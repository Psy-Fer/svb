//! Fused ex-zd tail: inverse-zigzag + delta prefix sum in one SIMD pass.
//!
//! `patched::decode_into` (the exception scan/scatter) stays a separate,
//! largely scalar pass — exceptions sit at arbitrary, data-dependent
//! positions, which doesn't map onto a fixed-width SIMD loop the way a
//! tag-driven codec does. This module instead fuses the two stages that
//! *are* uniformly SIMD-friendly — inverse-zigzag and the delta prefix sum —
//! into one pass over the already-reconstructed zigzag-delta (`zd`) array.
//! Same technique as `svbzd_fused.rs`/`vbz_fused.rs`: the branch-free zigzag
//! work fills the delta carry-chain's dependency stall, and this avoids
//! materializing an intermediate `Vec<i16>` of deltas between two full scans.
//!
//! The SIMD math itself is not new: it's `zigzag::decode_into_sse2`/
//! `decode_into_neon` (element-wise inverse-zigzag) combined with
//! `delta::decode_sse2_i16`/`decode_neon_i16`'s three-step prefix-sum scan,
//! recombined into a single loop instead of two.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Inverse-zigzag + delta-decode `zd`, starting the prefix sum from `initial`, appending to `out`.
pub(crate) fn decode_into(zd: &[u16], initial: i16, out: &mut Vec<i16>) {
    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-ssse3"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2(zd, initial, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon(zd, initial, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2(zd, initial, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon(zd, initial, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(
            any(feature = "simd-avx2", feature = "simd-ssse3"),
            target_arch = "x86_64"
        ),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(
            feature = "simd-auto",
            not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    decode_scalar(zd, initial, out);
}

#[inline]
fn unzigzag_one(code: u16) -> i16 {
    ((code >> 1) as i16) ^ -((code & 1) as i16)
}

#[allow(dead_code)]
fn decode_scalar(zd: &[u16], initial: i16, out: &mut Vec<i16>) {
    out.reserve(zd.len());
    let mut acc = initial;
    for &code in zd {
        acc = acc.wrapping_add(unzigzag_one(code));
        out.push(acc);
    }
}

// SSE2: 8 u16 codes per iteration.
// Inverse zigzag (element-wise, from zigzag::decode_into_sse2) feeds directly
// into the three-step prefix-sum scan (from delta::decode_sse2_i16) in the
// same register, instead of round-tripping through an intermediate buffer.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2(zd: &[u16], initial: i16, out: &mut Vec<i16>) {
    use core::arch::x86_64::*;

    let n = zd.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let result = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; zd slice bounds are valid.
            let v = _mm_loadu_si128(zd.as_ptr().add(i) as *const __m128i);

            // Inverse zigzag: shifted = v >> 1; sign = 0 - (v & 1).
            let one = _mm_set1_epi16(1);
            let zero = _mm_setzero_si128();
            let low_bit = _mm_and_si128(v, one);
            let sign = _mm_sub_epi16(zero, low_bit);
            let shifted = _mm_srli_epi16(v, 1);
            let delta = _mm_xor_si128(shifted, sign);

            // Three-step prefix-sum scan (all wrapping i16 arithmetic).
            let d = _mm_add_epi16(delta, _mm_slli_si128(delta, 2));
            let d = _mm_add_epi16(d, _mm_slli_si128(d, 4));
            let d = _mm_add_epi16(d, _mm_slli_si128(d, 8));
            _mm_add_epi16(d, _mm_set1_epi16(acc))
        };
        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 8 <= base + n.
            let out_ptr = out.as_mut_ptr().add(base + i) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
            // Element 7 is the prefix sum of all 8 deltas + acc = new accumulator.
            acc = _mm_extract_epi16(result, 7) as i16;
        }
        i += 8;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    // Scalar tail for n % 8 remaining values.
    for &code in &zd[simd_n..] {
        acc = acc.wrapping_add(unzigzag_one(code));
        out.push(acc);
    }
}

// NEON: 8 u16 codes per iteration. Same fusion as decode_sse2, using
// zigzag::decode_into_neon's element-wise unzigzag and delta::decode_neon_i16's
// vextq_s16-based three-step prefix-sum scan.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_neon(zd: &[u16], initial: i16, out: &mut Vec<i16>) {
    use core::arch::aarch64::*;

    let n = zd.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let result = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; zd slice bounds are valid.
            let v = vld1q_u16(zd.as_ptr().add(i));

            // Inverse zigzag: shifted = v >> 1; sign = 0 - (v & 1).
            let one = vdupq_n_u16(1);
            let zero16 = vdupq_n_u16(0);
            let low_bit = vandq_u16(v, one);
            let sign = vsubq_u16(zero16, low_bit);
            let shifted = vshrq_n_u16(v, 1);
            let delta = vreinterpretq_s16_u16(veorq_u16(shifted, sign));

            // Three-step prefix-sum scan (wrapping i16 arithmetic).
            let zero = vdupq_n_s16(0);
            let d = vaddq_s16(delta, vextq_s16(zero, delta, 7));
            let d = vaddq_s16(d, vextq_s16(zero, d, 6));
            let d = vaddq_s16(d, vextq_s16(zero, d, 4));
            vaddq_s16(d, vdupq_n_s16(acc))
        };
        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 8 <= base + n.
            vst1q_s16(out.as_mut_ptr().add(base + i), result);
            // Element 7 is the prefix sum of all 8 deltas + acc = new accumulator.
            acc = vgetq_lane_s16(result, 7);
        }
        i += 8;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    // Scalar tail for n % 8 remaining values.
    for &code in &zd[simd_n..] {
        acc = acc.wrapping_add(unzigzag_one(code));
        out.push(acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn reference(zd: &[u16], initial: i16) -> Vec<i16> {
        let mut acc = initial;
        zd.iter()
            .map(|&code| {
                acc = acc.wrapping_add(unzigzag_one(code));
                acc
            })
            .collect()
    }

    fn dispatch_matches_reference(zd: &[u16]) {
        let mut out = Vec::new();
        decode_into(zd, 0, &mut out);
        assert_eq!(out, reference(zd, 0));
    }

    #[test]
    fn empty() {
        dispatch_matches_reference(&[]);
    }

    #[test]
    fn small_tail_only() {
        for n in 0..8 {
            let zd: Vec<u16> = (0..n as u16).map(|i| i * 3 + 1).collect();
            dispatch_matches_reference(&zd);
        }
    }

    #[test]
    fn one_full_block_plus_tail() {
        let zd: Vec<u16> = (0..13u16).map(|i| i * 37 % 257).collect();
        dispatch_matches_reference(&zd);
    }

    #[test]
    fn large_input() {
        let zd: Vec<u16> = (0..1000u32).map(|i| ((i * 6151) % 65536) as u16).collect();
        dispatch_matches_reference(&zd);
    }

    #[test]
    fn extremes() {
        let zd = vec![0u16, 1, 65535, 65534, 32768, 32767];
        dispatch_matches_reference(&zd);
    }

    #[test]
    fn nonzero_initial_carry() {
        let zd: Vec<u16> = (0..20u16).map(|i| i * 5 + 2).collect();
        let mut out = Vec::new();
        decode_into(&zd, 1234, &mut out);
        assert_eq!(out, reference(&zd, 1234));
    }

    #[cfg(all(
        target_arch = "x86_64",
        any(feature = "simd-auto", feature = "simd-ssse3")
    ))]
    #[test]
    fn sse2_matches_reference_directly() {
        let zd: Vec<u16> = (0..37u16).map(|i| i * 91 % 401).collect();
        let mut out = Vec::new();
        unsafe { decode_sse2(&zd, 0, &mut out) };
        assert_eq!(out, reference(&zd, 0));
    }

    #[cfg(all(
        target_arch = "aarch64",
        any(feature = "simd-auto", feature = "simd-neon")
    ))]
    #[test]
    fn neon_matches_reference_directly() {
        let zd: Vec<u16> = (0..37u16).map(|i| i * 91 % 401).collect();
        let mut out = Vec::new();
        unsafe { decode_neon(&zd, 0, &mut out) };
        assert_eq!(out, reference(&zd, 0));
    }
}
