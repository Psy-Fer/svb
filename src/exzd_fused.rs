//! Fused ex-zd tail: inverse-zigzag + delta prefix sum + qts left-shift in
//! one SIMD pass.
//!
//! `patched::decode_into` (the exception scan/scatter) stays a separate,
//! largely scalar pass — exceptions sit at arbitrary, data-dependent
//! positions, which doesn't map onto a fixed-width SIMD loop the way a
//! tag-driven codec does. This module instead fuses the three stages that
//! *are* uniformly SIMD-friendly — inverse-zigzag, the delta prefix sum, and
//! the final qts left-shift — into one pass over the already-reconstructed
//! zigzag-delta (`zd`) array. Same technique as `svbzd_fused.rs`/
//! `vbz_fused.rs`: the branch-free zigzag work fills the delta carry-chain's
//! dependency stall, and this avoids materializing an intermediate
//! `Vec<i16>` of deltas between two full scans.
//!
//! The SIMD math itself is not new: it's `zigzag::decode_into_sse2`/
//! `decode_into_neon` (element-wise inverse-zigzag) combined with
//! `delta::decode_sse2_i16`/`decode_neon_i16`'s three-step prefix-sum scan,
//! recombined into a single loop instead of two.
//!
//! qts folding: `q` is loop-invariant (decided once, from the frame header,
//! before any of this runs), and left-shift distributes over addition even
//! under wrapping/modular arithmetic — `(a + b) << q == (a << q) + (b << q)`
//! holds exactly mod 2^16, regardless of intermediate wraparound. So shifting
//! the fully-accumulated per-block result right before the store, while
//! carrying the *unshifted* value forward as the next block's carry, gives
//! bit-identical output to running the whole decode unshifted and then
//! shifting the final array in a separate pass — one SIMD instruction per
//! block instead of a whole extra scan.
//!
//! A chunked variant that also folded `patched::decode_into`'s
//! reconstruction into this same loop (processing fixed-size chunks so
//! exception density could never fragment the SIMD width) was tried at
//! *two* chunk widths and measured slower than this simpler two-stage
//! version both times, across every data profile tested:
//!
//! - 8-wide (SSE2): re-entering the transform per chunk paid a fixed setup
//!   cost (register loads, bounds bookkeeping) that scaled with chunk
//!   count, and 8-wide chunking meant a lot of chunks.
//! - 16-wide (AVX2): halving the chunk count didn't help — AVX2's extra
//!   cross-lane bridge step (extract the low half's total, broadcast it,
//!   insert into the high half) adds more per-chunk instruction overhead
//!   than SSE2's simpler in-register 3-step scan ever needed, more than
//!   offsetting the benefit of paying the per-chunk cost half as often.
//!
//! Keeping the merge (`patched::decode_into`) and the transform (this
//! module) as two separate full-array passes is the faster design in
//! practice at both widths, not just simpler.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

/// Inverse-zigzag + delta-decode `zd`, applying a left-shift of `q` bits to
/// each reconstructed sample, starting the prefix sum from `initial`, and
/// appending to `out`.
///
/// `q` is masked to `0..=15` so the shift can never panic, even on a `q`
/// value read from untrusted/corrupted input (same convention as
/// [`crate::quantize::unshift_inplace`]).
pub(crate) fn decode_into(zd: &[u16], initial: i16, q: u8, out: &mut Vec<i16>) {
    let q = q & 15;
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        unsafe { decode_avx2(zd, initial, q, out) };
    }
    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2(zd, initial, q, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon(zd, initial, q, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        feature = "std",
        target_arch = "x86_64"
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime.
            unsafe { decode_avx2(zd, initial, q, out) };
        } else {
            // SAFETY: SSE2 is always available on x86_64.
            unsafe { decode_sse2(zd, initial, q, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon(zd, initial, q, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target
    // (including `simd-auto` on x86_64 without `std`, which can't do
    // runtime feature detection — see the crate-level no_std note).
    #[cfg(not(any(
        all(feature = "simd-avx2", target_arch = "x86_64"),
        all(
            feature = "simd-ssse3",
            not(feature = "simd-avx2"),
            target_arch = "x86_64"
        ),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(
            feature = "simd-auto",
            not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
            feature = "std",
            target_arch = "x86_64"
        ),
        all(
            feature = "simd-auto",
            not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
            target_arch = "aarch64"
        )
    )))]
    decode_scalar(zd, initial, q, out);
}

#[inline]
fn unzigzag_one(code: u16) -> i16 {
    ((code >> 1) as i16) ^ -((code & 1) as i16)
}

#[allow(dead_code)]
fn decode_scalar(zd: &[u16], initial: i16, q: u8, out: &mut Vec<i16>) {
    out.reserve(zd.len());
    let mut acc = initial;
    for &code in zd {
        acc = acc.wrapping_add(unzigzag_one(code));
        out.push(acc << q);
    }
}

// SSE2: 8 u16 codes per iteration.
// Inverse zigzag (element-wise, from zigzag::decode_into_sse2) feeds directly
// into the three-step prefix-sum scan (from delta::decode_sse2_i16) in the
// same register, instead of round-tripping through an intermediate buffer.
// The qts shift is applied to `result` (the unshifted accumulated value)
// only for the store; the carry extracted for the next iteration stays
// unshifted, since it needs to keep accumulating in the same domain as `zd`.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2(zd: &[u16], initial: i16, q: u8, out: &mut Vec<i16>) {
    use core::arch::x86_64::*;

    let n = zd.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut acc = initial;
    let mut i = 0usize;

    // SAFETY: _mm_cvtsi32_si128 has no preconditions and touches no memory.
    let q_vec = unsafe { _mm_cvtsi32_si128(q as i32) };

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
            // Element 7 of the unshifted `result` is the prefix sum of all 8
            // deltas + acc = new accumulator; qts shift is applied only to
            // the stored copy.
            acc = _mm_extract_epi16(result, 7) as i16;
            _mm_storeu_si128(out_ptr, _mm_sll_epi16(result, q_vec));
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
        out.push(acc << q);
    }
}

// AVX2: 16 u16 codes per iteration — same fusion as decode_sse2, but the
// prefix-sum scan needs a fourth step. `_mm256_slli_si256` (like all
// byte-granularity AVX2 shifts) operates *within each 128-bit lane*
// independently, not across the full 256 bits, so the first three
// shift-add steps (2/4/8-byte shifts) produce two independent 8-wide local
// prefix sums — the high half's partial sums don't yet include the low
// half's total. The fourth step bridges them: extract the low half's total
// (its last lane), broadcast it into a vector that's zero in the low half
// and that value in every high-half lane, and add — turning the high
// half's local sums into correct running sums. Only then is the
// inter-block carry (`acc`) broadcast-added to all 16 lanes uniformly.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
unsafe fn decode_avx2(zd: &[u16], initial: i16, q: u8, out: &mut Vec<i16>) {
    use core::arch::x86_64::*;

    let n = zd.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 16) * 16;

    let mut acc = initial;
    let mut i = 0usize;

    let q_vec = _mm_cvtsi32_si128(q as i32);

    while i < simd_n {
        let result = unsafe {
            // SAFETY: i + 16 <= simd_n <= n; zd slice bounds are valid.
            let v = _mm256_loadu_si256(zd.as_ptr().add(i) as *const __m256i);

            // Inverse zigzag: shifted = v >> 1; sign = 0 - (v & 1). AND/SUB/
            // SHIFT/XOR are full-width, no in-lane restriction.
            let one = _mm256_set1_epi16(1);
            let zero = _mm256_setzero_si256();
            let low_bit = _mm256_and_si256(v, one);
            let sign = _mm256_sub_epi16(zero, low_bit);
            let shifted = _mm256_srli_epi16(v, 1);
            let delta = _mm256_xor_si256(shifted, sign);

            // Three in-lane shift-add steps: two independent 8-wide local
            // prefix sums (low half = lanes 0..7, high half = lanes 8..15).
            let d = _mm256_add_epi16(delta, _mm256_slli_si256(delta, 2));
            let d = _mm256_add_epi16(d, _mm256_slli_si256(d, 4));
            let d = _mm256_add_epi16(d, _mm256_slli_si256(d, 8));

            // Bridge: add the low half's total to every lane of the high
            // half only, turning its local sums into correct running sums.
            let lo = _mm256_extracti128_si256(d, 0);
            let low_total = _mm_extract_epi16(lo, 7) as i16;
            let bridge_hi = _mm_set1_epi16(low_total);
            let bridge = _mm256_inserti128_si256(_mm256_setzero_si256(), bridge_hi, 1);
            let d = _mm256_add_epi16(d, bridge);

            // Broadcast-add the inter-block carry to all 16 lanes.
            _mm256_add_epi16(d, _mm256_set1_epi16(acc))
        };
        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 16 <= base + n.
            let out_ptr = out.as_mut_ptr().add(base + i) as *mut __m256i;
            // Lane 15 (last lane of the high half) of the unshifted `result`
            // is the prefix sum of all 16 deltas + acc = new accumulator;
            // qts shift is applied only to the stored copy.
            let hi = _mm256_extracti128_si256(result, 1);
            acc = _mm_extract_epi16(hi, 7) as i16;
            _mm256_storeu_si256(out_ptr, _mm256_sll_epi16(result, q_vec));
        }
        i += 16;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    // Scalar tail for n % 16 remaining values.
    for &code in &zd[simd_n..] {
        acc = acc.wrapping_add(unzigzag_one(code));
        out.push(acc << q);
    }
}

// NEON: 8 u16 codes per iteration. Same fusion as decode_sse2, using
// zigzag::decode_into_neon's element-wise unzigzag and delta::decode_neon_i16's
// vextq_s16-based three-step prefix-sum scan.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_neon(zd: &[u16], initial: i16, q: u8, out: &mut Vec<i16>) {
    use core::arch::aarch64::*;

    let n = zd.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut acc = initial;
    let mut i = 0usize;

    // SAFETY: vdupq_n_s16 has no preconditions and touches no memory.
    let q_vec = unsafe { vdupq_n_s16(q as i16) };

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
            // Element 7 of the unshifted `result` is the prefix sum of all 8
            // deltas + acc = new accumulator; qts shift is applied only to
            // the stored copy.
            acc = vgetq_lane_s16(result, 7);
            vst1q_s16(out.as_mut_ptr().add(base + i), vshlq_s16(result, q_vec));
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
        out.push(acc << q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn reference(zd: &[u16], initial: i16, q: u8) -> Vec<i16> {
        let mut acc = initial;
        zd.iter()
            .map(|&code| {
                acc = acc.wrapping_add(unzigzag_one(code));
                acc << (q & 15)
            })
            .collect()
    }

    fn dispatch_matches_reference(zd: &[u16], q: u8) {
        let mut out = Vec::new();
        decode_into(zd, 0, q, &mut out);
        assert_eq!(out, reference(zd, 0, q));
    }

    #[test]
    fn empty() {
        dispatch_matches_reference(&[], 0);
    }

    #[test]
    fn small_tail_only() {
        // 0..16 covers both SSE2/NEON's 8-wide tail boundary and AVX2's
        // 16-wide one, whichever the dispatch picks for the active features.
        for n in 0..16 {
            let zd: Vec<u16> = (0..n as u16).map(|i| i * 3 + 1).collect();
            dispatch_matches_reference(&zd, 0);
        }
    }

    #[test]
    fn one_full_block_plus_tail() {
        // 13 exercises SSE2/NEON's one-block-plus-5-tail; 29 additionally
        // exercises AVX2's one-block-plus-13-tail.
        for n in [13u16, 29] {
            let zd: Vec<u16> = (0..n).map(|i| i * 37 % 257).collect();
            dispatch_matches_reference(&zd, 0);
        }
    }

    #[test]
    fn large_input() {
        let zd: Vec<u16> = (0..1000u32).map(|i| ((i * 6151) % 65536) as u16).collect();
        dispatch_matches_reference(&zd, 0);
    }

    #[test]
    fn extremes() {
        let zd = vec![0u16, 1, 65535, 65534, 32768, 32767];
        dispatch_matches_reference(&zd, 0);
    }

    #[test]
    fn nonzero_initial_carry() {
        let zd: Vec<u16> = (0..20u16).map(|i| i * 5 + 2).collect();
        let mut out = Vec::new();
        decode_into(&zd, 1234, 0, &mut out);
        assert_eq!(out, reference(&zd, 1234, 0));
    }

    #[test]
    fn nonzero_shift() {
        for q in [0u8, 1, 3, 5, 15] {
            let zd: Vec<u16> = (0..37u16).map(|i| i * 91 % 401).collect();
            dispatch_matches_reference(&zd, q);
        }
    }

    #[test]
    fn shift_amount_masked_to_avoid_panic() {
        // q values >= 16 must not panic (masked to q & 15), even though the
        // encoder never produces them — decode must stay safe on corrupted input.
        let zd: Vec<u16> = (0..20u16).map(|i| i * 7 + 1).collect();
        let mut out = Vec::new();
        decode_into(&zd, 0, 255, &mut out);
        assert_eq!(out, reference(&zd, 0, 255));
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    #[test]
    fn avx2_matches_reference_directly() {
        // Exhaustively cover every tail length 0..=31 (one full 16-block
        // plus every possible remainder, and the zero/one/two-block cases)
        // to nail down the cross-lane bridge step at every boundary.
        for n in 0..=31usize {
            for q in [0u8, 1, 5, 15] {
                let zd: Vec<u16> = (0..n as u32).map(|i| ((i * 6151) % 65536) as u16).collect();
                let mut out = Vec::new();
                unsafe { decode_avx2(&zd, 0, q, &mut out) };
                assert_eq!(out, reference(&zd, 0, q), "n={n} q={q}");
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd-avx2"))]
    #[test]
    fn avx2_nonzero_initial_carry() {
        let zd: Vec<u16> = (0..40u16).map(|i| i * 5 + 2).collect();
        let mut out = Vec::new();
        unsafe { decode_avx2(&zd, 1234, 0, &mut out) };
        assert_eq!(out, reference(&zd, 1234, 0));
    }

    #[cfg(all(
        target_arch = "x86_64",
        any(feature = "simd-auto", feature = "simd-ssse3")
    ))]
    #[test]
    fn sse2_matches_reference_directly() {
        for q in [0u8, 1, 5] {
            let zd: Vec<u16> = (0..37u16).map(|i| i * 91 % 401).collect();
            let mut out = Vec::new();
            unsafe { decode_sse2(&zd, 0, q, &mut out) };
            assert_eq!(out, reference(&zd, 0, q));
        }
    }

    #[cfg(all(
        target_arch = "aarch64",
        any(feature = "simd-auto", feature = "simd-neon")
    ))]
    #[test]
    fn neon_matches_reference_directly() {
        for q in [0u8, 1, 5] {
            let zd: Vec<u16> = (0..37u16).map(|i| i * 91 % 401).collect();
            let mut out = Vec::new();
            unsafe { decode_neon(&zd, 0, q, &mut out) };
            assert_eq!(out, reference(&zd, 0, q));
        }
    }
}
