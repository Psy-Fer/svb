//! Zigzag encoding and decoding as a composable layer over signed integer types.
//!
//! Zigzag maps a signed integer to an unsigned integer so that values with
//! small absolute value produce small unsigned codes: `0→0`, `-1→1`, `1→2`,
//! `-2→3`, and so on.  This makes signed deltas compatible with StreamVByte
//! codecs that encode smaller unsigned values more compactly.
//!
//! The functions in this module accept any type that implements [`Zigzag`].

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

mod private {
    pub trait Sealed {}
}

/// Marker trait for signed integer types that support zigzag encoding.
///
/// This trait is sealed; it cannot be implemented outside this crate.
/// Implemented for `i16` (unsigned counterpart `u16`), `i32` (`u32`), and
/// `i64` (`u64`).
///
/// Zigzag is most useful after [`crate::delta`] encoding: delta produces
/// signed differences, and zigzag remaps them to small unsigned values that
/// StreamVByte can pack tightly.
pub trait Zigzag: private::Sealed + Sized + Copy {
    type Unsigned: Copy;
    fn encode_one(self) -> Self::Unsigned;
    fn decode_one(encoded: Self::Unsigned) -> Self;
    // Overridable decode dispatch; default is scalar. i16 overrides with SSE2/NEON.
    #[doc(hidden)]
    fn __decode_into(codes: &[Self::Unsigned], out: &mut Vec<Self>) {
        out.extend(codes.iter().copied().map(Self::decode_one));
    }
    #[doc(hidden)]
    fn __encode_into(samples: &[Self], out: &mut Vec<Self::Unsigned>) {
        out.extend(samples.iter().copied().map(Self::encode_one));
    }
}

impl private::Sealed for i16 {}
impl Zigzag for i16 {
    type Unsigned = u16;
    fn encode_one(self) -> u16 {
        ((self as u16) << 1) ^ ((self >> 15) as u16)
    }
    fn decode_one(n: u16) -> i16 {
        ((n >> 1) as i16) ^ (-((n & 1) as i16))
    }
    fn __decode_into(codes: &[u16], out: &mut Vec<i16>) {
        decode_into_i16(codes, out);
    }
}

fn decode_into_i16(codes: &[u16], out: &mut Vec<i16>) {
    #[cfg(all(any(feature = "simd-avx2", feature = "simd-sse2"), target_arch = "x86_64"))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_into_sse2(codes, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_into_neon(codes, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_into_sse2(codes, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_into_neon(codes, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(any(feature = "simd-avx2", feature = "simd-sse2"), target_arch = "x86_64"),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), any(target_arch = "x86_64", target_arch = "aarch64"))
    )))]
    out.extend(codes.iter().copied().map(i16::decode_one));
}

impl private::Sealed for i32 {}
impl Zigzag for i32 {
    type Unsigned = u32;
    fn encode_one(self) -> u32 {
        ((self as u32) << 1) ^ ((self >> 31) as u32)
    }
    fn decode_one(n: u32) -> i32 {
        ((n >> 1) as i32) ^ (-((n & 1) as i32))
    }
}

impl private::Sealed for i64 {}
impl Zigzag for i64 {
    type Unsigned = u64;
    fn encode_one(self) -> u64 {
        ((self as u64) << 1) ^ ((self >> 63) as u64)
    }
    fn decode_one(n: u64) -> i64 {
        ((n >> 1) as i64) ^ (-((n & 1) as i64))
    }
}

/// Zigzag-encode `samples`, mapping each signed value to a compact unsigned code.
///
/// # Examples
///
/// ```
/// # use svb::zigzag;
/// let codes: Vec<u16> = zigzag::encode(&[0i16, -1, 1, -2]);
/// assert_eq!(codes, [0, 1, 2, 3]);
/// ```
pub fn encode<T: Zigzag>(samples: &[T]) -> Vec<T::Unsigned> {
    let mut out = Vec::with_capacity(samples.len());
    T::__encode_into(samples, &mut out);
    out
}

/// Zigzag-encode `samples`, appending the unsigned codes to `out`.
pub fn encode_into<T: Zigzag>(samples: &[T], out: &mut Vec<T::Unsigned>) {
    T::__encode_into(samples, out);
}

/// Zigzag-decode `codes`, recovering the original signed values.
///
/// # Examples
///
/// ```
/// # use svb::zigzag;
/// let samples: Vec<i16> = zigzag::decode(&[0u16, 1, 2, 3]);
/// assert_eq!(samples, [0, -1, 1, -2]);
/// ```
pub fn decode<T: Zigzag>(codes: &[T::Unsigned]) -> Vec<T> {
    let mut out = Vec::with_capacity(codes.len());
    T::__decode_into(codes, &mut out);
    out
}

/// Zigzag-decode `codes`, appending the recovered signed values to `out`.
pub fn decode_into<T: Zigzag>(codes: &[T::Unsigned], out: &mut Vec<T>) {
    T::__decode_into(codes, out);
}

// SSE2 zigzag decode: 8 u16 values per iteration.
// SSE2 is baseline on x86_64 so no runtime feature check is needed.
// The scalar decode_one formula is: shifted = n >> 1; sign = -(n & 1); result = shifted ^ sign.
// In SIMD, -(n & 1) yields 0x0000 (if bit=0) or 0xFFFF (if bit=1) via _mm_sub_epi16(zero, bit).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_into_sse2(codes: &[u16], out: &mut Vec<i16>) {
    use core::arch::x86_64::*;

    let n = codes.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut i = 0usize;
    while i < simd_n {
        let result = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; codes slice bounds are valid.
            let v = _mm_loadu_si128(codes.as_ptr().add(i) as *const __m128i);
            let one = _mm_set1_epi16(1);
            let zero = _mm_setzero_si128();
            let low_bit = _mm_and_si128(v, one);
            // _mm_sub_epi16(zero, low_bit): 0 - 0 = 0x0000; 0 - 1 = 0xFFFF (wrapping i16)
            let sign = _mm_sub_epi16(zero, low_bit);
            let shifted = _mm_srli_epi16(v, 1);
            _mm_xor_si128(shifted, sign)
        };
        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 8 <= base + n.
            let out_ptr = out.as_mut_ptr().add(base + i) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
        }
        i += 8;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    // Scalar tail for n % 8 remaining values.
    out.extend(codes[simd_n..].iter().copied().map(i16::decode_one));
}

// NEON zigzag decode: 8 u16 values per iteration.
//
// Formula: shifted = v >> 1; sign = 0 - (v & 1)  [wrapping: gives 0 or 0xFFFF]
//          result = shifted XOR sign
// Operating on uint16x8_t throughout; final store reinterprets as int16x8_t.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_into_neon(codes: &[u16], out: &mut Vec<i16>) {
    use core::arch::aarch64::*;

    let n = codes.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut i = 0usize;
    while i < simd_n {
        // SAFETY: i + 8 <= simd_n <= n; codes slice bounds are valid.
        let v = unsafe { vld1q_u16(codes.as_ptr().add(i)) };
        let one = unsafe { vdupq_n_u16(1) };
        let zero = unsafe { vdupq_n_u16(0) };

        let low_bit = unsafe { vandq_u16(v, one) };
        // vsubq_u16(0, low_bit): 0-0=0x0000, 0-1=0xFFFF (wrapping)
        let sign = unsafe { vsubq_u16(zero, low_bit) };
        let shifted = unsafe { vshrq_n_u16(v, 1) };
        let result = unsafe { veorq_u16(shifted, sign) };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 8 <= base + n.
            // vreinterpretq_s16_u16: same bits, reinterpreted as i16.
            vst1q_s16(
                out.as_mut_ptr().add(base + i),
                vreinterpretq_s16_u16(result),
            );
        }
        i += 8;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    // Scalar tail for n % 8 remaining values.
    out.extend(codes[simd_n..].iter().copied().map(i16::decode_one));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    // ── i16 cross-path tests (SSE2 vs scalar) ────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    fn decode_both_i16(codes: &[u16]) -> (Vec<i16>, Vec<i16>) {
        let scalar_out: Vec<i16> = codes.iter().copied().map(i16::decode_one).collect();
        let mut simd_out = Vec::new();
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_into_sse2(codes, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_matches_scalar_known_values() {
        let codes: Vec<u16> = vec![0, 1, 2, 3, 65534, 65535, 0, 0];
        let (s, v) = decode_both_i16(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_matches_scalar_with_tail() {
        let codes: Vec<u16> = (0..11u16).collect();
        let (s, v) = decode_both_i16(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_matches_scalar_exhaustive_first_256() {
        let codes: Vec<u16> = (0u16..256).collect();
        let (s, v) = decode_both_i16(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_exhaustive_all_u16_values() {
        let codes: Vec<u16> = (u16::MIN..=u16::MAX).collect();
        let (s, v) = decode_both_i16(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_all_tail_lengths() {
        let pool: Vec<u16> = (0u16..16).collect();
        for n in 0..=16usize {
            let (s, v) = decode_both_i16(&pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_empty_and_small() {
        let (s, v) = decode_both_i16(&[]);
        assert_eq!(s, v);
        let (s, v) = decode_both_i16(&[0]);
        assert_eq!(s, v);
        let (s, v) = decode_both_i16(&[65535]);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_all_even_codes() {
        let codes: Vec<u16> = (0..64u16).map(|i| i * 2).collect();
        let (s, v) = decode_both_i16(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_all_odd_codes() {
        let codes: Vec<u16> = (0..64u16).map(|i| i * 2 + 1).collect();
        let (s, v) = decode_both_i16(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_large_input() {
        let codes: Vec<u16> = (0u32..10_000).map(|i| (i % 65536) as u16).collect();
        let (s, v) = decode_both_i16(&codes);
        assert_eq!(s, v);
    }

    // ── i16 NEON cross-path tests (aarch64) ──────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    fn decode_both_i16_neon(codes: &[u16]) -> (Vec<i16>, Vec<i16>) {
        let scalar_out: Vec<i16> = codes.iter().copied().map(i16::decode_one).collect();
        let mut simd_out = Vec::new();
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_into_neon(codes, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_known_values() {
        let codes: Vec<u16> = vec![0, 1, 2, 3, 65534, 65535, 0, 0];
        let (s, v) = decode_both_i16_neon(&codes);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_with_tail() {
        let codes: Vec<u16> = (0..11u16).collect();
        let (s, v) = decode_both_i16_neon(&codes);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_exhaustive_first_256() {
        let codes: Vec<u16> = (0u16..256).collect();
        let (s, v) = decode_both_i16_neon(&codes);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_exhaustive_all_u16_values() {
        let codes: Vec<u16> = (u16::MIN..=u16::MAX).collect();
        let (s, v) = decode_both_i16_neon(&codes);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_all_tail_lengths() {
        let pool: Vec<u16> = (0u16..16).collect();
        for n in 0..=16usize {
            let (s, v) = decode_both_i16_neon(&pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_all_even_codes() {
        let codes: Vec<u16> = (0..64u16).map(|i| i * 2).collect();
        let (s, v) = decode_both_i16_neon(&codes);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_all_odd_codes() {
        let codes: Vec<u16> = (0..64u16).map(|i| i * 2 + 1).collect();
        let (s, v) = decode_both_i16_neon(&codes);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_large_input() {
        let codes: Vec<u16> = (0u32..10_000).map(|i| (i % 65536) as u16).collect();
        let (s, v) = decode_both_i16_neon(&codes);
        assert_eq!(s, v);
    }

    // ── i16 correctness ───────────────────────────────────────────────────────

    #[test]
    fn i16_known_values() {
        let cases: &[(i16, u16)] = &[
            (0, 0),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (i16::MAX, 65534),
            (i16::MIN, 65535),
        ];
        for &(input, expected) in cases {
            assert_eq!(input.encode_one(), expected, "encode {input}");
            assert_eq!(i16::decode_one(expected), input, "decode {expected}");
        }
    }

    #[test]
    fn i16_exhaustive_roundtrip() {
        for raw in u16::MIN..=u16::MAX {
            let x = raw as i16;
            assert_eq!(i16::decode_one(x.encode_one()), x);
        }
    }

    #[test]
    fn i16_slice_roundtrip() {
        let samples: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        let decoded: Vec<i16> = decode(&encode(&samples));
        assert_eq!(decoded, samples);
    }

    #[test]
    fn i16_encode_into_appends() {
        let mut out = vec![99u16];
        encode_into(&[0i16, -1, 1], &mut out);
        assert_eq!(out, [99, 0, 1, 2]);
    }

    #[test]
    fn i16_small_values_encode_small() {
        for x in -127i16..=127 {
            assert!(x.encode_one() <= 254, "x={x} encoded to {} (>254)", x.encode_one());
        }
    }

    // ── i32 correctness ───────────────────────────────────────────────────────

    #[test]
    fn i32_known_values() {
        let cases: &[(i32, u32)] = &[
            (0, 0),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (i32::MAX, u32::MAX - 1),
            (i32::MIN, u32::MAX),
        ];
        for &(input, expected) in cases {
            assert_eq!(input.encode_one(), expected, "encode {input}");
            assert_eq!(i32::decode_one(expected), input, "decode {expected}");
        }
    }

    #[test]
    fn i32_slice_roundtrip() {
        let samples: Vec<i32> = vec![0, -1, 1, i32::MIN, i32::MAX, -1_000_000, 1_000_000];
        let decoded: Vec<i32> = decode(&encode(&samples));
        assert_eq!(decoded, samples);
    }

    #[test]
    fn i32_small_values_encode_small() {
        for x in -127i32..=127 {
            assert!(x.encode_one() <= 254, "x={x} encoded to {} (>254)", x.encode_one());
        }
    }

    #[test]
    fn i32_encode_into_appends() {
        let mut out = vec![99u32];
        encode_into(&[0i32, -1, 1], &mut out);
        assert_eq!(out, [99, 0, 1, 2]);
    }

    // ── i64 correctness ───────────────────────────────────────────────────────

    #[test]
    fn i64_known_values() {
        let cases: &[(i64, u64)] = &[
            (0, 0),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (i64::MAX, u64::MAX - 1),
            (i64::MIN, u64::MAX),
        ];
        for &(input, expected) in cases {
            assert_eq!(input.encode_one(), expected, "encode {input}");
            assert_eq!(i64::decode_one(expected), input, "decode {expected}");
        }
    }

    #[test]
    fn i64_slice_roundtrip() {
        let samples: Vec<i64> = vec![0, -1, 1, i64::MIN, i64::MAX, -1_000_000_000, 1_000_000_000];
        let decoded: Vec<i64> = decode(&encode(&samples));
        assert_eq!(decoded, samples);
    }

    #[test]
    fn i64_encode_into_appends() {
        let mut out = vec![99u64];
        encode_into(&[0i64, -1, 1], &mut out);
        assert_eq!(out, [99, 0, 1, 2]);
    }
}
