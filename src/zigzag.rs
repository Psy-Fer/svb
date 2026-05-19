#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

pub fn encode(samples: &[i16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(samples.len());
    encode_into(samples, &mut out);
    out
}

pub fn encode_into(samples: &[i16], out: &mut Vec<u16>) {
    out.extend(samples.iter().copied().map(encode_one));
}

pub fn decode(codes: &[u16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(codes.len());
    decode_into(codes, &mut out);
    out
}

pub fn decode_into(codes: &[u16], out: &mut Vec<i16>) {
    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    {
        decode_into_sse2(codes, out);
        return;
    }
    out.extend(codes.iter().copied().map(decode_one));
}

// SSE2 zigzag decode: 8 u16 values per iteration.
// SSE2 is baseline on x86_64 so no runtime feature check is needed.
// The scalar decode_one formula is: shifted = n >> 1; sign = -(n & 1); result = shifted ^ sign.
// In SIMD, -(n & 1) yields 0x0000 (if bit=0) or 0xFFFF (if bit=1) via _mm_sub_epi16(zero, bit).
#[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
fn decode_into_sse2(codes: &[u16], out: &mut Vec<i16>) {
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
    out.extend(codes[simd_n..].iter().copied().map(decode_one));
}

#[inline]
fn encode_one(x: i16) -> u16 {
    // Cast to u16 before left-shift to avoid i16 overflow on i16::MIN.
    // Right shift on i16 is arithmetic (sign-extending), yielding 0x0000 or 0xFFFF.
    ((x as u16) << 1) ^ ((x >> 15) as u16)
}

#[inline]
fn decode_one(n: u16) -> i16 {
    ((n >> 1) as i16) ^ (-((n & 1) as i16))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    // Cross-path: verify SSE2 and scalar produce identical output.
    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    fn decode_both(codes: &[u16]) -> (Vec<i16>, Vec<i16>) {
        let mut scalar_out = Vec::new();
        scalar_out.extend(codes.iter().copied().map(decode_one));
        let mut simd_out = Vec::new();
        decode_into_sse2(codes, &mut simd_out);
        (scalar_out, simd_out)
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_matches_scalar_known_values() {
        // Known zigzag codes: 0→0, 1→-1, 2→1, 3→-2, 65534→i16::MAX, 65535→i16::MIN
        let codes: Vec<u16> = vec![0, 1, 2, 3, 65534, 65535, 0, 0];
        let (s, v) = decode_both(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_matches_scalar_with_tail() {
        // 11 values — 8 via SIMD, 3 via scalar tail.
        let codes: Vec<u16> = (0..11u16).collect();
        let (s, v) = decode_both(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_matches_scalar_exhaustive_first_256() {
        // All 256 possible low-byte patterns (covers both 1-byte and sign cases).
        let codes: Vec<u16> = (0u16..256).collect();
        let (s, v) = decode_both(&codes);
        assert_eq!(s, v);
    }

    // ── exhaustive: all 65536 u16 values ─────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_exhaustive_all_u16_values() {
        // Every possible zigzag code through the SSE2 path must match scalar.
        let codes: Vec<u16> = (u16::MIN..=u16::MAX).collect();
        let (s, v) = decode_both(&codes);
        assert_eq!(s, v);
    }

    // ── all tail lengths ──────────────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_all_tail_lengths() {
        let pool: Vec<u16> = (0u16..16).collect();
        for n in 0..=16usize {
            let (s, v) = decode_both(&pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    // ── edge sizes ────────────────────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_empty_and_small() {
        // n=0 and n=1 exercise the early-return and pure-tail paths respectively.
        let (s, v) = decode_both(&[]);
        assert_eq!(s, v);
        let (s, v) = decode_both(&[0]);
        assert_eq!(s, v);
        let (s, v) = decode_both(&[65535]);
        assert_eq!(s, v);
    }

    // ── homogeneous bit patterns ──────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_all_even_codes() {
        // Even codes: low_bit = 0 → sign = 0 → result = code >> 1 (positive i16).
        let codes: Vec<u16> = (0..64u16).map(|i| i * 2).collect();
        let (s, v) = decode_both(&codes);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_all_odd_codes() {
        // Odd codes: low_bit = 1 → sign = 0xFFFF → result = (code>>1) ^ 0xFFFF = negative.
        let codes: Vec<u16> = (0..64u16).map(|i| i * 2 + 1).collect();
        let (s, v) = decode_both(&codes);
        assert_eq!(s, v);
    }

    // ── large input ───────────────────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_large_input() {
        let codes: Vec<u16> = (0u32..10_000).map(|i| (i % 65536) as u16).collect();
        let (s, v) = decode_both(&codes);
        assert_eq!(s, v);
    }

    #[test]
    fn known_values() {
        let cases: &[(i16, u16)] = &[
            (0, 0),
            (-1, 1),
            (1, 2),
            (-2, 3),
            (i16::MAX, 65534),
            (i16::MIN, 65535),
        ];
        for &(input, expected) in cases {
            assert_eq!(encode_one(input), expected, "encode {input}");
            assert_eq!(decode_one(expected), input, "decode {expected}");
        }
    }

    #[test]
    fn exhaustive_roundtrip() {
        // All 65536 i16 values must round-trip through zigzag.
        for raw in u16::MIN..=u16::MAX {
            let x = raw as i16;
            assert_eq!(decode_one(encode_one(x)), x);
        }
    }

    #[test]
    fn slice_roundtrip() {
        let samples: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        assert_eq!(decode(&encode(&samples)), samples);
    }

    #[test]
    fn encode_into_appends() {
        let mut out = vec![99u16];
        encode_into(&[0i16, -1, 1], &mut out);
        assert_eq!(out, [99, 0, 1, 2]);
    }

    #[test]
    fn small_values_encode_small() {
        // The point of zigzag: small absolute values → small unsigned codes.
        for x in -127i16..=127 {
            assert!(encode_one(x) <= 254, "x={x} encoded to {} (>254)", encode_one(x));
        }
    }
}
