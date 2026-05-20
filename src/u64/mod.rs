#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod shuffle;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod sse2;

// ── U64Coder1234 ──────────────────────────────────────────────────────────────

fn dispatch_encode_1234(values: &[u64], out: &mut Vec<u8>) {
    scalar::encode_into_1234(values, out);
}

fn dispatch_decode_1234(data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::decode_into_1234(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-sse2",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-sse2 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::decode_into_1234(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2"))
    ))]
    {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed at runtime.
                return unsafe { avx2::decode_into_1234(data, n, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::decode_into_1234(data, n, out) };
            }
        }
    }

    scalar::decode_into_1234(data, n, out)
}

/// StreamVByte codec for u64 values (1/2/3/4 bytes per value).
///
/// Same tag/width table as U32Classic but operates on `u64` slices. Values greater
/// than `u32::MAX` are silently truncated to their low 32 bits — this matches the
/// behaviour of other StreamVByte libraries and is defined, not accidental. Use
/// [`U64Coder1234::check_range`] before encoding if you need to detect out-of-range
/// values. For data that may genuinely exceed `u32::MAX`, use [`U64Coder1248`].
pub struct U64Coder1234;

impl U64Coder1234 {
    /// Returns the index of the first value that exceeds `u32::MAX`, or `None` if
    /// all values can be encoded without truncation.
    pub fn check_range(&self, values: &[u64]) -> Option<usize> {
        values.iter().position(|&v| v > u64::from(u32::MAX))
    }

    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_1234(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1234(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1234(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_1234(data, n, out)
    }
}

impl crate::coder::Coder for U64Coder1234 {
    type Elem = u64;

    fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1234(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        dispatch_decode_1234(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_1234(ctrl, n)
    }
}

// ── U64Coder1248 ──────────────────────────────────────────────────────────────

fn dispatch_encode_1248(values: &[u64], out: &mut Vec<u8>) {
    scalar::encode_into_1248(values, out);
}

fn dispatch_decode_1248(data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::decode_into_1248(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-sse2",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-sse2 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::decode_into_1248(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2"))
    ))]
    {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed at runtime.
                return unsafe { avx2::decode_into_1248(data, n, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::decode_into_1248(data, n, out) };
            }
        }
    }

    scalar::decode_into_1248(data, n, out)
}

/// StreamVByte codec for u64 values (1/2/4/8 bytes per value).
/// Covers the full u64 range. Values in 0x10000–0xFFFFFF use 4 bytes
/// (no 3-byte option); values in 0x100000000–u64::MAX use 8 bytes.
pub struct U64Coder1248;

impl U64Coder1248 {
    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_1248(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1248(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1248(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_1248(data, n, out)
    }
}

impl crate::coder::Coder for U64Coder1248 {
    type Elem = u64;

    fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1248(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        dispatch_decode_1248(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_1248(ctrl, n)
    }
}

#[cfg(test)]
mod cross_path {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86 {
        use super::super::{avx2, scalar, sse2};
        use std::vec::Vec;

        // ── helpers ──────────────────────────────────────────────────────────

        fn enc_1234(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_1234(v, &mut out);
            out
        }

        fn scalar_dec_1234(d: &[u8], n: usize) -> Vec<u64> {
            let mut out = Vec::new();
            scalar::decode_into_1234(d, n, &mut out).unwrap();
            out
        }

        fn ssse3_dec_1234(d: &[u8], n: usize) -> Option<Vec<u64>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::decode_into_1234(d, n, &mut out).unwrap() };
            Some(out)
        }

        fn avx2_dec_1234(d: &[u8], n: usize) -> Option<Vec<u64>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::decode_into_1234(d, n, &mut out).unwrap() };
            Some(out)
        }

        fn check_1234(values: &[u64]) {
            let n = values.len();
            let enc = enc_1234(values);
            let expected = scalar_dec_1234(&enc, n);
            if let Some(got) = ssse3_dec_1234(&enc, n) {
                assert_eq!(expected, got, "SSSE3 1234 mismatch n={n}");
            }
            if let Some(got) = avx2_dec_1234(&enc, n) {
                assert_eq!(expected, got, "AVX2 1234 mismatch n={n}");
            }
        }

        fn enc_1248(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_1248(v, &mut out);
            out
        }

        fn scalar_dec_1248(d: &[u8], n: usize) -> Vec<u64> {
            let mut out = Vec::new();
            scalar::decode_into_1248(d, n, &mut out).unwrap();
            out
        }

        fn ssse3_dec_1248(d: &[u8], n: usize) -> Option<Vec<u64>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::decode_into_1248(d, n, &mut out).unwrap() };
            Some(out)
        }

        fn avx2_dec_1248(d: &[u8], n: usize) -> Option<Vec<u64>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::decode_into_1248(d, n, &mut out).unwrap() };
            Some(out)
        }

        fn check_1248(values: &[u64]) {
            let n = values.len();
            let enc = enc_1248(values);
            let expected = scalar_dec_1248(&enc, n);
            if let Some(got) = ssse3_dec_1248(&enc, n) {
                assert_eq!(expected, got, "SSSE3 1248 mismatch n={n}");
            }
            if let Some(got) = avx2_dec_1248(&enc, n) {
                assert_eq!(expected, got, "AVX2 1248 mismatch n={n}");
            }
        }

        // ── U64Coder1234 tests ────────────────────────────────────────────────

        // exhaustive coverage of all 256 ctrl byte patterns
        #[test]
        fn all_ctrl_byte_values_1234() {
            for ctrl in 0u8..=255 {
                let values: Vec<u64> = (0..4u64)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => i + 1,           // 1-byte
                        1 => 0x100 + i,       // 2-byte
                        2 => 0x10000 + i,     // 3-byte
                        _ => 0x1000000 + i,   // 4-byte
                    })
                    .collect();
                check_1234(&values);
            }
        }

        #[test]
        fn all_tail_lengths_1234() {
            if ssse3_dec_1234(&enc_1234(&[1u64]), 1).is_none() {
                return;
            }
            let pool: Vec<u64> = (0..20u64)
                .map(|i| match i % 4 {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1000000 + i,
                })
                .collect();
            for n in 0..=20usize {
                let enc = enc_1234(&pool[..n]);
                let expected = scalar_dec_1234(&enc, n);
                if let Some(got) = ssse3_dec_1234(&enc, n) {
                    assert_eq!(expected, got, "SSSE3 1234 tail n={n}");
                }
                if let Some(got) = avx2_dec_1234(&enc, n) {
                    assert_eq!(expected, got, "AVX2 1234 tail n={n}");
                }
            }
        }

        #[test]
        fn homogeneous_tags_1234() {
            for base in [1u64, 0x100, 0x10000, 0x1000000] {
                let values: Vec<u64> = (0..32).map(|i| base + i).collect();
                check_1234(&values);
            }
        }

        // 16-byte guard: 4 × tag-3 (4 bytes each = 16 bytes data) then a short block
        #[test]
        fn ssse3_16byte_boundary_guard_1234() {
            let block1: Vec<u64> = (0x1000000u64..0x1000004).collect(); // 4 × tag-3
            let block2: Vec<u64> = vec![1, 0x100, 0x10000, 0x1000000];
            check_1234(&block1.into_iter().chain(block2).collect::<Vec<_>>());
        }

        // 32-byte guard: 8 × tag-3 exhausts the AVX2 32-byte window
        #[test]
        fn avx2_32byte_boundary_guard_1234() {
            let block12: Vec<u64> = (0x1000000u64..0x1000008).collect(); // 8 × tag-3
            let block3: Vec<u64> = vec![1, 0x100, 0x10000, 0x1000000];
            check_1234(&block12.into_iter().chain(block3).collect::<Vec<_>>());
        }

        #[test]
        fn boundary_values_1234() {
            let values: Vec<u64> = [
                0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX as u64,
            ]
            .iter()
            .copied()
            .cycle()
            .take(36)
            .collect();
            check_1234(&values);
        }

        #[test]
        fn large_input_1234() {
            let values: Vec<u64> = (0..10_000u64)
                .map(|i| match i % 4 {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFF_FFFF,
                    _ => 0x1000000 + i,
                })
                .collect();
            check_1234(&values);
        }

        #[test]
        fn empty_and_single_1234() {
            check_1234(&[]);
            for &v in &[0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, u32::MAX as u64] {
                check_1234(&[v]);
            }
        }

        // ── U64Coder1248 tests ────────────────────────────────────────────────

        // exhaustive coverage of all 256 ctrl byte patterns
        #[test]
        fn all_ctrl_byte_values_1248() {
            for ctrl in 0u8..=255 {
                let values: Vec<u64> = (0..4u64)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => i + 1,                    // 1-byte (1..=4)
                        1 => 0x100 + i,                // 2-byte
                        2 => 0x10000 + i,              // 4-byte
                        _ => 0x1_0000_0000 + i,        // 8-byte
                    })
                    .collect();
                check_1248(&values);
            }
        }

        #[test]
        fn all_tail_lengths_1248() {
            if ssse3_dec_1248(&enc_1248(&[1u64]), 1).is_none() {
                return;
            }
            let pool: Vec<u64> = (0..20u64)
                .map(|i| match i % 4 {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1_0000_0000 + i,
                })
                .collect();
            for n in 0..=20usize {
                let enc = enc_1248(&pool[..n]);
                let expected = scalar_dec_1248(&enc, n);
                if let Some(got) = ssse3_dec_1248(&enc, n) {
                    assert_eq!(expected, got, "SSSE3 1248 tail n={n}");
                }
                if let Some(got) = avx2_dec_1248(&enc, n) {
                    assert_eq!(expected, got, "AVX2 1248 tail n={n}");
                }
            }
        }

        #[test]
        fn homogeneous_tags_1248() {
            for base in [1u64, 0x100, 0x10000, 0x1_0000_0000] {
                let values: Vec<u64> = (0..32).map(|i| base + i).collect();
                check_1248(&values);
            }
        }

        // 32-byte guard: 4 × tag-3 (8 bytes each = 32 bytes data) then a short block
        #[test]
        fn ssse3_32byte_boundary_guard_1248() {
            let block1: Vec<u64> = (0x1_0000_0000u64..0x1_0000_0004).collect(); // 4 × 8-byte
            let block2: Vec<u64> = vec![1, 0x100, 0x10000, 0x1_0000_0000];
            check_1248(&block1.into_iter().chain(block2).collect::<Vec<_>>());
        }

        // 64-byte guard: 8 × tag-3 exhausts the AVX2 64-byte window
        #[test]
        fn avx2_64byte_boundary_guard_1248() {
            let block12: Vec<u64> = (0x1_0000_0000u64..0x1_0000_0008).collect(); // 8 × 8-byte
            let block3: Vec<u64> = vec![1, 0x100, 0x10000, 0x1_0000_0000];
            check_1248(&block12.into_iter().chain(block3).collect::<Vec<_>>());
        }

        #[test]
        fn boundary_values_1248() {
            let values: Vec<u64> = [
                0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF,
                0x1_0000_0000, u64::MAX,
            ]
            .iter()
            .copied()
            .cycle()
            .take(36)
            .collect();
            check_1248(&values);
        }

        #[test]
        fn large_input_1248() {
            let values: Vec<u64> = (0..10_000u64)
                .map(|i| match i % 4 {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFFFF_FFFF,
                    _ => 0x1_0000_0000 + i,
                })
                .collect();
            check_1248(&values);
        }

        #[test]
        fn empty_and_single_1248() {
            check_1248(&[]);
            for &v in &[0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, u64::MAX] {
                check_1248(&[v]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn check_range_all_safe() {
        assert_eq!(U64Coder1234.check_range(&[0, 0xFF, u32::MAX as u64]), None);
    }

    #[test]
    fn check_range_detects_first_bad() {
        let values = [1u64, 2, 0x1_0000_0000, 3, 0x2_0000_0000];
        assert_eq!(U64Coder1234.check_range(&values), Some(2));
    }

    #[test]
    fn check_range_empty() {
        assert_eq!(U64Coder1234.check_range(&[]), None);
    }

    #[test]
    fn check_range_first_element_bad() {
        assert_eq!(U64Coder1234.check_range(&[u64::MAX, 1, 2]), Some(0));
    }
}
