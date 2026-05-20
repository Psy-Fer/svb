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

// ── U32Classic ────────────────────────────────────────────────────────────────

fn dispatch_encode_classic(values: &[u32], out: &mut Vec<u8>) {
    scalar::encode_into_classic(values, out);
}

fn dispatch_decode_classic(data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::decode_into_classic(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-sse2",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-sse2 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::decode_into_classic(data, n, out) };
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
                return unsafe { avx2::decode_into_classic(data, n, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::decode_into_classic(data, n, out) };
            }
        }
    }

    scalar::decode_into_classic(data, n, out)
}

/// StreamVByte codec for u32 values.
/// 2-bit tags, 1/2/3/4 bytes per value. Wire-compatible with Lemire's C library.
pub struct U32Classic;

impl U32Classic {
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_classic(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_classic(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_classic(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_classic(data, n, out)
    }
}

impl crate::coder::Coder for U32Classic {
    type Elem = u32;

    fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_classic(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
        dispatch_decode_classic(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_classic(ctrl, n)
    }
}

// ── U32Variant0124 ────────────────────────────────────────────────────────────

fn dispatch_encode_0124(values: &[u32], out: &mut Vec<u8>) {
    scalar::encode_into_0124(values, out);
}

fn dispatch_decode_0124(data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
    scalar::decode_into_0124(data, n, out)
}

/// StreamVByte codec for u32 values.
/// 2-bit tags, 0/1/2/4 bytes per value. Zero values use 0 data bytes, making
/// this more compact than U32Classic for sparse (mostly-zero) data.
pub struct U32Variant0124;

impl U32Variant0124 {
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_0124(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_0124(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_0124(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_0124(data, n, out)
    }
}

impl crate::coder::Coder for U32Variant0124 {
    type Elem = u32;

    fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_0124(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
        dispatch_decode_0124(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_0124(ctrl, n)
    }
}

#[cfg(test)]
mod cross_path {
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86 {
        use super::super::{avx2, scalar, sse2};
        use std::vec::Vec;

        fn encode(values: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_classic(values, &mut out);
            out
        }

        fn scalar_decode(data: &[u8], n: usize) -> Vec<u32> {
            let mut out = Vec::new();
            scalar::decode_into_classic(data, n, &mut out).unwrap();
            out
        }

        fn ssse3_decode(data: &[u8], n: usize) -> Option<Vec<u32>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::decode_into_classic(data, n, &mut out).unwrap() };
            Some(out)
        }

        fn avx2_decode(data: &[u8], n: usize) -> Option<Vec<u32>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::decode_into_classic(data, n, &mut out).unwrap() };
            Some(out)
        }

        fn check(values: &[u32]) {
            let n = values.len();
            let enc = encode(values);
            let expected = scalar_decode(&enc, n);
            if let Some(got) = ssse3_decode(&enc, n) {
                assert_eq!(expected, got, "SSSE3 mismatch n={n}");
            }
            if let Some(got) = avx2_decode(&enc, n) {
                assert_eq!(expected, got, "AVX2 mismatch n={n}");
            }
        }

        // all 256 ctrl byte values — exhaustive coverage of TABLE
        #[test]
        fn all_ctrl_byte_values() {
            for ctrl in 0u8..=255 {
                let values: Vec<u32> = (0..4u32)
                    .map(|i| match (ctrl >> (2 * i)) & 3 {
                        0 => i,               // 0..3, fits in 1 byte
                        1 => 256 + i,         // fits in 2 bytes
                        2 => 65536 + i,       // fits in 3 bytes
                        _ => 16_777_216 + i,  // needs 4 bytes
                    })
                    .collect();
                check(&values);
            }
        }

        // n = 0..=20 covers all n%4 residues in both < 4-value and > 4-value cases
        #[test]
        fn all_tail_lengths() {
            if ssse3_decode(&encode(&[1u32]), 1).is_none() {
                return;
            }
            let pool: Vec<u32> = (0..20)
                .map(|i| match i % 4 {
                    0 => i,
                    1 => 256 + i,
                    2 => 65536 + i,
                    _ => 16_777_216 + i,
                })
                .collect();
            for n in 0..=20usize {
                let enc = encode(&pool[..n]);
                let expected = scalar_decode(&enc, n);
                if let Some(got) = ssse3_decode(&enc, n) {
                    assert_eq!(expected, got, "SSSE3 tail n={n}");
                }
                if let Some(got) = avx2_decode(&enc, n) {
                    assert_eq!(expected, got, "AVX2 tail n={n}");
                }
            }
        }

        // all values the same tag
        #[test]
        fn homogeneous_tags() {
            for tag in 0u32..4 {
                let base: u32 = [0, 256, 65536, 16_777_216][tag as usize];
                let values: Vec<u32> = (0..32).map(|i| base + i).collect();
                check(&values);
            }
        }

        // 16-byte boundary guard for SSSE3
        #[test]
        fn ssse3_16byte_boundary_guard() {
            // block1: 4 tag-3 values (16 bytes of data). After consuming, ≤15 bytes remain
            // for block2 → forces scalar tail for block2.
            let block1: Vec<u32> = (16_777_216..16_777_220).collect(); // tag=3
            let block2: Vec<u32> = vec![1, 2, 3, 4]; // tag=0
            check(&block1.into_iter().chain(block2).collect::<Vec<_>>());
        }

        // 32-byte boundary guard for AVX2
        #[test]
        fn avx2_32byte_boundary_guard() {
            // two blocks of tag-3 (32 bytes) then a mixed block → forces scalar tail
            let blocks12: Vec<u32> = (16_777_216..16_777_224).collect(); // 8 × tag-3
            let block3: Vec<u32> = vec![1, 256, 65536, 16_777_216];
            check(&blocks12.into_iter().chain(block3).collect::<Vec<_>>());
        }

        // boundary values: 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFFFF, 0x1000000, u32::MAX
        #[test]
        fn boundary_values() {
            let values: Vec<u32> = [
                0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX,
            ]
            .iter()
            .copied()
            .cycle()
            .take(32)
            .collect();
            check(&values);
        }

        // large input
        #[test]
        fn large_input() {
            let values: Vec<u32> = (0..10_000u32)
                .map(|i| match i % 4 {
                    0 => i % 256,
                    1 => 256 + i % 65536,
                    2 => 65536 + i % 16_777_216,
                    _ => 16_777_216 + i,
                })
                .collect();
            check(&values);
        }

        #[test]
        fn empty_and_single() {
            check(&[]);
            for &v in &[
                0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX,
            ] {
                check(&[v]);
            }
        }
    }
}
