//! StreamVByte codecs for `u32` values using 2-bit control tags.
//!
//! Two variants are provided:
//!
//! - [`U32Classic`]: tag encodes 1/2/3/4 data bytes; wire-compatible with
//!   Lemire's reference C library and the original StreamVByte paper.
//! - [`U32Variant0124`]: tag encodes 0/1/2/4 data bytes; zero values consume
//!   no data bytes, making this more compact for sparse (mostly-zero) input.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

mod scalar;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod shuffle;
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod sse2;
#[cfg(target_arch = "aarch64")]
mod neon;

// ── U32Classic ────────────────────────────────────────────────────────────────

fn dispatch_encode_classic(values: &[u32], out: &mut Vec<u8>) {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::encode_into_classic(values, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::encode_into_classic(values, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::encode_into_classic(values, out) };
    }

    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon"))
    ))]
    {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed at runtime.
                return unsafe { avx2::encode_into_classic(values, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::encode_into_classic(values, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::encode_into_classic(values, out) };
        }
    }

    scalar::encode_into_classic(values, out)
}

fn dispatch_decode_classic(data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::decode_into_classic(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::decode_into_classic(data, n, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::decode_into_classic(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon"))
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
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::decode_into_classic(data, n, out) };
        }
    }

    scalar::decode_into_classic(data, n, out)
}

/// StreamVByte codec for `u32` values using 2-bit tags encoding 1, 2, 3, or 4 data bytes per value.
///
/// Wire-compatible with Lemire's C library and the original StreamVByte paper.
/// All non-zero values use at least 1 data byte; use [`U32Variant0124`] if
/// your data contains many zeros and you want 0-byte encoding for them.
///
/// # Examples
///
/// ```
/// # use svb::u32::U32Classic;
/// let values: Vec<u32> = vec![0, 1, 256, 65536, u32::MAX];
/// let encoded = U32Classic.encode(&values);
/// let decoded = U32Classic.decode(&encoded, values.len()).unwrap();
/// assert_eq!(decoded, values);
/// ```
pub struct U32Classic;

impl U32Classic {
    /// Encode `values` and return a new `Vec<u8>` containing the control stream followed by the data stream.
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_classic(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_classic(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u32>`.
    ///
    /// `n` must equal the number of values that were originally encoded; a wrong
    /// value will produce incorrect output or a [`DecodeError`].
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_classic(data, n, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` values from `data`, appending them to `out`.
    ///
    /// `n` must equal the number of values that were originally encoded; a wrong
    /// value will produce incorrect output or a [`DecodeError`].
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
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::encode_into_0124(values, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::encode_into_0124(values, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::encode_into_0124(values, out) };
    }

    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon"))
    ))]
    {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed at runtime.
                return unsafe { avx2::encode_into_0124(values, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::encode_into_0124(values, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::encode_into_0124(values, out) };
        }
    }

    scalar::encode_into_0124(values, out)
}

fn dispatch_decode_0124(data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::decode_into_0124(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::decode_into_0124(data, n, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::decode_into_0124(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon"))
    ))]
    {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed at runtime.
                return unsafe { avx2::decode_into_0124(data, n, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::decode_into_0124(data, n, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::decode_into_0124(data, n, out) };
        }
    }

    scalar::decode_into_0124(data, n, out)
}

/// StreamVByte codec for `u32` values using 2-bit tags encoding 0, 1, 2, or 4 data bytes per value.
///
/// The key difference from [`U32Classic`] is that the value `0` encodes with
/// 0 data bytes (tag `00`), making this variant substantially more compact for
/// sparse data sets where most values are zero.  Note that there is no 3-byte
/// width; values in `0x10000..=0xFFFFFFFF` use 4 bytes.
///
/// # Examples
///
/// ```
/// # use svb::u32::U32Variant0124;
/// let values: Vec<u32> = vec![0, 0, 1, 0, 65536, 0];
/// let encoded = U32Variant0124.encode(&values);
/// let decoded = U32Variant0124.decode(&encoded, values.len()).unwrap();
/// assert_eq!(decoded, values);
/// ```
pub struct U32Variant0124;

impl U32Variant0124 {
    /// Encode `values` and return a new `Vec<u8>` containing the control stream followed by the data stream.
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_0124(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_0124(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u32>`.
    ///
    /// `n` must equal the number of values that were originally encoded; a wrong
    /// value will produce incorrect output or a [`DecodeError`].
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_0124(data, n, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` values from `data`, appending them to `out`.
    ///
    /// `n` must equal the number of values that were originally encoded; a wrong
    /// value will produce incorrect output or a [`DecodeError`].
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

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86_0124 {
        use super::super::{avx2, scalar, sse2};
        use std::vec::Vec;

        fn encode(values: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_0124(values, &mut out);
            out
        }

        fn scalar_decode(data: &[u8], n: usize) -> Vec<u32> {
            let mut out = Vec::new();
            scalar::decode_into_0124(data, n, &mut out).unwrap();
            out
        }

        fn ssse3_decode(data: &[u8], n: usize) -> Option<Vec<u32>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::decode_into_0124(data, n, &mut out).unwrap() };
            Some(out)
        }

        fn avx2_decode(data: &[u8], n: usize) -> Option<Vec<u32>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::decode_into_0124(data, n, &mut out).unwrap() };
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

        // exhaustive coverage of all 256 ctrl byte patterns
        #[test]
        fn all_ctrl_byte_values_0124() {
            for ctrl in 0u8..=255 {
                let values: Vec<u32> = (0..4u32)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => 0,          // tag 0: 0 bytes
                        1 => i + 1,      // tag 1: 1 byte (1..=255)
                        2 => 256 + i,    // tag 2: 2 bytes
                        _ => 65536 + i,  // tag 3: 4 bytes
                    })
                    .collect();
                check(&values);
            }
        }

        #[test]
        fn all_tail_lengths_0124() {
            if ssse3_decode(&encode(&[1u32]), 1).is_none() {
                return;
            }
            let pool: Vec<u32> = (0..20u32)
                .map(|i| match i % 4 {
                    0 => 0,
                    1 => i + 1,
                    2 => 256 + i,
                    _ => 65536 + i,
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

        #[test]
        fn homogeneous_tags_0124() {
            // all zeros (tag 0), all 1-byte, all 2-byte, all 4-byte
            for (tag, base) in [(0, 0u32), (1, 1), (2, 256), (3, 65536)] {
                let _ = tag;
                let values: Vec<u32> = (0..32).map(|i| base + i).collect();
                check(&values);
            }
        }

        // 16-byte boundary guard: 4 tag-3 values exhaust the SSSE3 16-byte window
        #[test]
        fn ssse3_16byte_boundary_guard_0124() {
            let block1: Vec<u32> = (65536..65540).collect(); // 4 × tag-3
            let block2: Vec<u32> = vec![0, 1, 256, 65536];  // mixed tags
            check(&block1.into_iter().chain(block2).collect::<Vec<_>>());
        }

        // 32-byte boundary guard: 8 tag-3 values exhaust the AVX2 32-byte window
        #[test]
        fn avx2_32byte_boundary_guard_0124() {
            let block12: Vec<u32> = (65536..65544).collect(); // 8 × tag-3
            let block3: Vec<u32> = vec![0, 1, 256, 65536];
            check(&block12.into_iter().chain(block3).collect::<Vec<_>>());
        }

        #[test]
        fn all_zeros_0124() {
            // all-zero input: ctrl bytes only, zero data bytes
            check(&vec![0u32; 32]);
        }

        #[test]
        fn boundary_values_0124() {
            let values: Vec<u32> = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX]
                .iter()
                .copied()
                .cycle()
                .take(32)
                .collect();
            check(&values);
        }

        #[test]
        fn empty_and_single_0124() {
            check(&[]);
            for &v in &[0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX] {
                check(&[v]);
            }
        }
    }

    // ── x86 encode cross-path tests ───────────────────────────────────────────
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86_encode {
        use super::super::{avx2, scalar, sse2};
        use std::vec::Vec;

        fn scalar_enc_classic(v: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_classic(v, &mut out);
            out
        }
        fn ssse3_enc_classic(v: &[u32]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::encode_into_classic(v, &mut out) };
            Some(out)
        }
        fn avx2_enc_classic(v: &[u32]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::encode_into_classic(v, &mut out) };
            Some(out)
        }
        fn scalar_enc_0124(v: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_0124(v, &mut out);
            out
        }
        fn ssse3_enc_0124(v: &[u32]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::encode_into_0124(v, &mut out) };
            Some(out)
        }
        fn avx2_enc_0124(v: &[u32]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::encode_into_0124(v, &mut out) };
            Some(out)
        }

        fn check_classic(values: &[u32]) {
            let expected = scalar_enc_classic(values);
            if let Some(got) = ssse3_enc_classic(values) {
                assert_eq!(expected, got, "SSSE3 classic encode mismatch n={}", values.len());
            }
            if let Some(got) = avx2_enc_classic(values) {
                assert_eq!(expected, got, "AVX2 classic encode mismatch n={}", values.len());
            }
        }
        fn check_0124(values: &[u32]) {
            let expected = scalar_enc_0124(values);
            if let Some(got) = ssse3_enc_0124(values) {
                assert_eq!(expected, got, "SSSE3 0124 encode mismatch n={}", values.len());
            }
            if let Some(got) = avx2_enc_0124(values) {
                assert_eq!(expected, got, "AVX2 0124 encode mismatch n={}", values.len());
            }
        }

        #[test]
        fn all_ctrl_byte_values_classic_enc() {
            for ctrl in 0u8..=255 {
                let values: Vec<u32> = (0..4u32)
                    .map(|i| match (ctrl >> (2 * i)) & 3 {
                        0 => i,
                        1 => 256 + i,
                        2 => 65536 + i,
                        _ => 16_777_216 + i,
                    })
                    .collect();
                check_classic(&values);
            }
        }
        #[test]
        fn all_ctrl_byte_values_0124_enc() {
            for ctrl in 0u8..=255 {
                let values: Vec<u32> = (0..4u32)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => 0,
                        1 => i + 1,
                        2 => 256 + i,
                        _ => 65536 + i,
                    })
                    .collect();
                check_0124(&values);
            }
        }
        #[test]
        fn all_tail_lengths_classic_enc() {
            if ssse3_enc_classic(&[1u32]).is_none() {
                return;
            }
            let pool: Vec<u32> = (0..20u32)
                .map(|i| match i % 4 {
                    0 => i,
                    1 => 256 + i,
                    2 => 65536 + i,
                    _ => 16_777_216 + i,
                })
                .collect();
            for n in 0..=20usize {
                check_classic(&pool[..n]);
            }
        }
        #[test]
        fn all_tail_lengths_0124_enc() {
            if ssse3_enc_0124(&[1u32]).is_none() {
                return;
            }
            let pool: Vec<u32> = (0..20u32)
                .map(|i| match i % 4 {
                    0 => 0,
                    1 => i + 1,
                    2 => 256 + i,
                    _ => 65536 + i,
                })
                .collect();
            for n in 0..=20usize {
                check_0124(&pool[..n]);
            }
        }
        #[test]
        fn homogeneous_tags_classic_enc() {
            for base in [0u32, 256, 65536, 16_777_216] {
                let values: Vec<u32> = (0..32).map(|i| base + i).collect();
                check_classic(&values);
            }
        }
        #[test]
        fn homogeneous_tags_0124_enc() {
            for base in [0u32, 1, 256, 65536] {
                let values: Vec<u32> = (0..32).map(|i| base + i).collect();
                check_0124(&values);
            }
        }
        #[test]
        fn boundary_values_classic_enc() {
            let values: Vec<u32> = [0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX]
                .iter().copied().cycle().take(32).collect();
            check_classic(&values);
        }
        #[test]
        fn boundary_values_0124_enc() {
            let values: Vec<u32> = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX]
                .iter().copied().cycle().take(32).collect();
            check_0124(&values);
        }
        #[test]
        fn large_input_classic_enc() {
            let values: Vec<u32> = (0..10_000u32)
                .map(|i| match i % 4 {
                    0 => i % 256,
                    1 => 256 + i % 65536,
                    2 => 65536 + i % 16_777_216,
                    _ => 16_777_216 + i,
                })
                .collect();
            check_classic(&values);
        }
        #[test]
        fn large_input_0124_enc() {
            let values: Vec<u32> = (0..10_000u32)
                .map(|i| match i % 4 {
                    0 => 0,
                    1 => (i % 255) + 1,
                    2 => 256 + i % 65536,
                    _ => 65536 + i,
                })
                .collect();
            check_0124(&values);
        }
        #[test]
        fn empty_and_single_classic_enc() {
            check_classic(&[]);
            for &v in &[0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX] {
                check_classic(&[v]);
            }
        }
        #[test]
        fn empty_and_single_0124_enc() {
            check_0124(&[]);
            for &v in &[0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX] {
                check_0124(&[v]);
            }
        }
    }

    // ── NEON cross-path tests (aarch64) ───────────────────────────────────────
    #[cfg(target_arch = "aarch64")]
    mod arm {
        use super::super::{neon, scalar};
        use std::vec::Vec;

        fn scalar_enc_classic(v: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_classic(v, &mut out);
            out
        }
        fn neon_enc_classic(v: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            unsafe { neon::encode_into_classic(v, &mut out) };
            out
        }
        fn scalar_dec_classic(d: &[u8], n: usize) -> Vec<u32> {
            let mut out = Vec::new();
            scalar::decode_into_classic(d, n, &mut out).unwrap();
            out
        }
        fn neon_dec_classic(d: &[u8], n: usize) -> Vec<u32> {
            let mut out = Vec::new();
            unsafe { neon::decode_into_classic(d, n, &mut out).unwrap() };
            out
        }
        fn scalar_enc_0124(v: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_0124(v, &mut out);
            out
        }
        fn neon_enc_0124(v: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            unsafe { neon::encode_into_0124(v, &mut out) };
            out
        }
        fn scalar_dec_0124(d: &[u8], n: usize) -> Vec<u32> {
            let mut out = Vec::new();
            scalar::decode_into_0124(d, n, &mut out).unwrap();
            out
        }
        fn neon_dec_0124(d: &[u8], n: usize) -> Vec<u32> {
            let mut out = Vec::new();
            unsafe { neon::decode_into_0124(d, n, &mut out).unwrap() };
            out
        }

        fn check_classic(values: &[u32]) {
            let n = values.len();
            let scalar_enc = scalar_enc_classic(values);
            let neon_enc = neon_enc_classic(values);
            assert_eq!(scalar_enc, neon_enc, "NEON classic encode mismatch n={n}");
            let scalar_dec = scalar_dec_classic(&scalar_enc, n);
            let neon_dec = neon_dec_classic(&scalar_enc, n);
            assert_eq!(scalar_dec, neon_dec, "NEON classic decode mismatch n={n}");
        }
        fn check_0124(values: &[u32]) {
            let n = values.len();
            let scalar_enc = scalar_enc_0124(values);
            let neon_enc = neon_enc_0124(values);
            assert_eq!(scalar_enc, neon_enc, "NEON 0124 encode mismatch n={n}");
            let scalar_dec = scalar_dec_0124(&scalar_enc, n);
            let neon_dec = neon_dec_0124(&scalar_enc, n);
            assert_eq!(scalar_dec, neon_dec, "NEON 0124 decode mismatch n={n}");
        }

        #[test]
        fn all_ctrl_byte_values_classic() {
            for ctrl in 0u8..=255 {
                let values: Vec<u32> = (0..4u32)
                    .map(|i| match (ctrl >> (2 * i)) & 3 {
                        0 => i,
                        1 => 256 + i,
                        2 => 65536 + i,
                        _ => 16_777_216 + i,
                    })
                    .collect();
                check_classic(&values);
            }
        }
        #[test]
        fn all_ctrl_byte_values_0124() {
            for ctrl in 0u8..=255 {
                let values: Vec<u32> = (0..4u32)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => 0,
                        1 => i + 1,
                        2 => 256 + i,
                        _ => 65536 + i,
                    })
                    .collect();
                check_0124(&values);
            }
        }
        #[test]
        fn all_tail_lengths_classic() {
            let pool: Vec<u32> = (0..20u32)
                .map(|i| match i % 4 {
                    0 => i,
                    1 => 256 + i,
                    2 => 65536 + i,
                    _ => 16_777_216 + i,
                })
                .collect();
            for n in 0..=20usize {
                check_classic(&pool[..n]);
            }
        }
        #[test]
        fn all_tail_lengths_0124() {
            let pool: Vec<u32> = (0..20u32)
                .map(|i| match i % 4 {
                    0 => 0,
                    1 => i + 1,
                    2 => 256 + i,
                    _ => 65536 + i,
                })
                .collect();
            for n in 0..=20usize {
                check_0124(&pool[..n]);
            }
        }
        #[test]
        fn homogeneous_tags_classic() {
            for base in [0u32, 256, 65536, 16_777_216] {
                let values: Vec<u32> = (0..32).map(|i| base + i).collect();
                check_classic(&values);
            }
        }
        #[test]
        fn homogeneous_tags_0124() {
            for base in [0u32, 1, 256, 65536] {
                let values: Vec<u32> = (0..32).map(|i| base + i).collect();
                check_0124(&values);
            }
        }
        #[test]
        fn large_input_classic() {
            let values: Vec<u32> = (0..10_000u32)
                .map(|i| match i % 4 {
                    0 => i % 256,
                    1 => 256 + i % 65536,
                    2 => 65536 + i % 16_777_216,
                    _ => 16_777_216 + i,
                })
                .collect();
            check_classic(&values);
        }
        #[test]
        fn large_input_0124() {
            let values: Vec<u32> = (0..10_000u32)
                .map(|i| match i % 4 {
                    0 => 0,
                    1 => (i % 255) + 1,
                    2 => 256 + i % 65536,
                    _ => 65536 + i,
                })
                .collect();
            check_0124(&values);
        }
        #[test]
        fn empty_and_single_classic() {
            check_classic(&[]);
            for &v in &[0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX] {
                check_classic(&[v]);
            }
        }
        #[test]
        fn empty_and_single_0124() {
            check_0124(&[]);
            for &v in &[0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX] {
                check_0124(&[v]);
            }
        }
    }
}
