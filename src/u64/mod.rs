//! StreamVByte codecs for `u64` values using 2-bit control tags.
//!
//! Two variants are provided:
//!
//! - [`U64Coder1234`]: tag encodes 1/2/3/4 data bytes per value, matching the
//!   tag table of [`crate::u32::U32Classic`]. Values above `u32::MAX` are
//!   silently truncated; call [`U64Coder1234::check_range`] first if this
//!   matters.
//! - [`U64Coder1248`]: tag encodes 1/2/4/8 data bytes per value, covering
//!   the full `u64` range at the cost of a 4-byte gap (values in
//!   `0x10000..=0xFFFFFFFF` use 4 bytes rather than 3).

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

// ── U64Coder1234 ──────────────────────────────────────────────────────────────

fn dispatch_encode_1234(values: &[u64], out: &mut Vec<u8>) {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::encode_into_1234(values, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::encode_into_1234(values, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::encode_into_1234(values, out) };
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
                return unsafe { avx2::encode_into_1234(values, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::encode_into_1234(values, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::encode_into_1234(values, out) };
        }
    }

    scalar::encode_into_1234(values, out)
}

fn dispatch_decode_1234(data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::decode_into_1234(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::decode_into_1234(data, n, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::decode_into_1234(data, n, out) };
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
                return unsafe { avx2::decode_into_1234(data, n, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::decode_into_1234(data, n, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::decode_into_1234(data, n, out) };
        }
    }

    scalar::decode_into_1234(data, n, out)
}

/// StreamVByte codec for `u64` values using 2-bit tags encoding 1, 2, 3, or 4 data bytes per value.
///
/// Same tag/width table as [`crate::u32::U32Classic`] but operates on `u64` slices.
/// Values greater than `u32::MAX` are silently truncated to their low 32 bits on
/// encode — this matches the behaviour of other StreamVByte libraries and is
/// defined, not accidental. Call [`U64Coder1234::check_range`] before encoding if
/// you need to detect out-of-range values. For data that may genuinely exceed
/// `u32::MAX`, use [`U64Coder1248`] instead.
///
/// # Examples
///
/// ```
/// # use svb::u64::U64Coder1234;
/// let values: Vec<u64> = vec![1, 256, 65536, u32::MAX as u64];
/// let encoded = U64Coder1234.encode(&values);
/// let decoded = U64Coder1234.decode(&encoded, values.len()).unwrap();
/// assert_eq!(decoded, values);
/// ```
pub struct U64Coder1234;

impl U64Coder1234 {
    /// Returns the index of the first value that exceeds `u32::MAX`, or `None` if
    /// all values fit within the 1–4 byte encoding range.
    ///
    /// Call this before [`encode`](U64Coder1234::encode) whenever the input may
    /// contain values larger than `u32::MAX`; encoding such values silently
    /// truncates them.
    pub fn check_range(&self, values: &[u64]) -> Option<usize> {
        values.iter().position(|&v| v > u64::from(u32::MAX))
    }

    /// Encode `values` and return a new `Vec<u8>` containing the control stream followed by the data stream.
    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_1234(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1234(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u64>`.
    ///
    /// `n` must equal the number of values that were originally encoded; a wrong
    /// value will produce incorrect output or a [`DecodeError`].
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1234(data, n, &mut out)?;
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
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::encode_into_1248(values, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::encode_into_1248(values, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::encode_into_1248(values, out) };
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
                return unsafe { avx2::encode_into_1248(values, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::encode_into_1248(values, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::encode_into_1248(values, out) };
        }
    }

    scalar::encode_into_1248(values, out)
}

fn dispatch_decode_1248(data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        return unsafe { avx2::decode_into_1248(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
        return unsafe { sse2::decode_into_1248(data, n, out) };
    }

    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::decode_into_1248(data, n, out) };
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
                return unsafe { avx2::decode_into_1248(data, n, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::decode_into_1248(data, n, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::decode_into_1248(data, n, out) };
        }
    }

    scalar::decode_into_1248(data, n, out)
}

/// StreamVByte codec for `u64` values using 2-bit tags encoding 1, 2, 4, or 8 data bytes per value.
///
/// Covers the full `u64` range without truncation. There is no 3-byte width:
/// values in `0x10000..=0xFFFFFFFF` use 4 bytes, and values above `0xFFFFFFFF`
/// use 8 bytes. Use [`U64Coder1234`] instead when all values are known to fit
/// within `u32::MAX` and you want the compact 3-byte option.
///
/// # Examples
///
/// ```
/// # use svb::u64::U64Coder1248;
/// let values: Vec<u64> = vec![1, 256, 65536, u64::MAX];
/// let encoded = U64Coder1248.encode(&values);
/// let decoded = U64Coder1248.decode(&encoded, values.len()).unwrap();
/// assert_eq!(decoded, values);
/// ```
pub struct U64Coder1248;

impl U64Coder1248 {
    /// Encode `values` and return a new `Vec<u8>` containing the control stream followed by the data stream.
    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_1248(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1248(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u64>`.
    ///
    /// `n` must equal the number of values that were originally encoded; a wrong
    /// value will produce incorrect output or a [`DecodeError`].
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1248(data, n, &mut out)?;
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

    // ── x86 encode cross-path tests ───────────────────────────────────────────
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86_encode {
        use super::super::{avx2, scalar, sse2};
        use std::vec::Vec;

        fn scalar_enc_1234(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_1234(v, &mut out);
            out
        }
        fn ssse3_enc_1234(v: &[u64]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::encode_into_1234(v, &mut out) };
            Some(out)
        }
        fn avx2_enc_1234(v: &[u64]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::encode_into_1234(v, &mut out) };
            Some(out)
        }
        fn scalar_enc_1248(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_1248(v, &mut out);
            out
        }
        fn ssse3_enc_1248(v: &[u64]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::encode_into_1248(v, &mut out) };
            Some(out)
        }
        fn avx2_enc_1248(v: &[u64]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::encode_into_1248(v, &mut out) };
            Some(out)
        }

        fn check_1234(values: &[u64]) {
            let expected = scalar_enc_1234(values);
            let n = values.len();
            if let Some(got) = ssse3_enc_1234(values) {
                assert_eq!(expected, got, "SSSE3 1234 encode mismatch n={n}");
            }
            if let Some(got) = avx2_enc_1234(values) {
                assert_eq!(expected, got, "AVX2 1234 encode mismatch n={n}");
            }
        }
        fn check_1248(values: &[u64]) {
            let expected = scalar_enc_1248(values);
            let n = values.len();
            if let Some(got) = ssse3_enc_1248(values) {
                assert_eq!(expected, got, "SSSE3 1248 encode mismatch n={n}");
            }
            if let Some(got) = avx2_enc_1248(values) {
                assert_eq!(expected, got, "AVX2 1248 encode mismatch n={n}");
            }
        }

        #[test]
        fn all_ctrl_byte_values_1234_enc() {
            for ctrl in 0u8..=255 {
                let values: Vec<u64> = (0..4u64)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => i + 1,
                        1 => 0x100 + i,
                        2 => 0x10000 + i,
                        _ => 0x1000000 + i,
                    })
                    .collect();
                check_1234(&values);
            }
        }
        #[test]
        fn all_ctrl_byte_values_1248_enc() {
            for ctrl in 0u8..=255 {
                let values: Vec<u64> = (0..4u64)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => i + 1,
                        1 => 0x100 + i,
                        2 => 0x10000 + i,
                        _ => 0x1_0000_0000 + i,
                    })
                    .collect();
                check_1248(&values);
            }
        }
        #[test]
        fn all_tail_lengths_1234_enc() {
            if ssse3_enc_1234(&[1u64]).is_none() {
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
                check_1234(&pool[..n]);
            }
        }
        #[test]
        fn all_tail_lengths_1248_enc() {
            if ssse3_enc_1248(&[1u64]).is_none() {
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
                check_1248(&pool[..n]);
            }
        }
        #[test]
        fn homogeneous_tags_1234_enc() {
            for base in [1u64, 0x100, 0x10000, 0x1000000] {
                let values: Vec<u64> = (0..32).map(|i| base + i).collect();
                check_1234(&values);
            }
        }
        #[test]
        fn homogeneous_tags_1248_enc() {
            for base in [1u64, 0x100, 0x10000, 0x1_0000_0000] {
                let values: Vec<u64> = (0..32).map(|i| base + i).collect();
                check_1248(&values);
            }
        }
        #[test]
        fn boundary_values_1234_enc() {
            let values: Vec<u64> = [
                0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX as u64,
            ]
            .iter().copied().cycle().take(36).collect();
            check_1234(&values);
        }
        #[test]
        fn boundary_values_1248_enc() {
            let values: Vec<u64> = [
                0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, 0x1_0000_0000, u64::MAX,
            ]
            .iter().copied().cycle().take(36).collect();
            check_1248(&values);
        }
        #[test]
        fn large_input_1234_enc() {
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
        fn large_input_1248_enc() {
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
        fn empty_and_single_1234_enc() {
            check_1234(&[]);
            for &v in &[0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, u32::MAX as u64] {
                check_1234(&[v]);
            }
        }
        #[test]
        fn empty_and_single_1248_enc() {
            check_1248(&[]);
            for &v in &[0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, u64::MAX] {
                check_1248(&[v]);
            }
        }
    }

    // ── NEON cross-path tests (aarch64) ───────────────────────────────────────
    #[cfg(target_arch = "aarch64")]
    mod arm {
        use super::super::{neon, scalar};
        use std::vec::Vec;

        fn scalar_enc_1234(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_1234(v, &mut out);
            out
        }
        fn neon_enc_1234(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            unsafe { neon::encode_into_1234(v, &mut out) };
            out
        }
        fn scalar_dec_1234(d: &[u8], n: usize) -> Vec<u64> {
            let mut out = Vec::new();
            scalar::decode_into_1234(d, n, &mut out).unwrap();
            out
        }
        fn neon_dec_1234(d: &[u8], n: usize) -> Vec<u64> {
            let mut out = Vec::new();
            unsafe { neon::decode_into_1234(d, n, &mut out).unwrap() };
            out
        }
        fn scalar_enc_1248(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            scalar::encode_into_1248(v, &mut out);
            out
        }
        fn neon_enc_1248(v: &[u64]) -> Vec<u8> {
            let mut out = Vec::new();
            unsafe { neon::encode_into_1248(v, &mut out) };
            out
        }
        fn scalar_dec_1248(d: &[u8], n: usize) -> Vec<u64> {
            let mut out = Vec::new();
            scalar::decode_into_1248(d, n, &mut out).unwrap();
            out
        }
        fn neon_dec_1248(d: &[u8], n: usize) -> Vec<u64> {
            let mut out = Vec::new();
            unsafe { neon::decode_into_1248(d, n, &mut out).unwrap() };
            out
        }

        fn check_1234(values: &[u64]) {
            let n = values.len();
            let scalar_enc = scalar_enc_1234(values);
            let neon_enc = neon_enc_1234(values);
            assert_eq!(scalar_enc, neon_enc, "NEON 1234 encode mismatch n={n}");
            let scalar_dec = scalar_dec_1234(&scalar_enc, n);
            let neon_dec = neon_dec_1234(&scalar_enc, n);
            assert_eq!(scalar_dec, neon_dec, "NEON 1234 decode mismatch n={n}");
        }
        fn check_1248(values: &[u64]) {
            let n = values.len();
            let scalar_enc = scalar_enc_1248(values);
            let neon_enc = neon_enc_1248(values);
            assert_eq!(scalar_enc, neon_enc, "NEON 1248 encode mismatch n={n}");
            let scalar_dec = scalar_dec_1248(&scalar_enc, n);
            let neon_dec = neon_dec_1248(&scalar_enc, n);
            assert_eq!(scalar_dec, neon_dec, "NEON 1248 decode mismatch n={n}");
        }

        #[test]
        fn all_ctrl_byte_values_1234() {
            for ctrl in 0u8..=255 {
                let values: Vec<u64> = (0..4u64)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => i + 1,
                        1 => 0x100 + i,
                        2 => 0x10000 + i,
                        _ => 0x1000000 + i,
                    })
                    .collect();
                check_1234(&values);
            }
        }
        #[test]
        fn all_ctrl_byte_values_1248() {
            for ctrl in 0u8..=255 {
                let values: Vec<u64> = (0..4u64)
                    .map(|i| match (ctrl >> (2 * i as usize)) & 3 {
                        0 => i + 1,
                        1 => 0x100 + i,
                        2 => 0x10000 + i,
                        _ => 0x1_0000_0000 + i,
                    })
                    .collect();
                check_1248(&values);
            }
        }
        #[test]
        fn all_tail_lengths_1234() {
            let pool: Vec<u64> = (0..20u64)
                .map(|i| match i % 4 {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1000000 + i,
                })
                .collect();
            for n in 0..=20usize {
                check_1234(&pool[..n]);
            }
        }
        #[test]
        fn all_tail_lengths_1248() {
            let pool: Vec<u64> = (0..20u64)
                .map(|i| match i % 4 {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1_0000_0000 + i,
                })
                .collect();
            for n in 0..=20usize {
                check_1248(&pool[..n]);
            }
        }
        #[test]
        fn homogeneous_tags_1234() {
            for base in [1u64, 0x100, 0x10000, 0x1000000] {
                let values: Vec<u64> = (0..32).map(|i| base + i).collect();
                check_1234(&values);
            }
        }
        #[test]
        fn homogeneous_tags_1248() {
            for base in [1u64, 0x100, 0x10000, 0x1_0000_0000] {
                let values: Vec<u64> = (0..32).map(|i| base + i).collect();
                check_1248(&values);
            }
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
        fn empty_and_single_1234() {
            check_1234(&[]);
            for &v in &[0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, u32::MAX as u64] {
                check_1234(&[v]);
            }
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
