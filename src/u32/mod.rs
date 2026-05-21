//! StreamVByte codecs for `u32` values using 2-bit control tags.
//!
//! Two variants are provided:
//!
//! - [`U32Classic`]: tag encodes 1/2/3/4 data bytes; wire-compatible with
//!   Lemire's reference C library and the original StreamVByte paper.
//! - [`U32Variant0124`]: tag encodes 0/1/2/4 data bytes; zero values consume
//!   no data bytes, making this more compact for sparse (mostly-zero) input.
//!
//! # Format
//!
//! ```text
//! [ ctrl_0 | ctrl_1 | … | ctrl_{ceil(n/4)-1} | data bytes … ]
//! ```
//!
//! The control stream occupies `ceil(n / 4)` bytes and precedes the data stream.
//! Within each control byte, two bits `[2k+1 : 2k]` encode the tag for the
//! `k`-th value in that group of four (bits 1:0 = value 0, bits 3:2 = value 1,
//! bits 5:4 = value 2, bits 7:6 = value 3).
//!
//! Tag encoding:
//!
//! | Tag | Classic bytes | Variant0124 bytes |
//! |-----|--------------|-------------------|
//! | 0   | 1            | 0 (value is zero) |
//! | 1   | 2            | 1                 |
//! | 2   | 3            | 2                 |
//! | 3   | 4            | 4                 |
//!
//! Data bytes follow in the same order as the values, with no padding between them.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod shuffle;
#[cfg(target_arch = "x86_64")]
mod sse2;

// ── U32Classic ────────────────────────────────────────────────────────────────

impl_dispatch_encode!(
    dispatch_encode_classic,
    u32,
    avx2::encode_into_classic,
    sse2::encode_into_classic,
    neon::encode_into_classic,
    scalar::encode_into_classic
);
impl_dispatch_decode!(
    dispatch_decode_classic,
    u32,
    avx2::decode_into_classic,
    sse2::decode_into_classic,
    neon::decode_into_classic,
    scalar::decode_into_classic
);

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
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Classic;
    /// let bytes = U32Classic.encode(&[0u32, 255, 256, 65536]);
    /// assert_eq!(U32Classic.decode(&bytes, 4).unwrap(), [0u32, 255, 256, 65536]);
    /// ```
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_classic(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Classic;
    /// let mut buf = Vec::new();
    /// U32Classic.encode_into(&[1u32, 2], &mut buf);
    /// U32Classic.encode_into(&[3u32, 4], &mut buf);
    /// ```
    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_classic(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u32>`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant (e.g., using [`U32Classic`] to decode data
    /// encoded by [`U32Variant0124`]) silently produces corrupt output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Classic;
    /// let bytes = U32Classic.encode(&[1u32, 65536, u32::MAX]);
    /// assert_eq!(U32Classic.decode(&bytes, 3).unwrap(), [1u32, 65536, u32::MAX]);
    /// ```
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_classic(data, n, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` values from `data`, appending them to `out`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant silently produces corrupt output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Classic;
    /// let bytes = U32Classic.encode(&[10u32, 20]);
    /// let mut out = vec![0u32];
    /// U32Classic.decode_into(&bytes, 2, &mut out).unwrap();
    /// assert_eq!(out, [0u32, 10, 20]);
    /// ```
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

impl_dispatch_encode!(
    dispatch_encode_0124,
    u32,
    avx2::encode_into_0124,
    sse2::encode_into_0124,
    neon::encode_into_0124,
    scalar::encode_into_0124
);
impl_dispatch_decode!(
    dispatch_decode_0124,
    u32,
    avx2::decode_into_0124,
    sse2::decode_into_0124,
    neon::decode_into_0124,
    scalar::decode_into_0124
);

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
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Variant0124;
    /// let bytes = U32Variant0124.encode(&[0u32, 0, 1, 256]);
    /// assert_eq!(U32Variant0124.decode(&bytes, 4).unwrap(), [0u32, 0, 1, 256]);
    /// ```
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_0124(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Variant0124;
    /// let mut buf = Vec::new();
    /// U32Variant0124.encode_into(&[0u32, 1], &mut buf);
    /// U32Variant0124.encode_into(&[0u32, 2], &mut buf);
    /// ```
    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_0124(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u32>`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant (e.g., using [`U32Classic`] to decode data
    /// encoded by [`U32Variant0124`]) silently produces corrupt output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Variant0124;
    /// let bytes = U32Variant0124.encode(&[0u32, 1, 65536]);
    /// assert_eq!(U32Variant0124.decode(&bytes, 3).unwrap(), [0u32, 1, 65536]);
    /// ```
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_0124(data, n, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` values from `data`, appending them to `out`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant silently produces corrupt output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u32::U32Variant0124;
    /// let bytes = U32Variant0124.encode(&[0u32, 1]);
    /// let mut out = vec![99u32];
    /// U32Variant0124.decode_into(&bytes, 2, &mut out).unwrap();
    /// assert_eq!(out, [99u32, 0, 1]);
    /// ```
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
    // ── x86 decode ────────────────────────────────────────────────────────────
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86 {
        // Generates decode cross-path tests for one u32 codec variant.
        // To add a new test case, add it once here; both variants are covered.
        macro_rules! impl_decode_tests {
            (
                scalar_enc   = $scalar_enc:path,
                ssse3_dec    = $ssse3_dec:path,
                avx2_dec     = $avx2_dec:path,
                scalar_dec   = $scalar_dec:path,
                val_arms     = |$iv:ident| { $($va:tt)+ },
                homog_bases  = [$($hb:expr),+],
                guard1_range = $g1r:expr,
                guard1_mixed = [$($gm1:expr),+],
                guard2_range = $g2r:expr,
                guard2_mixed = [$($gm2:expr),+],
                boundary     = [$($bv:expr),+],
                large_arms   = |$il:ident| { $($la:tt)+ },
                single_vals  = [$($sv:expr),+] $(,)?
            ) => {
                use std::vec::Vec;
                fn encode(values: &[u32]) -> Vec<u8> {
                    let mut out = Vec::new();
                    $scalar_enc(values, &mut out);
                    out
                }
                fn scalar_decode(data: &[u8], n: usize) -> Vec<u32> {
                    let mut out = Vec::new();
                    $scalar_dec(data, n, &mut out).unwrap();
                    out
                }
                fn ssse3_decode(data: &[u8], n: usize) -> Option<Vec<u32>> {
                    if !is_x86_feature_detected!("ssse3") { return None; }
                    let mut out = Vec::new();
                    unsafe { $ssse3_dec(data, n, &mut out).unwrap() };
                    Some(out)
                }
                fn avx2_decode(data: &[u8], n: usize) -> Option<Vec<u32>> {
                    if !is_x86_feature_detected!("avx2") { return None; }
                    let mut out = Vec::new();
                    unsafe { $avx2_dec(data, n, &mut out).unwrap() };
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
                #[test]
                fn all_ctrl_byte_values() {
                    for ctrl in 0u8..=255 {
                        let values: Vec<u32> = (0..4u32)
                            .map(|$iv| match (ctrl >> (2 * $iv)) & 3 { $($va)+ })
                            .collect();
                        check(&values);
                    }
                }
                #[test]
                fn all_tail_lengths() {
                    if ssse3_decode(&encode(&[1u32]), 1).is_none() { return; }
                    let pool: Vec<u32> = (0..20u32)
                        .map(|$iv| match $iv % 4 { $($va)+ })
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
                fn homogeneous_tags() {
                    for base in [$($hb),+] {
                        let values: Vec<u32> = (0u32..32).map(|i| base + i).collect();
                        check(&values);
                    }
                }
                #[test]
                fn ssse3_16byte_boundary_guard() {
                    let block1: Vec<u32> = ($g1r).collect();
                    let block2: Vec<u32> = vec![$($gm1),+];
                    check(&block1.into_iter().chain(block2).collect::<Vec<_>>());
                }
                #[test]
                fn avx2_32byte_boundary_guard() {
                    let block12: Vec<u32> = ($g2r).collect();
                    let block3: Vec<u32> = vec![$($gm2),+];
                    check(&block12.into_iter().chain(block3).collect::<Vec<_>>());
                }
                #[test]
                fn boundary_values() {
                    let values: Vec<u32> = [$($bv),+]
                        .iter().copied().cycle().take(32).collect();
                    check(&values);
                }
                #[test]
                fn large_input() {
                    let values: Vec<u32> = (0..10_000u32)
                        .map(|$il| match $il % 4 { $($la)+ })
                        .collect();
                    check(&values);
                }
                #[test]
                fn empty_and_single() {
                    check(&[]);
                    for &v in &[$($sv),+] { check(&[v]); }
                }
            };
        }

        mod classic {
            use super::super::super::{avx2, scalar, sse2};
            impl_decode_tests!(
                scalar_enc   = scalar::encode_into_classic,
                ssse3_dec    = sse2::decode_into_classic,
                avx2_dec     = avx2::decode_into_classic,
                scalar_dec   = scalar::decode_into_classic,
                val_arms     = |i| {
                    0 => i,
                    1 => 256 + i,
                    2 => 65536 + i,
                    _ => 16_777_216 + i,
                },
                homog_bases  = [0u32, 256, 65536, 16_777_216],
                guard1_range = 16_777_216u32..16_777_220,
                guard1_mixed = [1u32, 2, 3, 4],
                guard2_range = 16_777_216u32..16_777_224,
                guard2_mixed = [1u32, 256, 65536, 16_777_216],
                boundary     = [0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX],
                large_arms   = |i| {
                    0 => i % 256,
                    1 => 256 + i % 65536,
                    2 => 65536 + i % 16_777_216,
                    _ => 16_777_216 + i,
                },
                single_vals  = [0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX],
            );
        }

        mod variant0124 {
            use super::super::super::{avx2, scalar, sse2};
            impl_decode_tests!(
                scalar_enc   = scalar::encode_into_0124,
                ssse3_dec    = sse2::decode_into_0124,
                avx2_dec     = avx2::decode_into_0124,
                scalar_dec   = scalar::decode_into_0124,
                val_arms     = |i| {
                    0 => 0,
                    1 => i + 1,
                    2 => 256 + i,
                    _ => 65536 + i,
                },
                homog_bases  = [0u32, 1, 256, 65536],
                guard1_range = 65536u32..65540,
                guard1_mixed = [0u32, 1, 256, 65536],
                guard2_range = 65536u32..65544,
                guard2_mixed = [0u32, 1, 256, 65536],
                boundary     = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX],
                large_arms   = |i| {
                    0 => 0,
                    1 => (i % 255) + 1,
                    2 => 256 + i % 65536,
                    _ => 65536 + i,
                },
                single_vals  = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX],
            );
            #[test]
            fn all_zeros() {
                check(&vec![0u32; 32]);
            }
        }
    }

    // ── x86 encode ────────────────────────────────────────────────────────────
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86_encode {
        // Generates encode cross-path tests for one u32 codec variant.
        macro_rules! impl_encode_tests {
            (
                scalar_enc  = $scalar_enc:path,
                ssse3_enc   = $ssse3_enc:path,
                avx2_enc    = $avx2_enc:path,
                val_arms    = |$iv:ident| { $($va:tt)+ },
                homog_bases = [$($hb:expr),+],
                boundary    = [$($bv:expr),+],
                large_arms  = |$il:ident| { $($la:tt)+ },
                single_vals = [$($sv:expr),+] $(,)?
            ) => {
                use std::vec::Vec;
                fn scalar_enc_fn(v: &[u32]) -> Vec<u8> {
                    let mut out = Vec::new();
                    $scalar_enc(v, &mut out);
                    out
                }
                fn ssse3_enc_fn(v: &[u32]) -> Option<Vec<u8>> {
                    if !is_x86_feature_detected!("ssse3") { return None; }
                    let mut out = Vec::new();
                    unsafe { $ssse3_enc(v, &mut out) };
                    Some(out)
                }
                fn avx2_enc_fn(v: &[u32]) -> Option<Vec<u8>> {
                    if !is_x86_feature_detected!("avx2") { return None; }
                    let mut out = Vec::new();
                    unsafe { $avx2_enc(v, &mut out) };
                    Some(out)
                }
                fn check(values: &[u32]) {
                    let expected = scalar_enc_fn(values);
                    let n = values.len();
                    if let Some(got) = ssse3_enc_fn(values) {
                        assert_eq!(expected, got, "SSSE3 encode mismatch n={n}");
                    }
                    if let Some(got) = avx2_enc_fn(values) {
                        assert_eq!(expected, got, "AVX2 encode mismatch n={n}");
                    }
                }
                #[test]
                fn all_ctrl_byte_values() {
                    for ctrl in 0u8..=255 {
                        let values: Vec<u32> = (0..4u32)
                            .map(|$iv| match (ctrl >> (2 * $iv)) & 3 { $($va)+ })
                            .collect();
                        check(&values);
                    }
                }
                #[test]
                fn all_tail_lengths() {
                    if ssse3_enc_fn(&[1u32]).is_none() { return; }
                    let pool: Vec<u32> = (0..20u32)
                        .map(|$iv| match $iv % 4 { $($va)+ })
                        .collect();
                    for n in 0..=20usize { check(&pool[..n]); }
                }
                #[test]
                fn homogeneous_tags() {
                    for base in [$($hb),+] {
                        let values: Vec<u32> = (0u32..32).map(|i| base + i).collect();
                        check(&values);
                    }
                }
                #[test]
                fn boundary_values() {
                    let values: Vec<u32> = [$($bv),+]
                        .iter().copied().cycle().take(32).collect();
                    check(&values);
                }
                #[test]
                fn large_input() {
                    let values: Vec<u32> = (0..10_000u32)
                        .map(|$il| match $il % 4 { $($la)+ })
                        .collect();
                    check(&values);
                }
                #[test]
                fn empty_and_single() {
                    check(&[]);
                    for &v in &[$($sv),+] { check(&[v]); }
                }
            };
        }

        mod classic {
            use super::super::super::{avx2, scalar, sse2};
            impl_encode_tests!(
                scalar_enc  = scalar::encode_into_classic,
                ssse3_enc   = sse2::encode_into_classic,
                avx2_enc    = avx2::encode_into_classic,
                val_arms    = |i| {
                    0 => i,
                    1 => 256 + i,
                    2 => 65536 + i,
                    _ => 16_777_216 + i,
                },
                homog_bases = [0u32, 256, 65536, 16_777_216],
                boundary    = [0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX],
                large_arms  = |i| {
                    0 => i % 256,
                    1 => 256 + i % 65536,
                    2 => 65536 + i % 16_777_216,
                    _ => 16_777_216 + i,
                },
                single_vals = [0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX],
            );
        }

        mod variant0124 {
            use super::super::super::{avx2, scalar, sse2};
            impl_encode_tests!(
                scalar_enc  = scalar::encode_into_0124,
                ssse3_enc   = sse2::encode_into_0124,
                avx2_enc    = avx2::encode_into_0124,
                val_arms    = |i| {
                    0 => 0,
                    1 => i + 1,
                    2 => 256 + i,
                    _ => 65536 + i,
                },
                homog_bases = [0u32, 1, 256, 65536],
                boundary    = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX],
                large_arms  = |i| {
                    0 => 0,
                    1 => (i % 255) + 1,
                    2 => 256 + i % 65536,
                    _ => 65536 + i,
                },
                single_vals = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX],
            );
        }
    }

    // ── NEON ──────────────────────────────────────────────────────────────────
    #[cfg(target_arch = "aarch64")]
    mod arm {
        // Generates NEON encode+decode cross-path tests for one u32 codec variant.
        macro_rules! impl_neon_tests {
            (
                scalar_enc  = $scalar_enc:path,
                neon_enc    = $neon_enc:path,
                scalar_dec  = $scalar_dec:path,
                neon_dec    = $neon_dec:path,
                val_arms    = |$iv:ident| { $($va:tt)+ },
                homog_bases = [$($hb:expr),+],
                large_arms  = |$il:ident| { $($la:tt)+ },
                single_vals = [$($sv:expr),+] $(,)?
            ) => {
                use std::vec::Vec;
                fn scalar_enc_fn(v: &[u32]) -> Vec<u8> {
                    let mut out = Vec::new();
                    $scalar_enc(v, &mut out);
                    out
                }
                fn neon_enc_fn(v: &[u32]) -> Vec<u8> {
                    let mut out = Vec::new();
                    unsafe { $neon_enc(v, &mut out) };
                    out
                }
                fn scalar_dec_fn(d: &[u8], n: usize) -> Vec<u32> {
                    let mut out = Vec::new();
                    $scalar_dec(d, n, &mut out).unwrap();
                    out
                }
                fn neon_dec_fn(d: &[u8], n: usize) -> Vec<u32> {
                    let mut out = Vec::new();
                    unsafe { $neon_dec(d, n, &mut out).unwrap() };
                    out
                }
                fn check(values: &[u32]) {
                    let n = values.len();
                    let scalar_enc = scalar_enc_fn(values);
                    let neon_enc = neon_enc_fn(values);
                    assert_eq!(scalar_enc, neon_enc, "NEON encode mismatch n={n}");
                    let scalar_dec = scalar_dec_fn(&scalar_enc, n);
                    let neon_dec = neon_dec_fn(&scalar_enc, n);
                    assert_eq!(scalar_dec, neon_dec, "NEON decode mismatch n={n}");
                }
                #[test]
                fn all_ctrl_byte_values() {
                    for ctrl in 0u8..=255 {
                        let values: Vec<u32> = (0..4u32)
                            .map(|$iv| match (ctrl >> (2 * $iv)) & 3 { $($va)+ })
                            .collect();
                        check(&values);
                    }
                }
                #[test]
                fn all_tail_lengths() {
                    let pool: Vec<u32> = (0..20u32)
                        .map(|$iv| match $iv % 4 { $($va)+ })
                        .collect();
                    for n in 0..=20usize { check(&pool[..n]); }
                }
                #[test]
                fn homogeneous_tags() {
                    for base in [$($hb),+] {
                        let values: Vec<u32> = (0u32..32).map(|i| base + i).collect();
                        check(&values);
                    }
                }
                #[test]
                fn large_input() {
                    let values: Vec<u32> = (0..10_000u32)
                        .map(|$il| match $il % 4 { $($la)+ })
                        .collect();
                    check(&values);
                }
                #[test]
                fn empty_and_single() {
                    check(&[]);
                    for &v in &[$($sv),+] { check(&[v]); }
                }
            };
        }

        mod classic {
            use super::super::super::{neon, scalar};
            impl_neon_tests!(
                scalar_enc  = scalar::encode_into_classic,
                neon_enc    = neon::encode_into_classic,
                scalar_dec  = scalar::decode_into_classic,
                neon_dec    = neon::decode_into_classic,
                val_arms    = |i| {
                    0 => i,
                    1 => 256 + i,
                    2 => 65536 + i,
                    _ => 16_777_216 + i,
                },
                homog_bases = [0u32, 256, 65536, 16_777_216],
                large_arms  = |i| {
                    0 => i % 256,
                    1 => 256 + i % 65536,
                    2 => 65536 + i % 16_777_216,
                    _ => 16_777_216 + i,
                },
                single_vals = [0u32, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX],
            );
        }

        mod variant0124 {
            use super::super::super::{neon, scalar};
            impl_neon_tests!(
                scalar_enc  = scalar::encode_into_0124,
                neon_enc    = neon::encode_into_0124,
                scalar_dec  = scalar::decode_into_0124,
                neon_dec    = neon::decode_into_0124,
                val_arms    = |i| {
                    0 => 0,
                    1 => i + 1,
                    2 => 256 + i,
                    _ => 65536 + i,
                },
                homog_bases = [0u32, 1, 256, 65536],
                large_arms  = |i| {
                    0 => 0,
                    1 => (i % 255) + 1,
                    2 => 256 + i % 65536,
                    _ => 65536 + i,
                },
                single_vals = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX],
            );
        }
    }
}
