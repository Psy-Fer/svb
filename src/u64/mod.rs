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
//! | Tag | Coder1234 bytes | Coder1248 bytes |
//! |-----|-----------------|-----------------|
//! | 0   | 1               | 1               |
//! | 1   | 2               | 2               |
//! | 2   | 3               | 4               |
//! | 3   | 4               | 8               |
//!
//! Data bytes follow in the same order as the values, with no padding between them.

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

impl_dispatch_encode!(
    dispatch_encode_1234, u64,
    avx2::encode_into_1234, sse2::encode_into_1234,
    neon::encode_into_1234, scalar::encode_into_1234
);
impl_dispatch_decode!(
    dispatch_decode_1234, u64,
    avx2::decode_into_1234, sse2::decode_into_1234,
    neon::decode_into_1234, scalar::decode_into_1234
);

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
/// // Guard against silent truncation before encoding.
/// assert_eq!(U64Coder1234.check_range(&values), None);
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
    ///
    /// # Warning
    ///
    /// Values that exceed `u32::MAX` are silently truncated. Call
    /// [`check_range`](U64Coder1234::check_range) first if the input may contain
    /// values outside `0..=u32::MAX`.
    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_1234(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    ///
    /// # Warning
    ///
    /// Values that exceed `u32::MAX` are silently truncated. Call
    /// [`check_range`](U64Coder1234::check_range) first if the input may contain
    /// values outside `0..=u32::MAX`.
    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1234(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u64>`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant (e.g., using [`U64Coder1234`] to decode data
    /// encoded by [`U64Coder1248`]) silently produces corrupt output.
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1234(data, n, &mut out)?;
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

impl_dispatch_encode!(
    dispatch_encode_1248, u64,
    avx2::encode_into_1248, sse2::encode_into_1248,
    neon::encode_into_1248, scalar::encode_into_1248
);
impl_dispatch_decode!(
    dispatch_decode_1248, u64,
    avx2::decode_into_1248, sse2::decode_into_1248,
    neon::decode_into_1248, scalar::decode_into_1248
);

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
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant (e.g., using [`U64Coder1234`] to decode data
    /// encoded by [`U64Coder1248`]) silently produces corrupt output.
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1248(data, n, &mut out)?;
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
    // ── x86 decode ────────────────────────────────────────────────────────────
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86 {
        // Generates decode cross-path tests for one u64 codec variant.
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
                fn encode(values: &[u64]) -> Vec<u8> {
                    let mut out = Vec::new();
                    $scalar_enc(values, &mut out);
                    out
                }
                fn scalar_decode(data: &[u8], n: usize) -> Vec<u64> {
                    let mut out = Vec::new();
                    $scalar_dec(data, n, &mut out).unwrap();
                    out
                }
                fn ssse3_decode(data: &[u8], n: usize) -> Option<Vec<u64>> {
                    if !is_x86_feature_detected!("ssse3") { return None; }
                    let mut out = Vec::new();
                    unsafe { $ssse3_dec(data, n, &mut out).unwrap() };
                    Some(out)
                }
                fn avx2_decode(data: &[u8], n: usize) -> Option<Vec<u64>> {
                    if !is_x86_feature_detected!("avx2") { return None; }
                    let mut out = Vec::new();
                    unsafe { $avx2_dec(data, n, &mut out).unwrap() };
                    Some(out)
                }
                fn check(values: &[u64]) {
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
                        let values: Vec<u64> = (0..4u64)
                            .map(|$iv| match (ctrl >> (2 * $iv)) & 3 { $($va)+ })
                            .collect();
                        check(&values);
                    }
                }
                #[test]
                fn all_tail_lengths() {
                    if ssse3_decode(&encode(&[1u64]), 1).is_none() { return; }
                    let pool: Vec<u64> = (0..20u64)
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
                        let values: Vec<u64> = (0u64..32).map(|i| base + i).collect();
                        check(&values);
                    }
                }
                #[test]
                fn ssse3_boundary_guard() {
                    let block1: Vec<u64> = ($g1r).collect();
                    let block2: Vec<u64> = vec![$($gm1),+];
                    check(&block1.into_iter().chain(block2).collect::<Vec<_>>());
                }
                #[test]
                fn avx2_boundary_guard() {
                    let block12: Vec<u64> = ($g2r).collect();
                    let block3: Vec<u64> = vec![$($gm2),+];
                    check(&block12.into_iter().chain(block3).collect::<Vec<_>>());
                }
                #[test]
                fn boundary_values() {
                    let values: Vec<u64> = [$($bv),+]
                        .iter().copied().cycle().take(36).collect();
                    check(&values);
                }
                #[test]
                fn large_input() {
                    let values: Vec<u64> = (0..10_000u64)
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

        mod coder1234 {
            use super::super::super::{avx2, scalar, sse2};
            impl_decode_tests!(
                scalar_enc   = scalar::encode_into_1234,
                ssse3_dec    = sse2::decode_into_1234,
                avx2_dec     = avx2::decode_into_1234,
                scalar_dec   = scalar::decode_into_1234,
                val_arms     = |i| {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1000000 + i,
                },
                homog_bases  = [1u64, 0x100, 0x10000, 0x1000000],
                guard1_range = 0x1000000u64..0x1000004,
                guard1_mixed = [1u64, 0x100, 0x10000, 0x1000000],
                guard2_range = 0x1000000u64..0x1000008,
                guard2_mixed = [1u64, 0x100, 0x10000, 0x1000000],
                boundary     = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX as u64],
                large_arms   = |i| {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFF_FFFF,
                    _ => 0x1000000 + i,
                },
                single_vals  = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, u32::MAX as u64],
            );
        }

        mod coder1248 {
            use super::super::super::{avx2, scalar, sse2};
            impl_decode_tests!(
                scalar_enc   = scalar::encode_into_1248,
                ssse3_dec    = sse2::decode_into_1248,
                avx2_dec     = avx2::decode_into_1248,
                scalar_dec   = scalar::decode_into_1248,
                val_arms     = |i| {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1_0000_0000 + i,
                },
                homog_bases  = [1u64, 0x100, 0x10000, 0x1_0000_0000],
                guard1_range = 0x1_0000_0000u64..0x1_0000_0004,
                guard1_mixed = [1u64, 0x100, 0x10000, 0x1_0000_0000],
                guard2_range = 0x1_0000_0000u64..0x1_0000_0008,
                guard2_mixed = [1u64, 0x100, 0x10000, 0x1_0000_0000],
                boundary     = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, 0x1_0000_0000, u64::MAX],
                large_arms   = |i| {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFFFF_FFFF,
                    _ => 0x1_0000_0000 + i,
                },
                single_vals  = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, u64::MAX],
            );
        }
    }

    // ── x86 encode ────────────────────────────────────────────────────────────
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86_encode {
        // Generates encode cross-path tests for one u64 codec variant.
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
                fn scalar_enc_fn(v: &[u64]) -> Vec<u8> {
                    let mut out = Vec::new();
                    $scalar_enc(v, &mut out);
                    out
                }
                fn ssse3_enc_fn(v: &[u64]) -> Option<Vec<u8>> {
                    if !is_x86_feature_detected!("ssse3") { return None; }
                    let mut out = Vec::new();
                    unsafe { $ssse3_enc(v, &mut out) };
                    Some(out)
                }
                fn avx2_enc_fn(v: &[u64]) -> Option<Vec<u8>> {
                    if !is_x86_feature_detected!("avx2") { return None; }
                    let mut out = Vec::new();
                    unsafe { $avx2_enc(v, &mut out) };
                    Some(out)
                }
                fn check(values: &[u64]) {
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
                        let values: Vec<u64> = (0..4u64)
                            .map(|$iv| match (ctrl >> (2 * $iv)) & 3 { $($va)+ })
                            .collect();
                        check(&values);
                    }
                }
                #[test]
                fn all_tail_lengths() {
                    if ssse3_enc_fn(&[1u64]).is_none() { return; }
                    let pool: Vec<u64> = (0..20u64)
                        .map(|$iv| match $iv % 4 { $($va)+ })
                        .collect();
                    for n in 0..=20usize { check(&pool[..n]); }
                }
                #[test]
                fn homogeneous_tags() {
                    for base in [$($hb),+] {
                        let values: Vec<u64> = (0u64..32).map(|i| base + i).collect();
                        check(&values);
                    }
                }
                #[test]
                fn boundary_values() {
                    let values: Vec<u64> = [$($bv),+]
                        .iter().copied().cycle().take(36).collect();
                    check(&values);
                }
                #[test]
                fn large_input() {
                    let values: Vec<u64> = (0..10_000u64)
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

        mod coder1234 {
            use super::super::super::{avx2, scalar, sse2};
            impl_encode_tests!(
                scalar_enc  = scalar::encode_into_1234,
                ssse3_enc   = sse2::encode_into_1234,
                avx2_enc    = avx2::encode_into_1234,
                val_arms    = |i| {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1000000 + i,
                },
                homog_bases = [1u64, 0x100, 0x10000, 0x1000000],
                boundary    = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, 0x100_0000, u32::MAX as u64],
                large_arms  = |i| {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFF_FFFF,
                    _ => 0x1000000 + i,
                },
                single_vals = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, u32::MAX as u64],
            );
        }

        mod coder1248 {
            use super::super::super::{avx2, scalar, sse2};
            impl_encode_tests!(
                scalar_enc  = scalar::encode_into_1248,
                ssse3_enc   = sse2::encode_into_1248,
                avx2_enc    = avx2::encode_into_1248,
                val_arms    = |i| {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1_0000_0000 + i,
                },
                homog_bases = [1u64, 0x100, 0x10000, 0x1_0000_0000],
                boundary    = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, 0x1_0000_0000, u64::MAX],
                large_arms  = |i| {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFFFF_FFFF,
                    _ => 0x1_0000_0000 + i,
                },
                single_vals = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, u64::MAX],
            );
        }
    }

    // ── NEON ──────────────────────────────────────────────────────────────────
    #[cfg(target_arch = "aarch64")]
    mod arm {
        // Generates NEON encode+decode cross-path tests for one u64 codec variant.
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
                fn scalar_enc_fn(v: &[u64]) -> Vec<u8> {
                    let mut out = Vec::new();
                    $scalar_enc(v, &mut out);
                    out
                }
                fn neon_enc_fn(v: &[u64]) -> Vec<u8> {
                    let mut out = Vec::new();
                    unsafe { $neon_enc(v, &mut out) };
                    out
                }
                fn scalar_dec_fn(d: &[u8], n: usize) -> Vec<u64> {
                    let mut out = Vec::new();
                    $scalar_dec(d, n, &mut out).unwrap();
                    out
                }
                fn neon_dec_fn(d: &[u8], n: usize) -> Vec<u64> {
                    let mut out = Vec::new();
                    unsafe { $neon_dec(d, n, &mut out).unwrap() };
                    out
                }
                fn check(values: &[u64]) {
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
                        let values: Vec<u64> = (0..4u64)
                            .map(|$iv| match (ctrl >> (2 * $iv)) & 3 { $($va)+ })
                            .collect();
                        check(&values);
                    }
                }
                #[test]
                fn all_tail_lengths() {
                    let pool: Vec<u64> = (0..20u64)
                        .map(|$iv| match $iv % 4 { $($va)+ })
                        .collect();
                    for n in 0..=20usize { check(&pool[..n]); }
                }
                #[test]
                fn homogeneous_tags() {
                    for base in [$($hb),+] {
                        let values: Vec<u64> = (0u64..32).map(|i| base + i).collect();
                        check(&values);
                    }
                }
                #[test]
                fn large_input() {
                    let values: Vec<u64> = (0..10_000u64)
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

        mod coder1234 {
            use super::super::super::{neon, scalar};
            impl_neon_tests!(
                scalar_enc  = scalar::encode_into_1234,
                neon_enc    = neon::encode_into_1234,
                scalar_dec  = scalar::decode_into_1234,
                neon_dec    = neon::decode_into_1234,
                val_arms    = |i| {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1000000 + i,
                },
                homog_bases = [1u64, 0x100, 0x10000, 0x1000000],
                large_arms  = |i| {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFF_FFFF,
                    _ => 0x1000000 + i,
                },
                single_vals = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFF_FFFF, u32::MAX as u64],
            );
        }

        mod coder1248 {
            use super::super::super::{neon, scalar};
            impl_neon_tests!(
                scalar_enc  = scalar::encode_into_1248,
                neon_enc    = neon::encode_into_1248,
                scalar_dec  = scalar::decode_into_1248,
                neon_dec    = neon::decode_into_1248,
                val_arms    = |i| {
                    0 => i + 1,
                    1 => 0x100 + i,
                    2 => 0x10000 + i,
                    _ => 0x1_0000_0000 + i,
                },
                homog_bases = [1u64, 0x100, 0x10000, 0x1_0000_0000],
                large_arms  = |i| {
                    0 => (i % 255) + 1,
                    1 => 0x100 + i % 0xFFFF,
                    2 => 0x10000 + i % 0xFFFF_FFFF,
                    _ => 0x1_0000_0000 + i,
                },
                single_vals = [0u64, 1, 0xFF, 0x100, 0xFFFF, 0x10000, 0xFFFF_FFFF, u64::MAX],
            );
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
