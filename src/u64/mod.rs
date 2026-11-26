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

use crate::error::{DecodeError, EncodeError};

mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod shuffle;
#[cfg(target_arch = "x86_64")]
mod sse2;

// ── U64Coder1234 ──────────────────────────────────────────────────────────────

impl_dispatch! {
    fn dispatch_encode_1234(values: &[u64], out: &mut [u8]) -> usize;
    avx2::encode_to_slice_1234,
    sse2::encode_to_slice_1234,
    neon::encode_to_slice_1234,
    scalar::encode_to_slice_1234
}
impl_dispatch! {
    fn dispatch_decode_1234(data: &[u8], out: &mut [u64]) -> Result<usize, DecodeError>;
    avx2::decode_to_slice_1234,
    sse2::decode_to_slice_1234,
    neon::decode_to_slice_1234,
    scalar::decode_to_slice_1234
}

/// StreamVByte codec for `u64` values using 2-bit tags encoding 1, 2, 3, or 4 data bytes per value.
///
/// Same tag/width table as [`crate::u32::U32Classic`] but operates on `u64` slices.
/// Values greater than `u32::MAX` are silently truncated to their low 32 bits on
/// encode. This matches the behaviour of other StreamVByte libraries and is
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// assert_eq!(U64Coder1234.check_range(&[1u64, 255, u32::MAX as u64]), None);
    /// assert_eq!(U64Coder1234.check_range(&[1u64, u32::MAX as u64 + 1]), Some(1));
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// let values = vec![1u64, 256, 65536, u32::MAX as u64];
    /// assert_eq!(U64Coder1234.check_range(&values), None);
    /// let bytes = U64Coder1234.encode(&values);
    /// assert_eq!(U64Coder1234.decode(&bytes, values.len()).unwrap(), values);
    /// ```
    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    ///
    /// # Warning
    ///
    /// Values that exceed `u32::MAX` are silently truncated. Call
    /// [`check_range`](U64Coder1234::check_range) first if the input may contain
    /// values outside `0..=u32::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// let mut buf = Vec::new();
    /// U64Coder1234.encode_into(&[1u64, 2], &mut buf);
    /// U64Coder1234.encode_into(&[3u64, 4], &mut buf);
    /// ```
    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        let start = out.len();
        out.resize(start + self.max_encoded_len(values.len()), 0);
        let written = dispatch_encode_1234(values, &mut out[start..]);
        out.truncate(start + written);
    }

    /// Worst-case encoded size in bytes for `n` values: `ceil(n / 4)` control
    /// bytes plus 4 data bytes per value.
    ///
    /// Use this to size the buffer for [`encode_to_slice`](Self::encode_to_slice).
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// assert_eq!(U64Coder1234.max_encoded_len(0), 0);
    /// assert_eq!(U64Coder1234.max_encoded_len(5), 2 + 4 * 5);
    /// ```
    pub const fn max_encoded_len(&self, n: usize) -> usize {
        n.div_ceil(4).saturating_add(n.saturating_mul(4))
    }

    /// Encode `values` into the start of `out` and return the number of bytes
    /// written.
    ///
    /// `out` must be at least [`max_encoded_len(values.len())`](Self::max_encoded_len)
    /// bytes long, otherwise [`EncodeError::OutputTooSmall`] is returned and
    /// `out` is not modified. The check is against the worst case, so a
    /// shorter slice is rejected even if the actual encoded bytes would have
    /// fitted. Bytes in `out` after the returned length may be overwritten.
    ///
    /// # Warning
    ///
    /// Values that exceed `u32::MAX` are silently truncated. Call
    /// [`check_range`](U64Coder1234::check_range) first if the input may contain
    /// values outside `0..=u32::MAX`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// let values = [1u64, 256, 65536, u32::MAX as u64];
    /// let mut buf = [0u8; 64];
    /// let len = U64Coder1234.encode_to_slice(&values, &mut buf).unwrap();
    /// assert_eq!(&buf[..len], U64Coder1234.encode(&values));
    /// ```
    pub fn encode_to_slice(&self, values: &[u64], out: &mut [u8]) -> Result<usize, EncodeError> {
        let need = self.max_encoded_len(values.len());
        if out.len() < need {
            return Err(EncodeError::OutputTooSmall {
                need,
                have: out.len(),
            });
        }
        Ok(dispatch_encode_1234(values, out))
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// let bytes = U64Coder1234.encode(&[1u64, 256, 65536]);
    /// assert_eq!(U64Coder1234.decode(&bytes, 3).unwrap(), [1u64, 256, 65536]);
    /// ```
    #[allow(clippy::slow_vector_initialization)]
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        // Not `vec![0; n]`: that allocates with calloc, which measured ~30-40 ns
        // slower than malloc + fill for small n on glibc.
        crate::error::check_ctrl_len(data, n.div_ceil(4))?;
        let mut out = Vec::new();
        out.resize(n, 0);
        dispatch_decode_1234(data, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` values from `data`, appending them to `out`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`]. On error `out` is left
    /// at its original length.
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant silently produces corrupt output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// let bytes = U64Coder1234.encode(&[10u64, 20]);
    /// let mut out = vec![0u64];
    /// U64Coder1234.decode_into(&bytes, 2, &mut out).unwrap();
    /// assert_eq!(out, [0u64, 10, 20]);
    /// ```
    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), DecodeError> {
        crate::error::check_ctrl_len(data, n.div_ceil(4))?;
        let start = out.len();
        out.resize(start + n, 0);
        match dispatch_decode_1234(data, &mut out[start..]) {
            Ok(_) => Ok(()),
            Err(e) => {
                out.truncate(start);
                Err(e)
            }
        }
    }

    /// Decode exactly `out.len()` values from `data` into `out`.
    ///
    /// `out.len()` must equal the number of values that were originally
    /// encoded; a wrong length produces incorrect output or a [`DecodeError`].
    /// On error the contents of `out` are unspecified. As with
    /// [`decode`](Self::decode), the data must have been encoded by the same
    /// codec variant.
    ///
    /// Returns the number of bytes of `data` the encoded values took up
    /// (control stream plus data stream). Bytes after that are not read, so
    /// the values can be followed by other data. A caller that expects `data`
    /// to hold only these values can compare the count with `data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1234;
    /// let values = [1u64, 256, 65536, u32::MAX as u64];
    /// let bytes = U64Coder1234.encode(&values);
    /// let mut out = [0u64; 4];
    /// let used = U64Coder1234.decode_to_slice(&bytes, &mut out).unwrap();
    /// assert_eq!(used, bytes.len());
    /// assert_eq!(out, values);
    /// ```
    pub fn decode_to_slice(&self, data: &[u8], out: &mut [u64]) -> Result<usize, DecodeError> {
        dispatch_decode_1234(data, out)
    }
}

// ── U64Coder1248 ──────────────────────────────────────────────────────────────

impl_dispatch! {
    fn dispatch_encode_1248(values: &[u64], out: &mut [u8]) -> usize;
    avx2::encode_to_slice_1248,
    sse2::encode_to_slice_1248,
    neon::encode_to_slice_1248,
    scalar::encode_to_slice_1248
}
impl_dispatch! {
    fn dispatch_decode_1248(data: &[u8], out: &mut [u64]) -> Result<usize, DecodeError>;
    avx2::decode_to_slice_1248,
    sse2::decode_to_slice_1248,
    neon::decode_to_slice_1248,
    scalar::decode_to_slice_1248
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1248;
    /// let bytes = U64Coder1248.encode(&[1u64, 256, 1 << 32, u64::MAX]);
    /// assert_eq!(U64Coder1248.decode(&bytes, 4).unwrap(), [1u64, 256, 1 << 32, u64::MAX]);
    /// ```
    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1248;
    /// let mut buf = Vec::new();
    /// U64Coder1248.encode_into(&[1u64, 2], &mut buf);
    /// U64Coder1248.encode_into(&[u64::MAX], &mut buf);
    /// ```
    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        let start = out.len();
        out.resize(start + self.max_encoded_len(values.len()), 0);
        let written = dispatch_encode_1248(values, &mut out[start..]);
        out.truncate(start + written);
    }

    /// Worst-case encoded size in bytes for `n` values: `ceil(n / 4)` control
    /// bytes plus 8 data bytes per value.
    ///
    /// Use this to size the buffer for [`encode_to_slice`](Self::encode_to_slice).
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1248;
    /// assert_eq!(U64Coder1248.max_encoded_len(0), 0);
    /// assert_eq!(U64Coder1248.max_encoded_len(5), 2 + 8 * 5);
    /// ```
    pub const fn max_encoded_len(&self, n: usize) -> usize {
        n.div_ceil(4).saturating_add(n.saturating_mul(8))
    }

    /// Encode `values` into the start of `out` and return the number of bytes
    /// written.
    ///
    /// `out` must be at least [`max_encoded_len(values.len())`](Self::max_encoded_len)
    /// bytes long, otherwise [`EncodeError::OutputTooSmall`] is returned and
    /// `out` is not modified. The check is against the worst case, so a
    /// shorter slice is rejected even if the actual encoded bytes would have
    /// fitted. Bytes in `out` after the returned length may be overwritten.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1248;
    /// let values = [1u64, 256, 65536, u64::MAX];
    /// let mut buf = [0u8; 64];
    /// let len = U64Coder1248.encode_to_slice(&values, &mut buf).unwrap();
    /// assert_eq!(&buf[..len], U64Coder1248.encode(&values));
    /// ```
    pub fn encode_to_slice(&self, values: &[u64], out: &mut [u8]) -> Result<usize, EncodeError> {
        let need = self.max_encoded_len(values.len());
        if out.len() < need {
            return Err(EncodeError::OutputTooSmall {
                need,
                have: out.len(),
            });
        }
        Ok(dispatch_encode_1248(values, out))
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
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1248;
    /// let bytes = U64Coder1248.encode(&[1u64, 1 << 32, u64::MAX]);
    /// assert_eq!(U64Coder1248.decode(&bytes, 3).unwrap(), [1u64, 1 << 32, u64::MAX]);
    /// ```
    #[allow(clippy::slow_vector_initialization)]
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        // Not `vec![0; n]`: that allocates with calloc, which measured ~30-40 ns
        // slower than malloc + fill for small n on glibc.
        crate::error::check_ctrl_len(data, n.div_ceil(4))?;
        let mut out = Vec::new();
        out.resize(n, 0);
        dispatch_decode_1248(data, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` values from `data`, appending them to `out`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`]. On error `out` is left
    /// at its original length.
    ///
    /// # Warning
    ///
    /// The data must have been encoded by the same codec variant. Decoding bytes
    /// produced by a different variant silently produces corrupt output.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1248;
    /// let bytes = U64Coder1248.encode(&[10u64, 1 << 40]);
    /// let mut out = vec![0u64];
    /// U64Coder1248.decode_into(&bytes, 2, &mut out).unwrap();
    /// assert_eq!(out, [0u64, 10, 1 << 40]);
    /// ```
    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), DecodeError> {
        crate::error::check_ctrl_len(data, n.div_ceil(4))?;
        let start = out.len();
        out.resize(start + n, 0);
        match dispatch_decode_1248(data, &mut out[start..]) {
            Ok(_) => Ok(()),
            Err(e) => {
                out.truncate(start);
                Err(e)
            }
        }
    }

    /// Decode exactly `out.len()` values from `data` into `out`.
    ///
    /// `out.len()` must equal the number of values that were originally
    /// encoded; a wrong length produces incorrect output or a [`DecodeError`].
    /// On error the contents of `out` are unspecified. As with
    /// [`decode`](Self::decode), the data must have been encoded by the same
    /// codec variant.
    ///
    /// Returns the number of bytes of `data` the encoded values took up
    /// (control stream plus data stream). Bytes after that are not read, so
    /// the values can be followed by other data. A caller that expects `data`
    /// to hold only these values can compare the count with `data.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u64::U64Coder1248;
    /// let values = [1u64, 256, 65536, u64::MAX];
    /// let bytes = U64Coder1248.encode(&values);
    /// let mut out = [0u64; 4];
    /// let used = U64Coder1248.decode_to_slice(&bytes, &mut out).unwrap();
    /// assert_eq!(used, bytes.len());
    /// assert_eq!(out, values);
    /// ```
    pub fn decode_to_slice(&self, data: &[u8], out: &mut [u64]) -> Result<usize, DecodeError> {
        dispatch_decode_1248(data, out)
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
                max_len      = $max_len:expr,
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
                #[cfg(not(feature = "std"))]
                use alloc::{vec, vec::Vec};
                #[cfg(feature = "std")]
                use std::vec::Vec;
                fn encode(values: &[u64]) -> Vec<u8> {
                    crate::test_util::encode_checked(values, ($max_len)(values.len()), $scalar_enc)
                }
                fn scalar_decode(data: &[u8], n: usize) -> Vec<u64> {
                    let mut out = vec![0; n];
                    $scalar_dec(data, &mut out).unwrap();
                    out
                }
                fn ssse3_decode(data: &[u8], n: usize) -> Option<Vec<u64>> {
                    if !is_x86_feature_detected!("ssse3") { return None; }
                    let mut out = vec![0; n];
                    unsafe { $ssse3_dec(data, &mut out).unwrap() };
                    Some(out)
                }
                fn avx2_decode(data: &[u8], n: usize) -> Option<Vec<u64>> {
                    if !is_x86_feature_detected!("avx2") { return None; }
                    let mut out = vec![0; n];
                    unsafe { $avx2_dec(data, &mut out).unwrap() };
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

                // A truncation in the scalar tail must report the index in the
                // whole output, not the index within the tail. n % 4 != 0 puts
                // the truncated last value in the scalar remainder.
                #[test]
                fn tail_error_index_is_absolute() {
                    use crate::error::DecodeError;
                    let widest = *[$($sv),+].last().unwrap();
                    for n in [13usize, 22, 37] {
                        let values = vec![widest; n];
                        let mut enc = encode(&values);
                        enc.pop(); // last value loses its top byte
                        let check = |name: &str, r: Result<usize, DecodeError>| match r {
                            Err(DecodeError::DataTruncated { index }) => {
                                assert_eq!(index, n - 1, "{name} n={n}")
                            }
                            other => panic!("{name} n={n}: expected DataTruncated, got {other:?}"),
                        };
                        let mut out = vec![0; n];
                        check("scalar", $scalar_dec(&enc, &mut out));
                        if is_x86_feature_detected!("ssse3") {
                            check("ssse3", unsafe { $ssse3_dec(&enc, &mut out) });
                        }
                        if is_x86_feature_detected!("avx2") {
                            check("avx2", unsafe { $avx2_dec(&enc, &mut out) });
                        }
                    }
                }
            };
        }

        mod coder1234 {
            use super::super::super::{avx2, scalar, sse2};
            impl_decode_tests!(
                max_len      = |n: usize| crate::u64::U64Coder1234.max_encoded_len(n),
                scalar_enc   = scalar::encode_to_slice_1234,
                ssse3_dec    = sse2::decode_to_slice_1234,
                avx2_dec     = avx2::decode_to_slice_1234,
                scalar_dec   = scalar::decode_to_slice_1234,
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
                max_len      = |n: usize| crate::u64::U64Coder1248.max_encoded_len(n),
                scalar_enc   = scalar::encode_to_slice_1248,
                ssse3_dec    = sse2::decode_to_slice_1248,
                avx2_dec     = avx2::decode_to_slice_1248,
                scalar_dec   = scalar::decode_to_slice_1248,
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
                max_len      = $max_len:expr,
                scalar_enc  = $scalar_enc:path,
                ssse3_enc   = $ssse3_enc:path,
                avx2_enc    = $avx2_enc:path,
                val_arms    = |$iv:ident| { $($va:tt)+ },
                homog_bases = [$($hb:expr),+],
                boundary    = [$($bv:expr),+],
                large_arms  = |$il:ident| { $($la:tt)+ },
                single_vals = [$($sv:expr),+] $(,)?
            ) => {
                #[cfg(not(feature = "std"))]
                use alloc::{vec, vec::Vec};
                #[cfg(feature = "std")]
                use std::vec::Vec;
                fn scalar_enc_fn(v: &[u64]) -> Vec<u8> {
                    crate::test_util::encode_checked(v, ($max_len)(v.len()), $scalar_enc)
                }
                fn ssse3_enc_fn(v: &[u64]) -> Option<Vec<u8>> {
                    if !is_x86_feature_detected!("ssse3") { return None; }
                    Some(crate::test_util::encode_checked(v, ($max_len)(v.len()), |v, o| unsafe { $ssse3_enc(v, o) }))
                }
                fn avx2_enc_fn(v: &[u64]) -> Option<Vec<u8>> {
                    if !is_x86_feature_detected!("avx2") { return None; }
                    Some(crate::test_util::encode_checked(v, ($max_len)(v.len()), |v, o| unsafe { $avx2_enc(v, o) }))
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
                max_len      = |n: usize| crate::u64::U64Coder1234.max_encoded_len(n),
                scalar_enc  = scalar::encode_to_slice_1234,
                ssse3_enc   = sse2::encode_to_slice_1234,
                avx2_enc    = avx2::encode_to_slice_1234,
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
                max_len      = |n: usize| crate::u64::U64Coder1248.max_encoded_len(n),
                scalar_enc  = scalar::encode_to_slice_1248,
                ssse3_enc   = sse2::encode_to_slice_1248,
                avx2_enc    = avx2::encode_to_slice_1248,
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
                max_len      = $max_len:expr,
                scalar_enc  = $scalar_enc:path,
                neon_enc    = $neon_enc:path,
                scalar_dec  = $scalar_dec:path,
                neon_dec    = $neon_dec:path,
                val_arms    = |$iv:ident| { $($va:tt)+ },
                homog_bases = [$($hb:expr),+],
                large_arms  = |$il:ident| { $($la:tt)+ },
                single_vals = [$($sv:expr),+] $(,)?
            ) => {
                #[cfg(not(feature = "std"))]
                use alloc::{vec, vec::Vec};
                #[cfg(feature = "std")]
                use std::vec::Vec;
                fn scalar_enc_fn(v: &[u64]) -> Vec<u8> {
                    crate::test_util::encode_checked(v, ($max_len)(v.len()), $scalar_enc)
                }
                fn neon_enc_fn(v: &[u64]) -> Vec<u8> {
                    crate::test_util::encode_checked(v, ($max_len)(v.len()), |v, o| unsafe { $neon_enc(v, o) })
                }
                fn scalar_dec_fn(d: &[u8], n: usize) -> Vec<u64> {
                    let mut out = vec![0; n];
                    $scalar_dec(d, &mut out).unwrap();
                    out
                }
                fn neon_dec_fn(d: &[u8], n: usize) -> Vec<u64> {
                    let mut out = vec![0; n];
                    unsafe { $neon_dec(d, &mut out).unwrap() };
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

                // A truncation in the scalar tail must report the index in the
                // whole output, not the index within the tail.
                #[test]
                fn tail_error_index_is_absolute() {
                    use crate::error::DecodeError;
                    let widest = *[$($sv),+].last().unwrap();
                    for n in [13usize, 22, 37] {
                        let values = vec![widest; n];
                        let mut enc = scalar_enc_fn(&values);
                        enc.pop(); // last value loses its top byte
                        let mut out = vec![0; n];
                        for (name, r) in [
                            ("scalar", $scalar_dec(&enc, &mut out)),
                            ("neon", unsafe { $neon_dec(&enc, &mut out) }),
                        ] {
                            match r {
                                Err(DecodeError::DataTruncated { index }) => {
                                    assert_eq!(index, n - 1, "{name} n={n}")
                                }
                                other => panic!("{name} n={n}: expected DataTruncated, got {other:?}"),
                            }
                        }
                    }
                }
            };
        }

        mod coder1234 {
            use super::super::super::{neon, scalar};
            impl_neon_tests!(
                max_len      = |n: usize| crate::u64::U64Coder1234.max_encoded_len(n),
                scalar_enc  = scalar::encode_to_slice_1234,
                neon_enc    = neon::encode_to_slice_1234,
                scalar_dec  = scalar::decode_to_slice_1234,
                neon_dec    = neon::decode_to_slice_1234,
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
                max_len      = |n: usize| crate::u64::U64Coder1248.max_encoded_len(n),
                scalar_enc  = scalar::encode_to_slice_1248,
                neon_enc    = neon::encode_to_slice_1248,
                scalar_dec  = scalar::decode_to_slice_1248,
                neon_dec    = neon::decode_to_slice_1248,
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
