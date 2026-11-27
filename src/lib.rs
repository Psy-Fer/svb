//! Pure-Rust [StreamVByte](https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/) covering u16, u32, and u64 integer codecs with optional SIMD acceleration.
//!
//! # Codec variants
//!
//! | Type | Struct | Tag | Byte widths | Notes |
//! |------|--------|-----|-------------|-------|
//! | u16 | [`u16::Svb16`] | 1-bit | 1/2 | ONT VBZ format |
//! | u32 | [`u32::U32Classic`] | 2-bit | 1/2/3/4 | Lemire reference-compatible |
//! | u32 | [`u32::U32Variant0124`] | 2-bit | 0/1/2/4 | Sparse-data variant |
//! | u64 | [`u64::U64Coder1234`] | 2-bit | 1/2/3/4 | Values must fit in u32 |
//! | u64 | [`u64::U64Coder1248`] | 2-bit | 1/2/4/8 | Full u64 range |
//!
//! Delta and zigzag transforms are composable layers in [`delta`] and [`zigzag`].
//!
//! # Feature flags
//!
//! Enable `simd-auto` for runtime CPU detection (recommended). Use `simd-avx2`,
//! `simd-ssse3`, or `simd-neon` for compile-time SIMD when the target is known.
//! Disable `std` for `no_std` use. Without `alloc`, the slice functions
//! (`encode_to_slice`, `decode_to_slice`, the VBZ, VBZ-K and SVB-ZD encoders and
//! fused decoders) are available and never allocate. Enable `alloc` for the
//! rest: the `Vec` functions, ex-zd and the 3-pass decoders.
//!
//! **`no_std` note:** `simd-auto` on x86-64 requires `std` for
//! [`is_x86_feature_detected!`]. When `std` is disabled, `simd-auto` compiles
//! but silently falls back to scalar regardless of the CPU's actual capabilities.
//! Use `simd-avx2` or `simd-ssse3` with a compile-time target-feature flag
//! (`RUSTFLAGS="-C target-feature=+avx2"`) for SIMD in `no_std` builds.
#![cfg_attr(not(feature = "std"), no_std)]
#![deny(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

pub mod error;
pub use error::{DecodeError, EncodeError};

pub mod delta;
#[cfg(feature = "alloc")]
pub mod patched;
#[cfg(feature = "alloc")]
pub mod quantize;
pub mod zigzag;

// ── SIMD dispatch macro ───────────────────────────────────────────────────────
//
// Generates the 5-way cfg dispatch used by every codec variant. Each arm is
// individually gated so that exactly one branch is active at a time,
// preventing unreachable_code warnings when multiple simd-* features overlap.
// The scalar fallback is gated off in builds where a SIMD arm always returns
// (forced AVX2 or SSSE3 on x86_64, forced NEON or simd-auto on AArch64).
//
// Usage:
//   impl_dispatch! { fn name(a: A, b: B) -> R; avx2_fn, sse2_fn, neon_fn, scalar_fn }
//
// Call it with braces: rustfmt reformats parenthesised calls and breaks the `;`.
// Adding a new target architecture requires editing only the macro body here.

macro_rules! impl_dispatch {
    (fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;
     $avx2_fn:path, $sse2_fn:path, $neon_fn:path, $scalar_fn:path) => {
        fn $name($($arg: $ty),*) -> $ret {
            // In these builds a SIMD arm below always returns, so the scalar
            // call at the end is compiled out (it would be unreachable code).
            // This reference keeps the scalar function from being reported as
            // dead code there.
            #[cfg(any(
                all(target_arch = "x86_64", any(feature = "simd-avx2", feature = "simd-ssse3")),
                all(target_arch = "aarch64", any(feature = "simd-neon", feature = "simd-auto"))
            ))]
            let _ = $scalar_fn;
            #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
            {
                // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
                return unsafe { $avx2_fn($($arg),*) };
            }
            #[cfg(all(
                feature = "simd-ssse3",
                not(feature = "simd-avx2"),
                target_arch = "x86_64"
            ))]
            {
                // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
                return unsafe { $sse2_fn($($arg),*) };
            }
            #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
            {
                // SAFETY: NEON is mandatory on AArch64.
                return unsafe { $neon_fn($($arg),*) };
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
                        return unsafe { $avx2_fn($($arg),*) };
                    }
                    if is_x86_feature_detected!("ssse3") {
                        // SAFETY: SSSE3 confirmed at runtime.
                        return unsafe { $sse2_fn($($arg),*) };
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { $neon_fn($($arg),*) };
                }
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", any(feature = "simd-avx2", feature = "simd-ssse3")),
                all(target_arch = "aarch64", any(feature = "simd-neon", feature = "simd-auto"))
            )))]
            {
                return $scalar_fn($($arg),*);
            }
        }
    };
}

#[cfg(all(test, feature = "alloc"))]
pub(crate) mod test_util {
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    /// Run an encode kernel on a slice of exactly `max` bytes, followed by
    /// canary bytes, and check the kernel did not write past the end of its
    /// slice. Returns the encoded bytes.
    pub(crate) fn encode_checked<T>(
        values: &[T],
        max: usize,
        f: impl FnOnce(&[T], &mut [u8]) -> usize,
    ) -> Vec<u8> {
        const CANARY: u8 = 0xA5;
        let mut buf = vec![CANARY; max + 64];
        let written = f(values, &mut buf[..max]);
        assert!(written <= max);
        assert!(
            buf[max..].iter().all(|&b| b == CANARY),
            "encode wrote past max_encoded_len, n={}",
            values.len()
        );
        buf.truncate(written);
        buf
    }
}

pub mod u16;
pub mod u32;
pub mod u64;

// ── VBZ convenience pipeline ──────────────────────────────────────────────────
//
// Implements the three-stage inner codec used by Oxford Nanopore's POD5 format:
//   encode: i16 samples → delta → zigzag → SVB16 → Vec<u8>
//   decode: Vec<u8> → SVB16 → zigzag → delta → i16 samples
//
// The outer zstd layer is handled by the caller (e.g. pod5-rs).

#[cfg(feature = "alloc")]
pub use vbz::{decode_vbz, decode_vbz_into, decode_vbz_to_slice, encode_vbz, encode_vbz_into};
pub use vbz::{encode_vbz_to_slice, vbz_max_encoded_len};

mod vbz_fused;

mod svbzd_fused;

#[cfg(feature = "alloc")]
mod exzd_fused;

mod vbz {
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    #[cfg(feature = "alloc")]
    use crate::error::DecodeError;
    use crate::error::EncodeError;
    #[cfg(feature = "alloc")]
    use crate::{delta, zigzag};
    use crate::{u16::Svb16, vbz_fused};

    /// Encode `i16` samples through delta, zigzag, then SVB16, returning raw bytes ready to pass to zstd.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::{encode_vbz, decode_vbz};
    /// let samples = [10i16, 11, 12, 13];
    /// let encoded = encode_vbz(&samples);
    /// let decoded = decode_vbz(&encoded, samples.len()).unwrap();
    /// assert_eq!(decoded, samples);
    /// ```
    #[cfg(feature = "alloc")]
    pub fn encode_vbz(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_vbz_into(samples, &mut out);
        out
    }

    /// Encode `i16` samples through delta, zigzag, then SVB16, appending the result to `out`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::{encode_vbz_into, decode_vbz};
    /// let mut buf = Vec::new();
    /// encode_vbz_into(&[1i16, 2, 3], &mut buf);
    /// encode_vbz_into(&[4i16, 5, 6], &mut buf);
    /// ```
    #[cfg(feature = "alloc")]
    pub fn encode_vbz_into(samples: &[i16], out: &mut Vec<u8>) {
        let start = out.len();
        out.resize(start + vbz_max_encoded_len(samples.len()), 0);
        let written = vbz_fused::encode_to_slice(samples, &mut out[start..]);
        out.truncate(start + written);
    }

    /// Worst-case VBZ encoded size in bytes for `n` samples: `ceil(n / 8)`
    /// control bytes plus 2 data bytes per sample.
    ///
    /// Use this to size the buffer for [`encode_vbz_to_slice`].
    ///
    /// ```
    /// assert_eq!(svb::vbz_max_encoded_len(0), 0);
    /// assert_eq!(svb::vbz_max_encoded_len(9), 2 + 2 * 9);
    /// ```
    pub const fn vbz_max_encoded_len(n: usize) -> usize {
        Svb16.max_encoded_len(n)
    }

    /// Encode `i16` samples to VBZ into the start of `out` and return the
    /// number of bytes written.
    ///
    /// `out` must be at least [`vbz_max_encoded_len(samples.len())`](vbz_max_encoded_len)
    /// bytes long, otherwise [`EncodeError::OutputTooSmall`] is returned and
    /// `out` is not modified. The check is against the worst case, so a
    /// shorter slice is rejected even if the actual encoded bytes would have
    /// fitted. Bytes in `out` after the returned length may be overwritten.
    /// Nothing is allocated.
    ///
    /// ```
    /// let samples = [10i16, 11, 12, 13, 300];
    /// let mut buf = vec![0u8; svb::vbz_max_encoded_len(samples.len())];
    /// let len = svb::encode_vbz_to_slice(&samples, &mut buf).unwrap();
    /// assert_eq!(&buf[..len], svb::encode_vbz(&samples));
    /// ```
    pub fn encode_vbz_to_slice(samples: &[i16], out: &mut [u8]) -> Result<usize, EncodeError> {
        let need = vbz_max_encoded_len(samples.len());
        if out.len() < need {
            return Err(EncodeError::OutputTooSmall {
                need,
                have: out.len(),
            });
        }
        Ok(vbz_fused::encode_to_slice(samples, out))
    }

    /// Decode exactly `n` `i16` samples from SVB16 bytes (after zstd decompression).
    ///
    /// `data` must be exactly one frame: extra bytes after it give
    /// [`DecodeError::TrailingData`].
    ///
    /// `n` must equal the number of samples that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::{encode_vbz, decode_vbz};
    /// let samples = [10i16, 11, 12, 13];
    /// let encoded = encode_vbz(&samples);
    /// assert_eq!(decode_vbz(&encoded, samples.len()).unwrap(), samples);
    /// ```
    #[cfg(feature = "alloc")]
    pub fn decode_vbz(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
        let mut out = Vec::new();
        decode_vbz_into(data, n, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` `i16` samples from SVB16 bytes, appending them to `out`.
    ///
    /// `n` must equal the number of samples that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::{encode_vbz, decode_vbz_into};
    /// let encoded = encode_vbz(&[10i16, 20]);
    /// let mut out = vec![0i16];
    /// decode_vbz_into(&encoded, 2, &mut out).unwrap();
    /// assert_eq!(out, [0i16, 10, 20]);
    /// ```
    ///
    /// On error `out` is left at its original length.
    #[cfg(feature = "alloc")]
    pub fn decode_vbz_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
        crate::error::check_ctrl_len(data, n.div_ceil(8))?;
        let start = out.len();
        out.resize(start + n, 0);
        let result = decode_vbz_to_slice(data, &mut out[start..]);
        if result.is_err() {
            out.truncate(start);
        }
        result
    }

    /// Decode VBZ bytes into `out` with the 3-pass decoder (SVB16, zigzag,
    /// delta).
    ///
    /// The number of samples decoded is `out.len()`, which must equal the
    /// number that was encoded. On error the contents of `out` are
    /// unspecified. The SVB16 pass decodes into a temporary buffer; zigzag and
    /// delta then work in `out`. [`decode_vbz_fused_to_slice`](crate::decode_vbz_fused_to_slice)
    /// is faster and does not allocate.
    ///
    /// ```
    /// let samples = [10i16, 11, 12, 13, 300];
    /// let encoded = svb::encode_vbz(&samples);
    /// let mut out = [0i16; 5];
    /// svb::decode_vbz_to_slice(&encoded, &mut out).unwrap();
    /// assert_eq!(out, samples);
    /// ```
    // Not `vec![0; n]` for the temporary codes: that allocates with calloc,
    // which measured slower than malloc + fill for small n on glibc.
    #[allow(clippy::slow_vector_initialization)]
    #[cfg(feature = "alloc")]
    pub fn decode_vbz_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
        crate::error::check_ctrl_len(data, out.len().div_ceil(8))?;
        let mut codes = Vec::new();
        codes.resize(out.len(), 0u16);
        let used = Svb16.decode_to_slice(data, &mut codes)?;
        crate::error::check_used(used, data.len())?;
        zigzag::decode_to_slice(&codes, out);
        delta::decode_in_place(out);
        Ok(())
    }
}

/// Decode a VBZ-encoded byte stream into `i16` samples using a fused single-pass decoder.
///
/// `data` must be exactly one frame: extra bytes after it give
/// [`DecodeError::TrailingData`].
///
/// Identical output to [`decode_vbz`] but fuses SVB16, zigzag, and delta decode
/// into one SIMD loop. SVB16 and zigzag work fills the delta carry-chain stall,
/// so throughput approaches the delta-alone rate rather than the harmonic sum of
/// all three stages.
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
    let mut out = Vec::new();
    decode_vbz_fused_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode a VBZ-encoded byte stream, appending to `out`. See [`decode_vbz_fused`].
///
/// On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    vbz_fused::decode_into(data, n, out)
}

/// Decode a VBZ-encoded byte stream into `out` with the fused decoder.
///
/// The number of samples decoded is `out.len()`, which must equal the number
/// that was encoded. On error the contents of `out` are unspecified.
///
/// ```
/// let samples = [500i16, 510, 505, 520, 300, 301];
/// let encoded = svb::encode_vbz(&samples);
/// let mut out = [0i16; 6];
/// svb::decode_vbz_fused_to_slice(&encoded, &mut out).unwrap();
/// assert_eq!(out, samples);
/// ```
pub fn decode_vbz_fused_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    vbz_fused::decode_from_to_slice(data, 0, out)
}

/// Decode a VBZ stream whose running sum starts from `initial_carry`.
///
/// This decodes part of a longer VBZ stream: `initial_carry` is the last
/// sample before the part starts. VBZ-K uses it to decode its sub-streams
/// independently. A stream can be split after any multiple of 8 samples: each
/// control byte covers 8 samples, and their data takes 8 bytes plus one for
/// each set bit.
///
/// ```
/// let samples: Vec<i16> = (0..64).map(|i| i * 37 - 1000).collect();
/// let n = samples.len();
/// let encoded = svb::encode_vbz(&samples);
///
/// let n_half = 32;
/// let (ctrl, data) = encoded.split_at(n.div_ceil(8));
/// let (ctrl_a, ctrl_b) = ctrl.split_at(n_half / 8);
/// let data_a_len: usize = ctrl_a.iter().map(|c| 8 + c.count_ones() as usize).sum();
/// let (data_a, data_b) = data.split_at(data_a_len);
/// let stream_a = [ctrl_a, data_a].concat();
/// let stream_b = [ctrl_b, data_b].concat();
///
/// // The second half starts from the last sample of the first. If that value
/// // is stored with the data, as VBZ-K does, the halves can be decoded on
/// // separate threads.
/// let carry = samples[n_half - 1];
/// let a = svb::decode_vbz_fused_from(&stream_a, n_half, 0).unwrap();
/// let b = svb::decode_vbz_fused_from(&stream_b, n - n_half, carry).unwrap();
/// assert_eq!([a, b].concat(), samples);
/// ```
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_from(
    data: &[u8],
    n: usize,
    initial_carry: i16,
) -> Result<Vec<i16>, DecodeError> {
    let mut out = Vec::new();
    decode_vbz_fused_from_into(data, n, initial_carry, &mut out)?;
    Ok(out)
}

/// Decode a VBZ stream starting from `initial_carry`, appending to `out`.
///
/// See [`decode_vbz_fused_from`] for how to split a stream.
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_from_into(
    data: &[u8],
    n: usize,
    initial_carry: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    vbz_fused::decode_from_into(data, n, initial_carry, out)
}

/// Decode a VBZ stream starting from `initial_carry` into `out`, which
/// holds `out.len()` samples. On error the contents of `out` are unspecified.
///
/// See [`decode_vbz_fused_from`] for how to split a stream. Writing each
/// half into its own part of one output slice (for example with
/// `split_at_mut`) avoids concatenating the halves afterwards.
pub fn decode_vbz_fused_from_to_slice(
    data: &[u8],
    initial_carry: i16,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    vbz_fused::decode_from_to_slice(data, initial_carry, out)
}

#[cfg(all(test, feature = "alloc"))]
mod vbz_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode_vbz(&encode_vbz(&[]), 0).unwrap(), &[] as &[i16]);
    }

    #[test]
    fn roundtrip_single() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX] {
            assert_eq!(decode_vbz(&encode_vbz(&[v]), 1).unwrap(), [v]);
        }
    }

    #[test]
    fn roundtrip_flat_signal() {
        // Constant signal: deltas are all zero after the first → compresses well.
        let samples = vec![1000i16; 256];
        assert_eq!(decode_vbz(&encode_vbz(&samples), 256).unwrap(), samples);
    }

    #[test]
    fn roundtrip_ramp() {
        let samples: Vec<i16> = (0..128).collect();
        assert_eq!(decode_vbz(&encode_vbz(&samples), 128).unwrap(), samples);
    }

    #[test]
    fn roundtrip_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX];
        assert_eq!(decode_vbz(&encode_vbz(&samples), 4).unwrap(), samples);
    }

    #[test]
    fn encode_vbz_into_appends() {
        let mut out = encode_vbz(&[1i16, 2, 3]);
        let first_len = out.len();
        encode_vbz_into(&[4i16, 5, 6], &mut out);
        // Two independent blobs concatenated; decode each with its own n.
        let first = decode_vbz(&out[..first_len], 3).unwrap();
        let second = decode_vbz(&out[first_len..], 3).unwrap();
        assert_eq!(first, [1, 2, 3]);
        assert_eq!(second, [4, 5, 6]);
    }

    #[test]
    fn decode_vbz_into_appends() {
        let enc = encode_vbz(&[10i16, 20, 30]);
        let mut out = vec![99i16];
        decode_vbz_into(&enc, 3, &mut out).unwrap();
        assert_eq!(out, [99, 10, 20, 30]);
    }

    fn slice_case(n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| match i % 11 {
                0 => i16::MIN,
                5 => i16::MAX,
                _ => (i as i16).wrapping_mul(613) % 200,
            })
            .collect()
    }

    const SLICE_SIZES: [usize; 10] = [0, 1, 7, 8, 9, 15, 16, 17, 63, 1000];

    #[test]
    fn encode_to_slice_matches_vec_encode() {
        for n in SLICE_SIZES {
            let samples = slice_case(n);
            let enc = crate::test_util::encode_checked(&samples, vbz_max_encoded_len(n), |s, o| {
                encode_vbz_to_slice(s, o).unwrap()
            });
            assert_eq!(enc, encode_vbz(&samples), "n={n}");
        }
    }

    #[test]
    fn encode_to_slice_too_small() {
        let samples = [1i16, 2, 3, 4, 5, 6, 7, 8, 9];
        let need = vbz_max_encoded_len(9);
        assert_eq!(need, 20);
        let mut buf = vec![0xA5u8; need - 1];
        let err = encode_vbz_to_slice(&samples, &mut buf).unwrap_err();
        assert!(
            matches!(err, EncodeError::OutputTooSmall { need: 20, have: 19 }),
            "{err:?}"
        );
        assert!(buf.iter().all(|&b| b == 0xA5), "out was modified");
    }

    #[test]
    fn decode_to_slice_roundtrip() {
        for n in SLICE_SIZES {
            let samples = slice_case(n);
            let enc = encode_vbz(&samples);
            let mut out = vec![0x5A5Ai16; n];
            decode_vbz_to_slice(&enc, &mut out).unwrap();
            assert_eq!(out, samples, "n={n}");
        }
    }

    #[test]
    fn decode_truncated_is_an_error() {
        let samples = slice_case(65);
        let enc = encode_vbz(&samples);
        for len in 0..enc.len() {
            let mut out = vec![0i16; 65];
            assert!(
                decode_vbz_to_slice(&enc[..len], &mut out).is_err(),
                "len={len}"
            );
            let mut v = vec![7i16, 8];
            assert!(
                decode_vbz_into(&enc[..len], 65, &mut v).is_err(),
                "len={len}"
            );
            assert_eq!(v, [7, 8], "len={len}");
        }
    }
}

// ── VBZ-K format ──────────────────────────────────────────────────────────────
//
// K independent sub-streams, each decodable on its own thread.
// Header: [k: u8][(carry_i: i16, data_offset_i: u32) for i in 1..k][VBZ payload]
// n_sub = (n / k) & !7; last sub-chunk = n - (k-1)*n_sub

/// Encode samples to VBZ-K format — `k` independent sub-streams decodable in parallel.
///
/// **Experimental.** This format is not yet stabilised and may change in a future release.
/// It is provided for exploration and testing; do not use it for long-lived stored data
/// without pinning the crate version.
///
/// Header: `[k: u8][(carry_i: i16 LE, data_offset_i: u32 LE) for i in 1..k][VBZ payload]`
///
/// The header stores `k` in one byte, so a `k` above 255 is reduced to 255.
/// When `n_sub = (n / k) & !7` is 0 (fewer than `k*8` samples), the encoder
/// falls back to `k=1` (1-byte header + standard VBZ body).
///
/// # Panics
///
/// Panics if `k == 0`.
#[cfg(feature = "alloc")]
#[allow(clippy::slow_vector_initialization)]
pub fn encode_vbzk(samples: &[i16], k: usize) -> Vec<u8> {
    // Not `vec![0; len]`: that allocates with calloc, which measured 16%
    // slower at n=128.
    let mut out = Vec::new();
    out.resize(vbzk_max_encoded_len(samples.len(), k), 0);
    let len =
        encode_vbzk_to_slice(samples, k, &mut out).expect("buffer sized by vbzk_max_encoded_len");
    out.truncate(len);
    out
}

/// Worst-case VBZ-K encoded size in bytes for `n` samples split into `k`
/// sub-streams: the header for the `k` the encoder will actually use (see
/// [`encode_vbzk`]) plus [`vbz_max_encoded_len(n)`](vbz_max_encoded_len).
///
/// **Experimental**, like the rest of VBZ-K.
///
/// ```
/// // 64 samples split 4 ways: 1 + 3 * 6 header bytes.
/// assert_eq!(svb::vbzk_max_encoded_len(64, 4), 19 + svb::vbz_max_encoded_len(64));
/// // Too few samples for 4 sub-streams of 8, so k falls back to 1.
/// assert_eq!(svb::vbzk_max_encoded_len(16, 4), 1 + svb::vbz_max_encoded_len(16));
/// ```
///
/// # Panics
///
/// Panics if `k == 0`.
pub const fn vbzk_max_encoded_len(n: usize, k: usize) -> usize {
    vbzk_header_len(vbzk_effective_k(n, k)).saturating_add(vbz_max_encoded_len(n))
}

/// Encode samples to VBZ-K format into the start of `out` and return the
/// number of bytes written.
///
/// **Experimental**, like the rest of VBZ-K. `out` must be at least
/// [`vbzk_max_encoded_len(samples.len(), k)`](vbzk_max_encoded_len) bytes
/// long, otherwise [`EncodeError::OutputTooSmall`] is returned and `out` is
/// not modified. Bytes in `out` after the returned length may be overwritten.
///
/// ```
/// let samples: Vec<i16> = (0..100).map(|i| i * 3 - 150).collect();
/// let mut buf = vec![0u8; svb::vbzk_max_encoded_len(samples.len(), 4)];
/// let len = svb::encode_vbzk_to_slice(&samples, 4, &mut buf).unwrap();
/// assert_eq!(&buf[..len], svb::encode_vbzk(&samples, 4));
/// ```
///
/// # Panics
///
/// Panics if `k == 0`.
pub fn encode_vbzk_to_slice(
    samples: &[i16],
    k: usize,
    out: &mut [u8],
) -> Result<usize, EncodeError> {
    let n = samples.len();
    let need = vbzk_max_encoded_len(n, k);
    if out.len() < need {
        return Err(EncodeError::OutputTooSmall {
            need,
            have: out.len(),
        });
    }
    let effective_k = vbzk_effective_k(n, k);
    let n_sub = (n / effective_k) & !7;
    let (header, body) = out.split_at_mut(vbzk_header_len(effective_k));
    let body_len =
        encode_vbz_to_slice(samples, body).expect("buffer sized by vbzk_max_encoded_len");

    header[0] = effective_k as u8;
    // Entry i - 1 holds the carry into sub-stream i and the offset of its
    // first data byte, found by summing the data lengths of the ctrl bytes
    // before it. Each ctrl byte covers 8 samples and n_sub is a multiple of 8.
    let ctrl = &body[..n.div_ceil(8)];
    let mut data_offset: u32 = 0;
    let mut ctrl_idx = 0usize;
    for (i, entry) in header[1..].chunks_exact_mut(6).enumerate() {
        let split_pos = n_sub * (i + 1);
        while ctrl_idx < split_pos / 8 {
            data_offset += 8 + ctrl[ctrl_idx].count_ones();
            ctrl_idx += 1;
        }
        entry[..2].copy_from_slice(&samples[split_pos - 1].to_le_bytes());
        entry[2..].copy_from_slice(&data_offset.to_le_bytes());
    }
    Ok(header.len() + body_len)
}

// The number of sub-streams the encoder uses: `k` capped at 255 (the header
// stores it in one byte), or 1 when sub-streams of `(n / k) & !7` samples
// would be empty.
const fn vbzk_effective_k(n: usize, k: usize) -> usize {
    assert!(k != 0, "k must be >= 1");
    let k = if k > u8::MAX as usize {
        u8::MAX as usize
    } else {
        k
    };
    if k > 1 && (n / k) & !7 != 0 { k } else { 1 }
}

const fn vbzk_header_len(k: usize) -> usize {
    1 + (k - 1) * 6
}

/// Decode VBZ-K encoded data, returning a new `Vec<i16>`.
///
/// `data` must be exactly one frame: extra bytes after it give
/// [`DecodeError::TrailingData`].
#[cfg(feature = "alloc")]
pub fn decode_vbzk(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    decode_vbzk_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode VBZ-K encoded data sequentially (one sub-stream at a time), appending to `out`.
///
/// On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_vbzk_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    crate::error::check_ctrl_len(data, n.div_ceil(8))?;
    let start = out.len();
    out.resize(start + n, 0);
    let result = decode_vbzk_to_slice(data, &mut out[start..]);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Decode VBZ-K encoded data sequentially into `out`, which holds `out.len()`
/// samples. On error the contents of `out` are unspecified.
///
/// ```
/// let samples: Vec<i16> = (0..100).map(|i| i * 3 - 150).collect();
/// let encoded = svb::encode_vbzk(&samples, 4);
/// let mut out = vec![0i16; samples.len()];
/// svb::decode_vbzk_to_slice(&encoded, &mut out).unwrap();
/// assert_eq!(out, samples);
/// ```
pub fn decode_vbzk_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    let frame = parse_vbzk(data, out.len())?;
    // Check every sub-stream's offsets before decoding any of them, so a frame
    // with several faults reports the same error as the parallel decoder.
    for i in 0..frame.k {
        frame.stream(i)?;
    }
    for i in 0..frame.k {
        let s = frame.stream(i)?;
        decode_vbzk_stream(&s, &mut out[s.start..s.start + s.len])?;
    }
    Ok(())
}

// Decode one VBZ-K sub-stream into `out`. The sub-stream must use all of its
// data bytes, which end where the next sub-stream's start (or at the end of
// `data` for the last one).
fn decode_vbzk_stream(s: &VbzkStream<'_>, out: &mut [i16]) -> Result<(), DecodeError> {
    let used = vbz_fused::decode_parts_to_slice(s.ctrl, s.data, s.initial, out)
        .map_err(|e| offset_index(e, s.start))?;
    crate::error::check_used(s.data_offset + used, s.data_offset + s.data.len())
}

// One VBZ-K sub-stream: its ctrl and data bytes, where its samples go in the
// output, and the carry it starts from.
struct VbzkStream<'a> {
    ctrl: &'a [u8],
    data: &'a [u8],
    // Position of `data` in the whole VBZ-K frame.
    data_offset: usize,
    start: usize,
    len: usize,
    initial: i16,
}

// A VBZ-K frame with its header checked: the shared ctrl stream, the data
// bytes of all sub-streams, and what is needed to find each sub-stream.
struct VbzkFrame<'a> {
    data: &'a [u8],
    k: usize,
    n: usize,
    n_sub: usize,
    header_len: usize,
    ctrl: &'a [u8],
    data_bytes: &'a [u8],
}

// Parse a VBZ-K header for `n` samples. Checks that the header and the ctrl
// stream fit in `data`; the sub-stream offsets are checked by `stream`.
fn parse_vbzk(data: &[u8], n: usize) -> Result<VbzkFrame<'_>, DecodeError> {
    let Some(&k) = data.first() else {
        return Err(DecodeError::ControlStreamTooShort { need: 1, have: 0 });
    };
    let k = k as usize;
    if k == 0 {
        return Err(DecodeError::ControlStreamTooShort { need: 1, have: 0 });
    }
    let header_len = 1 + (k - 1) * 6;
    if data.len() < header_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: header_len,
            have: data.len(),
        });
    }

    let svb = &data[header_len..];
    let ctrl_len = n.div_ceil(8);
    if svb.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: svb.len(),
        });
    }
    Ok(VbzkFrame {
        data,
        k,
        n,
        n_sub: (n / k) & !7,
        header_len,
        ctrl: &svb[..ctrl_len],
        data_bytes: &svb[ctrl_len..],
    })
}

impl<'a> VbzkFrame<'a> {
    // Sub-stream i (for i >= 1) starts at header entry i - 1:
    // [carry: i16 LE][data offset: u32 LE]. Sub-stream 0 starts at 0 with
    // carry 0.
    fn entry(&self, i: usize) -> (i16, usize) {
        if i == 0 {
            return (0, 0);
        }
        let d = self.data;
        let off = 1 + (i - 1) * 6;
        let carry = i16::from_le_bytes([d[off], d[off + 1]]);
        let d_off = u32::from_le_bytes([d[off + 2], d[off + 3], d[off + 4], d[off + 5]]);
        (carry, d_off as usize)
    }

    // Sub-stream `i` (below `k`). A sub-stream whose data would start past the
    // end of the body, or before the previous one, returns `DataTruncated`
    // with the index of its first sample.
    fn stream(&self, i: usize) -> Result<VbzkStream<'a>, DecodeError> {
        let (k, n_sub) = (self.k, self.n_sub);
        let len = if i < k - 1 {
            n_sub
        } else {
            self.n - (k - 1) * n_sub
        };
        let start = i * n_sub;
        let (initial, d_start) = self.entry(i);
        let d_end = if i + 1 < k {
            self.entry(i + 1).1
        } else {
            self.data_bytes.len()
        };
        if d_start > d_end || d_end > self.data_bytes.len() {
            return Err(DecodeError::DataTruncated { index: start });
        }
        let ctrl_start = i * (n_sub / 8);
        Ok(VbzkStream {
            ctrl: &self.ctrl[ctrl_start..ctrl_start + len.div_ceil(8)],
            data: &self.data_bytes[d_start..d_end],
            data_offset: self.header_len + self.ctrl.len() + d_start,
            start,
            len,
            initial,
        })
    }
}

// Shift a `DataTruncated` index from a sub-stream to the whole output.
fn offset_index(e: DecodeError, base: usize) -> DecodeError {
    match e {
        DecodeError::DataTruncated { index } => DecodeError::DataTruncated {
            index: base + index,
        },
        other => other,
    }
}

/// Decode VBZ-K encoded data in parallel using `k` threads, returning a new `Vec<i16>`.
#[cfg(all(feature = "alloc", feature = "std"))]
pub fn decode_vbzk_parallel(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
    let mut out = Vec::new();
    decode_vbzk_parallel_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode VBZ-K encoded data in parallel using `k` threads, appending to `out`.
///
/// Spawns `k` threads with `std::thread::scope`; each thread decodes one sub-stream
/// independently. After all threads complete the results are concatenated in order.
///
/// **Thread-spawn overhead:** this function creates a fresh OS thread scope on every
/// call. For a tight loop over many chunks (e.g. all chunks in a POD5 file), prefer
/// maintaining a persistent thread pool and dispatching sub-streams via
/// [`decode_vbz_fused_from_into`] — the `vbzk_parallel` benchmark demonstrates the
/// pattern (pre-split each chunk's ctrl/data slices, send to workers, collect).
#[cfg(all(feature = "alloc", feature = "std"))]
pub fn decode_vbzk_parallel_into(
    data: &[u8],
    n: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    crate::error::check_ctrl_len(data, n.div_ceil(8))?;
    let start = out.len();
    out.resize(start + n, 0);
    let result = decode_vbzk_parallel_to_slice(data, &mut out[start..]);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Decode VBZ-K encoded data in parallel into `out`, which holds `out.len()`
/// samples. Each thread writes its sub-stream straight into its own part of
/// `out`. On error the contents of `out` are unspecified.
///
/// Has the same per-call thread-spawn cost as [`decode_vbzk_parallel_into`].
#[cfg(all(feature = "alloc", feature = "std"))]
pub fn decode_vbzk_parallel_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    let frame = parse_vbzk(data, out.len())?;
    let streams = (0..frame.k)
        .map(|i| frame.stream(i))
        .collect::<Result<Vec<_>, _>>()?;

    // Split `out` into one part per sub-stream, in order.
    let mut parts = Vec::with_capacity(streams.len());
    let mut rest = out;
    for s in &streams {
        let (part, tail) = rest.split_at_mut(s.len);
        parts.push(part);
        rest = tail;
    }

    std::thread::scope(|scope| {
        let handles: Vec<_> = streams
            .iter()
            .zip(parts)
            .map(|(s, part)| scope.spawn(move || decode_vbzk_stream(s, part)))
            .collect();
        // Join every thread before returning, then report the first error.
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or(Err(DecodeError::DataTruncated { index: 0 }))
            })
            .collect();
        results.into_iter().collect::<Result<(), DecodeError>>()
    })
}

#[cfg(all(test, feature = "alloc"))]
mod vbzk_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    // The 0.3.1 encoder, kept to check that encode_vbzk_to_slice writes the
    // same bytes.
    fn encode_vbzk_reference(samples: &[i16], k: usize) -> Vec<u8> {
        let n = samples.len();
        let n_sub = if k > 1 { (n / k) & !7 } else { 0 };
        let effective_k = if n_sub == 0 { 1 } else { k };
        let svb = encode_vbz(samples);
        let mut out = vec![effective_k as u8];
        if effective_k == 1 {
            out.extend_from_slice(&svb);
            return out;
        }
        let ctrl = &svb[..n.div_ceil(8)];
        let mut cumulative_data_offset: u32 = 0;
        let mut ctrl_byte_idx = 0usize;
        for i in 1..effective_k {
            let split_pos = n_sub * i;
            while ctrl_byte_idx < split_pos / 8 {
                cumulative_data_offset += 8 + ctrl[ctrl_byte_idx].count_ones();
                ctrl_byte_idx += 1;
            }
            out.extend_from_slice(&samples[split_pos - 1].to_le_bytes());
            out.extend_from_slice(&cumulative_data_offset.to_le_bytes());
        }
        out.extend_from_slice(&svb);
        out
    }

    #[test]
    fn encode_to_slice_matches_reference() {
        for n in [
            0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 100, 1000, 2048, 4099,
        ] {
            let samples = make_samples(n);
            for k in [1usize, 2, 3, 4, 7, 8, 16, 255] {
                let max = vbzk_max_encoded_len(n, k);
                let enc = crate::test_util::encode_checked(&samples, max, |s, o| {
                    encode_vbzk_to_slice(s, k, o).unwrap()
                });
                assert_eq!(enc, encode_vbzk_reference(&samples, k), "n={n} k={k}");
                assert_eq!(encode_vbzk(&samples, k), enc, "n={n} k={k}");
                assert_eq!(decode_vbzk(&enc, n).unwrap(), samples, "n={n} k={k}");
            }
        }
    }

    #[test]
    fn encode_to_slice_too_small() {
        let samples = make_samples(64);
        let need = vbzk_max_encoded_len(64, 4);
        assert_eq!(need, 1 + 3 * 6 + 8 + 2 * 64);
        let mut buf = vec![0xA5u8; need - 1];
        let err = encode_vbzk_to_slice(&samples, 4, &mut buf).unwrap_err();
        assert!(
            matches!(err, EncodeError::OutputTooSmall { need: nd, have } if nd == need && have == need - 1),
            "{err:?}"
        );
        assert!(buf.iter().all(|&b| b == 0xA5), "out was modified");
    }

    #[test]
    fn k_255_roundtrips() {
        // 255 * 8 samples is the smallest input that keeps all 255 sub-streams.
        let samples = make_samples(255 * 8 + 5);
        let enc = encode_vbzk(&samples, 255);
        assert_eq!(enc[0], 255);
        assert_eq!(decode_vbzk(&enc, samples.len()).unwrap(), samples);
    }

    #[test]
    fn k_above_255_is_capped() {
        // Before 0.4.0, k=256 wrote 0 in the header byte (decode failed) and
        // k=257 wrote 1 (decode returned the wrong samples).
        let samples = make_samples(4096);
        for k in [256usize, 257, 1000] {
            let enc = encode_vbzk(&samples, k);
            assert_eq!(enc, encode_vbzk(&samples, 255), "k={k}");
            assert_eq!(decode_vbzk(&enc, samples.len()).unwrap(), samples, "k={k}");
            assert_eq!(
                vbzk_max_encoded_len(4096, k),
                vbzk_max_encoded_len(4096, 255)
            );
        }
    }

    #[test]
    #[should_panic(expected = "k must be >= 1")]
    fn k_0_panics() {
        encode_vbzk(&make_samples(10), 0);
    }

    fn make_samples(n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let base = (i as i32 % 500 - 250) as i16;
                let noise = (i as i16).wrapping_mul(37) % 7 - 3;
                base.wrapping_add(noise)
            })
            .collect()
    }

    #[test]
    fn roundtrip_k1_to_k8_n8192() {
        let samples = make_samples(8192);
        for k in [1usize, 2, 4, 8] {
            let encoded = encode_vbzk(&samples, k);
            let decoded = decode_vbzk(&encoded, samples.len()).unwrap();
            assert_eq!(decoded, samples, "k={k} roundtrip failed");
        }
    }

    #[test]
    fn sequential_matches_vbz() {
        let samples = make_samples(8192);
        let expected = decode_vbz(&encode_vbz(&samples), samples.len()).unwrap();
        for k in [1usize, 2, 4, 8] {
            let encoded = encode_vbzk(&samples, k);
            let decoded = decode_vbzk(&encoded, samples.len()).unwrap();
            assert_eq!(decoded, expected, "k={k} does not match decode_vbz output");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_matches_vbz() {
        let samples = make_samples(8192);
        let expected = decode_vbz(&encode_vbz(&samples), samples.len()).unwrap();
        for k in [1usize, 2, 4, 8] {
            let encoded = encode_vbzk(&samples, k);
            let decoded = decode_vbzk_parallel(&encoded, samples.len()).unwrap();
            assert_eq!(
                decoded, expected,
                "k={k} parallel does not match decode_vbz"
            );
        }
    }

    #[test]
    fn k1_small_n() {
        // n=0
        let encoded = encode_vbzk(&[], 1);
        assert_eq!(decode_vbzk(&encoded, 0).unwrap(), Vec::<i16>::new());

        // n=4 (less than 8, so effectively k=1 even if k>1)
        let samples = vec![1i16, 2, 3, 4];
        let encoded = encode_vbzk(&samples, 1);
        assert_eq!(decode_vbzk(&encoded, 4).unwrap(), samples);

        // n=8
        let samples: Vec<i16> = (0..8).collect();
        let encoded = encode_vbzk(&samples, 1);
        assert_eq!(decode_vbzk(&encoded, 8).unwrap(), samples);
    }

    #[test]
    fn k_larger_than_useful_falls_back_to_k1() {
        // k=1000 with n=16: n_sub = (16/1000) & !7 = 0, so falls back to k=1
        let samples: Vec<i16> = (0..16).collect();
        let encoded = encode_vbzk(&samples, 1000);
        // The header byte should be 1 (effective_k fallback).
        assert_eq!(encoded[0], 1u8, "expected header k=1");
        let decoded = decode_vbzk(&encoded, 16).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn sequential_and_parallel_agree() {
        #[cfg(feature = "std")]
        {
            let samples = make_samples(8192);
            for k in [2usize, 4, 8] {
                let encoded = encode_vbzk(&samples, k);
                let seq = decode_vbzk(&encoded, samples.len()).unwrap();
                let par = decode_vbzk_parallel(&encoded, samples.len()).unwrap();
                assert_eq!(seq, par, "k={k} sequential != parallel");
            }
        }
    }

    #[test]
    fn roundtrip_unaligned_n() {
        // n=100 with k=2,3,4: last sub-stream size differs from n_sub, exercising
        // the `if i < k-1` branch in decode_vbzk_into with unequal sub-stream lengths.
        // k=2: n_sub=(100/2)&!7=48, last=52
        // k=3: n_sub=(100/3)&!7=32, last=36
        // k=4: n_sub=(100/4)&!7=24, last=28
        let samples = make_samples(100);
        for k in [2usize, 3, 4] {
            let encoded = encode_vbzk(&samples, k);
            let decoded = decode_vbzk(&encoded, samples.len()).unwrap();
            assert_eq!(decoded, samples, "k={k} unaligned roundtrip failed");
        }
    }

    #[test]
    fn decode_error_truncated_header() {
        // k=2 needs a 7-byte header (1 + 1*6). Supply only 6 bytes to hit the
        // `data.len() < header_len` error check.
        let data = [2u8, 0, 0, 0, 0, 0]; // 6 bytes; header_len = 7
        assert!(decode_vbzk(&data, 8).is_err());
    }

    #[test]
    fn decode_error_no_ctrl_after_header() {
        // data.len() == header_len exactly: VBZ body is empty.
        // Hits the `svb.len() < ctrl_len` check (0 < n.div_ceil(8)).
        let k = 2usize;
        #[cfg(not(feature = "std"))]
        use alloc::vec;
        let data: Vec<u8> = core::iter::once(k as u8).chain(vec![0u8; 6]).collect();
        assert!(decode_vbzk(&data, 8).is_err());
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_roundtrip_unaligned_n() {
        let samples = make_samples(100);
        for k in [2usize, 3, 4] {
            let encoded = encode_vbzk(&samples, k);
            let decoded = decode_vbzk_parallel(&encoded, samples.len()).unwrap();
            assert_eq!(
                decoded, samples,
                "k={k} parallel unaligned roundtrip failed"
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn parallel_decode_error_truncated_header() {
        let data = [2u8, 0, 0, 0, 0, 0];
        assert!(decode_vbzk_parallel(&data, 8).is_err());
    }

    #[test]
    fn to_slice_matches_vec_decode() {
        for k in [1usize, 2, 3, 4, 8] {
            for n in [0usize, 1, 7, 8, 9, 31, 64, 100, 257] {
                let samples = make_samples(n);
                let enc = encode_vbzk(&samples, k);
                let mut out = vec![0i16; n];
                decode_vbzk_to_slice(&enc, &mut out).unwrap();
                assert_eq!(out, samples, "seq k={k} n={n}");
                #[cfg(feature = "std")]
                {
                    let mut par = vec![0i16; n];
                    decode_vbzk_parallel_to_slice(&enc, &mut par).unwrap();
                    assert_eq!(par, samples, "par k={k} n={n}");
                }
            }
        }
    }

    // Offsets in the header are untrusted. These used to panic on slicing.
    #[test]
    fn corrupt_data_offsets_return_error() {
        let samples = make_samples(256);
        let enc = encode_vbzk(&samples, 4);
        // Header entry i (sub-stream i + 1) holds its data offset at 1 + 6i + 2.
        let past_end = |mut e: Vec<u8>| {
            e[3..7].copy_from_slice(&u32::MAX.to_le_bytes());
            e
        };
        let decreasing = |mut e: Vec<u8>| {
            // Sub-stream 3 starts before sub-stream 2.
            let off2 = u32::from_le_bytes([e[9], e[10], e[11], e[12]]);
            e[15..19].copy_from_slice(&(off2 - 1).to_le_bytes());
            e
        };
        for (name, bad) in [
            ("past_end", past_end(enc.clone())),
            ("decreasing", decreasing(enc.clone())),
        ] {
            assert!(decode_vbzk(&bad, 256).is_err(), "{name} seq");
            #[cfg(feature = "std")]
            assert!(decode_vbzk_parallel(&bad, 256).is_err(), "{name} par");
        }
    }

    #[test]
    fn error_index_is_absolute_across_sub_streams() {
        // k=2, n=37: sub-stream 1 holds values 16..37. All 1-byte codes, so
        // dropping the last byte leaves value 36 short.
        let samples = vec![2i16; 37];
        let mut enc = encode_vbzk(&samples, 2);
        enc.pop();
        let mut out = vec![0i16; 37];
        let err = decode_vbzk_to_slice(&enc, &mut out).unwrap_err();
        assert!(
            matches!(err, DecodeError::DataTruncated { index: 36 }),
            "seq {err:?}"
        );
        #[cfg(feature = "std")]
        {
            let err = decode_vbzk_parallel_to_slice(&enc, &mut out).unwrap_err();
            assert!(
                matches!(err, DecodeError::DataTruncated { index: 36 }),
                "par {err:?}"
            );
        }
        let mut v = vec![1i16];
        assert!(decode_vbzk_into(&enc, 37, &mut v).is_err());
        assert_eq!(v, [1]);
    }
}

#[cfg(all(test, feature = "alloc"))]
mod vbz_fused_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn fused_matches_reference_empty() {
        assert_eq!(
            decode_vbz_fused(&encode_vbz(&[]), 0).unwrap(),
            &[] as &[i16]
        );
    }

    #[test]
    fn fused_matches_reference_single() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX] {
            let enc = encode_vbz(&[v]);
            assert_eq!(
                decode_vbz_fused(&enc, 1).unwrap(),
                decode_vbz(&enc, 1).unwrap(),
            );
        }
    }

    #[test]
    fn fused_matches_reference_ramp() {
        let samples: Vec<i16> = (0..128).collect();
        let enc = encode_vbz(&samples);
        assert_eq!(
            decode_vbz_fused(&enc, 128).unwrap(),
            decode_vbz(&enc, 128).unwrap(),
        );
    }

    #[test]
    fn fused_matches_reference_large() {
        let samples: Vec<i16> = (0..1024)
            .map(|i| ((i % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3))
            .collect();
        let enc = encode_vbz(&samples);
        assert_eq!(
            decode_vbz_fused(&enc, 1024).unwrap(),
            decode_vbz(&enc, 1024).unwrap(),
        );
    }

    #[test]
    fn fused_matches_reference_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX];
        let enc = encode_vbz(&samples);
        assert_eq!(
            decode_vbz_fused(&enc, 4).unwrap(),
            decode_vbz(&enc, 4).unwrap(),
        );
    }
}

// ── SVB-ZD convenience pipeline ───────────────────────────────────────────────
//
// SVB-ZD is the BLOW5/slow5lib signal compression method (hasindu2008):
//   encode: i16 samples → widen to i32 → fused zigzag-delta (i32 domain) → U32Classic → Vec<u8>
//   decode: Vec<u8> → U32Classic → inverse-zigzag-delta (i32) → truncate to i16
//
// Wire format: identical to slow5lib SVB-ZD (SLOW5_COMPRESS_SVB_ZD).
// The 4-byte little-endian element-count prefix used by slow5lib is NOT included
// here; it is the caller's responsibility (as in the slow5lib Rust wrapper).

/// Encode `i16` samples through fused zigzag-delta (i32 domain) then U32Classic.
///
/// The fused zigzag-delta computes `zigzag32(sample[i] - prev)` for each element,
/// widening to i32 before differencing. This matches the BLOW5 SVB-ZD wire format.
///
/// # Examples
///
/// ```
/// # use svb::{encode_svbzd, decode_svbzd};
/// let samples = [10i16, 11, 12, 11, 9];
/// let encoded = encode_svbzd(&samples);
/// let decoded = decode_svbzd(&encoded, samples.len()).unwrap();
/// assert_eq!(decoded, samples);
/// ```
#[cfg(feature = "alloc")]
pub fn encode_svbzd(samples: &[i16]) -> Vec<u8> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    encode_svbzd_into(samples, &mut out);
    out
}

/// Encode `i16` samples to SVB-ZD, appending the result to `out`.
#[cfg(feature = "alloc")]
pub fn encode_svbzd_into(samples: &[i16], out: &mut Vec<u8>) {
    svbzd_fused::encode_into(samples, out);
}

/// Worst-case SVB-ZD encoded size in bytes for `n` samples: `ceil(n / 4)`
/// control bytes plus 4 data bytes per sample.
///
/// Use this to size the buffer for [`encode_svbzd_to_slice`]. Encoded i16
/// samples never use more than 3 data bytes each, but the bound is kept the
/// same as [`U32Classic::max_encoded_len`](crate::u32::U32Classic::max_encoded_len)
/// because the SIMD encoders rely on it to store 16 bytes at a time.
///
/// ```
/// assert_eq!(svb::svbzd_max_encoded_len(0), 0);
/// assert_eq!(svb::svbzd_max_encoded_len(5), 2 + 4 * 5);
/// ```
pub const fn svbzd_max_encoded_len(n: usize) -> usize {
    svbzd_fused::max_encoded_len(n)
}

/// Encode `i16` samples to SVB-ZD into the start of `out` and return the
/// number of bytes written.
///
/// `out` must be at least [`svbzd_max_encoded_len(samples.len())`](svbzd_max_encoded_len)
/// bytes long, otherwise [`EncodeError::OutputTooSmall`] is returned and `out`
/// is not modified. The check is against the worst case, so a shorter slice is
/// rejected even if the actual encoded bytes would have fitted. Bytes in `out`
/// after the returned length may be overwritten.
///
/// ```
/// let samples = [10i16, 11, 12, 11, 9];
/// let mut buf = [0u8; 32];
/// let len = svb::encode_svbzd_to_slice(&samples, &mut buf).unwrap();
/// assert_eq!(&buf[..len], svb::encode_svbzd(&samples));
/// ```
pub fn encode_svbzd_to_slice(samples: &[i16], out: &mut [u8]) -> Result<usize, EncodeError> {
    let need = svbzd_max_encoded_len(samples.len());
    if out.len() < need {
        return Err(EncodeError::OutputTooSmall {
            need,
            have: out.len(),
        });
    }
    Ok(svbzd_fused::encode_to_slice(samples, out))
}

/// Decode exactly `n` `i16` samples from SVB-ZD bytes (3-pass: U32Classic → zigzag → delta).
///
/// `data` must be exactly one frame: extra bytes after it give
/// [`DecodeError::TrailingData`].
///
/// `n` must equal the number of samples originally encoded; a wrong value
/// produces incorrect output or a [`DecodeError`].
///
/// # Examples
///
/// ```
/// # use svb::{encode_svbzd, decode_svbzd};
/// let samples = [10i16, 11, 12, 11, 9];
/// let encoded = encode_svbzd(&samples);
/// assert_eq!(decode_svbzd(&encoded, samples.len()).unwrap(), samples);
/// ```
#[cfg(feature = "alloc")]
pub fn decode_svbzd(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    decode_svbzd_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode exactly `n` `i16` samples from SVB-ZD bytes, appending to `out`.
///
/// On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_svbzd_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    crate::error::check_ctrl_len(data, n.div_ceil(4))?;
    let start = out.len();
    out.resize(start + n, 0);
    let result = decode_svbzd_to_slice(data, &mut out[start..]);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Decode SVB-ZD bytes into `out` with the 3-pass decoder (U32Classic,
/// zigzag, delta).
///
/// The number of samples decoded is `out.len()`, which must equal the number
/// that was encoded. On error the contents of `out` are unspecified. The
/// U32Classic pass decodes into a temporary buffer.
/// [`decode_svbzd_fused_to_slice`] is faster and does not allocate.
///
/// ```
/// let samples = [10i16, 11, 12, 11, 9];
/// let encoded = svb::encode_svbzd(&samples);
/// let mut out = [0i16; 5];
/// svb::decode_svbzd_to_slice(&encoded, &mut out).unwrap();
/// assert_eq!(out, samples);
/// ```
#[cfg(feature = "alloc")]
// Not `vec![0; n]` for the temporary codes: that allocates with calloc,
// which measured slower than malloc + fill for small n on glibc.
#[allow(clippy::slow_vector_initialization)]
pub fn decode_svbzd_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    crate::error::check_ctrl_len(data, out.len().div_ceil(4))?;
    let mut codes = Vec::new();
    codes.resize(out.len(), 0u32);
    let used = crate::u32::U32Classic.decode_to_slice(data, &mut codes)?;
    crate::error::check_used(used, data.len())?;
    let mut acc: i32 = 0;
    for (o, zz) in out.iter_mut().zip(codes) {
        let delta = ((zz >> 1) as i32) ^ -((zz & 1) as i32);
        acc = acc.wrapping_add(delta);
        *o = acc as i16;
    }
    Ok(())
}

/// Decode SVB-ZD bytes into `i16` samples using a fused single-pass decoder.
///
/// `data` must be exactly one frame: extra bytes after it give
/// [`DecodeError::TrailingData`].
///
/// Identical output to [`decode_svbzd`] but fuses U32Classic decode, inverse-zigzag,
/// and delta prefix sum into one SIMD loop. U32Classic and zigzag work fills the
/// delta carry-chain stall, so throughput approaches the delta-alone rate.
#[cfg(feature = "alloc")]
pub fn decode_svbzd_fused(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    decode_svbzd_fused_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode SVB-ZD bytes, appending to `out`. See [`decode_svbzd_fused`].
///
/// On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_svbzd_fused_into(
    data: &[u8],
    n: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    svbzd_fused::decode_into(data, n, out)
}

/// Decode SVB-ZD bytes into `out` with the fused decoder.
///
/// The number of samples decoded is `out.len()`, which must equal the number
/// that was encoded. On error the contents of `out` are unspecified.
///
/// ```
/// let samples = [10i16, 11, 12, 11, 9];
/// let encoded = svb::encode_svbzd(&samples);
/// let mut out = [0i16; 5];
/// svb::decode_svbzd_fused_to_slice(&encoded, &mut out).unwrap();
/// assert_eq!(out, samples);
/// ```
pub fn decode_svbzd_fused_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    svbzd_fused::decode_from_to_slice(data, 0, out)
}

/// Decode an SVB-ZD half-stream starting from an arbitrary `initial_carry` value.
///
/// `initial_carry` is the i32 accumulator at the end of the preceding sub-stream
/// (0 for the first stream). This is the building block for caller-side parallel
/// decode: split the SVB-ZD payload at a pre-computed midpoint and decode each
/// half on a separate thread.
#[cfg(feature = "alloc")]
pub fn decode_svbzd_fused_from(
    data: &[u8],
    n: usize,
    initial_carry: i32,
) -> Result<Vec<i16>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    decode_svbzd_fused_from_into(data, n, initial_carry, &mut out)?;
    Ok(out)
}

/// Decode an SVB-ZD half-stream starting from `initial_carry`, appending to `out`.
///
/// See [`decode_svbzd_fused_from`]. On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_svbzd_fused_from_into(
    data: &[u8],
    n: usize,
    initial_carry: i32,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    svbzd_fused::decode_from_into(data, n, initial_carry, out)
}

/// Decode an SVB-ZD half-stream starting from `initial_carry` into `out`,
/// which holds `out.len()` samples. On error the contents of `out` are
/// unspecified.
///
/// See [`decode_svbzd_fused_from`]. Writing each half into its own part of one
/// output slice (for example with `split_at_mut`) avoids concatenating the
/// halves afterwards.
pub fn decode_svbzd_fused_from_to_slice(
    data: &[u8],
    initial_carry: i32,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    svbzd_fused::decode_from_to_slice(data, initial_carry, out)
}

// ── ex-zd convenience pipeline ─────────────────────────────────────────────────
//
// ex-zd is the BLOW5/slow5lib signal compression method that improves on
// SVB-ZD with a quantization pre-pass and a PFOR-style exception scheme
// (SLOW5_COMPRESS_EX_ZD):
//   encode: i16 samples → qts shift → zigzag-delta (u16 domain) → patched(U32Classic) → Vec<u8>
//   decode: Vec<u8> → patched(U32Classic) → inverse-zigzag-delta (u16 domain) → un-shift → i16 samples
//
// Wire format: identical to slow5lib ex-zd (SLOW5_COMPRESS_EX_ZD), version 0.
// Unlike encode_vbz/encode_svbzd, the frame is self-describing (it embeds its
// own version byte and sample count), so decode_exzd takes no `n` parameter.

#[cfg(feature = "alloc")]
const EXZD_VERSION: u8 = 0;
#[cfg(feature = "alloc")]
const EXZD_HEADER_LEN: usize = 1 + 8 + 1; // version + nin(u64) + q

/// Encode `i16` samples through qts, zigzag-delta, then patched exception encoding.
///
/// # Examples
///
/// ```
/// # use svb::{encode_exzd, decode_exzd};
/// let samples = [100i16, -100, 0, 10000, 3442, 234, 2326, 346, 213, 234];
/// let encoded = encode_exzd(&samples);
/// let decoded = decode_exzd(&encoded).unwrap();
/// assert_eq!(decoded, samples);
/// ```
#[cfg(feature = "alloc")]
pub fn encode_exzd(samples: &[i16]) -> Vec<u8> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    encode_exzd_into(samples, &mut out);
    out
}

/// Encode `i16` samples to ex-zd, appending the result to `out`.
#[cfg(feature = "alloc")]
pub fn encode_exzd_into(samples: &[i16], out: &mut Vec<u8>) {
    let (q, zd) = exzd_transform(samples);
    let mut header = [0u8; EXZD_HEADER_LEN];
    write_exzd_header(samples.len(), q, &mut header);
    out.extend_from_slice(&header);
    if let Some((&first, rest)) = zd.split_first() {
        out.extend_from_slice(&first.to_le_bytes());
        patched::encode_into(rest, out);
    }
}

/// Worst-case ex-zd frame size in bytes for `n` samples: the 10-byte header,
/// then for `n > 0` the 2-byte first value and
/// [`patched::max_encoded_len(n - 1)`](patched::max_encoded_len).
///
/// Use this to size the buffer for [`encode_exzd_to_slice`]. The worst case
/// is about 8.5 bytes per sample, reached when every zigzag-delta value is an
/// exception. The real nanopore reads in `tests/vectors` encode to about
/// 1.03 bytes per sample, so this bound is much larger than the usual output.
///
/// ```
/// assert_eq!(svb::exzd_max_encoded_len(0), 10);
/// assert_eq!(svb::exzd_max_encoded_len(1), 10 + 2 + 12);
/// ```
#[cfg(feature = "alloc")]
pub const fn exzd_max_encoded_len(n: usize) -> usize {
    if n == 0 {
        EXZD_HEADER_LEN
    } else {
        (EXZD_HEADER_LEN + 2).saturating_add(patched::max_encoded_len(n - 1))
    }
}

/// Encode `i16` samples to ex-zd into the start of `out` and return the
/// number of bytes written.
///
/// `out` must be at least [`exzd_max_encoded_len(samples.len())`](exzd_max_encoded_len)
/// bytes long, otherwise [`EncodeError::OutputTooSmall`] is returned and `out`
/// is not modified. The check is against the worst case, so a shorter slice is
/// rejected even if the actual encoded bytes would have fitted. Bytes in `out`
/// after the returned length may be overwritten.
///
/// This writes the frame directly into `out`, but the qts, delta and zigzag
/// passes still allocate temporary buffers internally.
///
/// ```
/// let samples = [100i16, -100, 0, 10000, 3442];
/// let mut buf = vec![0u8; svb::exzd_max_encoded_len(samples.len())];
/// let len = svb::encode_exzd_to_slice(&samples, &mut buf).unwrap();
/// assert_eq!(&buf[..len], svb::encode_exzd(&samples));
/// ```
#[cfg(feature = "alloc")]
pub fn encode_exzd_to_slice(samples: &[i16], out: &mut [u8]) -> Result<usize, EncodeError> {
    let need = exzd_max_encoded_len(samples.len());
    if out.len() < need {
        return Err(EncodeError::OutputTooSmall {
            need,
            have: out.len(),
        });
    }
    let (q, zd) = exzd_transform(samples);
    let (header, rest) = out.split_at_mut(EXZD_HEADER_LEN);
    write_exzd_header(samples.len(), q, header);
    let Some((&first, zd_rest)) = zd.split_first() else {
        return Ok(EXZD_HEADER_LEN);
    };
    rest[..2].copy_from_slice(&first.to_le_bytes());
    // `rest[2..]` holds patched::max_encoded_len(n - 1) bytes, checked above.
    let len = patched::encode_to_slice(zd_rest, &mut rest[2..])
        .expect("buffer sized by exzd_max_encoded_len");
    Ok(EXZD_HEADER_LEN + 2 + len)
}

// qts shift, delta and zigzag. Returns the shift and the zigzag-delta values.
#[cfg(feature = "alloc")]
fn exzd_transform(samples: &[i16]) -> (u8, Vec<u16>) {
    let q = quantize::find_qts(samples, 5);
    let shifted_storage;
    let shifted: &[i16] = if q == 0 {
        samples
    } else {
        shifted_storage = quantize::apply_shift(samples, q);
        &shifted_storage
    };
    let deltas = delta::encode(shifted);
    (q, zigzag::encode(&deltas))
}

// Writes the version, sample count and shift into `out[..EXZD_HEADER_LEN]`.
#[cfg(feature = "alloc")]
fn write_exzd_header(nin: usize, q: u8, out: &mut [u8]) {
    out[0] = EXZD_VERSION;
    out[1..9].copy_from_slice(&(nin as u64).to_le_bytes());
    out[9] = q;
}

/// Decode ex-zd bytes into `i16` samples.
///
/// `data` must be exactly one frame: extra bytes after it give
/// [`DecodeError::TrailingData`].
///
/// Unlike [`decode_vbz`] / [`decode_svbzd`], the sample count is embedded in
/// the frame itself, so no separate `n` is needed.
///
/// # Examples
///
/// ```
/// # use svb::{encode_exzd, decode_exzd};
/// let samples = [10i16, 11, 12, 11, 9];
/// let encoded = encode_exzd(&samples);
/// assert_eq!(decode_exzd(&encoded).unwrap(), samples);
/// ```
#[cfg(feature = "alloc")]
pub fn decode_exzd(data: &[u8]) -> Result<Vec<i16>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    decode_exzd_into(data, &mut out)?;
    Ok(out)
}

/// Decode ex-zd bytes into `i16` samples, appending to `out`.
///
/// On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_exzd_into(data: &[u8], out: &mut Vec<i16>) -> Result<(), DecodeError> {
    decode_exzd_append(data, out, decode_exzd_to_slice)
}

/// Decode ex-zd bytes into `out` (3-pass: patched, then zigzag, then delta).
///
/// `out.len()` must equal the sample count stored in the frame, which
/// [`exzd_decoded_len`] returns; otherwise [`DecodeError::OutputLengthMismatch`]
/// is returned. On other errors the contents of `out` are unspecified.
///
/// ```
/// let samples = [10i16, 11, 12, 11, 9];
/// let encoded = svb::encode_exzd(&samples);
/// let mut out = vec![0i16; svb::exzd_decoded_len(&encoded).unwrap()];
/// svb::decode_exzd_to_slice(&encoded, &mut out).unwrap();
/// assert_eq!(out, samples);
/// ```
#[cfg(feature = "alloc")]
pub fn decode_exzd_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut zd = Vec::new();
    let Some(q) = decode_exzd_zd(data, out.len(), &mut zd, &mut Vec::new(), &mut Vec::new())?
    else {
        return Ok(());
    };
    zigzag::decode_to_slice(&zd, out);
    delta::decode_in_place(out);
    if q != 0 {
        quantize::unshift_inplace(out, q);
    }
    Ok(())
}

/// Number of samples in an ex-zd frame, read from its header.
///
/// Use this to size the output for [`decode_exzd_to_slice`],
/// [`decode_exzd_fused_to_slice`] and [`ExzdDecoder::decode_to_slice`]. It
/// also checks the version byte, and that `data` is long enough to hold that
/// many samples, so a corrupted sample count is rejected before anything is
/// allocated for it.
///
/// ```
/// let encoded = svb::encode_exzd(&[1i16, 2, 3]);
/// assert_eq!(svb::exzd_decoded_len(&encoded).unwrap(), 3);
/// ```
#[cfg(feature = "alloc")]
pub fn exzd_decoded_len(data: &[u8]) -> Result<usize, DecodeError> {
    parse_exzd_header(data).map(|(nin, _)| nin)
}

/// Decode ex-zd bytes into `i16` samples using a fused single-pass decoder.
///
/// `data` must be exactly one frame: extra bytes after it give
/// [`DecodeError::TrailingData`].
///
/// Identical output to [`decode_exzd`], but fuses inverse-zigzag and the
/// delta prefix sum into one SIMD loop rather than two separate scans. The
/// exception scan/scatter (`patched::decode_into`) is unchanged — exceptions
/// sit at arbitrary positions and aren't a natural fit for a fixed-width
/// SIMD loop, so it stays a distinct pass.
#[cfg(feature = "alloc")]
pub fn decode_exzd_fused(data: &[u8]) -> Result<Vec<i16>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    decode_exzd_fused_into(data, &mut out)?;
    Ok(out)
}

/// Decode ex-zd bytes, appending to `out`. See [`decode_exzd_fused`].
///
/// On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_exzd_fused_into(data: &[u8], out: &mut Vec<i16>) -> Result<(), DecodeError> {
    decode_exzd_append(data, out, decode_exzd_fused_to_slice)
}

/// Decode ex-zd bytes into `out` with the fused decoder. See
/// [`decode_exzd_fused`].
///
/// `out.len()` must equal the sample count stored in the frame, which
/// [`exzd_decoded_len`] returns; otherwise [`DecodeError::OutputLengthMismatch`]
/// is returned. On other errors the contents of `out` are unspecified.
///
/// ```
/// let samples = [10i16, 11, 12, 11, 9];
/// let encoded = svb::encode_exzd(&samples);
/// let mut out = vec![0i16; svb::exzd_decoded_len(&encoded).unwrap()];
/// svb::decode_exzd_fused_to_slice(&encoded, &mut out).unwrap();
/// assert_eq!(out, samples);
/// ```
#[cfg(feature = "alloc")]
pub fn decode_exzd_fused_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut zd = Vec::new();
    decode_exzd_fused_with_scratch(data, out, &mut zd, &mut Vec::new(), &mut Vec::new())
}

// Grows `out` by the frame's sample count, decodes into the new space with
// `decode`, and truncates `out` back on error.
#[cfg(feature = "alloc")]
fn decode_exzd_append(
    data: &[u8],
    out: &mut Vec<i16>,
    decode: impl FnOnce(&[u8], &mut [i16]) -> Result<(), DecodeError>,
) -> Result<(), DecodeError> {
    // exzd_decoded_len has checked that `data` can hold this many samples, so
    // a corrupted count cannot make this resize huge.
    let n = exzd_decoded_len(data)?;
    let start = out.len();
    out.resize(start + n, 0);
    let result = decode(data, &mut out[start..]);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

#[cfg(feature = "alloc")]
fn decode_exzd_fused_with_scratch(
    data: &[u8],
    out: &mut [i16],
    zd: &mut Vec<u16>,
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
) -> Result<(), DecodeError> {
    let Some(q) = decode_exzd_zd(data, out.len(), zd, ex_pos, ex_val)? else {
        return Ok(());
    };
    exzd_fused::decode_to_slice(zd, 0, q, out);
    Ok(())
}

/// Parse and check the ex-zd header. Returns `(nin, q)`.
///
/// For `nin > 0` this also checks that `data` holds the raw first value and at
/// least `nin - 1` more bytes. Every element after the first costs at least
/// one byte, whether stored as a literal or in the exception arrays, so a
/// shorter frame cannot be valid. `nin` is an untrusted header field, and
/// without this check a corrupted or crafted frame could make a caller
/// allocate an absurd amount of memory (or panic with "capacity overflow").
#[cfg(feature = "alloc")]
fn parse_exzd_header(data: &[u8]) -> Result<(usize, u8), DecodeError> {
    if data.len() < EXZD_HEADER_LEN {
        return Err(DecodeError::ControlStreamTooShort {
            need: EXZD_HEADER_LEN,
            have: data.len(),
        });
    }
    let version = data[0];
    if version != EXZD_VERSION {
        return Err(DecodeError::UnsupportedVersion { version });
    }
    let nin = u64::from_le_bytes([
        data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
    ]) as usize;
    let q = data[9];

    if nin > 0 {
        if data.len() < EXZD_HEADER_LEN + 2 {
            return Err(DecodeError::ControlStreamTooShort {
                need: EXZD_HEADER_LEN + 2,
                have: data.len(),
            });
        }
        let remaining = data.len() - (EXZD_HEADER_LEN + 2);
        if nin - 1 > remaining {
            return Err(DecodeError::ControlStreamTooShort {
                need: nin - 1,
                have: remaining,
            });
        }
    }
    Ok((nin, q))
}

/// Check the header against `out_len`, then reconstruct the frame's
/// zigzag-delta array into `zd` (raw `zd[0]` + patched/exception decode of
/// the rest), using `ex_pos` and `ex_val` as scratch.
///
/// Returns `Some(q)` with `zd` holding exactly `nin` values, or `None` for
/// `nin == 0` (nothing to decode). `zd` is only filled where it grows, so a
/// reused buffer costs nothing extra.
#[cfg(feature = "alloc")]
fn decode_exzd_zd(
    data: &[u8],
    out_len: usize,
    zd: &mut Vec<u16>,
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
) -> Result<Option<u8>, DecodeError> {
    let (nin, q) = parse_exzd_header(data)?;
    if out_len != nin {
        return Err(DecodeError::OutputLengthMismatch {
            need: nin,
            have: out_len,
        });
    }
    if nin == 0 {
        crate::error::check_used(EXZD_HEADER_LEN, data.len())?;
        return Ok(None);
    }
    // Every element is overwritten below. `resize` keeps the existing
    // elements, so a reused buffer is only filled where it grows.
    zd.resize(nin, 0);
    zd[0] = u16::from_le_bytes([data[EXZD_HEADER_LEN], data[EXZD_HEADER_LEN + 1]]);
    let used = patched::decode_to_slice_with_scratch(
        &data[EXZD_HEADER_LEN + 2..],
        &mut zd[1..],
        ex_pos,
        ex_val,
    )
    .map_err(|e| match e {
        // Positions in the patched part to positions in the frame.
        DecodeError::TrailingData { used, len } => DecodeError::TrailingData {
            used: EXZD_HEADER_LEN + 2 + used,
            len: EXZD_HEADER_LEN + 2 + len,
        },
        other => other,
    })?;
    crate::error::check_used(EXZD_HEADER_LEN + 2 + used, data.len())?;
    Ok(Some(q))
}

/// Reusable scratch buffers for repeatedly decoding ex-zd frames with
/// [`decode_exzd_fused`]'s algorithm, avoiding a fresh heap allocation on
/// every call.
///
/// The single-shot [`decode_exzd_fused`]/[`decode_exzd_fused_into`]
/// functions allocate their working buffers (the reconstructed zigzag-delta
/// array and the decoded exception position/residual lists) fresh on every
/// call — the right default for a one-off decode. But the typical ex-zd
/// workload is a BLOW5/nanopore file with many thousands of individual
/// reads, each decoded via its own call; in that pattern, `ExzdDecoder`
/// lets the same buffers be reused across calls instead of allocated and
/// freed every time.
///
/// # Examples
///
/// ```
/// # use svb::{encode_exzd, ExzdDecoder};
/// let mut decoder = ExzdDecoder::new();
/// let mut out = Vec::new();
/// for samples in [vec![1i16, 2, 3], vec![10, 20, 30]] {
///     let encoded = encode_exzd(&samples);
///     out.clear();
///     decoder.decode_into(&encoded, &mut out).unwrap();
///     assert_eq!(out, samples);
/// }
/// ```
#[cfg(feature = "alloc")]
#[derive(Default)]
pub struct ExzdDecoder {
    zd: Vec<u16>,
    ex_pos: Vec<u32>,
    ex_val: Vec<u32>,
}

#[cfg(feature = "alloc")]
impl ExzdDecoder {
    /// Create a new decoder with empty (unallocated) scratch buffers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode ex-zd bytes into `i16` samples, appending to `out`.
    ///
    /// Identical output to [`decode_exzd_fused`], but reuses this decoder's
    /// scratch buffers instead of allocating new ones on every call. On error
    /// `out` is left at its original length.
    pub fn decode_into(&mut self, data: &[u8], out: &mut Vec<i16>) -> Result<(), DecodeError> {
        decode_exzd_append(data, out, |d, o| self.decode_to_slice(d, o))
    }

    /// Decode ex-zd bytes into `out`, reusing this decoder's scratch buffers.
    ///
    /// `out.len()` must equal the sample count stored in the frame, which
    /// [`exzd_decoded_len`] returns; otherwise
    /// [`DecodeError::OutputLengthMismatch`] is returned. On other errors the
    /// contents of `out` are unspecified.
    ///
    /// ```
    /// # use svb::{encode_exzd, exzd_decoded_len, ExzdDecoder};
    /// let mut decoder = ExzdDecoder::new();
    /// let encoded = encode_exzd(&[1i16, 2, 3]);
    /// let mut out = vec![0i16; exzd_decoded_len(&encoded).unwrap()];
    /// decoder.decode_to_slice(&encoded, &mut out).unwrap();
    /// assert_eq!(out, [1, 2, 3]);
    /// ```
    pub fn decode_to_slice(&mut self, data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
        decode_exzd_fused_with_scratch(data, out, &mut self.zd, &mut self.ex_pos, &mut self.ex_val)
    }
}

#[cfg(all(test, feature = "alloc"))]
mod svbzd_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode_svbzd(&encode_svbzd(&[]), 0).unwrap(), &[] as &[i16]);
    }

    #[test]
    fn roundtrip_single() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX] {
            assert_eq!(decode_svbzd(&encode_svbzd(&[v]), 1).unwrap(), [v]);
        }
    }

    #[test]
    fn roundtrip_ramp() {
        let samples: Vec<i16> = (0..128).collect();
        assert_eq!(decode_svbzd(&encode_svbzd(&samples), 128).unwrap(), samples);
    }

    #[test]
    fn roundtrip_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX];
        assert_eq!(decode_svbzd(&encode_svbzd(&samples), 4).unwrap(), samples);
    }

    #[test]
    fn fused_matches_3pass() {
        let samples: Vec<i16> = (0..1024)
            .map(|i| ((i % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3))
            .collect();
        let enc = encode_svbzd(&samples);
        assert_eq!(
            decode_svbzd_fused(&enc, samples.len()).unwrap(),
            decode_svbzd(&enc, samples.len()).unwrap(),
        );
    }

    #[test]
    fn fused_matches_3pass_extremes() {
        let samples = vec![i16::MIN, i16::MAX, 0, -1, 1, i16::MIN, i16::MAX, 0];
        let enc = encode_svbzd(&samples);
        assert_eq!(
            decode_svbzd_fused(&enc, samples.len()).unwrap(),
            decode_svbzd(&enc, samples.len()).unwrap(),
        );
    }

    #[test]
    fn fused_from_parallel_split() {
        // Verify that split-stream decode gives the same output as decode_svbzd.
        let n = 128usize;
        let samples: Vec<i16> = (0..n).map(|i| (i as i16 * 17).wrapping_sub(500)).collect();
        let enc = encode_svbzd(&samples);

        // Find mid-point at n/2 = 64 (multiple of 4).
        let n_half = (n / 2) & !3;
        let ctrl_len = n.div_ceil(4);
        let ctrl_half = n_half / 4;
        let ctrl = &enc[..ctrl_len];
        let data_bytes = &enc[ctrl_len..];

        let mid_data_off: usize = ctrl[..ctrl_half]
            .iter()
            .map(|&cb| crate::u32::shuffle::DATA_LEN[cb as usize] as usize)
            .sum();
        let mid_carry = samples[n_half - 1] as i32;

        // Build flat [ctrl|data] sub-streams.
        let mut stream_a = ctrl[..ctrl_half].to_vec();
        stream_a.extend_from_slice(&data_bytes[..mid_data_off]);
        let mut stream_b = ctrl[ctrl_half..ctrl_len].to_vec();
        stream_b.extend_from_slice(&data_bytes[mid_data_off..]);

        let out_a = decode_svbzd_fused_from(&stream_a, n_half, 0).unwrap();
        let out_b = decode_svbzd_fused_from(&stream_b, n - n_half, mid_carry).unwrap();
        let mut combined = out_a;
        combined.extend_from_slice(&out_b);
        assert_eq!(combined, samples);

        // The same halves decoded straight into one output slice.
        let mut out = vec![0i16; n];
        let (half_a, half_b) = out.split_at_mut(n_half);
        decode_svbzd_fused_from_to_slice(&stream_a, 0, half_a).unwrap();
        decode_svbzd_fused_from_to_slice(&stream_b, mid_carry, half_b).unwrap();
        assert_eq!(out, samples);
    }

    #[test]
    fn encode_to_slice_matches_vec_encode() {
        for n in [0usize, 1, 5, 63, 1000] {
            let samples: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(613)).collect();
            let mut buf = vec![0xA5u8; svbzd_max_encoded_len(n)];
            let len = encode_svbzd_to_slice(&samples, &mut buf).unwrap();
            assert_eq!(&buf[..len], encode_svbzd(&samples), "n={n}");
        }
    }

    #[test]
    fn encode_to_slice_too_small() {
        let samples = [1i16, 2, 3, 4, 5];
        let need = svbzd_max_encoded_len(5);
        let mut buf = vec![0xA5u8; need - 1];
        let err = encode_svbzd_to_slice(&samples, &mut buf).unwrap_err();
        assert!(
            matches!(err, EncodeError::OutputTooSmall { need: 22, have: 21 }),
            "{err:?}"
        );
        assert!(buf.iter().all(|&b| b == 0xA5), "out was modified");
    }

    #[test]
    fn three_pass_to_slice_roundtrip() {
        for n in [0usize, 1, 3, 4, 5, 63, 1000] {
            let samples: Vec<i16> = (0..n)
                .map(|i| {
                    if i % 7 == 3 {
                        i16::MIN
                    } else {
                        (i as i16).wrapping_mul(613)
                    }
                })
                .collect();
            let enc = encode_svbzd(&samples);
            let mut out = vec![0x5A5Ai16; n];
            decode_svbzd_to_slice(&enc, &mut out).unwrap();
            assert_eq!(out, samples, "n={n}");
            let mut v = vec![42i16];
            decode_svbzd_into(&enc, n, &mut v).unwrap();
            assert_eq!(v[1..], samples, "n={n}");
            assert_eq!(v[0], 42);
        }
    }

    #[test]
    fn three_pass_truncated_is_an_error() {
        let samples: Vec<i16> = (0..65).map(|i| (i as i16).wrapping_mul(613)).collect();
        let enc = encode_svbzd(&samples);
        for len in 0..enc.len() {
            let mut out = vec![0i16; 65];
            assert!(
                decode_svbzd_to_slice(&enc[..len], &mut out).is_err(),
                "len={len}"
            );
            let mut v = vec![7i16, 8];
            assert!(
                decode_svbzd_into(&enc[..len], 65, &mut v).is_err(),
                "len={len}"
            );
            assert_eq!(v, [7, 8], "len={len}");
        }
    }

    #[test]
    fn fused_to_slice_matches_vec_decode() {
        for n in [0usize, 1, 5, 63, 1000] {
            let samples: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(613)).collect();
            let enc = encode_svbzd(&samples);
            let mut out = vec![0i16; n];
            decode_svbzd_fused_to_slice(&enc, &mut out).unwrap();
            assert_eq!(out, decode_svbzd(&enc, n).unwrap(), "n={n}");
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod exzd_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode_exzd(&encode_exzd(&[])).unwrap(), &[] as &[i16]);
    }

    #[test]
    fn roundtrip_single() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX] {
            assert_eq!(decode_exzd(&encode_exzd(&[v])).unwrap(), [v]);
        }
    }

    #[test]
    fn roundtrip_ramp() {
        let samples: Vec<i16> = (0..128).collect();
        assert_eq!(decode_exzd(&encode_exzd(&samples)).unwrap(), samples);
    }

    #[test]
    fn roundtrip_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX];
        assert_eq!(decode_exzd(&encode_exzd(&samples)).unwrap(), samples);
    }

    #[test]
    fn roundtrip_quantized() {
        // All multiples of 8 but not 16, so qts should find q = 3.
        let samples: Vec<i16> = vec![0, 8, -8, 16, -24, 800, -816, 8000];
        assert_eq!(crate::quantize::find_qts(&samples, 5), 3);
        let enc = encode_exzd(&samples);
        assert_eq!(enc[9], 3);
        assert_eq!(decode_exzd(&enc).unwrap(), samples);
    }

    #[test]
    fn roundtrip_mostly_exceptions() {
        // Large alternating jumps push nearly every zigzag-delta code past the
        // exception threshold, exercising the nex > n/5 (and nex > 1) path hard.
        let samples: Vec<i16> = (0..64)
            .map(|i| if i % 2 == 0 { 0 } else { 20000 })
            .collect();
        assert_eq!(decode_exzd(&encode_exzd(&samples)).unwrap(), samples);
    }

    #[test]
    fn roundtrip_c_reference_vectors() {
        // Sample sets from slow5lib's test/unit_test_press.c
        // (press_ex_one_valid / press_ex_big_valid / press_ex_exp_valid / press_ex_huge_valid).
        let one: Vec<i16> = vec![100];
        let big: Vec<i16> = vec![100, -100, 0, 10000, 3442, 234, 2326, 346, 213, 234];
        let exp: Vec<i16> = vec![1039, 588, 588, 593, 586, 574, 570, 585, 588, 586];
        let huge: Vec<i16> = vec![
            542, 543, 545, 547, 515, 415, 370, 409, 416, 419, 416, 455, 535, 542, 548, 539, 495,
            493, 489, 486, 486, 488, 476, 476, 563, 549, 556, 568, 535, 551, 545, 512, 416, 397,
            393,
        ];
        for samples in [one, big, exp, huge] {
            assert_eq!(decode_exzd(&encode_exzd(&samples)).unwrap(), samples);
        }
    }

    #[test]
    fn encode_exzd_into_appends() {
        let mut out = encode_exzd(&[1i16, 2, 3]);
        let first_len = out.len();
        encode_exzd_into(&[4i16, 5, 6], &mut out);
        let first = decode_exzd(&out[..first_len]).unwrap();
        let second = decode_exzd(&out[first_len..]).unwrap();
        assert_eq!(first, [1, 2, 3]);
        assert_eq!(second, [4, 5, 6]);
    }

    #[test]
    fn decode_exzd_into_appends() {
        let enc = encode_exzd(&[10i16, 20, 30]);
        let mut out = vec![99i16];
        decode_exzd_into(&enc, &mut out).unwrap();
        assert_eq!(out, [99, 10, 20, 30]);
    }

    #[test]
    fn decode_exzd_rejects_bad_version() {
        let mut enc = encode_exzd(&[1i16, 2, 3]);
        enc[0] = 1;
        assert!(matches!(
            decode_exzd(&enc),
            Err(DecodeError::UnsupportedVersion { version: 1 })
        ));
    }

    #[test]
    fn decode_exzd_rejects_truncated_header() {
        assert!(matches!(
            decode_exzd(&[0u8; 5]),
            Err(DecodeError::ControlStreamTooShort { .. })
        ));
    }

    #[test]
    fn decode_exzd_rejects_truncated_body() {
        let enc = encode_exzd(&[1i16, 2, 3, 4, 5]);
        assert!(decode_exzd(&enc[..enc.len() - 1]).is_err());
    }

    #[test]
    fn fused_matches_3pass_empty() {
        let enc = encode_exzd(&[]);
        assert_eq!(decode_exzd_fused(&enc).unwrap(), decode_exzd(&enc).unwrap(),);
    }

    #[test]
    fn fused_matches_3pass_signal() {
        let samples: Vec<i16> = (0..1024)
            .map(|i| ((i % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3))
            .collect();
        let enc = encode_exzd(&samples);
        assert_eq!(decode_exzd_fused(&enc).unwrap(), decode_exzd(&enc).unwrap(),);
    }

    #[test]
    fn fused_matches_3pass_extremes() {
        let samples = vec![i16::MIN, i16::MAX, 0, -1, 1, i16::MIN, i16::MAX, 0];
        let enc = encode_exzd(&samples);
        assert_eq!(decode_exzd_fused(&enc).unwrap(), decode_exzd(&enc).unwrap(),);
    }

    #[test]
    fn fused_matches_3pass_mostly_exceptions() {
        let samples: Vec<i16> = (0..64)
            .map(|i| if i % 2 == 0 { 0 } else { 20000 })
            .collect();
        let enc = encode_exzd(&samples);
        assert_eq!(decode_exzd_fused(&enc).unwrap(), decode_exzd(&enc).unwrap(),);
    }

    #[test]
    fn fused_matches_3pass_quantized() {
        let samples: Vec<i16> = vec![0, 8, -8, 16, -24, 800, -816, 8000];
        let enc = encode_exzd(&samples);
        assert_eq!(decode_exzd_fused(&enc).unwrap(), decode_exzd(&enc).unwrap(),);
    }

    #[test]
    fn fused_roundtrip_c_reference_vectors() {
        let one: Vec<i16> = vec![100];
        let big: Vec<i16> = vec![100, -100, 0, 10000, 3442, 234, 2326, 346, 213, 234];
        let exp: Vec<i16> = vec![1039, 588, 588, 593, 586, 574, 570, 585, 588, 586];
        let huge: Vec<i16> = vec![
            542, 543, 545, 547, 515, 415, 370, 409, 416, 419, 416, 455, 535, 542, 548, 539, 495,
            493, 489, 486, 486, 488, 476, 476, 563, 549, 556, 568, 535, 551, 545, 512, 416, 397,
            393,
        ];
        for samples in [one, big, exp, huge] {
            assert_eq!(decode_exzd_fused(&encode_exzd(&samples)).unwrap(), samples);
        }
    }

    #[test]
    fn decode_exzd_fused_into_appends() {
        let enc = encode_exzd(&[10i16, 20, 30]);
        let mut out = vec![99i16];
        decode_exzd_fused_into(&enc, &mut out).unwrap();
        assert_eq!(out, [99, 10, 20, 30]);
    }

    #[test]
    fn exzd_decoder_matches_single_shot() {
        let mut decoder = ExzdDecoder::new();
        let cases: Vec<Vec<i16>> = vec![
            vec![],
            vec![0],
            vec![i16::MIN, i16::MAX, 0, -1, 1, i16::MIN, i16::MAX, 0],
            (0..1024)
                .map(|i| ((i % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3))
                .collect(),
            (0..64)
                .map(|i| if i % 2 == 0 { 0i16 } else { 20000 })
                .collect(),
        ];
        for samples in cases {
            let enc = encode_exzd(&samples);
            let mut out = Vec::new();
            decoder.decode_into(&enc, &mut out).unwrap();
            assert_eq!(out, samples);
            assert_eq!(out, decode_exzd_fused(&enc).unwrap());
        }
    }

    #[test]
    fn exzd_decoder_into_appends() {
        let mut decoder = ExzdDecoder::new();
        let enc = encode_exzd(&[10i16, 20, 30]);
        let mut out = vec![99i16];
        decoder.decode_into(&enc, &mut out).unwrap();
        assert_eq!(out, [99, 10, 20, 30]);
    }

    // ── slice API ─────────────────────────────────────────────────────────────

    // Sample sets covering qts (q = 3), exceptions at several densities,
    // extremes and the empty frame.
    fn slice_cases() -> Vec<Vec<i16>> {
        vec![
            vec![],
            vec![7],
            vec![i16::MIN, i16::MAX, 0, -1, 1, i16::MIN, i16::MAX, 0],
            (0..1000).map(|i| ((i % 97) * 8 - 400) as i16).collect(),
            (0..1000)
                .map(|i| ((i % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3))
                .collect(),
            (0..300)
                .map(|i| {
                    if i % 5 == 0 {
                        (i * 977 % 30000) as i16
                    } else {
                        (i % 9) as i16
                    }
                })
                .collect(),
            (0..64)
                .map(|i| if i % 2 == 0 { 0i16 } else { 20000 })
                .collect(),
        ]
    }

    type SliceDecoder = fn(&[u8], &mut [i16]) -> Result<(), DecodeError>;

    fn slice_decoders() -> [(&'static str, SliceDecoder); 3] {
        [
            ("3pass", decode_exzd_to_slice),
            ("fused", decode_exzd_fused_to_slice),
            ("decoder", |d, o| ExzdDecoder::new().decode_to_slice(d, o)),
        ]
    }

    #[test]
    fn to_slice_roundtrip() {
        // One decoder reused across all cases, so its buffers grow and shrink.
        let mut decoder = ExzdDecoder::new();
        for samples in slice_cases() {
            let enc = encode_exzd(&samples);
            let n = exzd_decoded_len(&enc).unwrap();
            assert_eq!(n, samples.len());
            for (name, decode) in slice_decoders() {
                let mut out = vec![0x5A5Ai16; n];
                decode(&enc, &mut out).unwrap();
                assert_eq!(out, samples, "{name} n={n}");
            }
            let mut out = vec![0x5A5Ai16; n];
            decoder.decode_to_slice(&enc, &mut out).unwrap();
            assert_eq!(out, samples, "reused decoder n={n}");
        }
    }

    #[test]
    fn encode_to_slice_matches_vec_encode() {
        for samples in slice_cases() {
            let n = samples.len();
            let max = exzd_max_encoded_len(n);
            let enc = crate::test_util::encode_checked(&samples, max, |s, o| {
                encode_exzd_to_slice(s, o).unwrap()
            });
            assert_eq!(enc, encode_exzd(&samples), "n={n}");
            assert_eq!(decode_exzd(&enc).unwrap(), samples, "n={n}");
        }
    }

    #[test]
    fn encode_to_slice_worst_case_fits() {
        // Alternating extremes make every zigzag-delta value an exception
        // with a large residual, the largest output the encoder produces.
        for n in [1usize, 2, 3, 4, 5, 8, 9, 63, 64, 65, 1000] {
            let samples: Vec<i16> = (0..n)
                .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
                .collect();
            let enc =
                crate::test_util::encode_checked(&samples, exzd_max_encoded_len(n), |s, o| {
                    encode_exzd_to_slice(s, o).unwrap()
                });
            assert_eq!(decode_exzd(&enc).unwrap(), samples, "n={n}");
        }
    }

    #[test]
    fn encode_to_slice_too_small() {
        for samples in [vec![], vec![1i16, 2, 3, 4, 5]] {
            let need = exzd_max_encoded_len(samples.len());
            let mut buf = vec![0xA5u8; need - 1];
            let err = encode_exzd_to_slice(&samples, &mut buf).unwrap_err();
            assert!(
                matches!(err, EncodeError::OutputTooSmall { need: nd, have } if nd == need && have == need - 1),
                "{err:?}"
            );
            assert!(buf.iter().all(|&b| b == 0xA5), "out was modified");
        }
        assert_eq!(exzd_max_encoded_len(0), 10);
        assert_eq!(exzd_max_encoded_len(5), 10 + 2 + 12 + 2 * (1 + 4 * 4));
    }

    #[test]
    fn to_slice_length_mismatch() {
        for samples in [vec![], vec![1i16, 2, 3]] {
            let enc = encode_exzd(&samples);
            let n = samples.len();
            for (name, decode) in slice_decoders() {
                for have in core::iter::once(n + 1).chain(n.checked_sub(1)) {
                    let mut out = vec![0i16; have];
                    let err = decode(&enc, &mut out).unwrap_err();
                    assert!(
                        matches!(err, DecodeError::OutputLengthMismatch { need, have: h } if need == n && h == have),
                        "{name} n={n} have={have}: {err:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn decoded_len_checks_header() {
        let enc = encode_exzd(&[1i16, 2, 3, 4]);
        assert_eq!(exzd_decoded_len(&enc).unwrap(), 4);
        assert_eq!(exzd_decoded_len(&encode_exzd(&[])).unwrap(), 0);

        let mut bad_version = enc.clone();
        bad_version[0] = 1;
        assert!(matches!(
            exzd_decoded_len(&bad_version),
            Err(DecodeError::UnsupportedVersion { version: 1 })
        ));
        assert!(exzd_decoded_len(&enc[..EXZD_HEADER_LEN - 1]).is_err());

        // A sample count far beyond what the frame could hold is rejected,
        // so the Vec decoders never try to allocate for it.
        let mut huge = enc.clone();
        huge[1..9].copy_from_slice(&(u64::MAX / 4).to_le_bytes());
        assert!(exzd_decoded_len(&huge).is_err());
        let mut out = vec![5i16];
        assert!(decode_exzd_fused_into(&huge, &mut out).is_err());
        assert_eq!(out, [5]);
    }

    #[test]
    fn every_truncation_is_an_error() {
        for samples in slice_cases() {
            let enc = encode_exzd(&samples);
            let n = samples.len();
            for cut in 0..enc.len() {
                for (name, decode) in slice_decoders() {
                    let mut out = vec![0i16; n];
                    assert!(
                        decode(&enc[..cut], &mut out).is_err(),
                        "{name} n={n} cut={cut}"
                    );
                }
            }
        }
    }

    #[test]
    fn vec_wrappers_restore_length_on_error() {
        let samples: Vec<i16> = (0..100).map(|i| (i * 311 % 2000) as i16).collect();
        let enc = encode_exzd(&samples);
        let short = &enc[..enc.len() - 1];
        let mut out = vec![1i16, 2];
        assert!(decode_exzd_into(short, &mut out).is_err());
        assert_eq!(out, [1, 2]);
        assert!(decode_exzd_fused_into(short, &mut out).is_err());
        assert_eq!(out, [1, 2]);
        assert!(ExzdDecoder::new().decode_into(short, &mut out).is_err());
        assert_eq!(out, [1, 2]);
    }
}

// ── book examples ─────────────────────────────────────────────────────────────
//
// Each chapter of the mdbook in docs/src is included as a doc comment here, so
// `cargo test --doc` compiles and runs its Rust code blocks against this build
// of the crate. `mdbook test` cannot do that reliably: it needs `-L` pointing
// at a directory holding exactly one build of `svb`. Blocks that are not Rust
// are marked `text`, `toml` or `sh` in the chapters so they are skipped.
#[cfg(doctest)]
mod book {
    macro_rules! chapter {
        ($name:ident, $path:literal) => {
            #[doc = include_str!(concat!("../docs/src/", $path))]
            struct $name;
        };
    }

    chapter!(Index, "index.md");
    chapter!(GettingStarted, "getting-started.md");
    chapter!(Encoding, "encoding.md");
    chapter!(Codecs, "codecs/index.md");
    chapter!(CodecSvb16, "codecs/svb16.md");
    chapter!(CodecU32Classic, "codecs/u32-classic.md");
    chapter!(CodecU32Variant0124, "codecs/u32-variant0124.md");
    chapter!(CodecU64Coder1234, "codecs/u64-coder1234.md");
    chapter!(CodecU64Coder1248, "codecs/u64-coder1248.md");
    chapter!(Transforms, "transforms.md");
    chapter!(Vbz, "vbz.md");
    chapter!(Svbzd, "svbzd.md");
    chapter!(Exzd, "exzd.md");
    chapter!(Simd, "simd.md");
    chapter!(NoStd, "no-std.md");
    chapter!(WireCompatibility, "wire-compatibility.md");
    chapter!(Performance, "performance.md");
}
