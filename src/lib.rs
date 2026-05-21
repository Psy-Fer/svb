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
//! Disable `std` and enable `alloc` for `no_std` use; all codec functionality
//! requires at least the `alloc` feature.
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

pub mod error;
pub use error::DecodeError;

#[cfg(feature = "alloc")]
pub(crate) mod coder;

#[cfg(feature = "alloc")]
pub mod delta;
#[cfg(feature = "alloc")]
pub mod zigzag;

// ── SIMD dispatch macros ──────────────────────────────────────────────────────
//
// These macros generate the 5-way cfg dispatch used by every codec variant.
// Each arm is individually gated so that exactly one branch is active at a time,
// preventing unreachable_code warnings when multiple simd-* features overlap.
//
// Usage:
//   impl_dispatch_encode!(fn_name, ElemType, avx2_fn, sse2_fn, neon_fn, scalar_fn);
//   impl_dispatch_decode!(fn_name, ElemType, avx2_fn, sse2_fn, neon_fn, scalar_fn);
//
// Adding a new target architecture requires editing only the macro bodies here.

#[cfg(feature = "alloc")]
macro_rules! impl_dispatch_encode {
    ($name:ident, $T:ty, $avx2_fn:path, $sse2_fn:path, $neon_fn:path, $scalar_fn:path) => {
        fn $name(values: &[$T], out: &mut Vec<u8>) {
            #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
            {
                // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
                return unsafe { $avx2_fn(values, out) };
            }
            #[cfg(all(
                feature = "simd-ssse3",
                not(feature = "simd-avx2"),
                target_arch = "x86_64"
            ))]
            {
                // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
                return unsafe { $sse2_fn(values, out) };
            }
            #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
            {
                // SAFETY: NEON is mandatory on AArch64.
                return unsafe { $neon_fn(values, out) };
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
                        return unsafe { $avx2_fn(values, out) };
                    }
                    if is_x86_feature_detected!("ssse3") {
                        // SAFETY: SSSE3 confirmed at runtime.
                        return unsafe { $sse2_fn(values, out) };
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { $neon_fn(values, out) };
                }
            }
            $scalar_fn(values, out)
        }
    };
}

#[cfg(feature = "alloc")]
macro_rules! impl_dispatch_decode {
    ($name:ident, $T:ty, $avx2_fn:path, $sse2_fn:path, $neon_fn:path, $scalar_fn:path) => {
        fn $name(
            data: &[u8],
            n: usize,
            out: &mut Vec<$T>,
        ) -> Result<(), crate::error::DecodeError> {
            #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
            {
                // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
                return unsafe { $avx2_fn(data, n, out) };
            }
            #[cfg(all(
                feature = "simd-ssse3",
                not(feature = "simd-avx2"),
                target_arch = "x86_64"
            ))]
            {
                // SAFETY: simd-ssse3 feature declares SSSE3 is available at runtime.
                return unsafe { $sse2_fn(data, n, out) };
            }
            #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
            {
                // SAFETY: NEON is mandatory on AArch64.
                return unsafe { $neon_fn(data, n, out) };
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
                        return unsafe { $avx2_fn(data, n, out) };
                    }
                    if is_x86_feature_detected!("ssse3") {
                        // SAFETY: SSSE3 confirmed at runtime.
                        return unsafe { $sse2_fn(data, n, out) };
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { $neon_fn(data, n, out) };
                }
            }
            $scalar_fn(data, n, out)
        }
    };
}

#[cfg(feature = "alloc")]
pub mod u16;
#[cfg(feature = "alloc")]
pub mod u32;
#[cfg(feature = "alloc")]
pub mod u64;

// ── VBZ convenience pipeline ──────────────────────────────────────────────────
//
// Implements the three-stage inner codec used by Oxford Nanopore's POD5 format:
//   encode: i16 samples → delta → zigzag → SVB16 → Vec<u8>
//   decode: Vec<u8> → SVB16 → zigzag → delta → i16 samples
//
// The outer zstd layer is handled by the caller (e.g. pod5-rs).

#[cfg(feature = "alloc")]
pub use vbz::{decode_vbz, decode_vbz_into, encode_vbz, encode_vbz_into};

#[cfg(feature = "alloc")]
mod vbz {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    use crate::error::DecodeError;
    use crate::{delta, u16::Svb16, zigzag};

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
    pub fn encode_vbz_into(samples: &[i16], out: &mut Vec<u8>) {
        let deltas = delta::encode(samples);
        let codes = zigzag::encode(&deltas);
        Svb16.encode_into(&codes, out);
    }

    /// Decode exactly `n` `i16` samples from SVB16 bytes (after zstd decompression).
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
    pub fn decode_vbz(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
        let mut out = Vec::with_capacity(n);
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
    pub fn decode_vbz_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
        let codes = Svb16.decode(data, n)?;
        let deltas = zigzag::decode(&codes);
        delta::decode_into(&deltas, out);
        Ok(())
    }
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
}
