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

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec;

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
mod vbz_fused;

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

/// Decode a VBZ-encoded byte stream into `i16` samples using a fused single-pass decoder.
///
/// Identical output to [`decode_vbz`] but fuses SVB16, zigzag, and delta decode
/// into one SIMD loop. SVB16 and zigzag work fills the delta carry-chain stall,
/// so throughput approaches the delta-alone rate rather than the harmonic sum of
/// all three stages.
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
    let mut out = Vec::with_capacity(n);
    decode_vbz_fused_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode a VBZ-encoded byte stream, appending to `out`. See [`decode_vbz_fused`].
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    vbz_fused::decode_into(data, n, out)
}

/// Decode a VBZ half-stream starting from an arbitrary `initial_carry` value.
///
/// This is the building block for caller-side parallel decode of VBZ2 chunks.
/// Given a VBZ2-encoded payload, parse the 6-byte header for `mid_carry` and
/// `mid_data_offset`, split the SVB16 body into two independent sub-streams,
/// then decode each on a separate thread:
///
/// ```
/// # use svb::{encode_vbz2, decode_vbz_fused_from_into};
/// let n = 64usize;
/// let samples: Vec<i16> = (0..n as i16).collect();
/// let encoded = encode_vbz2(&samples);
///
/// let mid_carry      = i16::from_le_bytes([encoded[0], encoded[1]]);
/// let mid_data_offset = u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]) as usize;
/// let svb            = &encoded[6..];
/// let n_half         = (n / 2) & !7;
/// let ctrl_len       = n.div_ceil(8);
/// let ctrl_half      = n_half / 8;
///
/// // Reassemble each half as a flat [ctrl bytes | data bytes] stream.
/// let mut stream_a = svb[..ctrl_half].to_vec();
/// stream_a.extend_from_slice(&svb[ctrl_len..ctrl_len + mid_data_offset]);
/// let mut stream_b = svb[ctrl_half..ctrl_len].to_vec();
/// stream_b.extend_from_slice(&svb[ctrl_len + mid_data_offset..]);
///
/// let mut out_a = Vec::new();
/// let mut out_b = Vec::new();
/// // These two calls are independent and can run on separate threads:
/// decode_vbz_fused_from_into(&stream_a, n_half,      0,         &mut out_a).unwrap();
/// decode_vbz_fused_from_into(&stream_b, n - n_half,  mid_carry, &mut out_b).unwrap();
/// // out_a ++ out_b == samples
/// ```
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_from(
    data: &[u8],
    n: usize,
    initial_carry: i16,
) -> Result<Vec<i16>, DecodeError> {
    let mut out = Vec::with_capacity(n);
    decode_vbz_fused_from_into(data, n, initial_carry, &mut out)?;
    Ok(out)
}

/// Decode a VBZ half-stream starting from `initial_carry`, appending to `out`.
///
/// See [`decode_vbz_fused_from`] for the split-stream pattern.
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_from_into(
    data: &[u8],
    n: usize,
    initial_carry: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    vbz_fused::decode_from_into(data, n, initial_carry, out)
}

// ── VBZ2 format ───────────────────────────────────────────────────────────────
//
// VBZ2 adds a 6-byte header to the standard VBZ (SVB16) layout that stores
// the midpoint carry value and data-byte offset, enabling two-chain parallel
// decode without a pre-scan pass.
//
// Header layout (little-endian):
//   [0..2]  mid_carry:        i16  — samples[n_half - 1]
//   [2..6]  mid_data_offset:  u32  — SVB16 data bytes consumed by first n_half elements
//   [6..]   standard SVB16 layout: ctrl_len ctrl bytes + data bytes
//
// where n_half = (n / 2) & !7  (midpoint rounded down to multiple of 8).

/// Encode samples to VBZ2 format (standard VBZ with a 6-byte header enabling 2-chain decode).
///
/// The 6-byte header stores `mid_carry` (2 bytes) and `mid_data_offset` (4 bytes),
/// allowing `decode_vbz2` to skip the pre-scan and decode both halves in parallel.
///
/// **Deprecated in favour of [`encode_vbzk`].** VBZ2 is a fixed-k=2 predecessor;
/// VBZ-K generalises it to any number of sub-streams and uses a different header layout.
/// The two formats are **not interchangeable** — do not mix encoders and decoders.
#[cfg(feature = "alloc")]
#[deprecated(since = "0.1.0", note = "use encode_vbzk(samples, 2) instead")]
pub fn encode_vbz2(samples: &[i16]) -> Vec<u8> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let n = samples.len();
    if n == 0 {
        // 6-byte header with zeroes + empty body
        return vec![0u8; 6];
    }
    let n_half = (n / 2) & !7;
    // mid_carry = decoded value at position n_half - 1 = prefix sum of all deltas up to n_half.
    // Since samples[k] = sum(deltas[0..=k]) with initial=0, samples[n_half-1] = mid_carry.
    let mid_carry: i16 = if n_half > 0 { samples[n_half - 1] } else { 0 };

    let svb = encode_vbz(samples);
    let ctrl_len = n.div_ceil(8);
    let ctrl_half = n_half / 8;
    let ctrl = &svb[..ctrl_len];
    let mut mid_data_offset: u32 = 0;
    for &cb in &ctrl[..ctrl_half] {
        mid_data_offset += 8 + cb.count_ones();
    }

    let mut out = Vec::with_capacity(6 + svb.len());
    out.extend_from_slice(&mid_carry.to_le_bytes());
    out.extend_from_slice(&mid_data_offset.to_le_bytes());
    out.extend_from_slice(&svb);
    out
}

/// Decode VBZ2-encoded data (format produced by `encode_vbz2`).
#[cfg(feature = "alloc")]
#[deprecated(
    since = "0.1.0",
    note = "use decode_vbzk / decode_vbzk_parallel_into instead"
)]
#[allow(deprecated)]
pub fn decode_vbz2(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut out = Vec::new();
    decode_vbz2_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode VBZ2-encoded data into an existing Vec (avoids allocation if capacity is sufficient).
#[cfg(feature = "alloc")]
#[deprecated(
    since = "0.1.0",
    note = "use decode_vbzk_into / decode_vbzk_parallel_into instead"
)]
pub fn decode_vbz2_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    if data.len() < 6 {
        return Err(DecodeError::ControlStreamTooShort {
            need: 6,
            have: data.len(),
        });
    }
    let mid_carry = i16::from_le_bytes([data[0], data[1]]);
    let mid_data_offset = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;
    vbz_fused::decode_2chain_into(&data[6..], n, mid_carry, mid_data_offset, out)
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

#[cfg(all(test, feature = "alloc"))]
mod vbz2_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    /// Split a VBZ2 payload into two independent [ctrl|data] sub-streams.
    fn split_vbz2_streams(encoded: &[u8], n: usize) -> (i16, Vec<u8>, Vec<u8>, usize, usize) {
        let mid_carry = i16::from_le_bytes([encoded[0], encoded[1]]);
        let mid_data_offset =
            u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]) as usize;
        let svb = &encoded[6..];
        let n_half = (n / 2) & !7;
        let ctrl_len = n.div_ceil(8);
        let ctrl_half = n_half / 8;
        let mut stream_a = svb[..ctrl_half].to_vec();
        stream_a.extend_from_slice(&svb[ctrl_len..ctrl_len + mid_data_offset]);
        let mut stream_b = svb[ctrl_half..ctrl_len].to_vec();
        stream_b.extend_from_slice(&svb[ctrl_len + mid_data_offset..]);
        (mid_carry, stream_a, stream_b, n_half, n - n_half)
    }

    #[test]
    fn roundtrip_basic() {
        let samples: Vec<i16> = vec![100, 101, 103, 102, 98, 95, 97, 100];
        let encoded = encode_vbz2(&samples);
        let decoded = decode_vbz2(&encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn roundtrip_large() {
        let samples: Vec<i16> = (0..8192)
            .map(|i| {
                ((i as i32 % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3)
            })
            .collect();
        let encoded = encode_vbz2(&samples);
        let decoded = decode_vbz2(&encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn matches_vbz_output() {
        // VBZ2 must produce the same decoded output as standard VBZ.
        let samples: Vec<i16> = (0..1024)
            .map(|i| (i as i16 * 13).wrapping_sub(500))
            .collect();
        let decoded_vbz = decode_vbz(&encode_vbz(&samples), samples.len()).unwrap();
        let decoded_vbz2 = decode_vbz2(&encode_vbz2(&samples), samples.len()).unwrap();
        assert_eq!(decoded_vbz, decoded_vbz2);
    }

    #[test]
    fn roundtrip_empty() {
        let encoded = encode_vbz2(&[]);
        let decoded = decode_vbz2(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn roundtrip_small_n() {
        // n < 16 falls back to single-chain
        let samples: Vec<i16> = vec![10, 20, 15, 5, -10, -20, -5, 0, 10, 20, 15, 5, -10, -20, -5];
        let encoded = encode_vbz2(&samples);
        let decoded = decode_vbz2(&encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn roundtrip_extremes() {
        let samples = vec![
            i16::MIN,
            i16::MAX,
            0,
            -1,
            1,
            i16::MIN,
            i16::MAX,
            0,
            i16::MIN,
            i16::MAX,
            0,
            -1,
            1,
            i16::MIN,
            i16::MAX,
            0,
        ];
        let encoded = encode_vbz2(&samples);
        let decoded = decode_vbz2(&encoded, samples.len()).unwrap();
        assert_eq!(decoded, samples);
    }

    #[test]
    fn parallel_decode_correctness() {
        // Verify that a caller can split a VBZ2 payload and decode both halves
        // independently (on separate threads) and get the same result as decode_vbz2.
        let n = 8192usize;
        #[cfg(feature = "std")]
        use std::vec::Vec;
        let samples: Vec<i16> = (0..n)
            .map(|i| {
                ((i as i32 % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3)
            })
            .collect();
        let encoded = encode_vbz2(&samples);
        let (mid_carry, stream_a, stream_b, n_a, n_b) = split_vbz2_streams(&encoded, n);

        // Sequential version of the caller-side split (baseline for comparison).
        let out_a = decode_vbz_fused_from(&stream_a, n_a, 0).unwrap();
        let out_b = decode_vbz_fused_from(&stream_b, n_b, mid_carry).unwrap();
        let mut combined = out_a;
        combined.extend_from_slice(&out_b);
        assert_eq!(combined, samples);

        // Threaded version using std::thread::scope — the two calls are independent
        // and can safely run concurrently because they write into separate Vecs.
        #[cfg(feature = "std")]
        {
            let (out_a, out_b) = std::thread::scope(|s| {
                let ha = s.spawn(|| decode_vbz_fused_from(&stream_a, n_a, 0).unwrap());
                let hb = s.spawn(|| decode_vbz_fused_from(&stream_b, n_b, mid_carry).unwrap());
                (ha.join().unwrap(), hb.join().unwrap())
            });
            let mut combined = out_a;
            combined.extend_from_slice(&out_b);
            assert_eq!(combined, samples);
        }
    }
}

// ── VBZ-K format ──────────────────────────────────────────────────────────────
//
// Generalisation of VBZ2 to K independent sub-streams.
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
/// When `n_sub = (n / k) & !7` is 0 (fewer than `k*8` samples), the encoder
/// falls back to `k=1` (1-byte header + standard VBZ body).
///
/// # Panics
///
/// Panics if `k == 0`.
#[cfg(feature = "alloc")]
pub fn encode_vbzk(samples: &[i16], k: usize) -> Vec<u8> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    assert!(k != 0, "k must be >= 1");

    let n = samples.len();
    let n_sub = if k > 1 { (n / k) & !7 } else { 0 };
    let effective_k = if n_sub == 0 { 1 } else { k };

    let svb = encode_vbz(samples);

    if effective_k == 1 {
        let mut out = Vec::with_capacity(1 + svb.len());
        out.push(1u8);
        out.extend_from_slice(&svb);
        return out;
    }

    // Compute split points for i in 1..effective_k.
    let ctrl_len = n.div_ceil(8);
    let ctrl = &svb[..ctrl_len];

    // Accumulate data_offset incrementally as we walk ctrl bytes.
    let header_size = 1 + (effective_k - 1) * 6;
    let mut out = Vec::with_capacity(header_size + svb.len());
    out.push(effective_k as u8);

    let mut cumulative_data_offset: u32 = 0;
    let mut ctrl_byte_idx = 0usize;

    for i in 1..effective_k {
        let split_pos = n_sub * i;
        let ctrl_boundary = split_pos / 8;

        // Advance the cumulative data offset to ctrl_boundary.
        while ctrl_byte_idx < ctrl_boundary {
            cumulative_data_offset += 8 + ctrl[ctrl_byte_idx].count_ones();
            ctrl_byte_idx += 1;
        }

        let carry: i16 = samples[split_pos - 1];
        out.extend_from_slice(&carry.to_le_bytes());
        out.extend_from_slice(&cumulative_data_offset.to_le_bytes());
    }

    out.extend_from_slice(&svb);
    out
}

/// Decode VBZ-K encoded data, returning a new `Vec<i16>`.
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
#[cfg(feature = "alloc")]
pub fn decode_vbzk_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    if data.is_empty() {
        return Err(DecodeError::ControlStreamTooShort { need: 1, have: 0 });
    }

    let k = data[0] as usize;
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

    // Parse k-1 split points from the header.
    // carries[i] = initial carry for sub-stream i+1 (i.e., samples[n_sub*(i+1) - 1])
    // data_offsets[i] = data byte offset for sub-stream i+1's start (within data_bytes).
    // We build full arrays of length k+1 for the boundaries:
    //   sub_carry[0] = 0          (stream 0 starts with carry=0)
    //   sub_carry[i] = carries[i-1] for i in 1..k
    //   data_start[0] = 0
    //   data_start[i] = data_offsets[i-1] for i in 1..k
    //   data_start[k] = total data_bytes len (filled after parsing body length)

    #[cfg(not(feature = "std"))]
    use alloc::vec;

    let mut sub_carry = vec![0i16; k];
    let mut data_start = vec![0usize; k + 1];

    for i in 1..k {
        let off = 1 + (i - 1) * 6;
        let carry = i16::from_le_bytes([data[off], data[off + 1]]);
        let d_off = u32::from_le_bytes([data[off + 2], data[off + 3], data[off + 4], data[off + 5]])
            as usize;
        sub_carry[i] = carry;
        data_start[i] = d_off;
    }

    let svb = &data[header_len..];
    let ctrl_len = n.div_ceil(8);
    if svb.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: svb.len(),
        });
    }
    let ctrl = &svb[..ctrl_len];
    let data_bytes = &svb[ctrl_len..];

    // Set the final boundary: total data_bytes length.
    data_start[k] = data_bytes.len();

    let n_sub = (n / k) & !7;
    out.reserve(n);

    for i in 0..k {
        let sub_n = if i < k - 1 {
            n_sub
        } else {
            n - (k - 1) * n_sub
        };
        let ctrl_start = i * (n_sub / 8);
        let ctrl_end = ctrl_start + sub_n.div_ceil(8);
        let sub_ctrl = &ctrl[ctrl_start..ctrl_end];
        let sub_data = &data_bytes[data_start[i]..data_start[i + 1]];
        let initial = sub_carry[i];
        vbz_fused::decode_parts_into(sub_ctrl, sub_data, sub_n, initial, out)?;
    }

    Ok(())
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
///
/// **Format note:** VBZ-K and VBZ2 use different headers and are not interchangeable.
/// For any new format work, use VBZ-K (`encode_vbzk` / `decode_vbzk_*`).
#[cfg(all(feature = "alloc", feature = "std"))]
pub fn decode_vbzk_parallel_into(
    data: &[u8],
    n: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    if data.is_empty() {
        return Err(DecodeError::ControlStreamTooShort { need: 1, have: 0 });
    }

    let k = data[0] as usize;
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

    let mut sub_carry = vec![0i16; k];
    let mut data_start = vec![0usize; k + 1];

    for i in 1..k {
        let off = 1 + (i - 1) * 6;
        let carry = i16::from_le_bytes([data[off], data[off + 1]]);
        let d_off = u32::from_le_bytes([data[off + 2], data[off + 3], data[off + 4], data[off + 5]])
            as usize;
        sub_carry[i] = carry;
        data_start[i] = d_off;
    }

    let svb = &data[header_len..];
    let ctrl_len = n.div_ceil(8);
    if svb.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: svb.len(),
        });
    }
    let ctrl = &svb[..ctrl_len];
    let data_bytes = &svb[ctrl_len..];
    data_start[k] = data_bytes.len();

    let n_sub = (n / k) & !7;

    // Build per-sub-stream parameter list before entering the scope.
    struct SubStream<'a> {
        ctrl: &'a [u8],
        data: &'a [u8],
        sub_n: usize,
        initial: i16,
    }

    let streams: Vec<SubStream<'_>> = (0..k)
        .map(|i| {
            let sub_n = if i < k - 1 {
                n_sub
            } else {
                n - (k - 1) * n_sub
            };
            let ctrl_start = i * (n_sub / 8);
            let ctrl_end = ctrl_start + sub_n.div_ceil(8);
            SubStream {
                ctrl: &ctrl[ctrl_start..ctrl_end],
                data: &data_bytes[data_start[i]..data_start[i + 1]],
                sub_n,
                initial: sub_carry[i],
            }
        })
        .collect();

    // Decode each sub-stream in a separate thread.
    let results: Vec<Result<Vec<i16>, DecodeError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = streams
            .iter()
            .map(|s| {
                scope.spawn(move || {
                    let mut sub_out = Vec::with_capacity(s.sub_n);
                    vbz_fused::decode_parts_into(s.ctrl, s.data, s.sub_n, s.initial, &mut sub_out)?;
                    Ok(sub_out)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or(Err(DecodeError::DataTruncated { index: 0 }))
            })
            .collect()
    });

    out.reserve(n);
    for result in results {
        out.extend_from_slice(&result?);
    }
    Ok(())
}

#[cfg(all(test, feature = "alloc"))]
mod vbzk_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

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
            .map(|i| {
                ((i as i32 % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3)
            })
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
