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
use alloc::vec;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

pub mod error;
pub use error::{DecodeError, EncodeError};

#[cfg(feature = "alloc")]
pub mod delta;
#[cfg(feature = "alloc")]
pub mod patched;
#[cfg(feature = "alloc")]
pub mod quantize;
#[cfg(feature = "alloc")]
pub mod zigzag;

// ── SIMD dispatch macro ───────────────────────────────────────────────────────
//
// Generates the 5-way cfg dispatch used by every codec variant. Each arm is
// individually gated so that exactly one branch is active at a time,
// preventing unreachable_code warnings when multiple simd-* features overlap.
//
// Usage:
//   impl_dispatch! { fn name(a: A, b: B) -> R; avx2_fn, sse2_fn, neon_fn, scalar_fn }
//
// Call it with braces: rustfmt reformats parenthesised calls and breaks the `;`.
// Adding a new target architecture requires editing only the macro body here.

#[cfg(feature = "alloc")]
macro_rules! impl_dispatch {
    (fn $name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;
     $avx2_fn:path, $sse2_fn:path, $neon_fn:path, $scalar_fn:path) => {
        fn $name($($arg: $ty),*) -> $ret {
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
            $scalar_fn($($arg),*)
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
mod svbzd_fused;

#[cfg(feature = "alloc")]
mod exzd_fused;

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
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    vbz_fused::decode_from_to_slice(data, 0, out)
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

/// Decode a VBZ half-stream starting from `initial_carry` into `out`, which
/// holds `out.len()` samples. On error the contents of `out` are unspecified.
///
/// See [`decode_vbz_fused_from`] for the split-stream pattern. Writing each
/// half into its own part of one output slice (for example with
/// `split_at_mut`) avoids concatenating the halves afterwards.
#[cfg(feature = "alloc")]
pub fn decode_vbz_fused_from_to_slice(
    data: &[u8],
    initial_carry: i16,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    vbz_fused::decode_from_to_slice(data, initial_carry, out)
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
#[allow(deprecated)]
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
            .map(|i| ((i % 500 - 250) as i16).wrapping_add((i as i16).wrapping_mul(37) % 7 - 3))
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
///
/// On error `out` is left at its original length.
#[cfg(feature = "alloc")]
pub fn decode_vbzk_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
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
#[cfg(feature = "alloc")]
pub fn decode_vbzk_to_slice(data: &[u8], out: &mut [i16]) -> Result<(), DecodeError> {
    for s in parse_vbzk(data, out.len())? {
        let sub_out = &mut out[s.start..s.start + s.len];
        vbz_fused::decode_parts_to_slice(s.ctrl, s.data, s.initial, sub_out)
            .map_err(|e| offset_index(e, s.start))?;
    }
    Ok(())
}

// One VBZ-K sub-stream: its ctrl and data bytes, where its samples go in the
// output, and the carry it starts from.
#[cfg(feature = "alloc")]
struct VbzkStream<'a> {
    ctrl: &'a [u8],
    data: &'a [u8],
    start: usize,
    len: usize,
    initial: i16,
}

// Parse a VBZ-K header and split the body into its sub-streams. The data
// offsets in the header are checked: a sub-stream whose data would start past
// the end of the body, or before the previous one, returns `DataTruncated`
// with the index of that sub-stream's first sample.
#[cfg(feature = "alloc")]
fn parse_vbzk(data: &[u8], n: usize) -> Result<Vec<VbzkStream<'_>>, DecodeError> {
    if n == 0 {
        return Ok(Vec::new());
    }
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
    let ctrl = &svb[..ctrl_len];
    let data_bytes = &svb[ctrl_len..];

    // Sub-stream i (for i >= 1) starts at header entry i - 1:
    // [carry: i16 LE][data offset: u32 LE].
    let entry = |i: usize| -> (i16, usize) {
        if i == 0 {
            return (0, 0);
        }
        let off = 1 + (i - 1) * 6;
        let carry = i16::from_le_bytes([data[off], data[off + 1]]);
        let d_off =
            u32::from_le_bytes([data[off + 2], data[off + 3], data[off + 4], data[off + 5]]);
        (carry, d_off as usize)
    };

    let n_sub = (n / k) & !7;
    let mut streams = Vec::with_capacity(k);
    for i in 0..k {
        let len = if i < k - 1 {
            n_sub
        } else {
            n - (k - 1) * n_sub
        };
        let start = i * n_sub;
        let (initial, d_start) = entry(i);
        let d_end = if i + 1 < k {
            entry(i + 1).1
        } else {
            data_bytes.len()
        };
        if d_start > d_end || d_end > data_bytes.len() {
            return Err(DecodeError::DataTruncated { index: start });
        }
        let ctrl_start = i * (n_sub / 8);
        streams.push(VbzkStream {
            ctrl: &ctrl[ctrl_start..ctrl_start + len.div_ceil(8)],
            data: &data_bytes[d_start..d_end],
            start,
            len,
            initial,
        });
    }
    Ok(streams)
}

// Shift a `DataTruncated` index from a sub-stream to the whole output.
#[cfg(feature = "alloc")]
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
///
/// **Format note:** VBZ-K and VBZ2 use different headers and are not interchangeable.
/// For any new format work, use VBZ-K (`encode_vbzk` / `decode_vbzk_*`).
#[cfg(all(feature = "alloc", feature = "std"))]
pub fn decode_vbzk_parallel_into(
    data: &[u8],
    n: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
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
    let streams = parse_vbzk(data, out.len())?;

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
            .map(|(s, part)| {
                scope.spawn(move || {
                    vbz_fused::decode_parts_to_slice(s.ctrl, s.data, s.initial, part)
                        .map_err(|e| offset_index(e, s.start))
                })
            })
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
#[cfg(feature = "alloc")]
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
#[cfg(feature = "alloc")]
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

    let mut out = Vec::with_capacity(n);
    decode_svbzd_into(data, n, &mut out)?;
    Ok(out)
}

/// Decode exactly `n` `i16` samples from SVB-ZD bytes, appending to `out`.
#[cfg(feature = "alloc")]
pub fn decode_svbzd_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    let codes = crate::u32::U32Classic.decode(data, n)?;
    let mut acc: i32 = 0;
    for zz in codes {
        let delta = ((zz >> 1) as i32) ^ -((zz & 1) as i32);
        acc = acc.wrapping_add(delta);
        out.push(acc as i16);
    }
    Ok(())
}

/// Decode SVB-ZD bytes into `i16` samples using a fused single-pass decoder.
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

    let mut out = Vec::with_capacity(n);
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
#[cfg(feature = "alloc")]
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

    let mut out = Vec::with_capacity(n);
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
#[cfg(feature = "alloc")]
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

const EXZD_VERSION: u8 = 0;
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
    let nin = samples.len();
    let q = quantize::find_qts(samples, 5);

    let shifted_storage;
    let shifted: &[i16] = if q == 0 {
        samples
    } else {
        shifted_storage = quantize::apply_shift(samples, q);
        &shifted_storage
    };

    let deltas = delta::encode(shifted);
    let zd = zigzag::encode(&deltas);

    out.extend_from_slice(&EXZD_VERSION.to_le_bytes());
    out.extend_from_slice(&(nin as u64).to_le_bytes());
    out.extend_from_slice(&q.to_le_bytes());

    if nin > 0 {
        out.extend_from_slice(&zd[0].to_le_bytes());
        patched::encode_into(&zd[1..], out);
    }
}

/// Decode ex-zd bytes into `i16` samples.
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
#[cfg(feature = "alloc")]
pub fn decode_exzd_into(data: &[u8], out: &mut Vec<i16>) -> Result<(), DecodeError> {
    let Some((q, zd)) = decode_exzd_header_and_zd(data)? else {
        return Ok(());
    };

    let deltas = zigzag::decode(&zd);
    let start = out.len();
    delta::decode_into(&deltas, out);

    if q != 0 {
        quantize::unshift_inplace(&mut out[start..], q);
    }

    Ok(())
}

/// Decode ex-zd bytes into `i16` samples using a fused single-pass decoder.
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
#[cfg(feature = "alloc")]
pub fn decode_exzd_fused_into(data: &[u8], out: &mut Vec<i16>) -> Result<(), DecodeError> {
    let Some((q, zd)) = decode_exzd_header_and_zd(data)? else {
        return Ok(());
    };

    exzd_fused::decode_into(&zd, 0, q, out);

    Ok(())
}

/// Parse the ex-zd header and reconstruct the full `nin`-length zigzag-delta
/// array (raw `zd[0]` + patched/exception decode of the rest).
///
/// Returns `None` for `nin == 0` (nothing to decode). This is the shared
/// prefix between [`decode_exzd_into`] and [`decode_exzd_fused_into`]; they
/// differ only in how they turn `zd` into samples.
#[cfg(feature = "alloc")]
fn decode_exzd_header_and_zd(data: &[u8]) -> Result<Option<(u8, Vec<u16>)>, DecodeError> {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    let mut zd = Vec::new();
    let mut ex_pos = Vec::new();
    let mut ex_val = Vec::new();
    let q = decode_exzd_header_and_zd_with_scratch(data, &mut zd, &mut ex_pos, &mut ex_val)?;
    Ok(q.map(|q| (q, zd)))
}

/// Same as [`decode_exzd_header_and_zd`], but fills caller-supplied `zd`,
/// `ex_pos`, and `ex_val` scratch buffers instead of allocating fresh ones.
/// On success `zd` holds exactly `nin` values. Returns `Some(q)`, or
/// `None` for `nin == 0`. Used by [`ExzdDecoder`] to avoid a heap
/// allocation per decode when repeatedly decoding many small frames.
#[cfg(feature = "alloc")]
fn decode_exzd_header_and_zd_with_scratch(
    data: &[u8],
    zd: &mut Vec<u16>,
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
) -> Result<Option<u8>, DecodeError> {
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

    if nin == 0 {
        return Ok(None);
    }

    if data.len() < EXZD_HEADER_LEN + 2 {
        return Err(DecodeError::ControlStreamTooShort {
            need: EXZD_HEADER_LEN + 2,
            have: data.len(),
        });
    }
    let zd0 = u16::from_le_bytes([data[EXZD_HEADER_LEN], data[EXZD_HEADER_LEN + 1]]);

    // `nin` is an untrusted header field; a corrupted/crafted frame can claim
    // an absurd sample count. The patched-encoding payload can never be
    // shorter than `nin - 1` bytes (every element costs at least one byte,
    // whether stored as a literal or contributing to the exception arrays),
    // so this rejects bogus `nin` before it reaches `zd.reserve`, which would
    // otherwise panic with "capacity overflow" for a large enough value.
    let remaining = data.len() - (EXZD_HEADER_LEN + 2);
    if nin - 1 > remaining {
        return Err(DecodeError::ControlStreamTooShort {
            need: nin - 1,
            have: remaining,
        });
    }

    // Every element is overwritten below. `resize` keeps the existing
    // elements, so a reused buffer is only filled where it grows.
    zd.resize(nin, 0);
    zd[0] = zd0;
    patched::decode_to_slice_with_scratch(
        &data[EXZD_HEADER_LEN + 2..],
        &mut zd[1..],
        ex_pos,
        ex_val,
    )?;

    Ok(Some(q))
}

/// Reusable scratch buffers for repeatedly decoding ex-zd frames with
/// [`decode_exzd_fused`]'s algorithm, avoiding a fresh heap allocation on
/// every call.
///
/// The single-shot [`decode_exzd_fused`]/[`decode_exzd_fused_into`]
/// functions allocate their working buffers (the reconstructed zigzag-delta
/// array and the decoded exception position/residual lists) fresh on every
/// call — the right default for a
/// one-off decode. But the typical ex-zd workload is a BLOW5/nanopore file
/// with many thousands of individual reads, each decoded via its own call;
/// in that pattern, `ExzdDecoder` lets the same buffers be reused across
/// calls instead of allocated and freed every time.
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
    /// scratch buffers instead of allocating new ones on every call.
    pub fn decode_into(&mut self, data: &[u8], out: &mut Vec<i16>) -> Result<(), DecodeError> {
        let Some(q) = decode_exzd_header_and_zd_with_scratch(
            data,
            &mut self.zd,
            &mut self.ex_pos,
            &mut self.ex_val,
        )?
        else {
            return Ok(());
        };

        exzd_fused::decode_into(&self.zd, 0, q, out);
        Ok(())
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
}
