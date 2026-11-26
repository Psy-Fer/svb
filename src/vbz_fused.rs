//! Fused VBZ decoder: SVB16 + zigzag + delta in one SIMD pass.
//!
//! The three-stage VBZ pipeline (SVB16 → zigzag → delta) normally runs as
//! separate passes, each adding to total decode time. This fused version
//! collapses them into one loop, where SVB16 and zigzag work fills the
//! delta carry-chain stall (~8 cycles per 8-element block).
//!
//! The kernels write into a slice (`n = out.len()`); the `Vec` functions grow
//! the `Vec`, decode into the new space and truncate it again on error.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

// ── public entry points ───────────────────────────────────────────────────────

pub fn decode_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    decode_from_into(data, n, 0, out)
}

/// Decode a VBZ stream with an explicit starting carry value.
///
/// `initial` is the last decoded sample before this stream begins — 0 for the
/// first (or only) stream, or the carry stored for a sub-stream by VBZ-K.
pub fn decode_from_into(
    data: &[u8],
    n: usize,
    initial: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    crate::error::check_ctrl_len(data, n.div_ceil(8))?;
    let start = out.len();
    out.resize(start + n, 0);
    let result = decode_from_to_slice(data, initial, &mut out[start..]);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Decode `out.len()` values of a VBZ stream into `out`, starting from `initial`.
pub(crate) fn decode_from_to_slice(
    data: &[u8],
    initial: i16,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    let n = out.len();
    if n == 0 {
        return Ok(());
    }
    let ctrl_len = n.div_ceil(8);
    if data.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: data.len(),
        });
    }
    decode_parts_to_slice(&data[..ctrl_len], &data[ctrl_len..], initial, out)
}

/// Decode a pre-split VBZ sub-stream (`ctrl` and `data_bytes` already
/// separated) into `out`, which holds `out.len()` values.
///
/// Used by VBZ-K to decode each sub-stream without copying: the ctrl block and
/// data block of the full payload can be sliced at the pre-computed split points
/// and passed directly here, one call per sub-chunk. `ctrl` must hold at least
/// `out.len().div_ceil(8)` bytes. `DataTruncated` indices are relative to `out`.
pub(crate) fn decode_parts_to_slice(
    ctrl: &[u8],
    data_bytes: &[u8],
    initial: i16,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    if out.is_empty() {
        return Ok(());
    }
    assert!(ctrl.len() >= out.len().div_ceil(8));

    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-ssse3"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSSE3 is required for pshufb; simd-ssse3/simd-avx2 features
        // declare it available at compile time. SSE2 ops (shift/add/xor) are
        // always available on x86_64.
        return unsafe { decode_ssse3(ctrl, data_bytes, initial, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, initial, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        feature = "std",
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: runtime check confirms SSSE3; SSE2 always available on x86_64.
        if is_x86_feature_detected!("ssse3") {
            return unsafe { decode_ssse3(ctrl, data_bytes, initial, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, initial, out) };
    }
    decode_scalar(ctrl, data_bytes, initial, out, 0)
}

// ── scalar fallback ───────────────────────────────────────────────────────────

// Decodes `out.len()` values. Also used for the SIMD kernels' last 0-7 values,
// with `first_index` set to where the tail starts so that `DataTruncated`
// reports the index in the whole output.
fn decode_scalar(
    ctrl: &[u8],
    data: &[u8],
    initial: i16,
    out: &mut [i16],
    first_index: usize,
) -> Result<(), DecodeError> {
    let mut acc = initial;
    let mut data_pos = 0usize;
    for (i, o) in out.iter_mut().enumerate() {
        let bit = (ctrl[i / 8] >> (i % 8)) & 1;
        let raw = if bit == 0 {
            if data_pos >= data.len() {
                return Err(DecodeError::DataTruncated {
                    index: first_index + i,
                });
            }
            let v = data[data_pos] as u16;
            data_pos += 1;
            v
        } else {
            if data_pos + 2 > data.len() {
                return Err(DecodeError::DataTruncated {
                    index: first_index + i,
                });
            }
            let v = u16::from_le_bytes([data[data_pos], data[data_pos + 1]]);
            data_pos += 2;
            v
        };
        // zigzag decode: (raw >> 1) ^ -(raw & 1)  [wrapping u16 negate]
        let delta = ((raw >> 1) ^ (0u16.wrapping_sub(raw & 1))) as i16;
        acc = acc.wrapping_add(delta);
        *o = acc;
    }
    Ok(())
}

// ── SSSE3 / x86_64 ───────────────────────────────────────────────────────────

#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
// Unused under simd-auto without std, which has no runtime detection.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn decode_ssse3(
    ctrl: &[u8],
    data_bytes: &[u8],
    initial: i16,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    use crate::u16::shuffle::TABLE;
    use core::arch::x86_64::*;

    let n = out.len();
    let out_ptr = out.as_mut_ptr();

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;
    let mut acc = initial;

    // ── fast path: 16 or more data bytes remain ───────────────────────────────
    while decoded + 8 <= n {
        let cb = ctrl[ctrl_pos];
        let bytes_consumed = 8 + cb.count_ones() as usize;

        if data_pos + 16 > data_bytes.len() {
            break;
        }

        unsafe {
            // SAFETY: TABLE[cb] is 16 bytes. data_pos + 16 <= data_bytes.len().
            let shuf = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
            let chunk = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let u16s = _mm_shuffle_epi8(chunk, shuf);

            // Zigzag decode: (v >> 1) ^ -(v & 1)  [logical shift, wrapping negate]
            let lsb = _mm_and_si128(u16s, _mm_set1_epi16(1));
            let neg = _mm_sub_epi16(_mm_setzero_si128(), lsb);
            let delta = _mm_xor_si128(_mm_srli_epi16(u16s, 1), neg);

            // Delta prefix sum (3-pass log2 scan) + inter-block carry.
            let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 2));
            let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 4));
            let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 8));
            let result = _mm_add_epi16(delta, _mm_set1_epi16(acc));

            // SAFETY: decoded + 8 <= n = out.len().
            _mm_storeu_si128(out_ptr.add(decoded) as *mut __m128i, result);
            acc = _mm_extract_epi16(result, 7) as i16;
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 8;
    }

    // ── padded tail: for well-formed input, guard fired with full groups of 8
    // still remaining, fitting a zero-padded 32-byte buffer. `rem` and each
    // iteration's `bytes_consumed` are still re-validated below, since a
    // truncated/corrupted `data` (mismatched against the declared `n`) can't
    // be trusted to satisfy that bound. ─────────────────────────────────────
    if decoded + 8 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated { index: decoded });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 8 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = 8 + cb.count_ones() as usize;
            if padded_pos + bytes_consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated { index: decoded });
            }

            unsafe {
                // SAFETY: padded_pos + 16 <= padded.len() checked above.
                let shuf = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                let u16s = _mm_shuffle_epi8(chunk, shuf);

                let lsb = _mm_and_si128(u16s, _mm_set1_epi16(1));
                let neg = _mm_sub_epi16(_mm_setzero_si128(), lsb);
                let delta = _mm_xor_si128(_mm_srli_epi16(u16s, 1), neg);

                let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 2));
                let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 4));
                let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 8));
                let result = _mm_add_epi16(delta, _mm_set1_epi16(acc));

                _mm_storeu_si128(out_ptr.add(decoded) as *mut __m128i, result);
                acc = _mm_extract_epi16(result, 7) as i16;
            }

            padded_pos += bytes_consumed;
            data_pos += bytes_consumed;
            ctrl_pos += 1;
            decoded += 8;
        }
    }

    // Scalar tail: 0-7 remaining values.
    decode_scalar(
        &ctrl[ctrl_pos..],
        &data_bytes[data_pos..],
        acc,
        &mut out[decoded..],
        decoded,
    )
}

// ── NEON / AArch64 ───────────────────────────────────────────────────────────

#[cfg(all(
    any(feature = "simd-neon", feature = "simd-auto"),
    target_arch = "aarch64"
))]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn decode_neon(
    ctrl: &[u8],
    data_bytes: &[u8],
    initial: i16,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    use crate::u16::shuffle::TABLE;
    use core::arch::aarch64::*;

    let n = out.len();
    let out_ptr = out.as_mut_ptr();

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;
    let mut acc = initial;

    let zero = vdupq_n_s16(0);

    while decoded + 8 <= n {
        let cb = ctrl[ctrl_pos];
        let bytes_consumed = 8 + cb.count_ones() as usize;

        if data_pos + 16 > data_bytes.len() {
            break;
        }

        unsafe {
            // SAFETY: TABLE[cb] and data_bytes bounds verified above.
            let shuf = vld1q_u8(TABLE[cb as usize].as_ptr());
            let chunk = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            let u8s = vqtbl1q_u8(chunk, shuf);

            // Reinterpret byte pairs as u16 then zigzag decode.
            let u16s = vreinterpretq_u16_u8(u8s);
            let lsb = vandq_u16(u16s, vdupq_n_u16(1));
            let neg = vsubq_u16(vdupq_n_u16(0), lsb); // wrapping negate: 0→0, 1→0xFFFF
            let shifted = vshrq_n_u16(u16s, 1);
            let delta = vreinterpretq_s16_u16(veorq_u16(shifted, neg));

            // Delta prefix sum (3-pass log2 scan using vextq_s16) + carry.
            let delta = vaddq_s16(delta, vextq_s16(zero, delta, 7));
            let delta = vaddq_s16(delta, vextq_s16(zero, delta, 6));
            let delta = vaddq_s16(delta, vextq_s16(zero, delta, 4));
            let result = vaddq_s16(delta, vdupq_n_s16(acc));

            // SAFETY: decoded + 8 <= n = out.len().
            vst1q_s16(out_ptr.add(decoded), result);
            acc = vgetq_lane_s16(result, 7);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 8;
    }

    // Padded tail: `rem` and each iteration's `bytes_consumed` are validated
    // below, since a truncated/corrupted `data` (mismatched against the
    // declared `n`) can't be trusted to fit the fixed-size padded buffer.
    if decoded + 8 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated { index: decoded });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 8 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = 8 + cb.count_ones() as usize;
            if padded_pos + bytes_consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated { index: decoded });
            }

            unsafe {
                // SAFETY: padded_pos + 16 <= padded.len() checked above.
                let shuf = vld1q_u8(TABLE[cb as usize].as_ptr());
                let chunk = vld1q_u8(padded.as_ptr().add(padded_pos));
                let u8s = vqtbl1q_u8(chunk, shuf);

                let u16s = vreinterpretq_u16_u8(u8s);
                let lsb = vandq_u16(u16s, vdupq_n_u16(1));
                let neg = vsubq_u16(vdupq_n_u16(0), lsb);
                let shifted = vshrq_n_u16(u16s, 1);
                let delta = vreinterpretq_s16_u16(veorq_u16(shifted, neg));

                let delta = vaddq_s16(delta, vextq_s16(zero, delta, 7));
                let delta = vaddq_s16(delta, vextq_s16(zero, delta, 6));
                let delta = vaddq_s16(delta, vextq_s16(zero, delta, 4));
                let result = vaddq_s16(delta, vdupq_n_s16(acc));

                vst1q_s16(out_ptr.add(decoded), result);
                acc = vgetq_lane_s16(result, 7);
            }

            padded_pos += bytes_consumed;
            data_pos += bytes_consumed;
            ctrl_pos += 1;
            decoded += 8;
        }
    }

    // Scalar tail: 0-7 remaining values.
    decode_scalar(
        &ctrl[ctrl_pos..],
        &data_bytes[data_pos..],
        acc,
        &mut out[decoded..],
        decoded,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn encode_reference(values: &[i16]) -> Vec<u8> {
        let n = values.len();
        let ctrl_len = n.div_ceil(8);
        let mut ctrl = vec![0u8; ctrl_len];
        let mut data: Vec<u8> = Vec::new();
        let mut prev: i16 = 0;
        for (i, &v) in values.iter().enumerate() {
            let delta = v.wrapping_sub(prev) as u16;
            let zz =
                ((delta as i16).wrapping_shl(1) as u16) ^ ((delta as i16).wrapping_shr(15) as u16);
            prev = v;
            if zz <= 0xFF {
                data.push(zz as u8);
            } else {
                ctrl[i / 8] |= 1 << (i % 8);
                data.extend_from_slice(&zz.to_le_bytes());
            }
        }
        let mut out = ctrl;
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn roundtrip_n16() {
        let values: Vec<i16> = (0i16..16).map(|x| x * 100).collect();
        let encoded = encode_reference(&values);
        let mut decoded = Vec::new();
        decode_into(&encoded, 16, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn roundtrip_n17() {
        let values: Vec<i16> = (0i16..17).map(|x| x * 50 - 400).collect();
        let encoded = encode_reference(&values);
        let mut decoded = Vec::new();
        decode_into(&encoded, 17, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn roundtrip_n15() {
        let values: Vec<i16> = (0i16..15).map(|x| x * 33).collect();
        let encoded = encode_reference(&values);
        let mut decoded = Vec::new();
        decode_into(&encoded, 15, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn roundtrip_n32() {
        let values: Vec<i16> = (0i16..32).map(|x| x.wrapping_mul(200) - 3000).collect();
        let encoded = encode_reference(&values);
        let mut decoded = Vec::new();
        decode_into(&encoded, 32, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn roundtrip_n_zero() {
        let encoded: Vec<u8> = vec![];
        let mut decoded = Vec::new();
        decode_into(&encoded, 0, &mut decoded).unwrap();
        assert!(decoded.is_empty());
    }

    // ── slice API and cross-path ──────────────────────────────────────────────

    // Three kinds of data: all 1-byte codes, all 2-byte codes, and a mix.
    fn samples(n: usize, kind: u8) -> Vec<i16> {
        (0..n)
            .map(|i| match kind {
                0 => (i % 50) as i16 - 25,
                1 => (i as i16).wrapping_mul(1237),
                _ => {
                    if i % 3 == 0 {
                        (i as i16).wrapping_mul(997)
                    } else {
                        (i % 9) as i16
                    }
                }
            })
            .collect()
    }

    // Runs the scalar decoder and every SIMD kernel this CPU supports on the
    // same input, and checks they agree on success or failure and on output.
    fn check_paths(data: &[u8], n: usize, initial: i16) -> Option<Vec<i16>> {
        let ctrl_len = n.div_ceil(8);
        if data.len() < ctrl_len {
            return None;
        }
        let (ctrl, body) = data.split_at(ctrl_len);
        let mut scalar = vec![0i16; n];
        let scalar_ok = decode_scalar(ctrl, body, initial, &mut scalar, 0).is_ok();

        #[cfg(all(
            any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
            feature = "std",
            target_arch = "x86_64"
        ))]
        if std::is_x86_feature_detected!("ssse3") {
            let mut simd = vec![0i16; n];
            // SAFETY: SSSE3 confirmed at runtime.
            let ok = unsafe { decode_ssse3(ctrl, body, initial, &mut simd) }.is_ok();
            assert_eq!(ok, scalar_ok, "ssse3 ok n={n}");
            if ok {
                assert_eq!(simd, scalar, "ssse3 n={n}");
            }
        }
        #[cfg(all(
            any(feature = "simd-neon", feature = "simd-auto"),
            target_arch = "aarch64"
        ))]
        {
            let mut simd = vec![0i16; n];
            // SAFETY: NEON is mandatory on AArch64.
            let ok = unsafe { decode_neon(ctrl, body, initial, &mut simd) }.is_ok();
            assert_eq!(ok, scalar_ok, "neon ok n={n}");
            if ok {
                assert_eq!(simd, scalar, "neon n={n}");
            }
        }
        scalar_ok.then_some(scalar)
    }

    #[test]
    fn scalar_and_simd_agree_all_lengths() {
        for kind in 0..3 {
            for n in 0..=70usize {
                let values = samples(n, kind);
                let encoded = encode_reference(&values);
                let decoded = check_paths(&encoded, n, 0).expect("valid input");
                assert_eq!(decoded, values, "kind={kind} n={n}");

                // A nonzero starting carry shifts every output by that carry.
                let shifted = check_paths(&encoded, n, 1000).expect("valid input");
                let expected: Vec<i16> = values.iter().map(|v| v.wrapping_add(1000)).collect();
                assert_eq!(shifted, expected, "carry kind={kind} n={n}");

                // Every truncation of the data bytes must fail on every path.
                for cut in n.div_ceil(8)..encoded.len() {
                    assert!(
                        check_paths(&encoded[..cut], n, 0).is_none(),
                        "cut={cut} n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn tail_error_index_is_absolute() {
        // All 1-byte codes; dropping the last byte leaves the last value short.
        // Its index is in the scalar tail for these n on every path.
        for n in [13usize, 22, 37] {
            let values = vec![3i16; n];
            let mut encoded = encode_reference(&values);
            encoded.pop();
            let mut out = vec![0i16; n];
            let err = decode_from_to_slice(&encoded, 0, &mut out).unwrap_err();
            assert!(
                matches!(err, DecodeError::DataTruncated { index } if index == n - 1),
                "n={n}: {err:?}"
            );
        }
    }

    #[test]
    fn vec_wrapper_restores_length_on_error() {
        let values = samples(40, 2);
        let mut encoded = encode_reference(&values);
        encoded.pop();
        let mut out = vec![7i16, 8];
        assert!(decode_into(&encoded, 40, &mut out).is_err());
        assert_eq!(out, [7, 8]);
    }
}
