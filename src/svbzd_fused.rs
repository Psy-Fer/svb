//! Fused SVB-ZD decoder: U32Classic + inverse-zigzag + delta in one SIMD pass.
//!
//! SVB-ZD is the signal compression method used in hasindu2008's BLOW5/slow5lib.
//! Pipeline: i16 samples → widen to i32 → fused zigzag-delta → U32Classic → bytes.
//!
//! This fused decoder collapses U32Classic decode, inverse-zigzag, and delta
//! prefix sum into one SIMD loop. U32Classic and zigzag work execute during the
//! delta carry-chain stall, hiding most of their cost.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

// ── public entry points ───────────────────────────────────────────────────────

pub fn decode_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    decode_from_into(data, n, 0, out)
}

/// Decode an SVB-ZD stream with an explicit starting carry value.
///
/// `initial` is the i32 accumulator value from the end of the previous stream
/// (0 for the first stream, `mid_carry` from the header for later sub-streams).
pub fn decode_from_into(
    data: &[u8],
    n: usize,
    initial: i32,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    let ctrl_len = n.div_ceil(4);
    if data.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: data.len(),
        });
    }
    let ctrl = &data[..ctrl_len];
    let data_bytes = &data[ctrl_len..];
    out.reserve(n);
    decode_parts_into(ctrl, data_bytes, n, initial, out)
}

/// Decode a pre-split SVB-ZD sub-stream — `ctrl` and `data_bytes` already separated.
///
/// Used by SVB-ZD-K to decode each sub-stream without copying.
pub(crate) fn decode_parts_into(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    initial: i32,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    out.reserve(n);

    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-ssse3"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSSE3 is required for pshufb; simd-ssse3/simd-avx2 features
        // declare it available at compile time.
        return unsafe { decode_ssse3(ctrl, data_bytes, n, initial, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, n, initial, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: runtime check confirms SSSE3.
        if is_x86_feature_detected!("ssse3") {
            return unsafe { decode_ssse3(ctrl, data_bytes, n, initial, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, n, initial, out) };
    }
    decode_scalar(ctrl, data_bytes, n, initial, out)
}

// ── scalar fallback ───────────────────────────────────────────────────────────

fn decode_scalar(
    ctrl: &[u8],
    data: &[u8],
    n: usize,
    initial: i32,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    const WIDTHS: [usize; 4] = [1, 2, 3, 4];
    let mut acc = initial;
    let mut data_pos = 0usize;
    for i in 0..n {
        let tag = ((ctrl[i / 4] >> (2 * (i % 4))) & 3) as usize;
        let width = WIDTHS[tag];
        if data_pos + width > data.len() {
            return Err(DecodeError::DataTruncated { index: i });
        }
        let raw = match width {
            1 => data[data_pos] as u32,
            2 => u16::from_le_bytes([data[data_pos], data[data_pos + 1]]) as u32,
            3 => u32::from_le_bytes([data[data_pos], data[data_pos + 1], data[data_pos + 2], 0]),
            _ => u32::from_le_bytes([
                data[data_pos],
                data[data_pos + 1],
                data[data_pos + 2],
                data[data_pos + 3],
            ]),
        };
        data_pos += width;
        // Inverse zigzag32: (raw >> 1) ^ -(raw & 1)
        let delta = ((raw >> 1) as i32) ^ -((raw & 1) as i32);
        acc = acc.wrapping_add(delta);
        out.push(acc as i16);
    }
    Ok(())
}

// ── SSSE3 / x86_64 ───────────────────────────────────────────────────────────

// Truncate 4×i32 → 4×i16: select low 2 bytes of each i32 lane (wrapping cast).
// Bytes [0,1] ← i32[0]; [2,3] ← i32[1]; [4,5] ← i32[2]; [6,7] ← i32[3].
// Upper 8 bytes zeroed (0x80 = PSHUFB zero sentinel).
#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
static TRUNC_I32_I16: [u8; 16] =
    [0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80];

#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn decode_ssse3(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    initial: i32,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use crate::u32::shuffle::{DATA_LEN, TABLE};
    use core::arch::x86_64::*;

    let base = out.len();
    // SAFETY: caller already called out.reserve(n).
    let out_ptr = unsafe { out.as_mut_ptr().add(base) };

    // SAFETY: TRUNC_I32_I16 is 16 bytes; SSSE3 loadu has no alignment requirement.
    let trunc_mask =
        unsafe { _mm_loadu_si128(TRUNC_I32_I16.as_ptr() as *const __m128i) };

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;
    let mut acc = initial;

    // ── fast path: ≥16 data bytes remain ─────────────────────────────────────
    while decoded + 4 <= n {
        let cb = ctrl[ctrl_pos];
        let bytes_consumed = DATA_LEN[cb as usize] as usize;

        if data_pos + 16 > data_bytes.len() {
            break;
        }

        unsafe {
            // SAFETY: TABLE[cb] is 16 bytes. data_pos + 16 <= data_bytes.len().
            let shuf = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
            let chunk = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let u32s = _mm_shuffle_epi8(chunk, shuf);

            // Inverse zigzag32: (v >> 1) ^ -(v & 1)
            let lsb = _mm_and_si128(u32s, _mm_set1_epi32(1));
            let neg = _mm_sub_epi32(_mm_setzero_si128(), lsb);
            let delta = _mm_xor_si128(_mm_srli_epi32(u32s, 1), neg);

            // Delta prefix sum (2-pass log2 scan) + inter-block carry.
            let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 4));
            let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 8));
            let result = _mm_add_epi32(delta, _mm_set1_epi32(acc));

            // Extract carry: shift element 3 to position 0, then extract as i32.
            acc = _mm_cvtsi128_si32(_mm_srli_si128(result, 12));

            // Truncate 4×i32 → 4×i16 (wrapping, low bytes) and store 8 bytes.
            let packed = _mm_shuffle_epi8(result, trunc_mask);
            // SAFETY: decoded + 4 <= n; out was reserved for n more elements.
            _mm_storel_epi64(out_ptr.add(decoded) as *mut __m128i, packed);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 4;
    }

    // ── padded tail: guard fired but full groups of 4 may remain ─────────────
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = DATA_LEN[cb as usize] as usize;

            unsafe {
                // SAFETY: padded is 32 bytes; padded_pos <= rem <= 15;
                // load range [padded_pos, padded_pos+16) ⊆ [0,31) ⊆ [0,32).
                let shuf = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                let u32s = _mm_shuffle_epi8(chunk, shuf);

                let lsb = _mm_and_si128(u32s, _mm_set1_epi32(1));
                let neg = _mm_sub_epi32(_mm_setzero_si128(), lsb);
                let delta = _mm_xor_si128(_mm_srli_epi32(u32s, 1), neg);

                let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 4));
                let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 8));
                let result = _mm_add_epi32(delta, _mm_set1_epi32(acc));

                acc = _mm_cvtsi128_si32(_mm_srli_si128(result, 12));

                let packed = _mm_shuffle_epi8(result, trunc_mask);
                _mm_storel_epi64(out_ptr.add(decoded) as *mut __m128i, packed);
            }

            padded_pos += bytes_consumed;
            data_pos += bytes_consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
    }

    // SAFETY: every element in [base, base + decoded) was written above.
    unsafe { out.set_len(base + decoded) };

    // ── scalar tail: 0–3 remaining values ────────────────────────────────────
    if decoded < n {
        let tail = n - decoded;
        let ctrl_tail = &ctrl[ctrl_pos..];
        let data_tail = &data_bytes[data_pos..];
        let mut tail_pos = 0usize;
        const WIDTHS: [usize; 4] = [1, 2, 3, 4];
        for i in 0..tail {
            let tag = ((ctrl_tail[i / 4] >> (2 * (i % 4))) & 3) as usize;
            let width = WIDTHS[tag];
            if tail_pos + width > data_tail.len() {
                return Err(DecodeError::DataTruncated { index: decoded + i });
            }
            let raw = match width {
                1 => data_tail[tail_pos] as u32,
                2 => u16::from_le_bytes([data_tail[tail_pos], data_tail[tail_pos + 1]]) as u32,
                3 => u32::from_le_bytes([
                    data_tail[tail_pos],
                    data_tail[tail_pos + 1],
                    data_tail[tail_pos + 2],
                    0,
                ]),
                _ => u32::from_le_bytes([
                    data_tail[tail_pos],
                    data_tail[tail_pos + 1],
                    data_tail[tail_pos + 2],
                    data_tail[tail_pos + 3],
                ]),
            };
            tail_pos += width;
            let delta = ((raw >> 1) as i32) ^ -((raw & 1) as i32);
            acc = acc.wrapping_add(delta);
            out.push(acc as i16);
        }
    }

    Ok(())
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
    n: usize,
    initial: i32,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use crate::u32::shuffle::{DATA_LEN, TABLE};
    use core::arch::aarch64::*;

    let base = out.len();
    // SAFETY: caller already called out.reserve(n).
    let out_ptr = unsafe { out.as_mut_ptr().add(base) };

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;
    let mut acc = initial;

    let zero_s32 = vdupq_n_s32(0);

    // ── fast path: ≥16 data bytes remain ─────────────────────────────────────
    while decoded + 4 <= n {
        let cb = ctrl[ctrl_pos];
        let bytes_consumed = DATA_LEN[cb as usize] as usize;

        if data_pos + 16 > data_bytes.len() {
            break;
        }

        unsafe {
            // SAFETY: TABLE[cb] and data_bytes bounds verified above.
            let shuf = vld1q_u8(TABLE[cb as usize].as_ptr());
            let chunk = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            let u8s = vqtbl1q_u8(chunk, shuf);

            // Reinterpret as u32, inverse zigzag32.
            let u32s = vreinterpretq_u32_u8(u8s);
            let lsb = vandq_u32(u32s, vdupq_n_u32(1));
            let neg = vsubq_u32(vdupq_n_u32(0), lsb); // 0→0, 1→0xFFFFFFFF
            let shifted = vshrq_n_u32(u32s, 1);
            let delta = vreinterpretq_s32_u32(veorq_u32(shifted, neg));

            // Delta prefix sum (2-pass log2 scan) + inter-block carry.
            // vextq_s32(a, b, N) = elements [N..N+4] from concat(a, b).
            // vextq_s32(zero, d, 3) = [0, d[0], d[1], d[2]]  (shift right by 1 elem)
            // vextq_s32(zero, d, 2) = [0, 0, d[0], d[1]]     (shift right by 2 elem)
            let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 3));
            let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 2));
            let result = vaddq_s32(delta, vdupq_n_s32(acc));

            acc = vgetq_lane_s32(result, 3);

            // Narrow 4×i32 → 4×i16 (vmovn = wrapping truncation).
            let packed = vmovn_s32(result);
            // SAFETY: decoded + 4 <= n; out was reserved for n more elements.
            vst1_s16(out_ptr.add(decoded), packed);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 4;
    }

    // ── padded tail ───────────────────────────────────────────────────────────
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = DATA_LEN[cb as usize] as usize;

            unsafe {
                // SAFETY: padded is 32 bytes; padded_pos <= rem <= 15;
                // load range [padded_pos, padded_pos+16) ⊆ [0,31) ⊆ [0,32).
                let shuf = vld1q_u8(TABLE[cb as usize].as_ptr());
                let chunk = vld1q_u8(padded.as_ptr().add(padded_pos));
                let u8s = vqtbl1q_u8(chunk, shuf);

                let u32s = vreinterpretq_u32_u8(u8s);
                let lsb = vandq_u32(u32s, vdupq_n_u32(1));
                let neg = vsubq_u32(vdupq_n_u32(0), lsb);
                let shifted = vshrq_n_u32(u32s, 1);
                let delta = vreinterpretq_s32_u32(veorq_u32(shifted, neg));

                let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 3));
                let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 2));
                let result = vaddq_s32(delta, vdupq_n_s32(acc));

                acc = vgetq_lane_s32(result, 3);
                let packed = vmovn_s32(result);
                vst1_s16(out_ptr.add(decoded), packed);
            }

            padded_pos += bytes_consumed;
            data_pos += bytes_consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
    }

    // SAFETY: every element in [base, base + decoded) was written above.
    unsafe { out.set_len(base + decoded) };

    // ── scalar tail: 0–3 remaining values ────────────────────────────────────
    if decoded < n {
        let tail = n - decoded;
        let ctrl_tail = &ctrl[ctrl_pos..];
        let data_tail = &data_bytes[data_pos..];
        let mut tail_pos = 0usize;
        const WIDTHS: [usize; 4] = [1, 2, 3, 4];
        for i in 0..tail {
            let tag = ((ctrl_tail[i / 4] >> (2 * (i % 4))) & 3) as usize;
            let width = WIDTHS[tag];
            if tail_pos + width > data_tail.len() {
                return Err(DecodeError::DataTruncated { index: decoded + i });
            }
            let raw = match width {
                1 => data_tail[tail_pos] as u32,
                2 => u16::from_le_bytes([data_tail[tail_pos], data_tail[tail_pos + 1]]) as u32,
                3 => u32::from_le_bytes([
                    data_tail[tail_pos],
                    data_tail[tail_pos + 1],
                    data_tail[tail_pos + 2],
                    0,
                ]),
                _ => u32::from_le_bytes([
                    data_tail[tail_pos],
                    data_tail[tail_pos + 1],
                    data_tail[tail_pos + 2],
                    data_tail[tail_pos + 3],
                ]),
            };
            tail_pos += width;
            let delta = ((raw >> 1) as i32) ^ -((raw & 1) as i32);
            acc = acc.wrapping_add(delta);
            out.push(acc as i16);
        }
    }

    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    /// Scalar SVB-ZD encode used as the correctness reference in tests.
    fn encode_reference(samples: &[i16]) -> Vec<u8> {
        let n = samples.len();
        let ctrl_len = n.div_ceil(4);
        let mut ctrl = vec![0u8; ctrl_len];
        let mut data: Vec<u8> = Vec::new();
        let mut prev: i32 = 0;
        for (i, &s) in samples.iter().enumerate() {
            let v = s as i32;
            let delta = v.wrapping_sub(prev);
            let zz = ((delta << 1) ^ (delta >> 31)) as u32;
            prev = v;
            let (tag, width): (u8, usize) = if zz <= 0xFF {
                (0, 1)
            } else if zz <= 0xFFFF {
                (1, 2)
            } else if zz <= 0x00FF_FFFF {
                (2, 3)
            } else {
                (3, 4)
            };
            ctrl[i / 4] |= tag << (2 * (i % 4));
            data.extend_from_slice(&zz.to_le_bytes()[..width]);
        }
        let mut out = ctrl;
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn roundtrip_n4() {
        let samples: Vec<i16> = vec![100, 105, 103, 110];
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, 4, &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_n8() {
        let samples: Vec<i16> = (0i16..8).map(|x| x * 100).collect();
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, 8, &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_n5() {
        // 1 full SIMD group + 1 scalar tail
        let samples: Vec<i16> = vec![0, -1, 1, -2, 2];
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, 5, &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_n_zero() {
        let mut dec = Vec::new();
        decode_into(&[], 0, &mut dec).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn roundtrip_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX, 0, -1, 1];
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, samples.len(), &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_large() {
        let samples: Vec<i16> = (0..256).map(|i| (i as i16 * 13).wrapping_sub(400)).collect();
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, samples.len(), &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn initial_carry_splits() {
        // Encode a block, then decode the second half using a carry from the first.
        let samples: Vec<i16> = (0..32).map(|i| (i as i16 * 50) - 500).collect();
        let enc = encode_reference(&samples);
        let ctrl_len = samples.len().div_ceil(4);
        let ctrl = &enc[..ctrl_len];
        let data = &enc[ctrl_len..];

        // Compute data offset at n=16 (4 ctrl bytes).
        let mid_data_off: usize = ctrl[..4]
            .iter()
            .map(|&cb| crate::u32::shuffle::DATA_LEN[cb as usize] as usize)
            .sum();
        let mid_carry = samples[15] as i32;

        // Decode second half independently.
        let ctrl_b = &ctrl[4..];
        let data_b = &data[mid_data_off..];
        let mut buf_b = ctrl_b.to_vec();
        buf_b.extend_from_slice(data_b);

        let mut out_b = Vec::new();
        decode_from_into(&buf_b, 16, mid_carry, &mut out_b).unwrap();
        assert_eq!(out_b, samples[16..]);
    }
}
