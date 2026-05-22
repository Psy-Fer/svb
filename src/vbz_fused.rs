//! Fused VBZ decoder: SVB16 + zigzag + delta in one SIMD pass.
//!
//! The three-stage VBZ pipeline (SVB16 → zigzag → delta) normally runs as
//! separate passes, each adding to total decode time. This fused version
//! collapses them into one loop, where SVB16 and zigzag work fills the
//! delta carry-chain stall (~8 cycles per 8-element block).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

// ── public entry points ───────────────────────────────────────────────────────

pub fn decode_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
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
    let ctrl = &data[..ctrl_len];
    let data_bytes = &data[ctrl_len..];
    out.reserve(n);

    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-ssse3"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSSE3 is required for pshufb; simd-ssse3/simd-avx2 features
        // declare it available at compile time. SSE2 ops (shift/add/xor) are
        // always available on x86_64.
        return unsafe { decode_ssse3(ctrl, data_bytes, n, 0, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, n, 0, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: runtime check confirms SSSE3; SSE2 always available on x86_64.
        if is_x86_feature_detected!("ssse3") {
            unsafe { decode_ssse3(ctrl, data_bytes, n, 0, out) }
        } else {
            decode_scalar(ctrl, data_bytes, n, 0, out)
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, n, 0, out) };
    }
    #[cfg(not(any(
        all(any(feature = "simd-avx2", feature = "simd-ssse3"), target_arch = "x86_64"),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(
            feature = "simd-auto",
            not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    decode_scalar(ctrl, data_bytes, n, 0, out)
}

// ── scalar fallback ───────────────────────────────────────────────────────────

fn decode_scalar(
    ctrl: &[u8],
    data: &[u8],
    n: usize,
    initial: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    let mut acc = initial;
    let mut data_pos = 0usize;
    for i in 0..n {
        let bit = (ctrl[i / 8] >> (i % 8)) & 1;
        let raw = if bit == 0 {
            if data_pos >= data.len() {
                return Err(DecodeError::DataTruncated { index: i });
            }
            let v = data[data_pos] as u16;
            data_pos += 1;
            v
        } else {
            if data_pos + 2 > data.len() {
                return Err(DecodeError::DataTruncated { index: i });
            }
            let v = u16::from_le_bytes([data[data_pos], data[data_pos + 1]]);
            data_pos += 2;
            v
        };
        // zigzag decode: (raw >> 1) ^ -(raw & 1)  [wrapping u16 negate]
        let delta = ((raw >> 1) ^ (0u16.wrapping_sub(raw & 1))) as i16;
        acc = acc.wrapping_add(delta);
        out.push(acc);
    }
    Ok(())
}

// ── SSSE3 / x86_64 ───────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    initial: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use core::arch::x86_64::*;
    use crate::u16::shuffle::TABLE;

    let base = out.len();
    // SAFETY: caller already called out.reserve(n).
    let out_ptr = unsafe { out.as_mut_ptr().add(base) };

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

            // SAFETY: decoded + 8 <= n; out was reserved for n more elements.
            _mm_storeu_si128(out_ptr.add(decoded) as *mut __m128i, result);
            acc = _mm_extract_epi16(result, 7) as i16;
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 8;
    }

    // ── padded tail: guard fired but full groups of 8 may remain ─────────────
    if decoded + 8 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 8 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = 8 + cb.count_ones() as usize;

            unsafe {
                // SAFETY: padded is 32 bytes; padded_pos <= rem - 8 <= 7;
                // load range [padded_pos, padded_pos+16) ⊆ [0,23) ⊆ [0,32).
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

    // SAFETY: every element in [base, base + decoded) was written above.
    unsafe { out.set_len(base + decoded) };

    // ── scalar tail: 0–7 remaining values ────────────────────────────────────
    if decoded < n {
        let tail = n - decoded;
        let ctrl_tail = &ctrl[ctrl_pos..];
        let data_tail = &data_bytes[data_pos..];
        let mut data_tail_pos = 0usize;
        for i in 0..tail {
            let bit = (ctrl_tail[i / 8] >> (i % 8)) & 1;
            let raw = if bit == 0 {
                if data_tail_pos >= data_tail.len() {
                    return Err(DecodeError::DataTruncated { index: decoded + i });
                }
                let v = data_tail[data_tail_pos] as u16;
                data_tail_pos += 1;
                v
            } else {
                if data_tail_pos + 2 > data_tail.len() {
                    return Err(DecodeError::DataTruncated { index: decoded + i });
                }
                let v = u16::from_le_bytes([data_tail[data_tail_pos], data_tail[data_tail_pos + 1]]);
                data_tail_pos += 2;
                v
            };
            let delta = ((raw >> 1) ^ (0u16.wrapping_sub(raw & 1))) as i16;
            acc = acc.wrapping_add(delta);
            out.push(acc);
        }
    }

    Ok(())
}

// ── NEON / AArch64 ───────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn decode_neon(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    initial: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use core::arch::aarch64::*;
    use crate::u16::shuffle::TABLE;

    let base = out.len();
    // SAFETY: caller already called out.reserve(n).
    let out_ptr = unsafe { out.as_mut_ptr().add(base) };

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;
    let mut acc = initial;

    // SAFETY: NEON is mandatory on AArch64.
    let zero = unsafe { vdupq_n_s16(0) };

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

            // SAFETY: decoded + 8 <= n; out was reserved for n more elements.
            vst1q_s16(out_ptr.add(decoded), result);
            acc = vgetq_lane_s16(result, 7);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 8;
    }

    // Padded tail.
    if decoded + 8 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 8 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = 8 + cb.count_ones() as usize;

            unsafe {
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

    unsafe { out.set_len(base + decoded) };

    if decoded < n {
        let tail = n - decoded;
        let ctrl_tail = &ctrl[ctrl_pos..];
        let data_tail = &data_bytes[data_pos..];
        let mut data_tail_pos = 0usize;
        for i in 0..tail {
            let bit = (ctrl_tail[i / 8] >> (i % 8)) & 1;
            let raw = if bit == 0 {
                if data_tail_pos >= data_tail.len() {
                    return Err(DecodeError::DataTruncated { index: decoded + i });
                }
                let v = data_tail[data_tail_pos] as u16;
                data_tail_pos += 1;
                v
            } else {
                if data_tail_pos + 2 > data_tail.len() {
                    return Err(DecodeError::DataTruncated { index: decoded + i });
                }
                let v = u16::from_le_bytes([data_tail[data_tail_pos], data_tail[data_tail_pos + 1]]);
                data_tail_pos += 2;
                v
            };
            let delta = ((raw >> 1) ^ (0u16.wrapping_sub(raw & 1))) as i16;
            acc = acc.wrapping_add(delta);
            out.push(acc);
        }
    }

    Ok(())
}
