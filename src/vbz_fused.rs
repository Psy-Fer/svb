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
    decode_from_into(data, n, 0, out)
}

/// Decode a VBZ stream with an explicit starting carry value.
///
/// `initial` is the last decoded sample before this stream begins — 0 for the
/// first (or only) stream, `mid_carry` from the VBZ2 header for the second half.
pub fn decode_from_into(
    data: &[u8],
    n: usize,
    initial: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
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
    decode_parts_into(ctrl, data_bytes, n, initial, out)
}

/// Decode a pre-split VBZ sub-stream — `ctrl` and `data_bytes` are already separated.
///
/// Used by VBZ-K to decode each sub-stream without copying: the ctrl block and
/// data block of the full payload can be sliced at the pre-computed split points
/// and passed directly here, one call per sub-chunk.
pub(crate) fn decode_parts_into(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    initial: i16,
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
        // declare it available at compile time. SSE2 ops (shift/add/xor) are
        // always available on x86_64.
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
        // SAFETY: runtime check confirms SSSE3; SSE2 always available on x86_64.
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

#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn decode_ssse3(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    initial: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use crate::u16::shuffle::TABLE;
    use core::arch::x86_64::*;

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
                let v =
                    u16::from_le_bytes([data_tail[data_tail_pos], data_tail[data_tail_pos + 1]]);
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

#[cfg(all(
    any(feature = "simd-neon", feature = "simd-auto"),
    target_arch = "aarch64"
))]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn decode_neon(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    initial: i16,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use crate::u16::shuffle::TABLE;
    use core::arch::aarch64::*;

    let base = out.len();
    // SAFETY: caller already called out.reserve(n).
    let out_ptr = unsafe { out.as_mut_ptr().add(base) };

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
                let v =
                    u16::from_le_bytes([data_tail[data_tail_pos], data_tail[data_tail_pos + 1]]);
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

// ── 2-chain parallel decode ───────────────────────────────────────────────────

/// Public dispatch entry-point for 2-chain parallel decode.
///
/// `data` is the standard SVB16 layout (ctrl_len ctrl bytes + data bytes).
/// `mid_carry` and `mid_data_offset` come from the VBZ2 6-byte header.
pub(crate) fn decode_2chain_into(
    data: &[u8],
    n: usize,
    mid_carry: i16,
    mid_data_offset: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
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
    let n_half = (n / 2) & !7;

    // If n_half < 8 there is no meaningful mid-point; fall back to single-chain.
    if n_half < 8 {
        return decode_into(data, n, out);
    }

    if mid_data_offset > data_bytes.len() {
        return Err(DecodeError::DataTruncated { index: n_half });
    }

    out.reserve(n);

    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-ssse3"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSSE3 is required; simd-ssse3/simd-avx2 features declare it
        // available at compile time.
        return unsafe {
            decode_ssse3_2chain(ctrl, data_bytes, n, n_half, mid_carry, mid_data_offset, out)
        };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe {
            decode_neon_2chain(ctrl, data_bytes, n, n_half, mid_carry, mid_data_offset, out)
        };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: runtime check confirms SSSE3; SSE2 always available on x86_64.
        if is_x86_feature_detected!("ssse3") {
            return unsafe {
                decode_ssse3_2chain(ctrl, data_bytes, n, n_half, mid_carry, mid_data_offset, out)
            };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe {
            decode_neon_2chain(ctrl, data_bytes, n, n_half, mid_carry, mid_data_offset, out)
        };
    }
    decode_scalar_2chain(ctrl, data_bytes, n, mid_carry, mid_data_offset, out)
}

// ── scalar 2-chain fallback ───────────────────────────────────────────────────
//
// The scalar path has no dependency chain exposed to overlap, so we simply
// delegate to the regular scalar decoder (chain A) and an offset decoder
// (chain B starting from mid_carry).  We call decode_scalar twice: first for
// elements [0, n_half), then for elements [n_half, n), so output is appended
// in the correct order.

fn decode_scalar_2chain(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    mid_carry: i16,
    mid_data_offset: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    let n_half = (n / 2) & !7;
    let ctrl_half = n_half / 8;
    // Chain A: elements [0, n_half).
    decode_scalar(
        &ctrl[..ctrl_half],
        &data_bytes[..mid_data_offset],
        n_half,
        0,
        out,
    )?;
    // Chain B: elements [n_half, n).
    decode_scalar(
        &ctrl[ctrl_half..],
        &data_bytes[mid_data_offset..],
        n - n_half,
        mid_carry,
        out,
    )
}

// ── SSSE3 2-chain decode ──────────────────────────────────────────────────────

#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3_2chain(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    n_half: usize,
    mid_carry: i16,
    mid_data_offset: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use crate::u16::shuffle::TABLE;
    use core::arch::x86_64::*;

    let ctrl_half = n_half / 8;
    let base = out.len();
    // SAFETY: caller called out.reserve(n).
    let out_ptr_a = unsafe { out.as_mut_ptr().add(base) };
    // SAFETY: base + n_half < base + n <= capacity.
    let out_ptr_b = unsafe { out.as_mut_ptr().add(base + n_half) };

    let mut ctrl_pos = 0usize;
    let mut data_pos_a = 0usize;
    let mut data_pos_b = 0usize;
    let mut acc_a: i16 = 0;
    let mut acc_b: i16 = mid_carry;

    // Main 2-chain loop — both chains have ≥16 data bytes available.
    while ctrl_pos < ctrl_half
        && data_pos_a + 16 <= mid_data_offset
        && data_pos_b + 16 <= data_bytes.len() - mid_data_offset
    {
        let cb_a = ctrl[ctrl_pos];
        let cb_b = ctrl[ctrl_half + ctrl_pos];

        unsafe {
            // SAFETY: TABLE indexed by u8 (<256); data bounds verified in guard.
            let shuf_a = _mm_loadu_si128(TABLE[cb_a as usize].as_ptr() as *const __m128i);
            let u16s_a = _mm_shuffle_epi8(
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos_a) as *const __m128i),
                shuf_a,
            );
            let shuf_b = _mm_loadu_si128(TABLE[cb_b as usize].as_ptr() as *const __m128i);
            let u16s_b = _mm_shuffle_epi8(
                _mm_loadu_si128(
                    data_bytes.as_ptr().add(mid_data_offset + data_pos_b) as *const __m128i
                ),
                shuf_b,
            );

            // Zigzag decode both chains.
            let lsb_a = _mm_and_si128(u16s_a, _mm_set1_epi16(1));
            let delta_a = _mm_xor_si128(
                _mm_srli_epi16(u16s_a, 1),
                _mm_sub_epi16(_mm_setzero_si128(), lsb_a),
            );
            let lsb_b = _mm_and_si128(u16s_b, _mm_set1_epi16(1));
            let delta_b = _mm_xor_si128(
                _mm_srli_epi16(u16s_b, 1),
                _mm_sub_epi16(_mm_setzero_si128(), lsb_b),
            );

            // Both prefix sums BEFORE either carry extract — critical for ILP.
            // The CPU's OOO engine overlaps chain A's carry-extract latency with
            // chain B's prefix-sum arithmetic and vice versa.
            let da = _mm_add_epi16(delta_a, _mm_slli_si128(delta_a, 2));
            let da = _mm_add_epi16(da, _mm_slli_si128(da, 4));
            let da = _mm_add_epi16(da, _mm_slli_si128(da, 8));
            let db = _mm_add_epi16(delta_b, _mm_slli_si128(delta_b, 2));
            let db = _mm_add_epi16(db, _mm_slli_si128(db, 4));
            let db = _mm_add_epi16(db, _mm_slli_si128(db, 8));

            let ra = _mm_add_epi16(da, _mm_set1_epi16(acc_a));
            let rb = _mm_add_epi16(db, _mm_set1_epi16(acc_b));
            // SAFETY: ctrl_pos < ctrl_half, so ctrl_pos*8 < n_half; within reserved capacity.
            _mm_storeu_si128(out_ptr_a.add(ctrl_pos * 8) as *mut __m128i, ra);
            // SAFETY: out_ptr_b = base + n_half; ctrl_pos*8 < n - n_half; within capacity.
            _mm_storeu_si128(out_ptr_b.add(ctrl_pos * 8) as *mut __m128i, rb);
            acc_a = _mm_extract_epi16(ra, 7) as i16;
            acc_b = _mm_extract_epi16(rb, 7) as i16;
        }

        data_pos_a += 8 + cb_a.count_ones() as usize;
        data_pos_b += 8 + cb_b.count_ones() as usize;
        ctrl_pos += 1;
    }

    let ctrl_pos_break = ctrl_pos;

    // Finish chain A: remaining ctrl[ctrl_pos_break..ctrl_half].
    // n_half is always a multiple of 8, so chain A has no scalar tail.
    {
        let mut cp = ctrl_pos_break;
        let mut dpa = data_pos_a;

        // Fast sub-path while ≥16 chain-A data bytes remain.
        while cp < ctrl_half && dpa + 16 <= mid_data_offset {
            let cb = ctrl[cp];
            unsafe {
                // SAFETY: dpa + 16 <= mid_data_offset <= data_bytes.len(). TABLE[cb] is 16 bytes.
                let shuf = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(data_bytes.as_ptr().add(dpa) as *const __m128i);
                let u16s = _mm_shuffle_epi8(chunk, shuf);
                let lsb = _mm_and_si128(u16s, _mm_set1_epi16(1));
                let neg = _mm_sub_epi16(_mm_setzero_si128(), lsb);
                let delta = _mm_xor_si128(_mm_srli_epi16(u16s, 1), neg);
                let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 2));
                let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 4));
                let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 8));
                let result = _mm_add_epi16(delta, _mm_set1_epi16(acc_a));
                // SAFETY: cp < ctrl_half, so cp*8 < n_half; within reserved capacity.
                _mm_storeu_si128(out_ptr_a.add(cp * 8) as *mut __m128i, result);
                acc_a = _mm_extract_epi16(result, 7) as i16;
            }
            dpa += 8 + cb.count_ones() as usize;
            cp += 1;
        }

        // Padded tail for chain A (guard fired: <16 bytes remain).
        if cp < ctrl_half {
            let mut padded = [0u8; 32];
            let rem = mid_data_offset - dpa;
            padded[..rem].copy_from_slice(&data_bytes[dpa..mid_data_offset]);
            let mut ppos = 0usize;
            while cp < ctrl_half {
                let cb = ctrl[cp];
                unsafe {
                    // SAFETY: padded is 32 bytes; ppos ≤ rem ≤ 15; load [ppos,ppos+16) ⊆ [0,31) ⊆ [0,32).
                    let shuf = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                    let chunk = _mm_loadu_si128(padded.as_ptr().add(ppos) as *const __m128i);
                    let u16s = _mm_shuffle_epi8(chunk, shuf);
                    let lsb = _mm_and_si128(u16s, _mm_set1_epi16(1));
                    let neg = _mm_sub_epi16(_mm_setzero_si128(), lsb);
                    let delta = _mm_xor_si128(_mm_srli_epi16(u16s, 1), neg);
                    let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 2));
                    let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 4));
                    let delta = _mm_add_epi16(delta, _mm_slli_si128(delta, 8));
                    let result = _mm_add_epi16(delta, _mm_set1_epi16(acc_a));
                    // SAFETY: cp < ctrl_half, so cp*8 < n_half; within reserved capacity.
                    _mm_storeu_si128(out_ptr_a.add(cp * 8) as *mut __m128i, result);
                    acc_a = _mm_extract_epi16(result, 7) as i16;
                }
                ppos += 8 + cb.count_ones() as usize;
                cp += 1;
            }
        }
    }
    // out[base..base+n_half] fully written by chain A.

    // Set Vec len: chain A (n_half) + chain B decoded in main loop (ctrl_pos_break * 8).
    // SAFETY: all elements in [base, base + n_half + ctrl_pos_break * 8) were written above.
    unsafe { out.set_len(base + n_half + ctrl_pos_break * 8) };

    // Finish chain B: remaining (n - n_half) - ctrl_pos_break * 8 elements.
    let ctrl_b_rest = &ctrl[ctrl_half + ctrl_pos_break..];
    let data_b_rest = &data_bytes[mid_data_offset + data_pos_b..];
    let n_b_rem = (n - n_half) - ctrl_pos_break * 8;
    // SAFETY: same SSSE3 feature gate as this function.
    unsafe { decode_ssse3(ctrl_b_rest, data_b_rest, n_b_rem, acc_b, out) }
}

// ── NEON 2-chain decode ───────────────────────────────────────────────────────

#[cfg(all(
    any(feature = "simd-neon", feature = "simd-auto"),
    target_arch = "aarch64"
))]
#[target_feature(enable = "neon")]
unsafe fn decode_neon_2chain(
    ctrl: &[u8],
    data_bytes: &[u8],
    n: usize,
    n_half: usize,
    mid_carry: i16,
    mid_data_offset: usize,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    use crate::u16::shuffle::TABLE;
    use core::arch::aarch64::*;

    let ctrl_half = n_half / 8;
    let base = out.len();
    // SAFETY: caller called out.reserve(n).
    let out_ptr_a = unsafe { out.as_mut_ptr().add(base) };
    // SAFETY: base + n_half < base + n <= capacity.
    let out_ptr_b = unsafe { out.as_mut_ptr().add(base + n_half) };

    let mut ctrl_pos = 0usize;
    let mut data_pos_a = 0usize;
    let mut data_pos_b = 0usize;
    let mut acc_a: i16 = 0;
    let mut acc_b: i16 = mid_carry;

    let zero = vdupq_n_s16(0);

    // Main 2-chain loop — both chains have ≥16 data bytes available.
    while ctrl_pos < ctrl_half
        && data_pos_a + 16 <= mid_data_offset
        && data_pos_b + 16 <= data_bytes.len() - mid_data_offset
    {
        let cb_a = ctrl[ctrl_pos];
        let cb_b = ctrl[ctrl_half + ctrl_pos];

        unsafe {
            // SAFETY: TABLE indexed by u8 (<256); data bounds verified in guard.
            let shuf_a = vld1q_u8(TABLE[cb_a as usize].as_ptr());
            let chunk_a = vld1q_u8(data_bytes.as_ptr().add(data_pos_a));
            let u8s_a = vqtbl1q_u8(chunk_a, shuf_a);
            let shuf_b = vld1q_u8(TABLE[cb_b as usize].as_ptr());
            let chunk_b = vld1q_u8(data_bytes.as_ptr().add(mid_data_offset + data_pos_b));
            let u8s_b = vqtbl1q_u8(chunk_b, shuf_b);

            // Zigzag decode both chains.
            let u16s_a = vreinterpretq_u16_u8(u8s_a);
            let lsb_a = vandq_u16(u16s_a, vdupq_n_u16(1));
            let neg_a = vsubq_u16(vdupq_n_u16(0), lsb_a);
            let delta_a = vreinterpretq_s16_u16(veorq_u16(vshrq_n_u16(u16s_a, 1), neg_a));

            let u16s_b = vreinterpretq_u16_u8(u8s_b);
            let lsb_b = vandq_u16(u16s_b, vdupq_n_u16(1));
            let neg_b = vsubq_u16(vdupq_n_u16(0), lsb_b);
            let delta_b = vreinterpretq_s16_u16(veorq_u16(vshrq_n_u16(u16s_b, 1), neg_b));

            // Both prefix sums BEFORE either carry extract — critical for ILP.
            let da = vaddq_s16(delta_a, vextq_s16(zero, delta_a, 7));
            let da = vaddq_s16(da, vextq_s16(zero, da, 6));
            let da = vaddq_s16(da, vextq_s16(zero, da, 4));
            let db = vaddq_s16(delta_b, vextq_s16(zero, delta_b, 7));
            let db = vaddq_s16(db, vextq_s16(zero, db, 6));
            let db = vaddq_s16(db, vextq_s16(zero, db, 4));

            let ra = vaddq_s16(da, vdupq_n_s16(acc_a));
            let rb = vaddq_s16(db, vdupq_n_s16(acc_b));
            // SAFETY: ctrl_pos < ctrl_half, so ctrl_pos*8 < n_half; within reserved capacity.
            vst1q_s16(out_ptr_a.add(ctrl_pos * 8), ra);
            // SAFETY: out_ptr_b = base + n_half; ctrl_pos*8 < n - n_half; within capacity.
            vst1q_s16(out_ptr_b.add(ctrl_pos * 8), rb);
            acc_a = vgetq_lane_s16(ra, 7);
            acc_b = vgetq_lane_s16(rb, 7);
        }

        data_pos_a += 8 + cb_a.count_ones() as usize;
        data_pos_b += 8 + cb_b.count_ones() as usize;
        ctrl_pos += 1;
    }

    let ctrl_pos_break = ctrl_pos;

    // Finish chain A: remaining ctrl[ctrl_pos_break..ctrl_half].
    {
        let mut cp = ctrl_pos_break;
        let mut dpa = data_pos_a;

        // Fast sub-path while ≥16 chain-A data bytes remain.
        while cp < ctrl_half && dpa + 16 <= mid_data_offset {
            let cb = ctrl[cp];
            unsafe {
                // SAFETY: dpa + 16 <= mid_data_offset <= data_bytes.len(). TABLE[cb] is 16 bytes.
                let shuf = vld1q_u8(TABLE[cb as usize].as_ptr());
                let chunk = vld1q_u8(data_bytes.as_ptr().add(dpa));
                let u8s = vqtbl1q_u8(chunk, shuf);
                let u16s = vreinterpretq_u16_u8(u8s);
                let lsb = vandq_u16(u16s, vdupq_n_u16(1));
                let neg = vsubq_u16(vdupq_n_u16(0), lsb);
                let delta = vreinterpretq_s16_u16(veorq_u16(vshrq_n_u16(u16s, 1), neg));
                let delta = vaddq_s16(delta, vextq_s16(zero, delta, 7));
                let delta = vaddq_s16(delta, vextq_s16(zero, delta, 6));
                let delta = vaddq_s16(delta, vextq_s16(zero, delta, 4));
                let result = vaddq_s16(delta, vdupq_n_s16(acc_a));
                // SAFETY: cp < ctrl_half, so cp*8 < n_half; within reserved capacity.
                vst1q_s16(out_ptr_a.add(cp * 8), result);
                acc_a = vgetq_lane_s16(result, 7);
            }
            dpa += 8 + cb.count_ones() as usize;
            cp += 1;
        }

        // Padded tail for chain A (guard fired: <16 bytes remain).
        if cp < ctrl_half {
            let mut padded = [0u8; 32];
            let rem = mid_data_offset - dpa;
            padded[..rem].copy_from_slice(&data_bytes[dpa..mid_data_offset]);
            let mut ppos = 0usize;
            while cp < ctrl_half {
                let cb = ctrl[cp];
                unsafe {
                    // SAFETY: padded is 32 bytes; ppos ≤ rem ≤ 15; load [ppos, ppos+16) ⊆ [0,31) ⊆ [0,32).
                    let shuf = vld1q_u8(TABLE[cb as usize].as_ptr());
                    let chunk = vld1q_u8(padded.as_ptr().add(ppos));
                    let u8s = vqtbl1q_u8(chunk, shuf);
                    let u16s = vreinterpretq_u16_u8(u8s);
                    let lsb = vandq_u16(u16s, vdupq_n_u16(1));
                    let neg = vsubq_u16(vdupq_n_u16(0), lsb);
                    let delta = vreinterpretq_s16_u16(veorq_u16(vshrq_n_u16(u16s, 1), neg));
                    let delta = vaddq_s16(delta, vextq_s16(zero, delta, 7));
                    let delta = vaddq_s16(delta, vextq_s16(zero, delta, 6));
                    let delta = vaddq_s16(delta, vextq_s16(zero, delta, 4));
                    let result = vaddq_s16(delta, vdupq_n_s16(acc_a));
                    // SAFETY: cp < ctrl_half, so cp*8 < n_half; within reserved capacity.
                    vst1q_s16(out_ptr_a.add(cp * 8), result);
                    acc_a = vgetq_lane_s16(result, 7);
                }
                ppos += 8 + cb.count_ones() as usize;
                cp += 1;
            }
        }
    }
    // out[base..base+n_half] fully written by chain A.

    // Set Vec len: chain A (n_half) + chain B decoded in main loop (ctrl_pos_break * 8).
    // SAFETY: all elements in [base, base + n_half + ctrl_pos_break * 8) were written above.
    unsafe { out.set_len(base + n_half + ctrl_pos_break * 8) };

    // Finish chain B: remaining (n - n_half) - ctrl_pos_break * 8 elements.
    let ctrl_b_rest = &ctrl[ctrl_half + ctrl_pos_break..];
    let data_b_rest = &data_bytes[mid_data_offset + data_pos_b..];
    let n_b_rem = (n - n_half) - ctrl_pos_break * 8;
    // SAFETY: same NEON feature gate as this function.
    unsafe { decode_neon(ctrl_b_rest, data_b_rest, n_b_rem, acc_b, out) }
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
}
