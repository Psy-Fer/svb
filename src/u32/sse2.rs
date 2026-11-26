// SSSE3 decode path for U32Classic.
//
// The file is named sse2 per project convention; the instruction actually used
// is PSHUFB (_mm_shuffle_epi8), which is SSSE3 (Penryn 2007+).
// At runtime, dispatch checks is_x86_feature_detected!("ssse3") before calling.

use core::arch::x86_64::*;

use super::shuffle::{
    DATA_LEN, DATA_LEN_0124, ENCODE_TABLE_0124, ENCODE_TABLE_CLASSIC, TABLE, TABLE_0124,
};
use crate::error::DecodeError;

/// Encode `values` into U32Classic format using SSSE3 `PSHUFB`.
///
/// Processes 4 values per ctrl byte. The ctrl byte is computed using the signed
/// bias trick (unsigned comparison via i32 bias 0x80000000). `PSHUFB` packs the
/// variable-width data bytes in a single instruction. Stores 16 bytes per
/// iteration and advances the data pointer by the exact byte count for that
/// ctrl byte, so a store can write past the data encoded so far but never past
/// `ctrl_len + 4 * n`. Remaining values (n % 4) are handled by the scalar path.
///
/// # Safety
/// The executing CPU must support SSSE3. `out.len()` must be at least
/// `ctrl_len + 4 * values.len()`.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn encode_to_slice_classic(values: &[u32], out: &mut [u8]) -> usize {
    let n = values.len();
    let ctrl_len = n.div_ceil(4);
    debug_assert!(out.len() >= ctrl_len + 4 * n);
    // Zero the ctrl bytes so the scalar tail can OR into them.
    out[..ctrl_len].fill(0);

    let simd_n = (n / 4) * 4;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    // Bias constant: converts unsigned u32 to signed i32 range for comparisons.
    let bias = _mm_set1_epi32(i32::MIN);
    // Biased thresholds for Classic widths 1/2/3/4:
    // tag0: v <= 0xFF → width 1; tag1: v <= 0xFFFF → width 2;
    // tag2: v <= 0xFFFFFF → width 3; tag3: otherwise → width 4
    let t1 = _mm_set1_epi32(i32::MIN + 0xFF);
    let t2 = _mm_set1_epi32(i32::MIN + 0xFFFF);
    let t3 = _mm_set1_epi32(i32::MIN + 0xFF_FFFF);
    let zero = _mm_setzero_si128();

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        let v = unsafe {
            // SAFETY: i + 4 <= simd_n <= n; values slice bounds are valid.
            _mm_loadu_si128(values.as_ptr().add(i) as *const __m128i)
        };

        // Compute per-lane tags (0..3) using signed comparison with bias.
        let bv = _mm_add_epi32(v, bias);
        // c1[lane] = 0xFFFFFFFF if v[lane] > 0xFF, else 0
        let c1 = _mm_cmpgt_epi32(bv, t1);
        let c2 = _mm_cmpgt_epi32(bv, t2);
        let c3 = _mm_cmpgt_epi32(bv, t3);
        // Convert mask 0xFFFFFFFF → 1 via negation (-(-1) = 1, -(0) = 0).
        let b1 = _mm_sub_epi32(zero, c1);
        let b2 = _mm_sub_epi32(zero, c2);
        let b3 = _mm_sub_epi32(zero, c3);
        let tag_vec = _mm_add_epi32(_mm_add_epi32(b1, b2), b3); // 0,1,2,3 per lane

        // Pack 4 tag values (each in byte 0 of a 32-bit lane) into a ctrl byte.
        // PSHUFB: gather byte 0 of each 32-bit lane → positions 0,1,2,3.
        let tag_bytes = _mm_shuffle_epi8(
            tag_vec,
            _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0),
        );
        // tag_bytes[0..4] = [tag0, tag1, tag2, tag3], rest zero.
        // Extract as u32: bits 0..7=tag0, 8..15=tag1, 16..23=tag2, 24..31=tag3.
        let tags = _mm_cvtsi128_si32(tag_bytes) as u32;
        // Bit-pack to ctrl byte: tag_i occupies bits 2*i .. 2*i+1.
        let ctrl =
            ((tags & 0x3) | ((tags >> 6) & 0x0C) | ((tags >> 12) & 0x30) | ((tags >> 18) & 0xC0))
                as u8;

        unsafe {
            // SAFETY: block < ctrl_len <= out.len().
            *base_ptr.add(block) = ctrl;

            // Shuffle input bytes into packed output order, then store 16 bytes.
            // SAFETY: ENCODE_TABLE_CLASSIC[ctrl] is 16 bytes; ctrl < 256 (u8).
            let enc_mask =
                _mm_loadu_si128(ENCODE_TABLE_CLASSIC[ctrl as usize].as_ptr() as *const __m128i);
            let packed = _mm_shuffle_epi8(v, enc_mask);
            // SAFETY: values before this store took at most 4 bytes each and the
            // store covers 4 values of at most 4 bytes, so it ends at or before
            // ctrl_len + 4 * n <= out.len().
            _mm_storeu_si128(base_ptr.add(ctrl_len + data_pos) as *mut __m128i, packed);
        }

        data_pos += DATA_LEN[ctrl as usize] as usize;
        block += 1;
    }

    super::scalar::encode_tail_classic(values, simd_n, out, ctrl_len, data_pos)
}

/// Encode `values` into U32Variant0124 format using SSSE3 `PSHUFB`.
///
/// Identical structure to `encode_into_classic` but uses the 0124 tag thresholds
/// (0 bytes for zero, 1/2/4 bytes for 1-byte/2-byte/4-byte values) and the
/// corresponding encode table and data-length table.
///
/// # Safety
/// The executing CPU must support SSSE3. `out.len()` must be at least
/// `ctrl_len + 4 * values.len()`.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn encode_to_slice_0124(values: &[u32], out: &mut [u8]) -> usize {
    let n = values.len();
    let ctrl_len = n.div_ceil(4);
    debug_assert!(out.len() >= ctrl_len + 4 * n);
    // Zero the ctrl bytes so the scalar tail can OR into them.
    out[..ctrl_len].fill(0);

    let simd_n = (n / 4) * 4;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    // Bias constant for unsigned comparisons.
    let bias = _mm_set1_epi32(i32::MIN);
    // Variant0124: tag = (v>0) + (v>0xFF) + (v>0xFFFF)
    // Thresholds (biased): biased(0)=i32::MIN, biased(0xFF)=i32::MIN+0xFF, etc.
    // v > 0 (unsigned) ↔ biased_v > i32::MIN (i.e., biased_v > 0x80000000)
    let t0 = _mm_set1_epi32(i32::MIN); // threshold for v > 0
    let t1 = _mm_set1_epi32(i32::MIN + 0xFF); // threshold for v > 0xFF
    let t2 = _mm_set1_epi32(i32::MIN + 0xFFFF); // threshold for v > 0xFFFF
    let zero = _mm_setzero_si128();

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        let v = unsafe {
            // SAFETY: i + 4 <= simd_n <= n; values slice bounds are valid.
            _mm_loadu_si128(values.as_ptr().add(i) as *const __m128i)
        };

        let bv = _mm_add_epi32(v, bias);
        let c0 = _mm_cmpgt_epi32(bv, t0); // lane != 0 (v > 0 unsigned)
        let c1 = _mm_cmpgt_epi32(bv, t1); // v > 0xFF
        let c2 = _mm_cmpgt_epi32(bv, t2); // v > 0xFFFF
        let b0 = _mm_sub_epi32(zero, c0);
        let b1 = _mm_sub_epi32(zero, c1);
        let b2 = _mm_sub_epi32(zero, c2);
        let tag_vec = _mm_add_epi32(_mm_add_epi32(b0, b1), b2);

        let tag_bytes = _mm_shuffle_epi8(
            tag_vec,
            _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0),
        );
        let tags = _mm_cvtsi128_si32(tag_bytes) as u32;
        let ctrl =
            ((tags & 0x3) | ((tags >> 6) & 0x0C) | ((tags >> 12) & 0x30) | ((tags >> 18) & 0xC0))
                as u8;

        unsafe {
            // SAFETY: block < ctrl_len <= out.len().
            *base_ptr.add(block) = ctrl;

            // SAFETY: ENCODE_TABLE_0124[ctrl] is 16 bytes; ctrl < 256 (u8).
            let enc_mask =
                _mm_loadu_si128(ENCODE_TABLE_0124[ctrl as usize].as_ptr() as *const __m128i);
            let packed = _mm_shuffle_epi8(v, enc_mask);
            // SAFETY: values before this store took at most 4 bytes each and the
            // store covers 4 values of at most 4 bytes, so it ends at or before
            // ctrl_len + 4 * n <= out.len().
            _mm_storeu_si128(base_ptr.add(ctrl_len + data_pos) as *mut __m128i, packed);
        }

        data_pos += DATA_LEN_0124[ctrl as usize] as usize;
        block += 1;
    }

    super::scalar::encode_tail_0124(values, simd_n, out, ctrl_len, data_pos)
}

/// Decode `n` u32 values from a U32Classic-encoded buffer using SSSE3 `PSHUFB`.
///
/// Processes 4 values per ctrl byte. Falls back to the scalar path for any
/// trailing values when fewer than 16 data bytes remain (preventing an
/// out-of-bounds read on the unaligned 128-bit load).
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn decode_to_slice_classic(
    data: &[u8],
    out: &mut [u32],
) -> Result<usize, DecodeError> {
    let n = out.len();
    if n == 0 {
        return Ok(0);
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

    let out_ptr = out.as_mut_ptr();

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;

    while decoded + 4 <= n {
        let cb = ctrl[ctrl_pos];

        // Guard against an out-of-bounds unaligned load at the end of the buffer.
        // The maximum data bytes for 4 values is 16 (all 4-byte). If fewer than
        // 16 bytes remain we fall through to the scalar tail.
        if data_pos + 16 > data_bytes.len() {
            break;
        }

        let result = unsafe {
            // SAFETY: TABLE[cb] is 16 bytes; cb < 256 (u8).
            let mask = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
            // SAFETY: data_pos + 16 <= data_bytes.len() verified above.
            let chunk = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            _mm_shuffle_epi8(chunk, mask)
        };

        unsafe {
            // SAFETY: decoded + 4 <= n = out.len().
            let out_ptr = out_ptr.add(decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
        }

        data_pos += DATA_LEN[cb as usize] as usize;
        ctrl_pos += 1;
        decoded += 4;
    }

    // Padded tail: for well-formed input, guard fires (rem < 16) with complete
    // groups of 4 remaining, fitting a zero-padded 32-byte buffer. `rem` and
    // each iteration's `consumed` are still re-validated below, since a
    // truncated/corrupted `data` (mismatched against the declared `n`) can't
    // be trusted to satisfy that bound.
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated { index: decoded });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let consumed = DATA_LEN[cb as usize] as usize;
            if padded_pos + consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated { index: decoded });
            }
            // SAFETY: padded_pos + 16 <= padded.len() checked above.
            let result = unsafe {
                let mask = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                _mm_shuffle_epi8(chunk, mask)
            };
            unsafe {
                // SAFETY: decoded + 4 <= n = out.len().
                let out_ptr = out_ptr.add(decoded) as *mut __m128i;
                _mm_storeu_si128(out_ptr, result);
            }
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
    }

    // Scalar for n % 4 remainder (0–3 values).
    if decoded < n {
        data_pos += super::scalar::decode_classic_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            &mut out[decoded..],
            decoded,
        )?;
    }

    Ok(ctrl_len + data_pos)
}

/// Decode `n` u32 values from a U32Variant0124-encoded buffer using SSSE3 `PSHUFB`.
///
/// Identical structure to `decode_into_classic` but uses the 0124 shuffle and
/// data-length tables (tag widths 0/1/2/4 instead of 1/2/3/4).
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn decode_to_slice_0124(
    data: &[u8],
    out: &mut [u32],
) -> Result<usize, DecodeError> {
    let n = out.len();
    if n == 0 {
        return Ok(0);
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

    let out_ptr = out.as_mut_ptr();

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;

    while decoded + 4 <= n {
        let cb = ctrl[ctrl_pos];

        // Maximum data bytes for 4 values is 16 (all tag-3 → 4 bytes each).
        if data_pos + 16 > data_bytes.len() {
            break;
        }

        let result = unsafe {
            // SAFETY: TABLE_0124[cb] is 16 bytes; cb < 256 (u8).
            let mask = _mm_loadu_si128(TABLE_0124[cb as usize].as_ptr() as *const __m128i);
            // SAFETY: data_pos + 16 <= data_bytes.len() verified above.
            let chunk = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            _mm_shuffle_epi8(chunk, mask)
        };

        unsafe {
            // SAFETY: decoded + 4 <= n = out.len().
            let out_ptr = out_ptr.add(decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
        }

        data_pos += DATA_LEN_0124[cb as usize] as usize;
        ctrl_pos += 1;
        decoded += 4;
    }

    // Padded tail: for well-formed input, guard fires (rem < 16) with complete
    // groups of 4 remaining (DATA_LEN can be 0 for 0124), fitting a 32-byte
    // zero-padded buffer. `rem` and each iteration's `consumed` are still
    // re-validated below, since a truncated/corrupted `data` (mismatched
    // against the declared `n`) can't be trusted to satisfy that bound.
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated { index: decoded });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let consumed = DATA_LEN_0124[cb as usize] as usize;
            if padded_pos + consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated { index: decoded });
            }
            // SAFETY: padded_pos + 16 <= padded.len() checked above.
            let result = unsafe {
                let mask = _mm_loadu_si128(TABLE_0124[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                _mm_shuffle_epi8(chunk, mask)
            };
            unsafe {
                // SAFETY: decoded + 4 <= n = out.len().
                let out_ptr = out_ptr.add(decoded) as *mut __m128i;
                _mm_storeu_si128(out_ptr, result);
            }
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
    }

    // Scalar for n % 4 remainder (0–3 values).
    if decoded < n {
        data_pos += super::scalar::decode_0124_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            &mut out[decoded..],
            decoded,
        )?;
    }

    Ok(ctrl_len + data_pos)
}
