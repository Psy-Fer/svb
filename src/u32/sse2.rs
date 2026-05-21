// SSSE3 decode path for U32Classic.
//
// The file is named sse2 per project convention; the instruction actually used
// is PSHUFB (_mm_shuffle_epi8), which is SSSE3 (Penryn 2007+).
// At runtime, dispatch checks is_x86_feature_detected!("ssse3") before calling.

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{
    DATA_LEN, DATA_LEN_0124, ENCODE_TABLE_0124, ENCODE_TABLE_CLASSIC, TABLE, TABLE_0124,
};
use crate::error::DecodeError;

/// Encode `values` into U32Classic format using SSSE3 `PSHUFB`.
///
/// Processes 4 values per ctrl byte. The ctrl byte is computed using the signed
/// bias trick (unsigned comparison via i32 bias 0x80000000). `PSHUFB` packs the
/// variable-width data bytes in a single instruction. Stores 16 bytes per
/// iteration (overwriting into the +16-byte guard in the reserved capacity) and
/// advances the data pointer by the exact byte count for that ctrl byte.
/// Remaining values (n % 4) are handled by the scalar path.
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn encode_into_classic(values: &[u32], out: &mut Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return;
    }

    let ctrl_len = n.div_ceil(4);
    let ctrl_start = out.len();

    // Reserve ctrl bytes + worst-case data (4 bytes/value) + 16-byte SIMD overrun guard.
    out.reserve(ctrl_len + 4 * n + 16);
    // Zero-initialize ctrl bytes so the scalar tail can OR into them safely.
    out.resize(ctrl_start + ctrl_len, 0u8);

    let simd_n = (n / 4) * 4;
    let data_start = ctrl_start + ctrl_len;
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
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            // Shuffle input bytes into packed output order, then store 16 bytes.
            // SAFETY: ENCODE_TABLE_CLASSIC[ctrl] is 16 bytes; ctrl < 256 (u8).
            let enc_mask =
                _mm_loadu_si128(ENCODE_TABLE_CLASSIC[ctrl as usize].as_ptr() as *const __m128i);
            let packed = _mm_shuffle_epi8(v, enc_mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 4*n + 16 <= capacity.
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed);
        }

        data_pos += DATA_LEN[ctrl as usize] as usize;
        block += 1;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 4 remaining values.
    // The ctrl bytes are already allocated and zeroed; we only need to set bits and push data.
    for j in simd_n..n {
        let v = values[j];
        let (tag, count): (u8, usize) = if v <= 0xFF {
            (0, 1)
        } else if v <= 0xFFFF {
            (1, 2)
        } else if v <= 0xFF_FFFF {
            (2, 3)
        } else {
            (3, 4)
        };
        // SAFETY: ctrl_start + j/4 < ctrl_start + ctrl_len <= out.len().
        out[ctrl_start + j / 4] |= tag << ((j % 4) * 2);
        out.extend_from_slice(&v.to_le_bytes()[..count]);
    }
}

/// Encode `values` into U32Variant0124 format using SSSE3 `PSHUFB`.
///
/// Identical structure to `encode_into_classic` but uses the 0124 tag thresholds
/// (0 bytes for zero, 1/2/4 bytes for 1-byte/2-byte/4-byte values) and the
/// corresponding encode table and data-length table.
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn encode_into_0124(values: &[u32], out: &mut Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return;
    }

    let ctrl_len = n.div_ceil(4);
    let ctrl_start = out.len();

    // Reserve ctrl bytes + worst-case data (4 bytes/value) + 16-byte SIMD overrun guard.
    out.reserve(ctrl_len + 4 * n + 16);
    // Zero-initialize ctrl bytes so the scalar tail can OR into them safely.
    out.resize(ctrl_start + ctrl_len, 0u8);

    let simd_n = (n / 4) * 4;
    let data_start = ctrl_start + ctrl_len;
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
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            // SAFETY: ENCODE_TABLE_0124[ctrl] is 16 bytes; ctrl < 256 (u8).
            let enc_mask =
                _mm_loadu_si128(ENCODE_TABLE_0124[ctrl as usize].as_ptr() as *const __m128i);
            let packed = _mm_shuffle_epi8(v, enc_mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 4*n + 16 <= capacity.
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed);
        }

        data_pos += DATA_LEN_0124[ctrl as usize] as usize;
        block += 1;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 4 remaining values.
    // The ctrl bytes are already allocated and zeroed; we only need to set bits and push data.
    for j in simd_n..n {
        let v = values[j];
        let (tag, count): (u8, usize) = if v == 0 {
            (0, 0)
        } else if v <= 0xFF {
            (1, 1)
        } else if v <= 0xFFFF {
            (2, 2)
        } else {
            (3, 4)
        };
        // SAFETY: ctrl_start + j/4 < ctrl_start + ctrl_len <= out.len().
        out[ctrl_start + j / 4] |= tag << ((j % 4) * 2);
        if count > 0 {
            out.extend_from_slice(&v.to_le_bytes()[..count]);
        }
    }
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
pub(super) unsafe fn decode_into_classic(
    data: &[u8],
    n: usize,
    out: &mut Vec<u32>,
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
    let base = out.len();

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
            // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n, so
            // base + decoded + 4 <= base + n <= out.capacity().
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
        }

        data_pos += DATA_LEN[cb as usize] as usize;
        ctrl_pos += 1;
        decoded += 4;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    // Padded tail: guard fired (rem < 16) but complete groups of 4 may remain.
    // Copy the remaining data bytes into a zero-padded 32-byte buffer so every
    // 16-byte PSHUFB load is in-bounds (padded_pos ≤ rem−4 ≤ 11; [11,27) ⊆ [0,32)).
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            // SAFETY: padded is 32 bytes; padded_pos ≤ rem − DATA_LEN_min (≥4) ≤ 11;
            // load [padded_pos, padded_pos+16) ⊆ [0, 27) ⊆ [0, 32).
            let result = unsafe {
                let mask = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                _mm_shuffle_epi8(chunk, mask)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
                _mm_storeu_si128(out_ptr, result);
            }
            let consumed = DATA_LEN[cb as usize] as usize;
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe {
            out.set_len(base + decoded);
        }
    }

    // Scalar for n % 4 remainder (0–3 values).
    if decoded < n {
        super::scalar::decode_classic_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            n - decoded,
            out,
        )?;
    }

    Ok(())
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
pub(super) unsafe fn decode_into_0124(
    data: &[u8],
    n: usize,
    out: &mut Vec<u32>,
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
    let base = out.len();

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
            // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
        }

        data_pos += DATA_LEN_0124[cb as usize] as usize;
        ctrl_pos += 1;
        decoded += 4;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    // Padded tail: guard fired (rem < 16) but complete groups of 4 may remain.
    // For 0124, DATA_LEN can be 0, so padded_pos ≤ rem ≤ 15; [15,31) ⊆ [0,32).
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            // SAFETY: padded is 32 bytes; padded_pos ≤ rem ≤ 15;
            // load [padded_pos, padded_pos+16) ⊆ [0, 31) ⊆ [0, 32).
            let result = unsafe {
                let mask = _mm_loadu_si128(TABLE_0124[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                _mm_shuffle_epi8(chunk, mask)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
                _mm_storeu_si128(out_ptr, result);
            }
            let consumed = DATA_LEN_0124[cb as usize] as usize;
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe {
            out.set_len(base + decoded);
        }
    }

    // Scalar for n % 4 remainder (0–3 values).
    if decoded < n {
        super::scalar::decode_0124_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            n - decoded,
            out,
        )?;
    }

    Ok(())
}
