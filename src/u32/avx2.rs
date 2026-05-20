// AVX2 decode path for U32Classic.
//
// _mm256_shuffle_epi8 (VPSHUFB) operates as two independent 128-bit PSHUFB
// lanes. We process 2 ctrl bytes (8 u32 values) per iteration: c0 drives the
// lower lane, c1 the upper lane. Each lane's data is loaded independently
// from its start offset in the data stream.
//
// The scalar tail (< 8 remaining values, or final iteration with < 32 data
// bytes) is handled by the scalar path directly.

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{DATA_LEN, DATA_LEN_0124, ENCODE_TABLE_0124, ENCODE_TABLE_CLASSIC, TABLE, TABLE_0124};
use crate::error::DecodeError;

/// Encode `values` into U32Classic format using AVX2.
///
/// Processes 8 values (2 ctrl bytes) per iteration using two 128-bit PSHUFB
/// ops on the two halves of a 256-bit load. Ctrl bytes are computed with the
/// signed bias trick on 32-bit lanes. Remaining values (n % 8) are handled
/// by the scalar path.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    let simd_n = (n / 8) * 8;
    let data_start = ctrl_start + ctrl_len;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    // Bias and thresholds for Classic (same as SSSE3 path, but 256-bit).
    let bias = _mm256_set1_epi32(i32::MIN);
    let t1 = _mm256_set1_epi32(i32::MIN + 0xFF);
    let t2 = _mm256_set1_epi32(i32::MIN + 0xFFFF);
    let t3 = _mm256_set1_epi32(i32::MIN + 0xFF_FFFF);
    let zero256 = _mm256_setzero_si256();

    // PSHUFB mask to gather byte 0 of each 32-bit lane within a 128-bit half.
    let gather_lo = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        // Load 8 u32 values (32 bytes).
        let v = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; values slice bounds are valid.
            _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i)
        };

        // Compute per-lane tags for both 128-bit halves.
        let bv = _mm256_add_epi32(v, bias);
        let c1 = _mm256_cmpgt_epi32(bv, t1);
        let c2 = _mm256_cmpgt_epi32(bv, t2);
        let c3 = _mm256_cmpgt_epi32(bv, t3);
        let b1 = _mm256_sub_epi32(zero256, c1);
        let b2 = _mm256_sub_epi32(zero256, c2);
        let b3 = _mm256_sub_epi32(zero256, c3);
        let tag_vec = _mm256_add_epi32(_mm256_add_epi32(b1, b2), b3);

        // Extract the lower and upper 128-bit halves for tag packing.
        let tag_lo = _mm256_castsi256_si128(tag_vec);
        let tag_hi = _mm256_extracti128_si256(tag_vec, 1);

        let tag_bytes_lo = _mm_shuffle_epi8(tag_lo, gather_lo);
        let tags_lo = _mm_cvtsi128_si32(tag_bytes_lo) as u32;
        let c0 = ((tags_lo & 0x3)
            | ((tags_lo >> 6) & 0x0C)
            | ((tags_lo >> 12) & 0x30)
            | ((tags_lo >> 18) & 0xC0)) as u8;

        let tag_bytes_hi = _mm_shuffle_epi8(tag_hi, gather_lo);
        let tags_hi = _mm_cvtsi128_si32(tag_bytes_hi) as u32;
        let c1b = ((tags_hi & 0x3)
            | ((tags_hi >> 6) & 0x0C)
            | ((tags_hi >> 12) & 0x30)
            | ((tags_hi >> 18) & 0xC0)) as u8;

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len; block+1 also < ctrl_len.
            *base_ptr.add(ctrl_start + block) = c0;
            *base_ptr.add(ctrl_start + block + 1) = c1b;

            // Pack lower 4 values using PSHUFB on the lower 128-bit lane.
            let v_lo = _mm256_castsi256_si128(v);
            let enc_mask_lo =
                _mm_loadu_si128(ENCODE_TABLE_CLASSIC[c0 as usize].as_ptr() as *const __m128i);
            let packed_lo = _mm_shuffle_epi8(v_lo, enc_mask_lo);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed_lo);
            data_pos += DATA_LEN[c0 as usize] as usize;

            // Pack upper 4 values.
            let v_hi = _mm256_extracti128_si256(v, 1);
            let enc_mask_hi =
                _mm_loadu_si128(ENCODE_TABLE_CLASSIC[c1b as usize].as_ptr() as *const __m128i);
            let packed_hi = _mm_shuffle_epi8(v_hi, enc_mask_hi);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed_hi);
            data_pos += DATA_LEN[c1b as usize] as usize;
        }

        block += 2;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 8 remaining values.
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

/// Encode `values` into U32Variant0124 format using AVX2.
///
/// Identical structure to `encode_into_classic` but uses the 0124 tag thresholds
/// and encode table.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    let simd_n = (n / 8) * 8;
    let data_start = ctrl_start + ctrl_len;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    // Variant0124: tag = (v>0) + (v>0xFF) + (v>0xFFFF)
    let bias = _mm256_set1_epi32(i32::MIN);
    let t0 = _mm256_set1_epi32(i32::MIN);
    let t1 = _mm256_set1_epi32(i32::MIN + 0xFF);
    let t2 = _mm256_set1_epi32(i32::MIN + 0xFFFF);
    let zero256 = _mm256_setzero_si256();

    let gather_lo = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        let v = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; values slice bounds are valid.
            _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i)
        };

        let bv = _mm256_add_epi32(v, bias);
        let c0_mask = _mm256_cmpgt_epi32(bv, t0);
        let c1_mask = _mm256_cmpgt_epi32(bv, t1);
        let c2_mask = _mm256_cmpgt_epi32(bv, t2);
        let b0 = _mm256_sub_epi32(zero256, c0_mask);
        let b1 = _mm256_sub_epi32(zero256, c1_mask);
        let b2 = _mm256_sub_epi32(zero256, c2_mask);
        let tag_vec = _mm256_add_epi32(_mm256_add_epi32(b0, b1), b2);

        let tag_lo = _mm256_castsi256_si128(tag_vec);
        let tag_hi = _mm256_extracti128_si256(tag_vec, 1);

        let tag_bytes_lo = _mm_shuffle_epi8(tag_lo, gather_lo);
        let tags_lo = _mm_cvtsi128_si32(tag_bytes_lo) as u32;
        let c0 = ((tags_lo & 0x3)
            | ((tags_lo >> 6) & 0x0C)
            | ((tags_lo >> 12) & 0x30)
            | ((tags_lo >> 18) & 0xC0)) as u8;

        let tag_bytes_hi = _mm_shuffle_epi8(tag_hi, gather_lo);
        let tags_hi = _mm_cvtsi128_si32(tag_bytes_hi) as u32;
        let c1b = ((tags_hi & 0x3)
            | ((tags_hi >> 6) & 0x0C)
            | ((tags_hi >> 12) & 0x30)
            | ((tags_hi >> 18) & 0xC0)) as u8;

        unsafe {
            // SAFETY: ctrl_start + block and block+1 < ctrl_start + ctrl_len.
            *base_ptr.add(ctrl_start + block) = c0;
            *base_ptr.add(ctrl_start + block + 1) = c1b;

            let v_lo = _mm256_castsi256_si128(v);
            let enc_mask_lo =
                _mm_loadu_si128(ENCODE_TABLE_0124[c0 as usize].as_ptr() as *const __m128i);
            let packed_lo = _mm_shuffle_epi8(v_lo, enc_mask_lo);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed_lo);
            data_pos += DATA_LEN_0124[c0 as usize] as usize;

            let v_hi = _mm256_extracti128_si256(v, 1);
            let enc_mask_hi =
                _mm_loadu_si128(ENCODE_TABLE_0124[c1b as usize].as_ptr() as *const __m128i);
            let packed_hi = _mm_shuffle_epi8(v_hi, enc_mask_hi);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed_hi);
            data_pos += DATA_LEN_0124[c1b as usize] as usize;
        }

        block += 2;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 8 remaining values.
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

/// Decode `n` u32 values from a U32Classic-encoded buffer using AVX2 `VPSHUFB`.
///
/// Processes 8 values (2 ctrl bytes) per iteration.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    while decoded + 8 <= n {
        let c0 = ctrl[ctrl_pos];
        let c1 = ctrl[ctrl_pos + 1];
        let c0_bytes = DATA_LEN[c0 as usize] as usize;

        // Upper lane data starts c0_bytes after the lower lane. Worst case:
        // c0 and c1 both all-4-byte → c0_bytes = 16, c1_bytes = 16 → need 32.
        if data_pos + 32 > data_bytes.len() {
            break;
        }

        let result = unsafe {
            // SAFETY: TABLE indices are valid (c0, c1 are u8, so < 256).
            let mask_lo = _mm_loadu_si128(TABLE[c0 as usize].as_ptr() as *const __m128i);
            let mask_hi = _mm_loadu_si128(TABLE[c1 as usize].as_ptr() as *const __m128i);

            // SAFETY: data_pos + 32 <= data_bytes.len() checked above;
            // c0_bytes <= 16 so data_pos + c0_bytes + 16 <= data_pos + 32.
            let chunk_lo = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let chunk_hi =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos + c0_bytes) as *const __m128i);

            // Pack into 256-bit registers: lower 128 = c0 lane, upper 128 = c1 lane.
            let mask256 = _mm256_set_m128i(mask_hi, mask_lo);
            let data256 = _mm256_set_m128i(chunk_hi, chunk_lo);
            _mm256_shuffle_epi8(data256, mask256)
        };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 8 <= n.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m256i;
            _mm256_storeu_si256(out_ptr, result);
        }

        data_pos += c0_bytes + DATA_LEN[c1 as usize] as usize;
        ctrl_pos += 2;
        decoded += 8;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    // Scalar tail for any remaining values (< 8 remaining or < 32 data bytes).
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

/// Decode `n` u32 values from a U32Variant0124-encoded buffer using AVX2 `VPSHUFB`.
///
/// Identical structure to `decode_into_classic` but uses the 0124 shuffle and
/// data-length tables (tag widths 0/1/2/4 instead of 1/2/3/4).
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    while decoded + 8 <= n {
        let c0 = ctrl[ctrl_pos];
        let c1 = ctrl[ctrl_pos + 1];
        let c0_bytes = DATA_LEN_0124[c0 as usize] as usize;

        // Worst case: c0 and c1 both all-tag-3 → 16 + 16 = 32 bytes.
        if data_pos + 32 > data_bytes.len() {
            break;
        }

        let result = unsafe {
            // SAFETY: TABLE_0124 indices are valid (c0, c1 are u8 → < 256).
            let mask_lo = _mm_loadu_si128(TABLE_0124[c0 as usize].as_ptr() as *const __m128i);
            let mask_hi = _mm_loadu_si128(TABLE_0124[c1 as usize].as_ptr() as *const __m128i);

            // SAFETY: data_pos + 32 <= data_bytes.len() checked above;
            // c0_bytes <= 16 so data_pos + c0_bytes + 16 <= data_pos + 32.
            let chunk_lo = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let chunk_hi =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos + c0_bytes) as *const __m128i);

            let mask256 = _mm256_set_m128i(mask_hi, mask_lo);
            let data256 = _mm256_set_m128i(chunk_hi, chunk_lo);
            _mm256_shuffle_epi8(data256, mask256)
        };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 8 <= n.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m256i;
            _mm256_storeu_si256(out_ptr, result);
        }

        data_pos += c0_bytes + DATA_LEN_0124[c1 as usize] as usize;
        ctrl_pos += 2;
        decoded += 8;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

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
