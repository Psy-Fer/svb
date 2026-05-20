// AVX2 decode paths for U64Coder1234 and U64Coder1248.
//
// 1234: _mm256_cvtepu32_epi64 zero-extends each 4-u32 PSHUFB result to 4 u64.
//       Two ctrl bytes per iteration → 8 u64 per iteration.
//
// 1248: PSHUFB operates on pairs (2 u64 per register). Two ctrl bytes per
//       iteration → 4 PSHUFB ops → two 256-bit stores → 8 u64 per iteration.

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{DATA_LEN_1234, DATA_LEN_1248_PAIR, TABLE_1234, TABLE_1248_PAIR};
use crate::error::DecodeError;

/// Decode `n` u64 values from a U64Coder1234-encoded buffer using AVX2.
///
/// Processes 8 values (2 ctrl bytes) per iteration. Each PSHUFB result (4 u32)
/// is zero-extended to 4 u64 via `_mm256_cvtepu32_epi64`.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn decode_into_1234(
    data: &[u8],
    n: usize,
    out: &mut Vec<u64>,
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
        let c0_bytes = DATA_LEN_1234[c0 as usize] as usize;

        // Two ctrl bytes × max 16 data bytes each = 32 bytes worst case.
        if data_pos + 32 > data_bytes.len() {
            break;
        }

        let (u64s_c0, u64s_c1) = unsafe {
            // SAFETY: TABLE_1234 indices valid (u8 → < 256).
            let mask_c0 = _mm_loadu_si128(TABLE_1234[c0 as usize].as_ptr() as *const __m128i);
            let mask_c1 = _mm_loadu_si128(TABLE_1234[c1 as usize].as_ptr() as *const __m128i);
            // SAFETY: data_pos + 32 <= data_bytes.len() checked above;
            // c0_bytes <= 16, so data_pos + c0_bytes + 16 <= data_pos + 32.
            let chunk_c0 =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let chunk_c1 =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos + c0_bytes) as *const __m128i);
            let u32s_c0 = _mm_shuffle_epi8(chunk_c0, mask_c0);
            let u32s_c1 = _mm_shuffle_epi8(chunk_c1, mask_c1);
            // _mm256_cvtepu32_epi64: 4 u32 → 4 u64 (zero-extend), requires AVX2.
            (_mm256_cvtepu32_epi64(u32s_c0), _mm256_cvtepu32_epi64(u32s_c1))
        };

        unsafe {
            // SAFETY: decoded + 8 <= n; 8 u64s = 64 bytes = 2 × __m256i.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m256i;
            _mm256_storeu_si256(out_ptr, u64s_c0);
            _mm256_storeu_si256(out_ptr.add(1), u64s_c1);
        }

        data_pos += c0_bytes + DATA_LEN_1234[c1 as usize] as usize;
        ctrl_pos += 2;
        decoded += 8;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    if decoded < n {
        super::scalar::decode_1234_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            n - decoded,
            out,
        )?;
    }

    Ok(())
}

/// Decode `n` u64 values from a U64Coder1248-encoded buffer using AVX2.
///
/// Processes 8 values (2 ctrl bytes) per iteration. Each ctrl byte is split
/// into low and high nibbles, each indexing the 16-entry pair table (2 u64 per
/// PSHUFB). Two ctrl bytes → 4 PSHUFB ops → two `_mm256_set_m128i` results →
/// two 256-bit stores.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn decode_into_1248(
    data: &[u8],
    n: usize,
    out: &mut Vec<u64>,
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

        let lo_key0 = (c0 & 0x0F) as usize;
        let hi_key0 = (c0 >> 4) as usize;
        let lo_key1 = (c1 & 0x0F) as usize;
        let hi_key1 = (c1 >> 4) as usize;

        let lo_bytes0 = DATA_LEN_1248_PAIR[lo_key0] as usize;
        let hi_bytes0 = DATA_LEN_1248_PAIR[hi_key0] as usize;
        let lo_bytes1 = DATA_LEN_1248_PAIR[lo_key1] as usize;

        // Max per ctrl byte: 8+8+8+8 = 32 bytes. Two ctrl bytes: 64 bytes worst case.
        // Offsets: hi0 at +lo_bytes0 (≤16), lo1 at +(lo+hi)bytes0 (≤32),
        //          hi1 at +lo_bytes0+hi_bytes0+lo_bytes1 (≤48).
        // hi1 load ends at offset ≤48+16=64.
        if data_pos + 64 > data_bytes.len() {
            break;
        }

        let off_hi0 = data_pos + lo_bytes0;
        let off_lo1 = off_hi0 + hi_bytes0;
        let off_hi1 = off_lo1 + lo_bytes1;

        let (result0, result1) = unsafe {
            // SAFETY: all offsets + 16 <= data_pos + 64 <= data_bytes.len().
            let mask_lo0 =
                _mm_loadu_si128(TABLE_1248_PAIR[lo_key0].as_ptr() as *const __m128i);
            let chunk_lo0 =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let lo0 = _mm_shuffle_epi8(chunk_lo0, mask_lo0);

            let mask_hi0 =
                _mm_loadu_si128(TABLE_1248_PAIR[hi_key0].as_ptr() as *const __m128i);
            let chunk_hi0 =
                _mm_loadu_si128(data_bytes.as_ptr().add(off_hi0) as *const __m128i);
            let hi0 = _mm_shuffle_epi8(chunk_hi0, mask_hi0);

            let mask_lo1 =
                _mm_loadu_si128(TABLE_1248_PAIR[lo_key1].as_ptr() as *const __m128i);
            let chunk_lo1 =
                _mm_loadu_si128(data_bytes.as_ptr().add(off_lo1) as *const __m128i);
            let lo1 = _mm_shuffle_epi8(chunk_lo1, mask_lo1);

            let mask_hi1 =
                _mm_loadu_si128(TABLE_1248_PAIR[hi_key1].as_ptr() as *const __m128i);
            let chunk_hi1 =
                _mm_loadu_si128(data_bytes.as_ptr().add(off_hi1) as *const __m128i);
            let hi1 = _mm_shuffle_epi8(chunk_hi1, mask_hi1);

            // set_m128i(hi, lo): lower 128 bits = lo, upper 128 bits = hi.
            (_mm256_set_m128i(hi0, lo0), _mm256_set_m128i(hi1, lo1))
        };

        unsafe {
            // SAFETY: decoded + 8 <= n; 8 u64s = 64 bytes = 2 × __m256i.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m256i;
            _mm256_storeu_si256(out_ptr, result0);
            _mm256_storeu_si256(out_ptr.add(1), result1);
        }

        data_pos += lo_bytes0 + hi_bytes0 + lo_bytes1 + DATA_LEN_1248_PAIR[hi_key1] as usize;
        ctrl_pos += 2;
        decoded += 8;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    if decoded < n {
        super::scalar::decode_1248_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            n - decoded,
            out,
        )?;
    }

    Ok(())
}
