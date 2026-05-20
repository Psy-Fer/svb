// SSSE3 decode paths for U64Coder1234 and U64Coder1248.
//
// The file is named sse2 per project convention; the instruction used is
// PSHUFB (_mm_shuffle_epi8), which is SSSE3 (Penryn 2007+).

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{DATA_LEN_1234, DATA_LEN_1248_PAIR, TABLE_1234, TABLE_1248_PAIR};
use crate::error::DecodeError;

/// Decode `n` u64 values from a U64Coder1234-encoded buffer using SSSE3 `PSHUFB`.
///
/// Processes 4 values per ctrl byte. PSHUFB expands the data bytes into 4 u32
/// slots; `_mm_unpacklo/hi_epi32` zero-extends them to u64. Falls back to the
/// scalar path when fewer than 16 data bytes remain.
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
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

    let zero = _mm_setzero_si128();

    while decoded + 4 <= n {
        let cb = ctrl[ctrl_pos];

        // Maximum data bytes for 4 values (all tag-3 = 4 bytes each) is 16.
        if data_pos + 16 > data_bytes.len() {
            break;
        }

        let u32s = unsafe {
            // SAFETY: TABLE_1234[cb] is 16 bytes; cb < 256 (u8).
            let mask = _mm_loadu_si128(TABLE_1234[cb as usize].as_ptr() as *const __m128i);
            // SAFETY: data_pos + 16 <= data_bytes.len() verified above.
            let chunk = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            _mm_shuffle_epi8(chunk, mask)
        };

        // Zero-extend 4 × u32 → 4 × u64 across two 128-bit registers.
        // _mm_unpacklo_epi32(a, zero) = [a[31:0], 0, a[63:32], 0] = [u64[0], u64[1]]
        // _mm_unpackhi_epi32(a, zero) = [a[95:64], 0, a[127:96], 0] = [u64[2], u64[3]]
        let lo = _mm_unpacklo_epi32(u32s, zero);
        let hi = _mm_unpackhi_epi32(u32s, zero);

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n, so
            // base + decoded + 4 <= base + n <= out.capacity().
            // Each __m128i is 16 bytes = 2 u64s; lo covers [0,1], hi covers [2,3].
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, lo);
            _mm_storeu_si128(out_ptr.add(1), hi);
        }

        data_pos += DATA_LEN_1234[cb as usize] as usize;
        ctrl_pos += 1;
        decoded += 4;
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

/// Decode `n` u64 values from a U64Coder1248-encoded buffer using SSSE3 `PSHUFB`.
///
/// Processes 4 values (2 pairs) per ctrl byte. The ctrl byte is split into a
/// low nibble (tags for values 0 and 1) and high nibble (tags for values 2 and
/// 3), each indexing the 16-entry pair table. Falls back to the scalar path
/// when fewer than 32 data bytes remain (the worst-case for one ctrl byte).
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
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

    while decoded + 4 <= n {
        let cb = ctrl[ctrl_pos];
        let lo_key = (cb & 0x0F) as usize;
        let hi_key = (cb >> 4) as usize;
        let lo_bytes = DATA_LEN_1248_PAIR[lo_key] as usize;

        // Worst case: lo pair is 8+8=16 bytes, hi pair is 8+8=16 bytes → 32 total.
        if data_pos + 32 > data_bytes.len() {
            break;
        }

        let (lo_pair, hi_pair) = unsafe {
            // SAFETY: TABLE_1248_PAIR indices are < 16 (4-bit keys).
            let mask_lo =
                _mm_loadu_si128(TABLE_1248_PAIR[lo_key].as_ptr() as *const __m128i);
            // SAFETY: data_pos + 32 <= data_bytes.len() checked above; lo_bytes <= 16.
            let chunk_lo =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let lo = _mm_shuffle_epi8(chunk_lo, mask_lo);

            let mask_hi =
                _mm_loadu_si128(TABLE_1248_PAIR[hi_key].as_ptr() as *const __m128i);
            // SAFETY: data_pos + lo_bytes + 16 <= data_pos + 16 + 16 <= data_pos + 32.
            let chunk_hi =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos + lo_bytes) as *const __m128i);
            let hi = _mm_shuffle_epi8(chunk_hi, mask_hi);

            (lo, hi)
        };

        unsafe {
            // SAFETY: decoded + 4 <= n; 4 u64s = 32 bytes; each __m128i covers 2 u64s.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, lo_pair);
            _mm_storeu_si128(out_ptr.add(1), hi_pair);
        }

        data_pos += lo_bytes + DATA_LEN_1248_PAIR[hi_key] as usize;
        ctrl_pos += 1;
        decoded += 4;
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
