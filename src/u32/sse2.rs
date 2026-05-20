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

use super::shuffle::{DATA_LEN, DATA_LEN_0124, TABLE, TABLE_0124};
use crate::error::DecodeError;

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

    // Scalar tail: remaining values, or the final iteration that lacked ≥16 data bytes.
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
