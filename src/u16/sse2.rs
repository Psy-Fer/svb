// SSSE3 decode path for SVB16.
//
// The file is named sse2 per project convention; the instruction actually used
// is PSHUFB (_mm_shuffle_epi8), which is SSSE3 (Penryn 2007+).
// At runtime, dispatch checks is_x86_feature_detected!("ssse3") before calling.

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::TABLE;
use crate::error::DecodeError;

/// Decode `n` u16 values from an SVB16-encoded buffer using SSSE3 `PSHUFB`.
///
/// Processes 8 values per ctrl byte. Falls back to the scalar path for any
/// trailing values when fewer than 16 data bytes remain (preventing an
/// out-of-bounds read on the unaligned 128-bit load).
///
/// # Safety
/// The executing CPU must support SSSE3.
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn decode_into(
    data: &[u8],
    n: usize,
    out: &mut Vec<u16>,
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
    let base = out.len();

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;

    while decoded + 8 <= n {
        let cb = ctrl[ctrl_pos];
        let bytes_consumed = 8 + cb.count_ones() as usize;

        // Guard against an out-of-bounds unaligned load at the end of the buffer.
        // The maximum data bytes for 8 values is 16 (all 2-byte). If fewer than
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
            // SAFETY: out.reserve(n) ensures capacity; decoded + 8 <= n, so
            // base + decoded + 8 <= base + n <= out.capacity().
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 8;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    // Scalar tail: remaining values, or the final iteration that lacked ≥16 data bytes.
    if decoded < n {
        super::scalar::decode_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            n - decoded,
            out,
        )?;
    }

    Ok(())
}
