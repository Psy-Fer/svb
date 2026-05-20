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

use super::shuffle::{DATA_LEN, TABLE};
use crate::error::DecodeError;

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
