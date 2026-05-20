// NEON decode path for SVB16 (AArch64).
//
// vqtbl1q_u8 is the AArch64 equivalent of SSSE3 PSHUFB: it zeroes the output
// byte when the index byte is >= 16. Our table uses 0x80 for zero-fill slots,
// which satisfies both conditions, so the same shuffle table works unchanged.
// NEON is mandatory on AArch64, so no runtime feature check is needed.

use core::arch::aarch64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::TABLE;
use crate::error::DecodeError;

/// Decode `n` u16 values from an SVB16-encoded buffer using NEON `vqtbl1q_u8`.
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
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

        if data_pos + 16 > data_bytes.len() {
            break;
        }

        let result = unsafe {
            // SAFETY: TABLE[cb] is 16 bytes; cb < 256.
            let mask = vld1q_u8(TABLE[cb as usize].as_ptr());
            // SAFETY: data_pos + 16 <= data_bytes.len() verified above.
            let chunk = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            // vqtbl1q_u8: result[i] = chunk[mask[i]] if mask[i] < 16, else 0.
            // 0x80 >= 16 → zero fill, matching PSHUFB semantics.
            vqtbl1q_u8(chunk, mask)
        };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 8 <= n.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
            vst1q_u8(out_ptr, result);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 8;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

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
