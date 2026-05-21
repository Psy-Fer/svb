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

use super::shuffle::{ENCODE_TABLE, TABLE};
use crate::error::DecodeError;

/// Encode `values` into SVB16 format using NEON `vqtbl1q_u8`.
///
/// Processes 8 values per iteration. The ctrl byte is computed by shifting each
/// u16 right by 8, comparing with zero, narrowing to u8, weighting each lane
/// with a power-of-two vector, and summing with `vaddv_u8`. `vqtbl1q_u8` with
/// `ENCODE_TABLE[ctrl]` packs the data bytes. Remaining values (n % 8) are
/// handled by the scalar path.
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn encode_into(values: &[u16], out: &mut Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return;
    }

    let ctrl_len = n.div_ceil(8);
    let ctrl_start = out.len();

    // Reserve ctrl bytes + worst-case data (2 bytes/value) + 16-byte SIMD overrun guard.
    out.reserve(ctrl_len + 2 * n + 16);
    // Zero-initialize ctrl bytes so the scalar tail can OR into them safely.
    out.resize(ctrl_start + ctrl_len, 0u8);

    let simd_n = (n / 8) * 8;
    let data_start = ctrl_start + ctrl_len;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    // Weights for extracting a 1-bit-per-lane bitmask: each 0xFF lane ANDed with
    // its weight (1, 2, 4, … 128) gives that lane's contribution to the ctrl byte.
    let weights = unsafe { vld1_u8([1u8, 2, 4, 8, 16, 32, 64, 128].as_ptr()) };

    let mut block = 0usize;
    while block * 8 < simd_n {
        let i = block * 8;

        let v = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; values slice bounds are valid.
            vld1q_u16(values.as_ptr().add(i))
        };

        // Compute ctrl byte: bit k = 1 iff value k needs 2 bytes (high byte != 0).
        let hi = vshrq_n_u16(v, 8);                          // high byte in low position
        let nonzero = vcgtq_u16(hi, vdupq_n_u16(0));         // 0xFFFF or 0x0000 per lane
        let flags8 = vmovn_u16(nonzero);                      // uint8x8_t: 0xFF or 0x00
        let masked = vand_u8(flags8, weights);                 // 0 or the power-of-two weight
        let ctrl = vaddv_u8(masked);                          // horizontal sum = ctrl byte

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            // Reinterpret u16 values as bytes and shuffle into packed output order.
            let v_bytes = vreinterpretq_u8_u16(v);
            // SAFETY: ENCODE_TABLE[ctrl] is 16 bytes; ctrl < 256 (u8).
            let mask = vld1q_u8(ENCODE_TABLE[ctrl as usize].as_ptr());
            let packed = vqtbl1q_u8(v_bytes, mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 2*n + 16 <= capacity.
            vst1q_u8(base_ptr.add(data_start + data_pos), packed);
        }

        data_pos += 8 + ctrl.count_ones() as usize;
        block += 1;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 8 remaining values.
    for j in simd_n..n {
        let v = values[j];
        if v <= 0xFF {
            out.push(v as u8);
        } else {
            // SAFETY: ctrl_start + j/8 < ctrl_start + ctrl_len <= out.len().
            out[ctrl_start + j / 8] |= 1u8 << (j % 8);
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

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

    // Padded tail: guard fired (rem < 16) but complete groups of 8 may remain.
    // bytes_consumed ∈ [8,16]; padded_pos ≤ rem−8 ≤ 7; load [7,23) ⊆ [0,32). ✓
    if decoded + 8 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 8 <= n {
            let cb = ctrl[ctrl_pos];
            let result = unsafe {
                // SAFETY: padded is 32 bytes; padded_pos ≤ rem−8 ≤ 7;
                // load [padded_pos, padded_pos+16) ⊆ [0, 23) ⊆ [0, 32).
                let mask = vld1q_u8(TABLE[cb as usize].as_ptr());
                let chunk = vld1q_u8(padded.as_ptr().add(padded_pos));
                vqtbl1q_u8(chunk, mask)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 8 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
                vst1q_u8(out_ptr, result);
            }
            let consumed = 8 + cb.count_ones() as usize;
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 8;
        }
        unsafe { out.set_len(base + decoded); }
    }

    // Scalar for n % 8 remainder (0–7 values).
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
