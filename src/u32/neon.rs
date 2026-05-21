// NEON decode and encode paths for U32Classic and U32Variant0124 (AArch64).
//
// vqtbl1q_u8 is the AArch64 equivalent of SSSE3 PSHUFB: it zeroes the output
// byte when the index byte is >= 16. Our decode table uses 0x80 for zero-fill
// slots, satisfying both conditions.
//
// For encode, NEON provides vcgtq_u32 for UNSIGNED 32-bit comparison directly
// (no bias trick needed, unlike the x86 path). The ctrl byte is assembled via
// narrowing shifts and a weighted horizontal sum.

use core::arch::aarch64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{DATA_LEN, DATA_LEN_0124, ENCODE_TABLE_0124, ENCODE_TABLE_CLASSIC, TABLE, TABLE_0124};
use crate::error::DecodeError;

/// Encode `values` into U32Classic format using NEON `vqtbl1q_u8`.
///
/// Processes 4 values per ctrl byte. Tags are computed with `vcgtq_u32`
/// (unsigned comparison, no bias needed). `vqtbl1q_u8` packs the variable-width
/// data bytes. Stores 16 bytes per iteration (overwriting into the +16-byte guard
/// in the reserved capacity). Remaining values (n % 4) are handled by the scalar
/// path.
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
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

    // Weights for assembling the ctrl byte from 4 tags (each 0..3 in bits 0:1).
    // Lane k contributes tag_k << (2*k) to the ctrl byte.
    // We use: ctrl = tag0*1 + tag1*4 + tag2*16 + tag3*64
    // Pack weights as u8 (each tag fits in u8 after narrowing).
    let weights = unsafe { vld1_u8([1u8, 4, 16, 64, 0, 0, 0, 0].as_ptr()) };

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        let v = unsafe {
            // SAFETY: i + 4 <= simd_n <= n; values slice bounds are valid.
            vld1q_u32(values.as_ptr().add(i))
        };

        // Compute per-lane tags: tag = (v>0xFF) + (v>0xFFFF) + (v>0xFFFFFF)
        // vcgtq_u32 returns 0xFFFFFFFF or 0 per lane.
        let gt255 = vcgtq_u32(v, vdupq_n_u32(0xFF));
        let gt65535 = vcgtq_u32(v, vdupq_n_u32(0xFFFF));
        let gt16m = vcgtq_u32(v, vdupq_n_u32(0xFF_FFFF));
        // Convert 0xFFFFFFFF → 1 by logical shift right 31.
        let b1 = vshrq_n_u32::<31>(gt255);
        let b2 = vshrq_n_u32::<31>(gt65535);
        let b3 = vshrq_n_u32::<31>(gt16m);
        let tag_vec = vaddq_u32(vaddq_u32(b1, b2), b3); // 0,1,2,3 per lane (u32x4)

        // Narrow to u8 via two steps: u32x4 → u16x4 → u8x4, then assemble ctrl byte.
        let tag16 = vmovn_u32(tag_vec); // uint16x4_t: tags as u16
        let tag8 = vmovn_u16(vcombine_u16(tag16, vdup_n_u16(0))); // uint8x8_t: tags as u8
        // Multiply each tag by its weight and sum horizontally.
        let weighted = vmul_u8(tag8, weights); // uint8x8_t
        let ctrl = vaddv_u8(weighted); // horizontal sum → ctrl byte

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            // Reinterpret u32 values as bytes and shuffle into packed output order.
            let v_bytes = vreinterpretq_u8_u32(v);
            // SAFETY: ENCODE_TABLE_CLASSIC[ctrl] is 16 bytes; ctrl < 256 (u8).
            let mask = vld1q_u8(ENCODE_TABLE_CLASSIC[ctrl as usize].as_ptr());
            let packed = vqtbl1q_u8(v_bytes, mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 4*n + 16 <= capacity.
            vst1q_u8(base_ptr.add(data_start + data_pos), packed);
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

/// Encode `values` into U32Variant0124 format using NEON `vqtbl1q_u8`.
///
/// Identical structure to `encode_into_classic` but uses the 0124 thresholds:
/// tag = (v>0) + (v>0xFF) + (v>0xFFFF), with widths [0,1,2,4].
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
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

    let weights = unsafe { vld1_u8([1u8, 4, 16, 64, 0, 0, 0, 0].as_ptr()) };

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        let v = unsafe {
            // SAFETY: i + 4 <= simd_n <= n; values slice bounds are valid.
            vld1q_u32(values.as_ptr().add(i))
        };

        // Variant0124: tag = (v>0) + (v>0xFF) + (v>0xFFFF)
        // vceqzq_u32: 0xFFFFFFFF if v==0, else 0; we want (v>0) = !eq0
        // vcgtq_u32(v, vdupq_n_u32(0)) gives 0xFFFFFFFF if v > 0 (unsigned).
        let gt0 = vcgtq_u32(v, vdupq_n_u32(0));
        let gt255 = vcgtq_u32(v, vdupq_n_u32(0xFF));
        let gt65535 = vcgtq_u32(v, vdupq_n_u32(0xFFFF));
        let b0 = vshrq_n_u32::<31>(gt0);
        let b1 = vshrq_n_u32::<31>(gt255);
        let b2 = vshrq_n_u32::<31>(gt65535);
        let tag_vec = vaddq_u32(vaddq_u32(b0, b1), b2);

        let tag16 = vmovn_u32(tag_vec);
        let tag8 = vmovn_u16(vcombine_u16(tag16, vdup_n_u16(0)));
        let weighted = vmul_u8(tag8, weights);
        let ctrl = vaddv_u8(weighted);

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            let v_bytes = vreinterpretq_u8_u32(v);
            // SAFETY: ENCODE_TABLE_0124[ctrl] is 16 bytes; ctrl < 256 (u8).
            let mask = vld1q_u8(ENCODE_TABLE_0124[ctrl as usize].as_ptr());
            let packed = vqtbl1q_u8(v_bytes, mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 4*n + 16 <= capacity.
            vst1q_u8(base_ptr.add(data_start + data_pos), packed);
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

/// Decode `n` u32 values from a U32Classic-encoded buffer using NEON `vqtbl1q_u8`.
///
/// Processes 4 values per ctrl byte. Falls back to the scalar path for any
/// trailing values when fewer than 16 data bytes remain (preventing an
/// out-of-bounds read on the unaligned 128-bit load).
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
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
        // Maximum data bytes for 4 values is 16 (all 4-byte).
        if data_pos + 16 > data_bytes.len() {
            break;
        }

        let result = unsafe {
            // SAFETY: TABLE[cb] is 16 bytes; cb < 256 (u8).
            let mask = vld1q_u8(TABLE[cb as usize].as_ptr());
            // SAFETY: data_pos + 16 <= data_bytes.len() verified above.
            let chunk = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            // vqtbl1q_u8: result[i] = chunk[mask[i]] if mask[i] < 16, else 0.
            // 0x80 >= 16 → zero fill, matching PSHUFB semantics.
            vqtbl1q_u8(chunk, mask)
        };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
            vst1q_u8(out_ptr, result);
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
    // DATA_LEN ≥ 4 for Classic; padded_pos ≤ rem−4 ≤ 11; load [11,27) ⊆ [0,32). ✓
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let result = unsafe {
                // SAFETY: padded is 32 bytes; padded_pos ≤ rem−4 ≤ 11;
                // load [padded_pos, padded_pos+16) ⊆ [0, 27) ⊆ [0, 32).
                let mask = vld1q_u8(TABLE[cb as usize].as_ptr());
                let chunk = vld1q_u8(padded.as_ptr().add(padded_pos));
                vqtbl1q_u8(chunk, mask)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
                vst1q_u8(out_ptr, result);
            }
            let consumed = DATA_LEN[cb as usize] as usize;
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe { out.set_len(base + decoded); }
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

/// Decode `n` u32 values from a U32Variant0124-encoded buffer using NEON `vqtbl1q_u8`.
///
/// Identical structure to `decode_into_classic` but uses the 0124 shuffle and
/// data-length tables (tag widths 0/1/2/4 instead of 1/2/3/4).
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
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
            let mask = vld1q_u8(TABLE_0124[cb as usize].as_ptr());
            // SAFETY: data_pos + 16 <= data_bytes.len() verified above.
            let chunk = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            vqtbl1q_u8(chunk, mask)
        };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
            vst1q_u8(out_ptr, result);
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
    // For 0124 DATA_LEN can be 0; padded_pos ≤ rem ≤ 15; load [15,31) ⊆ [0,32). ✓
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let result = unsafe {
                // SAFETY: padded is 32 bytes; padded_pos ≤ rem ≤ 15;
                // load [padded_pos, padded_pos+16) ⊆ [0, 31) ⊆ [0, 32).
                let mask = vld1q_u8(TABLE_0124[cb as usize].as_ptr());
                let chunk = vld1q_u8(padded.as_ptr().add(padded_pos));
                vqtbl1q_u8(chunk, mask)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
                vst1q_u8(out_ptr, result);
            }
            let consumed = DATA_LEN_0124[cb as usize] as usize;
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe { out.set_len(base + decoded); }
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
