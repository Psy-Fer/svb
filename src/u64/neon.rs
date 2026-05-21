// NEON decode and encode paths for U64Coder1234 and U64Coder1248 (AArch64).
//
// vqtbl1q_u8 is the AArch64 equivalent of SSSE3 PSHUFB: it zeroes the output
// byte when the index byte is >= 16. Our decode table uses 0x80 for zero-fill
// slots, satisfying both conditions.
//
// For encode:
//   1234: vmovn_u64 narrows uint64x2_t → uint32x2_t (taking low 32 bits), then
//         the Classic u32 NEON encode path applies.
//   1248: scalar tag computation (values can be full u64) + SIMD packing with
//         ENCODE_TABLE_1248_PAIR and vqtbl1q_u8.

use core::arch::aarch64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{
    DATA_LEN_1234, DATA_LEN_1248_PAIR, ENCODE_TABLE_1234, ENCODE_TABLE_1248_PAIR, TABLE_1234,
    TABLE_1248_PAIR,
};
use crate::error::DecodeError;

/// Encode `values` into U64Coder1234 format using NEON `vqtbl1q_u8`.
///
/// Processes 4 values per ctrl byte. The low 32 bits of each u64 are extracted
/// via `vmovn_u64`; tags are computed with `vcgtq_u32` (unsigned). Values
/// > u32::MAX are silently truncated. Remaining values (n % 4) are handled by
/// the scalar path.
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn encode_into_1234(values: &[u64], out: &mut Vec<u8>) {
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

    // Weights for assembling the ctrl byte from 4 tags (each 0..3).
    let weights = unsafe { vld1_u8([1u8, 4, 16, 64, 0, 0, 0, 0].as_ptr()) };

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        // Load 4 u64 values as two uint64x2_t registers.
        let lo_vals = unsafe {
            // SAFETY: i + 2 <= simd_n <= n; values slice bounds are valid.
            vld1q_u64(values.as_ptr().add(i))
        };
        let hi_vals = unsafe {
            // SAFETY: i + 4 <= simd_n <= n.
            vld1q_u64(values.as_ptr().add(i + 2))
        };

        // Narrow 4 u64 → 4 u32 (take low 32 bits of each u64).
        // vmovn_u64: uint64x2_t → uint32x2_t (low 32 bits of each lane)
        let lo_u32 = vmovn_u64(lo_vals); // uint32x2_t: [v0_low32, v1_low32]
        let hi_u32 = vmovn_u64(hi_vals); // uint32x2_t: [v2_low32, v3_low32]
        let v32 = vcombine_u32(lo_u32, hi_u32); // uint32x4_t: [v0,v1,v2,v3] low32

        // Compute per-lane tags exactly as U32Classic NEON encode.
        let gt255 = vcgtq_u32(v32, vdupq_n_u32(0xFF));
        let gt65535 = vcgtq_u32(v32, vdupq_n_u32(0xFFFF));
        let gt16m = vcgtq_u32(v32, vdupq_n_u32(0xFF_FFFF));
        let b1 = vshrq_n_u32::<31>(gt255);
        let b2 = vshrq_n_u32::<31>(gt65535);
        let b3 = vshrq_n_u32::<31>(gt16m);
        let tag_vec = vaddq_u32(vaddq_u32(b1, b2), b3);

        let tag16 = vmovn_u32(tag_vec);
        let tag8 = vmovn_u16(vcombine_u16(tag16, vdup_n_u16(0)));
        let weighted = vmul_u8(tag8, weights);
        let ctrl = vaddv_u8(weighted);

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            let v_bytes = vreinterpretq_u8_u32(v32);
            // SAFETY: ENCODE_TABLE_1234[ctrl] is 16 bytes; ctrl < 256 (u8).
            let mask = vld1q_u8(ENCODE_TABLE_1234[ctrl as usize].as_ptr());
            let packed = vqtbl1q_u8(v_bytes, mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 4*n + 16 <= capacity.
            vst1q_u8(base_ptr.add(data_start + data_pos), packed);
        }

        data_pos += DATA_LEN_1234[ctrl as usize] as usize;
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
        out.extend_from_slice(&(v as u32).to_le_bytes()[..count]);
    }
}

/// Encode `values` into U64Coder1248 format using NEON `vqtbl1q_u8`.
///
/// Processes 4 values (2 pairs) per ctrl byte. Tags are computed via scalar
/// code (to handle the full u64 range); SIMD is used for data packing via
/// `vqtbl1q_u8` on 16-byte pair registers.
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn encode_into_1248(values: &[u64], out: &mut Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return;
    }

    let ctrl_len = n.div_ceil(4);
    let ctrl_start = out.len();

    // Reserve ctrl bytes + worst-case data (8 bytes/value) + 16-byte SIMD overrun guard.
    out.reserve(ctrl_len + 8 * n + 16);
    // Zero-initialize ctrl bytes so the scalar tail can OR into them safely.
    out.resize(ctrl_start + ctrl_len, 0u8);

    let simd_n = (n / 4) * 4;
    let data_start = ctrl_start + ctrl_len;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    let tag1248 = |v: u64| -> u8 {
        if v <= 0xFF {
            0
        } else if v <= 0xFFFF {
            1
        } else if v <= 0xFFFF_FFFF {
            2
        } else {
            3
        }
    };

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        // Compute tags for all 4 values via scalar code.
        let v0 = unsafe { *values.as_ptr().add(i) };
        let v1 = unsafe { *values.as_ptr().add(i + 1) };
        let v2 = unsafe { *values.as_ptr().add(i + 2) };
        let v3 = unsafe { *values.as_ptr().add(i + 3) };
        let t0 = tag1248(v0);
        let t1 = tag1248(v1);
        let t2 = tag1248(v2);
        let t3 = tag1248(v3);

        let ctrl = t0 | (t1 << 2) | (t2 << 4) | (t3 << 6);
        let lo_key = (ctrl & 0x0F) as usize;
        let hi_key = (ctrl >> 4) as usize;

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            // Pack lo pair (values 0,1): load 16 bytes (2 u64s), apply vqtbl1q_u8.
            // SAFETY: i + 2 <= simd_n <= n; pointer valid.
            let pair_lo = vld1q_u8(values.as_ptr().add(i) as *const u8);
            // SAFETY: ENCODE_TABLE_1248_PAIR[lo_key] is 16 bytes; lo_key < 16.
            let mask_lo = vld1q_u8(ENCODE_TABLE_1248_PAIR[lo_key].as_ptr());
            let packed_lo = vqtbl1q_u8(pair_lo, mask_lo);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            vst1q_u8(base_ptr.add(data_start + data_pos), packed_lo);
            data_pos += DATA_LEN_1248_PAIR[lo_key] as usize;

            // Pack hi pair (values 2,3).
            // SAFETY: i + 4 <= simd_n <= n; pointer valid.
            let pair_hi = vld1q_u8(values.as_ptr().add(i + 2) as *const u8);
            // SAFETY: ENCODE_TABLE_1248_PAIR[hi_key] is 16 bytes; hi_key < 16.
            let mask_hi = vld1q_u8(ENCODE_TABLE_1248_PAIR[hi_key].as_ptr());
            let packed_hi = vqtbl1q_u8(pair_hi, mask_hi);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            vst1q_u8(base_ptr.add(data_start + data_pos), packed_hi);
            data_pos += DATA_LEN_1248_PAIR[hi_key] as usize;
        }

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
        } else if v <= 0xFFFF_FFFF {
            (2, 4)
        } else {
            (3, 8)
        };
        // SAFETY: ctrl_start + j/4 < ctrl_start + ctrl_len <= out.len().
        out[ctrl_start + j / 4] |= tag << ((j % 4) * 2);
        out.extend_from_slice(&v.to_le_bytes()[..count]);
    }
}

/// Decode `n` u64 values from a U64Coder1234-encoded buffer using NEON `vqtbl1q_u8`.
///
/// Processes 4 values per ctrl byte. PSHUFB expands the data bytes into 4 u32
/// slots; `vmovl_u32` zero-extends them to u64. Falls back to the scalar path
/// when fewer than 16 data bytes remain.
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
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

    while decoded + 4 <= n {
        let cb = ctrl[ctrl_pos];

        // Maximum data bytes for 4 values (all tag-3 = 4 bytes each) is 16.
        if data_pos + 16 > data_bytes.len() {
            break;
        }

        let u32s = unsafe {
            // SAFETY: TABLE_1234[cb] is 16 bytes; cb < 256 (u8).
            let mask = vld1q_u8(TABLE_1234[cb as usize].as_ptr());
            // SAFETY: data_pos + 16 <= data_bytes.len() verified above.
            let chunk = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            vreinterpretq_u32_u8(vqtbl1q_u8(chunk, mask))
        };

        // Zero-extend 4 × u32 → 4 × u64 across two uint64x2_t registers.
        // vmovl_u32: uint32x2_t → uint64x2_t (zero-extend each lane)
        let lo = vmovl_u32(vget_low_u32(u32s)); // uint64x2_t: [u64[0], u64[1]]
        let hi = vmovl_high_u32(u32s); // uint64x2_t: [u64[2], u64[3]]

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n, so
            // base + decoded + 4 <= base + n <= out.capacity().
            let out_ptr = out.as_mut_ptr().add(base + decoded);
            vst1q_u64(out_ptr, lo);
            vst1q_u64(out_ptr.add(2), hi);
        }

        data_pos += DATA_LEN_1234[cb as usize] as usize;
        ctrl_pos += 1;
        decoded += 4;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    // Padded tail: guard fired (rem < 16) but complete groups of 4 may remain.
    // DATA_LEN_1234 ≥ 4; padded_pos ≤ rem−4 ≤ 11; load [11,27) ⊆ [0,32). ✓
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let u32s = unsafe {
                // SAFETY: padded is 32 bytes; padded_pos ≤ rem−4 ≤ 11;
                // load [padded_pos, padded_pos+16) ⊆ [0, 27) ⊆ [0, 32).
                let mask = vld1q_u8(TABLE_1234[cb as usize].as_ptr());
                let chunk = vld1q_u8(padded.as_ptr().add(padded_pos));
                vreinterpretq_u32_u8(vqtbl1q_u8(chunk, mask))
            };
            let lo = vmovl_u32(vget_low_u32(u32s));
            let hi = vmovl_high_u32(u32s);
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded);
                vst1q_u64(out_ptr, lo);
                vst1q_u64(out_ptr.add(2), hi);
            }
            let consumed = DATA_LEN_1234[cb as usize] as usize;
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe { out.set_len(base + decoded); }
    }

    // Scalar for n % 4 remainder (0–3 values).
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

/// Decode `n` u64 values from a U64Coder1248-encoded buffer using NEON `vqtbl1q_u8`.
///
/// Processes 4 values (2 pairs) per ctrl byte. The ctrl byte is split into a
/// low nibble (tags for values 0 and 1) and high nibble (tags for values 2 and
/// 3), each indexing the 16-entry pair table. Falls back to the scalar path
/// when fewer than 32 data bytes remain (the worst-case for one ctrl byte).
///
/// # Safety
/// Must run on AArch64 (NEON is mandatory on that architecture).
#[allow(dead_code)]
#[target_feature(enable = "neon")]
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

        // The hi load starts at data_pos + lo_bytes and reads 16 bytes.
        if data_pos + lo_bytes + 16 > data_bytes.len() {
            break;
        }

        let (lo_pair, hi_pair) = unsafe {
            // SAFETY: TABLE_1248_PAIR indices are < 16 (4-bit keys).
            let mask_lo = vld1q_u8(TABLE_1248_PAIR[lo_key].as_ptr());
            // SAFETY: data_pos + lo_bytes + 16 <= data_bytes.len() checked above;
            // lo load: data_pos + 16 ≤ data_pos + lo_bytes + 16 (lo_bytes ≥ 0).
            let chunk_lo = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            let lo = vqtbl1q_u8(chunk_lo, mask_lo);

            let mask_hi = vld1q_u8(TABLE_1248_PAIR[hi_key].as_ptr());
            let chunk_hi = vld1q_u8(data_bytes.as_ptr().add(data_pos + lo_bytes));
            let hi = vqtbl1q_u8(chunk_hi, mask_hi);

            (lo, hi)
        };

        unsafe {
            // SAFETY: decoded + 4 <= n; 4 u64s = 32 bytes = 2 × uint8x16_t.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
            vst1q_u8(out_ptr, lo_pair);
            vst1q_u8(out_ptr.add(16), hi_pair);
        }

        data_pos += lo_bytes + DATA_LEN_1248_PAIR[hi_key] as usize;
        ctrl_pos += 1;
        decoded += 4;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    // Padded tail: guard fired (rem < lo_bytes+16 ≤ 32) but groups of 4 may remain.
    // At hi load: padded_pos+lo_bytes ≤ rem−hi_bytes ≤ 29; [29,45) ⊆ [0,64). ✓
    if decoded + 4 <= n {
        let mut padded = [0u8; 64];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let lo_key = (cb & 0x0F) as usize;
            let hi_key = (cb >> 4) as usize;
            let lo_bytes = DATA_LEN_1248_PAIR[lo_key] as usize;
            let (lo_pair, hi_pair) = unsafe {
                // SAFETY: padded is 64 bytes; padded_pos+lo_bytes ≤ rem−hi_bytes ≤ 29;
                // lo load [padded_pos, padded_pos+16) ⊆ [0,46) ⊆ [0,64);
                // hi load [padded_pos+lo_bytes, padded_pos+lo_bytes+16) ⊆ [0,45) ⊆ [0,64).
                let mask_lo = vld1q_u8(TABLE_1248_PAIR[lo_key].as_ptr());
                let chunk_lo = vld1q_u8(padded.as_ptr().add(padded_pos));
                let lo = vqtbl1q_u8(chunk_lo, mask_lo);
                let mask_hi = vld1q_u8(TABLE_1248_PAIR[hi_key].as_ptr());
                let chunk_hi = vld1q_u8(padded.as_ptr().add(padded_pos + lo_bytes));
                let hi = vqtbl1q_u8(chunk_hi, mask_hi);
                (lo, hi)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut u8;
                vst1q_u8(out_ptr, lo_pair);
                vst1q_u8(out_ptr.add(16), hi_pair);
            }
            let consumed = lo_bytes + DATA_LEN_1248_PAIR[hi_key] as usize;
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe { out.set_len(base + decoded); }
    }

    // Scalar for n % 4 remainder (0–3 values).
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
