// SSSE3 decode paths for U64Coder1234 and U64Coder1248.
//
// The file is named sse2 per project convention; the instruction used is
// PSHUFB (_mm_shuffle_epi8), which is SSSE3 (Penryn 2007+).

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{
    DATA_LEN_1234, DATA_LEN_1248_PAIR, ENCODE_TABLE_1234, ENCODE_TABLE_1248_PAIR, TABLE_1234,
    TABLE_1248_PAIR,
};
use crate::error::DecodeError;

/// Encode `values` into U64Coder1234 format using SSSE3 `PSHUFB`.
///
/// Processes 4 values per ctrl byte. Values must fit in u32 (values > u32::MAX
/// are silently truncated to their low 32 bits). The low 32 bits of each u64
/// are narrowed to a u32 register; the Classic encode path is then applied.
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
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

    // Bias and thresholds for Classic tag computation (same as U32Classic SSSE3 encode).
    let bias = _mm_set1_epi32(i32::MIN);
    let t1 = _mm_set1_epi32(i32::MIN + 0xFF);
    let t2 = _mm_set1_epi32(i32::MIN + 0xFFFF);
    let t3 = _mm_set1_epi32(i32::MIN + 0xFF_FFFF);
    let zero = _mm_setzero_si128();

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        // Load 4 u64 values as two 128-bit registers.
        let lo128 = unsafe {
            // SAFETY: i + 2 <= simd_n <= n; values slice bounds are valid.
            _mm_loadu_si128(values.as_ptr().add(i) as *const __m128i)
        };
        let hi128 = unsafe {
            // SAFETY: i + 4 <= simd_n <= n.
            _mm_loadu_si128(values.as_ptr().add(i + 2) as *const __m128i)
        };

        // Narrow 4 u64 values to 4 u32 (taking low 32 bits of each u64).
        // _mm_shuffle_epi32 with imm 0x88 = 0b10001000:
        //   dst[0]=src[0], dst[1]=src[2], dst[2]=src[0], dst[3]=src[2]
        // In u32 terms: src = [v0_lo, v0_hi, v1_lo, v1_hi]
        //   result = [v0_lo, v1_lo, v0_lo, v1_lo] (low halves in positions 0,1)
        let lo_pair = _mm_shuffle_epi32(lo128, 0x88); // [v0_lo, v1_lo, v0_lo, v1_lo]
        let hi_pair = _mm_shuffle_epi32(hi128, 0x88); // [v2_lo, v3_lo, v2_lo, v3_lo]
        // _mm_unpacklo_epi64: lower 64 bits of each → [v0_lo, v1_lo, v2_lo, v3_lo]
        let v32 = _mm_unpacklo_epi64(lo_pair, hi_pair);

        // Compute tags exactly as U32Classic SSSE3 encode.
        let bv = _mm_add_epi32(v32, bias);
        let c1 = _mm_cmpgt_epi32(bv, t1);
        let c2 = _mm_cmpgt_epi32(bv, t2);
        let c3 = _mm_cmpgt_epi32(bv, t3);
        let b1 = _mm_sub_epi32(zero, c1);
        let b2 = _mm_sub_epi32(zero, c2);
        let b3 = _mm_sub_epi32(zero, c3);
        let tag_vec = _mm_add_epi32(_mm_add_epi32(b1, b2), b3);

        let tag_bytes = _mm_shuffle_epi8(
            tag_vec,
            _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0),
        );
        let tags = _mm_cvtsi128_si32(tag_bytes) as u32;
        let ctrl =
            ((tags & 0x3) | ((tags >> 6) & 0x0C) | ((tags >> 12) & 0x30) | ((tags >> 18) & 0xC0))
                as u8;

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            // SAFETY: ENCODE_TABLE_1234[ctrl] is 16 bytes; ctrl < 256 (u8).
            let enc_mask =
                _mm_loadu_si128(ENCODE_TABLE_1234[ctrl as usize].as_ptr() as *const __m128i);
            let packed = _mm_shuffle_epi8(v32, enc_mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 4*n + 16 <= capacity.
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed);
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

/// Encode `values` into U64Coder1248 format using SSSE3 `PSHUFB`.
///
/// Processes 4 values (2 pairs) per ctrl byte. Tags are computed via scalar
/// code (SSSE3 lacks 64-bit comparison); SIMD is used only for data packing
/// via `PSHUFB` on the 16-byte pair registers.
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
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

    // Helper closure for 1248 tag computation (scalar, since SSSE3 has no 64-bit cmpgt).
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

            // Pack lo pair (values 0,1): load 16 bytes (2 u64s), apply PSHUFB.
            // SAFETY: i + 2 <= simd_n <= n; pointer valid.
            let pair_lo = _mm_loadu_si128(values.as_ptr().add(i) as *const __m128i);
            // SAFETY: ENCODE_TABLE_1248_PAIR[lo_key] is 16 bytes; lo_key < 16.
            let enc_mask_lo =
                _mm_loadu_si128(ENCODE_TABLE_1248_PAIR[lo_key].as_ptr() as *const __m128i);
            let packed_lo = _mm_shuffle_epi8(pair_lo, enc_mask_lo);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_lo,
            );
            data_pos += DATA_LEN_1248_PAIR[lo_key] as usize;

            // Pack hi pair (values 2,3).
            // SAFETY: i + 4 <= simd_n <= n; pointer valid.
            let pair_hi = _mm_loadu_si128(values.as_ptr().add(i + 2) as *const __m128i);
            // SAFETY: ENCODE_TABLE_1248_PAIR[hi_key] is 16 bytes; hi_key < 16.
            let enc_mask_hi =
                _mm_loadu_si128(ENCODE_TABLE_1248_PAIR[hi_key].as_ptr() as *const __m128i);
            let packed_hi = _mm_shuffle_epi8(pair_hi, enc_mask_hi);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_hi,
            );
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

    // Padded tail: for well-formed input, guard fires (rem < 16) with complete
    // groups of 4 remaining, fitting a zero-padded 32-byte buffer. `rem` and
    // each iteration's `consumed` are still re-validated below, since a
    // truncated/corrupted `data` (mismatched against the declared `n`) can't
    // be trusted to satisfy that bound.
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated {
                index: base + decoded,
            });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let consumed = DATA_LEN_1234[cb as usize] as usize;
            if padded_pos + consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated {
                    index: base + decoded,
                });
            }
            let u32s = unsafe {
                // SAFETY: padded_pos + 16 <= padded.len() checked above.
                let mask = _mm_loadu_si128(TABLE_1234[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                _mm_shuffle_epi8(chunk, mask)
            };
            let lo = _mm_unpacklo_epi32(u32s, zero);
            let hi = _mm_unpackhi_epi32(u32s, zero);
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
                _mm_storeu_si128(out_ptr, lo);
                _mm_storeu_si128(out_ptr.add(1), hi);
            }
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe {
            out.set_len(base + decoded);
        }
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

        // The hi load starts at data_pos + lo_bytes and reads 16 bytes.
        if data_pos + lo_bytes + 16 > data_bytes.len() {
            break;
        }

        let (lo_pair, hi_pair) = unsafe {
            // SAFETY: TABLE_1248_PAIR indices are < 16 (4-bit keys).
            let mask_lo = _mm_loadu_si128(TABLE_1248_PAIR[lo_key].as_ptr() as *const __m128i);
            // SAFETY: data_pos + lo_bytes + 16 <= data_bytes.len() checked above;
            // lo load: data_pos + 16 ≤ data_pos + lo_bytes + 16 (lo_bytes ≥ 0).
            let chunk_lo = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let lo = _mm_shuffle_epi8(chunk_lo, mask_lo);

            let mask_hi = _mm_loadu_si128(TABLE_1248_PAIR[hi_key].as_ptr() as *const __m128i);
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

    // Padded tail: guard fired when rem < lo_bytes + 16 (lo_bytes ≤ 16 → rem ≤ 31), but
    // complete groups of 4 may still remain. Copy the remaining bytes into a 64-byte
    // zero-padded stack buffer so SIMD loads are always in-bounds.
    //
    // Bound derivation for hi load at padded_pos + lo_bytes:
    //   (1) Guard condition: rem < lo_bytes_guard + 16 ≤ 16 + 16 = 32 → rem ≤ 31.
    //   (2) At each iteration the current group's data (lo_bytes + hi_bytes bytes) starts
    //       at padded_pos; all consumed groups fit within rem:
    //       padded_pos + lo_bytes + hi_bytes ≤ rem → padded_pos + lo_bytes ≤ rem − hi_bytes.
    //   (3) hi_bytes = DATA_LEN_1248_PAIR[hi_key] ≥ 2 (min pair = 1+1 = 2).
    //   (4) padded_pos + lo_bytes ≤ rem − hi_bytes ≤ 31 − 2 = 29.
    //   (5) Hi load end: 29 + 16 = 45 ≤ 64. ✓  Lo load end: 29 + 16 = 45 ≤ 64. ✓
    if decoded + 4 <= n {
        let mut padded = [0u8; 64];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated {
                index: base + decoded,
            });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let lo_key = (cb & 0x0F) as usize;
            let hi_key = (cb >> 4) as usize;
            let lo_bytes = DATA_LEN_1248_PAIR[lo_key] as usize;
            let hi_bytes = DATA_LEN_1248_PAIR[hi_key] as usize;
            let consumed = lo_bytes + hi_bytes;
            // The hi load starts at padded_pos + lo_bytes and also reads 16
            // bytes, so it - not the lo load - is the binding bound check.
            if padded_pos + consumed > rem || padded_pos + lo_bytes + 16 > padded.len() {
                return Err(DecodeError::DataTruncated {
                    index: base + decoded,
                });
            }
            let (lo_pair, hi_pair) = unsafe {
                // SAFETY: padded_pos + lo_bytes + 16 <= padded.len() checked above;
                // the lo load (ending at padded_pos + 16) is within that same bound.
                let mask_lo = _mm_loadu_si128(TABLE_1248_PAIR[lo_key].as_ptr() as *const __m128i);
                let chunk_lo = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                let lo = _mm_shuffle_epi8(chunk_lo, mask_lo);
                let mask_hi = _mm_loadu_si128(TABLE_1248_PAIR[hi_key].as_ptr() as *const __m128i);
                let chunk_hi =
                    _mm_loadu_si128(padded.as_ptr().add(padded_pos + lo_bytes) as *const __m128i);
                let hi = _mm_shuffle_epi8(chunk_hi, mask_hi);
                (lo, hi)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 4 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
                _mm_storeu_si128(out_ptr, lo_pair);
                _mm_storeu_si128(out_ptr.add(1), hi_pair);
            }
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
        unsafe {
            out.set_len(base + decoded);
        }
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
