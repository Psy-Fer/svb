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

use super::shuffle::{
    DATA_LEN_1234, DATA_LEN_1248_PAIR, ENCODE_TABLE_1234, ENCODE_TABLE_1248_PAIR, TABLE_1234,
    TABLE_1248_PAIR,
};
use crate::error::DecodeError;

/// Encode `values` into U64Coder1234 format using AVX2.
///
/// Processes 8 values (2 ctrl bytes) per iteration. The low 32 bits of each
/// u64 are extracted using 256-bit shuffles; two 128-bit PSHUFB ops pack the
/// data. Values > u32::MAX are silently truncated.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    let simd_n = (n / 8) * 8;
    let data_start = ctrl_start + ctrl_len;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    // Bias and thresholds for Classic tag computation (256-bit).
    let bias = _mm256_set1_epi32(i32::MIN);
    let t1 = _mm256_set1_epi32(i32::MIN + 0xFF);
    let t2 = _mm256_set1_epi32(i32::MIN + 0xFFFF);
    let t3 = _mm256_set1_epi32(i32::MIN + 0xFF_FFFF);
    let zero256 = _mm256_setzero_si256();

    let gather_lo = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);

    let mut block = 0usize;
    while block * 4 < simd_n {
        let i = block * 4;

        // Load 8 u64 values as four 128-bit registers (2 u64 each).
        let r0 = unsafe { _mm_loadu_si128(values.as_ptr().add(i) as *const __m128i) };
        let r1 = unsafe { _mm_loadu_si128(values.as_ptr().add(i + 2) as *const __m128i) };
        let r2 = unsafe { _mm_loadu_si128(values.as_ptr().add(i + 4) as *const __m128i) };
        let r3 = unsafe { _mm_loadu_si128(values.as_ptr().add(i + 6) as *const __m128i) };

        // Narrow each pair of u64 → pair of u32 (low 32 bits), combine to 4 u32.
        // _mm_shuffle_epi32 0x88 = gathers u32 lanes 0 and 2 (the low halves of u64s).
        let p0 = _mm_unpacklo_epi64(_mm_shuffle_epi32(r0, 0x88), _mm_shuffle_epi32(r1, 0x88));
        let p1 = _mm_unpacklo_epi64(_mm_shuffle_epi32(r2, 0x88), _mm_shuffle_epi32(r3, 0x88));

        // Combine into 256-bit for vectorised tag computation.
        let v32_256 = _mm256_set_m128i(p1, p0);

        let bv = _mm256_add_epi32(v32_256, bias);
        let c1 = _mm256_cmpgt_epi32(bv, t1);
        let c2 = _mm256_cmpgt_epi32(bv, t2);
        let c3 = _mm256_cmpgt_epi32(bv, t3);
        let b1 = _mm256_sub_epi32(zero256, c1);
        let b2 = _mm256_sub_epi32(zero256, c2);
        let b3 = _mm256_sub_epi32(zero256, c3);
        let tag_vec = _mm256_add_epi32(_mm256_add_epi32(b1, b2), b3);

        let tag_lo = _mm256_castsi256_si128(tag_vec);
        let tag_hi = _mm256_extracti128_si256(tag_vec, 1);

        let tag_bytes_lo = _mm_shuffle_epi8(tag_lo, gather_lo);
        let tags_lo = _mm_cvtsi128_si32(tag_bytes_lo) as u32;
        let c0 = ((tags_lo & 0x3)
            | ((tags_lo >> 6) & 0x0C)
            | ((tags_lo >> 12) & 0x30)
            | ((tags_lo >> 18) & 0xC0)) as u8;

        let tag_bytes_hi = _mm_shuffle_epi8(tag_hi, gather_lo);
        let tags_hi = _mm_cvtsi128_si32(tag_bytes_hi) as u32;
        let c1b = ((tags_hi & 0x3)
            | ((tags_hi >> 6) & 0x0C)
            | ((tags_hi >> 12) & 0x30)
            | ((tags_hi >> 18) & 0xC0)) as u8;

        unsafe {
            // SAFETY: ctrl_start + block and block+1 < ctrl_start + ctrl_len.
            *base_ptr.add(ctrl_start + block) = c0;
            *base_ptr.add(ctrl_start + block + 1) = c1b;

            // Pack first 4 values (lo half of v32_256).
            let enc_mask_lo =
                _mm_loadu_si128(ENCODE_TABLE_1234[c0 as usize].as_ptr() as *const __m128i);
            let packed_lo = _mm_shuffle_epi8(p0, enc_mask_lo);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_lo,
            );
            data_pos += DATA_LEN_1234[c0 as usize] as usize;

            // Pack second 4 values (hi half of v32_256).
            let enc_mask_hi =
                _mm_loadu_si128(ENCODE_TABLE_1234[c1b as usize].as_ptr() as *const __m128i);
            let packed_hi = _mm_shuffle_epi8(p1, enc_mask_hi);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_hi,
            );
            data_pos += DATA_LEN_1234[c1b as usize] as usize;
        }

        block += 2;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 8 remaining values.
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

/// Encode `values` into U64Coder1248 format using AVX2.
///
/// Processes 8 values (2 ctrl bytes) per iteration. Tags are computed via
/// scalar code (AVX2 lacks practical 64-bit comparison without AVX-512);
/// SIMD is used for data packing via four `PSHUFB` ops per iteration.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    let simd_n = (n / 8) * 8;
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

        // Compute tags for all 8 values via scalar code.
        let v0 = unsafe { *values.as_ptr().add(i) };
        let v1 = unsafe { *values.as_ptr().add(i + 1) };
        let v2 = unsafe { *values.as_ptr().add(i + 2) };
        let v3 = unsafe { *values.as_ptr().add(i + 3) };
        let v4 = unsafe { *values.as_ptr().add(i + 4) };
        let v5 = unsafe { *values.as_ptr().add(i + 5) };
        let v6 = unsafe { *values.as_ptr().add(i + 6) };
        let v7 = unsafe { *values.as_ptr().add(i + 7) };

        let c0 = tag1248(v0) | (tag1248(v1) << 2) | (tag1248(v2) << 4) | (tag1248(v3) << 6);
        let c1 = tag1248(v4) | (tag1248(v5) << 2) | (tag1248(v6) << 4) | (tag1248(v7) << 6);

        let lo_key0 = (c0 & 0x0F) as usize;
        let hi_key0 = (c0 >> 4) as usize;
        let lo_key1 = (c1 & 0x0F) as usize;
        let hi_key1 = (c1 >> 4) as usize;

        unsafe {
            // SAFETY: ctrl_start + block and block+1 < ctrl_start + ctrl_len.
            *base_ptr.add(ctrl_start + block) = c0;
            *base_ptr.add(ctrl_start + block + 1) = c1;

            // Pack pairs for ctrl byte c0.
            let pair_lo0 = _mm_loadu_si128(values.as_ptr().add(i) as *const __m128i);
            let enc_mask_lo0 =
                _mm_loadu_si128(ENCODE_TABLE_1248_PAIR[lo_key0].as_ptr() as *const __m128i);
            let packed_lo0 = _mm_shuffle_epi8(pair_lo0, enc_mask_lo0);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_lo0,
            );
            data_pos += DATA_LEN_1248_PAIR[lo_key0] as usize;

            let pair_hi0 = _mm_loadu_si128(values.as_ptr().add(i + 2) as *const __m128i);
            let enc_mask_hi0 =
                _mm_loadu_si128(ENCODE_TABLE_1248_PAIR[hi_key0].as_ptr() as *const __m128i);
            let packed_hi0 = _mm_shuffle_epi8(pair_hi0, enc_mask_hi0);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_hi0,
            );
            data_pos += DATA_LEN_1248_PAIR[hi_key0] as usize;

            // Pack pairs for ctrl byte c1.
            let pair_lo1 = _mm_loadu_si128(values.as_ptr().add(i + 4) as *const __m128i);
            let enc_mask_lo1 =
                _mm_loadu_si128(ENCODE_TABLE_1248_PAIR[lo_key1].as_ptr() as *const __m128i);
            let packed_lo1 = _mm_shuffle_epi8(pair_lo1, enc_mask_lo1);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_lo1,
            );
            data_pos += DATA_LEN_1248_PAIR[lo_key1] as usize;

            let pair_hi1 = _mm_loadu_si128(values.as_ptr().add(i + 6) as *const __m128i);
            let enc_mask_hi1 =
                _mm_loadu_si128(ENCODE_TABLE_1248_PAIR[hi_key1].as_ptr() as *const __m128i);
            let packed_hi1 = _mm_shuffle_epi8(pair_hi1, enc_mask_hi1);
            // SAFETY: data_start + data_pos + 16 <= capacity.
            _mm_storeu_si128(
                base_ptr.add(data_start + data_pos) as *mut __m128i,
                packed_hi1,
            );
            data_pos += DATA_LEN_1248_PAIR[hi_key1] as usize;
        }

        block += 2;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 8 remaining values.
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

        // The c1 load starts at data_pos + c0_bytes and reads 16 bytes.
        if data_pos + c0_bytes + 16 > data_bytes.len() {
            break;
        }

        let (u64s_c0, u64s_c1) = unsafe {
            // SAFETY: TABLE_1234 indices valid (u8 → < 256).
            let mask_c0 = _mm_loadu_si128(TABLE_1234[c0 as usize].as_ptr() as *const __m128i);
            let mask_c1 = _mm_loadu_si128(TABLE_1234[c1 as usize].as_ptr() as *const __m128i);
            // SAFETY: data_pos + c0_bytes + 16 <= data_bytes.len() checked above;
            // c0 load: data_pos + 16 ≤ data_pos + c0_bytes + 16 (c0_bytes ≥ 1).
            let chunk_c0 = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let chunk_c1 =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos + c0_bytes) as *const __m128i);
            let u32s_c0 = _mm_shuffle_epi8(chunk_c0, mask_c0);
            let u32s_c1 = _mm_shuffle_epi8(chunk_c1, mask_c1);
            // _mm256_cvtepu32_epi64: 4 u32 → 4 u64 (zero-extend), requires AVX2.
            (
                _mm256_cvtepu32_epi64(u32s_c0),
                _mm256_cvtepu32_epi64(u32s_c1),
            )
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

    // SSE2-style padded tail: for well-formed input, guard fires
    // (rem < c0_bytes + 16 ≤ 32) with groups of 4 remaining, fitting a
    // zero-padded 64-byte buffer. `rem` and each iteration's `consumed` are
    // still re-validated below, since a truncated/corrupted `data`
    // (mismatched against the declared `n`) can't be trusted to satisfy that
    // bound.
    if decoded + 4 <= n {
        let zero = _mm_setzero_si128();
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

        // The hi1 load starts at data_pos + lo_bytes0 + hi_bytes0 + lo_bytes1 and reads 16 bytes.
        if data_pos + lo_bytes0 + hi_bytes0 + lo_bytes1 + 16 > data_bytes.len() {
            break;
        }

        let off_hi0 = data_pos + lo_bytes0;
        let off_lo1 = off_hi0 + hi_bytes0;
        let off_hi1 = off_lo1 + lo_bytes1;

        let (result0, result1) = unsafe {
            // SAFETY: guard ensures data_pos + lo_bytes0 + hi_bytes0 + lo_bytes1 + 16
            // <= data_bytes.len(); all four offsets + 16 are ≤ that bound.
            let mask_lo0 = _mm_loadu_si128(TABLE_1248_PAIR[lo_key0].as_ptr() as *const __m128i);
            let chunk_lo0 = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let lo0 = _mm_shuffle_epi8(chunk_lo0, mask_lo0);

            let mask_hi0 = _mm_loadu_si128(TABLE_1248_PAIR[hi_key0].as_ptr() as *const __m128i);
            let chunk_hi0 = _mm_loadu_si128(data_bytes.as_ptr().add(off_hi0) as *const __m128i);
            let hi0 = _mm_shuffle_epi8(chunk_hi0, mask_hi0);

            let mask_lo1 = _mm_loadu_si128(TABLE_1248_PAIR[lo_key1].as_ptr() as *const __m128i);
            let chunk_lo1 = _mm_loadu_si128(data_bytes.as_ptr().add(off_lo1) as *const __m128i);
            let lo1 = _mm_shuffle_epi8(chunk_lo1, mask_lo1);

            let mask_hi1 = _mm_loadu_si128(TABLE_1248_PAIR[hi_key1].as_ptr() as *const __m128i);
            let chunk_hi1 = _mm_loadu_si128(data_bytes.as_ptr().add(off_hi1) as *const __m128i);
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

    // SSE2-style padded tail: guard fires when rem < lo0+hi0+lo1+16 (each term ≤ 16 →
    // rem ≤ 63). Complete groups of 4 may still remain. Copy remaining bytes into a
    // 96-byte zero-padded stack buffer so SIMD loads are always in-bounds.
    //
    // Bound derivation for hi load at padded_pos + lo_bytes (1 ctrl byte / 4 values):
    //   (1) Guard: rem < lo0+hi0+lo1+16 ≤ 16+16+16+16 = 64 → rem ≤ 63.
    //   (2) At each padded iteration the group's data (lo_bytes + hi_bytes) starts at
    //       padded_pos; all groups fit within rem:
    //       padded_pos + lo_bytes + hi_bytes ≤ rem → padded_pos + lo_bytes ≤ rem − hi_bytes.
    //   (3) hi_bytes = DATA_LEN_1248_PAIR[hi_key] ≥ 2.
    //   (4) padded_pos + lo_bytes ≤ rem − hi_bytes ≤ 63 − 2 = 61.
    //   (5) Hi load end: 61 + 16 = 77 ≤ 96. ✓  Lo load end: 61 + 16 = 77 ≤ 96. ✓
    if decoded + 4 <= n {
        let mut padded = [0u8; 96];
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
