// AVX2 decode path for SVB16.
//
// _mm256_shuffle_epi8 (VPSHUFB) operates as two independent 128-bit PSHUFB
// lanes. We process 2 ctrl bytes (16 values) per iteration: c0 drives the
// lower lane, c1 the upper lane. Each lane's data is loaded independently
// from its start offset in the data stream.
//
// The scalar tail (≤ 15 remaining values, or final iteration with < 32 data
// bytes) is handled by the scalar path directly.

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{ENCODE_TABLE, TABLE};
use crate::error::DecodeError;

/// Encode `values` into SVB16 format using AVX2.
///
/// Processes 16 values (2 ctrl bytes) per iteration. A single 256-bit load
/// covers both 8-value groups; ctrl bytes c0 and c1 are extracted from one
/// `_mm256_movemask_epi8` call. `_mm256_shuffle_epi8` packs both groups in
/// a single VPSHUFB; each lane is stored independently to its data position.
/// Remaining values (n % 16) are handled by the scalar path.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    let simd_n = (n / 16) * 16;
    let data_start = ctrl_start + ctrl_len;
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    // block counts ctrl bytes; each iteration writes 2 ctrl bytes (16 values).
    let mut block = 0usize;
    while block * 8 < simd_n {
        let i = block * 8;

        // Load 16 u16 values (32 bytes) into a 256-bit register.
        let v = unsafe {
            // SAFETY: i + 16 <= simd_n <= n; values slice bounds are valid.
            _mm256_loadu_si256(values.as_ptr().add(i) as *const __m256i)
        };

        // Compute ctrl bytes for both 8-value groups.
        // _mm256_packs_epi16 operates in two independent 128-bit lanes:
        //   lower lane → bits 0..7 and 8..15 of movemask → c0 (bits 0..7)
        //   upper lane → bits 16..23 and 24..31 of movemask → c1 (bits 16..23)
        let hi = _mm256_srli_epi16(v, 8);
        let needs_two = _mm256_cmpgt_epi16(hi, _mm256_setzero_si256());
        let ctrl_packed = _mm256_packs_epi16(needs_two, needs_two);
        let movemask = _mm256_movemask_epi8(ctrl_packed);
        let c0 = (movemask & 0xFF) as u8;
        let c1 = ((movemask >> 16) & 0xFF) as u8;

        unsafe {
            // SAFETY: block and block+1 < ctrl_len (block * 8 < simd_n ≤ n,
            // so block + 1 ≤ n/8 = ctrl_len when n is a multiple of 8; the
            // AVX2 loop only runs when simd_n ≥ 16, guaranteeing two ctrl bytes).
            *base_ptr.add(ctrl_start + block) = c0;
            *base_ptr.add(ctrl_start + block + 1) = c1;

            // Pack both groups with one VPSHUFB (two independent 128-bit lanes).
            let enc_mask_lo =
                _mm_loadu_si128(ENCODE_TABLE[c0 as usize].as_ptr() as *const __m128i);
            let enc_mask_hi =
                _mm_loadu_si128(ENCODE_TABLE[c1 as usize].as_ptr() as *const __m128i);
            let enc_mask = _mm256_set_m128i(enc_mask_hi, enc_mask_lo);
            let packed = _mm256_shuffle_epi8(v, enc_mask);

            // Store lower lane (c0). Overruns by up to 8 bytes into c1's space;
            // the c1 store below overwrites those bytes with real data.
            // SAFETY: data_start + data_pos + 16 ≤ data_start + 2*n + 16 ≤ capacity.
            let lo128 = _mm256_castsi256_si128(packed);
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, lo128);
            data_pos += 8 + c0.count_ones() as usize;

            // Store upper lane (c1). Overruns at most 8 bytes into the +16 guard.
            let hi128 = _mm256_extracti128_si256(packed, 1);
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, hi128);
            data_pos += 8 + c1.count_ones() as usize;
        }

        block += 2;
    }

    unsafe {
        // SAFETY: elements [data_start, data_start + data_pos) were written above.
        out.set_len(data_start + data_pos);
    }

    // Scalar tail for n % 16 remaining values.
    for j in simd_n..n {
        let v = values[j];
        if v <= 0xFF {
            out.push(v as u8);
        } else {
            // SAFETY: ctrl_start + j/8 < ctrl_start + ctrl_len ≤ out.len().
            out[ctrl_start + j / 8] |= 1u8 << (j % 8);
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Decode `n` u16 values from an SVB16-encoded buffer using AVX2 `VPSHUFB`.
///
/// Processes 16 values (2 ctrl bytes) per iteration.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
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

    while decoded + 16 <= n {
        let c0 = ctrl[ctrl_pos];
        let c1 = ctrl[ctrl_pos + 1];
        let c0_bytes = 8 + c0.count_ones() as usize;

        // The hi-lane load starts at data_pos + c0_bytes and reads 16 bytes.
        if data_pos + c0_bytes + 16 > data_bytes.len() {
            break;
        }

        let result = unsafe {
            // SAFETY: TABLE indices are valid (c0, c1 are u8, so < 256).
            let mask_lo = _mm_loadu_si128(TABLE[c0 as usize].as_ptr() as *const __m128i);
            let mask_hi = _mm_loadu_si128(TABLE[c1 as usize].as_ptr() as *const __m128i);

            // SAFETY: data_pos + c0_bytes + 16 <= data_bytes.len() checked above;
            // lo load: data_pos + 16 ≤ data_pos + c0_bytes + 16 (c0_bytes ≥ 8).
            let chunk_lo = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let chunk_hi =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos + c0_bytes) as *const __m128i);

            // Pack into 256-bit registers: lower 128 = c0 lane, upper 128 = c1 lane.
            let mask256 = _mm256_set_m128i(mask_hi, mask_lo);
            let data256 = _mm256_set_m128i(chunk_hi, chunk_lo);
            _mm256_shuffle_epi8(data256, mask256)
        };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; decoded + 16 <= n.
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m256i;
            _mm256_storeu_si256(out_ptr, result);
        }

        data_pos += c0_bytes + 8 + c1.count_ones() as usize;
        ctrl_pos += 2;
        decoded += 16;
    }

    unsafe {
        // SAFETY: every element in [base, base + decoded) was written above.
        out.set_len(base + decoded);
    }

    // SSE2-style padded tail: guard fired (rem < c0_bytes+16 ≤ 32) but groups of 8 remain.
    // c0_bytes ∈ [8,16]; rem ≤ 31; padded_pos ≤ rem−8 ≤ 23; load [23,39) ⊆ [0,64). ✓
    if decoded + 8 <= n {
        let mut padded = [0u8; 64];
        let rem = data_bytes.len() - data_pos;
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 8 <= n {
            let cb = ctrl[ctrl_pos];
            let result = unsafe {
                // SAFETY: padded is 64 bytes; padded_pos ≤ rem−8 ≤ 23;
                // load [padded_pos, padded_pos+16) ⊆ [0, 39) ⊆ [0, 64).
                let mask = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                _mm_shuffle_epi8(chunk, mask)
            };
            unsafe {
                // SAFETY: out.reserve(n) ensures capacity; decoded + 8 <= n.
                let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
                _mm_storeu_si128(out_ptr, result);
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
