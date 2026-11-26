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

use super::shuffle::{ENCODE_TABLE, TABLE};
use crate::error::DecodeError;

/// Encode `values` into `out` using AVX2 and return the number of bytes
/// written.
///
/// Processes 16 values (2 ctrl bytes) per iteration. A single 256-bit load
/// covers both 8-value groups; ctrl bytes c0 and c1 are extracted from one
/// `_mm256_movemask_epi8` call. `_mm256_shuffle_epi8` packs both groups in
/// a single VPSHUFB; each lane is stored independently to its data position.
/// Stores can write past the data encoded so far but never past
/// `ctrl_len + 2 * values.len()` (see the SAFETY comment on the stores).
/// Remaining values (n % 16) are handled by the scalar path.
///
/// # Safety
/// The executing CPU must support AVX2. `out.len()` must be at least
/// `ctrl_len + 2 * values.len()`.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn encode_to_slice(values: &[u16], out: &mut [u8]) -> usize {
    let n = values.len();
    let ctrl_len = n.div_ceil(8);
    // Zero the ctrl bytes so the scalar tail can OR into them.
    out[..ctrl_len].fill(0);

    let simd_n = (n / 16) * 16;
    debug_assert!(out.len() >= ctrl_len + 2 * n);
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
            // SAFETY: i + 16 <= n, so block + 1 = i / 8 + 1 < ctrl_len.
            *base_ptr.add(block) = c0;
            *base_ptr.add(block + 1) = c1;

            // Pack both groups with one VPSHUFB (two independent 128-bit lanes).
            let enc_mask_lo = _mm_loadu_si128(ENCODE_TABLE[c0 as usize].as_ptr() as *const __m128i);
            let enc_mask_hi = _mm_loadu_si128(ENCODE_TABLE[c1 as usize].as_ptr() as *const __m128i);
            let enc_mask = _mm256_set_m128i(enc_mask_hi, enc_mask_lo);
            let packed = _mm256_shuffle_epi8(v, enc_mask);

            // Store lower lane (c0). Overruns by up to 8 bytes into c1's space;
            // the c1 store below overwrites those bytes with real data.
            // SAFETY: the i values before this block took at most 2 bytes
            // each, so data_pos <= 2 * i. The store ends at
            // ctrl_len + data_pos + 16 <= ctrl_len + 2 * (i + 8) <= ctrl_len + 2 * n.
            let lo128 = _mm256_castsi256_si128(packed);
            _mm_storeu_si128(base_ptr.add(ctrl_len + data_pos) as *mut __m128i, lo128);
            data_pos += 8 + c0.count_ones() as usize;

            // Store upper lane (c1).
            // SAFETY: now data_pos <= 2 * (i + 8). The store ends at
            // ctrl_len + data_pos + 16 <= ctrl_len + 2 * (i + 16) <= ctrl_len + 2 * n.
            let hi128 = _mm256_extracti128_si256(packed, 1);
            _mm_storeu_si128(base_ptr.add(ctrl_len + data_pos) as *mut __m128i, hi128);
            data_pos += 8 + c1.count_ones() as usize;
        }

        block += 2;
    }

    super::scalar::encode_tail(values, block * 8, out, ctrl_len, data_pos)
}

/// Decode `out.len()` u16 values from an SVB16-encoded buffer using AVX2
/// `VPSHUFB`.
///
/// Processes 16 values (2 ctrl bytes) per iteration.
///
/// # Safety
/// The executing CPU must support AVX2.
#[allow(dead_code)]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn decode_to_slice(data: &[u8], out: &mut [u16]) -> Result<(), DecodeError> {
    let n = out.len();
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
    let out_ptr = out.as_mut_ptr();

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
            // SAFETY: decoded + 16 <= n = out.len().
            _mm256_storeu_si256(out_ptr.add(decoded) as *mut __m256i, result);
        }

        data_pos += c0_bytes + 8 + c1.count_ones() as usize;
        ctrl_pos += 2;
        decoded += 16;
    }

    // SSE2-style padded tail: for well-formed input, guard fires
    // (rem < c0_bytes+16 ≤ 32) with groups of 8 remaining, fitting a
    // zero-padded 64-byte buffer. `rem` and each iteration's `consumed` are
    // still re-validated below, since a truncated/corrupted `data`
    // (mismatched against the declared `n`) can't be trusted to satisfy that
    // bound.
    if decoded + 8 <= n {
        let mut padded = [0u8; 64];
        // The main loop takes several groups at a time, so it can also stop
        // because it ran out of values, with every byte after the frame still
        // in `data_bytes`. Those bytes are ignored, so copy at most what the
        // buffer holds. tests/trailing_bytes.rs checks that the frame's
        // remaining groups always fit.
        let rem = (data_bytes.len() - data_pos).min(padded.len());
        padded[..rem].copy_from_slice(&data_bytes[data_pos..data_pos + rem]);
        let mut padded_pos = 0usize;

        while decoded + 8 <= n {
            let cb = ctrl[ctrl_pos];
            let consumed = 8 + cb.count_ones() as usize;
            if padded_pos + consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated { index: decoded });
            }
            let result = unsafe {
                // SAFETY: padded_pos + 16 <= padded.len() checked above.
                let mask = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                _mm_shuffle_epi8(chunk, mask)
            };
            unsafe {
                // SAFETY: decoded + 8 <= n = out.len().
                _mm_storeu_si128(out_ptr.add(decoded) as *mut __m128i, result);
            }
            padded_pos += consumed;
            data_pos += consumed;
            ctrl_pos += 1;
            decoded += 8;
        }
    }

    // Scalar for n % 8 remainder (0–7 values).
    if decoded < n {
        super::scalar::decode_from_raw(
            &ctrl[ctrl_pos..],
            &data_bytes[data_pos..],
            &mut out[decoded..],
            decoded,
        )?;
    }

    Ok(())
}
