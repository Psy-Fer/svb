// SSSE3 decode path for SVB16.
//
// The file is named sse2 per project convention; the instruction actually used
// is PSHUFB (_mm_shuffle_epi8), which is SSSE3 (Penryn 2007+).
// At runtime, dispatch checks is_x86_feature_detected!("ssse3") before calling.

use core::arch::x86_64::*;

use super::shuffle::{ENCODE_TABLE, TABLE};
use crate::error::DecodeError;

/// Encode `values` into `out` using SSSE3 `PSHUFB` and return the number of
/// bytes written.
///
/// Processes 8 values per iteration. Uses a 256-entry encode shuffle table to
/// pack variable-width bytes in a single instruction. Each iteration stores 16
/// bytes and advances the data position by the real byte count, so a store can
/// write past the data encoded so far. It never writes past
/// `ctrl_len + 2 * values.len()` (see the SAFETY comment on the store).
/// Remaining values (n % 8) are handled by the scalar path.
///
/// # Safety
/// The executing CPU must support SSSE3. `out.len()` must be at least
/// `ctrl_len + 2 * values.len()`.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn encode_to_slice(values: &[u16], out: &mut [u8]) -> usize {
    let n = values.len();
    let ctrl_len = n.div_ceil(8);
    // Zero the ctrl bytes so the scalar tail can OR into them.
    out[..ctrl_len].fill(0);

    let simd_n = (n / 8) * 8;
    debug_assert!(out.len() >= ctrl_len + 2 * n);
    let base_ptr = out.as_mut_ptr();
    let mut data_pos = 0usize;

    let mut block = 0usize;
    while block * 8 < simd_n {
        let i = block * 8;

        let v = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; values slice bounds are valid.
            _mm_loadu_si128(values.as_ptr().add(i) as *const __m128i)
        };

        // Compute ctrl byte: bit k = 1 iff high byte of value k is non-zero (value > 255).
        // _mm_srli_epi16 shifts each u16 right by 8, leaving the high byte in bits 7:0.
        // _mm_cmpgt_epi16 on [0,255] values is signed-safe (all non-negative as i16).
        // _mm_packs_epi16 saturates 0xFFFF→0x80, 0x0000→0x00; movemask extracts MSBs.
        let hi = _mm_srli_epi16(v, 8);
        let needs_two = _mm_cmpgt_epi16(hi, _mm_setzero_si128());
        let ctrl = _mm_movemask_epi8(_mm_packs_epi16(needs_two, needs_two)) as u8;

        unsafe {
            // SAFETY: block < ctrl_len <= out.len().
            *base_ptr.add(block) = ctrl;

            // Shuffle input bytes into packed output order, then store 16 bytes.
            // SAFETY: ENCODE_TABLE[ctrl] is 16 bytes; ctrl < 256 (u8).
            let mask = _mm_loadu_si128(ENCODE_TABLE[ctrl as usize].as_ptr() as *const __m128i);
            let packed = _mm_shuffle_epi8(v, mask);
            // SAFETY: the i values before this block took at most 2 bytes each,
            // so data_pos <= 2 * i, and i + 8 <= n. The store ends at
            // ctrl_len + data_pos + 16 <= ctrl_len + 2 * (i + 8) <= ctrl_len + 2 * n.
            _mm_storeu_si128(base_ptr.add(ctrl_len + data_pos) as *mut __m128i, packed);
        }

        data_pos += 8 + ctrl.count_ones() as usize;
        block += 1;
    }

    super::scalar::encode_tail(values, block * 8, out, ctrl_len, data_pos)
}

/// Decode `out.len()` u16 values from an SVB16-encoded buffer using SSSE3
/// `PSHUFB`.
///
/// Processes 8 values per ctrl byte. Falls back to the scalar path for any
/// trailing values when fewer than 16 data bytes remain (preventing an
/// out-of-bounds read on the unaligned 128-bit load).
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
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

    while decoded + 8 <= n {
        let cb = ctrl[ctrl_pos];
        let bytes_consumed = 8 + cb.count_ones() as usize;

        // Guard against an out-of-bounds unaligned load at the end of the buffer.
        // The maximum data bytes for 8 values is 16 (all 2-byte). If fewer than
        // 16 bytes remain we fall through to the padded tail.
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
            // SAFETY: decoded + 8 <= n = out.len().
            _mm_storeu_si128(out_ptr.add(decoded) as *mut __m128i, result);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 8;
    }

    // Padded tail: for well-formed input, guard fires (rem < 16) with complete
    // groups of 8 remaining, fitting a zero-padded 32-byte buffer. `rem` and
    // each iteration's `consumed` are still re-validated below, since a
    // truncated/corrupted `data` (mismatched against the declared `n`) can't
    // be trusted to satisfy that bound.
    if decoded + 8 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated { index: decoded });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
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
