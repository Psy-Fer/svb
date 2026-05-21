// SSSE3 decode path for SVB16.
//
// The file is named sse2 per project convention; the instruction actually used
// is PSHUFB (_mm_shuffle_epi8), which is SSSE3 (Penryn 2007+).
// At runtime, dispatch checks is_x86_feature_detected!("ssse3") before calling.

use core::arch::x86_64::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use super::shuffle::{ENCODE_TABLE, TABLE};
use crate::error::DecodeError;

/// Encode `values` into SVB16 format using SSSE3 `PSHUFB`.
///
/// Processes 8 values per iteration. Uses a 256-entry encode shuffle table to
/// pack variable-width bytes in a single instruction. Stores 16 bytes per
/// iteration (overwriting into the +16-byte guard in the reserved capacity) and
/// advances the data pointer by the exact byte count for that ctrl byte.
/// Remaining values (n % 8) are handled by the scalar path.
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
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
        // These are pure-computation intrinsics, safe within #[target_feature(enable="ssse3")].
        let hi = _mm_srli_epi16(v, 8);
        let needs_two = _mm_cmpgt_epi16(hi, _mm_setzero_si128());
        let ctrl = _mm_movemask_epi8(_mm_packs_epi16(needs_two, needs_two)) as u8;

        unsafe {
            // SAFETY: ctrl_start + block < ctrl_start + ctrl_len <= out.len().
            *base_ptr.add(ctrl_start + block) = ctrl;

            // Shuffle input bytes into packed output order, then store 16 bytes.
            // SAFETY: ENCODE_TABLE[ctrl] is 16 bytes; ctrl < 256 (u8).
            let mask = _mm_loadu_si128(ENCODE_TABLE[ctrl as usize].as_ptr() as *const __m128i);
            let packed = _mm_shuffle_epi8(v, mask);
            // SAFETY: data_start + data_pos + 16 <= data_start + 2*n + 16 <= out.capacity().
            _mm_storeu_si128(base_ptr.add(data_start + data_pos) as *mut __m128i, packed);
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
            // SAFETY: ctrl_start + j/8 < ctrl_start + ctrl_len <= new out.len()
            //         (ctrl bytes are always within the initialized region).
            out[ctrl_start + j / 8] |= 1u8 << (j % 8);
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Decode `n` u16 values from an SVB16-encoded buffer using SSSE3 `PSHUFB`.
///
/// Processes 8 values per ctrl byte. Falls back to the scalar path for any
/// trailing values when fewer than 16 data bytes remain (preventing an
/// out-of-bounds read on the unaligned 128-bit load).
///
/// # Safety
/// The executing CPU must support SSSE3.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
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

        // Guard against an out-of-bounds unaligned load at the end of the buffer.
        // The maximum data bytes for 8 values is 16 (all 2-byte). If fewer than
        // 16 bytes remain we fall through to the scalar tail.
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
            // SAFETY: out.reserve(n) ensures capacity; decoded + 8 <= n, so
            // base + decoded + 8 <= base + n <= out.capacity().
            let out_ptr = out.as_mut_ptr().add(base + decoded) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
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
        unsafe {
            out.set_len(base + decoded);
        }
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
