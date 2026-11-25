//! Fused SVB-ZD decoder: U32Classic + inverse-zigzag + delta in one SIMD pass.
//!
//! SVB-ZD is the signal compression method used in hasindu2008's BLOW5/slow5lib.
//! Pipeline: i16 samples → widen to i32 → fused zigzag-delta → U32Classic → bytes.
//!
//! This fused decoder collapses U32Classic decode, inverse-zigzag, and delta
//! prefix sum into one SIMD loop. U32Classic and zigzag work execute during the
//! delta carry-chain stall, hiding most of their cost.
//!
//! The decode kernels write into a slice (`n = out.len()`); the `Vec` functions
//! grow the `Vec`, decode into the new space and truncate it again on error.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

// ── encode: public entry points ───────────────────────────────────────────────

/// Worst-case encoded size for `n` samples: `ceil(n / 4)` control bytes plus 4
/// data bytes per sample, the same as `U32Classic`.
pub(crate) const fn max_encoded_len(n: usize) -> usize {
    crate::u32::U32Classic.max_encoded_len(n)
}

pub fn encode_into(samples: &[i16], out: &mut Vec<u8>) {
    let start = out.len();
    out.resize(start + max_encoded_len(samples.len()), 0);
    let written = encode_to_slice(samples, &mut out[start..]);
    out.truncate(start + written);
}

/// Encode `samples` into the start of `out` and return the number of bytes
/// written. `out` must be at least `max_encoded_len(samples.len())` bytes.
pub(crate) fn encode_to_slice(samples: &[i16], out: &mut [u8]) -> usize {
    assert!(out.len() >= max_encoded_len(samples.len()));
    if samples.is_empty() {
        return 0;
    }
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 declares AVX2 available at compile time.
        return unsafe { encode_avx2(samples, out) };
    }
    #[cfg(all(
        feature = "simd-ssse3",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-ssse3 declares SSSE3 available at compile time.
        return unsafe { encode_ssse3(samples, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { encode_neon(samples, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon"))
    ))]
    {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: runtime check confirms AVX2.
                return unsafe { encode_avx2(samples, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: runtime check confirms SSSE3.
                return unsafe { encode_ssse3(samples, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { encode_neon(samples, out) };
        }
    }
    encode_scalar(samples, out)
}

fn encode_scalar(samples: &[i16], out: &mut [u8]) -> usize {
    let ctrl_len = samples.len().div_ceil(4);
    let (ctrl, data) = out.split_at_mut(ctrl_len);
    ctrl_len + encode_tail(samples, 0, 0, ctrl, data, 0)
}

// Scalar fused zigzag-delta + U32Classic encode of `samples[start..]`, with
// `prev` the sample before `start` and `data_pos` the data bytes written so
// far. Used for the whole input by `encode_scalar` and for the last 0-7 values
// by the SIMD kernels. Clears the control bytes it writes to, since `out` may
// hold old data. Returns the new `data_pos`.
fn encode_tail(
    samples: &[i16],
    start: usize,
    mut prev: i32,
    ctrl: &mut [u8],
    data: &mut [u8],
    mut data_pos: usize,
) -> usize {
    let n = samples.len();
    ctrl[start / 4..n.div_ceil(4)].fill(0);
    for (j, &s) in samples.iter().enumerate().skip(start) {
        let v = s as i32;
        let delta = v.wrapping_sub(prev);
        let zz = ((delta << 1) ^ (delta >> 31)) as u32;
        prev = v;
        let (tag, count): (u8, usize) = if zz <= 0xFF {
            (0, 1)
        } else if zz <= 0xFFFF {
            (1, 2)
        } else if zz <= 0xFF_FFFF {
            (2, 3)
        } else {
            (3, 4)
        };
        ctrl[j / 4] |= tag << ((j % 4) * 2);
        data[data_pos..data_pos + count].copy_from_slice(&zz.to_le_bytes()[..count]);
        data_pos += count;
    }
    data_pos
}

// Overrun bound for the SIMD kernels below: each value takes at most 4 data
// bytes, so a group of 4 starting at value `i` is stored at `data_pos <= 4 * i`.
// Its 16-byte store ends at `<= 4 * i + 16 <= 4 * n`, since `i + 4 <= n`, and
// `data` holds `4 * n` bytes (see `max_encoded_len`).

// ── encode: AVX2 / x86_64 ────────────────────────────────────────────────────

// Fused zigzag-delta + U32Classic StreamVByte encode, 8 i16s per iteration.
//
// _mm256_cvtepi16_epi32 widens 8 i16→i32 in one instruction.
// _mm_alignr_epi8 builds the "previous-sample" vector per 128-bit half:
//   low half:  [prev, s0, s1, s2]  via alignr(curr_lo, set1(prev), 12)
//   high half: [s3,   s4, s5, s6]  via alignr(curr_hi, curr_lo,   12)
// Zigzag = (delta<<1) ^ (delta>>31) using 256-bit shifts — no bias trick needed.
// StreamVByte packing via ENCODE_TABLE_CLASSIC (same as avx2::encode_into_classic).
#[allow(dead_code)]
#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-auto"),
    target_arch = "x86_64"
))]
#[target_feature(enable = "avx2")]
unsafe fn encode_avx2(samples: &[i16], out: &mut [u8]) -> usize {
    use crate::u32::shuffle::{DATA_LEN, ENCODE_TABLE_CLASSIC};
    use core::arch::x86_64::*;

    let n = samples.len();
    let ctrl_len = n.div_ceil(4);
    let (ctrl, data) = out.split_at_mut(ctrl_len);
    let data_ptr = data.as_mut_ptr();

    let simd_n = (n / 8) * 8;
    let mut data_pos = 0usize;
    let mut prev: i32 = 0;

    let bias = _mm256_set1_epi32(i32::MIN);
    let t1 = _mm256_set1_epi32(i32::MIN + 0xFF);
    let t2 = _mm256_set1_epi32(i32::MIN + 0xFFFF);
    let t3 = _mm256_set1_epi32(i32::MIN + 0xFF_FFFF);
    let zero256 = _mm256_setzero_si256();
    let gather_lo = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);

    let mut i = 0usize;
    let mut block = 0usize;

    while i + 8 <= simd_n {
        // SAFETY: i + 8 <= n; 16-byte load of 8 i16s is within the slice.
        let raw_i16 = unsafe { _mm_loadu_si128(samples.as_ptr().add(i) as *const __m128i) };
        let curr = _mm256_cvtepi16_epi32(raw_i16); // 8 × i32

        // Build prev_shifted = [prev,s0,s1,s2 | s3,s4,s5,s6] using per-half alignr.
        let lo = _mm256_castsi256_si128(curr); // [s0,s1,s2,s3]
        let hi = _mm256_extracti128_si256(curr, 1); // [s4,s5,s6,s7]
        let prev_lo = _mm_alignr_epi8(lo, _mm_set1_epi32(prev), 12); // [prev,s0,s1,s2]
        let prev_hi = _mm_alignr_epi8(hi, lo, 12); // [s3,s4,s5,s6]
        let prev_shifted = _mm256_inserti128_si256(_mm256_castsi128_si256(prev_lo), prev_hi, 1);

        let delta = _mm256_sub_epi32(curr, prev_shifted);
        let zigzag = _mm256_xor_si256(_mm256_slli_epi32(delta, 1), _mm256_srai_epi32(delta, 31));

        // Carry: last element of hi (= s7).
        prev = _mm_cvtsi128_si32(_mm_srli_si128(hi, 12));

        // Tag computation via signed bias (unsigned compare on i32).
        let bv = _mm256_add_epi32(zigzag, bias);
        let c1m = _mm256_cmpgt_epi32(bv, t1);
        let c2m = _mm256_cmpgt_epi32(bv, t2);
        let c3m = _mm256_cmpgt_epi32(bv, t3);
        let tag_vec = _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_sub_epi32(zero256, c1m),
                _mm256_sub_epi32(zero256, c2m),
            ),
            _mm256_sub_epi32(zero256, c3m),
        );

        let tlo = _mm256_castsi256_si128(tag_vec);
        let thi = _mm256_extracti128_si256(tag_vec, 1);
        let raw_lo = _mm_cvtsi128_si32(_mm_shuffle_epi8(tlo, gather_lo)) as u32;
        let c0 = ((raw_lo & 0x3)
            | ((raw_lo >> 6) & 0x0C)
            | ((raw_lo >> 12) & 0x30)
            | ((raw_lo >> 18) & 0xC0)) as u8;
        let raw_hi = _mm_cvtsi128_si32(_mm_shuffle_epi8(thi, gather_lo)) as u32;
        let c1b = ((raw_hi & 0x3)
            | ((raw_hi >> 6) & 0x0C)
            | ((raw_hi >> 12) & 0x30)
            | ((raw_hi >> 18) & 0xC0)) as u8;

        // Raw writes: bounds-checked indexing here made the kernel 5-10% slower.
        unsafe {
            // SAFETY: block = i / 4 and i + 8 <= n, so block + 1 < ctrl_len.
            *ctrl.as_mut_ptr().add(block) = c0;
            *ctrl.as_mut_ptr().add(block + 1) = c1b;
            let v_lo = _mm256_castsi256_si128(zigzag);
            let enc_lo =
                _mm_loadu_si128(ENCODE_TABLE_CLASSIC[c0 as usize].as_ptr() as *const __m128i);
            // SAFETY: data_pos + 16 <= data.len(), see the overrun bound above.
            _mm_storeu_si128(
                data_ptr.add(data_pos) as *mut __m128i,
                _mm_shuffle_epi8(v_lo, enc_lo),
            );
            data_pos += DATA_LEN[c0 as usize] as usize;

            let v_hi = _mm256_extracti128_si256(zigzag, 1);
            let enc_hi =
                _mm_loadu_si128(ENCODE_TABLE_CLASSIC[c1b as usize].as_ptr() as *const __m128i);
            // SAFETY: as above, for the group starting at value i + 4.
            _mm_storeu_si128(
                data_ptr.add(data_pos) as *mut __m128i,
                _mm_shuffle_epi8(v_hi, enc_hi),
            );
            data_pos += DATA_LEN[c1b as usize] as usize;
        }

        block += 2;
        i += 8;
    }

    // Scalar tail: the remaining values.
    ctrl_len + encode_tail(samples, i, prev, ctrl, data, data_pos)
}

// ── encode: SSSE3 / x86_64 ───────────────────────────────────────────────────

// Fused zigzag-delta + U32Classic encode, 4 i16s per iteration.
//
// Sign-extension without SSE4.1: _mm_srai_epi16(raw,15) produces the sign word
// (0x0000 or 0xFFFF) for each i16; _mm_unpacklo_epi16(raw, signs) interleaves
// [s0_lo, s0_hi, s1_lo, s1_hi, ...] which reinterpreted as i32 gives the
// sign-extended values.
//
// _mm_alignr_epi8(a, b, 12) = [b[12..15], a[0..11]] in bytes
//   = [b[3] as i32, a[0], a[1], a[2]] = [prev, s0, s1, s2] when b = set1(prev).
#[allow(dead_code)]
#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
#[target_feature(enable = "ssse3")]
unsafe fn encode_ssse3(samples: &[i16], out: &mut [u8]) -> usize {
    use crate::u32::shuffle::{DATA_LEN, ENCODE_TABLE_CLASSIC};
    use core::arch::x86_64::*;

    let n = samples.len();
    let ctrl_len = n.div_ceil(4);
    let (ctrl, data) = out.split_at_mut(ctrl_len);
    let data_ptr = data.as_mut_ptr();

    let simd_n = (n / 4) * 4;
    let mut data_pos = 0usize;
    let mut prev: i32 = 0;

    let bias = _mm_set1_epi32(i32::MIN);
    let t1 = _mm_set1_epi32(i32::MIN + 0xFF);
    let t2 = _mm_set1_epi32(i32::MIN + 0xFFFF);
    let t3 = _mm_set1_epi32(i32::MIN + 0xFF_FFFF);
    let zero = _mm_setzero_si128();
    let gather_lo = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0);

    let mut i = 0usize;
    let mut block = 0usize;

    while i + 4 <= simd_n {
        // Load 4 i16s (8 bytes) via unaligned u64 read; upper 8 bytes zeroed.
        // SAFETY: i + 4 <= n → at least 8 more bytes available in the slice.
        let lo = unsafe { (samples.as_ptr().add(i) as *const u64).read_unaligned() };
        let raw = _mm_set_epi64x(0, lo as i64);

        // Sign-extend 4 × i16 → 4 × i32 without SSE4.1.
        let signs = _mm_srai_epi16(raw, 15); // 0x0000 or 0xFFFF per lane
        let curr = _mm_unpacklo_epi16(raw, signs); // 4 × i32

        // [prev, s0, s1, s2] = alignr(curr, set1(prev), 12).
        let prev_shifted = _mm_alignr_epi8(curr, _mm_set1_epi32(prev), 12);
        let delta = _mm_sub_epi32(curr, prev_shifted);
        let zigzag = _mm_xor_si128(_mm_slli_epi32(delta, 1), _mm_srai_epi32(delta, 31));

        prev = _mm_cvtsi128_si32(_mm_srli_si128(curr, 12));

        let bv = _mm_add_epi32(zigzag, bias);
        let c1m = _mm_cmpgt_epi32(bv, t1);
        let c2m = _mm_cmpgt_epi32(bv, t2);
        let c3m = _mm_cmpgt_epi32(bv, t3);
        let tag_vec = _mm_add_epi32(
            _mm_add_epi32(_mm_sub_epi32(zero, c1m), _mm_sub_epi32(zero, c2m)),
            _mm_sub_epi32(zero, c3m),
        );
        let tags = _mm_cvtsi128_si32(_mm_shuffle_epi8(tag_vec, gather_lo)) as u32;
        let cb =
            ((tags & 0x3) | ((tags >> 6) & 0x0C) | ((tags >> 12) & 0x30) | ((tags >> 18) & 0xC0))
                as u8;

        unsafe {
            // SAFETY: block = i / 4 and i + 4 <= n, so block < ctrl_len.
            *ctrl.as_mut_ptr().add(block) = cb;
            let enc = _mm_loadu_si128(ENCODE_TABLE_CLASSIC[cb as usize].as_ptr() as *const __m128i);
            // SAFETY: data_pos + 16 <= data.len(), see the overrun bound above.
            _mm_storeu_si128(
                data_ptr.add(data_pos) as *mut __m128i,
                _mm_shuffle_epi8(zigzag, enc),
            );
        }

        data_pos += DATA_LEN[cb as usize] as usize;
        block += 1;
        i += 4;
    }

    // Scalar tail: the remaining values.
    ctrl_len + encode_tail(samples, i, prev, ctrl, data, data_pos)
}

// ── encode: NEON / AArch64 ───────────────────────────────────────────────────

// Fused zigzag-delta + U32Classic encode, 4 i16s per iteration.
//
// vmovl_s16 widens 4 i16 → 4 i32 (sign-extended).
// vextq_s32(vdupq_n_s32(prev), curr, 3) extracts elements [3..7) of
//   concat([prev,prev,prev,prev], [s0,s1,s2,s3]) = [prev, s0, s1, s2].
// vcgtq_u32 for unsigned tag comparison (no bias trick needed).
// vqtbl1q_u8 for data packing (same as neon::encode_into_classic).
#[allow(dead_code)]
#[cfg(all(
    any(feature = "simd-neon", feature = "simd-auto"),
    target_arch = "aarch64"
))]
#[target_feature(enable = "neon")]
unsafe fn encode_neon(samples: &[i16], out: &mut [u8]) -> usize {
    use crate::u32::shuffle::{DATA_LEN, ENCODE_TABLE_CLASSIC};
    use core::arch::aarch64::*;

    let n = samples.len();
    let ctrl_len = n.div_ceil(4);
    let (ctrl, data) = out.split_at_mut(ctrl_len);
    let data_ptr = data.as_mut_ptr();

    let simd_n = (n / 4) * 4;
    let mut data_pos = 0usize;
    let mut prev: i32 = 0;

    // SAFETY: array literal; no alignment requirement.
    let weights = unsafe { vld1_u8([1u8, 4, 16, 64, 0, 0, 0, 0].as_ptr()) };

    let mut i = 0usize;
    let mut block = 0usize;

    while i + 4 <= simd_n {
        // SAFETY: i + 4 <= n; vld1_s16 loads 4 × i16 = 8 bytes.
        let raw_s16 = unsafe { vld1_s16(samples.as_ptr().add(i)) };
        let curr = vmovl_s16(raw_s16); // int32x4_t

        // prev_shifted = [prev, s0, s1, s2].
        let prev_shifted = vextq_s32(vdupq_n_s32(prev), curr, 3);
        let delta = vsubq_s32(curr, prev_shifted);

        // Zigzag32: (delta << 1) ^ (delta >> 31).
        let delta_u32 = vreinterpretq_u32_s32(delta);
        let zigzag = veorq_u32(
            vshlq_n_u32::<1>(delta_u32),
            vreinterpretq_u32_s32(vshrq_n_s32::<31>(delta)),
        );

        prev = vgetq_lane_s32(curr, 3);

        let gt255 = vcgtq_u32(zigzag, vdupq_n_u32(0xFF));
        let gt65535 = vcgtq_u32(zigzag, vdupq_n_u32(0xFFFF));
        let gt16m = vcgtq_u32(zigzag, vdupq_n_u32(0xFF_FFFF));
        let tag_vec = vaddq_u32(
            vaddq_u32(vshrq_n_u32::<31>(gt255), vshrq_n_u32::<31>(gt65535)),
            vshrq_n_u32::<31>(gt16m),
        );
        let tag16 = vmovn_u32(tag_vec);
        let tag8 = vmovn_u16(vcombine_u16(tag16, vdup_n_u16(0)));
        let cb = vaddv_u8(vmul_u8(tag8, weights));

        unsafe {
            // SAFETY: block = i / 4 and i + 4 <= n, so block < ctrl_len.
            *ctrl.as_mut_ptr().add(block) = cb;
            let mask = vld1q_u8(ENCODE_TABLE_CLASSIC[cb as usize].as_ptr());
            // SAFETY: data_pos + 16 <= data.len(), see the overrun bound above.
            vst1q_u8(
                data_ptr.add(data_pos),
                vqtbl1q_u8(vreinterpretq_u8_u32(zigzag), mask),
            );
        }

        data_pos += DATA_LEN[cb as usize] as usize;
        block += 1;
        i += 4;
    }

    // Scalar tail: the remaining values.
    ctrl_len + encode_tail(samples, i, prev, ctrl, data, data_pos)
}

// ── decode: public entry points ───────────────────────────────────────────────

pub fn decode_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
    decode_from_into(data, n, 0, out)
}

/// Decode an SVB-ZD stream with an explicit starting carry value.
///
/// `initial` is the i32 accumulator value from the end of the previous stream
/// (0 for the first stream).
pub fn decode_from_into(
    data: &[u8],
    n: usize,
    initial: i32,
    out: &mut Vec<i16>,
) -> Result<(), DecodeError> {
    let start = out.len();
    out.resize(start + n, 0);
    let result = decode_from_to_slice(data, initial, &mut out[start..]);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Decode `out.len()` values of an SVB-ZD stream into `out`, starting from `initial`.
pub(crate) fn decode_from_to_slice(
    data: &[u8],
    initial: i32,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    let n = out.len();
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
    decode_parts_to_slice(&data[..ctrl_len], &data[ctrl_len..], initial, out)
}

/// Decode a pre-split SVB-ZD stream (`ctrl` and `data_bytes` already
/// separated) into `out`, which holds `out.len()` values. `ctrl` must hold at
/// least `out.len().div_ceil(4)` bytes.
fn decode_parts_to_slice(
    ctrl: &[u8],
    data_bytes: &[u8],
    initial: i32,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    if out.is_empty() {
        return Ok(());
    }
    assert!(ctrl.len() >= out.len().div_ceil(4));

    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-ssse3"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSSE3 is required for pshufb; simd-ssse3/simd-avx2 features
        // declare it available at compile time.
        return unsafe { decode_ssse3(ctrl, data_bytes, initial, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, initial, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        feature = "std",
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: runtime check confirms SSSE3.
        if is_x86_feature_detected!("ssse3") {
            return unsafe { decode_ssse3(ctrl, data_bytes, initial, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { decode_neon(ctrl, data_bytes, initial, out) };
    }
    decode_scalar(ctrl, data_bytes, initial, out, 0)
}

// ── scalar fallback ───────────────────────────────────────────────────────────

// Decodes `out.len()` values. Also used for the SIMD kernels' last 0-3 values,
// with `first_index` set to where the tail starts so that `DataTruncated`
// reports the index in the whole output.
fn decode_scalar(
    ctrl: &[u8],
    data: &[u8],
    initial: i32,
    out: &mut [i16],
    first_index: usize,
) -> Result<(), DecodeError> {
    const WIDTHS: [usize; 4] = [1, 2, 3, 4];
    let mut acc = initial;
    let mut data_pos = 0usize;
    for (i, o) in out.iter_mut().enumerate() {
        let tag = ((ctrl[i / 4] >> (2 * (i % 4))) & 3) as usize;
        let width = WIDTHS[tag];
        if data_pos + width > data.len() {
            return Err(DecodeError::DataTruncated {
                index: first_index + i,
            });
        }
        let raw = match width {
            1 => data[data_pos] as u32,
            2 => u16::from_le_bytes([data[data_pos], data[data_pos + 1]]) as u32,
            3 => u32::from_le_bytes([data[data_pos], data[data_pos + 1], data[data_pos + 2], 0]),
            _ => u32::from_le_bytes([
                data[data_pos],
                data[data_pos + 1],
                data[data_pos + 2],
                data[data_pos + 3],
            ]),
        };
        data_pos += width;
        // Inverse zigzag32: (raw >> 1) ^ -(raw & 1)
        let delta = ((raw >> 1) as i32) ^ -((raw & 1) as i32);
        acc = acc.wrapping_add(delta);
        *o = acc as i16;
    }
    Ok(())
}

// ── SSSE3 / x86_64 ───────────────────────────────────────────────────────────

// Truncate 4×i32 → 4×i16: select low 2 bytes of each i32 lane (wrapping cast).
// Bytes [0,1] ← i32[0]; [2,3] ← i32[1]; [4,5] ← i32[2]; [6,7] ← i32[3].
// Upper 8 bytes zeroed (0x80 = PSHUFB zero sentinel).
#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
// Unused under simd-auto without std, which has no runtime detection.
#[allow(dead_code)]
static TRUNC_I32_I16: [u8; 16] = [
    0, 1, 4, 5, 8, 9, 12, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
];

#[cfg(all(
    any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
    target_arch = "x86_64"
))]
// Unused under simd-auto without std, which has no runtime detection.
#[allow(dead_code)]
#[target_feature(enable = "ssse3")]
unsafe fn decode_ssse3(
    ctrl: &[u8],
    data_bytes: &[u8],
    initial: i32,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    use crate::u32::shuffle::{DATA_LEN, TABLE};
    use core::arch::x86_64::*;

    let n = out.len();
    let out_ptr = out.as_mut_ptr();

    // SAFETY: TRUNC_I32_I16 is 16 bytes; SSSE3 loadu has no alignment requirement.
    let trunc_mask = unsafe { _mm_loadu_si128(TRUNC_I32_I16.as_ptr() as *const __m128i) };

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;
    let mut acc = initial;

    // ── fast path: 2 ctrl bytes (8 values) per iteration ────────────────────
    while decoded + 8 <= n {
        if data_pos + 32 > data_bytes.len() {
            break;
        }
        let cb0 = ctrl[ctrl_pos] as usize;
        let cb1 = ctrl[ctrl_pos + 1] as usize;
        let bytes_a = DATA_LEN[cb0] as usize;

        unsafe {
            // SAFETY: data_pos + 32 <= data_bytes.len(); bytes_a <= 16, so
            // the group-B load [data_pos+bytes_a, data_pos+bytes_a+16) ⊆ [data_pos, data_pos+32).
            let shuf_a = _mm_loadu_si128(TABLE[cb0].as_ptr() as *const __m128i);
            let chunk_a = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let u32s_a = _mm_shuffle_epi8(chunk_a, shuf_a);
            let lsb_a = _mm_and_si128(u32s_a, _mm_set1_epi32(1));
            let neg_a = _mm_sub_epi32(_mm_setzero_si128(), lsb_a);
            let delta_a = _mm_xor_si128(_mm_srli_epi32(u32s_a, 1), neg_a);
            let delta_a = _mm_add_epi32(delta_a, _mm_slli_si128(delta_a, 4));
            let delta_a = _mm_add_epi32(delta_a, _mm_slli_si128(delta_a, 8));
            let result_a = _mm_add_epi32(delta_a, _mm_set1_epi32(acc));
            let acc_a = _mm_cvtsi128_si32(_mm_srli_si128(result_a, 12));

            let shuf_b = _mm_loadu_si128(TABLE[cb1].as_ptr() as *const __m128i);
            let chunk_b =
                _mm_loadu_si128(data_bytes.as_ptr().add(data_pos + bytes_a) as *const __m128i);
            let u32s_b = _mm_shuffle_epi8(chunk_b, shuf_b);
            let lsb_b = _mm_and_si128(u32s_b, _mm_set1_epi32(1));
            let neg_b = _mm_sub_epi32(_mm_setzero_si128(), lsb_b);
            let delta_b = _mm_xor_si128(_mm_srli_epi32(u32s_b, 1), neg_b);
            let delta_b = _mm_add_epi32(delta_b, _mm_slli_si128(delta_b, 4));
            let delta_b = _mm_add_epi32(delta_b, _mm_slli_si128(delta_b, 8));
            let result_b = _mm_add_epi32(delta_b, _mm_set1_epi32(acc_a));
            acc = _mm_cvtsi128_si32(_mm_srli_si128(result_b, 12));

            // Pack 8×i32 → 8×i16 (wrapping truncation) into one 128-bit store.
            let packed_a = _mm_shuffle_epi8(result_a, trunc_mask);
            let packed_b = _mm_shuffle_epi8(result_b, trunc_mask);
            // SAFETY: decoded + 8 <= n = out.len().
            _mm_storeu_si128(
                out_ptr.add(decoded) as *mut __m128i,
                _mm_unpacklo_epi64(packed_a, packed_b),
            );
        }

        let bytes_b = DATA_LEN[cb1] as usize;
        data_pos += bytes_a + bytes_b;
        ctrl_pos += 2;
        decoded += 8;
    }

    // ── single-group fast path: 1 ctrl byte (4 values), ≥16 data bytes ───────
    while decoded + 4 <= n && data_pos + 16 <= data_bytes.len() {
        let cb = ctrl[ctrl_pos] as usize;
        let bytes_consumed = DATA_LEN[cb] as usize;

        unsafe {
            // SAFETY: data_pos + 16 <= data_bytes.len(); TABLE[cb] is 16 bytes.
            let shuf = _mm_loadu_si128(TABLE[cb].as_ptr() as *const __m128i);
            let chunk = _mm_loadu_si128(data_bytes.as_ptr().add(data_pos) as *const __m128i);
            let u32s = _mm_shuffle_epi8(chunk, shuf);
            let lsb = _mm_and_si128(u32s, _mm_set1_epi32(1));
            let neg = _mm_sub_epi32(_mm_setzero_si128(), lsb);
            let delta = _mm_xor_si128(_mm_srli_epi32(u32s, 1), neg);
            let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 4));
            let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 8));
            let result = _mm_add_epi32(delta, _mm_set1_epi32(acc));
            acc = _mm_cvtsi128_si32(_mm_srli_si128(result, 12));
            let packed = _mm_shuffle_epi8(result, trunc_mask);
            // SAFETY: decoded + 4 <= n = out.len().
            _mm_storel_epi64(out_ptr.add(decoded) as *mut __m128i, packed);
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 4;
    }

    // ── padded tail: for well-formed input, guard fired with full groups of 4
    // still remaining, fitting a zero-padded 32-byte buffer. `rem` and each
    // iteration's `bytes_consumed` are still re-validated below, since a
    // truncated/corrupted `data` (mismatched against the declared `n`) can't
    // be trusted to satisfy that bound. ─────────────────────────────────────
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated { index: decoded });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = DATA_LEN[cb as usize] as usize;
            if padded_pos + bytes_consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated { index: decoded });
            }

            unsafe {
                // SAFETY: padded_pos + 16 <= padded.len() checked above.
                let shuf = _mm_loadu_si128(TABLE[cb as usize].as_ptr() as *const __m128i);
                let chunk = _mm_loadu_si128(padded.as_ptr().add(padded_pos) as *const __m128i);
                let u32s = _mm_shuffle_epi8(chunk, shuf);

                let lsb = _mm_and_si128(u32s, _mm_set1_epi32(1));
                let neg = _mm_sub_epi32(_mm_setzero_si128(), lsb);
                let delta = _mm_xor_si128(_mm_srli_epi32(u32s, 1), neg);

                let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 4));
                let delta = _mm_add_epi32(delta, _mm_slli_si128(delta, 8));
                let result = _mm_add_epi32(delta, _mm_set1_epi32(acc));

                acc = _mm_cvtsi128_si32(_mm_srli_si128(result, 12));

                let packed = _mm_shuffle_epi8(result, trunc_mask);
                _mm_storel_epi64(out_ptr.add(decoded) as *mut __m128i, packed);
            }

            padded_pos += bytes_consumed;
            data_pos += bytes_consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
    }

    // Scalar tail: 0-3 remaining values.
    decode_scalar(
        &ctrl[ctrl_pos..],
        &data_bytes[data_pos..],
        acc,
        &mut out[decoded..],
        decoded,
    )
}

// ── NEON / AArch64 ───────────────────────────────────────────────────────────

#[cfg(all(
    any(feature = "simd-neon", feature = "simd-auto"),
    target_arch = "aarch64"
))]
#[target_feature(enable = "neon")]
unsafe fn decode_neon(
    ctrl: &[u8],
    data_bytes: &[u8],
    initial: i32,
    out: &mut [i16],
) -> Result<(), DecodeError> {
    use crate::u32::shuffle::{DATA_LEN, TABLE};
    use core::arch::aarch64::*;

    let n = out.len();
    let out_ptr = out.as_mut_ptr();

    let mut ctrl_pos = 0usize;
    let mut data_pos = 0usize;
    let mut decoded = 0usize;
    let mut acc = initial;

    let zero_s32 = vdupq_n_s32(0);

    // ── fast path: 2 ctrl bytes (8 values) per iteration ────────────────────
    while decoded + 8 <= n {
        if data_pos + 32 > data_bytes.len() {
            break;
        }
        let cb0 = ctrl[ctrl_pos] as usize;
        let cb1 = ctrl[ctrl_pos + 1] as usize;
        let bytes_a = DATA_LEN[cb0] as usize;

        unsafe {
            // SAFETY: data_pos + 32 <= data_bytes.len(); bytes_a <= 16, so
            // the group-B load [data_pos+bytes_a, data_pos+bytes_a+16) ⊆ [data_pos, data_pos+32).
            let shuf_a = vld1q_u8(TABLE[cb0].as_ptr());
            let chunk_a = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            let u32s_a = vreinterpretq_u32_u8(vqtbl1q_u8(chunk_a, shuf_a));
            let lsb_a = vandq_u32(u32s_a, vdupq_n_u32(1));
            let neg_a = vsubq_u32(vdupq_n_u32(0), lsb_a);
            let delta_a = vreinterpretq_s32_u32(veorq_u32(vshrq_n_u32(u32s_a, 1), neg_a));
            let delta_a = vaddq_s32(delta_a, vextq_s32(zero_s32, delta_a, 3));
            let delta_a = vaddq_s32(delta_a, vextq_s32(zero_s32, delta_a, 2));
            let result_a = vaddq_s32(delta_a, vdupq_n_s32(acc));
            let acc_a = vgetq_lane_s32(result_a, 3);

            let shuf_b = vld1q_u8(TABLE[cb1].as_ptr());
            let chunk_b = vld1q_u8(data_bytes.as_ptr().add(data_pos + bytes_a));
            let u32s_b = vreinterpretq_u32_u8(vqtbl1q_u8(chunk_b, shuf_b));
            let lsb_b = vandq_u32(u32s_b, vdupq_n_u32(1));
            let neg_b = vsubq_u32(vdupq_n_u32(0), lsb_b);
            let delta_b = vreinterpretq_s32_u32(veorq_u32(vshrq_n_u32(u32s_b, 1), neg_b));
            let delta_b = vaddq_s32(delta_b, vextq_s32(zero_s32, delta_b, 3));
            let delta_b = vaddq_s32(delta_b, vextq_s32(zero_s32, delta_b, 2));
            let result_b = vaddq_s32(delta_b, vdupq_n_s32(acc_a));
            acc = vgetq_lane_s32(result_b, 3);

            // Narrow 8×i32 → 8×i16 (wrapping) and store 16 bytes.
            // SAFETY: decoded + 8 <= n = out.len().
            vst1q_s16(
                out_ptr.add(decoded),
                vcombine_s16(vmovn_s32(result_a), vmovn_s32(result_b)),
            );
        }

        let bytes_b = DATA_LEN[cb1] as usize;
        data_pos += bytes_a + bytes_b;
        ctrl_pos += 2;
        decoded += 8;
    }

    // ── single-group fast path: 1 ctrl byte (4 values), ≥16 data bytes ───────
    while decoded + 4 <= n && data_pos + 16 <= data_bytes.len() {
        let cb = ctrl[ctrl_pos] as usize;
        let bytes_consumed = DATA_LEN[cb] as usize;

        unsafe {
            // SAFETY: data_pos + 16 <= data_bytes.len(); TABLE[cb] is 16 bytes.
            let shuf = vld1q_u8(TABLE[cb].as_ptr());
            let chunk = vld1q_u8(data_bytes.as_ptr().add(data_pos));
            let u32s = vreinterpretq_u32_u8(vqtbl1q_u8(chunk, shuf));
            let lsb = vandq_u32(u32s, vdupq_n_u32(1));
            let neg = vsubq_u32(vdupq_n_u32(0), lsb);
            let delta = vreinterpretq_s32_u32(veorq_u32(vshrq_n_u32(u32s, 1), neg));
            let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 3));
            let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 2));
            let result = vaddq_s32(delta, vdupq_n_s32(acc));
            acc = vgetq_lane_s32(result, 3);
            // SAFETY: decoded + 4 <= n = out.len().
            vst1_s16(out_ptr.add(decoded), vmovn_s32(result));
        }

        data_pos += bytes_consumed;
        ctrl_pos += 1;
        decoded += 4;
    }

    // ── padded tail: for well-formed input, guard fired with full groups of 4
    // still remaining, fitting a zero-padded 32-byte buffer. `rem` and each
    // iteration's `bytes_consumed` are still re-validated below, since a
    // truncated/corrupted `data` (mismatched against the declared `n`) can't
    // be trusted to satisfy that bound. ─────────────────────────────────────
    if decoded + 4 <= n {
        let mut padded = [0u8; 32];
        let rem = data_bytes.len() - data_pos;
        if rem > padded.len() {
            return Err(DecodeError::DataTruncated { index: decoded });
        }
        padded[..rem].copy_from_slice(&data_bytes[data_pos..]);
        let mut padded_pos = 0usize;

        while decoded + 4 <= n {
            let cb = ctrl[ctrl_pos];
            let bytes_consumed = DATA_LEN[cb as usize] as usize;
            if padded_pos + bytes_consumed > rem || padded_pos + 16 > padded.len() {
                return Err(DecodeError::DataTruncated { index: decoded });
            }

            unsafe {
                // SAFETY: padded_pos + 16 <= padded.len() checked above.
                let shuf = vld1q_u8(TABLE[cb as usize].as_ptr());
                let chunk = vld1q_u8(padded.as_ptr().add(padded_pos));
                let u8s = vqtbl1q_u8(chunk, shuf);

                let u32s = vreinterpretq_u32_u8(u8s);
                let lsb = vandq_u32(u32s, vdupq_n_u32(1));
                let neg = vsubq_u32(vdupq_n_u32(0), lsb);
                let shifted = vshrq_n_u32(u32s, 1);
                let delta = vreinterpretq_s32_u32(veorq_u32(shifted, neg));

                let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 3));
                let delta = vaddq_s32(delta, vextq_s32(zero_s32, delta, 2));
                let result = vaddq_s32(delta, vdupq_n_s32(acc));

                acc = vgetq_lane_s32(result, 3);
                let packed = vmovn_s32(result);
                vst1_s16(out_ptr.add(decoded), packed);
            }

            padded_pos += bytes_consumed;
            data_pos += bytes_consumed;
            ctrl_pos += 1;
            decoded += 4;
        }
    }

    // Scalar tail: 0-3 remaining values.
    decode_scalar(
        &ctrl[ctrl_pos..],
        &data_bytes[data_pos..],
        acc,
        &mut out[decoded..],
        decoded,
    )
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    /// Scalar SVB-ZD encode used as the correctness reference in tests.
    fn encode_reference(samples: &[i16]) -> Vec<u8> {
        let n = samples.len();
        let ctrl_len = n.div_ceil(4);
        let mut ctrl = vec![0u8; ctrl_len];
        let mut data: Vec<u8> = Vec::new();
        let mut prev: i32 = 0;
        for (i, &s) in samples.iter().enumerate() {
            let v = s as i32;
            let delta = v.wrapping_sub(prev);
            let zz = ((delta << 1) ^ (delta >> 31)) as u32;
            prev = v;
            let (tag, width): (u8, usize) = if zz <= 0xFF {
                (0, 1)
            } else if zz <= 0xFFFF {
                (1, 2)
            } else if zz <= 0x00FF_FFFF {
                (2, 3)
            } else {
                (3, 4)
            };
            ctrl[i / 4] |= tag << (2 * (i % 4));
            data.extend_from_slice(&zz.to_le_bytes()[..width]);
        }
        let mut out = ctrl;
        out.extend_from_slice(&data);
        out
    }

    #[test]
    fn encode_matches_reference_n4() {
        let samples: Vec<i16> = vec![100, 105, 103, 110];
        assert_eq!(encode_into_vec(&samples), encode_reference(&samples));
    }

    #[test]
    fn encode_matches_reference_n8() {
        let samples: Vec<i16> = (0i16..8).map(|x| x * 100).collect();
        assert_eq!(encode_into_vec(&samples), encode_reference(&samples));
    }

    #[test]
    fn encode_matches_reference_n5() {
        let samples: Vec<i16> = vec![0, -1, 1, -2, 2];
        assert_eq!(encode_into_vec(&samples), encode_reference(&samples));
    }

    #[test]
    fn encode_matches_reference_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX, 0, -1, 1];
        assert_eq!(encode_into_vec(&samples), encode_reference(&samples));
    }

    #[test]
    fn encode_matches_reference_large() {
        let samples: Vec<i16> = (0..256)
            .map(|i| (i as i16 * 13).wrapping_sub(400))
            .collect();
        assert_eq!(encode_into_vec(&samples), encode_reference(&samples));
    }

    #[test]
    fn encode_empty() {
        assert_eq!(encode_into_vec(&[]), Vec::<u8>::new());
    }

    fn encode_into_vec(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_into(samples, &mut out);
        out
    }

    #[test]
    fn roundtrip_n4() {
        let samples: Vec<i16> = vec![100, 105, 103, 110];
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, 4, &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_n8() {
        let samples: Vec<i16> = (0i16..8).map(|x| x * 100).collect();
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, 8, &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_n5() {
        // 1 full SIMD group + 1 scalar tail
        let samples: Vec<i16> = vec![0, -1, 1, -2, 2];
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, 5, &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_n_zero() {
        let mut dec = Vec::new();
        decode_into(&[], 0, &mut dec).unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn roundtrip_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX, 0, -1, 1];
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, samples.len(), &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn roundtrip_large() {
        let samples: Vec<i16> = (0..256)
            .map(|i| (i as i16 * 13).wrapping_sub(400))
            .collect();
        let enc = encode_reference(&samples);
        let mut dec = Vec::new();
        decode_into(&enc, samples.len(), &mut dec).unwrap();
        assert_eq!(dec, samples);
    }

    #[test]
    fn initial_carry_splits() {
        // Encode a block, then decode the second half using a carry from the first.
        let samples: Vec<i16> = (0..32).map(|i| (i as i16 * 50) - 500).collect();
        let enc = encode_reference(&samples);
        let ctrl_len = samples.len().div_ceil(4);
        let ctrl = &enc[..ctrl_len];
        let data = &enc[ctrl_len..];

        // Compute data offset at n=16 (4 ctrl bytes).
        let mid_data_off: usize = ctrl[..4]
            .iter()
            .map(|&cb| crate::u32::shuffle::DATA_LEN[cb as usize] as usize)
            .sum();
        let mid_carry = samples[15] as i32;

        // Decode second half independently.
        let ctrl_b = &ctrl[4..];
        let data_b = &data[mid_data_off..];
        let mut buf_b = ctrl_b.to_vec();
        buf_b.extend_from_slice(data_b);

        let mut out_b = Vec::new();
        decode_from_into(&buf_b, 16, mid_carry, &mut out_b).unwrap();
        assert_eq!(out_b, samples[16..]);
    }

    // ── slice API and cross-path ──────────────────────────────────────────────

    // Zigzag codes of four kinds: all 1-byte, all 2-byte, 1-3 bytes mixed (what
    // encoding i16 samples produces) and all four widths mixed. 4-byte codes
    // only appear in data not made by the encoder, but the decoder accepts them.
    fn codes(n: usize, kind: u8) -> Vec<u32> {
        (0..n as u32)
            .map(|i| match kind {
                0 => i % 200,
                1 => 256 + (i * 1237) % 60000,
                2 => [i % 7, 300 + i, 70000 + i * 31][i as usize % 3],
                _ => (i.wrapping_mul(2_654_435_761)) >> (8 * (i % 4)),
            })
            .collect()
    }

    // Packs codes as U32Classic: ctrl bytes then data bytes.
    fn pack(codes: &[u32]) -> Vec<u8> {
        let mut ctrl = vec![0u8; codes.len().div_ceil(4)];
        let mut data = Vec::new();
        for (i, &c) in codes.iter().enumerate() {
            let width = match c {
                0..=0xFF => 1,
                0x100..=0xFFFF => 2,
                0x1_0000..=0xFF_FFFF => 3,
                _ => 4,
            };
            ctrl[i / 4] |= ((width - 1) as u8) << (2 * (i % 4));
            data.extend_from_slice(&c.to_le_bytes()[..width]);
        }
        ctrl.extend_from_slice(&data);
        ctrl
    }

    fn expected(codes: &[u32], initial: i32) -> Vec<i16> {
        let mut acc = initial;
        codes
            .iter()
            .map(|&zz| {
                acc = acc.wrapping_add(((zz >> 1) as i32) ^ -((zz & 1) as i32));
                acc as i16
            })
            .collect()
    }

    // Runs the scalar decoder and every SIMD kernel this CPU supports on the
    // same input, and checks they agree on success or failure and on output.
    fn check_paths(data: &[u8], n: usize, initial: i32) -> Option<Vec<i16>> {
        let ctrl_len = n.div_ceil(4);
        if data.len() < ctrl_len {
            return None;
        }
        let (ctrl, body) = data.split_at(ctrl_len);
        let mut scalar = vec![0i16; n];
        let scalar_ok = decode_scalar(ctrl, body, initial, &mut scalar, 0).is_ok();

        #[cfg(all(
            any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
            feature = "std",
            target_arch = "x86_64"
        ))]
        if std::is_x86_feature_detected!("ssse3") {
            let mut simd = vec![0i16; n];
            // SAFETY: SSSE3 confirmed at runtime.
            let ok = unsafe { decode_ssse3(ctrl, body, initial, &mut simd) }.is_ok();
            assert_eq!(ok, scalar_ok, "ssse3 ok n={n}");
            if ok {
                assert_eq!(simd, scalar, "ssse3 n={n}");
            }
        }
        #[cfg(all(
            any(feature = "simd-neon", feature = "simd-auto"),
            target_arch = "aarch64"
        ))]
        {
            let mut simd = vec![0i16; n];
            // SAFETY: NEON is mandatory on AArch64.
            let ok = unsafe { decode_neon(ctrl, body, initial, &mut simd) }.is_ok();
            assert_eq!(ok, scalar_ok, "neon ok n={n}");
            if ok {
                assert_eq!(simd, scalar, "neon n={n}");
            }
        }
        scalar_ok.then_some(scalar)
    }

    #[test]
    fn scalar_and_simd_agree_all_lengths() {
        for kind in 0..4 {
            for n in 0..=70usize {
                let c = codes(n, kind);
                let encoded = pack(&c);
                let decoded = check_paths(&encoded, n, 0).expect("valid input");
                assert_eq!(decoded, expected(&c, 0), "kind={kind} n={n}");

                // A nonzero starting carry shifts every output by that carry.
                let carry = 1000;
                let shifted = check_paths(&encoded, n, carry).expect("valid input");
                assert_eq!(shifted, expected(&c, carry), "carry kind={kind} n={n}");

                // Every truncation of the data bytes must fail on every path.
                for cut in n.div_ceil(4)..encoded.len() {
                    assert!(
                        check_paths(&encoded[..cut], n, 0).is_none(),
                        "cut={cut} n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn to_slice_matches_vec_decode() {
        for n in [0usize, 1, 7, 64, 1000] {
            let samples: Vec<i16> = (0..n).map(|i| (i as i16).wrapping_mul(997)).collect();
            let enc = encode_reference(&samples);
            let mut out = vec![0i16; n];
            decode_from_to_slice(&enc, 0, &mut out).unwrap();
            assert_eq!(out, samples, "n={n}");
        }
    }

    #[test]
    fn short_ctrl_stream_is_an_error() {
        let mut out = [0i16; 9];
        let err = decode_from_to_slice(&[0, 0], 0, &mut out).unwrap_err();
        assert!(
            matches!(err, DecodeError::ControlStreamTooShort { need: 3, have: 2 }),
            "{err:?}"
        );
    }

    #[test]
    fn tail_error_index_is_absolute() {
        // All 1-byte codes; dropping the last byte leaves the last value short.
        // Its index is in the scalar tail for these n on every path.
        for n in [13usize, 22, 37] {
            let mut encoded = pack(&codes(n, 0));
            encoded.pop();
            let mut out = vec![0i16; n];
            let err = decode_from_to_slice(&encoded, 0, &mut out).unwrap_err();
            assert!(
                matches!(err, DecodeError::DataTruncated { index } if index == n - 1),
                "n={n}: {err:?}"
            );
        }
    }

    #[test]
    fn vec_wrapper_restores_length_on_error() {
        let mut encoded = pack(&codes(40, 2));
        encoded.pop();
        let mut out = vec![7i16, 8];
        assert!(decode_into(&encoded, 40, &mut out).is_err());
        assert_eq!(out, [7, 8]);
    }

    // ── slice encode ──────────────────────────────────────────────────────────

    // i16 samples giving 1-byte, 2-byte, 3-byte and mixed zigzag codes. Kind 2
    // alternates between i16::MIN and i16::MAX, the largest possible deltas.
    fn encode_samples(n: usize, kind: u8) -> Vec<i16> {
        (0..n)
            .map(|i| match kind {
                0 => (i % 50) as i16 - 25,
                1 => (i as i16).wrapping_mul(1237) % 16000,
                2 => [i16::MIN, i16::MAX][i % 2],
                _ => (i as i16).wrapping_mul(i as i16).wrapping_mul(31),
            })
            .collect()
    }

    // Runs every encode kernel this CPU supports through `encode_checked`,
    // which fills the buffer with canary bytes and checks nothing is written
    // past `max_encoded_len`.
    fn check_encode_paths(samples: &[i16]) -> Vec<u8> {
        let max = max_encoded_len(samples.len());
        let reference = encode_reference(samples);
        let n = samples.len();
        let scalar = crate::test_util::encode_checked(samples, max, encode_scalar);
        assert_eq!(scalar, reference, "scalar n={n}");

        #[cfg(all(
            any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-auto"),
            feature = "std",
            target_arch = "x86_64"
        ))]
        if !samples.is_empty() && std::is_x86_feature_detected!("ssse3") {
            // SAFETY: SSSE3 confirmed at runtime.
            let f = |v: &[i16], o: &mut [u8]| unsafe { encode_ssse3(v, o) };
            let simd = crate::test_util::encode_checked(samples, max, f);
            assert_eq!(simd, reference, "ssse3 n={n}");
        }
        #[cfg(all(
            any(feature = "simd-avx2", feature = "simd-auto"),
            feature = "std",
            target_arch = "x86_64"
        ))]
        if !samples.is_empty() && std::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime.
            let f = |v: &[i16], o: &mut [u8]| unsafe { encode_avx2(v, o) };
            let simd = crate::test_util::encode_checked(samples, max, f);
            assert_eq!(simd, reference, "avx2 n={n}");
        }
        #[cfg(all(
            any(feature = "simd-neon", feature = "simd-auto"),
            target_arch = "aarch64"
        ))]
        if !samples.is_empty() {
            // SAFETY: NEON is mandatory on AArch64.
            let f = |v: &[i16], o: &mut [u8]| unsafe { encode_neon(v, o) };
            let simd = crate::test_util::encode_checked(samples, max, f);
            assert_eq!(simd, reference, "neon n={n}");
        }
        reference
    }

    #[test]
    fn encode_paths_agree_all_lengths() {
        for kind in 0..4 {
            for n in 0..=70usize {
                let samples = encode_samples(n, kind);
                let encoded = check_encode_paths(&samples);
                let mut decoded = vec![0i16; n];
                decode_from_to_slice(&encoded, 0, &mut decoded).unwrap();
                assert_eq!(decoded, samples, "kind={kind} n={n}");
            }
        }
    }

    #[test]
    fn encode_into_appends() {
        let a = encode_samples(13, 3);
        let b = encode_samples(30, 1);
        let mut out = vec![1u8, 2, 3];
        encode_into(&a, &mut out);
        encode_into(&b, &mut out);
        let mut expected = vec![1u8, 2, 3];
        expected.extend(encode_reference(&a));
        expected.extend(encode_reference(&b));
        assert_eq!(out, expected);
    }
}
