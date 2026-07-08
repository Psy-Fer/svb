//! PFOR-style patched/exception encoding as a composable layer over `u16` values.
//!
//! Values that fit in a byte (`<= u8::MAX`) are stored as literal bytes, in
//! original stream order. Values that don't fit are pulled out as
//! exceptions: their positions and residual values (`value - 256`) are
//! recorded separately and [`crate::u32::U32Classic`]-encoded. This pays off
//! when exceptions are rare — e.g. the tail of a zigzag-delta signal stream,
//! where most residuals are small but occasional spikes need the full range.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::DecodeError;
use crate::u32::U32Classic;

const THRESHOLD: u16 = u8::MAX as u16;

fn too_short(need: usize, have: usize) -> DecodeError {
    DecodeError::ControlStreamTooShort { need, have }
}

/// Encode `values`, appending the patched/exception representation to `out`.
///
/// # Examples
///
/// ```
/// # use svb::patched;
/// let mut out = Vec::new();
/// patched::encode_into(&[1u16, 300, 2], &mut out);
/// let mut decoded = Vec::new();
/// patched::decode_into(&out, 3, &mut decoded).unwrap();
/// assert_eq!(decoded, [1u16, 300, 2]);
/// ```
pub fn encode_into(values: &[u16], out: &mut Vec<u8>) {
    let mut ex_pos: Vec<u32> = Vec::new();
    let mut ex_val: Vec<u32> = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v > THRESHOLD {
            ex_pos.push(i as u32);
            ex_val.push((v - THRESHOLD - 1) as u32);
        }
    }

    let nex = ex_pos.len() as u32;
    out.extend_from_slice(&nex.to_le_bytes());

    if nex > 1 {
        let mut pos_delta = Vec::with_capacity(ex_pos.len());
        pos_delta.push(ex_pos[0]);
        for w in ex_pos.windows(2) {
            pos_delta.push(w[1] - w[0] - 1);
        }

        let mut pos_bytes = Vec::new();
        U32Classic.encode_into(&pos_delta, &mut pos_bytes);
        out.extend_from_slice(&(pos_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&pos_bytes);

        let mut val_bytes = Vec::new();
        U32Classic.encode_into(&ex_val, &mut val_bytes);
        out.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&val_bytes);
    } else if nex == 1 {
        out.extend_from_slice(&ex_pos[0].to_le_bytes());
        out.extend_from_slice(&ex_val[0].to_le_bytes());
    }

    let mut j = 0;
    for (i, &v) in values.iter().enumerate() {
        if j < ex_pos.len() && i as u32 == ex_pos[j] {
            j += 1;
        } else {
            out.push(v as u8);
        }
    }
}

/// Decode exactly `n` values from the start of `data`, appending them to `out`.
///
/// Returns the number of bytes consumed from `data`. `n` must equal the
/// number of values that were originally encoded, same convention as
/// [`crate::u32::U32Classic::decode`].
///
/// # Examples
///
/// ```
/// # use svb::patched;
/// let mut enc = Vec::new();
/// patched::encode_into(&[1u16, 300, 2], &mut enc);
/// let mut out = Vec::new();
/// let consumed = patched::decode_into(&enc, 3, &mut out).unwrap();
/// assert_eq!(consumed, enc.len());
/// assert_eq!(out, [1u16, 300, 2]);
/// ```
pub fn decode_into(data: &[u8], n: usize, out: &mut Vec<u16>) -> Result<usize, DecodeError> {
    if data.len() < 4 {
        return Err(too_short(4, data.len()));
    }
    let nex = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let mut offset = 4;
    let n_literal = n.saturating_sub(nex);

    if nex == 0 {
        // No exceptions: the literal stream is the entire output, so widen
        // straight into `out` — no intermediate buffer or scatter step to pay
        // for. This is the common case for well-compressing signal (ex-zd's
        // own encoder warns once exceptions exceed ~20%), so it's worth
        // special-casing rather than routing it through the general path
        // below, which would cost an extra allocation and copy for nothing.
        if data.len() < offset + n_literal {
            return Err(too_short(offset + n_literal, data.len()));
        }
        widen_into(&data[offset..offset + n_literal], out);
        return Ok(offset + n_literal);
    }

    // Locate the exception metadata and the literal region (reading only the
    // length prefixes for nex > 1, not decoding the blobs yet) before
    // touching either. This lets the literal widen below — usually the
    // larger of the two independent operations — be issued *before* the
    // U32Classic decode / raw 8-byte read, instead of strictly serializing
    // "decode exceptions, then widen": neither depends on the other's
    // output, only on `data`, so ordering them this way gives the CPU more
    // independent work to overlap.
    let pos_bytes_range;
    let val_bytes_range;
    if nex == 1 {
        if data.len() < offset + 8 {
            return Err(too_short(offset + 8, data.len()));
        }
        pos_bytes_range = offset..offset + 4;
        val_bytes_range = offset + 4..offset + 8;
        offset += 8;
    } else {
        if data.len() < offset + 4 {
            return Err(too_short(offset + 4, data.len()));
        }
        let nex_pos_press = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        if data.len() < offset + nex_pos_press {
            return Err(too_short(offset + nex_pos_press, data.len()));
        }
        pos_bytes_range = offset..offset + nex_pos_press;
        offset += nex_pos_press;

        if data.len() < offset + 4 {
            return Err(too_short(offset + 4, data.len()));
        }
        let nex_press = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        if data.len() < offset + nex_press {
            return Err(too_short(offset + nex_press, data.len()));
        }
        val_bytes_range = offset..offset + nex_press;
        offset += nex_press;
    }

    if data.len() < offset + n_literal {
        return Err(too_short(offset + n_literal, data.len()));
    }
    let mut literal: Vec<u16> = Vec::new();
    widen_into(&data[offset..offset + n_literal], &mut literal);

    let (ex_pos, ex_val): (Vec<u32>, Vec<u32>) = if nex == 1 {
        let p = &data[pos_bytes_range];
        let v = &data[val_bytes_range];
        (
            [u32::from_le_bytes([p[0], p[1], p[2], p[3]])].into(),
            [u32::from_le_bytes([v[0], v[1], v[2], v[3]])].into(),
        )
    } else {
        let mut pos_delta = U32Classic.decode(&data[pos_bytes_range], nex)?;
        for i in 1..pos_delta.len() {
            let prev = pos_delta[i - 1];
            pos_delta[i] = pos_delta[i].wrapping_add(prev).wrapping_add(1);
        }
        let ex_val = U32Classic.decode(&data[val_bytes_range], nex)?;
        (pos_delta, ex_val)
    };

    // `pos < prev_pos || pos >= n` guards against a corrupted/adversarial
    // position stream (e.g. non-increasing after wraparound in the delta
    // reconstruction above) sending `run_len` negative or `literal` slicing
    // out of bounds.
    out.reserve(n);
    let mut cursor = 0usize;
    let mut prev_pos = 0usize;
    for (j, &pos) in ex_pos.iter().enumerate() {
        let pos = pos as usize;
        if pos < prev_pos || pos >= n {
            return Err(too_short(offset + n_literal, data.len()));
        }
        let run_len = pos - prev_pos;
        out.extend_from_slice(&literal[cursor..cursor + run_len]);
        cursor += run_len;

        out.push((ex_val[j] as u16).wrapping_add(THRESHOLD + 1));
        prev_pos = pos + 1;
    }
    out.extend_from_slice(&literal[cursor..]);

    Ok(offset + n_literal)
}

/// Widen a run of literal bytes to `u16` (zero-extend), appending to `out`.
fn widen_into(bytes: &[u8], out: &mut Vec<u16>) {
    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-ssse3"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { widen_sse2(bytes, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { widen_neon(bytes, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { widen_sse2(bytes, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { widen_neon(bytes, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(
            any(feature = "simd-avx2", feature = "simd-ssse3"),
            target_arch = "x86_64"
        ),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(
            feature = "simd-auto",
            not(any(feature = "simd-avx2", feature = "simd-ssse3", feature = "simd-neon")),
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )))]
    widen_scalar(bytes, out);
}

#[allow(dead_code)]
fn widen_scalar(bytes: &[u8], out: &mut Vec<u16>) {
    out.extend(bytes.iter().map(|&b| b as u16));
}

// SSE2: 16 bytes per iteration, widened to two 8-lane u16 registers via
// unpacklo/unpackhi with a zero register (SSE2-only zero-extend; pmovzx is
// SSE4.1). Same technique noted in svbzd_fused.rs's sign-extension comment.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn widen_sse2(bytes: &[u8], out: &mut Vec<u16>) {
    use core::arch::x86_64::*;

    let n = bytes.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 16) * 16;

    // SAFETY: _mm_setzero_si128 has no preconditions and touches no memory.
    let zero = unsafe { _mm_setzero_si128() };
    let mut i = 0usize;
    while i < simd_n {
        unsafe {
            // SAFETY: i + 16 <= simd_n <= n; bytes slice bounds are valid.
            let v = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
            let lo = _mm_unpacklo_epi8(v, zero);
            let hi = _mm_unpackhi_epi8(v, zero);
            // SAFETY: out.reserve(n) ensures capacity; base + i + 16 <= base + n.
            let out_ptr = out.as_mut_ptr().add(base + i) as *mut __m128i;
            _mm_storeu_si128(out_ptr, lo);
            _mm_storeu_si128(out_ptr.add(1), hi);
        }
        i += 16;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    out.extend(bytes[simd_n..].iter().map(|&b| b as u16));
}

#[cfg(test)]
mod widen_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn reference(bytes: &[u8]) -> Vec<u16> {
        bytes.iter().map(|&b| b as u16).collect()
    }

    #[test]
    fn dispatch_matches_reference_all_boundary_lengths() {
        for n in 0..=40 {
            let bytes: Vec<u8> = (0..n as u32).map(|i| (i * 37 % 256) as u8).collect();
            let mut out = Vec::new();
            widen_into(&bytes, &mut out);
            assert_eq!(out, reference(&bytes), "n={n}");
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        any(feature = "simd-auto", feature = "simd-ssse3")
    ))]
    #[test]
    fn sse2_matches_reference_directly() {
        for n in 0..=40 {
            let bytes: Vec<u8> = (0..n as u32).map(|i| (i * 37 % 256) as u8).collect();
            let mut out = Vec::new();
            unsafe { widen_sse2(&bytes, &mut out) };
            assert_eq!(out, reference(&bytes), "n={n}");
        }
    }

    #[cfg(all(
        target_arch = "aarch64",
        any(feature = "simd-auto", feature = "simd-neon")
    ))]
    #[test]
    fn neon_matches_reference_directly() {
        for n in 0..=40 {
            let bytes: Vec<u8> = (0..n as u32).map(|i| (i * 37 % 256) as u8).collect();
            let mut out = Vec::new();
            unsafe { widen_neon(&bytes, &mut out) };
            assert_eq!(out, reference(&bytes), "n={n}");
        }
    }
}

// NEON: 16 bytes per iteration, widened via vmovl_u8 on the low/high halves.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn widen_neon(bytes: &[u8], out: &mut Vec<u16>) {
    use core::arch::aarch64::*;

    let n = bytes.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 16) * 16;

    let mut i = 0usize;
    while i < simd_n {
        unsafe {
            // SAFETY: i + 16 <= simd_n <= n; bytes slice bounds are valid.
            let v = vld1q_u8(bytes.as_ptr().add(i));
            let lo = vmovl_u8(vget_low_u8(v));
            let hi = vmovl_u8(vget_high_u8(v));
            // SAFETY: out.reserve(n) ensures capacity; base + i + 16 <= base + n.
            vst1q_u16(out.as_mut_ptr().add(base + i), lo);
            vst1q_u16(out.as_mut_ptr().add(base + i + 8), hi);
        }
        i += 16;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    out.extend(bytes[simd_n..].iter().map(|&b| b as u16));
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn roundtrip(values: &[u16]) {
        let mut enc = Vec::new();
        encode_into(values, &mut enc);
        let mut out = Vec::new();
        let consumed = decode_into(&enc, values.len(), &mut out).unwrap();
        assert_eq!(consumed, enc.len());
        assert_eq!(out, values);
    }

    #[test]
    fn no_exceptions() {
        roundtrip(&(0..40).collect::<Vec<u16>>());
    }

    #[test]
    fn single_exception() {
        roundtrip(&[1u16, 300, 2, 3]);
    }

    #[test]
    fn consecutive_exceptions_zero_run_between() {
        // Adjacent exception positions: run_len == 0 between them.
        roundtrip(&[1u16, 300, 400, 500, 2]);
    }

    #[test]
    fn exceptions_at_start_and_end() {
        roundtrip(&[300u16, 1, 2, 3, 400]);
    }

    #[test]
    fn all_exceptions() {
        roundtrip(&[300u16, 400, 500, 600, 700]);
    }

    #[test]
    fn runs_crossing_simd_width() {
        // 20 literal bytes before the exception, 20 after: crosses the 16-byte
        // SIMD width on both sides of a single exception.
        let mut values: Vec<u16> = (0..20).map(|i| i % 250).collect();
        values.push(9999);
        values.extend((0..20).map(|i| i % 250));
        roundtrip(&values);
    }

    #[test]
    fn empty() {
        roundtrip(&[]);
    }

    #[test]
    fn decode_rejects_truncated_header() {
        assert!(decode_into(&[0u8, 0, 0], 1, &mut Vec::new()).is_err());
    }

    #[test]
    fn decode_rejects_out_of_order_positions() {
        // nex=2, positions [5, u32::MAX] via a wrapping delta reconstruction
        // that produces a non-increasing sequence — must error, not panic.
        let mut data = Vec::new();
        data.extend_from_slice(&2u32.to_le_bytes());
        let mut pos_bytes = Vec::new();
        U32Classic.encode_into(&[5u32, u32::MAX], &mut pos_bytes);
        data.extend_from_slice(&(pos_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&pos_bytes);
        let mut val_bytes = Vec::new();
        U32Classic.encode_into(&[0u32, 0], &mut val_bytes);
        data.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
        data.extend_from_slice(&val_bytes);
        data.extend_from_slice(&[0u8; 10]); // literal padding; should error before this matters

        let mut out = Vec::new();
        assert!(decode_into(&data, 10, &mut out).is_err());
    }

    #[test]
    fn decode_rejects_position_out_of_bounds() {
        // nex=1, position = 100 but n=3.
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&100u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&[0u8; 3]);

        let mut out = Vec::new();
        assert!(decode_into(&data, 3, &mut out).is_err());
    }
}
