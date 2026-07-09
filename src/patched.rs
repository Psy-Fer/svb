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
    let mut literal = Vec::new();
    let mut ex_pos = Vec::new();
    let mut ex_val = Vec::new();
    decode_into_with_scratch(data, n, out, &mut literal, &mut ex_pos, &mut ex_val)
}

/// Same as [`decode_into`], but reuses caller-supplied scratch buffers
/// (cleared internally) instead of allocating a fresh literal/exception
/// buffer on every call. Used by [`crate::ExzdDecoder`] to avoid a heap
/// allocation per decode when repeatedly decoding many small frames (the
/// typical BLOW5/nanopore workload — many thousands of individual reads).
pub(crate) fn decode_into_with_scratch(
    data: &[u8],
    n: usize,
    out: &mut Vec<u16>,
    literal: &mut Vec<u16>,
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
) -> Result<usize, DecodeError> {
    // `nex * 7 >= n` (>=~14.3% exceptions): the merge_runs/merge_walk
    // crossover density, found empirically with `merge_density_sweep`
    // (`cargo test --release -- --ignored --nocapture merge_density_sweep`)
    // and consistent across n=128/1024/8192 — not the format's own ~20%
    // "compression may not be ideal" warning threshold, which turned out to
    // be more conservative than the actual crossover.
    decode_into_with_scratch_choosing(data, n, out, literal, ex_pos, ex_val, |nex, n| {
        nex.saturating_mul(7) >= n
    })
}

/// Same as [`decode_into_with_scratch`], but `use_walk_merge(nex, n)` picks
/// the merge strategy instead of the hardcoded density threshold — the hook
/// [`decode_into_with_scratch`] uses in production, and that
/// `#[cfg(test)]` benchmarks use to force one strategy or the other when
/// sweeping for the actual crossover density (see `merge_density_sweep`
/// below).
fn decode_into_with_scratch_choosing(
    data: &[u8],
    n: usize,
    out: &mut Vec<u16>,
    literal: &mut Vec<u16>,
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
    use_walk_merge: impl Fn(usize, usize) -> bool,
) -> Result<usize, DecodeError> {
    literal.clear();
    ex_pos.clear();
    ex_val.clear();

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
    widen_into(&data[offset..offset + n_literal], literal);

    if nex == 1 {
        let p = &data[pos_bytes_range];
        let v = &data[val_bytes_range];
        ex_pos.push(u32::from_le_bytes([p[0], p[1], p[2], p[3]]));
        ex_val.push(u32::from_le_bytes([v[0], v[1], v[2], v[3]]));
    } else {
        U32Classic.decode_into(&data[pos_bytes_range], nex, ex_pos)?;
        for i in 1..ex_pos.len() {
            let prev = ex_pos[i - 1];
            ex_pos[i] = ex_pos[i].wrapping_add(prev).wrapping_add(1);
        }
        U32Classic.decode_into(&data[val_bytes_range], nex, ex_val)?;
    }

    // Validate once, up front, so neither merge strategy below needs to
    // check for corrupted/adversarial input (non-increasing or
    // out-of-range exception positions).
    let mut prev_pos = 0usize;
    for &pos in ex_pos.iter() {
        let pos = pos as usize;
        if pos < prev_pos || pos >= n {
            return Err(too_short(offset + n_literal, data.len()));
        }
        prev_pos = pos + 1;
    }

    // Two merge strategies, picked by exception density:
    //
    // - `merge_runs`: batches each stretch of non-exception values into one
    //   `extend_from_slice` (a vectorized memcpy), interrupted only by a
    //   single `push` per exception. Wins when exceptions are rare (long
    //   runs, few interruptions) — the common case for well-compressing
    //   signal.
    // - `merge_walk`: writes every output element individually through a
    //   raw pointer (no per-call overhead, no bounds/capacity checks).
    //   Loses to `merge_runs` on long runs (a scalar per-element loop can't
    //   beat a vectorized memcpy) but wins once runs get short enough that
    //   `merge_runs`'s per-run call overhead dominates — crossover measured
    //   at ~14-15% exception density (see `merge_density_sweep`), matching
    //   the technique slow5lib's C reference (`ex_depress`) uses.
    out.reserve(n);
    if use_walk_merge(nex, n) {
        merge_walk(n, ex_pos, ex_val, literal, out);
    } else {
        merge_runs(ex_pos, ex_val, literal, out);
    }

    Ok(offset + n_literal)
}

/// Merge literals and exceptions back into `out` by copying each
/// non-exception run in one `extend_from_slice` call. See the dispatch
/// comment in [`decode_into_with_scratch`] for when this wins.
fn merge_runs(ex_pos: &[u32], ex_val: &[u32], literal: &[u16], out: &mut Vec<u16>) {
    let mut cursor = 0usize;
    let mut prev_pos = 0usize;
    for (j, &pos) in ex_pos.iter().enumerate() {
        let pos = pos as usize;
        let run_len = pos - prev_pos;
        out.extend_from_slice(&literal[cursor..cursor + run_len]);
        cursor += run_len;

        out.push((ex_val[j] as u16).wrapping_add(THRESHOLD + 1));
        prev_pos = pos + 1;
    }
    out.extend_from_slice(&literal[cursor..]);
}

/// Merge literals and exceptions back into `out` with one raw-pointer write
/// per output element (no bounds/capacity checks). See the dispatch comment
/// in [`decode_into_with_scratch`] for when this wins.
///
/// A two-phase variant (scatter exceptions into their final positions
/// first, then a separate walk that only ever fills literals) was tried,
/// matching slow5lib's C reference (`ex_depress`) structure exactly — it
/// measured no difference from this single combined loop. Investigating why
/// C still edges this out at high exception density (~9-11% at n=1024/8192)
/// found the gap isn't really about C vs Rust: compiling the *same* C
/// source with clang instead of gcc is itself 15-18% faster, since both
/// clang and rustc share the LLVM backend. Disassembling this function
/// showed LLVM 4×-unrolling the scatter phase and splitting the walk phase
/// into multiple specialized loop variants for the Rust version, vs. the
/// single compact loop it emits for the structurally-identical C — likely
/// driven by the stronger `noalias` guarantees Rust's `&mut` references let
/// rustc emit, giving LLVM's unrolling heuristics different (here not
/// better) incentives than the equivalent raw-pointer C. Not chased
/// further: closing this would mean fighting LLVM's cost model rather than
/// fixing an identifiable inefficiency in this code.
fn merge_walk(n: usize, ex_pos: &[u32], ex_val: &[u32], literal: &[u16], out: &mut Vec<u16>) {
    let out_base = out.len();
    // SAFETY: caller (`decode_into_with_scratch`) already called
    // `out.reserve(n)`, guaranteeing capacity for n more elements from
    // out_base; the loop below writes each index in [0, n) exactly once via
    // out_ptr, so out.set_len(out_base + n) afterwards is sound.
    let out_ptr = unsafe { out.as_mut_ptr().add(out_base) };
    debug_assert_eq!(ex_pos.len(), ex_val.len());
    let mut lit_cursor = 0usize;
    let mut j = 0usize;
    for i in 0..n {
        // SAFETY: `j < ex_pos.len()` is checked here, and `ex_pos`/`ex_val`
        // are always populated with the same length (both decoded from
        // `nex` items in `decode_into_with_scratch`) — but that invariant
        // lives in the caller, not in a form LLVM can see across the two
        // independent slices, so plain `ex_val[j]` indexing paid a real
        // (if perfectly-predicted and measurably free) bounds check here
        // despite being provably in range.
        if j < ex_pos.len() && unsafe { *ex_pos.get_unchecked(j) as usize == i } {
            // SAFETY: i < n, out_ptr has room for n elements.
            unsafe {
                *out_ptr.add(i) = (*ex_val.get_unchecked(j) as u16).wrapping_add(THRESHOLD + 1)
            };
            j += 1;
        } else {
            // SAFETY: exactly n - nex literal values were widened by the
            // caller, and lit_cursor advances once per non-exception i in
            // [0, n), so it never reaches literal.len(); i < n so
            // out_ptr.add(i) is valid.
            unsafe {
                *out_ptr.add(i) = *literal.get_unchecked(lit_cursor);
            }
            lit_cursor += 1;
        }
    }
    // SAFETY: every index in [0, n) was written above.
    unsafe { out.set_len(out_base + n) };
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

    #[cfg(feature = "std")]
    #[test]
    #[ignore = "manual perf sweep, not part of CI - run with \
                `cargo test --release --features simd-avx2 -- --ignored --nocapture merge_density_sweep`"]
    fn merge_density_sweep() {
        use std::time::Instant;

        fn gen_values(n: usize, density_pct: usize) -> Vec<u16> {
            let stride = (100 / density_pct.max(1)).max(1);
            (0..n)
                .map(|i| {
                    if i % stride == 0 {
                        300 + (i % 5000) as u16
                    } else {
                        (i % 200) as u16
                    }
                })
                .collect()
        }

        fn time_strategy(enc: &[u8], n: usize, iters: usize, force_walk: bool) -> f64 {
            let mut out = Vec::new();
            let mut literal = Vec::new();
            let mut ex_pos = Vec::new();
            let mut ex_val = Vec::new();
            // Warm up.
            decode_into_with_scratch_choosing(
                enc,
                n,
                &mut out,
                &mut literal,
                &mut ex_pos,
                &mut ex_val,
                |_, _| force_walk,
            )
            .unwrap();

            let t0 = Instant::now();
            for _ in 0..iters {
                out.clear();
                decode_into_with_scratch_choosing(
                    enc,
                    n,
                    &mut out,
                    &mut literal,
                    &mut ex_pos,
                    &mut ex_val,
                    |_, _| force_walk,
                )
                .unwrap();
            }
            t0.elapsed().as_secs_f64() / iters as f64
        }

        for n in [128usize, 1024, 8192] {
            let iters = 20_000;
            eprintln!("--- n={n} ---");
            eprintln!("density%  runs(ns)  walk(ns)  walk/runs  winner");
            for density_pct in [8, 10, 12, 13, 14, 15, 16, 18, 20, 25, 30] {
                let values = gen_values(n, density_pct);
                let mut enc = Vec::new();
                encode_into(&values, &mut enc);

                let runs_ns = time_strategy(&enc, n, iters, false) * 1e9;
                let walk_ns = time_strategy(&enc, n, iters, true) * 1e9;
                let winner = if walk_ns < runs_ns { "walk" } else { "runs" };
                eprintln!(
                    "{density_pct:>7}  {runs_ns:>9.1}  {walk_ns:>9.1}  {:>9.3}  {winner}",
                    walk_ns / runs_ns
                );
            }
        }
    }
}
