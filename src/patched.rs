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

use crate::u32::U32Classic;
use crate::{DecodeError, EncodeError};

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
    let (ex_pos, ex_val) = find_exceptions(values);
    // Size from the actual exception count rather than `max_encoded_len`,
    // which assumes every value is an exception (about 8.5 bytes per value).
    let bound = bound_for(values.len(), ex_pos.len());
    let start = out.len();
    out.resize(start + bound, 0);
    let written = encode_parts(values, ex_pos, &ex_val, &mut out[start..]);
    out.truncate(start + written);
}

/// Worst-case encoded size in bytes for `n` values, reached when every value
/// is an exception: a 4-byte exception count, two 4-byte length prefixes and
/// two [`U32Classic`]-encoded arrays of `n` values.
///
/// Use this to size the buffer for [`encode_to_slice`].
///
/// # Examples
///
/// ```
/// # use svb::patched;
/// assert_eq!(patched::max_encoded_len(0), 12);
/// assert_eq!(patched::max_encoded_len(5), 12 + 2 * (2 + 4 * 5));
/// ```
pub const fn max_encoded_len(n: usize) -> usize {
    bound_for(n, n)
}

/// Encode `values` into the start of `out` and return the number of bytes
/// written.
///
/// `out` must be at least [`max_encoded_len(values.len())`](max_encoded_len)
/// bytes long, otherwise [`EncodeError::OutputTooSmall`] is returned and `out`
/// is not modified. The check is against the worst case, so a shorter slice is
/// rejected even if the actual encoded bytes would have fitted. Bytes in `out`
/// after the returned length may be overwritten.
///
/// # Examples
///
/// ```
/// # use svb::patched;
/// let values = [1u16, 300, 2];
/// let mut buf = vec![0u8; patched::max_encoded_len(values.len())];
/// let len = patched::encode_to_slice(&values, &mut buf).unwrap();
/// let mut expected = Vec::new();
/// patched::encode_into(&values, &mut expected);
/// assert_eq!(&buf[..len], expected);
/// ```
pub fn encode_to_slice(values: &[u16], out: &mut [u8]) -> Result<usize, EncodeError> {
    let need = max_encoded_len(values.len());
    if out.len() < need {
        return Err(EncodeError::OutputTooSmall {
            need,
            have: out.len(),
        });
    }
    let (ex_pos, ex_val) = find_exceptions(values);
    Ok(encode_parts(values, ex_pos, &ex_val, out))
}

// Upper bound on the encoded size of `n` values of which `nex` are
// exceptions. `nex >= 2` is the largest case: 12 bytes of count and length
// prefixes, two U32Classic arrays of `nex` values, and `n - nex` literals.
// That is also at least the `nex <= 1` sizes (4 + n and 12 + n - 1).
const fn bound_for(n: usize, nex: usize) -> usize {
    let side = U32Classic.max_encoded_len(nex);
    12usize
        .saturating_add(side.saturating_mul(2))
        .saturating_add(n.saturating_sub(nex))
}

// Positions and residuals (`value - 256`) of the values that do not fit in a
// byte.
fn find_exceptions(values: &[u16]) -> (Vec<u32>, Vec<u32>) {
    let mut ex_pos: Vec<u32> = Vec::new();
    let mut ex_val: Vec<u32> = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v > THRESHOLD {
            ex_pos.push(i as u32);
            ex_val.push((v - THRESHOLD - 1) as u32);
        }
    }
    (ex_pos, ex_val)
}

// Writes the encoding into `out`, which must hold at least
// `bound_for(values.len(), ex_pos.len())` bytes. Returns the bytes written.
fn encode_parts(values: &[u16], mut ex_pos: Vec<u32>, ex_val: &[u32], out: &mut [u8]) -> usize {
    let nex = ex_pos.len();
    out[..4].copy_from_slice(&(nex as u32).to_le_bytes());
    let mut offset = 4;

    if nex > 1 {
        // Positions are stored as gaps: the first position, then the number
        // of values between each exception and the next.
        for i in (1..nex).rev() {
            ex_pos[i] = ex_pos[i] - ex_pos[i - 1] - 1;
        }
        // U32Classic.encode_to_slice needs its worst case free; the
        // remaining space after each length prefix is at least that (see
        // `bound_for`).
        for arr in [&ex_pos[..], ex_val] {
            let len = U32Classic
                .encode_to_slice(arr, &mut out[offset + 4..])
                .expect("buffer sized by bound_for");
            out[offset..offset + 4].copy_from_slice(&(len as u32).to_le_bytes());
            offset += 4 + len;
        }
    } else if nex == 1 {
        out[offset..offset + 4].copy_from_slice(&ex_pos[0].to_le_bytes());
        out[offset + 4..offset + 8].copy_from_slice(&ex_val[0].to_le_bytes());
        offset += 8;
    }

    let literals = &mut out[offset..offset + (values.len() - nex)];
    let mut j = 0;
    for &v in values {
        if v <= THRESHOLD {
            literals[j] = v as u8;
            j += 1;
        }
    }
    offset + j
}

/// Decode exactly `n` values from the start of `data`, appending them to `out`.
///
/// Returns the number of bytes consumed from `data`. `n` must equal the
/// number of values that were originally encoded, same convention as
/// [`crate::u32::U32Classic::decode`]. On error `out` is left at its original
/// length.
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
    let mut ex_pos = Vec::new();
    let mut ex_val = Vec::new();
    decode_into_with_scratch(data, n, out, &mut ex_pos, &mut ex_val)
}

/// Decode `out.len()` values from the start of `data` into `out`.
///
/// Returns the number of bytes consumed from `data`. `out.len()` must equal
/// the number of values that were originally encoded. On error the contents
/// of `out` are unspecified.
///
/// # Examples
///
/// ```
/// # use svb::patched;
/// let mut enc = Vec::new();
/// patched::encode_into(&[1u16, 300, 2], &mut enc);
/// let mut out = [0u16; 3];
/// let consumed = patched::decode_to_slice(&enc, &mut out).unwrap();
/// assert_eq!(consumed, enc.len());
/// assert_eq!(out, [1u16, 300, 2]);
/// ```
pub fn decode_to_slice(data: &[u8], out: &mut [u16]) -> Result<usize, DecodeError> {
    let mut ex_pos = Vec::new();
    let mut ex_val = Vec::new();
    decode_to_slice_with_scratch(data, out, &mut ex_pos, &mut ex_val)
}

/// Same as [`decode_into`], but reuses caller-supplied scratch buffers
/// (cleared internally) instead of allocating fresh exception buffers on
/// every call. Used by [`crate::ExzdDecoder`] to avoid a heap
/// allocation per decode when repeatedly decoding many small frames (the
/// typical BLOW5/nanopore workload — many thousands of individual reads).
pub(crate) fn decode_into_with_scratch(
    data: &[u8],
    n: usize,
    out: &mut Vec<u16>,
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
) -> Result<usize, DecodeError> {
    // Every encoding of n > 0 values is at least n + 4 bytes: the 4-byte
    // count, then one byte per literal and at least one byte per exception
    // (8 bytes for a single exception), so a large `n` with a short input
    // fails here instead of after allocating `n` values.
    if n > 0 && data.len() < n + 4 {
        return Err(too_short(n + 4, data.len()));
    }
    let start = out.len();
    out.resize(start + n, 0);
    let result = decode_to_slice_with_scratch(data, &mut out[start..], ex_pos, ex_val);
    if result.is_err() {
        out.truncate(start);
    }
    result
}

/// Same as [`decode_to_slice`], with caller-supplied scratch buffers.
pub(crate) fn decode_to_slice_with_scratch(
    data: &[u8],
    out: &mut [u16],
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
) -> Result<usize, DecodeError> {
    // `nex * 5 >= n` (>= 20% exceptions): the merge_runs/merge_walk
    // crossover density, found with `merge_density_sweep`
    // (`cargo test --release -- --ignored --nocapture merge_density_sweep`).
    // With the in-place merges it is 18-20% at n=128/1024/8192; the 0.3.1
    // merges, which copied from a separate literal buffer, crossed at 14-15%.
    decode_to_slice_with_scratch_choosing(data, out, ex_pos, ex_val, |nex, n| {
        nex.saturating_mul(5) >= n
    })
}

/// Same as [`decode_to_slice_with_scratch`], but `use_walk_merge(nex, n)` picks
/// the merge strategy instead of the hardcoded density threshold — the hook
/// [`decode_to_slice_with_scratch`] uses in production, and that
/// `#[cfg(test)]` benchmarks use to force one strategy or the other when
/// sweeping for the actual crossover density (see `merge_density_sweep`
/// below).
fn decode_to_slice_with_scratch_choosing(
    data: &[u8],
    out: &mut [u16],
    ex_pos: &mut Vec<u32>,
    ex_val: &mut Vec<u32>,
    use_walk_merge: impl Fn(usize, usize) -> bool,
) -> Result<usize, DecodeError> {
    let n = out.len();
    ex_pos.clear();
    ex_val.clear();

    if data.len() < 4 {
        return Err(too_short(4, data.len()));
    }
    let nex = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    // Rejected before the exception arrays are decoded, which would
    // otherwise allocate `nex` values from an untrusted count. The position
    // check below rejects nex > n as well, but only after decoding.
    if nex > n {
        return Err(DecodeError::DataTruncated { index: n });
    }
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
        widen_to_slice(&data[offset..offset + n_literal], out);
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
    // The literals go to the front of `out`; the merge below moves them to
    // their final positions. n_literal <= n, so this fits.
    widen_to_slice(&data[offset..offset + n_literal], &mut out[..n_literal]);

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
    // out-of-range exception positions). Strictly increasing positions below
    // `n` also mean nex <= n, so there are exactly `n - nex` literals.
    let mut prev_pos = 0usize;
    for &pos in ex_pos.iter() {
        let pos = pos as usize;
        if pos < prev_pos || pos >= n {
            return Err(too_short(offset + n_literal, data.len()));
        }
        prev_pos = pos + 1;
    }

    // Two merge strategies, picked by exception density. Both work in place,
    // from the back of `out`: the literals sit in `out[..n - nex]` and each
    // one only ever moves right, so working backwards never overwrites a
    // literal before it has been moved.
    //
    // - `merge_runs`: moves each stretch of non-exception values with one
    //   `copy_within` (a vectorized memmove), interrupted only by a single
    //   write per exception. Wins when exceptions are rare (long runs, few
    //   interruptions), the common case for well-compressing signal.
    // - `merge_walk`: writes every output element individually through a
    //   raw pointer (no per-call overhead, no bounds checks). Loses to
    //   `merge_runs` on long runs but wins once runs get short enough that
    //   `merge_runs`'s per-run call overhead dominates, at about 18-20%
    //   exception density (see `merge_density_sweep`).
    if use_walk_merge(nex, n) {
        merge_walk(ex_pos, ex_val, out);
    } else {
        merge_runs(ex_pos, ex_val, out);
    }

    Ok(offset + n_literal)
}

/// Move the literals in `out[..n - nex]` to their final positions and write
/// the exceptions, one `copy_within` per run of literals. See the dispatch
/// comment in [`decode_to_slice_with_scratch`] for when this wins.
fn merge_runs(ex_pos: &[u32], ex_val: &[u32], out: &mut [u16]) {
    // `lit_end` is the end of the literals not yet moved, `out_end` the end
    // of the part of `out` not yet final. Between them are as many slots as
    // there are exceptions left to place.
    let mut lit_end = out.len() - ex_pos.len();
    let mut out_end = out.len();
    for (&pos, &val) in ex_pos.iter().zip(ex_val).rev() {
        let pos = pos as usize;
        let run_len = out_end - pos - 1;
        out.copy_within(lit_end - run_len..lit_end, pos + 1);
        out[pos] = (val as u16).wrapping_add(THRESHOLD + 1);
        lit_end -= run_len;
        out_end = pos;
    }
    // The literals before the first exception are already in place.
    debug_assert_eq!(lit_end, out_end);
}

/// Move the literals in `out[..n - nex]` to their final positions and write
/// the exceptions, with one raw-pointer write per output element (no bounds
/// checks), working from the back. See the dispatch comment in
/// [`decode_to_slice_with_scratch`] for when this wins.
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
fn merge_walk(ex_pos: &[u32], ex_val: &[u32], out: &mut [u16]) {
    let n = out.len();
    let out_ptr = out.as_mut_ptr();
    debug_assert_eq!(ex_pos.len(), ex_val.len());
    // `lit` is one past the next literal to move; `j` is the number of
    // exceptions not yet placed.
    let mut lit = n - ex_pos.len();
    let mut j = ex_pos.len();
    for i in (0..n).rev() {
        // SAFETY: `j > 0` is checked here, and `ex_pos`/`ex_val` are always
        // populated with the same length (both decoded from `nex` items in
        // `decode_to_slice_with_scratch`), but LLVM cannot see that across
        // the two slices, so plain `ex_val[j - 1]` would keep a bounds check.
        if j > 0 && unsafe { *ex_pos.get_unchecked(j - 1) as usize == i } {
            // SAFETY: i < n = out.len().
            unsafe {
                *out_ptr.add(i) = (*ex_val.get_unchecked(j - 1) as u16).wrapping_add(THRESHOLD + 1)
            };
            j -= 1;
        } else {
            // Positions i + 1..n are final: n - 1 - i of them, of which
            // nex - j are exceptions, so lit = i + 1 - j. The j exceptions
            // left all lie in 0..=i, and i is not one of them, so j <= i and
            // lit >= 1. out[lit - 1] = out[i - j] has not been written yet,
            // since every write so far went to a position above i.
            lit -= 1;
            // SAFETY: lit <= i < n = out.len(), see above.
            unsafe {
                *out_ptr.add(i) = *out_ptr.add(lit);
            }
        }
    }
}

/// Widen a run of literal bytes to `u16` (zero-extend) into `out`, which must
/// be the same length as `bytes`.
fn widen_to_slice(bytes: &[u8], out: &mut [u16]) {
    assert_eq!(bytes.len(), out.len());
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
fn widen_scalar(bytes: &[u8], out: &mut [u16]) {
    for (o, &b) in out.iter_mut().zip(bytes) {
        *o = b as u16;
    }
}

// SSE2: 16 bytes per iteration, widened to two 8-lane u16 registers via
// unpacklo/unpackhi with a zero register (SSE2-only zero-extend; pmovzx is
// SSE4.1). Same technique noted in svbzd_fused.rs's sign-extension comment.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn widen_sse2(bytes: &[u8], out: &mut [u16]) {
    use core::arch::x86_64::*;

    let n = bytes.len();
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
            // SAFETY: i + 16 <= n = out.len() (checked by widen_to_slice).
            let out_ptr = out.as_mut_ptr().add(i) as *mut __m128i;
            _mm_storeu_si128(out_ptr, lo);
            _mm_storeu_si128(out_ptr.add(1), hi);
        }
        i += 16;
    }
    widen_scalar(&bytes[simd_n..], &mut out[simd_n..]);
}

#[cfg(test)]
mod widen_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

    fn reference(bytes: &[u8]) -> Vec<u16> {
        bytes.iter().map(|&b| b as u16).collect()
    }

    #[test]
    fn dispatch_matches_reference_all_boundary_lengths() {
        for n in 0..=40 {
            let bytes: Vec<u8> = (0..n as u32).map(|i| (i * 37 % 256) as u8).collect();
            let mut out = vec![0u16; n];
            widen_to_slice(&bytes, &mut out);
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
            let mut out = vec![0u16; n];
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
            let mut out = vec![0u16; n];
            unsafe { widen_neon(&bytes, &mut out) };
            assert_eq!(out, reference(&bytes), "n={n}");
        }
    }
}

// NEON: 16 bytes per iteration, widened via vmovl_u8 on the low/high halves.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn widen_neon(bytes: &[u8], out: &mut [u16]) {
    use core::arch::aarch64::*;

    let n = bytes.len();
    let simd_n = (n / 16) * 16;

    let mut i = 0usize;
    while i < simd_n {
        unsafe {
            // SAFETY: i + 16 <= simd_n <= n; bytes slice bounds are valid.
            let v = vld1q_u8(bytes.as_ptr().add(i));
            let lo = vmovl_u8(vget_low_u8(v));
            let hi = vmovl_u8(vget_high_u8(v));
            // SAFETY: i + 16 <= n = out.len() (checked by widen_to_slice).
            vst1q_u16(out.as_mut_ptr().add(i), lo);
            vst1q_u16(out.as_mut_ptr().add(i + 8), hi);
        }
        i += 16;
    }
    widen_scalar(&bytes[simd_n..], &mut out[simd_n..]);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};

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

    // ── slice API ─────────────────────────────────────────────────────────────

    // The 0.3.1 `Vec` encoder, kept as an independent reference.
    fn encode_reference(values: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut ex_pos: Vec<u32> = Vec::new();
        let mut ex_val: Vec<u32> = Vec::new();
        for (i, &v) in values.iter().enumerate() {
            if v > THRESHOLD {
                ex_pos.push(i as u32);
                ex_val.push((v - THRESHOLD - 1) as u32);
            }
        }
        out.extend_from_slice(&(ex_pos.len() as u32).to_le_bytes());
        if ex_pos.len() > 1 {
            let mut pos_delta = vec![ex_pos[0]];
            for w in ex_pos.windows(2) {
                pos_delta.push(w[1] - w[0] - 1);
            }
            for arr in [&pos_delta, &ex_val] {
                let bytes = U32Classic.encode(arr);
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(&bytes);
            }
        } else if ex_pos.len() == 1 {
            out.extend_from_slice(&ex_pos[0].to_le_bytes());
            out.extend_from_slice(&ex_val[0].to_le_bytes());
        }
        for &v in values {
            if v <= THRESHOLD {
                out.push(v as u8);
            }
        }
        out
    }

    // `n` values with roughly `pct` percent exceptions, spread over the full
    // u16 range so the residuals take 1 and 2 bytes.
    fn values(n: usize, pct: usize) -> Vec<u16> {
        (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761);
                if (h >> 8) % 100 < pct as u32 {
                    256 + (h % 65280) as u16
                } else {
                    (h >> 24) as u16
                }
            })
            .collect()
    }

    const PCTS: [usize; 5] = [0, 3, 15, 50, 100];

    #[test]
    fn encode_to_slice_matches_reference() {
        for pct in PCTS {
            for n in (0..=70).chain([1000, 5000]) {
                let v = values(n, pct);
                let expected = encode_reference(&v);
                let got = crate::test_util::encode_checked(&v, max_encoded_len(n), |v, o| {
                    encode_to_slice(v, o).unwrap()
                });
                assert_eq!(got, expected, "pct={pct} n={n}");
                let mut via_vec = vec![9u8];
                encode_into(&v, &mut via_vec);
                assert_eq!(via_vec[1..], expected[..], "encode_into pct={pct} n={n}");
            }
        }
    }

    #[test]
    fn encode_large_position_gaps() {
        // Gaps over 65535 values give 3-byte position deltas.
        let mut v = vec![0u16; 200_000];
        for i in [0usize, 70_000, 70_001, 199_999] {
            v[i] = 40_000;
        }
        let expected = encode_reference(&v);
        let mut buf = vec![0xA5u8; max_encoded_len(v.len())];
        let len = encode_to_slice(&v, &mut buf).unwrap();
        assert_eq!(buf[..len], expected[..]);
        let mut out = vec![0u16; v.len()];
        assert_eq!(decode_to_slice(&expected, &mut out).unwrap(), len);
        assert_eq!(out, v);
    }

    #[test]
    fn encode_to_slice_too_small() {
        let v = values(10, 50);
        let need = max_encoded_len(10);
        let mut buf = vec![0xA5u8; need - 1];
        let err = encode_to_slice(&v, &mut buf).unwrap_err();
        assert!(
            matches!(err, EncodeError::OutputTooSmall { need: n, have } if n == need && have == need - 1),
            "{err:?}"
        );
        assert!(buf.iter().all(|&b| b == 0xA5), "out was modified");
    }

    #[test]
    fn decode_to_slice_both_merges() {
        for pct in PCTS {
            for n in (0..=70).chain([1000, 5000]) {
                let v = values(n, pct);
                let enc = encode_reference(&v);
                for walk in [false, true] {
                    // Stale data in every buffer: the decoder must overwrite
                    // all of `out` and clear its scratch.
                    let mut out = vec![0xBEEFu16; n];
                    let mut ex_pos = vec![1u32; 3];
                    let mut ex_val = vec![1u32; 3];
                    let consumed = decode_to_slice_with_scratch_choosing(
                        &enc,
                        &mut out,
                        &mut ex_pos,
                        &mut ex_val,
                        |_, _| walk,
                    )
                    .unwrap();
                    assert_eq!(consumed, enc.len(), "pct={pct} n={n} walk={walk}");
                    assert_eq!(out, v, "pct={pct} n={n} walk={walk}");
                }
            }
        }
    }

    #[test]
    fn decode_rejects_every_truncation() {
        for pct in PCTS {
            for n in [1usize, 2, 17, 40] {
                let v = values(n, pct);
                let enc = encode_reference(&v);
                for cut in 0..enc.len() {
                    let mut out = vec![0u16; n];
                    assert!(
                        decode_to_slice(&enc[..cut], &mut out).is_err(),
                        "pct={pct} n={n} cut={cut}"
                    );
                }
            }
        }
    }

    #[test]
    fn decode_into_appends_and_restores_on_error() {
        let a = values(30, 15);
        let enc = encode_reference(&a);
        let mut out = vec![1u16, 2];
        decode_into(&enc, 30, &mut out).unwrap();
        assert_eq!(out[..2], [1, 2]);
        assert_eq!(out[2..], a[..]);

        let mut out = vec![1u16, 2];
        assert!(decode_into(&enc[..enc.len() - 1], 30, &mut out).is_err());
        assert_eq!(out, [1, 2]);
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
            let mut out = vec![0u16; n];
            let mut ex_pos = Vec::new();
            let mut ex_val = Vec::new();
            // Warm up.
            decode_to_slice_with_scratch_choosing(
                enc,
                &mut out,
                &mut ex_pos,
                &mut ex_val,
                |_, _| force_walk,
            )
            .unwrap();

            let t0 = Instant::now();
            for _ in 0..iters {
                decode_to_slice_with_scratch_choosing(
                    enc,
                    &mut out,
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
