//! Delta encoding and decoding as a composable layer over any integer type.
//!
//! Delta encoding replaces each value with the difference from the previous
//! value, which typically produces smaller numbers that compress well with
//! StreamVByte. Decoding reconstructs the original values by computing a
//! running prefix sum over the delta sequence.
//!
//! The functions in this module accept any type that implements [`Delta`].
//! Call [`encode`] / [`decode`] for standalone sequences, or the
//! `_with_initial` / `_into` variants for streaming use.

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

mod private {
    pub trait Sealed {}
}

/// Marker trait for types that support delta encoding and decoding.
///
/// This trait is sealed; it cannot be implemented outside this crate.
/// Implemented for `i16`, `i32`, `i64`, `u32`, and `u64`.
///
/// Choose the concrete type based on your data:
/// - Use `i16`, `i32`, or `i64` when the sequence is non-monotone and you
///   plan to follow delta encoding with [`crate::zigzag`] to map signed
///   differences back to small unsigned values.
/// - Use `u32` or `u64` for sorted or non-decreasing sequences where all
///   differences are non-negative.
///
/// All arithmetic is wrapping, so overflow is defined and lossless.
pub trait Delta: private::Sealed + Copy + Default {
    #[doc(hidden)]
    fn __sub(self, rhs: Self) -> Self;
    #[doc(hidden)]
    fn __add(self, rhs: Self) -> Self;
    // Overridable decode dispatch; default is scalar.
    #[doc(hidden)]
    fn __decode_into(initial: Self, deltas: &[Self], out: &mut Vec<Self>) {
        let mut acc = initial;
        for &d in deltas {
            acc = acc.__add(d);
            out.push(acc);
        }
    }
}

// ── Scalar fallback helpers ───────────────────────────────────────────────────

#[allow(dead_code)]
fn decode_scalar<T: Copy + Delta>(initial: T, deltas: &[T], out: &mut Vec<T>) {
    let mut acc = initial;
    for &d in deltas {
        acc = acc.__add(d);
        out.push(acc);
    }
}

// ── i16 ──────────────────────────────────────────────────────────────────────

impl private::Sealed for i16 {}
impl Delta for i16 {
    fn __sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
    fn __add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
    fn __decode_into(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
        decode_into_i16(initial, deltas, out);
    }
}

// cfg condition: "any explicit compile-time SIMD feature applies for i16 on this arch"
// OR "simd-auto applies and we are on a SIMD-capable arch"
// Used to cfg-out the scalar fallback to avoid unreachable_code.
fn decode_into_i16(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
    #[cfg(all(
        any(feature = "simd-avx2", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2_i16(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i16(initial, deltas, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2_i16(initial, deltas, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i16(initial, deltas, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(any(feature = "simd-avx2", feature = "simd-sse2"), target_arch = "x86_64"),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), any(target_arch = "x86_64", target_arch = "aarch64"))
    )))]
    decode_scalar(initial, deltas, out);
}

// ── i32 ──────────────────────────────────────────────────────────────────────

impl private::Sealed for i32 {}
impl Delta for i32 {
    fn __sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
    fn __add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
    fn __decode_into(initial: i32, deltas: &[i32], out: &mut Vec<i32>) {
        decode_into_i32(initial, deltas, out);
    }
}

fn decode_into_i32(initial: i32, deltas: &[i32], out: &mut Vec<i32>) {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        unsafe { decode_avx2_i32(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"))]
    {
        // SAFETY: simd-sse2 feature declares SSE2 is available at runtime.
        unsafe { decode_sse2_i32(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i32(initial, deltas, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        feature = "std",
        target_arch = "x86_64"
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime.
            unsafe { decode_avx2_i32(initial, deltas, out) };
        } else {
            // SAFETY: SSE2 is always available on x86_64.
            unsafe { decode_sse2_i32(initial, deltas, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i32(initial, deltas, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(feature = "simd-avx2", target_arch = "x86_64"),
        all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), feature = "std", target_arch = "x86_64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), target_arch = "aarch64")
    )))]
    decode_scalar(initial, deltas, out);
}

// ── u32 ──────────────────────────────────────────────────────────────────────

impl private::Sealed for u32 {}
impl Delta for u32 {
    fn __sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
    fn __add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
    fn __decode_into(initial: u32, deltas: &[u32], out: &mut Vec<u32>) {
        decode_into_u32(initial, deltas, out);
    }
}

fn decode_into_u32(initial: u32, deltas: &[u32], out: &mut Vec<u32>) {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        unsafe { decode_avx2_u32(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"))]
    {
        // SAFETY: simd-sse2 feature declares SSE2 is available at runtime.
        unsafe { decode_sse2_u32(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_u32(initial, deltas, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        feature = "std",
        target_arch = "x86_64"
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime.
            unsafe { decode_avx2_u32(initial, deltas, out) };
        } else {
            // SAFETY: SSE2 is always available on x86_64.
            unsafe { decode_sse2_u32(initial, deltas, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_u32(initial, deltas, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(feature = "simd-avx2", target_arch = "x86_64"),
        all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), feature = "std", target_arch = "x86_64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), target_arch = "aarch64")
    )))]
    decode_scalar(initial, deltas, out);
}

// ── i64 ──────────────────────────────────────────────────────────────────────

impl private::Sealed for i64 {}
impl Delta for i64 {
    fn __sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
    fn __add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
    fn __decode_into(initial: i64, deltas: &[i64], out: &mut Vec<i64>) {
        decode_into_i64(initial, deltas, out);
    }
}

fn decode_into_i64(initial: i64, deltas: &[i64], out: &mut Vec<i64>) {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        unsafe { decode_avx2_i64(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"))]
    {
        // SAFETY: simd-sse2 feature declares SSE2 is available at runtime.
        unsafe { decode_sse2_i64(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i64(initial, deltas, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        feature = "std",
        target_arch = "x86_64"
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime.
            unsafe { decode_avx2_i64(initial, deltas, out) };
        } else {
            // SAFETY: SSE2 is always available on x86_64.
            unsafe { decode_sse2_i64(initial, deltas, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i64(initial, deltas, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(feature = "simd-avx2", target_arch = "x86_64"),
        all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), feature = "std", target_arch = "x86_64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), target_arch = "aarch64")
    )))]
    decode_scalar(initial, deltas, out);
}

// ── u64 ──────────────────────────────────────────────────────────────────────

impl private::Sealed for u64 {}
impl Delta for u64 {
    fn __sub(self, rhs: Self) -> Self { self.wrapping_sub(rhs) }
    fn __add(self, rhs: Self) -> Self { self.wrapping_add(rhs) }
    fn __decode_into(initial: u64, deltas: &[u64], out: &mut Vec<u64>) {
        decode_into_u64(initial, deltas, out);
    }
}

fn decode_into_u64(initial: u64, deltas: &[u64], out: &mut Vec<u64>) {
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares AVX2 is available at runtime.
        unsafe { decode_avx2_u64(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"))]
    {
        // SAFETY: simd-sse2 feature declares SSE2 is available at runtime.
        unsafe { decode_sse2_u64(initial, deltas, out) };
    }
    #[cfg(all(feature = "simd-neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_u64(initial, deltas, out) };
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        feature = "std",
        target_arch = "x86_64"
    ))]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed at runtime.
            unsafe { decode_avx2_u64(initial, deltas, out) };
        } else {
            // SAFETY: SSE2 is always available on x86_64.
            unsafe { decode_sse2_u64(initial, deltas, out) };
        }
    }
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_u64(initial, deltas, out) };
    }
    // Scalar fallback: only compiled when no SIMD path covers this target.
    #[cfg(not(any(
        all(feature = "simd-avx2", target_arch = "x86_64"),
        all(feature = "simd-sse2", not(feature = "simd-avx2"), target_arch = "x86_64"),
        all(feature = "simd-neon", target_arch = "aarch64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), feature = "std", target_arch = "x86_64"),
        all(feature = "simd-auto", not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon")), target_arch = "aarch64")
    )))]
    decode_scalar(initial, deltas, out);
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Delta-encode `samples`, using `T::default()` (typically `0`) as the implicit value before the first element.
///
/// Each output element is `samples[i] - samples[i-1]` (wrapping), with `0`
/// used as `samples[-1]`.
///
/// # Examples
///
/// ```
/// # use svb::delta;
/// let deltas = delta::encode(&[10i16, 11, 13]);
/// assert_eq!(deltas, [10, 1, 2]);
/// ```
pub fn encode<T: Delta>(samples: &[T]) -> Vec<T> {
    encode_with_initial(T::default(), samples)
}

/// Delta-encode `samples`, treating `initial` as the value logically preceding the first element.
///
/// Pass `0` (or `T::default()`) for a standalone sequence; pass the last value
/// from the previous chunk when encoding a stream in multiple pieces.
///
/// # Examples
///
/// ```
/// # use svb::delta;
/// // Encode two chunks of a stream so decode can be chained.
/// let chunk1 = delta::encode_with_initial(0i16, &[10, 11, 13]);
/// let chunk2 = delta::encode_with_initial(13i16, &[14, 20]);
/// assert_eq!(chunk1, [10, 1, 2]);
/// assert_eq!(chunk2, [1, 6]);
/// ```
pub fn encode_with_initial<T: Delta>(initial: T, samples: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(samples.len());
    encode_with_initial_into(initial, samples, &mut out);
    out
}

/// Delta-encode `samples` using `T::default()` as the initial value, appending the result to `out`.
pub fn encode_into<T: Delta>(samples: &[T], out: &mut Vec<T>) {
    encode_with_initial_into(T::default(), samples, out);
}

/// Delta-decode `deltas`, using `T::default()` (typically `0`) as the initial accumulator.
///
/// This is the inverse of [`encode`]; pass the same sequence of delta values
/// that `encode` produced to recover the original samples.
///
/// # Examples
///
/// ```
/// # use svb::delta;
/// let samples = delta::decode(&[10i16, 1, 2]);
/// assert_eq!(samples, [10, 11, 13]);
/// ```
pub fn decode<T: Delta>(deltas: &[T]) -> Vec<T> {
    decode_with_initial(T::default(), deltas)
}

/// Delta-decode `deltas`, starting the prefix sum from `initial`.
///
/// Use `initial = 0` (or `T::default()`) for a standalone sequence; for
/// streaming use, pass the last decoded value from the previous chunk so the
/// prefix sum continues from where it left off.
///
/// # Examples
///
/// ```
/// # use svb::delta;
/// // Decode two independently-encoded chunks:
/// let s1 = delta::decode_with_initial(0i16, &[10i16, 1, 2]);   // [10, 11, 13]
/// let s2 = delta::decode_with_initial(13i16, &[1i16, 6]);       // [14, 20]
/// assert_eq!(s1, [10, 11, 13]);
/// assert_eq!(s2, [14, 20]);
/// ```
pub fn decode_with_initial<T: Delta>(initial: T, deltas: &[T]) -> Vec<T> {
    let mut out = Vec::with_capacity(deltas.len());
    T::__decode_into(initial, deltas, &mut out);
    out
}

/// Delta-decode `deltas` using `T::default()` as the initial accumulator, appending the result to `out`.
pub fn decode_into<T: Delta>(deltas: &[T], out: &mut Vec<T>) {
    T::__decode_into(T::default(), deltas, out);
}

fn encode_with_initial_into<T: Delta>(initial: T, samples: &[T], out: &mut Vec<T>) {
    let mut prev = initial;
    for &s in samples {
        out.push(s.__sub(prev));
        prev = s;
    }
}

// ── SSE2 implementations (x86_64) ─────────────────────────────────────────────

// SSE2 prefix-sum delta decode: 8 i16 values per iteration.
//
// Three-step scan builds all 8 prefix sums in-register:
//   v += shl_1(v)  →  pairwise running sums
//   v += shl_2(v)  →  4-element running sums
//   v += shl_4(v)  →  8-element prefix sums (all starting from d0)
// Then add the inter-block accumulator `acc` to all 8 lanes and extract
// element 7 (the cumulative sum of all 8 deltas + acc) as the new accumulator.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2_i16(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
    use core::arch::x86_64::*;

    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let result = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; deltas slice bounds are valid.
            let v = _mm_loadu_si128(deltas.as_ptr().add(i) as *const __m128i);
            // Three-step prefix-sum scan (all wrapping i16 arithmetic).
            let v = _mm_add_epi16(v, _mm_slli_si128(v, 2));
            let v = _mm_add_epi16(v, _mm_slli_si128(v, 4));
            let v = _mm_add_epi16(v, _mm_slli_si128(v, 8));
            // Broadcast acc to all lanes and add.
            _mm_add_epi16(v, _mm_set1_epi16(acc))
        };
        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 8 <= base + n.
            let out_ptr = out.as_mut_ptr().add(base + i) as *mut __m128i;
            _mm_storeu_si128(out_ptr, result);
            // Element 7 is the prefix sum of all 8 deltas + acc = new accumulator.
            acc = _mm_extract_epi16(result, 7) as i16;
        }
        i += 8;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    // Scalar tail for n % 8 remaining values.
    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

// SSE2 prefix-sum delta decode for i32: 4 values per iteration.
//
// Two-step scan:
//   v += shl_1_elem(v)  [shift by 4 bytes]
//   v += shl_2_elem(v)  [shift by 8 bytes]
// Then add acc broadcast, extract element 3 as new acc (SSE2-only via shuffle).
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2_i32_inner(initial: i32, deltas: *const i32, n: usize, out: *mut i32) -> i32 {
    use core::arch::x86_64::*;

    let simd_n = (n / 4) * 4;
    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: caller ensures i + 4 <= n and pointers are valid.
            _mm_loadu_si128(deltas.add(i) as *const __m128i)
        };
        // Two-step prefix-sum (wrapping i32 = wrapping u32 at the bit level).
        let v = unsafe { _mm_add_epi32(v, _mm_slli_si128(v, 4)) };
        let v = unsafe { _mm_add_epi32(v, _mm_slli_si128(v, 8)) };
        // Add inter-block accumulator to all lanes.
        let result = unsafe { _mm_add_epi32(v, _mm_set1_epi32(acc)) };
        unsafe {
            // SAFETY: out pointer valid for i + 4 elements.
            _mm_storeu_si128(out.add(i) as *mut __m128i, result);
            // Extract element 3 as new accumulator (SSE2-only; avoids SSE4.1
            // _mm_extract_epi32). Shuffle broadcasts element 3 to all positions,
            // then cvtsi128_si32 reads the low 32 bits.
            acc = _mm_cvtsi128_si32(_mm_shuffle_epi32(result, 0xFF_u32 as i32));
        }
        i += 4;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2_i32(initial: i32, deltas: &[i32], out: &mut Vec<i32>) {
    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 4) * 4;

    // SAFETY: pointers derived from valid slices/vecs; capacity reserved above.
    let acc = unsafe { decode_sse2_i32_inner(initial, deltas.as_ptr(), n, out.as_mut_ptr().add(base)) };
    unsafe {
        // SAFETY: elements [base, base + simd_n) were written by decode_sse2_i32_inner.
        out.set_len(base + simd_n);
    }

    let mut acc = acc;
    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2_u32(initial: u32, deltas: &[u32], out: &mut Vec<u32>) {
    // Wrapping add is identical for i32 and u32 at the bit level.
    // SAFETY: u32 and i32 have the same size/alignment; wrapping arithmetic is identical.
    // We cast the output Vec pointer to reuse the i32 implementation without copying data.
    let deltas_i32 = unsafe {
        core::slice::from_raw_parts(deltas.as_ptr() as *const i32, deltas.len())
    };
    let out_i32 = unsafe { &mut *(out as *mut Vec<u32> as *mut Vec<i32>) };
    unsafe { decode_sse2_i32(initial as i32, deltas_i32, out_i32) };
}

// SSE2 prefix-sum delta decode for i64: 2 values per iteration.
//
// One-step scan:
//   v += shl_1_elem(v)  [shift by 8 bytes]
// Then add acc broadcast, extract element 1 (high 64 bits) as new acc.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2_i64_inner(initial: i64, deltas: *const i64, n: usize, out: *mut i64) -> i64 {
    use core::arch::x86_64::*;

    let simd_n = (n / 2) * 2;
    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: caller ensures i + 2 <= n and pointers are valid.
            _mm_loadu_si128(deltas.add(i) as *const __m128i)
        };
        // One-step prefix-sum (wrapping i64 arithmetic).
        let v = unsafe { _mm_add_epi64(v, _mm_slli_si128(v, 8)) };
        // Add inter-block accumulator to all lanes.
        let result = unsafe { _mm_add_epi64(v, _mm_set1_epi64x(acc)) };
        unsafe {
            // SAFETY: out pointer valid for i + 2 elements.
            _mm_storeu_si128(out.add(i) as *mut __m128i, result);
            // Extract element 1 (high 64 bits) as new accumulator.
            acc = _mm_cvtsi128_si64(_mm_unpackhi_epi64(result, result));
        }
        i += 2;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2_i64(initial: i64, deltas: &[i64], out: &mut Vec<i64>) {
    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 2) * 2;

    // SAFETY: pointers derived from valid slices/vecs; capacity reserved above.
    let acc = unsafe { decode_sse2_i64_inner(initial, deltas.as_ptr(), n, out.as_mut_ptr().add(base)) };
    unsafe {
        // SAFETY: elements [base, base + simd_n) were written by decode_sse2_i64_inner.
        out.set_len(base + simd_n);
    }

    let mut acc = acc;
    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_sse2_u64(initial: u64, deltas: &[u64], out: &mut Vec<u64>) {
    // Wrapping add is identical for i64 and u64 at the bit level.
    // SAFETY: u64 and i64 have the same size/alignment; wrapping arithmetic is identical.
    let deltas_i64 = unsafe {
        core::slice::from_raw_parts(deltas.as_ptr() as *const i64, deltas.len())
    };
    let out_i64 = unsafe { &mut *(out as *mut Vec<u64> as *mut Vec<i64>) };
    unsafe { decode_sse2_i64(initial as i64, deltas_i64, out_i64) };
}

// ── AVX2 implementations (x86_64) ─────────────────────────────────────────────

// AVX2 prefix-sum delta decode for i32/u32: 8 values per iteration.
//
// AVX2 has two independent 128-bit lanes; _mm256_slli_si256 shifts each lane
// independently. Steps:
//   1. Intra-lane 2-step prefix sum (same as SSE2 but on 256 bits).
//   2. Carry the sum of the lo lane's element 3 into the hi lane.
//   3. Add inter-block accumulator.
//   4. Extract hi-lane element 3 as new acc.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_avx2_i32_inner(initial: i32, deltas: *const i32, n: usize, out: *mut i32) -> i32 {
    use core::arch::x86_64::*;

    let simd_n = (n / 8) * 8;
    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: caller ensures i + 8 <= n and pointers are valid.
            _mm256_loadu_si256(deltas.add(i) as *const __m256i)
        };

        // Step 1: intra-lane prefix sums (each 128-bit lane independently).
        let v = unsafe { _mm256_add_epi32(v, _mm256_slli_si256(v, 4)) };
        let v = unsafe { _mm256_add_epi32(v, _mm256_slli_si256(v, 8)) };

        // Step 2: carry lo lane's element-3 sum into the hi lane.
        // Extract lo 128-bit lane, broadcast element 3 to all positions.
        let lo128 = unsafe { _mm256_castsi256_si128(v) };
        // _mm_shuffle_epi32 with 0xFF broadcasts element 3.
        let p3 = unsafe { _mm_shuffle_epi32(lo128, 0xFF_u32 as i32) };
        // Place p3 only in hi 128 bits, zero in lo.
        let carry = unsafe { _mm256_set_m128i(p3, _mm_setzero_si128()) };
        let v = unsafe { _mm256_add_epi32(v, carry) };

        // Step 3: add inter-block accumulator to all lanes.
        let result = unsafe { _mm256_add_epi32(v, _mm256_set1_epi32(acc)) };

        unsafe {
            // SAFETY: out pointer valid for i + 8 elements.
            _mm256_storeu_si256(out.add(i) as *mut __m256i, result);
            // Extract hi lane element 3 as new accumulator.
            // _mm256_extracti128_si256(result, 1) gets the hi 128-bit lane.
            // Shuffle broadcasts element 3; cvtsi128_si32 reads low 32 bits.
            let hi128 = _mm256_extracti128_si256(result, 1);
            acc = _mm_cvtsi128_si32(_mm_shuffle_epi32(hi128, 0xFF_u32 as i32));
        }
        i += 8;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_avx2_i32(initial: i32, deltas: &[i32], out: &mut Vec<i32>) {
    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    // SAFETY: pointers derived from valid slices/vecs; capacity reserved above.
    let acc = unsafe { decode_avx2_i32_inner(initial, deltas.as_ptr(), n, out.as_mut_ptr().add(base)) };
    unsafe {
        // SAFETY: elements [base, base + simd_n) were written by decode_avx2_i32_inner.
        out.set_len(base + simd_n);
    }

    // Process remaining with SSE2 then scalar tail.
    let rem = &deltas[simd_n..];
    let sse2_n = (rem.len() / 4) * 4;
    let acc = if sse2_n > 0 {
        // SAFETY: pointers derived from valid slices/vecs; capacity reserved above.
        let a = unsafe {
            decode_sse2_i32_inner(acc, rem.as_ptr(), rem.len(), out.as_mut_ptr().add(base + simd_n))
        };
        unsafe {
            // SAFETY: elements written by decode_sse2_i32_inner.
            out.set_len(base + simd_n + sse2_n);
        }
        a
    } else {
        acc
    };

    let mut acc = acc;
    for &d in &rem[sse2_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_avx2_u32(initial: u32, deltas: &[u32], out: &mut Vec<u32>) {
    // SAFETY: u32 and i32 have the same size/alignment; wrapping arithmetic is identical.
    let deltas_i32 = unsafe {
        core::slice::from_raw_parts(deltas.as_ptr() as *const i32, deltas.len())
    };
    let out_i32 = unsafe { &mut *(out as *mut Vec<u32> as *mut Vec<i32>) };
    unsafe { decode_avx2_i32(initial as i32, deltas_i32, out_i32) };
}

// AVX2 prefix-sum delta decode for i64/u64: 4 values per iteration.
//
// Similar to i32 AVX2 but 64-bit lanes:
//   1. Intra-lane 1-step prefix sum.
//   2. Carry lo lane's element-1 into hi lane.
//   3. Add acc, extract new acc from hi lane element 1.
#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_avx2_i64_inner(initial: i64, deltas: *const i64, n: usize, out: *mut i64) -> i64 {
    use core::arch::x86_64::*;

    let simd_n = (n / 4) * 4;
    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: caller ensures i + 4 <= n and pointers are valid.
            _mm256_loadu_si256(deltas.add(i) as *const __m256i)
        };

        // Step 1: intra-lane prefix sum (1 step for 2 elements/lane).
        let v = unsafe { _mm256_add_epi64(v, _mm256_slli_si256(v, 8)) };

        // Step 2: carry lo lane's element-1 into hi lane.
        let lo128 = unsafe { _mm256_castsi256_si128(v) };
        // _mm_unpackhi_epi64 duplicates element 1 of lo lane into both positions.
        let p1 = unsafe { _mm_unpackhi_epi64(lo128, lo128) };
        let carry = unsafe { _mm256_set_m128i(p1, _mm_setzero_si128()) };
        let v = unsafe { _mm256_add_epi64(v, carry) };

        // Step 3: add inter-block accumulator.
        let result = unsafe { _mm256_add_epi64(v, _mm256_set1_epi64x(acc)) };

        unsafe {
            // SAFETY: out pointer valid for i + 4 elements.
            _mm256_storeu_si256(out.add(i) as *mut __m256i, result);
            // Extract hi lane element 1 as new accumulator.
            let hi128 = _mm256_extracti128_si256(result, 1);
            acc = _mm_cvtsi128_si64(_mm_unpackhi_epi64(hi128, hi128));
        }
        i += 4;
    }
    acc
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_avx2_i64(initial: i64, deltas: &[i64], out: &mut Vec<i64>) {
    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 4) * 4;

    // SAFETY: pointers derived from valid slices/vecs; capacity reserved above.
    let acc = unsafe { decode_avx2_i64_inner(initial, deltas.as_ptr(), n, out.as_mut_ptr().add(base)) };
    unsafe {
        // SAFETY: elements [base, base + simd_n) were written by decode_avx2_i64_inner.
        out.set_len(base + simd_n);
    }

    // Process remaining with SSE2 then scalar tail.
    let rem = &deltas[simd_n..];
    let sse2_n = (rem.len() / 2) * 2;
    let acc = if sse2_n > 0 {
        // SAFETY: pointers derived from valid slices/vecs; capacity reserved above.
        let a = unsafe {
            decode_sse2_i64_inner(acc, rem.as_ptr(), rem.len(), out.as_mut_ptr().add(base + simd_n))
        };
        unsafe {
            // SAFETY: elements written by decode_sse2_i64_inner.
            out.set_len(base + simd_n + sse2_n);
        }
        a
    } else {
        acc
    };

    let mut acc = acc;
    for &d in &rem[sse2_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
unsafe fn decode_avx2_u64(initial: u64, deltas: &[u64], out: &mut Vec<u64>) {
    // SAFETY: u64 and i64 have the same size/alignment; wrapping arithmetic is identical.
    let deltas_i64 = unsafe {
        core::slice::from_raw_parts(deltas.as_ptr() as *const i64, deltas.len())
    };
    let out_i64 = unsafe { &mut *(out as *mut Vec<u64> as *mut Vec<i64>) };
    unsafe { decode_avx2_i64(initial as i64, deltas_i64, out_i64) };
}

// ── NEON implementations (aarch64) ────────────────────────────────────────────

// NEON prefix-sum delta decode for i16: 8 values per iteration.
//
// Three-step scan using vextq_s16 for element-shift:
//   vextq_s16(zero, v, 7) → [0, v[0], v[1], ..., v[6]]  (shift left 1 element)
//   vextq_s16(zero, v, 6) → [0, 0, v[0], ..., v[5]]     (shift left 2 elements)
//   vextq_s16(zero, v, 4) → [0, 0, 0, 0, v[0], v[1], v[2], v[3]] (shift left 4)
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_neon_i16(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
    use core::arch::aarch64::*;

    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 8) * 8;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: i + 8 <= simd_n <= n; deltas slice bounds are valid.
            vld1q_s16(deltas.as_ptr().add(i))
        };
        let zero = unsafe { vdupq_n_s16(0) };

        // Three-step prefix-sum scan (wrapping i16 arithmetic).
        // vextq_s16(a, b, n): result[k] = if k + n < 8 { a[k+n] } else { b[k+n-8] }
        // So vextq_s16(zero, v, 7) = [zero[7], v[0..7]] = [0, v[0], ..., v[6]]
        let v = unsafe { vaddq_s16(v, vextq_s16(zero, v, 7)) };
        let v = unsafe { vaddq_s16(v, vextq_s16(zero, v, 6)) };
        let v = unsafe { vaddq_s16(v, vextq_s16(zero, v, 4)) };

        // Add inter-block accumulator to all lanes.
        let result = unsafe { vaddq_s16(v, vdupq_n_s16(acc)) };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 8 <= base + n.
            vst1q_s16(out.as_mut_ptr().add(base + i), result);
            // Element 7 is the prefix sum of all 8 deltas + acc = new accumulator.
            acc = vgetq_lane_s16(result, 7);
        }
        i += 8;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    // Scalar tail for n % 8 remaining values.
    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

// NEON prefix-sum delta decode for i32: 4 values per iteration.
//
// Two-step scan:
//   vextq_s32(zero, v, 3) → [0, v[0], v[1], v[2]]   (shift left 1 element)
//   vextq_s32(zero, v, 2) → [0, 0, v[0], v[1]]       (shift left 2 elements)
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_neon_i32(initial: i32, deltas: &[i32], out: &mut Vec<i32>) {
    use core::arch::aarch64::*;

    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 4) * 4;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: i + 4 <= simd_n <= n; deltas slice bounds are valid.
            vld1q_s32(deltas.as_ptr().add(i))
        };
        let zero = unsafe { vdupq_n_s32(0) };

        // Two-step prefix-sum scan (wrapping i32 arithmetic).
        let v = unsafe { vaddq_s32(v, vextq_s32(zero, v, 3)) };
        let v = unsafe { vaddq_s32(v, vextq_s32(zero, v, 2)) };

        // Add inter-block accumulator to all lanes.
        let result = unsafe { vaddq_s32(v, vdupq_n_s32(acc)) };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 4 <= base + n.
            vst1q_s32(out.as_mut_ptr().add(base + i), result);
            acc = vgetq_lane_s32(result, 3);
        }
        i += 4;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

// NEON prefix-sum delta decode for u32: 4 values per iteration.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_neon_u32(initial: u32, deltas: &[u32], out: &mut Vec<u32>) {
    use core::arch::aarch64::*;

    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 4) * 4;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: i + 4 <= simd_n <= n; deltas slice bounds are valid.
            vld1q_u32(deltas.as_ptr().add(i))
        };
        let zero = unsafe { vdupq_n_u32(0) };

        // Two-step prefix-sum scan (wrapping u32 arithmetic).
        let v = unsafe { vaddq_u32(v, vextq_u32(zero, v, 3)) };
        let v = unsafe { vaddq_u32(v, vextq_u32(zero, v, 2)) };

        // Add inter-block accumulator to all lanes.
        let result = unsafe { vaddq_u32(v, vdupq_n_u32(acc)) };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 4 <= base + n.
            vst1q_u32(out.as_mut_ptr().add(base + i), result);
            acc = vgetq_lane_u32(result, 3);
        }
        i += 4;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

// NEON prefix-sum delta decode for i64: 2 values per iteration.
//
// One-step scan:
//   vextq_s64(zero, v, 1) → [0, v[0]]   (shift left 1 element)
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_neon_i64(initial: i64, deltas: &[i64], out: &mut Vec<i64>) {
    use core::arch::aarch64::*;

    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 2) * 2;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: i + 2 <= simd_n <= n; deltas slice bounds are valid.
            vld1q_s64(deltas.as_ptr().add(i))
        };
        let zero = unsafe { vdupq_n_s64(0) };

        // One-step prefix-sum scan (wrapping i64 arithmetic).
        let v = unsafe { vaddq_s64(v, vextq_s64(zero, v, 1)) };

        // Add inter-block accumulator to all lanes.
        let result = unsafe { vaddq_s64(v, vdupq_n_s64(acc)) };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 2 <= base + n.
            vst1q_s64(out.as_mut_ptr().add(base + i), result);
            acc = vgetq_lane_s64(result, 1);
        }
        i += 2;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

// NEON prefix-sum delta decode for u64: 2 values per iteration.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
unsafe fn decode_neon_u64(initial: u64, deltas: &[u64], out: &mut Vec<u64>) {
    use core::arch::aarch64::*;

    let n = deltas.len();
    out.reserve(n);
    let base = out.len();
    let simd_n = (n / 2) * 2;

    let mut acc = initial;
    let mut i = 0usize;

    while i < simd_n {
        let v = unsafe {
            // SAFETY: i + 2 <= simd_n <= n; deltas slice bounds are valid.
            vld1q_u64(deltas.as_ptr().add(i))
        };
        let zero = unsafe { vdupq_n_u64(0) };

        // One-step prefix-sum scan (wrapping u64 arithmetic).
        let v = unsafe { vaddq_u64(v, vextq_u64(zero, v, 1)) };

        // Add inter-block accumulator to all lanes.
        let result = unsafe { vaddq_u64(v, vdupq_n_u64(acc)) };

        unsafe {
            // SAFETY: out.reserve(n) ensures capacity; base + i + 2 <= base + n.
            vst1q_u64(out.as_mut_ptr().add(base + i), result);
            acc = vgetq_lane_u64(result, 1);
        }
        i += 2;
    }
    unsafe {
        // SAFETY: elements [base, base + simd_n) were all written above.
        out.set_len(base + simd_n);
    }

    for &d in &deltas[simd_n..] {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    // ── i16 cross-path tests (SSE2 vs scalar) ────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    fn decode_both_i16(initial: i16, deltas: &[i16]) -> (Vec<i16>, Vec<i16>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2_i16(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_matches_scalar_exact_block() {
        let deltas: Vec<i16> = vec![1, 2, 3, 4, -1, -2, -3, -4];
        let (s, v) = decode_both_i16(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_matches_scalar_with_tail() {
        let deltas: Vec<i16> = vec![10, -5, 3, 0, -100, 200, i16::MAX, i16::MIN, 1, 2, 3];
        let (s, v) = decode_both_i16(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_matches_scalar_nonzero_initial() {
        let deltas: Vec<i16> = (0..40).map(|i| i as i16).collect();
        let (s, v) = decode_both_i16(100, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_matches_scalar_wrapping() {
        let deltas: Vec<i16> = (0..16)
            .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
            .collect();
        let (s, v) = decode_both_i16(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_all_tail_lengths() {
        let pool: Vec<i16> = (0..16).map(|i| (i * 3 - 20) as i16).collect();
        for n in 0..=16usize {
            let (s, v) = decode_both_i16(5, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_accumulator_carry_multiple_blocks() {
        let deltas: Vec<i16> = (0..24).map(|i| (i as i16).wrapping_mul(7)).collect();
        let (s, v) = decode_both_i16(-100, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_accumulator_carry_at_wrap_boundary() {
        let mut deltas = vec![i16::MAX, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
        let (s, v) = decode_both_i16(0, &deltas);
        assert_eq!(s, v);
        deltas = vec![i16::MIN, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0];
        let (s, v) = decode_both_i16(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_monotone_increasing() {
        let deltas = vec![1i16; 33];
        let (s, v) = decode_both_i16(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_all_zero_deltas() {
        let deltas = vec![0i16; 32];
        let (s, v) = decode_both_i16(42, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i16_large_input() {
        let deltas: Vec<i16> = (0..512i32)
            .map(|i| ((i * 31 + 17) % 257 - 128) as i16)
            .collect();
        let (s, v) = decode_both_i16(1000, &deltas);
        assert_eq!(s, v);
    }

    // ── i32 SSE2 cross-path tests ─────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    fn decode_both_i32_sse2(initial: i32, deltas: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2_i32(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i32_exact_block() {
        let deltas: Vec<i32> = vec![1, 2, 3, 4];
        let (s, v) = decode_both_i32_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i32_with_tail() {
        let deltas: Vec<i32> = vec![10, -5, 3, 0, 100, -200, 7];
        let (s, v) = decode_both_i32_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i32_nonzero_initial() {
        let deltas: Vec<i32> = (0..40).map(|i| i as i32).collect();
        let (s, v) = decode_both_i32_sse2(1000, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i32_wrapping() {
        let deltas: Vec<i32> = (0..8)
            .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
            .collect();
        let (s, v) = decode_both_i32_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i32_all_tail_lengths() {
        let pool: Vec<i32> = (0..8).map(|i| (i * 3 - 10) as i32).collect();
        for n in 0..=8usize {
            let (s, v) = decode_both_i32_sse2(5, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i32_multiple_blocks() {
        let deltas: Vec<i32> = (0..12).map(|i| (i as i32).wrapping_mul(7)).collect();
        let (s, v) = decode_both_i32_sse2(-100, &deltas);
        assert_eq!(s, v);
    }

    // ── u32 SSE2 cross-path tests ─────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    fn decode_both_u32_sse2(initial: u32, deltas: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2_u32(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_u32_sorted() {
        let deltas: Vec<u32> = vec![100, 200, 150, 500];
        let (s, v) = decode_both_u32_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_u32_wrapping() {
        let deltas: Vec<u32> = vec![u32::MAX, 1, u32::MAX, 1, u32::MAX, 1, u32::MAX, 1];
        let (s, v) = decode_both_u32_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_u32_all_tail_lengths() {
        let pool: Vec<u32> = (0..8u32).map(|i| i * 100).collect();
        for n in 0..=8usize {
            let (s, v) = decode_both_u32_sse2(50, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    // ── i64 SSE2 cross-path tests ─────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    fn decode_both_i64_sse2(initial: i64, deltas: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2_i64(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i64_exact_block() {
        let deltas: Vec<i64> = vec![1, 2];
        let (s, v) = decode_both_i64_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i64_with_tail() {
        let deltas: Vec<i64> = vec![10, -5, 3];
        let (s, v) = decode_both_i64_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i64_nonzero_initial() {
        let deltas: Vec<i64> = (0..20).map(|i| i as i64).collect();
        let (s, v) = decode_both_i64_sse2(1000, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i64_wrapping() {
        let deltas: Vec<i64> = vec![i64::MAX, i64::MIN, i64::MAX, i64::MIN];
        let (s, v) = decode_both_i64_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_i64_all_tail_lengths() {
        let pool: Vec<i64> = (0..4).map(|i| (i * 100 - 150) as i64).collect();
        for n in 0..=4usize {
            let (s, v) = decode_both_i64_sse2(5, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    // ── u64 SSE2 cross-path tests ─────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    fn decode_both_u64_sse2(initial: u64, deltas: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: SSE2 is always available on x86_64.
        unsafe { decode_sse2_u64(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_u64_sorted() {
        let deltas: Vec<u64> = vec![1_000_000, 2_000_000];
        let (s, v) = decode_both_u64_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_u64_wrapping() {
        let deltas: Vec<u64> = vec![u64::MAX, 1, u64::MAX, 1];
        let (s, v) = decode_both_u64_sse2(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"))]
    #[test]
    fn sse2_u64_all_tail_lengths() {
        let pool: Vec<u64> = (0..4u64).map(|i| i * 1_000_000_000).collect();
        for n in 0..=4usize {
            let (s, v) = decode_both_u64_sse2(100, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    // ── AVX2 cross-path tests ─────────────────────────────────────────────────

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    fn decode_both_i32_avx2(initial: i32, deltas: &[i32]) -> Option<(Vec<i32>, Vec<i32>)> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: AVX2 confirmed by is_x86_feature_detected! above.
        unsafe { decode_avx2_i32(initial, deltas, &mut simd_out) };
        Some((scalar_out, simd_out))
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i32_exact_block() {
        let deltas: Vec<i32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        if let Some((s, v)) = decode_both_i32_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i32_with_tail() {
        let deltas: Vec<i32> = vec![10, -5, 3, 0, 100, -200, 7, 3, 1, 2, 3];
        if let Some((s, v)) = decode_both_i32_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i32_nonzero_initial() {
        let deltas: Vec<i32> = (0..40).map(|i| i as i32).collect();
        if let Some((s, v)) = decode_both_i32_avx2(1000, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i32_wrapping() {
        let deltas: Vec<i32> = (0..16)
            .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
            .collect();
        if let Some((s, v)) = decode_both_i32_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i32_all_tail_lengths() {
        let pool: Vec<i32> = (0..16).map(|i| (i * 3 - 10) as i32).collect();
        for n in 0..=16usize {
            if let Some((s, v)) = decode_both_i32_avx2(5, &pool[..n]) {
                assert_eq!(s, v, "tail n={n}");
            }
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    fn decode_both_u32_avx2(initial: u32, deltas: &[u32]) -> Option<(Vec<u32>, Vec<u32>)> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: AVX2 confirmed by is_x86_feature_detected! above.
        unsafe { decode_avx2_u32(initial, deltas, &mut simd_out) };
        Some((scalar_out, simd_out))
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_u32_sorted() {
        let deltas: Vec<u32> = (0..16u32).map(|i| i * 100).collect();
        if let Some((s, v)) = decode_both_u32_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_u32_wrapping() {
        let deltas: Vec<u32> = (0..8).map(|i| if i % 2 == 0 { u32::MAX } else { 1 }).collect();
        if let Some((s, v)) = decode_both_u32_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    fn decode_both_i64_avx2(initial: i64, deltas: &[i64]) -> Option<(Vec<i64>, Vec<i64>)> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: AVX2 confirmed by is_x86_feature_detected! above.
        unsafe { decode_avx2_i64(initial, deltas, &mut simd_out) };
        Some((scalar_out, simd_out))
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i64_exact_block() {
        let deltas: Vec<i64> = vec![1, 2, 3, 4];
        if let Some((s, v)) = decode_both_i64_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i64_with_tail() {
        let deltas: Vec<i64> = vec![10, -5, 3, 0, 100];
        if let Some((s, v)) = decode_both_i64_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i64_nonzero_initial() {
        let deltas: Vec<i64> = (0..20).map(|i| i as i64).collect();
        if let Some((s, v)) = decode_both_i64_avx2(1000, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_i64_wrapping() {
        let deltas: Vec<i64> = (0..8)
            .map(|i| if i % 2 == 0 { i64::MAX } else { i64::MIN })
            .collect();
        if let Some((s, v)) = decode_both_i64_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    fn decode_both_u64_avx2(initial: u64, deltas: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
        if !is_x86_feature_detected!("avx2") {
            return None;
        }
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: AVX2 confirmed by is_x86_feature_detected! above.
        unsafe { decode_avx2_u64(initial, deltas, &mut simd_out) };
        Some((scalar_out, simd_out))
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_u64_sorted() {
        let deltas: Vec<u64> = (0..8u64).map(|i| i * 1_000_000_000).collect();
        if let Some((s, v)) = decode_both_u64_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    #[cfg(all(any(feature = "simd-auto", feature = "simd-avx2"), target_arch = "x86_64", feature = "std"))]
    #[test]
    fn avx2_u64_wrapping() {
        let deltas: Vec<u64> = (0..4).map(|i| if i % 2 == 0 { u64::MAX } else { 1 }).collect();
        if let Some((s, v)) = decode_both_u64_avx2(0, &deltas) {
            assert_eq!(s, v);
        }
    }

    // ── NEON cross-path tests (aarch64) ───────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    fn decode_both_neon_i16(initial: i16, deltas: &[i16]) -> (Vec<i16>, Vec<i16>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i16(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i16_matches_scalar_exact_block() {
        let deltas: Vec<i16> = vec![1, 2, 3, 4, -1, -2, -3, -4];
        let (s, v) = decode_both_neon_i16(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i16_matches_scalar_with_tail() {
        let deltas: Vec<i16> = vec![10, -5, 3, 0, -100, 200, i16::MAX, i16::MIN, 1, 2, 3];
        let (s, v) = decode_both_neon_i16(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i16_nonzero_initial() {
        let deltas: Vec<i16> = (0..40).map(|i| i as i16).collect();
        let (s, v) = decode_both_neon_i16(100, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i16_all_tail_lengths() {
        let pool: Vec<i16> = (0..16).map(|i| (i * 3 - 20) as i16).collect();
        for n in 0..=16usize {
            let (s, v) = decode_both_neon_i16(5, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i16_large_input() {
        let deltas: Vec<i16> = (0..512i32)
            .map(|i| ((i * 31 + 17) % 257 - 128) as i16)
            .collect();
        let (s, v) = decode_both_neon_i16(1000, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    fn decode_both_neon_i32(initial: i32, deltas: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i32(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i32_exact_block() {
        let deltas: Vec<i32> = vec![1, 2, 3, 4];
        let (s, v) = decode_both_neon_i32(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i32_with_tail() {
        let deltas: Vec<i32> = vec![10, -5, 3, 0, 100, -200, 7];
        let (s, v) = decode_both_neon_i32(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i32_nonzero_initial() {
        let deltas: Vec<i32> = (0..40).map(|i| i as i32).collect();
        let (s, v) = decode_both_neon_i32(1000, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i32_all_tail_lengths() {
        let pool: Vec<i32> = (0..8).map(|i| (i * 3 - 10) as i32).collect();
        for n in 0..=8usize {
            let (s, v) = decode_both_neon_i32(5, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn decode_both_neon_u32(initial: u32, deltas: &[u32]) -> (Vec<u32>, Vec<u32>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_u32(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u32_sorted() {
        let deltas: Vec<u32> = vec![100, 200, 150, 500];
        let (s, v) = decode_both_neon_u32(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u32_wrapping() {
        let deltas: Vec<u32> = vec![u32::MAX, 1, u32::MAX, 1, u32::MAX, 1, u32::MAX, 1];
        let (s, v) = decode_both_neon_u32(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u32_all_tail_lengths() {
        let pool: Vec<u32> = (0..8u32).map(|i| i * 100).collect();
        for n in 0..=8usize {
            let (s, v) = decode_both_neon_u32(50, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn decode_both_neon_i64(initial: i64, deltas: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_i64(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i64_exact_block() {
        let deltas: Vec<i64> = vec![1, 2];
        let (s, v) = decode_both_neon_i64(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i64_with_tail() {
        let deltas: Vec<i64> = vec![10, -5, 3];
        let (s, v) = decode_both_neon_i64(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i64_nonzero_initial() {
        let deltas: Vec<i64> = (0..20).map(|i| i as i64).collect();
        let (s, v) = decode_both_neon_i64(1000, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_i64_wrapping() {
        let deltas: Vec<i64> = vec![i64::MAX, i64::MIN, i64::MAX, i64::MIN];
        let (s, v) = decode_both_neon_i64(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    fn decode_both_neon_u64(initial: u64, deltas: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let mut scalar_out = Vec::new();
        let mut acc = initial;
        for &d in deltas {
            acc = acc.wrapping_add(d);
            scalar_out.push(acc);
        }
        let mut simd_out = Vec::new();
        // SAFETY: NEON is mandatory on AArch64.
        unsafe { decode_neon_u64(initial, deltas, &mut simd_out) };
        (scalar_out, simd_out)
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u64_sorted() {
        let deltas: Vec<u64> = vec![1_000_000, 2_000_000];
        let (s, v) = decode_both_neon_u64(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u64_wrapping() {
        let deltas: Vec<u64> = vec![u64::MAX, 1, u64::MAX, 1];
        let (s, v) = decode_both_neon_u64(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_u64_all_tail_lengths() {
        let pool: Vec<u64> = (0..4u64).map(|i| i * 1_000_000_000).collect();
        for n in 0..=4usize {
            let (s, v) = decode_both_neon_u64(100, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    // ── i16 roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(&[] as &[i16])), &[] as &[i16]);
    }

    #[test]
    fn roundtrip_single() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX] {
            assert_eq!(decode(&encode(&[v])), &[v]);
        }
    }

    #[test]
    fn roundtrip_sequence() {
        let samples: Vec<i16> = (-128..=127).collect();
        assert_eq!(decode(&encode(&samples)), samples);
    }

    #[test]
    fn encode_produces_differences() {
        let samples = [10i16, 20, 15, 30];
        let deltas = encode(&samples);
        assert_eq!(deltas, [10, 10, -5, 15]);
    }

    #[test]
    fn encode_wraps_on_overflow() {
        let samples = [i16::MAX, i16::MIN];
        let deltas = encode(&samples);
        assert_eq!(deltas[0], i16::MAX);
        assert_eq!(deltas[1], i16::MIN.wrapping_sub(i16::MAX));
        assert_eq!(decode(&deltas), samples);
    }

    #[test]
    fn encode_with_initial_nonzero() {
        let samples = [10i16, 20, 30];
        let deltas = encode_with_initial(5, &samples);
        assert_eq!(deltas, [5, 10, 10]);
        assert_eq!(decode_with_initial(5, &deltas), samples);
    }

    #[test]
    fn encode_into_appends() {
        let mut out = vec![99i16];
        encode_into(&[3i16, 6, 9], &mut out);
        assert_eq!(out, [99, 3, 3, 3]);
    }

    // ── u32 roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn u32_roundtrip_sorted() {
        let values: Vec<u32> = vec![100, 200, 350, 700, 1000];
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn u32_roundtrip_wrapping() {
        let values: Vec<u32> = vec![10, 5, u32::MAX, 0];
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn u32_encode_produces_differences() {
        let values: Vec<u32> = vec![100u32, 200, 350];
        let deltas = encode(&values);
        assert_eq!(deltas, [100u32, 100, 150]);
    }

    #[test]
    fn u32_encode_with_initial() {
        let values: Vec<u32> = vec![200u32, 300, 500];
        let deltas = encode_with_initial(100u32, &values);
        assert_eq!(deltas, [100u32, 100, 200]);
        assert_eq!(decode_with_initial(100u32, &deltas), values);
    }

    // ── u64 roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn u64_roundtrip_sorted() {
        let values: Vec<u64> = vec![0, 1_000_000, 1_000_000_000, u64::MAX / 2];
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn u64_roundtrip_wrapping() {
        let values: Vec<u64> = vec![u64::MAX, 0, u64::MAX];
        assert_eq!(decode(&encode(&values)), values);
    }

    // ── i32 roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn i32_roundtrip() {
        let values: Vec<i32> = vec![-1000, 500, i32::MIN, i32::MAX, 0];
        assert_eq!(decode(&encode(&values)), values);
    }

    #[test]
    fn i32_encode_produces_differences() {
        let values: Vec<i32> = vec![10i32, 20, 15, 30];
        let deltas = encode(&values);
        assert_eq!(deltas, [10i32, 10, -5, 15]);
    }

    // ── i64 roundtrip ─────────────────────────────────────────────────────────

    #[test]
    fn i64_roundtrip() {
        let values: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX];
        assert_eq!(decode(&encode(&values)), values);
    }
}
