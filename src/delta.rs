#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

pub fn encode(samples: &[i16]) -> Vec<i16> {
    encode_with_initial(0, samples)
}

pub fn encode_with_initial(initial: i16, samples: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(samples.len());
    encode_with_initial_into(initial, samples, &mut out);
    out
}

pub fn encode_into(samples: &[i16], out: &mut Vec<i16>) {
    encode_with_initial_into(0, samples, out);
}

pub fn decode(deltas: &[i16]) -> Vec<i16> {
    decode_with_initial(0, deltas)
}

pub fn decode_with_initial(initial: i16, deltas: &[i16]) -> Vec<i16> {
    let mut out = Vec::with_capacity(deltas.len());
    decode_with_initial_into(initial, deltas, &mut out);
    out
}

pub fn decode_into(deltas: &[i16], out: &mut Vec<i16>) {
    decode_with_initial_into(0, deltas, out);
}

fn encode_with_initial_into(initial: i16, samples: &[i16], out: &mut Vec<i16>) {
    let mut prev = initial;
    for &s in samples {
        out.push(s.wrapping_sub(prev));
        prev = s;
    }
}

fn decode_with_initial_into(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    decode_sse2(initial, deltas, out);
    #[cfg(not(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    )))]
    decode_scalar(initial, deltas, out);
}

#[cfg_attr(
    all(any(feature = "simd-auto", feature = "simd-sse2"), target_arch = "x86_64"),
    allow(dead_code)
)]
fn decode_scalar(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
    let mut acc = initial;
    for &d in deltas {
        acc = acc.wrapping_add(d);
        out.push(acc);
    }
}

// SSE2 prefix-sum delta decode: 8 i16 values per iteration.
//
// Three-step scan builds all 8 prefix sums in-register:
//   v += shl_1(v)  →  pairwise running sums
//   v += shl_2(v)  →  4-element running sums
//   v += shl_4(v)  →  8-element prefix sums (all starting from d0)
// Then add the inter-block accumulator `acc` to all 8 lanes and extract
// element 7 (the cumulative sum of all 8 deltas + acc) as the new accumulator.
#[cfg(all(
    any(feature = "simd-auto", feature = "simd-sse2"),
    target_arch = "x86_64"
))]
fn decode_sse2(initial: i16, deltas: &[i16], out: &mut Vec<i16>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    // Cross-path: verify SSE2 and scalar produce identical output.
    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    fn decode_both(initial: i16, deltas: &[i16]) -> (Vec<i16>, Vec<i16>) {
        let mut scalar_out = Vec::new();
        decode_scalar(initial, deltas, &mut scalar_out);
        let mut simd_out = Vec::new();
        decode_sse2(initial, deltas, &mut simd_out);
        (scalar_out, simd_out)
    }

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_matches_scalar_exact_block() {
        // Exactly 8 values — exercises the SIMD loop with no tail.
        let deltas: Vec<i16> = vec![1, 2, 3, 4, -1, -2, -3, -4];
        let (s, v) = decode_both(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_matches_scalar_with_tail() {
        // 11 values — 8 via SIMD, 3 via scalar tail.
        let deltas: Vec<i16> = vec![10, -5, 3, 0, -100, 200, i16::MAX, i16::MIN, 1, 2, 3];
        let (s, v) = decode_both(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_matches_scalar_nonzero_initial() {
        let deltas: Vec<i16> = (0..40).map(|i| i as i16).collect();
        let (s, v) = decode_both(100, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_matches_scalar_wrapping() {
        // Wrapping arithmetic: alternating MAX/MIN deltas.
        let deltas: Vec<i16> = (0..16)
            .map(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN })
            .collect();
        let (s, v) = decode_both(0, &deltas);
        assert_eq!(s, v);
    }

    // ── all tail lengths ──────────────────────────────────────────────────────
    //
    // n = 0..=16 covers n%8 = 0,1,2,3,4,5,6,7 for both one-block (n<8) and
    // two-block (8≤n≤16) cases, exercising the scalar tail for every residue.

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_all_tail_lengths() {
        let pool: Vec<i16> = (0..16).map(|i| (i * 3 - 20) as i16).collect();
        for n in 0..=16usize {
            let (s, v) = decode_both(5, &pool[..n]);
            assert_eq!(s, v, "tail n={n}");
        }
    }

    // ── inter-block accumulator carry ─────────────────────────────────────────
    //
    // Verifies that the value at element 7 of block N becomes the initial
    // accumulator for block N+1.

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_accumulator_carry_multiple_blocks() {
        // 3 full blocks (24 values) with initial = -100.
        let deltas: Vec<i16> = (0..24).map(|i| (i as i16).wrapping_mul(7)).collect();
        let (s, v) = decode_both(-100, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_accumulator_carry_at_wrap_boundary() {
        // Block 1 ends with prefix-sum = i16::MAX; block 2 starts adding from there.
        // Craft block 1 so its cumulative sum = i16::MAX with initial=0:
        //   deltas = [i16::MAX, 0, 0, 0, 0, 0, 0, 0]
        // Then block 2 delta[0] = 1 should wrap to i16::MIN.
        let mut deltas = vec![i16::MAX, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0];
        let (s, v) = decode_both(0, &deltas);
        assert_eq!(s, v);

        // Reverse: block 1 ends at i16::MIN, block 2 delta[0] = -1 wraps to i16::MAX.
        deltas = vec![i16::MIN, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0];
        let (s, v) = decode_both(0, &deltas);
        assert_eq!(s, v);
    }

    // ── special sequences ─────────────────────────────────────────────────────

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_monotone_increasing() {
        // All deltas = 1: output should be initial+1, initial+2, ...
        let deltas = vec![1i16; 33]; // 4 full blocks + 1 scalar tail
        let (s, v) = decode_both(0, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_all_zero_deltas() {
        // All deltas = 0: every output equals initial.
        let deltas = vec![0i16; 32];
        let (s, v) = decode_both(42, &deltas);
        assert_eq!(s, v);
    }

    #[cfg(all(
        any(feature = "simd-auto", feature = "simd-sse2"),
        target_arch = "x86_64"
    ))]
    #[test]
    fn sse2_large_input() {
        // 512 values — many SIMD blocks.
        let deltas: Vec<i16> = (0..512i32)
            .map(|i| ((i * 31 + 17) % 257 - 128) as i16)
            .collect();
        let (s, v) = decode_both(1000, &deltas);
        assert_eq!(s, v);
    }

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(&[])), &[] as &[i16]);
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
        // i16::MIN - i16::MAX wraps
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
}
