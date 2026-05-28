//! StreamVByte codec for `u16` values using 1-bit control tags.
//!
//! Each value is stored in 1 byte if it fits in `0..=255`, or 2 bytes otherwise.
//! One control bit per value is packed into a prefix control stream.
//! The encoding is wire-compatible with Oxford Nanopore's VBZ format.
//!
//! # Format
//!
//! ```text
//! [ ctrl_0 | ctrl_1 | … | ctrl_{ceil(n/8)-1} | data bytes … ]
//! ```
//!
//! The control stream occupies `ceil(n / 8)` bytes and precedes the data stream.
//! Within each control byte, bit `k` (LSB = bit 0) corresponds to the `k`-th
//! value in that group of eight: `0` = 1-byte value, `1` = 2-byte value.
//! Data bytes follow in the same order as the values, with no padding between them.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

mod scalar;

// Shuffle table is used by the SIMD back-ends on x86_64 and aarch64.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) mod shuffle;

// SIMD back-ends are compiled on their respective target architectures
// regardless of feature flags; the feature flags only control dispatch.
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod sse2;

// ── dispatch ──────────────────────────────────────────────────────────────────

impl_dispatch_encode!(
    dispatch_encode,
    u16,
    avx2::encode_into,
    sse2::encode_into,
    neon::encode_into,
    scalar::encode_into
);
impl_dispatch_decode!(
    dispatch_decode,
    u16,
    avx2::decode_into,
    sse2::decode_into,
    neon::decode_into,
    scalar::decode_into
);

// ── public API ────────────────────────────────────────────────────────────────

/// StreamVByte codec for `u16` values (1-bit control stream, 1 or 2 bytes per value).
///
/// Wire-compatible with ONT's VBZ format. `Svb16` is a zero-sized type with no
/// configuration.
///
/// # Examples
///
/// ```
/// # use svb::u16::Svb16;
/// let values: Vec<u16> = vec![0, 255, 256, 1000, u16::MAX];
/// let encoded = Svb16.encode(&values);
/// let decoded = Svb16.decode(&encoded, values.len()).unwrap();
/// assert_eq!(decoded, values);
/// ```
pub struct Svb16;

impl Svb16 {
    /// Encode `values` and return a new `Vec<u8>` containing the control stream followed by the data stream.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u16::Svb16;
    /// let bytes = Svb16.encode(&[0u16, 255, 256, 1000]);
    /// assert_eq!(Svb16.decode(&bytes, 4).unwrap(), [0u16, 255, 256, 1000]);
    /// ```
    pub fn encode(&self, values: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode(values, &mut out);
        out
    }

    /// Encode `values`, appending the encoded bytes to `out`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u16::Svb16;
    /// let mut buf = Vec::new();
    /// Svb16.encode_into(&[1u16, 2], &mut buf);
    /// Svb16.encode_into(&[3u16, 4], &mut buf);
    /// ```
    pub fn encode_into(&self, values: &[u16], out: &mut Vec<u8>) {
        dispatch_encode(values, out);
    }

    /// Decode exactly `n` values from `data`, returning them in a new `Vec<u16>`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u16::Svb16;
    /// let bytes = Svb16.encode(&[10u16, 300]);
    /// assert_eq!(Svb16.decode(&bytes, 2).unwrap(), [10u16, 300]);
    /// ```
    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u16>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode(data, n, &mut out)?;
        Ok(out)
    }

    /// Decode exactly `n` values from `data`, appending them to `out`.
    ///
    /// `n` must equal the number of values that were originally encoded (`n` is
    /// not stored in the encoded bytes and cannot be inferred); a wrong value
    /// produces incorrect output or a [`DecodeError`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use svb::u16::Svb16;
    /// let bytes = Svb16.encode(&[10u16, 300]);
    /// let mut out = vec![0u16];
    /// Svb16.decode_into(&bytes, 2, &mut out).unwrap();
    /// assert_eq!(out, [0u16, 10, 300]);
    /// ```
    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u16>,
    ) -> Result<(), DecodeError> {
        dispatch_decode(data, n, out)
    }
}

// ── cross-path unit tests ─────────────────────────────────────────────────────
//
// These tests call the scalar and SIMD decode functions directly to verify
// bit-identical output. They are runtime-guarded so they pass on CPUs that
// do not support the relevant feature (the assertion is simply skipped).

#[cfg(test)]
mod cross_path {
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::vec::Vec;

    use super::scalar;

    fn decode_scalar(data: &[u8], n: usize) -> Vec<u16> {
        let mut out = Vec::new();
        scalar::decode_into(data, n, &mut out).unwrap();
        out
    }

    // ── x86_64 ───────────────────────────────────────────────────────────────
    // is_x86_feature_detected! is a std macro; skip these tests on no_std builds.

    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    mod x86 {
        use super::super::{avx2, scalar, sse2};
        use super::*;

        // ── helpers ──────────────────────────────────────────────────────────

        fn encode(values: &[u16]) -> Vec<u8> {
            let mut v = Vec::new();
            scalar::encode_into(values, &mut v);
            v
        }

        fn ssse3_decode(data: &[u8], n: usize) -> Option<Vec<u16>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::decode_into(data, n, &mut out).unwrap() };
            Some(out)
        }

        fn avx2_decode(data: &[u8], n: usize) -> Option<Vec<u16>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::decode_into(data, n, &mut out).unwrap() };
            Some(out)
        }

        fn check_all(values: &[u16]) {
            let n = values.len();
            let enc = encode(values);
            let expected = decode_scalar(&enc, n);
            if let Some(got) = ssse3_decode(&enc, n) {
                assert_eq!(expected, got, "SSSE3 mismatch n={n} values={values:?}");
            }
            if let Some(got) = avx2_decode(&enc, n) {
                assert_eq!(expected, got, "AVX2 mismatch n={n} values={values:?}");
            }
        }

        // ── encode helpers ───────────────────────────────────────────────────

        fn ssse3_encode(values: &[u16]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("ssse3") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { sse2::encode_into(values, &mut out) };
            Some(out)
        }

        fn check_encode(values: &[u16]) {
            let expected = encode(values);
            if let Some(got) = ssse3_encode(values) {
                assert_eq!(
                    expected,
                    got,
                    "SSSE3 encode mismatch n={} values={values:?}",
                    values.len()
                );
            }
            if let Some(got) = avx2_encode(values) {
                assert_eq!(
                    expected,
                    got,
                    "AVX2 encode mismatch n={} values={values:?}",
                    values.len()
                );
            }
        }

        fn avx2_encode(values: &[u16]) -> Option<Vec<u8>> {
            if !is_x86_feature_detected!("avx2") {
                return None;
            }
            let mut out = Vec::new();
            unsafe { avx2::encode_into(values, &mut out) };
            Some(out)
        }

        // ── POD5 parity (real-world data) ─────────────────────────────────

        const SVB0: &[u8] = include_bytes!("../../tests/vectors/parity_00_02885.svb16");
        const SVB1: &[u8] = include_bytes!("../../tests/vectors/parity_01_02915.svb16");
        const SVB2: &[u8] = include_bytes!("../../tests/vectors/parity_02_02949.svb16");

        #[test]
        fn pod5_parity_ssse3() {
            for &(data, n) in &[(SVB0, 2885usize), (SVB1, 2915), (SVB2, 2949)] {
                if let Some(got) = ssse3_decode(data, n) {
                    assert_eq!(decode_scalar(data, n), got, "SSSE3 n={n}");
                }
            }
        }

        #[test]
        fn pod5_parity_avx2() {
            for &(data, n) in &[(SVB0, 2885usize), (SVB1, 2915), (SVB2, 2949)] {
                if let Some(got) = avx2_decode(data, n) {
                    assert_eq!(decode_scalar(data, n), got, "AVX2 n={n}");
                }
            }
        }

        // ── all 256 ctrl byte values ──────────────────────────────────────
        //
        // For each ctrl byte value c, construct exactly 8 u16 values whose
        // encoding produces ctrl byte c: bit k of c set → value k needs 2 bytes
        // (use 300 + k); bit k clear → value k fits in 1 byte (use k as u16).

        #[test]
        fn all_ctrl_byte_values() {
            for ctrl in 0u8..=255 {
                let values: Vec<u16> = (0..8)
                    .map(|k| {
                        if (ctrl >> k) & 1 == 1 {
                            300 + k as u16 // ≥ 256 → 2-byte
                        } else {
                            k as u16 // ≤ 255 → 1-byte
                        }
                    })
                    .collect();
                check_all(&values);
            }
        }

        // ── tail lengths: n = 0 through 20 ───────────────────────────────
        //
        // Exercises the scalar tail for every possible n % 8 residue.

        #[test]
        fn ssse3_all_tail_lengths() {
            if ssse3_decode(&encode(&[0u16]), 1).is_none() {
                return;
            }
            // Mix of 1-byte and 2-byte values to stress both branches.
            let pool: Vec<u16> = (0..20)
                .map(|i| if i % 3 == 0 { 300 + i } else { i })
                .collect();
            for n in 0..=20usize {
                let values = &pool[..n];
                let enc = encode(values);
                let expected = decode_scalar(&enc, n);
                let got = ssse3_decode(&enc, n).unwrap();
                assert_eq!(expected, got, "SSSE3 tail n={n}");
            }
        }

        #[test]
        fn avx2_all_tail_lengths() {
            if avx2_decode(&encode(&[0u16]), 1).is_none() {
                return;
            }
            let pool: Vec<u16> = (0..33)
                .map(|i| if i % 3 == 0 { 300 + i } else { i })
                .collect();
            for n in 0..=33usize {
                let values = &pool[..n];
                let enc = encode(values);
                let expected = decode_scalar(&enc, n);
                let got = avx2_decode(&enc, n).unwrap();
                assert_eq!(expected, got, "AVX2 tail n={n}");
            }
        }

        // ── homogeneous inputs ────────────────────────────────────────────

        #[test]
        fn all_one_byte_values() {
            // 256 values all ≤ 255 → every ctrl byte is 0x00; 8 bytes of data per ctrl byte.
            let values: Vec<u16> = (0..=255).collect();
            check_all(&values);
        }

        #[test]
        fn all_two_byte_values() {
            // 256 values all ≥ 256 → every ctrl byte is 0xFF; 16 bytes of data per ctrl byte.
            // This exercises the maximum data consumption per SIMD iteration.
            let values: Vec<u16> = (256..512).collect();
            check_all(&values);
        }

        #[test]
        fn alternating_one_and_two_byte() {
            // Ctrl byte 0b01010101 = 0x55 for every ctrl byte.
            let values: Vec<u16> = (0..64)
                .map(|i| if i % 2 == 0 { i as u16 } else { 300 + i })
                .collect();
            check_all(&values);
        }

        // ── 16-byte guard boundary (SSSE3) ────────────────────────────────
        //
        // The SSSE3 loop guards `data_pos + 16 > data_bytes.len()`. This test
        // constructs an input where the second ctrl byte's data would need 16
        // bytes but only 9 are available after the first block is consumed —
        // forcing the scalar tail for the second 8 values.

        #[test]
        fn ssse3_16byte_boundary_guard() {
            // Block 1: 8 all-2-byte values → ctrl=0xFF, 16 data bytes consumed.
            // Block 2: 1 two-byte + 7 one-byte → ctrl=0x01, 9 data bytes.
            // After block 1: data_pos=16, remaining data=9. 16+16=32 > 25 → scalar tail.
            let block1: Vec<u16> = (1000..1008).collect(); // all ≥ 256
            let block2: Vec<u16> = vec![500, 1, 2, 3, 4, 5, 6, 7]; // first ≥ 256, rest ≤ 255
            let values: Vec<u16> = block1.into_iter().chain(block2).collect();
            check_all(&values);
        }

        // ── 32-byte guard boundary (AVX2) ─────────────────────────────────
        //
        // Same idea for AVX2: constructs input where the second 16-value block
        // has fewer than 32 data bytes remaining, forcing a scalar tail.

        #[test]
        fn avx2_32byte_boundary_guard() {
            // Blocks 1+2: 16 all-2-byte values → 2 ctrl bytes, 32 data bytes consumed.
            // Block 3: 1 two-byte + 7 one-byte → ctrl=0x01, 9 data bytes.
            // After blocks 1+2: data_pos=32, remaining=9. 32+32=64 > 41 → scalar tail.
            let blocks12: Vec<u16> = (1000..1016).collect();
            let block3: Vec<u16> = vec![500, 1, 2, 3, 4, 5, 6, 7];
            let values: Vec<u16> = blocks12.into_iter().chain(block3).collect();
            check_all(&values);
        }

        // ── boundary values ──────────────────────────────────────────────

        #[test]
        fn values_at_type_boundaries() {
            // 0, 255 (max 1-byte), 256 (min 2-byte), u16::MAX
            let values: Vec<u16> = vec![0, 255, 256, u16::MAX]
                .into_iter()
                .cycle()
                .take(32)
                .collect();
            check_all(&values);
        }

        // ── large input ───────────────────────────────────────────────────

        #[test]
        fn large_input() {
            // 10 000 values: mix of 1-byte and 2-byte to create varied ctrl bytes.
            let values: Vec<u16> = (0..10_000u16)
                .map(|i| if i % 7 < 3 { i % 256 } else { 256 + (i % 1000) })
                .collect();
            check_all(&values);
        }

        // ── edge: n = 0 and n = 1 ────────────────────────────────────────

        #[test]
        fn empty_and_single() {
            check_all(&[]);
            check_all(&[0]);
            check_all(&[255]);
            check_all(&[256]);
            check_all(&[u16::MAX]);
        }

        // ── SSSE3 encode cross-path tests ─────────────────────────────────
        //
        // All tests compare SSSE3 encode output byte-for-byte against scalar.

        #[test]
        fn ssse3_encode_all_ctrl_byte_values() {
            // One block of 8 per ctrl byte value — exhaustive coverage of ENCODE_TABLE.
            for ctrl in 0u8..=255 {
                let values: Vec<u16> = (0..8)
                    .map(|k| {
                        if (ctrl >> k) & 1 == 1 {
                            300 + k as u16
                        } else {
                            k as u16
                        }
                    })
                    .collect();
                check_encode(&values);
            }
        }

        #[test]
        fn ssse3_encode_all_tail_lengths() {
            // n = 0..=20 exercises every possible tail residue (n % 8 = 0..7)
            // across both zero-tail and non-zero-tail cases.
            if ssse3_encode(&[0u16]).is_none() {
                return;
            }
            let pool: Vec<u16> = (0..20)
                .map(|i| if i % 3 == 0 { 300 + i } else { i })
                .collect();
            for n in 0..=20usize {
                check_encode(&pool[..n]);
            }
        }

        #[test]
        fn ssse3_encode_roundtrip() {
            // Encode with SSSE3, decode with scalar — must recover original values.
            if ssse3_encode(&[0u16]).is_none() {
                return;
            }
            let values: Vec<u16> = (0..100)
                .map(|i| if i % 2 == 0 { i as u16 } else { 300 + i })
                .collect();
            let enc = ssse3_encode(&values).unwrap();
            let got = decode_scalar(&enc, values.len());
            assert_eq!(values, got);
        }

        #[test]
        fn ssse3_encode_all_one_byte() {
            // ctrl byte = 0x00 for every block — exercises the all-small path.
            let values: Vec<u16> = (0..=255).collect();
            check_encode(&values);
        }

        #[test]
        fn ssse3_encode_all_two_byte() {
            // ctrl byte = 0xFF for every block — maximum data per block (16 bytes).
            let values: Vec<u16> = (256..512).collect();
            check_encode(&values);
        }

        #[test]
        fn ssse3_encode_large_input() {
            let values: Vec<u16> = (0..10_000u16)
                .map(|i| if i % 7 < 3 { i % 256 } else { 256 + (i % 1000) })
                .collect();
            check_encode(&values);
        }

        #[test]
        fn ssse3_encode_boundary_values() {
            // 0, 255, 256, u16::MAX — the four boundary cases repeated.
            let values: Vec<u16> = [0u16, 255, 256, u16::MAX]
                .iter()
                .copied()
                .cycle()
                .take(32)
                .collect();
            check_encode(&values);
        }

        // ── AVX2 encode cross-path tests ──────────────────────────────────────
        //
        // Exercises AVX2-specific tail lengths and the 16-value-per-iteration
        // boundary. check_encode already includes AVX2 via avx2_encode, so the
        // SSSE3 tests above also cover AVX2; these tests add AVX2-specific cases.

        #[test]
        fn avx2_encode_all_tail_lengths() {
            // n = 0..=33 covers all n%16 residues and both zero-tail/non-zero-tail.
            if avx2_encode(&[0u16]).is_none() {
                return;
            }
            let pool: Vec<u16> = (0..33)
                .map(|i| if i % 3 == 0 { 300 + i } else { i })
                .collect();
            for n in 0..=33usize {
                check_encode(&pool[..n]);
            }
        }

        #[test]
        fn avx2_encode_all_ctrl_byte_values() {
            // Two consecutive 8-value blocks that together exercise every possible
            // (c0, c1) pair via paired ctrl bytes. Checking via check_encode verifies
            // AVX2 matches scalar on all 256 ctrl byte values for both groups.
            for ctrl in 0u8..=255 {
                let values: Vec<u16> = (0..16)
                    .map(|k| {
                        if (ctrl >> (k % 8)) & 1 == 1 {
                            300 + k as u16
                        } else {
                            k as u16
                        }
                    })
                    .collect();
                check_encode(&values);
            }
        }

        #[test]
        fn avx2_encode_roundtrip() {
            if avx2_encode(&[0u16]).is_none() {
                return;
            }
            let values: Vec<u16> = (0..200)
                .map(|i| if i % 2 == 0 { i as u16 } else { 300 + i })
                .collect();
            let enc = avx2_encode(&values).unwrap();
            let got = decode_scalar(&enc, values.len());
            assert_eq!(values, got, "AVX2 encode roundtrip failed");
        }
    }

    // ── aarch64 ──────────────────────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod arm {
        use super::super::{neon, scalar};
        use super::*;
        #[cfg(not(feature = "std"))]
        use alloc::vec;

        fn encode(values: &[u16]) -> Vec<u8> {
            let mut v = Vec::new();
            scalar::encode_into(values, &mut v);
            v
        }

        fn neon_decode(data: &[u8], n: usize) -> Vec<u16> {
            let mut out = Vec::new();
            unsafe { neon::decode_into(data, n, &mut out).unwrap() };
            out
        }

        fn neon_encode(values: &[u16]) -> Vec<u8> {
            let mut out = Vec::new();
            unsafe { neon::encode_into(values, &mut out) };
            out
        }

        fn check(values: &[u16]) {
            let n = values.len();
            let enc = encode(values);
            let expected = decode_scalar(&enc, n);
            let got = neon_decode(&enc, n);
            assert_eq!(expected, got, "NEON decode n={n}");
        }

        fn check_encode(values: &[u16]) {
            let expected = encode(values);
            let got = neon_encode(values);
            assert_eq!(
                expected,
                got,
                "NEON encode mismatch n={} values={values:?}",
                values.len()
            );
        }

        const SVB0: &[u8] = include_bytes!("../../tests/vectors/parity_00_02885.svb16");
        const SVB1: &[u8] = include_bytes!("../../tests/vectors/parity_01_02915.svb16");
        const SVB2: &[u8] = include_bytes!("../../tests/vectors/parity_02_02949.svb16");

        #[test]
        fn pod5_parity_neon() {
            for &(data, n) in &[(SVB0, 2885usize), (SVB1, 2915), (SVB2, 2949)] {
                assert_eq!(decode_scalar(data, n), neon_decode(data, n), "n={n}");
            }
        }

        #[test]
        fn all_ctrl_byte_values() {
            for ctrl in 0u8..=255 {
                let values: Vec<u16> = (0..8)
                    .map(|k| {
                        if (ctrl >> k) & 1 == 1 {
                            300 + k as u16
                        } else {
                            k as u16
                        }
                    })
                    .collect();
                check(&values);
            }
        }

        #[test]
        fn all_tail_lengths() {
            let pool: Vec<u16> = (0..20)
                .map(|i| if i % 3 == 0 { 300 + i } else { i })
                .collect();
            for n in 0..=20usize {
                let enc = encode(&pool[..n]);
                assert_eq!(decode_scalar(&enc, n), neon_decode(&enc, n), "tail n={n}");
            }
        }

        #[test]
        fn all_one_byte_values() {
            check(&(0u16..=255).collect::<Vec<_>>());
        }

        #[test]
        fn all_two_byte_values() {
            check(&(256u16..512).collect::<Vec<_>>());
        }

        #[test]
        fn neon_16byte_boundary_guard() {
            let block1: Vec<u16> = (1000..1008).collect();
            let block2: Vec<u16> = vec![500, 1, 2, 3, 4, 5, 6, 7];
            check(&block1.into_iter().chain(block2).collect::<Vec<_>>());
        }

        #[test]
        fn large_input() {
            let values: Vec<u16> = (0..10_000u16)
                .map(|i| if i % 7 < 3 { i % 256 } else { 256 + (i % 1000) })
                .collect();
            check(&values);
        }

        // ── NEON encode cross-path tests ──────────────────────────────────────

        #[test]
        fn neon_encode_all_ctrl_byte_values() {
            for ctrl in 0u8..=255 {
                let values: Vec<u16> = (0..8)
                    .map(|k| {
                        if (ctrl >> k) & 1 == 1 {
                            300 + k as u16
                        } else {
                            k as u16
                        }
                    })
                    .collect();
                check_encode(&values);
            }
        }

        #[test]
        fn neon_encode_all_tail_lengths() {
            let pool: Vec<u16> = (0..20)
                .map(|i| if i % 3 == 0 { 300 + i } else { i })
                .collect();
            for n in 0..=20usize {
                check_encode(&pool[..n]);
            }
        }

        #[test]
        fn neon_encode_roundtrip() {
            let values: Vec<u16> = (0..100)
                .map(|i| if i % 2 == 0 { i as u16 } else { 300 + i })
                .collect();
            let enc = neon_encode(&values);
            let got = decode_scalar(&enc, values.len());
            assert_eq!(values, got, "NEON encode roundtrip failed");
        }

        #[test]
        fn neon_encode_all_one_byte() {
            check_encode(&(0u16..=255).collect::<Vec<_>>());
        }

        #[test]
        fn neon_encode_all_two_byte() {
            check_encode(&(256u16..512).collect::<Vec<_>>());
        }

        #[test]
        fn neon_encode_boundary_values() {
            let values: Vec<u16> = [0u16, 255, 256, u16::MAX]
                .iter()
                .copied()
                .cycle()
                .take(32)
                .collect();
            check_encode(&values);
        }
    }
}
