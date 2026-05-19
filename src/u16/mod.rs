#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

mod scalar;

// Shuffle table is used by the SIMD back-ends on x86_64 and aarch64.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod shuffle;

// SIMD back-ends are compiled on their respective target architectures
// regardless of feature flags; the feature flags only control dispatch.
#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod sse2;
#[cfg(target_arch = "aarch64")]
mod neon;

// ── dispatch ──────────────────────────────────────────────────────────────────

fn dispatch_encode(values: &[u16], out: &mut Vec<u8>) {
    scalar::encode_into(values, out);
}

fn dispatch_decode(data: &[u8], n: usize, out: &mut Vec<u16>) -> Result<(), DecodeError> {
    // Explicit compile-time paths (highest priority, ordered best → worst).
    // Each guard uses a `not(...)` condition on stronger features so that at
    // most one branch is active — preventing unreachable_code warnings when
    // multiple simd-* features are enabled simultaneously.
    #[cfg(all(feature = "simd-avx2", target_arch = "x86_64"))]
    {
        // SAFETY: simd-avx2 feature declares that AVX2 is available at runtime.
        return unsafe { avx2::decode_into(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-sse2",
        not(feature = "simd-avx2"),
        target_arch = "x86_64"
    ))]
    {
        // SAFETY: simd-sse2 feature declares that SSSE3 is available at runtime.
        return unsafe { sse2::decode_into(data, n, out) };
    }

    #[cfg(all(
        feature = "simd-neon",
        not(any(feature = "simd-avx2", feature = "simd-sse2")),
        target_arch = "aarch64"
    ))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { neon::decode_into(data, n, out) };
    }

    // Runtime auto-detection — requires std for is_x86_feature_detected! on x86.
    #[cfg(all(
        feature = "simd-auto",
        not(any(feature = "simd-avx2", feature = "simd-sse2", feature = "simd-neon"))
    ))]
    {
        #[cfg(all(feature = "std", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 confirmed at runtime.
                return unsafe { avx2::decode_into(data, n, out) };
            }
            if is_x86_feature_detected!("ssse3") {
                // SAFETY: SSSE3 confirmed at runtime.
                return unsafe { sse2::decode_into(data, n, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is mandatory on AArch64.
            return unsafe { neon::decode_into(data, n, out) };
        }
    }

    scalar::decode_into(data, n, out)
}

// ── public API ────────────────────────────────────────────────────────────────

/// StreamVByte codec for u16 values (1-bit control stream, 1 or 2 bytes per value).
///
/// Wire-compatible with ONT's VBZ format.
pub struct Svb16;

impl Svb16 {
    pub fn encode(&self, values: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u16], out: &mut Vec<u8>) {
        dispatch_encode(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u16>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u16>,
    ) -> Result<(), DecodeError> {
        dispatch_decode(data, n, out)
    }
}

impl crate::coder::Coder for Svb16 {
    type Elem = u16;

    fn encode_into(&self, values: &[u16], out: &mut Vec<u8>) {
        dispatch_encode(values, out);
    }

    fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<Self::Elem>,
    ) -> Result<(), DecodeError> {
        dispatch_decode(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len(ctrl, n)
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
        use super::*;
        use super::super::{avx2, sse2};

        const SVB0: &[u8] = include_bytes!("../../tests/vectors/parity_00_02885.svb16");
        const SVB1: &[u8] = include_bytes!("../../tests/vectors/parity_01_02915.svb16");
        const SVB2: &[u8] = include_bytes!("../../tests/vectors/parity_02_02949.svb16");

        #[test]
        fn ssse3_matches_scalar() {
            if !is_x86_feature_detected!("ssse3") {
                return;
            }
            for &(data, n) in &[(SVB0, 2885usize), (SVB1, 2915), (SVB2, 2949)] {
                let expected = decode_scalar(data, n);
                let got = unsafe {
                    let mut out = Vec::new();
                    sse2::decode_into(data, n, &mut out).unwrap();
                    out
                };
                assert_eq!(expected, got, "SSSE3 vs scalar mismatch (n={n})");
            }
        }

        #[test]
        fn avx2_matches_scalar() {
            if !is_x86_feature_detected!("avx2") {
                return;
            }
            for &(data, n) in &[(SVB0, 2885usize), (SVB1, 2915), (SVB2, 2949)] {
                let expected = decode_scalar(data, n);
                let got = unsafe {
                    let mut out = Vec::new();
                    avx2::decode_into(data, n, &mut out).unwrap();
                    out
                };
                assert_eq!(expected, got, "AVX2 vs scalar mismatch (n={n})");
            }
        }
    }

    // ── aarch64 ──────────────────────────────────────────────────────────────

    #[cfg(target_arch = "aarch64")]
    mod arm {
        use super::*;
        use super::super::neon;

        const SVB0: &[u8] = include_bytes!("../../tests/vectors/parity_00_02885.svb16");
        const SVB1: &[u8] = include_bytes!("../../tests/vectors/parity_01_02915.svb16");
        const SVB2: &[u8] = include_bytes!("../../tests/vectors/parity_02_02949.svb16");

        #[test]
        fn neon_matches_scalar() {
            for &(data, n) in &[(SVB0, 2885usize), (SVB1, 2915), (SVB2, 2949)] {
                let expected = decode_scalar(data, n);
                let got = unsafe {
                    let mut out = Vec::new();
                    neon::decode_into(data, n, &mut out).unwrap();
                    out
                };
                assert_eq!(expected, got, "NEON vs scalar mismatch (n={n})");
            }
        }
    }
}
