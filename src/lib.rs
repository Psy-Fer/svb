#![cfg_attr(not(feature = "std"), no_std)]
#![deny(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(all(not(feature = "std"), feature = "alloc"))]
extern crate alloc;

pub mod error;
pub use error::DecodeError;

#[cfg(feature = "alloc")]
pub(crate) mod coder;

#[cfg(feature = "alloc")]
pub mod delta;
#[cfg(feature = "alloc")]
pub mod zigzag;

#[cfg(feature = "alloc")]
pub mod u16;
#[cfg(feature = "alloc")]
pub mod u32;
#[cfg(feature = "alloc")]
pub mod u64;

// ── VBZ convenience pipeline ──────────────────────────────────────────────────
//
// Implements the three-stage inner codec used by Oxford Nanopore's POD5 format:
//   encode: i16 samples → delta → zigzag → SVB16 → Vec<u8>
//   decode: Vec<u8> → SVB16 → zigzag → delta → i16 samples
//
// The outer zstd layer is handled by the caller (e.g. pod5-rs).

#[cfg(feature = "alloc")]
pub use vbz::{decode_vbz, decode_vbz_into, encode_vbz, encode_vbz_into};

#[cfg(feature = "alloc")]
mod vbz {
    #[cfg(feature = "std")]
    use std::vec::Vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::error::DecodeError;
    use crate::{delta, u16::Svb16, zigzag};

    /// Encode `i16` samples through delta → zigzag → SVB16.
    /// Returns raw SVB16 bytes ready to pass to zstd.
    pub fn encode_vbz(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_vbz_into(samples, &mut out);
        out
    }

    /// Encode into a caller-supplied buffer, appending the SVB16 bytes.
    pub fn encode_vbz_into(samples: &[i16], out: &mut Vec<u8>) {
        let deltas = delta::encode(samples);
        let codes = zigzag::encode(&deltas);
        Svb16.encode_into(&codes, out);
    }

    /// Decode `n` `i16` samples from SVB16 bytes (after zstd decompression).
    pub fn decode_vbz(data: &[u8], n: usize) -> Result<Vec<i16>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        decode_vbz_into(data, n, &mut out)?;
        Ok(out)
    }

    /// Decode into a caller-supplied buffer, appending the decoded samples.
    pub fn decode_vbz_into(data: &[u8], n: usize, out: &mut Vec<i16>) -> Result<(), DecodeError> {
        let codes = Svb16.decode(data, n)?;
        let deltas = zigzag::decode(&codes);
        delta::decode_into(&deltas, out);
        Ok(())
    }
}

#[cfg(all(test, feature = "alloc"))]
mod vbz_tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode_vbz(&encode_vbz(&[]), 0).unwrap(), &[] as &[i16]);
    }

    #[test]
    fn roundtrip_single() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX] {
            assert_eq!(decode_vbz(&encode_vbz(&[v]), 1).unwrap(), [v]);
        }
    }

    #[test]
    fn roundtrip_flat_signal() {
        // Constant signal: deltas are all zero after the first → compresses well.
        let samples = vec![1000i16; 256];
        assert_eq!(decode_vbz(&encode_vbz(&samples), 256).unwrap(), samples);
    }

    #[test]
    fn roundtrip_ramp() {
        let samples: Vec<i16> = (0..128).collect();
        assert_eq!(decode_vbz(&encode_vbz(&samples), 128).unwrap(), samples);
    }

    #[test]
    fn roundtrip_extremes() {
        let samples = vec![i16::MIN, i16::MAX, i16::MIN, i16::MAX];
        assert_eq!(decode_vbz(&encode_vbz(&samples), 4).unwrap(), samples);
    }

    #[test]
    fn encode_vbz_into_appends() {
        let mut out = encode_vbz(&[1i16, 2, 3]);
        let first_len = out.len();
        encode_vbz_into(&[4i16, 5, 6], &mut out);
        // Two independent blobs concatenated; decode each with its own n.
        let first = decode_vbz(&out[..first_len], 3).unwrap();
        let second = decode_vbz(&out[first_len..], 3).unwrap();
        assert_eq!(first, [1, 2, 3]);
        assert_eq!(second, [4, 5, 6]);
    }

    #[test]
    fn decode_vbz_into_appends() {
        let enc = encode_vbz(&[10i16, 20, 30]);
        let mut out = vec![99i16];
        decode_vbz_into(&enc, 3, &mut out).unwrap();
        assert_eq!(out, [99, 10, 20, 30]);
    }
}
