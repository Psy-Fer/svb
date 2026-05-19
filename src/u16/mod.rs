#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::DecodeError;

mod scalar;

// Dispatch functions — SIMD back-ends will replace these calls when enabled.
fn dispatch_encode(values: &[u16], out: &mut Vec<u8>) {
    scalar::encode_into(values, out);
}

fn dispatch_decode(data: &[u8], n: usize, out: &mut Vec<u16>) -> Result<(), DecodeError> {
    scalar::decode_into(data, n, out)
}

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
