#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::DecodeError;

mod scalar;

// ── U32Classic ────────────────────────────────────────────────────────────────

fn dispatch_encode_classic(values: &[u32], out: &mut Vec<u8>) {
    scalar::encode_into_classic(values, out);
}

fn dispatch_decode_classic(data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
    scalar::decode_into_classic(data, n, out)
}

/// StreamVByte codec for u32 values.
/// 2-bit tags, 1/2/3/4 bytes per value. Wire-compatible with Lemire's C library.
pub struct U32Classic;

impl U32Classic {
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_classic(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_classic(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_classic(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_classic(data, n, out)
    }
}

impl crate::coder::Coder for U32Classic {
    type Elem = u32;

    fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_classic(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
        dispatch_decode_classic(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_classic(ctrl, n)
    }
}

// ── U32Variant0124 ────────────────────────────────────────────────────────────

fn dispatch_encode_0124(values: &[u32], out: &mut Vec<u8>) {
    scalar::encode_into_0124(values, out);
}

fn dispatch_decode_0124(data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
    scalar::decode_into_0124(data, n, out)
}

/// StreamVByte codec for u32 values.
/// 2-bit tags, 0/1/2/4 bytes per value. Zero values use 0 data bytes, making
/// this more compact than U32Classic for sparse (mostly-zero) data.
pub struct U32Variant0124;

impl U32Variant0124 {
    pub fn encode(&self, values: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_0124(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_0124(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_0124(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u32>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_0124(data, n, out)
    }
}

impl crate::coder::Coder for U32Variant0124 {
    type Elem = u32;

    fn encode_into(&self, values: &[u32], out: &mut Vec<u8>) {
        dispatch_encode_0124(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u32>) -> Result<(), DecodeError> {
        dispatch_decode_0124(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_0124(ctrl, n)
    }
}
