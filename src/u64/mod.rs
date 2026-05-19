#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::DecodeError;

mod scalar;

// ── U64Coder1234 ──────────────────────────────────────────────────────────────

fn dispatch_encode_1234(values: &[u64], out: &mut Vec<u8>) {
    scalar::encode_into_1234(values, out);
}

fn dispatch_decode_1234(data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
    scalar::decode_into_1234(data, n, out)
}

/// StreamVByte codec for u64 values (1/2/3/4 bytes per value).
///
/// Same tag/width table as U32Classic but operates on `u64` slices. Values greater
/// than `u32::MAX` are silently truncated to their low 32 bits — this matches the
/// behaviour of other StreamVByte libraries and is defined, not accidental. Use
/// [`U64Coder1234::check_range`] before encoding if you need to detect out-of-range
/// values. For data that may genuinely exceed `u32::MAX`, use [`U64Coder1248`].
pub struct U64Coder1234;

impl U64Coder1234 {
    /// Returns the index of the first value that exceeds `u32::MAX`, or `None` if
    /// all values can be encoded without truncation.
    pub fn check_range(&self, values: &[u64]) -> Option<usize> {
        values.iter().position(|&v| v > u64::from(u32::MAX))
    }

    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_1234(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1234(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1234(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_1234(data, n, out)
    }
}

impl crate::coder::Coder for U64Coder1234 {
    type Elem = u64;

    fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1234(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        dispatch_decode_1234(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_1234(ctrl, n)
    }
}

// ── U64Coder1248 ──────────────────────────────────────────────────────────────

fn dispatch_encode_1248(values: &[u64], out: &mut Vec<u8>) {
    scalar::encode_into_1248(values, out);
}

fn dispatch_decode_1248(data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
    scalar::decode_into_1248(data, n, out)
}

/// StreamVByte codec for u64 values (1/2/4/8 bytes per value).
/// Covers the full u64 range. Values in 0x10000–0xFFFFFF use 4 bytes
/// (no 3-byte option); values in 0x100000000–u64::MAX use 8 bytes.
pub struct U64Coder1248;

impl U64Coder1248 {
    pub fn encode(&self, values: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        dispatch_encode_1248(values, &mut out);
        out
    }

    pub fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1248(values, out);
    }

    pub fn decode(&self, data: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        dispatch_decode_1248(data, n, &mut out)?;
        Ok(out)
    }

    pub fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<u64>,
    ) -> Result<(), DecodeError> {
        dispatch_decode_1248(data, n, out)
    }
}

impl crate::coder::Coder for U64Coder1248 {
    type Elem = u64;

    fn encode_into(&self, values: &[u64], out: &mut Vec<u8>) {
        dispatch_encode_1248(values, out);
    }

    fn decode_into(&self, data: &[u8], n: usize, out: &mut Vec<u64>) -> Result<(), DecodeError> {
        dispatch_decode_1248(data, n, out)
    }

    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize {
        scalar::encoded_data_len_1248(ctrl, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    #[test]
    fn check_range_all_safe() {
        assert_eq!(U64Coder1234.check_range(&[0, 0xFF, u32::MAX as u64]), None);
    }

    #[test]
    fn check_range_detects_first_bad() {
        let values = [1u64, 2, 0x1_0000_0000, 3, 0x2_0000_0000];
        assert_eq!(U64Coder1234.check_range(&values), Some(2));
    }

    #[test]
    fn check_range_empty() {
        assert_eq!(U64Coder1234.check_range(&[]), None);
    }

    #[test]
    fn check_range_first_element_bad() {
        assert_eq!(U64Coder1234.check_range(&[u64::MAX, 1, 2]), Some(0));
    }
}
