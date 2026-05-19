#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::vec::Vec;

use crate::error::DecodeError;

pub(crate) trait Coder {
    type Elem: Copy;

    fn encode_into(&self, values: &[Self::Elem], out: &mut Vec<u8>);

    fn decode_into(
        &self,
        data: &[u8],
        n: usize,
        out: &mut Vec<Self::Elem>,
    ) -> Result<(), DecodeError>;

    /// Returns the number of data bytes consumed by `n` values given their control bytes.
    fn encoded_data_len(&self, ctrl: &[u8], n: usize) -> usize;

    fn encode(&self, values: &[Self::Elem]) -> Vec<u8> {
        let mut out = Vec::new();
        self.encode_into(values, &mut out);
        out
    }

    fn decode(&self, data: &[u8], n: usize) -> Result<Vec<Self::Elem>, DecodeError> {
        let mut out = Vec::with_capacity(n);
        self.decode_into(data, n, &mut out)?;
        Ok(out)
    }
}
