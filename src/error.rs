use thiserror::Error;

/// Errors that can occur when decoding a StreamVByte-encoded byte slice.
///
/// # Examples
///
/// ```
/// # use svb::{u32::U32Classic, DecodeError};
/// // Decoding from an empty buffer when n > 0 → ControlStreamTooShort.
/// match U32Classic.decode(&[], 4) {
///     Err(DecodeError::ControlStreamTooShort { need, have }) => {
///         assert_eq!(need, 1);
///         assert_eq!(have, 0);
///     }
///     _ => panic!("expected ControlStreamTooShort"),
/// }
/// ```
#[derive(Debug, Error)]
pub enum DecodeError {
    /// The data stream ended before all `n` values could be decoded.
    ///
    /// `index` is the zero-based index of the first value whose bytes were
    /// missing. This usually means `n` was larger than the number of values
    /// that were actually encoded.
    #[error("data truncated: expected more bytes at value {index}")]
    DataTruncated { index: usize },
    /// The control (tag) stream is shorter than required for `n` values.
    ///
    /// `need` is the number of control bytes required; `have` is how many
    /// were present in `data`.
    #[error("control stream shorter than expected: need {need} bytes, have {have}")]
    ControlStreamTooShort { need: usize, have: usize },
}
