use core::error::Error;
use core::fmt;

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
#[derive(Debug)]
pub enum DecodeError {
    /// The data stream ended before all `n` values could be decoded.
    ///
    /// `index` is the zero-based index of the first value whose bytes were
    /// missing. This usually means `n` was larger than the number of values
    /// that were actually encoded.
    DataTruncated { index: usize },
    /// The control (tag) stream is shorter than required for `n` values.
    ///
    /// `need` is the number of control bytes required; `have` is how many
    /// were present in `data`.
    ControlStreamTooShort { need: usize, have: usize },
    /// The frame's version byte is not one this crate knows how to decode.
    ///
    /// Wire formats that embed a version byte (e.g. ex-zd) use this to
    /// signal forward-incompatible changes rather than silently
    /// misinterpreting the payload.
    UnsupportedVersion { version: u8 },
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DataTruncated { index } => {
                write!(f, "data truncated: expected more bytes at value {index}")
            }
            Self::ControlStreamTooShort { need, have } => write!(
                f,
                "control stream shorter than expected: need {need} bytes, have {have}"
            ),
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported format version: {version}")
            }
        }
    }
}

impl Error for DecodeError {}
