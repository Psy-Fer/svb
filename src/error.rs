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
    /// The output slice is not the length the frame says it holds.
    ///
    /// Returned by the `_to_slice` decoders of self-describing formats
    /// (ex-zd), whose frame stores its own sample count. `need` is that count;
    /// `have` is the length of the output slice.
    OutputLengthMismatch { need: usize, have: usize },
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
            Self::OutputLengthMismatch { need, have } => write!(
                f,
                "output length does not match the frame: need {need} values, have {have}"
            ),
        }
    }
}

impl Error for DecodeError {}

/// Errors that can occur when encoding into a caller-supplied byte slice.
///
/// # Examples
///
/// ```
/// # use svb::{u16::Svb16, EncodeError};
/// let mut out = [0u8; 2];
/// match Svb16.encode_to_slice(&[1u16, 2, 3], &mut out) {
///     Err(EncodeError::OutputTooSmall { need, have }) => {
///         assert_eq!(need, Svb16.max_encoded_len(3));
///         assert_eq!(have, 2);
///     }
///     _ => panic!("expected OutputTooSmall"),
/// }
/// ```
#[derive(Debug)]
pub enum EncodeError {
    /// The output slice is shorter than the worst-case encoded size.
    ///
    /// `need` is the worst-case size for the input (see the codec's
    /// `max_encoded_len`); `have` is the length of the output slice. The check
    /// is made before encoding starts, so a slice that would have fitted the
    /// actual encoded bytes is still rejected if it is below `need`.
    OutputTooSmall { need: usize, have: usize },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutputTooSmall { need, have } => {
                write!(f, "output buffer too small: need {need} bytes, have {have}")
            }
        }
    }
}

impl Error for EncodeError {}
