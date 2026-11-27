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
///
/// The enum is `#[non_exhaustive]`, so a `match` on it needs a `_` arm. New
/// variants can then be added without a breaking release.
#[derive(Debug)]
#[non_exhaustive]
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
    /// The input holds more bytes than the encoded values use.
    ///
    /// Returned by the pipeline decoders (VBZ, VBZ-K, SVB-ZD and ex-zd), which
    /// expect `data` to be exactly one frame. `used` is where the values ended
    /// according to the control stream, and `len` is where they should have
    /// ended. For a whole frame `len` is `data.len()`. For a part of a frame
    /// whose length is stored in the frame (a VBZ-K sub-stream, or a `patched`
    /// exception array) it is the end of that part, and both are positions in
    /// `data`. A corrupted control stream that claims fewer bytes than were
    /// written gives this error. The codec `decode_to_slice` functions do not
    /// return it; they return the number of bytes used instead.
    TrailingData { used: usize, len: usize },
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
            Self::TrailingData { used, len } => write!(
                f,
                "bytes left over after the encoded values: they end at byte {used}, expected {len}"
            ),
        }
    }
}

impl Error for DecodeError {}

// Fails if `data` is shorter than the `ctrl_len`-byte control stream. The
// `Vec` decode wrappers call this before allocating `n` values, so a large
// `n` with a short input fails without the allocation. The decoders make the
// same check and return the same error.
#[cfg(feature = "alloc")]
pub(crate) fn check_ctrl_len(data: &[u8], ctrl_len: usize) -> Result<(), DecodeError> {
    if data.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: data.len(),
        });
    }
    Ok(())
}

// Fails if a pipeline decoder used fewer than all `len` bytes of its input.
// `used` is never above `len`: a decoder that needs more returns an error.
pub(crate) fn check_used(used: usize, len: usize) -> Result<(), DecodeError> {
    if used != len {
        return Err(DecodeError::TrailingData { used, len });
    }
    Ok(())
}

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
