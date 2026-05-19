use thiserror::Error;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("data truncated: expected more bytes at value {index}")]
    DataTruncated { index: usize },
    #[error("control stream shorter than expected: need {need} bytes, have {have}")]
    ControlStreamTooShort { need: usize, have: usize },
}
