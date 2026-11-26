//! Checks shared by the fuzz targets. Included with `#[path = "common.rs"]`;
//! not a fuzz target itself.

#![allow(dead_code)]

use core::fmt::Debug;

const CANARY: u8 = 0xA5;

/// Value the slice decoders' output buffers are filled with before decoding,
/// so a decoder that skips writing part of `out` is caught.
pub const STALE_I16: i16 = 0x5A5A;

/// Run an `encode_to_slice` call with a buffer of exactly `max` bytes,
/// followed by canary bytes the call cannot see. Checks the encoder wrote
/// nothing past `max` (a SIMD store through a raw pointer could) and returns
/// the encoded bytes.
pub fn encode_checked<E: Debug>(
    max: usize,
    f: impl FnOnce(&mut [u8]) -> Result<usize, E>,
) -> Vec<u8> {
    let mut buf = vec![CANARY; max + 64];
    let len = f(&mut buf[..max]).expect("encode_to_slice failed with a max_encoded_len buffer");
    assert!(len <= max, "returned length {len} > max {max}");
    assert!(
        buf[max..].iter().all(|&b| b == CANARY),
        "encode wrote past max_encoded_len"
    );
    buf.truncate(len);
    buf
}

/// A buffer one byte shorter than `max` must be rejected and left unchanged.
pub fn check_too_small<E: Debug>(max: usize, f: impl FnOnce(&mut [u8]) -> Result<usize, E>) {
    if max == 0 {
        return;
    }
    let mut buf = vec![CANARY; max - 1];
    assert!(f(&mut buf).is_err(), "accepted a buffer shorter than max");
    assert!(
        buf.iter().all(|&b| b == CANARY),
        "too-small buffer was modified"
    );
}

/// Check a `Vec` `_into` call against the slice result. `prefix` was in the
/// `Vec` before the call. On error the `Vec` must be back to `prefix`; on
/// success it must be `prefix` followed by `expected`.
pub fn check_into<T: PartialEq + Debug, E: Debug>(
    prefix: &[T],
    v: &[T],
    result: &Result<(), E>,
    expected: Option<&[T]>,
) {
    match (result, expected) {
        (Err(_), None) => assert_eq!(v, prefix, "_into did not restore the Vec on error"),
        (Ok(()), Some(exp)) => {
            assert_eq!(
                &v[..prefix.len()],
                prefix,
                "_into changed the existing values"
            );
            assert_eq!(&v[prefix.len()..], exp, "_into and _to_slice disagree");
        }
        _ => panic!(
            "_into returned {result:?}, slice decode returned {}",
            expected.is_some()
        ),
    }
}

/// Two decoders of the same format must both fail, or both succeed with the
/// same output.
pub fn check_agree<T: PartialEq + Debug, E: Debug, F: Debug>(
    name_a: &str,
    a: (&Result<(), E>, &[T]),
    name_b: &str,
    b: (&Result<(), F>, &[T]),
) {
    assert_eq!(
        a.0.is_ok(),
        b.0.is_ok(),
        "{name_a} returned {:?}, {name_b} returned {:?}",
        a.0,
        b.0
    );
    if a.0.is_ok() {
        assert_eq!(a.1, b.1, "{name_a} and {name_b} decoded different values");
    }
}

pub fn i16s(data: &[u8]) -> Vec<i16> {
    data.chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// Split the input into a declared count (first two bytes, little-endian)
/// and the encoded bytes.
pub fn split_n(data: &[u8]) -> Option<(usize, &[u8])> {
    if data.len() < 2 {
        return None;
    }
    Some((u16::from_le_bytes([data[0], data[1]]) as usize, &data[2..]))
}
