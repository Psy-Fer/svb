//! Fuzz `patched` encode -> decode round-trip.
//!
//! Arbitrary bytes are reinterpreted as little-endian u16 values, so every
//! exception density from none to all is reachable. `encode_to_slice` must
//! match `encode_into`, stay within `max_encoded_len` and reject a shorter
//! buffer, and both decoders must return the input and consume every byte.

#![no_main]
use libfuzzer_sys::fuzz_target;
use svb::patched;

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let values: Vec<u16> = data
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    let n = values.len();

    let mut encoded = Vec::new();
    patched::encode_into(&values, &mut encoded);
    let max = patched::max_encoded_len(n);
    assert_eq!(
        encode_checked(max, |o| patched::encode_to_slice(&values, o)),
        encoded
    );
    check_too_small(max, |o| patched::encode_to_slice(&values, o));

    let mut out = vec![0x5A5Au16; n];
    let consumed = patched::decode_to_slice(&encoded, &mut out).expect("slice decode failed");
    assert_eq!(consumed, encoded.len());
    assert_eq!(out, values);
    let mut v = vec![7u16];
    let consumed = patched::decode_into(&encoded, n, &mut v).expect("decode failed");
    assert_eq!(consumed, encoded.len());
    assert_eq!(v[0], 7);
    assert_eq!(v[1..], values);
});
