//! Fuzz SVB-ZD encode -> decode round-trip.
//!
//! Arbitrary bytes are reinterpreted as little-endian i16 samples. Every
//! encoder must produce the same bytes, `encode_svbzd_to_slice` must stay
//! within `svbzd_max_encoded_len` and reject a shorter buffer, and every
//! decoder must return the input.

#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let samples = i16s(data);
    let n = samples.len();

    let encoded = svb::encode_svbzd(&samples);
    let max = svb::svbzd_max_encoded_len(n);
    assert_eq!(
        encode_checked(max, |o| svb::encode_svbzd_to_slice(&samples, o)),
        encoded
    );
    check_too_small(max, |o| svb::encode_svbzd_to_slice(&samples, o));
    let mut appended = vec![9u8];
    svb::encode_svbzd_into(&samples, &mut appended);
    assert_eq!(appended[0], 9);
    assert_eq!(appended[1..], encoded);

    assert_eq!(
        svb::decode_svbzd(&encoded, n).expect("3-pass decode failed"),
        samples
    );
    assert_eq!(
        svb::decode_svbzd_fused(&encoded, n).expect("fused decode failed"),
        samples
    );
    let mut out = vec![STALE_I16; n];
    svb::decode_svbzd_to_slice(&encoded, &mut out).expect("3-pass slice decode failed");
    assert_eq!(out, samples);
    out.fill(STALE_I16);
    svb::decode_svbzd_fused_to_slice(&encoded, &mut out).expect("fused slice decode failed");
    assert_eq!(out, samples);
});
