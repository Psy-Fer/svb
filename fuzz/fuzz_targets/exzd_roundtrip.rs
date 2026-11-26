//! Fuzz ex-zd encode -> decode round-trip.
//!
//! Arbitrary bytes are reinterpreted as little-endian i16 samples. Every
//! encoder must produce the same bytes, `encode_exzd_to_slice` must stay
//! within `exzd_max_encoded_len` and reject a shorter buffer, and every
//! decoder must return the input. The `ExzdDecoder` is kept between inputs.

#![no_main]
use libfuzzer_sys::fuzz_target;
use std::cell::RefCell;
use svb::ExzdDecoder;

#[path = "common.rs"]
mod common;
use common::*;

thread_local! {
    static DECODER: RefCell<ExzdDecoder> = RefCell::new(ExzdDecoder::new());
}

fuzz_target!(|data: &[u8]| {
    let samples = i16s(data);
    let n = samples.len();

    let encoded = svb::encode_exzd(&samples);
    let max = svb::exzd_max_encoded_len(n);
    assert_eq!(
        encode_checked(max, |o| svb::encode_exzd_to_slice(&samples, o)),
        encoded
    );
    check_too_small(max, |o| svb::encode_exzd_to_slice(&samples, o));
    let mut appended = vec![9u8];
    svb::encode_exzd_into(&samples, &mut appended);
    assert_eq!(appended[0], 9);
    assert_eq!(appended[1..], encoded);

    assert_eq!(
        svb::exzd_decoded_len(&encoded).expect("decoded_len failed"),
        n
    );
    assert_eq!(
        svb::decode_exzd(&encoded).expect("3-pass decode failed"),
        samples
    );
    assert_eq!(
        svb::decode_exzd_fused(&encoded).expect("fused decode failed"),
        samples
    );
    let mut out = vec![STALE_I16; n];
    svb::decode_exzd_to_slice(&encoded, &mut out).expect("3-pass slice decode failed");
    assert_eq!(out, samples);
    out.fill(STALE_I16);
    svb::decode_exzd_fused_to_slice(&encoded, &mut out).expect("fused slice decode failed");
    assert_eq!(out, samples);

    DECODER.with_borrow_mut(|decoder| {
        out.fill(STALE_I16);
        decoder
            .decode_to_slice(&encoded, &mut out)
            .expect("ExzdDecoder slice decode failed");
        assert_eq!(out, samples);
        let mut v = Vec::new();
        decoder
            .decode_into(&encoded, &mut v)
            .expect("ExzdDecoder decode failed");
        assert_eq!(v, samples);
    });
});
