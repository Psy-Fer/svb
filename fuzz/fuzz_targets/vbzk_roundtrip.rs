//! Fuzz VBZ-K (experimental format) encode -> decode round-trip.
//!
//! Input layout: [k_lo, k_hi, ...samples as little-endian i16...]
//! k = max(1, u16::from_le_bytes([k_lo, k_hi])), so values above 255 test
//! the cap. `encode_vbzk_to_slice` must match `encode_vbzk`, stay within
//! `vbzk_max_encoded_len` and reject a shorter buffer, and the sequential and
//! parallel decoders must return the input.

#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let Some((k, rest)) = split_n(data) else {
        return;
    };
    let k = k.max(1);
    let samples = i16s(rest);
    let n = samples.len();

    let encoded = svb::encode_vbzk(&samples, k);
    assert!(
        encoded[0] >= 1 && usize::from(encoded[0]) <= k.min(255),
        "header k={}",
        encoded[0]
    );
    let max = svb::vbzk_max_encoded_len(n, k);
    assert_eq!(
        encode_checked(max, |o| svb::encode_vbzk_to_slice(&samples, k, o)),
        encoded
    );
    check_too_small(max, |o| svb::encode_vbzk_to_slice(&samples, k, o));

    assert_eq!(
        svb::decode_vbzk(&encoded, n).expect("decode failed"),
        samples
    );
    let mut out = vec![STALE_I16; n];
    svb::decode_vbzk_to_slice(&encoded, &mut out).expect("slice decode failed");
    assert_eq!(out, samples);
    out.fill(STALE_I16);
    svb::decode_vbzk_parallel_to_slice(&encoded, &mut out).expect("parallel slice decode failed");
    assert_eq!(out, samples);
});
