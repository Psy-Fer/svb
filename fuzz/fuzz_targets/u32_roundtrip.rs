//! Fuzz U32Classic and U32Variant0124 encode -> decode round-trips.
//!
//! Input is reinterpreted as little-endian u32 values. Both codecs must
//! round-trip every input exactly, through both the `Vec` and slice
//! functions, and `encode_to_slice` must stay within `max_encoded_len` and
//! reject a shorter buffer.

#![no_main]
use libfuzzer_sys::fuzz_target;
use svb::u32::{U32Classic, U32Variant0124};

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let values: Vec<u32> = data
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let enc = U32Classic.encode(&values);
    let max = U32Classic.max_encoded_len(values.len());
    assert_eq!(
        encode_checked(max, |o| U32Classic.encode_to_slice(&values, o)),
        enc
    );
    check_too_small(max, |o| U32Classic.encode_to_slice(&values, o));
    let dec = U32Classic
        .decode(&enc, values.len())
        .expect("U32Classic round-trip failed");
    assert_eq!(dec, values);
    let mut out = vec![0x5A5A_5A5Au32; values.len()];
    U32Classic
        .decode_to_slice(&enc, &mut out)
        .expect("U32Classic slice decode failed");
    assert_eq!(out, values);

    let enc = U32Variant0124.encode(&values);
    let max = U32Variant0124.max_encoded_len(values.len());
    assert_eq!(
        encode_checked(max, |o| U32Variant0124.encode_to_slice(&values, o)),
        enc
    );
    check_too_small(max, |o| U32Variant0124.encode_to_slice(&values, o));
    let dec = U32Variant0124
        .decode(&enc, values.len())
        .expect("U32Variant0124 round-trip failed");
    assert_eq!(dec, values);
    let mut out = vec![0x5A5A_5A5Au32; values.len()];
    U32Variant0124
        .decode_to_slice(&enc, &mut out)
        .expect("U32Variant0124 slice decode failed");
    assert_eq!(out, values);
});
