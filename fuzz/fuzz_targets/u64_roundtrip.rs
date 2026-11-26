//! Fuzz U64Coder1248 and U64Coder1234 encode -> decode round-trips.
//!
//! U64Coder1248 covers the full u64 range so no range check is needed.
//! U64Coder1234 silently truncates values above u32::MAX, so we mask inputs
//! to u32::MAX to get a clean round-trip. Both go through the `Vec` and slice
//! functions, and `encode_to_slice` must stay within `max_encoded_len` and
//! reject a shorter buffer.

#![no_main]
use libfuzzer_sys::fuzz_target;
use svb::u64::{U64Coder1234, U64Coder1248};

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let values: Vec<u64> = data
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();

    let enc = U64Coder1248.encode(&values);
    let max = U64Coder1248.max_encoded_len(values.len());
    assert_eq!(
        encode_checked(max, |o| U64Coder1248.encode_to_slice(&values, o)),
        enc
    );
    check_too_small(max, |o| U64Coder1248.encode_to_slice(&values, o));
    let dec = U64Coder1248
        .decode(&enc, values.len())
        .expect("U64Coder1248 round-trip failed");
    assert_eq!(dec, values);
    let mut out = vec![0x5A5A_5A5A_5A5A_5A5Au64; values.len()];
    U64Coder1248
        .decode_to_slice(&enc, &mut out)
        .expect("U64Coder1248 slice decode failed");
    assert_eq!(out, values);

    let values32: Vec<u64> = values.iter().map(|&v| v & 0xFFFF_FFFF).collect();
    let enc = U64Coder1234.encode(&values32);
    let max = U64Coder1234.max_encoded_len(values32.len());
    assert_eq!(
        encode_checked(max, |o| U64Coder1234.encode_to_slice(&values32, o)),
        enc
    );
    check_too_small(max, |o| U64Coder1234.encode_to_slice(&values32, o));
    let dec = U64Coder1234
        .decode(&enc, values32.len())
        .expect("U64Coder1234 round-trip failed");
    assert_eq!(dec, values32);
    let mut out = vec![0x5A5A_5A5A_5A5A_5A5Au64; values32.len()];
    U64Coder1234
        .decode_to_slice(&enc, &mut out)
        .expect("U64Coder1234 slice decode failed");
    assert_eq!(out, values32);
});
