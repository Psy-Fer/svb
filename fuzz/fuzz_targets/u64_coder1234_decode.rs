//! Fuzz U64Coder1234 decode: arbitrary bytes must never panic (only return Err or Ok).
//!
//! Input layout: [n_lo, n_hi, ...encoded_bytes...]
//! n = u16::from_le_bytes([n_lo, n_hi]) is the declared value count.
//!
//! `decode`, `decode_to_slice` and `decode_into` must agree, and
//! `decode_into` must restore the `Vec` on error.

#![no_main]
use libfuzzer_sys::fuzz_target;
use svb::u64::U64Coder1234;

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let Some((n, rest)) = split_n(data) else {
        return;
    };

    let mut out = vec![0x5A5A_5A5A_5A5A_5A5Au64; n];
    let r = U64Coder1234.decode_to_slice(rest, &mut out);
    let expected = r.is_ok().then_some(&out[..]);
    match U64Coder1234.decode(rest, n) {
        Ok(v) => assert_eq!(Some(&v[..]), expected),
        Err(_) => assert!(expected.is_none()),
    }
    let prefix = [1u64, 2, 3];
    let mut v = prefix.to_vec();
    let r = U64Coder1234.decode_into(rest, n, &mut v);
    check_into(&prefix, &v, &r, expected);
});
