//! Fuzz `patched` decode: arbitrary bytes must never panic.
//!
//! Input layout: [n_lo, n_hi, ...encoded_bytes...]
//! n = u16::from_le_bytes([n_lo, n_hi]) is the declared value count.
//!
//! `decode_to_slice` and `decode_into` must agree on the values and the
//! number of bytes consumed, and `decode_into` must restore the `Vec` on
//! error.

#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let Some((n, rest)) = split_n(data) else {
        return;
    };

    let mut out = vec![0x5A5Au16; n];
    let r_slice = svb::patched::decode_to_slice(rest, &mut out);
    if let Ok(consumed) = r_slice {
        assert!(consumed <= rest.len());
    }

    let prefix = [1u16, 2, 3];
    let mut v = prefix.to_vec();
    let r_vec = svb::patched::decode_into(rest, n, &mut v);
    match (&r_slice, &r_vec) {
        (Ok(a), Ok(b)) => assert_eq!(a, b, "consumed byte counts differ"),
        (Err(_), Err(_)) => {}
        _ => panic!("decode_to_slice returned {r_slice:?}, decode_into returned {r_vec:?}"),
    }
    let expected = r_slice.is_ok().then_some(&out[..]);
    check_into(&prefix, &v, &r_vec.map(|_| ()), expected);
});
