//! Fuzz VBZ decode: arbitrary bytes must never panic (only return Err or Ok).
//!
//! Input layout: [n_lo, n_hi, ...encoded_bytes...]
//! n = u16::from_le_bytes([n_lo, n_hi]) is the declared value count.
//!
//! The 3-pass and fused decoders must agree, the `Vec` functions must match
//! the slice functions, and the `_into` functions must restore the `Vec` on
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

    let mut three = vec![STALE_I16; n];
    let r_three = svb::decode_vbz_to_slice(rest, &mut three);
    let mut fused = vec![STALE_I16; n];
    let r_fused = svb::decode_vbz_fused_to_slice(rest, &mut fused);
    check_agree(
        "decode_vbz_to_slice",
        (&r_three, &three),
        "decode_vbz_fused_to_slice",
        (&r_fused, &fused),
    );
    let expected = r_fused.is_ok().then_some(&fused[..]);

    match svb::decode_vbz(rest, n) {
        Ok(v) => assert_eq!(Some(&v[..]), expected),
        Err(_) => assert!(expected.is_none()),
    }
    match svb::decode_vbz_fused(rest, n) {
        Ok(v) => assert_eq!(Some(&v[..]), expected),
        Err(_) => assert!(expected.is_none()),
    }

    let prefix = [1i16, 2, 3];
    let mut v = prefix.to_vec();
    let r = svb::decode_vbz_into(rest, n, &mut v);
    check_into(&prefix, &v, &r, expected);
    let mut v = prefix.to_vec();
    let r = svb::decode_vbz_fused_into(rest, n, &mut v);
    check_into(&prefix, &v, &r, expected);
});
