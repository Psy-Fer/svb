//! Fuzz VBZ-K decode (experimental format): arbitrary bytes must never panic.
//!
//! Input layout: [n_lo, n_hi, ...encoded_bytes...]
//! n = u16::from_le_bytes([n_lo, n_hi]) is the declared sample count.
//!
//! The sequential and parallel decoders must agree, the `Vec` functions must
//! match the slice functions, and the `_into` functions must restore the
//! `Vec` on error.

#![no_main]
use libfuzzer_sys::fuzz_target;

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let Some((n, rest)) = split_n(data) else {
        return;
    };

    let mut seq = vec![STALE_I16; n];
    let r_seq = svb::decode_vbzk_to_slice(rest, &mut seq);
    let mut par = vec![STALE_I16; n];
    let r_par = svb::decode_vbzk_parallel_to_slice(rest, &mut par);
    check_agree(
        "decode_vbzk_to_slice",
        (&r_seq, &seq),
        "decode_vbzk_parallel_to_slice",
        (&r_par, &par),
    );
    let expected = r_seq.is_ok().then_some(&seq[..]);

    match svb::decode_vbzk(rest, n) {
        Ok(v) => assert_eq!(Some(&v[..]), expected),
        Err(_) => assert!(expected.is_none()),
    }
    let prefix = [1i16, 2, 3];
    let mut v = prefix.to_vec();
    let r = svb::decode_vbzk_into(rest, n, &mut v);
    check_into(&prefix, &v, &r, expected);
    let mut v = prefix.to_vec();
    let r = svb::decode_vbzk_parallel_into(rest, n, &mut v);
    check_into(&prefix, &v, &r, expected);
});
