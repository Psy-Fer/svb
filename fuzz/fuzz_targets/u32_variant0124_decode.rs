//! Fuzz U32Variant0124 decode: arbitrary bytes must never panic (only return Err or Ok).
//!
//! Input layout: [n_lo, n_hi, ...encoded_bytes...]
//! n = u16::from_le_bytes([n_lo, n_hi]) is the declared value count.
//!
//! `decode`, `decode_to_slice` and `decode_into` must agree, and
//! `decode_into` must restore the `Vec` on error. The byte count from
//! `decode_to_slice` must be the end of the values.

#![no_main]
use libfuzzer_sys::fuzz_target;
use svb::u32::U32Variant0124;

#[path = "common.rs"]
mod common;
use common::*;

fuzz_target!(|data: &[u8]| {
    let Some((n, rest)) = split_n(data) else {
        return;
    };

    let mut out = vec![0x5A5A_5A5Au32; n];
    let r = U32Variant0124.decode_to_slice(rest, &mut out);
    if let Ok(used) = r {
        // The count is where the values end: decoding only those bytes must
        // give the same result, and one byte fewer must fail (the last byte
        // of a frame is always needed).
        assert!(used <= rest.len(), "used {used} > len {}", rest.len());
        let mut again = out.clone();
        again.fill(0);
        let r2 = U32Variant0124.decode_to_slice(&rest[..used], &mut again);
        assert_eq!(r2.ok(), Some(used), "decode of the first {used} bytes");
        assert_eq!(again, out);
        if used > 0 {
            let r3 = U32Variant0124.decode_to_slice(&rest[..used - 1], &mut again);
            assert!(r3.is_err(), "decoded with one byte fewer than {used}");
        }
    }
    let expected = r.is_ok().then_some(&out[..]);
    match U32Variant0124.decode(rest, n) {
        Ok(v) => assert_eq!(Some(&v[..]), expected),
        Err(_) => assert!(expected.is_none()),
    }
    let prefix = [1u32, 2, 3];
    let mut v = prefix.to_vec();
    let r = U32Variant0124.decode_into(rest, n, &mut v);
    check_into(&prefix, &v, &r, expected);
});
