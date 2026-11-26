//! Fuzz ex-zd decode: arbitrary bytes must never panic (only return Err or Ok).
//!
//! Unlike the other codec decode fuzz targets, ex-zd's frame is
//! self-describing (version + sample count + q embedded in the header), so
//! the raw fuzz input is fed directly to the decoder with no separate `n`.
//!
//! The 3-pass decoder, the fused decoder and a reused `ExzdDecoder` must
//! agree. The decoder is kept between inputs, so its grow-only buffers carry
//! state from earlier frames. The slice decoders must reject an output of
//! the wrong length, and the `_into` functions must restore the `Vec` on
//! error.

#![no_main]
use libfuzzer_sys::fuzz_target;
use std::cell::RefCell;
use svb::{DecodeError, ExzdDecoder};

#[path = "common.rs"]
mod common;
use common::*;

thread_local! {
    static DECODER: RefCell<ExzdDecoder> = RefCell::new(ExzdDecoder::new());
}

fuzz_target!(|data: &[u8]| {
    DECODER.with_borrow_mut(|decoder| check(data, decoder));
});

fn check(data: &[u8], decoder: &mut ExzdDecoder) {
    let prefix = [1i16, 2, 3];
    let Ok(n) = svb::exzd_decoded_len(data) else {
        // A frame whose header is rejected must fail everywhere.
        assert!(svb::decode_exzd(data).is_err());
        assert!(svb::decode_exzd_fused(data).is_err());
        let mut v = prefix.to_vec();
        assert!(decoder.decode_into(data, &mut v).is_err());
        assert_eq!(v, prefix);
        return;
    };

    let mut three = vec![STALE_I16; n];
    let r_three = svb::decode_exzd_to_slice(data, &mut three);
    let mut fused = vec![STALE_I16; n];
    let r_fused = svb::decode_exzd_fused_to_slice(data, &mut fused);
    let mut reused = vec![STALE_I16; n];
    let r_reused = decoder.decode_to_slice(data, &mut reused);
    check_agree(
        "decode_exzd_to_slice",
        (&r_three, &three),
        "decode_exzd_fused_to_slice",
        (&r_fused, &fused),
    );
    check_agree(
        "decode_exzd_fused_to_slice",
        (&r_fused, &fused),
        "ExzdDecoder::decode_to_slice",
        (&r_reused, &reused),
    );
    let expected = r_fused.is_ok().then_some(&fused[..]);

    match svb::decode_exzd(data) {
        Ok(v) => assert_eq!(Some(&v[..]), expected),
        Err(_) => assert!(expected.is_none()),
    }
    match svb::decode_exzd_fused(data) {
        Ok(v) => assert_eq!(Some(&v[..]), expected),
        Err(_) => assert!(expected.is_none()),
    }
    let mut v = prefix.to_vec();
    let r = svb::decode_exzd_into(data, &mut v);
    check_into(&prefix, &v, &r, expected);
    let mut v = prefix.to_vec();
    let r = svb::decode_exzd_fused_into(data, &mut v);
    check_into(&prefix, &v, &r, expected);
    let mut v = prefix.to_vec();
    let r = decoder.decode_into(data, &mut v);
    check_into(&prefix, &v, &r, expected);

    // An output of the wrong length is rejected before anything else.
    for have in std::iter::once(n + 1).chain(n.checked_sub(1)) {
        let mut out = vec![0i16; have];
        for r in [
            svb::decode_exzd_to_slice(data, &mut out),
            svb::decode_exzd_fused_to_slice(data, &mut out),
            decoder.decode_to_slice(data, &mut out),
        ] {
            assert!(
                matches!(r, Err(DecodeError::OutputLengthMismatch { need, have: h }) if need == n && h == have),
                "n={n} have={have}: {r:?}"
            );
        }
    }
}
