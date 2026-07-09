//! Fuzz ex-zd decode: arbitrary bytes must never panic (only return Err or Ok).
//!
//! Unlike the other codec decode fuzz targets, ex-zd's frame is
//! self-describing (version + sample count + q embedded in the header), so
//! the raw fuzz input is fed directly to the decoder with no separate `n`.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = svb::decode_exzd(data);
    let _ = svb::decode_exzd_fused(data);
});
