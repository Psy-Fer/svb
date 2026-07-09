//! Fuzz SVB-ZD decode: arbitrary bytes must never panic (only return Err or Ok).
//!
//! Input layout: [n_lo, n_hi, ...encoded_bytes...]
//! n = u16::from_le_bytes([n_lo, n_hi]) is the declared sample count.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let n = u16::from_le_bytes([data[0], data[1]]) as usize;
    let rest = &data[2..];
    let _ = svb::decode_svbzd(rest, n);
    let _ = svb::decode_svbzd_fused(rest, n);
});
