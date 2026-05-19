//! Fuzz Svb16.decode: arbitrary bytes must never panic (only return Err).
//!
//! Input layout: [n_lo, n_hi, ...encoded_bytes...]

#![no_main]
use libfuzzer_sys::fuzz_target;
use svb::u16::Svb16;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let n = u16::from_le_bytes([data[0], data[1]]) as usize;
    let _ = Svb16.decode(&data[2..], n);
});
