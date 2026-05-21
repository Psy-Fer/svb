//! Fuzz U64Coder1248 encode→decode round-trip.
//!
//! U64Coder1248 covers the full u64 range so no range check is needed.
//! U64Coder1234 silently truncates values above u32::MAX, so we mask inputs
//! to u32::MAX to get a clean round-trip.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let values: Vec<u64> = data
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect();

    // 1248: full range round-trip
    let enc = svb::u64::U64Coder1248.encode(&values);
    let dec = svb::u64::U64Coder1248.decode(&enc, values.len()).expect("1248 round-trip failed");
    assert_eq!(dec, values);

    // 1234: mask to u32::MAX to avoid silent truncation
    let values32: Vec<u64> = values.iter().map(|&v| v & 0xFFFF_FFFF).collect();
    let enc = svb::u64::U64Coder1234.encode(&values32);
    let dec = svb::u64::U64Coder1234.decode(&enc, values32.len()).expect("1234 round-trip failed");
    assert_eq!(dec, values32);
});
