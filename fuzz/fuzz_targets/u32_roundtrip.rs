//! Fuzz U32Classic and U32Variant0124 encode→decode round-trips.
//!
//! Input is reinterpreted as little-endian u32 values. Both codecs must
//! round-trip every input exactly.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let values: Vec<u32> = data
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let enc = svb::u32::U32Classic.encode(&values);
    let dec = svb::u32::U32Classic.decode(&enc, values.len()).expect("classic round-trip failed");
    assert_eq!(dec, values);

    let enc = svb::u32::U32Variant0124.encode(&values);
    let dec = svb::u32::U32Variant0124.decode(&enc, values.len()).expect("0124 round-trip failed");
    assert_eq!(dec, values);
});
