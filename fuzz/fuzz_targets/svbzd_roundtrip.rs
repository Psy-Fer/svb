//! Fuzz SVB-ZD encode -> decode round-trip: decode_svbzd(encode_svbzd(input))
//! == input, and decode_svbzd_fused must agree with it exactly.
//!
//! Arbitrary bytes are reinterpreted as little-endian i16 samples.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    let encoded = svb::encode_svbzd(&samples);

    let decoded = svb::decode_svbzd(&encoded, samples.len()).expect("3-pass decode failed");
    assert_eq!(decoded, samples);

    let fused =
        svb::decode_svbzd_fused(&encoded, samples.len()).expect("fused decode failed");
    assert_eq!(fused, samples);
});
