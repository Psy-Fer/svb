//! Fuzz encode→decode round-trip: encode_vbz(decode_vbz(encode_vbz(input))) == input.
//!
//! Arbitrary bytes are reinterpreted as little-endian i16 samples. The
//! encode→decode round-trip must be identity for any valid input.

#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();
    let encoded = svb::encode_vbz(&samples);
    let decoded = svb::decode_vbz(&encoded, samples.len()).expect("round-trip decode failed");
    assert_eq!(decoded, samples);
});
