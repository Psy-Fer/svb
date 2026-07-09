//! Fuzz ex-zd encode -> decode round-trip: decode_exzd(encode_exzd(input)) ==
//! input, and decode_exzd_fused / ExzdDecoder must agree with it exactly.
//!
//! Arbitrary bytes are reinterpreted as little-endian i16 samples.

#![no_main]
use libfuzzer_sys::fuzz_target;
use svb::ExzdDecoder;

fuzz_target!(|data: &[u8]| {
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    let encoded = svb::encode_exzd(&samples);

    let decoded = svb::decode_exzd(&encoded).expect("3-pass decode failed");
    assert_eq!(decoded, samples);

    let fused = svb::decode_exzd_fused(&encoded).expect("fused decode failed");
    assert_eq!(fused, samples);

    let mut decoder = ExzdDecoder::new();
    let mut out = Vec::new();
    decoder
        .decode_into(&encoded, &mut out)
        .expect("ExzdDecoder decode failed");
    assert_eq!(out, samples);
});
