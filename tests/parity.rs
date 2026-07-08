//! Parity tests: fixed byte vectors derived from the algorithm spec (svb_ref_doc.md).
//! These test the *public API* end-to-end, independent of which back-end is selected.
//! If any of these fail, the wire format has drifted from the spec.

use svb::{
    decode_exzd, decode_vbz, encode_exzd, encode_vbz,
    u16::Svb16,
    u32::{U32Classic, U32Variant0124},
    u64::{U64Coder1234, U64Coder1248},
    zigzag,
};

// ── §2.2 Zigzag ──────────────────────────────────────────────────────────────

#[test]
fn zigzag_known_values_encode() {
    let cases: &[(i16, u16)] = &[
        (0, 0),
        (-1, 1),
        (1, 2),
        (-2, 3),
        (i16::MAX, 65534),
        (i16::MIN, 65535),
    ];
    for &(input, expected) in cases {
        assert_eq!(
            zigzag::encode(&[input]),
            [expected],
            "zigzag encode {input}"
        );
    }
}

#[test]
fn zigzag_known_values_decode() {
    let cases: &[(u16, i16)] = &[
        (0, 0),
        (1, -1),
        (2, 1),
        (3, -2),
        (65534, i16::MAX),
        (65535, i16::MIN),
    ];
    for &(input, expected) in cases {
        assert_eq!(
            zigzag::decode::<i16>(&[input]),
            [expected],
            "zigzag decode {input}"
        );
    }
}

// ── §2.3 SVB16 ───────────────────────────────────────────────────────────────

/// Spec example: [5, 300, 0, 1000]
/// ctrl[0] = 0x0A (bit1 and bit3 set for the two-byte values)
/// data    = [0x05, 0x2C, 0x01, 0x00, 0xE8, 0x03]
#[test]
fn svb16_spec_example_encode() {
    assert_eq!(
        Svb16.encode(&[5u16, 300, 0, 1000]),
        [0x0A, 0x05, 0x2C, 0x01, 0x00, 0xE8, 0x03]
    );
}

#[test]
fn svb16_spec_example_decode() {
    let data = [0x0Au8, 0x05, 0x2C, 0x01, 0x00, 0xE8, 0x03];
    assert_eq!(Svb16.decode(&data, 4).unwrap(), [5u16, 300, 0, 1000]);
}

// ── §2.4 U32Classic ──────────────────────────────────────────────────────────

/// Spec example: [1, 256, 65536, 0xFFFFFFFF]
/// ctrl[0] = 0xE4 (tags: 0, 1, 2, 3 packed LSB-first)
#[test]
fn u32_classic_spec_example_encode() {
    assert_eq!(
        U32Classic.encode(&[1u32, 256, 65536, 0xFFFF_FFFF]),
        [
            0xE4, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF
        ]
    );
}

#[test]
fn u32_classic_spec_example_decode() {
    let data = [
        0xE4u8, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF,
    ];
    assert_eq!(
        U32Classic.decode(&data, 4).unwrap(),
        [1u32, 256, 65536, 0xFFFF_FFFF]
    );
}

// ── §2.4 U32Variant0124 ──────────────────────────────────────────────────────

/// Spec example: [0, 1, 255, 256, 65535, 65536, 0xFFFFFFFF]
/// ctrl = [0x94, 0x3E]
#[test]
fn u32_variant0124_spec_example_encode() {
    assert_eq!(
        U32Variant0124.encode(&[0u32, 1, 255, 256, 65535, 65536, 0xFFFF_FFFF]),
        [
            0x94, 0x3E, 0x01, 0xFF, 0x00, 0x01, 0xFF, 0xFF, 0x00, 0x00, 0x01, 0x00, 0xFF, 0xFF,
            0xFF, 0xFF
        ]
    );
}

#[test]
fn u32_variant0124_spec_example_decode() {
    let data = [
        0x94u8, 0x3E, 0x01, 0xFF, 0x00, 0x01, 0xFF, 0xFF, 0x00, 0x00, 0x01, 0x00, 0xFF, 0xFF, 0xFF,
        0xFF,
    ];
    assert_eq!(
        U32Variant0124.decode(&data, 7).unwrap(),
        [0u32, 1, 255, 256, 65535, 65536, 0xFFFF_FFFF]
    );
}

// ── §2.4 U64Coder1234 ────────────────────────────────────────────────────────

/// Spec example: [0, 0xFFFFFF, 0xFFFFFFFF]
/// ctrl[0] = 0x38
#[test]
fn u64_coder1234_spec_example_encode() {
    assert_eq!(
        U64Coder1234.encode(&[0u64, 0xFF_FFFF, 0xFFFF_FFFF]),
        [0x38, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]
    );
}

#[test]
fn u64_coder1234_spec_example_decode() {
    let data = [0x38u8, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
    assert_eq!(
        U64Coder1234.decode(&data, 3).unwrap(),
        [0u64, 0xFF_FFFF, 0xFFFF_FFFF]
    );
}

// ── §2.4 U64Coder1248 ────────────────────────────────────────────────────────

/// Spec example: [0, 0x1_0000_0000]
/// ctrl[0] = 0x0C
#[test]
fn u64_coder1248_spec_example_encode() {
    assert_eq!(
        U64Coder1248.encode(&[0u64, 0x1_0000_0000u64]),
        [0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
    );
}

#[test]
fn u64_coder1248_spec_example_decode() {
    let data = [0x0Cu8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
    assert_eq!(
        U64Coder1248.decode(&data, 2).unwrap(),
        [0u64, 0x1_0000_0000u64]
    );
}

// ── VBZ pipeline known vector ─────────────────────────────────────────────────

/// Hand-derived vector for [0, 1, -1, 2, -2]:
///   delta  → [0, 1, -2, 3, -4]
///   zigzag → [0, 2, 3, 6, 7]   (all ≤ 255 → all 1-byte in SVB16)
///   SVB16  → ctrl=[0x00], data=[0x00, 0x02, 0x03, 0x06, 0x07]
#[test]
fn vbz_known_vector_encode() {
    assert_eq!(
        encode_vbz(&[0i16, 1, -1, 2, -2]),
        [0x00, 0x00, 0x02, 0x03, 0x06, 0x07]
    );
}

#[test]
fn vbz_known_vector_decode() {
    let data = [0x00u8, 0x00, 0x02, 0x03, 0x06, 0x07];
    assert_eq!(decode_vbz(&data, 5).unwrap(), [0i16, 1, -1, 2, -2]);
}

/// VBZ vector with a 2-byte SVB16 value:
///   input  → [1000]
///   delta  → [1000]
///   zigzag → [2000]   (2000 > 255 → 2-byte in SVB16)
///   SVB16  → ctrl=[0x01], data=[0xD0, 0x07]
#[test]
fn vbz_known_vector_two_byte() {
    assert_eq!(encode_vbz(&[1000i16]), [0x01, 0xD0, 0x07]);
    assert_eq!(decode_vbz(&[0x01u8, 0xD0, 0x07], 1).unwrap(), [1000i16]);
}

// ── POD5 parity vectors ───────────────────────────────────────────────────────
//
// Raw SVB16 bytes extracted from examples/small.pod5 (post-zstd-decompress).
// Expected i16 signal from pod5.Reader.read.signal (the pod5 Python library).
// Our decode_vbz must produce bit-identical output to the pod5 decoder.
//
// Extraction script: see the commit that added tests/vectors/.

fn load_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn load_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn load_u64(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
        .collect()
}

#[test]
fn pod5_parity_read0() {
    let svb16 = include_bytes!("vectors/parity_00_02885.svb16");
    let expected = load_i16(include_bytes!("vectors/parity_00_02885.i16"));
    assert_eq!(decode_vbz(svb16, 2885).unwrap(), expected);
}

#[test]
fn pod5_parity_read1() {
    let svb16 = include_bytes!("vectors/parity_01_02915.svb16");
    let expected = load_i16(include_bytes!("vectors/parity_01_02915.i16"));
    assert_eq!(decode_vbz(svb16, 2915).unwrap(), expected);
}

#[test]
fn pod5_parity_read2() {
    let svb16 = include_bytes!("vectors/parity_02_02949.svb16");
    let expected = load_i16(include_bytes!("vectors/parity_02_02949.i16"));
    assert_eq!(decode_vbz(svb16, 2949).unwrap(), expected);
}

// ── U32 / U64 regression vectors ─────────────────────────────────────────────
//
// Self-generated from the Rust implementation (validated against spec examples
// and property-based roundtrip tests; see manifest.json for details).
// These guard against future regressions — any change to encode/decode wire
// format will cause these tests to fail.
//
// To regenerate: cargo run --bin gen_test_vectors

#[test]
fn u32_classic_regression_vector() {
    let enc = include_bytes!("vectors/u32_classic_256.enc");
    let expected = load_u32(include_bytes!("vectors/u32_classic_256.raw"));
    assert_eq!(U32Classic.decode(enc, 256).unwrap(), expected);
    // Verify encode produces the same bytes.
    assert_eq!(U32Classic.encode(&expected), enc);
}

#[test]
fn u32_variant0124_regression_vector() {
    let enc = include_bytes!("vectors/u32_variant0124_256.enc");
    let expected = load_u32(include_bytes!("vectors/u32_variant0124_256.raw"));
    assert_eq!(U32Variant0124.decode(enc, 256).unwrap(), expected);
    assert_eq!(U32Variant0124.encode(&expected), enc);
}

#[test]
fn u64_coder1234_regression_vector() {
    let enc = include_bytes!("vectors/u64_coder1234_256.enc");
    let expected = load_u64(include_bytes!("vectors/u64_coder1234_256.raw"));
    assert_eq!(U64Coder1234.decode(enc, 256).unwrap(), expected);
    assert_eq!(U64Coder1234.encode(&expected), enc);
}

#[test]
fn u64_coder1248_regression_vector() {
    let enc = include_bytes!("vectors/u64_coder1248_256.enc");
    let expected = load_u64(include_bytes!("vectors/u64_coder1248_256.raw"));
    assert_eq!(U64Coder1248.decode(enc, 256).unwrap(), expected);
    assert_eq!(U64Coder1248.encode(&expected), enc);
}

// ── ex-zd parity vectors ──────────────────────────────────────────────────────
//
// Byte-exact fixtures generated by calling slow5lib's
// slow5_ptr_compress_solo(SLOW5_COMPRESS_EX_ZD, ...) on the same sample
// arrays used by its own unit tests (test/unit_test_press.c:
// press_ex_one_valid / press_ex_big_valid / press_ex_exp_valid /
// press_ex_huge_valid). If any of these fail, encode_exzd/decode_exzd have
// drifted from slow5lib's wire format.

macro_rules! exzd_parity_test {
    ($name:ident, $stem:literal) => {
        #[test]
        fn $name() {
            let enc = include_bytes!(concat!("vectors/", $stem, ".enc"));
            let expected = load_i16(include_bytes!(concat!("vectors/", $stem, ".raw")));
            assert_eq!(decode_exzd(enc).unwrap(), expected);
            assert_eq!(encode_exzd(&expected), enc);
        }
    };
}

exzd_parity_test!(exzd_c_reference_one, "exzd_one_1");
exzd_parity_test!(exzd_c_reference_big, "exzd_big_10");
exzd_parity_test!(exzd_c_reference_exp, "exzd_exp_10");
exzd_parity_test!(exzd_c_reference_huge, "exzd_huge_35");
