//! Property-based and deterministic round-trip tests for all codec types.

use proptest::prelude::*;
use svb::{
    decode_vbz, encode_vbz,
    u16::Svb16,
    u32::{U32Classic, U32Variant0124},
    u64::{U64Coder1234, U64Coder1248},
};

// ── proptest round-trips ──────────────────────────────────────────────────────

proptest! {
    #[test]
    fn svb16_roundtrip(values in proptest::collection::vec(any::<u16>(), 0usize..=4096)) {
        let n = values.len();
        let enc = Svb16.encode(&values);
        prop_assert_eq!(Svb16.decode(&enc, n).unwrap(), values);
    }

    #[test]
    fn u32_classic_roundtrip(values in proptest::collection::vec(any::<u32>(), 0usize..=4096)) {
        let n = values.len();
        let enc = U32Classic.encode(&values);
        prop_assert_eq!(U32Classic.decode(&enc, n).unwrap(), values);
    }

    #[test]
    fn u32_variant0124_roundtrip(values in proptest::collection::vec(any::<u32>(), 0usize..=4096)) {
        let n = values.len();
        let enc = U32Variant0124.encode(&values);
        prop_assert_eq!(U32Variant0124.decode(&enc, n).unwrap(), values);
    }

    #[test]
    fn u64_coder1234_roundtrip(
        // Constrain to u32::MAX so values encode without truncation.
        values in proptest::collection::vec(0u64..=u64::from(u32::MAX), 0usize..=4096)
    ) {
        let n = values.len();
        let enc = U64Coder1234.encode(&values);
        prop_assert_eq!(U64Coder1234.decode(&enc, n).unwrap(), values);
    }

    #[test]
    fn u64_coder1248_roundtrip(values in proptest::collection::vec(any::<u64>(), 0usize..=4096)) {
        let n = values.len();
        let enc = U64Coder1248.encode(&values);
        prop_assert_eq!(U64Coder1248.decode(&enc, n).unwrap(), values);
    }

    #[test]
    fn vbz_roundtrip(samples in proptest::collection::vec(any::<i16>(), 0usize..=4096)) {
        let n = samples.len();
        let enc = encode_vbz(&samples);
        prop_assert_eq!(decode_vbz(&enc, n).unwrap(), samples);
    }
}

// ── deterministic VBZ edge cases ─────────────────────────────────────────────

#[test]
fn vbz_empty() {
    assert_eq!(decode_vbz(&encode_vbz(&[]), 0).unwrap(), &[] as &[i16]);
}

#[test]
fn vbz_single_zero() {
    let v = vec![0i16];
    assert_eq!(decode_vbz(&encode_vbz(&v), 1).unwrap(), v);
}

#[test]
fn vbz_single_min() {
    let v = vec![i16::MIN];
    assert_eq!(decode_vbz(&encode_vbz(&v), 1).unwrap(), v);
}

#[test]
fn vbz_single_max() {
    let v = vec![i16::MAX];
    assert_eq!(decode_vbz(&encode_vbz(&v), 1).unwrap(), v);
}

#[test]
fn vbz_all_zeros() {
    let v = vec![0i16; 512];
    assert_eq!(decode_vbz(&encode_vbz(&v), v.len()).unwrap(), v);
}

#[test]
fn vbz_all_min() {
    let v = vec![i16::MIN; 512];
    assert_eq!(decode_vbz(&encode_vbz(&v), v.len()).unwrap(), v);
}

#[test]
fn vbz_all_max() {
    let v = vec![i16::MAX; 512];
    assert_eq!(decode_vbz(&encode_vbz(&v), v.len()).unwrap(), v);
}

#[test]
fn vbz_monotone_increasing() {
    let v: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    assert_eq!(decode_vbz(&encode_vbz(&v), v.len()).unwrap(), v);
}

#[test]
fn vbz_monotone_decreasing() {
    let v: Vec<i16> = (i16::MIN..=i16::MAX).rev().collect();
    assert_eq!(decode_vbz(&encode_vbz(&v), v.len()).unwrap(), v);
}

#[test]
fn vbz_alternating_extremes() {
    let v: Vec<i16> = (0..512)
        .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
        .collect();
    assert_eq!(decode_vbz(&encode_vbz(&v), v.len()).unwrap(), v);
}
