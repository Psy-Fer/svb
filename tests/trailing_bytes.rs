//! Decoders must ignore bytes after the end of a frame, on every back-end.
//!
//! Before 0.4.0 the AVX2 decoders of the five codecs returned
//! `DataTruncated` when a frame was followed by more bytes than their tail
//! buffer holds (64-96 bytes), while the scalar, SSSE3 and NEON decoders
//! accepted them. The 3-pass VBZ and SVB-ZD decoders were affected through
//! `Svb16` and `U32Classic`. These tests decode each frame with 0,
//! 1, 31 and 200 extra bytes, for every count up to 100 (all tail shapes of
//! every kernel), with values at the narrowest, widest and mixed widths.
//! CI runs them under each SIMD feature.
//!
//! The codec `decode_to_slice` functions must also return the length of the
//! frame, whatever follows it.

use svb::u16::Svb16;
use svb::u32::{U32Classic, U32Variant0124};
use svb::u64::{U64Coder1234, U64Coder1248};

const EXTRA: [usize; 4] = [0, 1, 31, 200];

fn with_extra(enc: &[u8], extra: usize) -> Vec<u8> {
    let mut data = enc.to_vec();
    data.extend((0..extra).map(|i| (i as u8).wrapping_mul(37) | 0x80));
    data
}

// Narrowest, widest and mixed widths for a codec whose widest value is `max`.
fn value_sets<T: Copy>(n: usize, zero: T, max: T, mixed: impl Fn(usize) -> T) -> [Vec<T>; 3] {
    [vec![zero; n], vec![max; n], (0..n).map(mixed).collect()]
}

macro_rules! check_codec {
    ($codec:expr, $t:ty, $max:expr) => {
        for n in 0..=100usize {
            let sets = value_sets::<$t>(n, 0, $max, |i| match i % 5 {
                0 => 0,
                1 => $max,
                2 => 300,
                3 => (70000u64 & $max as u64) as $t,
                _ => ((i as u64).wrapping_mul(1_000_003) & $max as u64) as $t,
            });
            for values in sets {
                let enc = $codec.encode(&values);
                for extra in EXTRA {
                    let data = with_extra(&enc, extra);
                    let ctx = format!("{} n={n} extra={extra}", stringify!($codec));
                    assert_eq!($codec.decode(&data, n).expect(&ctx), values, "{ctx}");
                    let mut out = vec![1 as $t; n];
                    let used = $codec.decode_to_slice(&data, &mut out).expect(&ctx);
                    assert_eq!(out, values, "{ctx}");
                    assert_eq!(used, enc.len(), "{ctx}: bytes used");
                }
            }
        }
    };
}

#[test]
fn svb16() {
    check_codec!(Svb16, u16, u16::MAX);
}

#[test]
fn u32_classic() {
    check_codec!(U32Classic, u32, u32::MAX);
}

#[test]
fn u32_variant0124() {
    check_codec!(U32Variant0124, u32, u32::MAX);
}

#[test]
fn u64_coder1234() {
    check_codec!(U64Coder1234, u64, u32::MAX as u64);
}

#[test]
fn u64_coder1248() {
    check_codec!(U64Coder1248, u64, u64::MAX);
}

fn sample_sets(n: usize) -> [Vec<i16>; 3] {
    [
        vec![7; n],
        (0..n)
            .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
            .collect(),
        (0..n)
            .map(|i| match i % 4 {
                0 => i16::MIN,
                1 => 3,
                _ => (i as i16).wrapping_mul(613),
            })
            .collect(),
    ]
}

#[test]
fn vbz() {
    for n in 0..=100usize {
        for samples in sample_sets(n) {
            let enc = svb::encode_vbz(&samples);
            for extra in EXTRA {
                let data = with_extra(&enc, extra);
                let ctx = format!("n={n} extra={extra}");
                assert_eq!(svb::decode_vbz(&data, n).expect(&ctx), samples, "{ctx}");
                assert_eq!(
                    svb::decode_vbz_fused(&data, n).expect(&ctx),
                    samples,
                    "{ctx}"
                );
                let mut out = vec![1i16; n];
                svb::decode_vbz_fused_to_slice(&data, &mut out).expect(&ctx);
                assert_eq!(out, samples, "{ctx}");
            }
        }
    }
}

#[test]
fn svbzd() {
    for n in 0..=100usize {
        for samples in sample_sets(n) {
            let enc = svb::encode_svbzd(&samples);
            for extra in EXTRA {
                let data = with_extra(&enc, extra);
                let ctx = format!("n={n} extra={extra}");
                assert_eq!(svb::decode_svbzd(&data, n).expect(&ctx), samples, "{ctx}");
                assert_eq!(
                    svb::decode_svbzd_fused(&data, n).expect(&ctx),
                    samples,
                    "{ctx}"
                );
                let mut out = vec![1i16; n];
                svb::decode_svbzd_fused_to_slice(&data, &mut out).expect(&ctx);
                assert_eq!(out, samples, "{ctx}");
            }
        }
    }
}
