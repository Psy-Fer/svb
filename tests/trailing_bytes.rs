//! Bytes after the end of a frame, on every back-end.
//!
//! The codec decoders accept them, and `decode_to_slice` returns the length
//! of the frame whatever follows it. Before 0.4.0 the AVX2 decoders of the
//! five codecs returned `DataTruncated` when a frame was followed by more
//! bytes than their tail buffer holds (64-96 bytes), while the scalar, SSSE3
//! and NEON decoders accepted them.
//!
//! The pipeline decoders (VBZ, VBZ-K, SVB-ZD, ex-zd) expect exactly one frame
//! and must return `TrailingData` for any extra bytes. They must also catch a
//! control stream that claims fewer bytes than were written, which used to
//! decode as `Ok` with wrong samples.
//!
//! Each frame is decoded with 0, 1, 31 and 200 extra bytes, for every count up
//! to 100 (all tail shapes of every kernel), with values at the narrowest,
//! widest and mixed widths. CI runs these tests under each SIMD feature.

use svb::DecodeError;
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

// Every decoder of one pipeline, as `(name, decode)`. Each returns the
// decoded samples, so the `Vec`, `_into` and `_to_slice` forms can be checked
// the same way.
type Decode = Box<dyn Fn(&[u8], usize) -> Result<Vec<i16>, DecodeError>>;

// `_into`: decode after a prefix, which must be left alone on error.
fn into(f: impl Fn(&[u8], usize, &mut Vec<i16>) -> Result<(), DecodeError> + 'static) -> Decode {
    Box::new(move |data, n| {
        let mut v = vec![-7i16, 8];
        match f(data, n, &mut v) {
            Ok(()) => {
                assert_eq!(v[..2], [-7, 8], "_into changed the prefix");
                Ok(v[2..].to_vec())
            }
            Err(e) => {
                assert_eq!(v, [-7, 8], "_into did not restore the Vec on error");
                Err(e)
            }
        }
    })
}

fn to_slice(f: impl Fn(&[u8], &mut [i16]) -> Result<(), DecodeError> + 'static) -> Decode {
    Box::new(move |data, n| {
        let mut v = vec![0x5A5Ai16; n];
        f(data, &mut v).map(|()| v)
    })
}

fn vbz_decoders() -> Vec<(&'static str, Decode)> {
    vec![
        ("decode_vbz", Box::new(svb::decode_vbz)),
        ("decode_vbz_into", into(svb::decode_vbz_into)),
        ("decode_vbz_to_slice", to_slice(svb::decode_vbz_to_slice)),
        ("decode_vbz_fused", Box::new(svb::decode_vbz_fused)),
        ("decode_vbz_fused_into", into(svb::decode_vbz_fused_into)),
        (
            "decode_vbz_fused_to_slice",
            to_slice(svb::decode_vbz_fused_to_slice),
        ),
        (
            "decode_vbz_fused_from",
            Box::new(|d, n| svb::decode_vbz_fused_from(d, n, 0)),
        ),
        (
            "decode_vbz_fused_from_into",
            into(|d, n, v| svb::decode_vbz_fused_from_into(d, n, 0, v)),
        ),
        (
            "decode_vbz_fused_from_to_slice",
            to_slice(|d, o| svb::decode_vbz_fused_from_to_slice(d, 0, o)),
        ),
    ]
}

fn vbzk_decoders() -> Vec<(&'static str, Decode)> {
    #[cfg_attr(not(feature = "std"), allow(unused_mut))] // only std adds the parallel decoders
    let mut d: Vec<(&'static str, Decode)> = vec![
        ("decode_vbzk", Box::new(svb::decode_vbzk)),
        ("decode_vbzk_into", into(svb::decode_vbzk_into)),
        ("decode_vbzk_to_slice", to_slice(svb::decode_vbzk_to_slice)),
    ];
    #[cfg(feature = "std")]
    d.extend::<[(&'static str, Decode); 3]>([
        ("decode_vbzk_parallel", Box::new(svb::decode_vbzk_parallel)),
        (
            "decode_vbzk_parallel_into",
            into(svb::decode_vbzk_parallel_into),
        ),
        (
            "decode_vbzk_parallel_to_slice",
            to_slice(svb::decode_vbzk_parallel_to_slice),
        ),
    ]);
    d
}

fn svbzd_decoders() -> Vec<(&'static str, Decode)> {
    vec![
        ("decode_svbzd", Box::new(svb::decode_svbzd)),
        ("decode_svbzd_into", into(svb::decode_svbzd_into)),
        (
            "decode_svbzd_to_slice",
            to_slice(svb::decode_svbzd_to_slice),
        ),
        ("decode_svbzd_fused", Box::new(svb::decode_svbzd_fused)),
        (
            "decode_svbzd_fused_into",
            into(svb::decode_svbzd_fused_into),
        ),
        (
            "decode_svbzd_fused_to_slice",
            to_slice(svb::decode_svbzd_fused_to_slice),
        ),
        (
            "decode_svbzd_fused_from",
            Box::new(|d, n| svb::decode_svbzd_fused_from(d, n, 0)),
        ),
        (
            "decode_svbzd_fused_from_into",
            into(|d, n, v| svb::decode_svbzd_fused_from_into(d, n, 0, v)),
        ),
        (
            "decode_svbzd_fused_from_to_slice",
            to_slice(|d, o| svb::decode_svbzd_fused_from_to_slice(d, 0, o)),
        ),
    ]
}

// ex-zd frames store their own count; `n` is only used to size `_to_slice`.
fn exzd_decoders() -> Vec<(&'static str, Decode)> {
    vec![
        ("decode_exzd", Box::new(|d, _| svb::decode_exzd(d))),
        (
            "decode_exzd_into",
            into(|d, _, v| svb::decode_exzd_into(d, v)),
        ),
        ("decode_exzd_to_slice", to_slice(svb::decode_exzd_to_slice)),
        (
            "decode_exzd_fused",
            Box::new(|d, _| svb::decode_exzd_fused(d)),
        ),
        (
            "decode_exzd_fused_into",
            into(|d, _, v| svb::decode_exzd_fused_into(d, v)),
        ),
        (
            "decode_exzd_fused_to_slice",
            to_slice(svb::decode_exzd_fused_to_slice),
        ),
        (
            "ExzdDecoder::decode_into",
            into(|d, _, v| svb::ExzdDecoder::new().decode_into(d, v)),
        ),
        (
            "ExzdDecoder::decode_to_slice",
            to_slice(|d, o| svb::ExzdDecoder::new().decode_to_slice(d, o)),
        ),
    ]
}

// Each frame must decode on its own, and every decoder must reject it with
// any bytes after it, reporting where the frame ended and the input length.
fn check_pipeline(
    decoders: &[(&'static str, Decode)],
    encode: impl Fn(&[i16]) -> Vec<u8>,
    max_n: usize,
) {
    for n in 0..=max_n {
        for samples in sample_sets(n) {
            let enc = encode(&samples);
            for extra in EXTRA {
                let data = with_extra(&enc, extra);
                for (name, decode) in decoders {
                    let ctx = format!("{name} n={n} extra={extra}");
                    let r = decode(&data, n);
                    if extra == 0 {
                        assert_eq!(r.expect(&ctx), samples, "{ctx}");
                    } else {
                        match r {
                            Err(DecodeError::TrailingData { used, len }) => {
                                assert_eq!((used, len), (enc.len(), data.len()), "{ctx}")
                            }
                            other => panic!("{ctx}: expected TrailingData, got {other:?}"),
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn vbz() {
    check_pipeline(&vbz_decoders(), svb::encode_vbz, 100);
}

#[test]
fn vbzk() {
    for k in [1, 2, 4] {
        check_pipeline(&vbzk_decoders(), |s| svb::encode_vbzk(s, k), 100);
    }
}

#[test]
fn svbzd() {
    check_pipeline(&svbzd_decoders(), svb::encode_svbzd, 100);
}

#[test]
fn exzd() {
    check_pipeline(&exzd_decoders(), svb::encode_exzd, 100);
}

// A control stream that claims fewer bytes than were written (the pod5lib
// report): lower one value's width tag in a valid frame. The data stream then
// has one byte left over, which used to decode as `Ok` with wrong samples.
// Every decoder must now return `TrailingData`.
fn check_under_claim(
    decoders: &[(&'static str, Decode)],
    encoded: &[u8],
    n: usize,
    tag_bits: usize,
    ctrl_start: usize,
) {
    let tags_per_byte = 8 / tag_bits;
    let mask = (1u8 << tag_bits) - 1;
    let mut tested = 0;
    for i in 0..n {
        let (byte, shift) = (
            ctrl_start + i / tags_per_byte,
            (i % tags_per_byte) * tag_bits,
        );
        let tag = (encoded[byte] >> shift) & mask;
        if tag == 0 {
            continue;
        }
        let mut data = encoded.to_vec();
        data[byte] = (data[byte] & !(mask << shift)) | ((tag - 1) << shift);
        for (name, decode) in decoders {
            match decode(&data, n) {
                Err(DecodeError::TrailingData { used, len }) => {
                    assert_eq!(
                        (used, len),
                        (data.len() - 1, data.len()),
                        "{name} value {i}"
                    )
                }
                other => panic!("{name} value {i}: expected TrailingData, got {other:?}"),
            }
        }
        tested += 1;
    }
    assert!(tested > 0, "no tag to lower");
}

#[test]
fn under_claiming_ctrl() {
    for n in [1usize, 7, 8, 9, 33, 100] {
        // Alternating extremes: every VBZ value is 2 bytes and every SVB-ZD
        // value 3 bytes, so any tag can be lowered.
        let samples: Vec<i16> = (0..n)
            .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
            .collect();
        check_under_claim(&vbz_decoders(), &svb::encode_vbz(&samples), n, 1, 0);
        // k = 1: a 1-byte header, then a single VBZ stream.
        check_under_claim(&vbzk_decoders(), &svb::encode_vbzk(&samples, 1), n, 1, 1);
        check_under_claim(&svbzd_decoders(), &svb::encode_svbzd(&samples), n, 2, 0);
    }
}

// ex-zd: lower a tag in the control stream of the exception values array
// inside the patched part of the frame. The array's stored length then has
// one byte too many.
#[test]
fn under_claiming_exzd_exception_array() {
    // Large jumps: every delta after the first is an exception, and each
    // exception value needs 2 bytes.
    let samples: Vec<i16> = (0..40)
        .map(|i| if i % 2 == 0 { 0 } else { 20000 })
        .collect();
    let enc = svb::encode_exzd(&samples);
    let n = samples.len();
    // Frame: 10-byte header, 2-byte first value, then the patched part:
    // [nex u32][pos length u32][pos array][val length u32][val array][literals].
    let p = 12;
    let u32_at = |i: usize| u32::from_le_bytes(enc[i..i + 4].try_into().unwrap()) as usize;
    let nex = u32_at(p);
    assert!(nex >= 2, "need the length-prefixed layout");
    let val_len_at = p + 8 + u32_at(p + 4);
    let val_start = val_len_at + 4;
    let val_end = val_start + u32_at(val_len_at);
    let tag = enc[val_start] & 3;
    assert!(tag > 0, "first exception value should be wider than 1 byte");
    let mut data = enc.clone();
    data[val_start] -= 1;
    for (name, decode) in exzd_decoders() {
        match decode(&data, n) {
            Err(DecodeError::TrailingData { used, len }) => {
                assert_eq!((used, len), (val_end - 1, val_end), "{name}")
            }
            other => panic!("{name}: expected TrailingData, got {other:?}"),
        }
    }
    let mut out = vec![0u16; n - 1];
    match svb::patched::decode_to_slice(&data[p..], &mut out) {
        Err(DecodeError::TrailingData { used, len }) => {
            assert_eq!((used, len), (val_end - 1 - p, val_end - p), "patched")
        }
        other => panic!("patched: expected TrailingData, got {other:?}"),
    }
}

// VBZ-K: lower a tag in the first of two sub-streams. Its data then ends one
// byte before the second sub-stream starts, which the header records.
#[test]
fn under_claiming_vbzk_first_sub_stream() {
    let n: usize = 64;
    let samples: Vec<i16> = (0..n)
        .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
        .collect();
    let mut data = svb::encode_vbzk(&samples, 2);
    // Header: k, then [carry i16][data offset u32] for sub-stream 1.
    assert_eq!(data[0], 2);
    let d_off = u32::from_le_bytes(data[3..7].try_into().unwrap()) as usize;
    let sub1_start = 7 + n.div_ceil(8) + d_off;
    data[7] &= !1; // first value of sub-stream 0: 2 bytes -> 1 byte
    for (name, decode) in vbzk_decoders() {
        match decode(&data, n) {
            Err(DecodeError::TrailingData { used, len }) => {
                assert_eq!((used, len), (sub1_start - 1, sub1_start), "{name}")
            }
            other => panic!("{name}: expected TrailingData, got {other:?}"),
        }
    }
}
