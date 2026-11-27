//! The slice API that is available without the `alloc` feature.
//!
//! Every call here is to a function that works on caller-supplied slices, so
//! this file also builds and runs with `--no-default-features`. The test
//! itself uses `Vec` and `std` (integration tests always link std); only the
//! crate under test is built without `alloc`.
//!
//! The codecs are checked against the stored test vectors. The fused VBZ and
//! SVB-ZD decoders and the SVB-ZD encoder are checked against the same pipeline
//! built from the `delta`, `zigzag` and codec slice functions.

use svb::u16::Svb16;
use svb::u32::{U32Classic, U32Variant0124};
use svb::u64::{U64Coder1234, U64Coder1248};
use svb::{DecodeError, delta, zigzag};

fn le_i16(b: &[u8]) -> Vec<i16> {
    b.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}
fn le_u32(b: &[u8]) -> Vec<u32> {
    b.chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
fn le_u64(b: &[u8]) -> Vec<u64> {
    b.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// Encodes `values` with `encode`, checks the bytes equal `enc`, then decodes
// `enc` with `decode` and checks the values and the byte count.
fn check_codec<T: Copy + Default + PartialEq + core::fmt::Debug>(
    values: &[T],
    enc: &[u8],
    max_len: usize,
    encode: impl Fn(&[T], &mut [u8]) -> usize,
    decode: impl Fn(&[u8], &mut [T]) -> Result<usize, DecodeError>,
) {
    let mut buf = vec![0u8; max_len];
    let written = encode(values, &mut buf);
    assert_eq!(&buf[..written], enc);

    let mut out = vec![T::default(); values.len()];
    assert_eq!(decode(enc, &mut out).unwrap(), enc.len());
    assert_eq!(out, values);
}

#[test]
fn u32_codecs_match_vectors() {
    let raw = le_u32(include_bytes!("vectors/u32_classic_256.raw"));
    check_codec(
        &raw,
        include_bytes!("vectors/u32_classic_256.enc"),
        U32Classic.max_encoded_len(raw.len()),
        |v, o| U32Classic.encode_to_slice(v, o).unwrap(),
        |d, o| U32Classic.decode_to_slice(d, o),
    );
    let raw = le_u32(include_bytes!("vectors/u32_variant0124_256.raw"));
    check_codec(
        &raw,
        include_bytes!("vectors/u32_variant0124_256.enc"),
        U32Variant0124.max_encoded_len(raw.len()),
        |v, o| U32Variant0124.encode_to_slice(v, o).unwrap(),
        |d, o| U32Variant0124.decode_to_slice(d, o),
    );
}

#[test]
fn u64_codecs_match_vectors() {
    let raw = le_u64(include_bytes!("vectors/u64_coder1234_256.raw"));
    check_codec(
        &raw,
        include_bytes!("vectors/u64_coder1234_256.enc"),
        U64Coder1234.max_encoded_len(raw.len()),
        |v, o| U64Coder1234.encode_to_slice(v, o).unwrap(),
        |d, o| U64Coder1234.decode_to_slice(d, o),
    );
    let raw = le_u64(include_bytes!("vectors/u64_coder1248_256.raw"));
    check_codec(
        &raw,
        include_bytes!("vectors/u64_coder1248_256.enc"),
        U64Coder1248.max_encoded_len(raw.len()),
        |v, o| U64Coder1248.encode_to_slice(v, o).unwrap(),
        |d, o| U64Coder1248.decode_to_slice(d, o),
    );
}

// Values on each side of every byte-width boundary. The encoded length is
// the control bytes plus the width of each value, so a value given the wrong
// width changes it even when the round trip still works.
#[test]
fn width_boundaries() {
    fn check<T: Copy + Default + PartialEq + core::fmt::Debug>(
        values: &[T],
        want_len: usize,
        max_len: usize,
        encode: impl Fn(&[T], &mut [u8]) -> usize,
        decode: impl Fn(&[u8], &mut [T]) -> Result<usize, DecodeError>,
    ) {
        let mut buf = vec![0u8; max_len];
        let len = encode(values, &mut buf);
        assert_eq!(len, want_len);
        let mut out = vec![T::default(); values.len()];
        assert_eq!(decode(&buf[..len], &mut out).unwrap(), len);
        assert_eq!(out, values);
    }

    // 1 control byte, widths 1 + 2 + 2.
    let v = [0xFFu16, 0x100, 0xFFFF];
    check(
        &v,
        1 + 5,
        Svb16.max_encoded_len(v.len()),
        |v, o| Svb16.encode_to_slice(v, o).unwrap(),
        |d, o| Svb16.decode_to_slice(d, o),
    );
    // 2 control bytes, widths 1 + 2 + 2 + 3 + 3 + 4 + 4.
    let v = [
        0xFFu32,
        0x100,
        0xFFFF,
        0x1_0000,
        0xFF_FFFF,
        0x100_0000,
        u32::MAX,
    ];
    check(
        &v,
        2 + 19,
        U32Classic.max_encoded_len(v.len()),
        |v, o| U32Classic.encode_to_slice(v, o).unwrap(),
        |d, o| U32Classic.decode_to_slice(d, o),
    );
    // 2 control bytes, widths 0 + 1 + 1 + 2 + 2 + 4 + 4.
    let v = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x1_0000, u32::MAX];
    check(
        &v,
        2 + 14,
        U32Variant0124.max_encoded_len(v.len()),
        |v, o| U32Variant0124.encode_to_slice(v, o).unwrap(),
        |d, o| U32Variant0124.decode_to_slice(d, o),
    );
    // 2 control bytes, widths 1 + 2 + 2 + 3 + 3 + 4 + 4.
    let v = [
        0xFFu64,
        0x100,
        0xFFFF,
        0x1_0000,
        0xFF_FFFF,
        0x100_0000,
        0xFFFF_FFFF,
    ];
    check(
        &v,
        2 + 19,
        U64Coder1234.max_encoded_len(v.len()),
        |v, o| U64Coder1234.encode_to_slice(v, o).unwrap(),
        |d, o| U64Coder1234.decode_to_slice(d, o),
    );
    // 2 control bytes, widths 1 + 2 + 2 + 4 + 4 + 8 + 8.
    let v = [
        0xFFu64,
        0x100,
        0xFFFF,
        0x1_0000,
        0xFFFF_FFFF,
        0x1_0000_0000,
        u64::MAX,
    ];
    check(
        &v,
        2 + 29,
        U64Coder1248.max_encoded_len(v.len()),
        |v, o| U64Coder1248.encode_to_slice(v, o).unwrap(),
        |d, o| U64Coder1248.decode_to_slice(d, o),
    );
}

// (VBZ bytes, samples) for the POD5 reads decoded by ONT's reference decoder.
fn pod5_reads() -> [(&'static [u8], Vec<i16>); 3] {
    [
        (
            include_bytes!("vectors/parity_00_02885.svb16"),
            le_i16(include_bytes!("vectors/parity_00_02885.i16")),
        ),
        (
            include_bytes!("vectors/parity_01_02915.svb16"),
            le_i16(include_bytes!("vectors/parity_01_02915.i16")),
        ),
        (
            include_bytes!("vectors/parity_02_02949.svb16"),
            le_i16(include_bytes!("vectors/parity_02_02949.i16")),
        ),
    ]
}

// Delta then zigzag with the slice functions: the values Svb16 encodes.
fn vbz_codes(samples: &[i16]) -> Vec<u16> {
    let mut deltas = vec![0i16; samples.len()];
    delta::encode_to_slice(samples, &mut deltas);
    let mut codes = vec![0u16; samples.len()];
    zigzag::encode_to_slice(&deltas, &mut codes);
    codes
}

#[test]
fn vbz_pipeline_from_slice_functions_matches_vectors() {
    for (enc, samples) in pod5_reads() {
        let n = samples.len();
        let codes = vbz_codes(&samples);
        check_codec(
            &codes,
            enc,
            Svb16.max_encoded_len(n),
            |v, o| Svb16.encode_to_slice(v, o).unwrap(),
            |d, o| Svb16.decode_to_slice(d, o),
        );

        // The inverse steps, also with the slice functions.
        let mut deltas = vec![0i16; n];
        zigzag::decode_to_slice(&codes, &mut deltas);
        let mut out = vec![0i16; n];
        delta::decode_to_slice(&deltas, &mut out);
        assert_eq!(out, samples);
    }
}

#[test]
fn vbz_fused_matches_vectors() {
    for (enc, samples) in pod5_reads() {
        let mut out = vec![0i16; samples.len()];
        svb::decode_vbz_fused_to_slice(enc, &mut out).unwrap();
        assert_eq!(out, samples);

        // One extra byte after the frame is rejected.
        let mut longer = enc.to_vec();
        longer.push(0);
        assert!(matches!(
            svb::decode_vbz_fused_to_slice(&longer, &mut out),
            Err(DecodeError::TrailingData { .. })
        ));
    }
}

#[test]
fn vbz_fused_from_decodes_the_second_half() {
    for (_, samples) in pod5_reads() {
        let mid = samples.len() / 2;
        let codes = vbz_codes(&samples);
        let mut enc = vec![0u8; Svb16.max_encoded_len(samples.len() - mid)];
        let len = Svb16.encode_to_slice(&codes[mid..], &mut enc).unwrap();

        let mut out = vec![0i16; samples.len() - mid];
        svb::decode_vbz_fused_from_to_slice(&enc[..len], samples[mid - 1], &mut out).unwrap();
        assert_eq!(out, samples[mid..]);
    }
}

// SVB-ZD built from the slice functions: widen to i32, delta, zigzag, U32Classic.
fn svbzd_reference(samples: &[i16], initial: i32) -> Vec<u8> {
    let wide: Vec<i32> = samples.iter().map(|&s| s as i32).collect();
    let mut deltas = vec![0i32; wide.len()];
    delta::encode_with_initial_to_slice(initial, &wide, &mut deltas);
    let mut codes = vec![0u32; wide.len()];
    zigzag::encode_to_slice(&deltas, &mut codes);
    let mut enc = vec![0u8; U32Classic.max_encoded_len(codes.len())];
    let len = U32Classic.encode_to_slice(&codes, &mut enc).unwrap();
    enc.truncate(len);
    enc
}

#[test]
fn svbzd_encode_and_fused_decode() {
    for (_, samples) in pod5_reads() {
        let n = samples.len();
        let reference = svbzd_reference(&samples, 0);
        let mut enc = vec![0u8; svb::svbzd_max_encoded_len(n)];
        let len = svb::encode_svbzd_to_slice(&samples, &mut enc).unwrap();
        assert_eq!(enc[..len], reference);

        let mut out = vec![0i16; n];
        svb::decode_svbzd_fused_to_slice(&reference, &mut out).unwrap();
        assert_eq!(out, samples);

        let mid = n / 2;
        let second = svbzd_reference(&samples[mid..], samples[mid - 1] as i32);
        let mut out = vec![0i16; n - mid];
        svb::decode_svbzd_fused_from_to_slice(&second, samples[mid - 1] as i32, &mut out).unwrap();
        assert_eq!(out, samples[mid..]);

        let mut longer = reference.clone();
        longer.push(0);
        let mut out = vec![0i16; n];
        assert!(matches!(
            svb::decode_svbzd_fused_to_slice(&longer, &mut out),
            Err(DecodeError::TrailingData { .. })
        ));
    }
}

#[test]
fn long_read_decodes() {
    // The longest POD5 read (101988 samples), through Svb16 and the fused
    // decoder.
    let enc: &[u8] = include_bytes!("vectors/bench_00_101988.svb16");
    let samples = le_i16(include_bytes!("vectors/bench_00_101988.i16"));
    let mut codes = vec![0u16; samples.len()];
    assert_eq!(Svb16.decode_to_slice(enc, &mut codes).unwrap(), enc.len());
    assert_eq!(codes, vbz_codes(&samples));

    let mut out = vec![0i16; samples.len()];
    svb::decode_vbz_fused_to_slice(enc, &mut out).unwrap();
    assert_eq!(out, samples);
}

// A VBZ-K frame built from the format description: byte k, then for each
// sub-stream i >= 1 the carry (the sample before its first one, i16 LE) and
// the offset of its first data byte (u32 LE), then the VBZ bytes of all the
// samples. Sub-streams 0..k-1 hold `(n / k) & !7` samples each.
fn vbzk_frame(samples: &[i16], k: usize) -> Vec<u8> {
    let n = samples.len();
    let n_sub = (n / k) & !7;
    assert!(k == 1 || n_sub > 0);
    let codes = vbz_codes(samples);
    let mut vbz = vec![0u8; Svb16.max_encoded_len(n)];
    let len = Svb16.encode_to_slice(&codes, &mut vbz).unwrap();
    vbz.truncate(len);

    let ctrl = &vbz[..n.div_ceil(8)];
    let width = |j: usize| 1 + ((ctrl[j / 8] >> (j % 8)) & 1) as u32;
    let mut frame = vec![k as u8];
    for i in 1..k {
        let split = i * n_sub;
        let offset: u32 = (0..split).map(width).sum();
        frame.extend_from_slice(&samples[split - 1].to_le_bytes());
        frame.extend_from_slice(&offset.to_le_bytes());
    }
    frame.extend_from_slice(&vbz);
    frame
}

#[test]
fn vbzk_decode() {
    for (_, samples) in pod5_reads() {
        let n = samples.len();
        for k in [1, 2, 4, 7] {
            let frame = vbzk_frame(&samples, k);
            let mut enc = vec![0u8; svb::vbzk_max_encoded_len(n, k)];
            let len = svb::encode_vbzk_to_slice(&samples, k, &mut enc).unwrap();
            assert_eq!(enc[..len], frame, "k={k}");

            let mut out = vec![0i16; n];
            svb::decode_vbzk_to_slice(&frame, &mut out).unwrap();
            assert_eq!(out, samples, "k={k}");

            let mut longer = frame.clone();
            longer.push(0);
            assert!(matches!(
                svb::decode_vbzk_to_slice(&longer, &mut out),
                Err(DecodeError::TrailingData { .. })
            ));
        }

        // Sub-stream 2 of 4 with a data offset past the end of the frame. That
        // offset is also where sub-stream 1's data ends, so sub-stream 1 is the
        // first one out of range.
        let mut bad = vbzk_frame(&samples, 4);
        let entry = 1 + 6 + 2; // header entry for sub-stream 2, offset field
        bad[entry..entry + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut out = vec![0i16; n];
        let n_sub = (n / 4) & !7;
        assert!(matches!(
            svb::decode_vbzk_to_slice(&bad, &mut out),
            Err(DecodeError::DataTruncated { index }) if index == n_sub
        ));
    }
}

// A frame with two faults: sub-stream 0 claims one data byte fewer than it
// has (one 2-byte tag cleared), and sub-stream 2's data ends past the end of
// the frame. Every offset is checked before any sub-stream is decoded, so the
// bad offset is reported, as the parallel decoder does.
#[test]
fn vbzk_offsets_are_checked_before_decoding() {
    for (_, samples) in pod5_reads() {
        let n = samples.len();
        let k = 4;
        let n_sub = (n / k) & !7;
        let mut frame = vbzk_frame(&samples, k);
        let header_len = 1 + (k - 1) * 6;

        let j = (0..n_sub)
            .find(|&j| frame[header_len + j / 8] >> (j % 8) & 1 == 1)
            .expect("a 2-byte value in sub-stream 0");
        frame[header_len + j / 8] &= !(1 << (j % 8));
        let mut out = vec![0i16; n];
        // On its own, the cleared tag gives TrailingData for sub-stream 0.
        assert!(matches!(
            svb::decode_vbzk_to_slice(&frame, &mut out),
            Err(DecodeError::TrailingData { .. })
        ));

        let entry = 1 + 2 * 6 + 2; // offset field of sub-stream 3, the end of 2
        frame[entry..entry + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let is_offset_error =
            |r| matches!(r, Err(DecodeError::DataTruncated { index }) if index == 2 * n_sub);
        assert!(is_offset_error(svb::decode_vbzk_to_slice(&frame, &mut out)));
        #[cfg(feature = "std")]
        assert!(is_offset_error(svb::decode_vbzk_parallel_to_slice(
            &frame, &mut out
        )));
    }
}

// encode_vbz_to_slice against delta, zigzag and Svb16 run as separate passes.
// The encoder works in blocks of 1024 samples; the lengths below include
// every length up to 600 and the lengths around one and two blocks, so the
// carry between blocks and short last blocks are covered.
#[test]
fn vbz_encode_matches_separate_passes() {
    let mut x = 0x1234_5678u32;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        x
    };
    // Mostly small steps with some large jumps, so both widths occur.
    let mut samples = vec![0i16; 2100];
    let mut s = 0i16;
    for v in samples.iter_mut() {
        let r = next();
        let step = if r % 5 == 0 {
            r as i16
        } else {
            (r % 200) as i16 - 100
        };
        s = s.wrapping_add(step);
        *v = s;
    }
    let check = |samples: &[i16]| {
        let codes = vbz_codes(samples);
        let mut want = vec![0u8; Svb16.max_encoded_len(samples.len())];
        let want_len = Svb16.encode_to_slice(&codes, &mut want).unwrap();

        let mut got = vec![0xA5u8; svb::vbz_max_encoded_len(samples.len())];
        let len = svb::encode_vbz_to_slice(samples, &mut got).unwrap();
        assert_eq!(got[..len], want[..want_len], "n={}", samples.len());
    };
    for n in (0..=600).chain(1000..=1050).chain(2030..=2100) {
        check(&samples[..n]);
    }
    for (enc, samples) in pod5_reads() {
        check(&samples);
        let mut got = vec![0u8; svb::vbz_max_encoded_len(samples.len())];
        let len = svb::encode_vbz_to_slice(&samples, &mut got).unwrap();
        assert_eq!(got[..len], *enc);
    }

    // One byte short of the worst case is rejected before anything is written.
    let mut small = vec![0xA5u8; svb::vbz_max_encoded_len(9) - 1];
    assert!(matches!(
        svb::encode_vbz_to_slice(&samples[..9], &mut small),
        Err(svb::EncodeError::OutputTooSmall { .. })
    ));
    assert!(small.iter().all(|&b| b == 0xA5));
}
