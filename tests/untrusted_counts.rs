//! A large value count with a short input must fail without allocating the
//! output first.
//!
//! The count often comes from a file header, so it cannot be trusted. Each
//! call below asks for 2^40 values from 10 bytes. If a decoder allocated
//! before checking, the allocation (2-8 TB) fails and the test process
//! aborts.

use svb::DecodeError;
use svb::u16::Svb16;
use svb::u32::{U32Classic, U32Variant0124};
use svb::u64::{U64Coder1234, U64Coder1248};

const HUGE: usize = 1 << 40;
const DATA: [u8; 10] = [0; 10];

fn assert_short<T: core::fmt::Debug>(name: &str, r: Result<T, DecodeError>) {
    assert!(
        matches!(r, Err(DecodeError::ControlStreamTooShort { .. })),
        "{name}: {r:?}"
    );
}

#[test]
fn codecs() {
    assert_short("Svb16.decode", Svb16.decode(&DATA, HUGE));
    assert_short(
        "Svb16.decode_into",
        Svb16.decode_into(&DATA, HUGE, &mut Vec::new()),
    );
    assert_short("U32Classic.decode", U32Classic.decode(&DATA, HUGE));
    assert_short(
        "U32Classic.decode_into",
        U32Classic.decode_into(&DATA, HUGE, &mut Vec::new()),
    );
    assert_short("U32Variant0124.decode", U32Variant0124.decode(&DATA, HUGE));
    assert_short(
        "U32Variant0124.decode_into",
        U32Variant0124.decode_into(&DATA, HUGE, &mut Vec::new()),
    );
    assert_short("U64Coder1234.decode", U64Coder1234.decode(&DATA, HUGE));
    assert_short(
        "U64Coder1234.decode_into",
        U64Coder1234.decode_into(&DATA, HUGE, &mut Vec::new()),
    );
    assert_short("U64Coder1248.decode", U64Coder1248.decode(&DATA, HUGE));
    assert_short(
        "U64Coder1248.decode_into",
        U64Coder1248.decode_into(&DATA, HUGE, &mut Vec::new()),
    );
}

#[test]
fn pipelines() {
    let mut v = vec![1i16, 2];
    assert_short("decode_vbz", svb::decode_vbz(&DATA, HUGE));
    assert_short("decode_vbz_into", svb::decode_vbz_into(&DATA, HUGE, &mut v));
    assert_short("decode_vbz_fused", svb::decode_vbz_fused(&DATA, HUGE));
    assert_short(
        "decode_vbz_fused_into",
        svb::decode_vbz_fused_into(&DATA, HUGE, &mut v),
    );
    assert_short(
        "decode_vbz_fused_from",
        svb::decode_vbz_fused_from(&DATA, HUGE, 0),
    );
    assert_short(
        "decode_vbz_fused_from_into",
        svb::decode_vbz_fused_from_into(&DATA, HUGE, 0, &mut v),
    );
    assert_short("decode_vbzk", svb::decode_vbzk(&DATA, HUGE));
    assert_short(
        "decode_vbzk_into",
        svb::decode_vbzk_into(&DATA, HUGE, &mut v),
    );
    #[cfg(feature = "std")]
    assert_short(
        "decode_vbzk_parallel_into",
        svb::decode_vbzk_parallel_into(&DATA, HUGE, &mut v),
    );
    assert_short("decode_svbzd", svb::decode_svbzd(&DATA, HUGE));
    assert_short(
        "decode_svbzd_into",
        svb::decode_svbzd_into(&DATA, HUGE, &mut v),
    );
    assert_short("decode_svbzd_fused", svb::decode_svbzd_fused(&DATA, HUGE));
    assert_short(
        "decode_svbzd_fused_into",
        svb::decode_svbzd_fused_into(&DATA, HUGE, &mut v),
    );
    assert_short(
        "decode_svbzd_fused_from",
        svb::decode_svbzd_fused_from(&DATA, HUGE, 0),
    );
    assert_short(
        "decode_svbzd_fused_from_into",
        svb::decode_svbzd_fused_from_into(&DATA, HUGE, 0, &mut v),
    );
    assert_eq!(v, [1, 2], "an _into function changed the Vec on error");
}

#[test]
fn patched_count() {
    assert_short(
        "patched::decode_into",
        svb::patched::decode_into(&DATA, HUGE, &mut Vec::new()),
    );
}

#[test]
fn patched_exception_count_above_n() {
    // n = 9 and enough bytes for the length check, but the exception count
    // in the frame is u32::MAX. It must be rejected before the exception
    // arrays are decoded.
    let mut data = vec![0xFF, 0xFF, 0xFF, 0xFF];
    data.extend([0u8; 32]);
    let mut out = [0u16; 9];
    let r = svb::patched::decode_to_slice(&data, &mut out);
    // Decoding the exception arrays would fail too, with
    // ControlStreamTooShort; this error comes only from the early check.
    assert!(
        matches!(r, Err(DecodeError::DataTruncated { index: 9 })),
        "{r:?}"
    );
}
