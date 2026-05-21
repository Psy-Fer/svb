//! Wire-compatibility tests between `svb` and `streamvbyte64`.
//!
//! Each test encodes data with one library and decodes it with the other, then
//! asserts the round-trip is lossless.  A failure here means the two libraries
//! produce incompatible wire formats and is a critical regression.
//!
//! `streamvbyte64` keeps tags and data in separate buffers; `svb` concatenates
//! them (tags first).  `split_flat` handles the conversion.
//!
//! Note: `streamvbyte64` processes values in groups of 4 and may produce
//! undefined results for n that is not a multiple of 4, so all test sizes are
//! multiples of 4.

use streamvbyte64::Coder as _;
use svb::u32::{U32Classic, U32Variant0124};
use svb::u64::{U64Coder1234, U64Coder1248};

// Split svb's flat [tags | data] buffer into separate (tags, data) slices
// for streamvbyte64's decoder.
fn split_flat(encoded: &[u8], n: usize) -> (&[u8], &[u8]) {
    let ctrl_len = n.div_ceil(4);
    (&encoded[..ctrl_len], &encoded[ctrl_len..])
}

// Concatenate streamvbyte64's separate (tags, data) into svb's flat format.
fn join_flat(tags: &[u8], data: &[u8]) -> Vec<u8> {
    let mut flat = tags.to_vec();
    flat.extend_from_slice(data);
    flat
}

// ── test data generators ──────────────────────────────────────────────────────

fn u32_mixed(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| match i % 4 {
            0 => i % 256,
            1 => i % 65536,
            2 => i % 16_777_216,
            _ => i,
        })
        .collect()
}

fn u32_sparse(n: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| if i % 3 == 0 { 0 } else { i % 256 })
        .collect()
}

fn u64_wide(n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| match i % 4 {
            0 => i % 256,
            1 => i % 65536,
            2 => i % 0xFFFF_FFFF,
            _ => i | 0x0100_0000_0000_0000,
        })
        .collect()
}

// Values that fit in u32, for testing U64Coder1234 ↔ Coder1234 parity.
fn u64_u32range(n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| match i % 4 {
            0 => i % 256,
            1 => i % 65536,
            2 => i % 16_777_216,
            _ => i % (u32::MAX as u64),
        })
        .collect()
}

const SIZES: &[usize] = &[4, 128, 1024];

// ── U32Classic (svb) ↔ Coder1234 (streamvbyte64) ─────────────────────────────

#[test]
fn u32_classic_svb_encode_sv64_decode() {
    let sv64 = streamvbyte64::Coder1234::new();
    for &n in SIZES {
        let values = u32_mixed(n);
        let encoded = U32Classic.encode(&values);
        let (tags, data) = split_flat(&encoded, n);
        let mut decoded = vec![0u32; n];
        sv64.decode(tags, data, &mut decoded);
        assert_eq!(decoded, values, "n={n}");
    }
}

#[test]
fn u32_classic_sv64_encode_svb_decode() {
    let sv64 = streamvbyte64::Coder1234::new();
    for &n in SIZES {
        let values = u32_mixed(n);
        let (tl, dl) = streamvbyte64::Coder1234::max_compressed_bytes(n);
        let mut tags = vec![0u8; tl];
        let mut data = vec![0u8; dl];
        let used = sv64.encode(&values, &mut tags, &mut data);
        let flat = join_flat(&tags, &data[..used]);
        assert_eq!(U32Classic.decode(&flat, n).unwrap(), values, "n={n}");
    }
}

// ── U32Variant0124 (svb) ↔ Coder0124 (streamvbyte64) ─────────────────────────

#[test]
fn u32_variant0124_svb_encode_sv64_decode() {
    let sv64 = streamvbyte64::Coder0124::new();
    for &n in SIZES {
        let values = u32_sparse(n);
        let encoded = U32Variant0124.encode(&values);
        let (tags, data) = split_flat(&encoded, n);
        let mut decoded = vec![0u32; n];
        sv64.decode(tags, data, &mut decoded);
        assert_eq!(decoded, values, "n={n}");
    }
}

#[test]
fn u32_variant0124_sv64_encode_svb_decode() {
    let sv64 = streamvbyte64::Coder0124::new();
    for &n in SIZES {
        let values = u32_sparse(n);
        let (tl, dl) = streamvbyte64::Coder0124::max_compressed_bytes(n);
        let mut tags = vec![0u8; tl];
        let mut data = vec![0u8; dl];
        let used = sv64.encode(&values, &mut tags, &mut data);
        let flat = join_flat(&tags, &data[..used]);
        assert_eq!(U32Variant0124.decode(&flat, n).unwrap(), values, "n={n}");
    }
}

// ── U64Coder1248 (svb) ↔ Coder1248 (streamvbyte64) ───────────────────────────

#[test]
fn u64_coder1248_svb_encode_sv64_decode() {
    let sv64 = streamvbyte64::Coder1248::new();
    for &n in SIZES {
        let values = u64_wide(n);
        let encoded = U64Coder1248.encode(&values);
        let (tags, data) = split_flat(&encoded, n);
        let mut decoded = vec![0u64; n];
        sv64.decode(tags, data, &mut decoded);
        assert_eq!(decoded, values, "n={n}");
    }
}

#[test]
fn u64_coder1248_sv64_encode_svb_decode() {
    let sv64 = streamvbyte64::Coder1248::new();
    for &n in SIZES {
        let values = u64_wide(n);
        let (tl, dl) = streamvbyte64::Coder1248::max_compressed_bytes(n);
        let mut tags = vec![0u8; tl];
        let mut data = vec![0u8; dl];
        let used = sv64.encode(&values, &mut tags, &mut data);
        let flat = join_flat(&tags, &data[..used]);
        assert_eq!(U64Coder1248.decode(&flat, n).unwrap(), values, "n={n}");
    }
}

// ── U64Coder1234 (svb) ↔ Coder1234 (streamvbyte64) ───────────────────────────
//
// U64Coder1234 has the same tag/width table and wire format as Coder1234 (u32);
// values are stored in 1–4 bytes with zero-extension to u64 on decode.
// Both libraries must produce identical tag bytes and data bytes for the same
// values (as long as those values fit within u32::MAX).

#[test]
fn u64_coder1234_svb_encode_sv64_decode() {
    let sv64 = streamvbyte64::Coder1234::new();
    for &n in SIZES {
        let values_u64 = u64_u32range(n);
        let values_u32: Vec<u32> = values_u64.iter().map(|&v| v as u32).collect();
        let encoded = U64Coder1234.encode(&values_u64);
        let (tags, data) = split_flat(&encoded, n);
        let mut decoded = vec![0u32; n];
        sv64.decode(tags, data, &mut decoded);
        assert_eq!(decoded, values_u32, "n={n}");
    }
}

#[test]
fn u64_coder1234_sv64_encode_svb_decode() {
    let sv64 = streamvbyte64::Coder1234::new();
    for &n in SIZES {
        let values_u64 = u64_u32range(n);
        let values_u32: Vec<u32> = values_u64.iter().map(|&v| v as u32).collect();
        let (tl, dl) = streamvbyte64::Coder1234::max_compressed_bytes(n);
        let mut tags = vec![0u8; tl];
        let mut data = vec![0u8; dl];
        let used = sv64.encode(&values_u32, &mut tags, &mut data);
        let flat = join_flat(&tags, &data[..used]);
        assert_eq!(U64Coder1234.decode(&flat, n).unwrap(), values_u64, "n={n}");
    }
}
