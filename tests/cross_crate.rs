//! Cross-crate parity tests: `svb` ↔ `streamvbyte64`.
//!
//! For each shared codec variant we verify:
//!   A) `streamvbyte64` encodes → `svb` decodes → original values
//!   B) `svb` encodes → `streamvbyte64` decodes → original values
//!
//! Covered variants:
//!   - `U32Classic`    ↔ `streamvbyte64::Coder1234` (u32, widths 1/2/3/4)
//!   - `U32Variant0124`↔ `streamvbyte64::Coder0124` (u32, widths 0/1/2/4)
//!   - `U64Coder1248`  ↔ `streamvbyte64::Coder1248` (u64, widths 1/2/4/8)
//!
//! `U64Coder1234` and `Svb16`/VBZ have no counterpart in `streamvbyte64` and
//! are not tested here.
//!
//! # Wire format notes
//!
//! `svb` stores `ceil(n/4)` control bytes immediately followed by data bytes in
//! a single `Vec<u8>`.  `streamvbyte64` keeps the tag and data streams in
//! separate buffers.  The two layouts are trivially interconverted:
//!
//!   svb → sv64:  split at `ctrl_len = n.div_ceil(4)` bytes
//!   sv64 → svb:  concatenate `tags ++ data`
//!
//! `streamvbyte64::Coder::encode` panics if `values.len() % 4 != 0`, so every
//! test vector length must be a multiple of 4.  "Single element" and other small
//! counts are rounded up to the nearest multiple of 4 (padding with a neutral
//! value that fits in 1 byte).
//!
//! # Key boundary values by codec
//!
//! U32Classic / Coder1234 (tag → bytes):  0→1, 1→2, 2→3, 3→4
//!   fit-in-1: 0x00–0xFF   fit-in-2: 0x100–0xFFFF
//!   fit-in-3: 0x10000–0xFFFFFF   fit-in-4: 0x1000000–0xFFFFFFFF
//!
//! U32Variant0124 / Coder0124 (tag → bytes):  0→0, 1→1, 2→2, 3→4  (no 3-byte)
//!   0-byte: exactly 0   1-byte: 0x01–0xFF   2-byte: 0x100–0xFFFF
//!   4-byte: 0x10000–0xFFFFFFFF
//!
//! U64Coder1248 / Coder1248 (tag → bytes):  0→1, 1→2, 2→4, 3→8
//!   1-byte: 0x01–0xFF   2-byte: 0x100–0xFFFF
//!   4-byte: 0x10000–0xFFFFFFFF   8-byte: 0x100000000–0xFFFFFFFFFFFFFFFF

use streamvbyte64::Coder as _;
use svb::u32::{U32Classic, U32Variant0124};
use svb::u64::U64Coder1248;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Build a `streamvbyte64` tag+data pair from a value slice, returning
/// `(tags, data, bytes_used_in_data)`.
fn sv64_encode_u32<C: streamvbyte64::Coder<Elem = u32>>(
    coder: &C,
    values: &[u32],
) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(values.len() % 4, 0, "streamvbyte64 requires len%4==0");
    let (tag_len, data_len) = C::max_compressed_bytes(values.len());
    let mut tags = vec![0u8; tag_len];
    let mut data = vec![0u8; data_len];
    let used = coder.encode(values, &mut tags, &mut data);
    data.truncate(used);
    (tags, data)
}

fn sv64_encode_u64<C: streamvbyte64::Coder<Elem = u64>>(
    coder: &C,
    values: &[u64],
) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(values.len() % 4, 0, "streamvbyte64 requires len%4==0");
    let (tag_len, data_len) = C::max_compressed_bytes(values.len());
    let mut tags = vec![0u8; tag_len];
    let mut data = vec![0u8; data_len];
    let used = coder.encode(values, &mut tags, &mut data);
    data.truncate(used);
    (tags, data)
}

/// Decode via `streamvbyte64` given a tag buffer and data buffer.
fn sv64_decode_u32<C: streamvbyte64::Coder<Elem = u32>>(
    coder: &C,
    tags: &[u8],
    data: &[u8],
    n: usize,
) -> Vec<u32> {
    let mut out = vec![0u32; n];
    coder.decode(tags, data, &mut out);
    out
}

fn sv64_decode_u64<C: streamvbyte64::Coder<Elem = u64>>(
    coder: &C,
    tags: &[u8],
    data: &[u8],
    n: usize,
) -> Vec<u64> {
    let mut out = vec![0u64; n];
    coder.decode(tags, data, &mut out);
    out
}

/// Split an `svb`-encoded buffer into (tags, data) for `streamvbyte64`.
fn split_svb_u32(encoded: &[u8], n: usize) -> (&[u8], &[u8]) {
    let ctrl_len = n.div_ceil(4);
    (&encoded[..ctrl_len], &encoded[ctrl_len..])
}

fn split_svb_u64(encoded: &[u8], n: usize) -> (&[u8], &[u8]) {
    let ctrl_len = n.div_ceil(4);
    (&encoded[..ctrl_len], &encoded[ctrl_len..])
}

/// Concatenate `streamvbyte64` tags + data into a flat buffer for `svb::decode`.
fn join_sv64(tags: &[u8], data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tags.len() + data.len());
    buf.extend_from_slice(tags);
    buf.extend_from_slice(data);
    buf
}

// ── U32Classic ↔ Coder1234 ────────────────────────────────────────────────────

mod u32_classic {
    use super::*;
    use streamvbyte64::Coder1234;

    // Direction A: sv64 encodes, svb decodes.
    fn sv64_enc_svb_dec(values: &[u32]) {
        let coder = Coder1234::new();
        let (tags, data) = sv64_encode_u32(&coder, values);
        let flat = join_sv64(&tags, &data);
        let got = U32Classic.decode(&flat, values.len()).unwrap();
        assert_eq!(got, values, "sv64→svb mismatch n={}", values.len());
    }

    // Direction B: svb encodes, sv64 decodes.
    fn svb_enc_sv64_dec(values: &[u32]) {
        let n = values.len();
        let encoded = U32Classic.encode(values);
        let (tags, data) = split_svb_u32(&encoded, n);
        let coder = Coder1234::new();
        let got = sv64_decode_u32(&coder, tags, data, n);
        assert_eq!(got, values, "svb→sv64 mismatch n={n}");
    }

    fn check(values: &[u32]) {
        sv64_enc_svb_dec(values);
        svb_enc_sv64_dec(values);
    }

    // All-small: every value fits in 1 byte (tag 0).
    #[test]
    fn all_small() {
        let v: Vec<u32> = (0..8u32).map(|i| i % 200).collect();
        check(&v);
    }

    // All-large: every value needs 4 bytes (tag 3).
    #[test]
    fn all_large() {
        let v: Vec<u32> = (0..8u32).map(|i| 0x1000000 + i).collect();
        check(&v);
    }

    // Mixed: one value per tag width.
    #[test]
    fn mixed_widths() {
        let v: Vec<u32> = (0..8u32)
            .map(|i| match i % 4 {
                0 => i % 256,       // 1-byte
                1 => 0x100 + i,     // 2-byte
                2 => 0x10000 + i,   // 3-byte
                _ => 0x1000000 + i, // 4-byte
            })
            .collect();
        check(&v);
    }

    // Empty slice.
    #[test]
    fn empty() {
        check(&[]);
    }

    // Single element, padded to 4.
    #[test]
    fn single_element() {
        for &v in &[
            0u32,
            0x01,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0xFF_FFFF,
            0x100_0000,
            u32::MAX,
        ] {
            // pad to multiple of 4 with a neutral 1-byte value
            let vals = [v, 1, 1, 1];
            sv64_enc_svb_dec(&vals);
            svb_enc_sv64_dec(&vals);
        }
    }

    // Large slice (1024 values).
    #[test]
    fn large_1024() {
        let v: Vec<u32> = (0..1024u32)
            .map(|i| match i % 4 {
                0 => i % 256,
                1 => 0x100 + i % 0xFFFF,
                2 => 0x10000 + i % 0xFF_FFFF,
                _ => 0x1000000 + i,
            })
            .collect();
        check(&v);
    }

    // Boundary values (all four tag thresholds).
    #[test]
    fn boundary_values() {
        let pool: Vec<u32> = [
            0u32,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0xFF_FFFF,
            0x100_0000,
            u32::MAX,
        ]
        .iter()
        .copied()
        .cycle()
        .take(32)
        .collect();
        check(&pool);
    }

    // All-zeros edge case.
    #[test]
    fn all_zeros() {
        let v = vec![0u32; 8];
        check(&v);
    }
}

// ── U32Variant0124 ↔ Coder0124 ────────────────────────────────────────────────

mod u32_variant0124 {
    use super::*;
    use streamvbyte64::Coder0124;

    fn sv64_enc_svb_dec(values: &[u32]) {
        let coder = Coder0124::new();
        let (tags, data) = sv64_encode_u32(&coder, values);
        let flat = join_sv64(&tags, &data);
        let got = U32Variant0124.decode(&flat, values.len()).unwrap();
        assert_eq!(got, values, "sv64→svb mismatch n={}", values.len());
    }

    fn svb_enc_sv64_dec(values: &[u32]) {
        let n = values.len();
        let encoded = U32Variant0124.encode(values);
        let (tags, data) = split_svb_u32(&encoded, n);
        let coder = Coder0124::new();
        let got = sv64_decode_u32(&coder, tags, data, n);
        assert_eq!(got, values, "svb→sv64 mismatch n={n}");
    }

    fn check(values: &[u32]) {
        sv64_enc_svb_dec(values);
        svb_enc_sv64_dec(values);
    }

    // All-small: values fit in 1 byte (tag 1).
    #[test]
    fn all_small() {
        let v: Vec<u32> = (0..8u32).map(|i| (i % 200) + 1).collect();
        check(&v);
    }

    // All-large: values need 4 bytes (tag 3).
    #[test]
    fn all_large() {
        let v: Vec<u32> = (0..8u32).map(|i| 0x10000 + i).collect();
        check(&v);
    }

    // Mixed widths: 0-byte, 1-byte, 2-byte, 4-byte.
    #[test]
    fn mixed_widths() {
        let v: Vec<u32> = (0..8u32)
            .map(|i| match i % 4 {
                0 => 0,             // 0-byte
                1 => (i % 255) + 1, // 1-byte
                2 => 0x100 + i,     // 2-byte
                _ => 0x10000 + i,   // 4-byte
            })
            .collect();
        check(&v);
    }

    // Empty slice.
    #[test]
    fn empty() {
        check(&[]);
    }

    // Single element, padded to 4.
    #[test]
    fn single_element() {
        for &v in &[0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX] {
            let vals = [v, 1, 1, 1];
            sv64_enc_svb_dec(&vals);
            svb_enc_sv64_dec(&vals);
        }
    }

    // Large slice (1024 values).
    #[test]
    fn large_1024() {
        let v: Vec<u32> = (0..1024u32)
            .map(|i| match i % 4 {
                0 => 0,
                1 => (i % 255) + 1,
                2 => 0x100 + i % 0xFFFF,
                _ => 0x10000 + i,
            })
            .collect();
        check(&v);
    }

    // Boundary values.
    #[test]
    fn boundary_values() {
        let pool: Vec<u32> = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX]
            .iter()
            .copied()
            .cycle()
            .take(28)
            .collect();
        check(&pool);
    }

    // All-zeros: entire payload in control bytes, no data bytes.
    #[test]
    fn all_zeros() {
        let v = vec![0u32; 8];
        check(&v);
    }
}

// ── U64Coder1248 ↔ Coder1248 ─────────────────────────────────────────────────

mod u64_coder1248 {
    use super::*;
    use streamvbyte64::Coder1248;

    fn sv64_enc_svb_dec(values: &[u64]) {
        let coder = Coder1248::new();
        let (tags, data) = sv64_encode_u64(&coder, values);
        let flat = join_sv64(&tags, &data);
        let got = U64Coder1248.decode(&flat, values.len()).unwrap();
        assert_eq!(got, values, "sv64→svb mismatch n={}", values.len());
    }

    fn svb_enc_sv64_dec(values: &[u64]) {
        let n = values.len();
        let encoded = U64Coder1248.encode(values);
        let (tags, data) = split_svb_u64(&encoded, n);
        let coder = Coder1248::new();
        let got = sv64_decode_u64(&coder, tags, data, n);
        assert_eq!(got, values, "svb→sv64 mismatch n={n}");
    }

    fn check(values: &[u64]) {
        sv64_enc_svb_dec(values);
        svb_enc_sv64_dec(values);
    }

    // All-small: 1-byte values (tag 0).
    #[test]
    fn all_small() {
        let v: Vec<u64> = (0..8u64).map(|i| (i % 200) + 1).collect();
        check(&v);
    }

    // All-large: 8-byte values (tag 3).
    #[test]
    fn all_large() {
        let v: Vec<u64> = (0..8u64).map(|i| 0x1_0000_0000 + i).collect();
        check(&v);
    }

    // Mixed widths: 1-byte, 2-byte, 4-byte, 8-byte.
    #[test]
    fn mixed_widths() {
        let v: Vec<u64> = (0..8u64)
            .map(|i| match i % 4 {
                0 => (i % 255) + 1,     // 1-byte
                1 => 0x100 + i,         // 2-byte
                2 => 0x10000 + i,       // 4-byte
                _ => 0x1_0000_0000 + i, // 8-byte
            })
            .collect();
        check(&v);
    }

    // Empty slice.
    #[test]
    fn empty() {
        check(&[]);
    }

    // Single element, padded to 4.
    #[test]
    fn single_element() {
        for &v in &[
            1u64,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0xFFFF_FFFF,
            0x1_0000_0000,
            u64::MAX,
        ] {
            let vals = [v, 1, 1, 1];
            sv64_enc_svb_dec(&vals);
            svb_enc_sv64_dec(&vals);
        }
    }

    // Large slice (1024 values).
    #[test]
    fn large_1024() {
        let v: Vec<u64> = (0..1024u64)
            .map(|i| match i % 4 {
                0 => (i % 255) + 1,
                1 => 0x100 + i % 0xFFFF,
                2 => 0x10000 + i % 0xFFFF_FFFF,
                _ => 0x1_0000_0000 + i,
            })
            .collect();
        check(&v);
    }

    // Boundary values (all tag thresholds).
    #[test]
    fn boundary_values() {
        let pool: Vec<u64> = [
            1u64,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0xFFFF_FFFF,
            0x1_0000_0000,
            u64::MAX,
        ]
        .iter()
        .copied()
        .cycle()
        .take(32)
        .collect();
        check(&pool);
    }

    // Full u64 range check (values that need 8 bytes).
    #[test]
    fn full_u64_range() {
        let v: Vec<u64> = [u64::MAX, u64::MAX - 1, 0x1_0000_0000, 0xFFFF_FFFF_FFFF_FF00]
            .iter()
            .copied()
            .cycle()
            .take(8)
            .collect();
        check(&v);
    }
}
