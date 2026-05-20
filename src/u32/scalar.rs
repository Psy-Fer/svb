#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

// ── shared helpers ────────────────────────────────────────────────────────────

/// Extract the 2-bit tag for value `i` from the control stream.
#[inline]
fn get_tag(ctrl: &[u8], i: usize) -> u8 {
    (ctrl[i / 4] >> ((i % 4) * 2)) & 0x03
}

// ── U32Classic (Lemire) ───────────────────────────────────────────────────────
// tag → byte width: tag + 1  (0→1, 1→2, 2→3, 3→4)
// value ranges:  0x00–0xFF→tag0, 0x100–0xFFFF→tag1,
//                0x10000–0xFFFFFF→tag2, 0x1000000–0xFFFFFFFF→tag3

pub(super) fn encode_into_classic(values: &[u32], out: &mut Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return;
    }
    let ctrl_len = n.div_ceil(4);
    let ctrl_start = out.len();
    out.resize(ctrl_start + ctrl_len, 0u8);

    for (i, &v) in values.iter().enumerate() {
        let (tag, count): (u8, usize) = if v <= 0xFF {
            (0, 1)
        } else if v <= 0xFFFF {
            (1, 2)
        } else if v <= 0xFF_FFFF {
            (2, 3)
        } else {
            (3, 4)
        };
        out[ctrl_start + i / 4] |= tag << ((i % 4) * 2);
        out.extend_from_slice(&v.to_le_bytes()[..count]);
    }
}

/// Decode `n` U32Classic values from pre-split `ctrl` and `data` byte slices.
///
/// Unlike [`decode_into_classic`], this function does not split the buffer —
/// the caller is responsible for passing the ctrl and data streams separately.
/// Used by SIMD decode paths to handle the scalar tail.
pub(super) fn decode_classic_from_raw(
    ctrl: &[u8],
    data: &[u8],
    n: usize,
    out: &mut Vec<u32>,
) -> Result<(), DecodeError> {
    out.reserve(n);
    let mut pos = 0usize;
    for i in 0..n {
        let tag = (ctrl[i / 4] >> ((i % 4) * 2)) & 3;
        let count = (tag + 1) as usize;
        if pos + count > data.len() {
            return Err(DecodeError::DataTruncated { index: i });
        }
        let mut bytes = [0u8; 4];
        bytes[..count].copy_from_slice(&data[pos..pos + count]);
        out.push(u32::from_le_bytes(bytes));
        pos += count;
    }
    Ok(())
}

pub(super) fn decode_into_classic(
    data: &[u8],
    n: usize,
    out: &mut Vec<u32>,
) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    let ctrl_len = n.div_ceil(4);
    if data.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: data.len(),
        });
    }
    let ctrl = &data[..ctrl_len];
    let mut pos = ctrl_len;
    out.reserve(n);

    for i in 0..n {
        let tag = get_tag(ctrl, i);
        let count = (tag + 1) as usize;
        if pos + count > data.len() {
            return Err(DecodeError::DataTruncated { index: i });
        }
        let mut bytes = [0u8; 4];
        bytes[..count].copy_from_slice(&data[pos..pos + count]);
        out.push(u32::from_le_bytes(bytes));
        pos += count;
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn encoded_data_len_classic(ctrl: &[u8], n: usize) -> usize {
    // byte_width = tag + 1, so data_len = n + sum(tag_i)
    let mut sum = n;
    let full = n / 4;
    let rem = n % 4;
    for &b in &ctrl[..full] {
        sum += ((b & 0x03) + ((b >> 2) & 0x03) + ((b >> 4) & 0x03) + ((b >> 6) & 0x03)) as usize;
    }
    for j in 0..rem {
        sum += ((ctrl[full] >> (j * 2)) & 0x03) as usize;
    }
    sum
}

// ── U32Variant0124 ────────────────────────────────────────────────────────────
// tag → byte width: [0, 1, 2, 4]
// value ranges:  0→tag0 (0 bytes), 0x01–0xFF→tag1,
//                0x100–0xFFFF→tag2, 0x10000–0xFFFFFFFF→tag3 (no 3-byte option)

const WIDTHS_0124: [usize; 4] = [0, 1, 2, 4];

pub(super) fn encode_into_0124(values: &[u32], out: &mut Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return;
    }
    let ctrl_len = n.div_ceil(4);
    let ctrl_start = out.len();
    out.resize(ctrl_start + ctrl_len, 0u8);

    for (i, &v) in values.iter().enumerate() {
        let (tag, count): (u8, usize) = if v == 0 {
            (0, 0)
        } else if v <= 0xFF {
            (1, 1)
        } else if v <= 0xFFFF {
            (2, 2)
        } else {
            (3, 4)
        };
        out[ctrl_start + i / 4] |= tag << ((i % 4) * 2);
        if count > 0 {
            out.extend_from_slice(&v.to_le_bytes()[..count]);
        }
    }
}

pub(super) fn decode_into_0124(
    data: &[u8],
    n: usize,
    out: &mut Vec<u32>,
) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    let ctrl_len = n.div_ceil(4);
    if data.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: data.len(),
        });
    }
    let ctrl = &data[..ctrl_len];
    let mut pos = ctrl_len;
    out.reserve(n);

    for i in 0..n {
        let tag = get_tag(ctrl, i);
        let count = WIDTHS_0124[tag as usize];
        if count == 0 {
            out.push(0);
        } else {
            if pos + count > data.len() {
                return Err(DecodeError::DataTruncated { index: i });
            }
            let mut bytes = [0u8; 4];
            bytes[..count].copy_from_slice(&data[pos..pos + count]);
            out.push(u32::from_le_bytes(bytes));
            pos += count;
        }
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn encoded_data_len_0124(ctrl: &[u8], n: usize) -> usize {
    let mut sum = 0usize;
    let full = n / 4;
    let rem = n % 4;
    for &b in &ctrl[..full] {
        for j in 0..4 {
            sum += WIDTHS_0124[((b >> (j * 2)) & 0x03) as usize];
        }
    }
    for j in 0..rem {
        sum += WIDTHS_0124[((ctrl[full] >> (j * 2)) & 0x03) as usize];
    }
    sum
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn enc_classic(v: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_into_classic(v, &mut out);
        out
    }
    fn dec_classic(d: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::new();
        decode_into_classic(d, n, &mut out)?;
        Ok(out)
    }
    fn enc_0124(v: &[u32]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_into_0124(v, &mut out);
        out
    }
    fn dec_0124(d: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = Vec::new();
        decode_into_0124(d, n, &mut out)?;
        Ok(out)
    }

    // ── U32Classic spec example ───────────────────────────────────────────────

    #[test]
    fn classic_spec_example_encode() {
        let got = enc_classic(&[1, 256, 65536, 0xFFFF_FFFF]);
        assert_eq!(
            got,
            [
                0xE4, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF
            ]
        );
    }

    #[test]
    fn classic_spec_example_decode() {
        let data = [
            0xE4u8, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF,
        ];
        assert_eq!(dec_classic(&data, 4).unwrap(), [1, 256, 65536, 0xFFFF_FFFF]);
    }

    // ── U32Classic round-trips ────────────────────────────────────────────────

    #[test]
    fn classic_roundtrip_empty() {
        assert_eq!(dec_classic(&enc_classic(&[]), 0).unwrap(), &[] as &[u32]);
    }

    #[test]
    fn classic_roundtrip_boundaries() {
        let vals = [
            0u32,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0xFF_FFFF,
            0x100_0000,
            u32::MAX,
        ];
        assert_eq!(dec_classic(&enc_classic(&vals), vals.len()).unwrap(), vals);
    }

    #[test]
    fn classic_roundtrip_all_same_tag() {
        // 8 values that each fit in 1 byte → ctrl = [0x00, 0x00]
        let vals: Vec<u32> = (0..8).collect();
        let enc = enc_classic(&vals);
        assert_eq!(enc[..2], [0x00, 0x00]); // all tag-0
        assert_eq!(dec_classic(&enc, 8).unwrap(), vals);
    }

    #[test]
    fn classic_ctrl_byte_layout() {
        // 4 values each with a different tag to verify bit packing
        let enc = enc_classic(&[1, 256, 65536, 0xFFFF_FFFF]);
        assert_eq!(enc[0], 0xE4); // 0b11_10_01_00
    }

    // ── U32Classic encoded_data_len ───────────────────────────────────────────

    #[test]
    fn classic_data_len() {
        let enc = enc_classic(&[1, 256, 65536, 0xFFFF_FFFF]);
        let ctrl_len = 1usize;
        assert_eq!(
            encoded_data_len_classic(&enc[..ctrl_len], 4),
            enc.len() - ctrl_len
        );
    }

    // ── U32Classic errors ─────────────────────────────────────────────────────

    #[test]
    fn classic_error_ctrl_too_short() {
        // n=5 needs ceil(5/4)=2 ctrl bytes
        assert!(matches!(
            dec_classic(&[0x00], 5),
            Err(DecodeError::ControlStreamTooShort { need: 2, have: 1 })
        ));
    }

    #[test]
    fn classic_error_data_truncated() {
        // ctrl says first value needs 2 bytes (tag=1), but only ctrl byte present
        assert!(matches!(
            dec_classic(&[0x01], 1),
            Err(DecodeError::DataTruncated { index: 0 })
        ));
    }

    // ── U32Variant0124 spec example ───────────────────────────────────────────

    #[test]
    fn v0124_spec_example_encode() {
        let got = enc_0124(&[0, 1, 255, 256, 65535, 65536, 0xFFFF_FFFF]);
        assert_eq!(
            got,
            [
                0x94, 0x3E, 0x01, 0xFF, 0x00, 0x01, 0xFF, 0xFF, 0x00, 0x00, 0x01, 0x00, 0xFF, 0xFF,
                0xFF, 0xFF
            ]
        );
    }

    #[test]
    fn v0124_spec_example_decode() {
        let data = [
            0x94u8, 0x3E, 0x01, 0xFF, 0x00, 0x01, 0xFF, 0xFF, 0x00, 0x00, 0x01, 0x00, 0xFF, 0xFF,
            0xFF, 0xFF,
        ];
        assert_eq!(
            dec_0124(&data, 7).unwrap(),
            [0, 1, 255, 256, 65535, 65536, 0xFFFF_FFFF]
        );
    }

    // ── U32Variant0124 round-trips ────────────────────────────────────────────

    #[test]
    fn v0124_roundtrip_boundaries() {
        let vals = [0u32, 1, 0xFF, 0x100, 0xFFFF, 0x10000, u32::MAX];
        assert_eq!(dec_0124(&enc_0124(&vals), vals.len()).unwrap(), vals);
    }

    #[test]
    fn v0124_zero_uses_no_data_bytes() {
        // 4 zeros → ctrl = [0x00], data = [] (0 bytes), total = 1 byte
        let enc = enc_0124(&[0, 0, 0, 0]);
        assert_eq!(enc, [0x00]);
        assert_eq!(dec_0124(&enc, 4).unwrap(), [0, 0, 0, 0]);
    }

    #[test]
    fn v0124_mid_range_uses_4_bytes() {
        // 0x10000–0xFFFFFF must use 4 bytes (no 3-byte option)
        let vals = [0x10000u32, 0xFF_FFFFu32];
        let enc = enc_0124(&vals);
        // both get tag 3 → ctrl[0] = 0b00_00_11_11 = 0x0F, data = 4+4=8 bytes
        assert_eq!(enc[0], 0x0F);
        assert_eq!(enc.len(), 1 + 8);
        assert_eq!(dec_0124(&enc, 2).unwrap(), vals);
    }

    // ── U32Variant0124 encoded_data_len ──────────────────────────────────────

    #[test]
    fn v0124_data_len() {
        let vals = [0, 1, 255, 256, 65535, 65536, 0xFFFF_FFFFu32];
        let enc = enc_0124(&vals);
        let ctrl_len = vals.len().div_ceil(4);
        assert_eq!(
            encoded_data_len_0124(&enc[..ctrl_len], vals.len()),
            enc.len() - ctrl_len
        );
    }

    // ── U32Variant0124 errors ─────────────────────────────────────────────────

    #[test]
    fn v0124_error_data_truncated() {
        // ctrl says first value has tag 1 (1 byte), but no data follows
        assert!(matches!(
            dec_0124(&[0x01], 1),
            Err(DecodeError::DataTruncated { index: 0 })
        ));
    }
}
