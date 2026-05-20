#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::error::DecodeError;

// ── shared helpers ────────────────────────────────────────────────────────────

#[inline]
fn get_tag(ctrl: &[u8], i: usize) -> u8 {
    (ctrl[i / 4] >> ((i % 4) * 2)) & 0x03
}

// ── U64Coder1234 ──────────────────────────────────────────────────────────────
// tag → byte width: tag + 1  (0→1, 1→2, 2→3, 3→4)
// Same tag/width table as U32Classic but element type is u64.
// Precondition: all values must fit in u32 (≤ 0xFFFF_FFFF).
// Violating this is a precondition error; debug_assert fires in debug builds,
// release builds silently truncate to the low 4 bytes.

pub(super) fn encode_into_1234(values: &[u64], out: &mut Vec<u8>) {
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
        // Cast to u32 before taking LE bytes so we always get exactly 4 bytes.
        out.extend_from_slice(&(v as u32).to_le_bytes()[..count]);
    }
}

/// Decode `n` U64Coder1234 values from pre-split `ctrl` and `data` byte slices.
///
/// Used by SIMD decode paths to handle the scalar tail.
pub(super) fn decode_1234_from_raw(
    ctrl: &[u8],
    data: &[u8],
    n: usize,
    out: &mut Vec<u64>,
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
        out.push(u32::from_le_bytes(bytes) as u64);
        pos += count;
    }
    Ok(())
}

/// Decode `n` U64Coder1248 values from pre-split `ctrl` and `data` byte slices.
///
/// Used by SIMD decode paths to handle the scalar tail.
pub(super) fn decode_1248_from_raw(
    ctrl: &[u8],
    data: &[u8],
    n: usize,
    out: &mut Vec<u64>,
) -> Result<(), DecodeError> {
    out.reserve(n);
    let mut pos = 0usize;
    for i in 0..n {
        let tag = (ctrl[i / 4] >> ((i % 4) * 2)) & 3;
        let count = WIDTHS_1248[tag as usize];
        if pos + count > data.len() {
            return Err(DecodeError::DataTruncated { index: i });
        }
        let mut bytes = [0u8; 8];
        bytes[..count].copy_from_slice(&data[pos..pos + count]);
        out.push(u64::from_le_bytes(bytes));
        pos += count;
    }
    Ok(())
}

pub(super) fn decode_into_1234(
    data: &[u8],
    n: usize,
    out: &mut Vec<u64>,
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
        out.push(u32::from_le_bytes(bytes) as u64);
        pos += count;
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn encoded_data_len_1234(ctrl: &[u8], n: usize) -> usize {
    // Identical formula to U32Classic: data_len = n + sum(tag_i)
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

// ── U64Coder1248 ──────────────────────────────────────────────────────────────
// tag → byte width: [1, 2, 4, 8]
// value ranges: 0x00–0xFF→tag0, 0x100–0xFFFF→tag1,
//               0x10000–0xFFFFFFFF→tag2 (no 3-byte option),
//               0x100000000–u64::MAX→tag3

const WIDTHS_1248: [usize; 4] = [1, 2, 4, 8];

pub(super) fn encode_into_1248(values: &[u64], out: &mut Vec<u8>) {
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
        } else if v <= 0xFFFF_FFFF {
            (2, 4)
        } else {
            (3, 8)
        };
        out[ctrl_start + i / 4] |= tag << ((i % 4) * 2);
        out.extend_from_slice(&v.to_le_bytes()[..count]);
    }
}

pub(super) fn decode_into_1248(
    data: &[u8],
    n: usize,
    out: &mut Vec<u64>,
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
        let count = WIDTHS_1248[tag as usize];
        if pos + count > data.len() {
            return Err(DecodeError::DataTruncated { index: i });
        }
        let mut bytes = [0u8; 8];
        bytes[..count].copy_from_slice(&data[pos..pos + count]);
        out.push(u64::from_le_bytes(bytes));
        pos += count;
    }
    Ok(())
}

#[allow(dead_code)]
pub(super) fn encoded_data_len_1248(ctrl: &[u8], n: usize) -> usize {
    let mut sum = 0usize;
    let full = n / 4;
    let rem = n % 4;
    for &b in &ctrl[..full] {
        for j in 0..4 {
            sum += WIDTHS_1248[((b >> (j * 2)) & 0x03) as usize];
        }
    }
    for j in 0..rem {
        sum += WIDTHS_1248[((ctrl[full] >> (j * 2)) & 0x03) as usize];
    }
    sum
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn enc_1234(v: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_into_1234(v, &mut out);
        out
    }
    fn dec_1234(d: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::new();
        decode_into_1234(d, n, &mut out)?;
        Ok(out)
    }
    fn enc_1248(v: &[u64]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_into_1248(v, &mut out);
        out
    }
    fn dec_1248(d: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = Vec::new();
        decode_into_1248(d, n, &mut out)?;
        Ok(out)
    }

    // ── U64Coder1234 spec example ─────────────────────────────────────────────

    #[test]
    fn coder1234_spec_example_encode() {
        let got = enc_1234(&[0, 0xFF_FFFF, 0xFFFF_FFFF]);
        assert_eq!(got, [0x38, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn coder1234_spec_example_decode() {
        let data = [0x38u8, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(dec_1234(&data, 3).unwrap(), [0, 0xFF_FFFF, 0xFFFF_FFFF]);
    }

    // ── U64Coder1234 round-trips ──────────────────────────────────────────────

    #[test]
    fn coder1234_roundtrip_empty() {
        assert_eq!(dec_1234(&enc_1234(&[]), 0).unwrap(), &[] as &[u64]);
    }

    #[test]
    fn coder1234_roundtrip_boundaries() {
        let vals = [
            0u64,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0xFF_FFFF,
            0x100_0000,
            u32::MAX as u64,
        ];
        assert_eq!(dec_1234(&enc_1234(&vals), vals.len()).unwrap(), vals);
    }

    #[test]
    fn coder1234_data_len() {
        let vals = [0u64, 0xFF_FFFF, 0xFFFF_FFFF];
        let enc = enc_1234(&vals);
        let ctrl_len = vals.len().div_ceil(4);
        assert_eq!(
            encoded_data_len_1234(&enc[..ctrl_len], vals.len()),
            enc.len() - ctrl_len
        );
    }

    // ── U64Coder1234 truncation behaviour ────────────────────────────────────

    #[test]
    fn coder1234_truncates_large_values() {
        // Values > u32::MAX are truncated to their low 32 bits — defined behaviour.
        let large = 0x1_DEAD_BEEFu64; // low 32 bits = 0xDEAD_BEEF
        let enc = enc_1234(&[large]);
        // Decoded value should be 0xDEAD_BEEF (low 32 bits), not the original.
        assert_eq!(dec_1234(&enc, 1).unwrap(), [0xDEAD_BEEFu64]);
    }

    // ── U64Coder1234 errors ───────────────────────────────────────────────────

    #[test]
    fn coder1234_error_ctrl_too_short() {
        assert!(matches!(
            dec_1234(&[0x00], 5),
            Err(DecodeError::ControlStreamTooShort { need: 2, have: 1 })
        ));
    }

    #[test]
    fn coder1234_error_data_truncated() {
        // tag=1 means 2 bytes needed, but only ctrl byte present
        assert!(matches!(
            dec_1234(&[0x01], 1),
            Err(DecodeError::DataTruncated { index: 0 })
        ));
    }

    // ── U64Coder1248 spec example ─────────────────────────────────────────────

    #[test]
    fn coder1248_spec_example_encode() {
        let got = enc_1248(&[0u64, 0x1_0000_0000u64]);
        assert_eq!(
            got,
            [0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]
        );
    }

    #[test]
    fn coder1248_spec_example_decode() {
        let data = [0x0Cu8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00];
        assert_eq!(dec_1248(&data, 2).unwrap(), [0u64, 0x1_0000_0000u64]);
    }

    // ── U64Coder1248 round-trips ──────────────────────────────────────────────

    #[test]
    fn coder1248_roundtrip_empty() {
        assert_eq!(dec_1248(&enc_1248(&[]), 0).unwrap(), &[] as &[u64]);
    }

    #[test]
    fn coder1248_roundtrip_boundaries() {
        let vals = [
            0u64,
            0xFF,
            0x100,
            0xFFFF,
            0x10000,
            0xFFFF_FFFF,
            0x1_0000_0000,
            u64::MAX,
        ];
        assert_eq!(dec_1248(&enc_1248(&vals), vals.len()).unwrap(), vals);
    }

    #[test]
    fn coder1248_mid_range_uses_4_bytes() {
        // 0x10000–0xFFFFFF: no 3-byte option, must use 4 bytes
        let vals = [0x10000u64, 0xFF_FFFFu64];
        let enc = enc_1248(&vals);
        // both get tag 2 → ctrl[0] = 0b00_00_10_10 = 0x0A, data = 4+4 bytes
        assert_eq!(enc[0], 0x0A);
        assert_eq!(enc.len(), 1 + 8);
        assert_eq!(dec_1248(&enc, 2).unwrap(), vals);
    }

    #[test]
    fn coder1248_data_len() {
        let vals = [0u64, 0x1_0000_0000u64];
        let enc = enc_1248(&vals);
        let ctrl_len = vals.len().div_ceil(4);
        assert_eq!(
            encoded_data_len_1248(&enc[..ctrl_len], vals.len()),
            enc.len() - ctrl_len
        );
    }

    // ── U64Coder1248 errors ───────────────────────────────────────────────────

    #[test]
    fn coder1248_error_ctrl_too_short() {
        assert!(matches!(
            dec_1248(&[0x00], 5),
            Err(DecodeError::ControlStreamTooShort { need: 2, have: 1 })
        ));
    }

    #[test]
    fn coder1248_error_data_truncated() {
        // tag=3 means 8 bytes needed, but only ctrl byte present
        assert!(matches!(
            dec_1248(&[0x0C], 1),
            Err(DecodeError::DataTruncated { index: 0 })
        ));
    }
}
