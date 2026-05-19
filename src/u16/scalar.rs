#[cfg(feature = "std")]
use std::vec::Vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::error::DecodeError;

pub(super) fn encode_into(values: &[u16], out: &mut Vec<u8>) {
    let n = values.len();
    if n == 0 {
        return;
    }
    let ctrl_len = n.div_ceil(8);
    let ctrl_start = out.len();
    out.resize(ctrl_start + ctrl_len, 0u8);

    for (i, &v) in values.iter().enumerate() {
        if v <= 0xFF {
            out.push(v as u8);
        } else {
            out[ctrl_start + i / 8] |= 1 << (i % 8);
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

pub(super) fn decode_into(
    data: &[u8],
    n: usize,
    out: &mut Vec<u16>,
) -> Result<(), DecodeError> {
    if n == 0 {
        return Ok(());
    }
    let ctrl_len = n.div_ceil(8);
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
        let bit = (ctrl[i / 8] >> (i % 8)) & 1;
        if bit == 0 {
            if pos >= data.len() {
                return Err(DecodeError::DataTruncated { index: i });
            }
            out.push(data[pos] as u16);
            pos += 1;
        } else {
            if pos + 2 > data.len() {
                return Err(DecodeError::DataTruncated { index: i });
            }
            out.push(u16::from_le_bytes([data[pos], data[pos + 1]]));
            pos += 2;
        }
    }
    Ok(())
}

/// Number of data bytes consumed by `n` values given their control stream.
/// Each value occupies 1 byte (ctrl bit 0) or 2 bytes (ctrl bit 1), so
/// total = n + popcount(first n bits of ctrl).
pub(super) fn encoded_data_len(ctrl: &[u8], n: usize) -> usize {
    let full = n / 8;
    let rem = n % 8;
    let mut ones: usize = ctrl[..full].iter().map(|b| b.count_ones() as usize).sum();
    if rem > 0 {
        let mask = (1u8 << rem) - 1;
        ones += (ctrl[full] & mask).count_ones() as usize;
    }
    n + ones
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec;

    fn encode(values: &[u16]) -> Vec<u8> {
        let mut out = Vec::new();
        encode_into(values, &mut out);
        out
    }

    fn decode(data: &[u8], n: usize) -> Result<Vec<u16>, DecodeError> {
        let mut out = Vec::new();
        decode_into(data, n, &mut out)?;
        Ok(out)
    }

    // ── spec example (§2.3) ──────────────────────────────────────────────────

    #[test]
    fn spec_example_encode() {
        // values: [5, 300, 0, 1000]
        // ctrl[0]: bit0=0(5), bit1=1(300), bit2=0(0), bit3=1(1000) = 0b00001010 = 0x0A
        // data: 0x05, 0x2C 0x01, 0x00, 0xE8 0x03
        let got = encode(&[5, 300, 0, 1000]);
        assert_eq!(got, [0x0A, 0x05, 0x2C, 0x01, 0x00, 0xE8, 0x03]);
    }

    #[test]
    fn spec_example_decode() {
        let data = [0x0Au8, 0x05, 0x2C, 0x01, 0x00, 0xE8, 0x03];
        assert_eq!(decode(&data, 4).unwrap(), [5, 300, 0, 1000]);
    }

    // ── round-trips ──────────────────────────────────────────────────────────

    #[test]
    fn roundtrip_empty() {
        assert_eq!(decode(&encode(&[]), 0).unwrap(), &[] as &[u16]);
    }

    #[test]
    fn roundtrip_all_one_byte() {
        let vals: Vec<u16> = (0..=255).collect();
        let enc = encode(&vals);
        // ctrl: ceil(256/8) = 32 bytes all zero; data: 256 bytes
        assert_eq!(enc.len(), 32 + 256);
        assert_eq!(decode(&enc, 256).unwrap(), vals);
    }

    #[test]
    fn roundtrip_all_two_byte() {
        let vals: Vec<u16> = (256..=511).collect();
        let enc = encode(&vals);
        // ctrl: 32 bytes all 0xFF; data: 512 bytes
        assert_eq!(enc.len(), 32 + 512);
        assert_eq!(decode(&enc, 256).unwrap(), vals);
    }

    #[test]
    fn roundtrip_boundary_values() {
        let vals = [0u16, 255, 256, u16::MAX];
        assert_eq!(decode(&encode(&vals), 4).unwrap(), vals);
    }

    #[test]
    fn roundtrip_single() {
        for v in [0u16, 1, 255, 256, u16::MAX] {
            assert_eq!(decode(&encode(&[v]), 1).unwrap(), [v]);
        }
    }

    // ── LSB-first bit packing across byte boundary ────────────────────────

    #[test]
    fn ctrl_bit_packing_across_boundary() {
        // 9 values: first 8 pack into ctrl[0], value 9 packs into ctrl[1]
        let mut vals = [0u16; 9];
        vals[8] = 1000; // only this one needs 2 bytes
        let enc = encode(&vals);
        // ctrl[0] = 0x00 (all 1-byte), ctrl[1] = 0x01 (bit0 set for val[8])
        assert_eq!(enc[0], 0x00);
        assert_eq!(enc[1], 0x01);
        assert_eq!(decode(&enc, 9).unwrap(), vals);
    }

    // ── encoded_data_len ─────────────────────────────────────────────────────

    #[test]
    fn data_len_all_one_byte() {
        let ctrl = [0x00u8; 4]; // 32 values, all 1-byte
        assert_eq!(encoded_data_len(&ctrl, 32), 32);
    }

    #[test]
    fn data_len_all_two_byte() {
        let ctrl = [0xFFu8; 4]; // 32 values, all 2-byte
        assert_eq!(encoded_data_len(&ctrl, 32), 64);
    }

    #[test]
    fn data_len_partial_last_byte() {
        // 9 values: ctrl[0]=0xFF (8 two-byte), ctrl[1]=0x01 (1 two-byte, 1 set bit)
        let ctrl = [0xFFu8, 0x01];
        assert_eq!(encoded_data_len(&ctrl, 9), 9 + 9); // all 9 are two-byte
    }

    // ── error cases ──────────────────────────────────────────────────────────

    #[test]
    fn error_ctrl_too_short() {
        // Need ceil(9/8)=2 ctrl bytes, only provide 1
        let err = decode(&[0x00], 9).unwrap_err();
        assert!(matches!(
            err,
            DecodeError::ControlStreamTooShort { need: 2, have: 1 }
        ));
    }

    #[test]
    fn error_data_truncated_one_byte() {
        // ctrl says first value is 1-byte, but no data bytes follow
        let err = decode(&[0x00], 1).unwrap_err();
        assert!(matches!(err, DecodeError::DataTruncated { index: 0 }));
    }

    #[test]
    fn error_data_truncated_two_byte() {
        // ctrl says first value is 2-byte, but only 1 data byte follows
        let err = decode(&[0x01, 0xAB], 1).unwrap_err();
        assert!(matches!(err, DecodeError::DataTruncated { index: 0 }));
    }
}
