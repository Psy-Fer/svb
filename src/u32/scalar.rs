use crate::error::DecodeError;

// The encode functions below take an `out` slice of at least
// `ctrl_len + 4 * values.len()` bytes; the caller checks this. They panic on a
// shorter slice.

// ── U32Classic (Lemire) ───────────────────────────────────────────────────────
// tag → byte width: tag + 1  (0→1, 1→2, 2→3, 3→4)
// value ranges:  0x00–0xFF→tag0, 0x100–0xFFFF→tag1,
//                0x10000–0xFFFFFF→tag2, 0x1000000–0xFFFFFFFF→tag3

/// Encode `values` into `out` and return the number of bytes written.
pub(super) fn encode_to_slice_classic(values: &[u32], out: &mut [u8]) -> usize {
    let ctrl_len = values.len().div_ceil(4);
    out[..ctrl_len].fill(0);
    encode_tail_classic(values, 0, out, ctrl_len, 0)
}

/// Encode `values[start..]`, continuing an encode a SIMD path has started.
///
/// `start` must be a multiple of 4, and `data_pos` is the number of data bytes
/// already written after the `ctrl_len` control bytes. Control bytes from
/// `start / 4` onwards must be zero. Returns the total bytes written.
pub(super) fn encode_tail_classic(
    values: &[u32],
    start: usize,
    out: &mut [u8],
    ctrl_len: usize,
    mut data_pos: usize,
) -> usize {
    let (ctrl, data) = out.split_at_mut(ctrl_len);
    for (i, &v) in values.iter().enumerate().skip(start) {
        let (tag, count): (u8, usize) = if v <= 0xFF {
            (0, 1)
        } else if v <= 0xFFFF {
            (1, 2)
        } else if v <= 0xFF_FFFF {
            (2, 3)
        } else {
            (3, 4)
        };
        ctrl[i / 4] |= tag << ((i % 4) * 2);
        data[data_pos..data_pos + count].copy_from_slice(&v.to_le_bytes()[..count]);
        data_pos += count;
    }
    ctrl_len + data_pos
}

/// Decode `out.len()` U32Classic values from pre-split `ctrl` and `data` byte
/// slices. Used by the SIMD decode paths for their scalar tail. `first_index`
/// is the index of `out[0]` in the whole output, used in error reports.
pub(super) fn decode_classic_from_raw(
    ctrl: &[u8],
    data: &[u8],
    out: &mut [u32],
    first_index: usize,
) -> Result<(), DecodeError> {
    let mut pos = 0usize;
    for (i, o) in out.iter_mut().enumerate() {
        let tag = (ctrl[i / 4] >> ((i % 4) * 2)) & 3;
        let count = (tag + 1) as usize;
        if pos + count > data.len() {
            return Err(DecodeError::DataTruncated {
                index: first_index + i,
            });
        }
        let mut bytes = [0u8; 4];
        bytes[..count].copy_from_slice(&data[pos..pos + count]);
        *o = u32::from_le_bytes(bytes);
        pos += count;
    }
    Ok(())
}

/// Decode `out.len()` U32Classic values from `data`.
pub(super) fn decode_to_slice_classic(data: &[u8], out: &mut [u32]) -> Result<(), DecodeError> {
    let ctrl_len = check_ctrl_len(data, out.len())?;
    decode_classic_from_raw(&data[..ctrl_len], &data[ctrl_len..], out, 0)
}

/// Return the control stream length for `n` values, or an error if `data` is
/// shorter than that.
fn check_ctrl_len(data: &[u8], n: usize) -> Result<usize, DecodeError> {
    let ctrl_len = n.div_ceil(4);
    if data.len() < ctrl_len {
        return Err(DecodeError::ControlStreamTooShort {
            need: ctrl_len,
            have: data.len(),
        });
    }
    Ok(ctrl_len)
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

/// Encode `values` into `out` and return the number of bytes written.
pub(super) fn encode_to_slice_0124(values: &[u32], out: &mut [u8]) -> usize {
    let ctrl_len = values.len().div_ceil(4);
    out[..ctrl_len].fill(0);
    encode_tail_0124(values, 0, out, ctrl_len, 0)
}

/// Encode `values[start..]`, continuing an encode a SIMD path has started.
/// Same contract as [`encode_tail_classic`].
pub(super) fn encode_tail_0124(
    values: &[u32],
    start: usize,
    out: &mut [u8],
    ctrl_len: usize,
    mut data_pos: usize,
) -> usize {
    let (ctrl, data) = out.split_at_mut(ctrl_len);
    for (i, &v) in values.iter().enumerate().skip(start) {
        let (tag, count): (u8, usize) = if v == 0 {
            (0, 0)
        } else if v <= 0xFF {
            (1, 1)
        } else if v <= 0xFFFF {
            (2, 2)
        } else {
            (3, 4)
        };
        ctrl[i / 4] |= tag << ((i % 4) * 2);
        data[data_pos..data_pos + count].copy_from_slice(&v.to_le_bytes()[..count]);
        data_pos += count;
    }
    ctrl_len + data_pos
}

/// Decode `out.len()` Variant0124 values from pre-split `ctrl` and `data` byte
/// slices. Same contract as [`decode_classic_from_raw`].
pub(super) fn decode_0124_from_raw(
    ctrl: &[u8],
    data: &[u8],
    out: &mut [u32],
    first_index: usize,
) -> Result<(), DecodeError> {
    let mut pos = 0usize;
    for (i, o) in out.iter_mut().enumerate() {
        let tag = (ctrl[i / 4] >> ((i % 4) * 2)) & 3;
        let count = WIDTHS_0124[tag as usize];
        if pos + count > data.len() {
            return Err(DecodeError::DataTruncated {
                index: first_index + i,
            });
        }
        let mut bytes = [0u8; 4];
        bytes[..count].copy_from_slice(&data[pos..pos + count]);
        *o = u32::from_le_bytes(bytes);
        pos += count;
    }
    Ok(())
}

/// Decode `out.len()` Variant0124 values from `data`.
pub(super) fn decode_to_slice_0124(data: &[u8], out: &mut [u32]) -> Result<(), DecodeError> {
    let ctrl_len = check_ctrl_len(data, out.len())?;
    decode_0124_from_raw(&data[..ctrl_len], &data[ctrl_len..], out, 0)
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
    use alloc::{vec, vec::Vec};

    fn enc_classic(v: &[u32]) -> Vec<u8> {
        let mut out = vec![0u8; v.len().div_ceil(4) + 4 * v.len()];
        let len = encode_to_slice_classic(v, &mut out);
        out.truncate(len);
        out
    }
    fn dec_classic(d: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = vec![0u32; n];
        decode_to_slice_classic(d, &mut out)?;
        Ok(out)
    }
    fn enc_0124(v: &[u32]) -> Vec<u8> {
        let mut out = vec![0u8; v.len().div_ceil(4) + 4 * v.len()];
        let len = encode_to_slice_0124(v, &mut out);
        out.truncate(len);
        out
    }
    fn dec_0124(d: &[u8], n: usize) -> Result<Vec<u32>, DecodeError> {
        let mut out = vec![0u32; n];
        decode_to_slice_0124(d, &mut out)?;
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

    #[test]
    fn classic_data_len_tag_lt_3_in_pos3() {
        // Existing test has tag=3 in position 3 of the ctrl byte, where
        // `(b>>6)&0x03 == (b>>6)|0x03 == 3`, masking the & vs | mutation.
        // tags: 0,1,2,0 → ctrl=0x24 → (b>>6)=0x00, so & vs | is detectable.
        let vals = [1u32, 256, 65536, 0xFF];
        let enc = enc_classic(&vals);
        assert_eq!(encoded_data_len_classic(&enc[..1], 4), enc.len() - 1);
    }

    #[test]
    fn classic_data_len_with_tail() {
        // n=6: full=1 (main loop runs) + rem=2 (tail loop runs for j=0 and j=1).
        // rem=1 is insufficient: when j=0 the shift is 0 so >> and << are identical
        // and j*2 == j/2 == 0. rem=2 exercises j=1 where shift=2, making those
        // mutations detectable.
        let vals = [1u32, 256, 65536, 0xFF, 0xFFFF, 65536];
        let enc = enc_classic(&vals);
        let ctrl_len = vals.len().div_ceil(4);
        assert_eq!(
            encoded_data_len_classic(&enc[..ctrl_len], vals.len()),
            enc.len() - ctrl_len
        );
    }

    #[test]
    fn classic_data_len_two_ctrl_bytes() {
        // n=8: full=2 — two complete ctrl bytes, exercising the main loop across
        // multiple iterations with varied tags.
        let vals = [1u32, 256, 65536, 0xFF, 0xFFFF, 0xFF_FFFF, 0xFF_FFFF, 1];
        let enc = enc_classic(&vals);
        let ctrl_len = vals.len().div_ceil(4);
        assert_eq!(
            encoded_data_len_classic(&enc[..ctrl_len], vals.len()),
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

    #[test]
    fn v0124_data_len_sparse_ctrl() {
        // ctrl=0x01: only position-0 has tag=1 (1 byte), rest tag=0 (0 bytes).
        // With j+2 mutation the bit extraction reads a different slot and sums to 0 ≠ 1,
        // catching the `* with +` and `* with /` mutations in the inner j loop.
        let vals = [1u32, 0, 0, 0];
        let enc = enc_0124(&vals);
        assert_eq!(encoded_data_len_0124(&enc[..1], 4), enc.len() - 1);
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
