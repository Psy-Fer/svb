use crate::error::DecodeError;

// The encode functions below take an `out` slice of at least
// `ctrl_len + W * values.len()` bytes, where W is the widest value (4 for
// Coder1234, 8 for Coder1248); the caller checks this. They panic on a shorter
// slice.

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

// ── U64Coder1234 ──────────────────────────────────────────────────────────────
// tag → byte width: tag + 1  (0→1, 1→2, 2→3, 3→4)
// Same tag/width table as U32Classic but element type is u64.
// Values above u32::MAX are truncated to their low 4 bytes (documented on
// U64Coder1234).

/// Encode `values` into `out` and return the number of bytes written.
pub(super) fn encode_to_slice_1234(values: &[u64], out: &mut [u8]) -> usize {
    let ctrl_len = values.len().div_ceil(4);
    out[..ctrl_len].fill(0);
    encode_tail_1234(values, 0, out, ctrl_len, 0)
}

/// Encode `values[start..]`, continuing an encode a SIMD path has started.
///
/// `start` must be a multiple of 4, and `data_pos` is the number of data bytes
/// already written after the `ctrl_len` control bytes. Control bytes from
/// `start / 4` onwards must be zero. Returns the total bytes written.
pub(super) fn encode_tail_1234(
    values: &[u64],
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
        // Cast to u32 before taking LE bytes so we always get exactly 4 bytes.
        data[data_pos..data_pos + count].copy_from_slice(&(v as u32).to_le_bytes()[..count]);
        data_pos += count;
    }
    ctrl_len + data_pos
}

/// Decode `out.len()` U64Coder1234 values from pre-split `ctrl` and `data`
/// byte slices. Used by the SIMD decode paths for their scalar tail.
/// `first_index` is the index of `out[0]` in the whole output, used in error
/// reports.
/// Returns the number of `data` bytes read.
pub(super) fn decode_1234_from_raw(
    ctrl: &[u8],
    data: &[u8],
    out: &mut [u64],
    first_index: usize,
) -> Result<usize, DecodeError> {
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
        *o = u32::from_le_bytes(bytes) as u64;
        pos += count;
    }
    Ok(pos)
}

/// Decode `out.len()` U64Coder1234 values from `data`.
pub(super) fn decode_to_slice_1234(data: &[u8], out: &mut [u64]) -> Result<usize, DecodeError> {
    let ctrl_len = check_ctrl_len(data, out.len())?;
    Ok(ctrl_len + decode_1234_from_raw(&data[..ctrl_len], &data[ctrl_len..], out, 0)?)
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

/// Encode `values` into `out` and return the number of bytes written.
pub(super) fn encode_to_slice_1248(values: &[u64], out: &mut [u8]) -> usize {
    let ctrl_len = values.len().div_ceil(4);
    out[..ctrl_len].fill(0);
    encode_tail_1248(values, 0, out, ctrl_len, 0)
}

/// Encode `values[start..]`, continuing an encode a SIMD path has started.
/// Same contract as [`encode_tail_1234`].
pub(super) fn encode_tail_1248(
    values: &[u64],
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
        } else if v <= 0xFFFF_FFFF {
            (2, 4)
        } else {
            (3, 8)
        };
        ctrl[i / 4] |= tag << ((i % 4) * 2);
        data[data_pos..data_pos + count].copy_from_slice(&v.to_le_bytes()[..count]);
        data_pos += count;
    }
    ctrl_len + data_pos
}

/// Decode `out.len()` U64Coder1248 values from pre-split `ctrl` and `data`
/// byte slices. Same contract as [`decode_1234_from_raw`].
pub(super) fn decode_1248_from_raw(
    ctrl: &[u8],
    data: &[u8],
    out: &mut [u64],
    first_index: usize,
) -> Result<usize, DecodeError> {
    let mut pos = 0usize;
    for (i, o) in out.iter_mut().enumerate() {
        let tag = (ctrl[i / 4] >> ((i % 4) * 2)) & 3;
        let count = WIDTHS_1248[tag as usize];
        if pos + count > data.len() {
            return Err(DecodeError::DataTruncated {
                index: first_index + i,
            });
        }
        let mut bytes = [0u8; 8];
        bytes[..count].copy_from_slice(&data[pos..pos + count]);
        *o = u64::from_le_bytes(bytes);
        pos += count;
    }
    Ok(pos)
}

/// Decode `out.len()` U64Coder1248 values from `data`.
pub(super) fn decode_to_slice_1248(data: &[u8], out: &mut [u64]) -> Result<usize, DecodeError> {
    let ctrl_len = check_ctrl_len(data, out.len())?;
    Ok(ctrl_len + decode_1248_from_raw(&data[..ctrl_len], &data[ctrl_len..], out, 0)?)
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
    use alloc::{vec, vec::Vec};

    fn enc_1234(v: &[u64]) -> Vec<u8> {
        let mut out = vec![0u8; v.len().div_ceil(4) + 4 * v.len()];
        let len = encode_to_slice_1234(v, &mut out);
        out.truncate(len);
        out
    }
    fn dec_1234(d: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = vec![0u64; n];
        decode_to_slice_1234(d, &mut out)?;
        Ok(out)
    }
    fn enc_1248(v: &[u64]) -> Vec<u8> {
        let mut out = vec![0u8; v.len().div_ceil(4) + 8 * v.len()];
        let len = encode_to_slice_1248(v, &mut out);
        out.truncate(len);
        out
    }
    fn dec_1248(d: &[u8], n: usize) -> Result<Vec<u64>, DecodeError> {
        let mut out = vec![0u64; n];
        decode_to_slice_1248(d, &mut out)?;
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

    #[test]
    fn coder1234_data_len_full_loop() {
        // Existing n=3 test has full=0, leaving the main loop (line 136-138) dead.
        // n=5: full=1 (main loop runs once) + rem=1 (tail runs once).
        // tags: 0,1,2,0,1 → ctrl byte has non-max tags, making & vs | detectable.
        let vals = [1u64, 256, 65536, 0xFF, 0xFFFF];
        let enc = enc_1234(&vals);
        let ctrl_len = vals.len().div_ceil(4);
        assert_eq!(
            encoded_data_len_1234(&enc[..ctrl_len], vals.len()),
            enc.len() - ctrl_len
        );
    }

    #[test]
    fn coder1234_data_len_nonzero_pos3_tag() {
        // The previous test has tag=0 in position 3 of the ctrl byte, so mutations
        // on the `(b>>6)` term (col 74) and the final `+` (col 68) are transparent
        // (subtracting or shifting 0 gives the same result). tag=2 in position 3
        // makes both mutations detectable.
        // tags: 0,0,0,2,0 → ctrl[0]=0x80
        let vals = [1u64, 1, 1, 65537, 1];
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

    #[test]
    fn coder1248_data_len_full_loop() {
        // Existing n=2 test has full=0, so the inner j-loop over a complete ctrl byte
        // never runs. n=5: full=1 (inner loop runs for all 4 slots) + rem=1.
        // 1→tag0(1B), 256→tag1(2B), 65536→tag2(4B), 0xFFFF_FFFF→tag2(4B), 1→tag0(1B)
        let vals = [1u64, 256, 65536, 0xFFFF_FFFF, 1];
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
