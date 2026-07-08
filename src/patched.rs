//! PFOR-style patched/exception encoding as a composable layer over `u16` values.
//!
//! Values that fit in a byte (`<= u8::MAX`) are stored as literal bytes, in
//! original stream order. Values that don't fit are pulled out as
//! exceptions: their positions and residual values (`value - 256`) are
//! recorded separately and [`crate::u32::U32Classic`]-encoded. This pays off
//! when exceptions are rare — e.g. the tail of a zigzag-delta signal stream,
//! where most residuals are small but occasional spikes need the full range.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::DecodeError;
use crate::u32::U32Classic;

const THRESHOLD: u16 = u8::MAX as u16;

fn too_short(need: usize, have: usize) -> DecodeError {
    DecodeError::ControlStreamTooShort { need, have }
}

/// Encode `values`, appending the patched/exception representation to `out`.
///
/// # Examples
///
/// ```
/// # use svb::patched;
/// let mut out = Vec::new();
/// patched::encode_into(&[1u16, 300, 2], &mut out);
/// let mut decoded = Vec::new();
/// patched::decode_into(&out, 3, &mut decoded).unwrap();
/// assert_eq!(decoded, [1u16, 300, 2]);
/// ```
pub fn encode_into(values: &[u16], out: &mut Vec<u8>) {
    let mut ex_pos: Vec<u32> = Vec::new();
    let mut ex_val: Vec<u32> = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v > THRESHOLD {
            ex_pos.push(i as u32);
            ex_val.push((v - THRESHOLD - 1) as u32);
        }
    }

    let nex = ex_pos.len() as u32;
    out.extend_from_slice(&nex.to_le_bytes());

    if nex > 1 {
        let mut pos_delta = Vec::with_capacity(ex_pos.len());
        pos_delta.push(ex_pos[0]);
        for w in ex_pos.windows(2) {
            pos_delta.push(w[1] - w[0] - 1);
        }

        let mut pos_bytes = Vec::new();
        U32Classic.encode_into(&pos_delta, &mut pos_bytes);
        out.extend_from_slice(&(pos_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&pos_bytes);

        let mut val_bytes = Vec::new();
        U32Classic.encode_into(&ex_val, &mut val_bytes);
        out.extend_from_slice(&(val_bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&val_bytes);
    } else if nex == 1 {
        out.extend_from_slice(&ex_pos[0].to_le_bytes());
        out.extend_from_slice(&ex_val[0].to_le_bytes());
    }

    let mut j = 0;
    for (i, &v) in values.iter().enumerate() {
        if j < ex_pos.len() && i as u32 == ex_pos[j] {
            j += 1;
        } else {
            out.push(v as u8);
        }
    }
}

/// Decode exactly `n` values from the start of `data`, appending them to `out`.
///
/// Returns the number of bytes consumed from `data`. `n` must equal the
/// number of values that were originally encoded, same convention as
/// [`crate::u32::U32Classic::decode`].
///
/// # Examples
///
/// ```
/// # use svb::patched;
/// let mut enc = Vec::new();
/// patched::encode_into(&[1u16, 300, 2], &mut enc);
/// let mut out = Vec::new();
/// let consumed = patched::decode_into(&enc, 3, &mut out).unwrap();
/// assert_eq!(consumed, enc.len());
/// assert_eq!(out, [1u16, 300, 2]);
/// ```
pub fn decode_into(data: &[u8], n: usize, out: &mut Vec<u16>) -> Result<usize, DecodeError> {
    if data.len() < 4 {
        return Err(too_short(4, data.len()));
    }
    let nex = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let mut offset = 4;

    let mut ex_pos: Vec<u32> = Vec::new();
    let mut ex_val: Vec<u32> = Vec::new();

    if nex > 1 {
        let nex = nex as usize;

        if data.len() < offset + 4 {
            return Err(too_short(offset + 4, data.len()));
        }
        let nex_pos_press = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        if data.len() < offset + nex_pos_press {
            return Err(too_short(offset + nex_pos_press, data.len()));
        }
        let mut pos_delta = U32Classic.decode(&data[offset..offset + nex_pos_press], nex)?;
        offset += nex_pos_press;

        for i in 1..pos_delta.len() {
            let prev = pos_delta[i - 1];
            pos_delta[i] = pos_delta[i].wrapping_add(prev).wrapping_add(1);
        }
        ex_pos = pos_delta;

        if data.len() < offset + 4 {
            return Err(too_short(offset + 4, data.len()));
        }
        let nex_press = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        offset += 4;
        if data.len() < offset + nex_press {
            return Err(too_short(offset + nex_press, data.len()));
        }
        ex_val = U32Classic.decode(&data[offset..offset + nex_press], nex)?;
        offset += nex_press;
    } else if nex == 1 {
        if data.len() < offset + 8 {
            return Err(too_short(offset + 8, data.len()));
        }
        let pos = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        let val = u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        offset += 8;
        ex_pos.push(pos);
        ex_val.push(val);
    }

    let nex = nex as usize;
    let n_literal = n.saturating_sub(nex);
    if data.len() < offset + n_literal {
        return Err(too_short(offset + n_literal, data.len()));
    }

    out.reserve(n);
    let mut j = 0;
    let mut lit = 0;
    for i in 0..n {
        if j < ex_pos.len() && ex_pos[j] as usize == i {
            out.push((ex_val[j] as u16).wrapping_add(THRESHOLD + 1));
            j += 1;
        } else {
            out.push(data[offset + lit] as u16);
            lit += 1;
        }
    }
    offset += lit;

    Ok(offset)
}
