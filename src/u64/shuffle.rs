// Shuffle tables shared by the SSSE3 and AVX2 decode paths.
// Always compiled on x86_64 to catch compile errors even when no SIMD
// feature is active; dead_code suppressed because use is feature-gated.
#![allow(dead_code)]

// ── U64Coder1234 decode table ─────────────────────────────────────────────────
//
// Entry `c` is the 16-byte PSHUFB mask that expands the compact data bytes for
// ctrl byte `c` into 4 × u32 (little-endian) in a 128-bit register.
// Tag widths: tag+1 (0→1, 1→2, 2→3, 3→4). Identical structure to U32Classic.
// After PSHUFB, each u32 slot is zero-extended to u64 before output.

const fn make_decode_1234() -> [[u8; 16]; 256] {
    let mut table = [[0u8; 16]; 256];
    let mut ctrl = 0usize;
    while ctrl < 256 {
        let mut src = 0u8;
        let mut i = 0usize;
        while i < 4 {
            let tag = (ctrl >> (2 * i)) & 3;
            let width = tag + 1;
            let base = 4 * i;
            let mut b = 0usize;
            while b < 4 {
                table[ctrl][base + b] = if b < width { src + b as u8 } else { 0x80 };
                b += 1;
            }
            src += width as u8;
            i += 1;
        }
        ctrl += 1;
    }
    table
}

pub(super) static TABLE_1234: [[u8; 16]; 256] = make_decode_1234();

const fn make_data_len_1234() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut ctrl = 0usize;
    while ctrl < 256 {
        let mut sum = 0u8;
        let mut i = 0usize;
        while i < 4 {
            sum += ((ctrl >> (2 * i)) & 3) as u8 + 1;
            i += 1;
        }
        table[ctrl] = sum;
        ctrl += 1;
    }
    table
}

pub(super) static DATA_LEN_1234: [u8; 256] = make_data_len_1234();

// ── U64Coder1248 pair decode table ────────────────────────────────────────────
//
// Tag widths: [1, 2, 4, 8]. A value can be up to 8 bytes, so we process values
// in pairs: 2 values × 8 output bytes = one 16-byte register.
//
// The table is indexed by a 4-bit key:
//   bits 1:0 = tag for the first value in the pair (val0)
//   bits 3:2 = tag for the second value in the pair (val1)
//
// Entry is a 16-byte PSHUFB mask that shuffles compact data bytes into two
// 8-byte u64 output slots (little-endian). The high bit (0x80) produces zero.

const fn make_decode_1248_pair() -> [[u8; 16]; 16] {
    const WIDTHS: [usize; 4] = [1, 2, 4, 8];
    let mut table = [[0u8; 16]; 16];
    let mut key = 0usize;
    while key < 16 {
        let tag0 = key & 3;
        let tag1 = (key >> 2) & 3;
        let w0 = WIDTHS[tag0];
        let w1 = WIDTHS[tag1];
        // val0 occupies output bytes 0..8
        let mut b = 0usize;
        while b < 8 {
            table[key][b] = if b < w0 { b as u8 } else { 0x80 };
            b += 1;
        }
        // val1 occupies output bytes 8..16; its source bytes start at w0
        let src1 = w0 as u8;
        b = 0;
        while b < 8 {
            table[key][8 + b] = if b < w1 { src1 + b as u8 } else { 0x80 };
            b += 1;
        }
        key += 1;
    }
    table
}

pub(super) static TABLE_1248_PAIR: [[u8; 16]; 16] = make_decode_1248_pair();

const fn make_data_len_1248_pair() -> [u8; 16] {
    const WIDTHS: [u8; 4] = [1, 2, 4, 8];
    let mut table = [0u8; 16];
    let mut key = 0usize;
    while key < 16 {
        table[key] = WIDTHS[key & 3] + WIDTHS[(key >> 2) & 3];
        key += 1;
    }
    table
}

pub(super) static DATA_LEN_1248_PAIR: [u8; 16] = make_data_len_1248_pair();
