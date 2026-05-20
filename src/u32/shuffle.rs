// Shuffle tables shared by the SSSE3 and AVX2 decode paths.
// Always compiled on x86_64 to catch compile errors even when no SIMD
// feature is active; dead_code suppressed because use is feature-gated.
#![allow(dead_code)]

// ── Decode table ─────────────────────────────────────────────────────────────
//
// Entry `c` is the 16-byte PSHUFB mask that expands the variable-width data
// bytes for control byte value `c` into 4 fixed-width u32 output slots
// (little-endian).
//
// For each value i (0..4):
//   tag = (c >> (2*i)) & 3,  width = tag + 1
//   output bytes [4*i .. 4*i+4]:
//     first `width` bytes → src+0, src+1, ... (data byte indices)
//     remaining bytes     → 0x80  (PSHUFB zero-fill sentinel)
//
// Both PSHUFB (_mm_shuffle_epi8) and VPSHUFB (_mm256_shuffle_epi8) zero the
// output byte when the mask byte has its high bit set. 0x80 satisfies this,
// so the same table works for both SSE and AVX2.

const fn make_decode() -> [[u8; 16]; 256] {
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
                if b < width {
                    table[ctrl][base + b] = src + b as u8;
                } else {
                    table[ctrl][base + b] = 0x80;
                }
                b += 1;
            }
            src += width as u8;
            i += 1;
        }
        ctrl += 1;
    }
    table
}

pub(super) static TABLE: [[u8; 16]; 256] = make_decode();

// ── Data-length table ─────────────────────────────────────────────────────────
//
// Entry `c` = number of data bytes consumed when decoding 4 u32 values with
// control byte `c`.
//
// data_len(c) = sum of (((c >> (2*i)) & 3) + 1) for i in 0..4

const fn make_data_len() -> [u8; 256] {
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

pub(super) static DATA_LEN: [u8; 256] = make_data_len();
