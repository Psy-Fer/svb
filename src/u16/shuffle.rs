// Shuffle table shared by the SSSE3 and NEON decode paths.
//
// Entry `c` is the 16-byte PSHUFB / vqtbl1q_u8 mask that expands the
// variable-width data bytes for control byte value `c` into 8 fixed-width
// u16 output slots (little-endian).
//
// Bit k of `c`: 0 = 1-byte value, 1 = 2-byte value.
//   output byte 2k   → index of value's low data byte in the input chunk
//   output byte 2k+1 → index of high byte (2-byte) or 0x80 (zero fill, 1-byte)
//
// Both PSHUFB and vqtbl1q_u8 zero the output byte when the mask byte has
// its high bit set (≥ 16 and ≥ 128 respectively). 0x80 satisfies both, so
// the same table works for SSE/NEON without modification.

const fn make() -> [[u8; 16]; 256] {
    let mut table = [[0u8; 16]; 256];
    let mut ctrl = 0usize;
    while ctrl < 256 {
        let mut src = 0u8;
        let mut i = 0usize;
        while i < 8 {
            let two_byte = (ctrl >> i) & 1 != 0;
            table[ctrl][2 * i] = src;
            if two_byte {
                table[ctrl][2 * i + 1] = src + 1;
                src += 2;
            } else {
                table[ctrl][2 * i + 1] = 0x80;
                src += 1;
            }
            i += 1;
        }
        ctrl += 1;
    }
    table
}

pub(super) static TABLE: [[u8; 16]; 256] = make();
