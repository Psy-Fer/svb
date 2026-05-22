# Encoding Guide

This page explains how StreamVByte, delta, and zigzag work, when to use each, and how they compose. For the API itself see [Delta and Zigzag](transforms.md), [Codec Variants](codecs/index.md), and the [API reference](https://docs.rs/svb).

---

## Fixed-width integers and why they waste space

Every integer type has a fixed storage width. A `u32` always occupies 4 bytes, regardless of its value:

```
value         memory (big-endian for clarity)   bytes used
──────────────────────────────────────────────────────────
         1    00 00 00 01                        4 (3 wasted)
       300    00 00 01 2C                        4 (2 wasted)
    75 000    00 01 24 F8                        4 (1 wasted)
16 000 000    00 F4 24 00                        4 (1 wasted)
 4 294 967 295  FF FF FF FF                      4 (none wasted)
```

The high-order bytes are zero whenever the value is small. Those zeros carry no information, yet they occupy the same storage as any other byte. For a `u32` array where most values are below 256, three quarters of the storage is zero-padding.

This matters in practice because integer arrays in real applications are rarely uniformly distributed across the full type range. File offsets, timestamps, sensor readings, and index lists all tend to cluster at small magnitudes relative to the maximum the type can hold. An array of one million `u32` values representing document word frequencies, for example, might use only the bottom 12 bits of each element, leaving 20 bits per value (over 2 MB per million elements) as wasted zeros.

Variable-byte encoding solves this by storing only the bytes that carry information and recording how many bytes each value used. StreamVByte is a specific variable-byte scheme designed to make that decoding fast with SIMD.

---

## StreamVByte

Most integers in real data are small. Fixed-width encoding wastes bytes on the high-order zeros; StreamVByte stores only the bytes that carry information.

The key design decision is *where* to put the width metadata. Naive variable-byte schemes (such as SQLite's `varint` or Protocol Buffers' `LEN`) interleave a length prefix with each value, so the decoder must branch on every element, and SIMD cannot help. StreamVByte separates the metadata into a **control stream** and the integer bytes into a **data stream**:

```
┌──────────────────────────────────┐
│ control stream                   │
│  tag tag tag tag tag tag tag tag  │  ← 2 bits per value (u32), 1 bit (u16)
└──────────────────────────────────┘
┌──────────────────────────────────┐
│ data stream                      │
│  [value 0 bytes][value 1 bytes]… │  ← tightly packed, no separators
└──────────────────────────────────┘
```

Because all the widths for a block of values live in one or two control bytes, a SIMD decoder can read them all at once, look up a pre-built shuffle table, and unpack 4–8 values in a single `pshufb` instruction. The data stream is a plain byte array with no branch points.

### A concrete example

Four `u32` values encoded with `U32Classic` (1/2/3/4-byte widths):

```
values:   [   1,   300,  75000,   5 ]
widths:   [ 1 B,   2 B,    3 B, 1 B ]  ← determined by value magnitude
tags:     [  00,    01,     10,  00 ]  ← 2-bit tag per value

control byte:  0b_00_10_01_00  (4 tags packed LSB-first)

data bytes:    01 | 2C 01 | F8 24 01 | 05
               └1┘  └─300─┘  └─75000─┘  └5┘
```

The full encoded output is **5 bytes** (1 control + 4 data) for four 32-bit values that would require 16 bytes in fixed-width form.

### Codec variant selection

The five variants differ in which byte widths they support and what element type they encode:

| Variant | Element | Byte widths | Best for |
|---|---|---|---|
| `Svb16` | `u16` | 1 / 2 | 16-bit data; values mostly ≤ 255 |
| `U32Classic` | `u32` | 1 / 2 / 3 / 4 | General-purpose u32; compatible with Lemire C library |
| `U32Variant0124` | `u32` | 0 / 1 / 2 / 4 | Sparse data with many exact zeros (0 bytes stored) |
| `U64Coder1234` | `u64` | 1 / 2 / 3 / 4 | u64 values known to fit in u32 |
| `U64Coder1248` | `u64` | 1 / 2 / 4 / 8 | Full u64 range |

`U32Variant0124` skips the 3-byte width and adds a 0-byte width: a zero value stores no data bytes at all, only its tag. This is a significant win for sparse arrays where many values are exactly zero.

---

## Delta encoding

Delta encoding replaces each value with its *difference* from the previous one:

```
original:  [ 1000,  1003,  1007,  1004,  1010 ]
                ↘      ↘      ↘      ↘
deltas:    [ 1000,    +3,    +4,    -3,    +6  ]
```

The first delta is the first value itself (difference from an implicit zero, or from a caller-supplied carry). Every subsequent delta is `values[i] - values[i-1]`.

For sequences where adjacent values are close together (sorted integers, time-series measurements, sensor readings), the deltas are much smaller than the raw values. Smaller values encode to fewer bytes in any variable-byte scheme.

### When delta helps

| Data pattern | Example | Delta effect |
|---|---|---|
| Sorted / monotone | File offsets, timestamps | Deltas are small positive integers |
| Slowly drifting | Temperature readings | Deltas cluster near zero |
| Periodic / oscillating | ADC signal samples | Deltas small if bandwidth is limited |
| Uniformly random | Hash values | No benefit; deltas are as large as the values |

Delta encoding is a **lossless, reversible transform**. Decoding is a prefix sum: `values[i] = deltas[0] + deltas[1] + … + deltas[i]`. The serial dependency between elements is the main cost; see [Performance](performance.md) for how the SIMD prefix-sum implementation handles it.

### Signed vs unsigned

`delta` in `svb` is implemented for `i16`, `i32`, `i64`, `u32`, and `u64`. For non-monotone data (where values can decrease), use a **signed** type, as the deltas will be negative and a signed representation preserves that. For guaranteed non-decreasing sequences (file offsets, sorted timestamps), an **unsigned** type is fine and avoids the overhead of zigzag.

---

## Zigzag encoding

Variable-byte codecs assign shorter encodings to smaller *non-negative* integers. A signed delta of −1 would be stored as `0xFFFFFFFF` (4 bytes) in a `u32` codec, with no compression at all.

Zigzag solves this by remapping signed integers to unsigned so that small absolute values map to small codes:

```
signed →  unsigned
     0 →  0
    -1 →  1
    +1 →  2
    -2 →  3
    +2 →  4
    -3 →  5
    +3 →  6
   ...
```

The formula is `(n << 1) ^ (n >> (bits - 1))`, two bitwise ops with no branches. Decoding is `(n >> 1) ^ -(n & 1)`. Both directions are branchless and LLVM auto-vectorizes them.

After zigzag, a signed delta of −1 becomes the unsigned value 1, which encodes in a single byte. A delta of +127 becomes 254, still a single byte. Only values with absolute magnitude above 127 spill into a second byte.

---

## Composing the three

Delta → zigzag → StreamVByte is a standard pipeline for compressing integer sequences that are slowly varying or oscillating. Each stage does one job:

```
raw values
  │
  ▼  delta encode
differences (signed, small magnitude)
  │
  ▼  zigzag encode
differences (unsigned, small magnitude)
  │
  ▼  StreamVByte encode
compact byte stream
```

### Worked example

Five `i16` ADC-style samples:

```
stage         values
───────────────────────────────────────────────
raw           [ 1000,  1003,  1007,  1004,  1010 ]
after delta   [ 1000,     3,     4,    -3,     6 ]
after zigzag  [ 2000,     6,     8,     5,    12 ]
after SVB16   2 ctrl bytes + 6 data bytes  (vs 10 raw bytes)
```

The first value (1000) stays large because it is the absolute anchor. The subsequent values (the deltas) all fit in a single byte after zigzag. In practice, for signals with small bandwidth relative to their absolute level, the per-value cost quickly drops to 1 byte once the anchor is amortised over the chunk.

### Choosing what to compose

| Data | Recipe |
|---|---|
| Sorted unsigned integers | delta → `U32Classic` or `U64Coder1248` |
| Non-monotone integers | delta → zigzag → `U32Classic` |
| 16-bit oscillating signal | delta → zigzag → `Svb16` (= the VBZ pipeline) |
| Sparse data with many zeros | `U32Variant0124` alone, or delta first if it helps |
| Already-small unsigned values | `U32Classic` or `Svb16` directly |

---

## Further reading

- **StreamVByte paper**: Lemire, Kurz, Rupp. *Stream VByte: Faster Byte-Oriented Integer Compression* (2017). [arxiv.org/abs/1709.08990](https://arxiv.org/abs/1709.08990)
- **Lemire's blog post** introducing StreamVByte with benchmarks: [lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/](https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/)
- **Zigzag encoding** as used in Protocol Buffers (good concise reference): [protobuf.dev/programming-guides/encoding/#signed-ints](https://protobuf.dev/programming-guides/encoding/#signed-ints)
