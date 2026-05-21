# VBZ Pipeline

The VBZ pipeline compresses 16-bit ADC signal data as used in Oxford Nanopore POD5 files. It chains four stages:

```
raw i16 samples
  →  delta encode    (1st-order differences; small values dominate)
  →  zigzag encode   (signed i16 → unsigned u16; small |values| → small codes)
  →  SVB16 encode    (StreamVByte-16: 1-bit control stream)
  →  zstd            (outer entropy coding; NOT part of this crate)
```

The `svb` crate handles stages 1–3. The outer zstd layer is left to the caller.

## High-level API

```rust
use svb::{encode_vbz, decode_vbz};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];

// Encode: i16 → delta → zigzag → SVB16 bytes
let encoded = encode_vbz(&samples);

// Decode: SVB16 bytes → zigzag → delta → i16
let decoded = decode_vbz(&encoded, samples.len()).unwrap();
assert_eq!(decoded, samples);
```

## Low-level / into variants

For zero-allocation usage or building larger buffers:

```rust
use svb::{encode_vbz_into, decode_vbz_into};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];
let mut buf: Vec<u8> = Vec::new();
encode_vbz_into(&samples, &mut buf);

let mut out: Vec<i16> = Vec::new();
decode_vbz_into(&buf, samples.len(), &mut out).unwrap();
assert_eq!(out, samples);
```
