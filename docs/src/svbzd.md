# SVB-ZD Pipeline

The SVB-ZD pipeline compresses 16-bit signal data into a compact byte stream. It is wire-compatible with the [slow5lib](https://github.com/hasindu2008/slow5lib) `SLOW5_COMPRESS_SVB_ZD` format used in BLOW5 files. The pipeline chains three stages:

```
raw i16 samples
  →  widen to i32
  →  fused zigzag-delta (delta of differences → zigzag32 to make values unsigned)
  →  U32Classic encode  (2-bit control stream, 1–4 bytes per value)
  →  zstd               (outer entropy coding; NOT part of this crate)
```

The key difference from VBZ is the element width: SVB-ZD widens i16 → i32 before the zigzag-delta step, so it uses the U32Classic codec rather than SVB16. This costs one extra bit of tag width but removes SVB16's 2-byte cap: values that overflow i16 after delta (e.g. baseline resets) are encoded correctly without truncation.

## High-level API

```rust
use svb::{encode_svbzd, decode_svbzd};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];

// Encode: i16 → widen → zigzag-delta → U32Classic bytes
let encoded = encode_svbzd(&samples);

// Decode: U32Classic bytes → unzigzag-undelta → i16
let decoded = decode_svbzd(&encoded, samples.len()).unwrap();
assert_eq!(decoded, samples);
```

## Low-level / into variants

For zero-allocation usage or appending to an existing buffer:

```rust
use svb::{encode_svbzd_into, decode_svbzd_into};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];
let mut buf: Vec<u8> = Vec::new();
encode_svbzd_into(&samples, &mut buf);

let mut out: Vec<i16> = Vec::new();
decode_svbzd_into(&buf, samples.len(), &mut out).unwrap();
assert_eq!(out, samples);
```

## Fused decode

`decode_svbzd_fused` collapses all three decode stages (U32Classic, unzigzag, undelta) into a single SIMD loop. This avoids intermediate buffers and is the preferred path for high-throughput BLOW5 reads:

```rust
use svb::decode_svbzd_fused;

let decoded = decode_svbzd_fused(&encoded, samples.len()).unwrap();
```

`decode_svbzd_fused_into` appends into an existing `Vec<i16>`.

## Parallel decode with fused_from

`decode_svbzd_fused_from` accepts a caller-supplied initial carry value, enabling independent decoding of any sub-stream that starts at a known split point. This is the building block for parallel decoding:

```rust
use svb::decode_svbzd_fused_from;

// Decode second half independently, with known carry from midpoint
let half_b = decode_svbzd_fused_from(&stream_b, n - n_half, mid_carry).unwrap();
```

The `_into` variant (`decode_svbzd_fused_from_into`) appends into an existing `Vec<i16>`.

## Wire format

The encoded byte layout is identical to a `U32Classic`-encoded `Vec<u32>` where each u32 is `zigzag32(samples[i].widened() - samples[i-1].widened())`. There is no additional header; the caller is responsible for tracking `n` (the number of original i16 samples).

The zigzag32 mapping is `(delta << 1) ^ (delta >> 31)`, the same convention used by slow5lib.
