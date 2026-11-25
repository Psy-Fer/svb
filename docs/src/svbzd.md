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

## Encoding into your own buffer

`encode_svbzd_to_slice` writes into a caller-supplied `&mut [u8]` and returns
the number of bytes written. The buffer must be at least
`svbzd_max_encoded_len(n)` bytes (`ceil(n / 4) + 4 * n`), otherwise
`EncodeError::OutputTooSmall` is returned. Encoded i16 samples never use more
than 3 data bytes each, but the SIMD encoders store 16 bytes at a time and need
the 4-byte bound to stay inside the buffer.

```rust
use svb::{encode_svbzd, encode_svbzd_to_slice, svbzd_max_encoded_len};

let samples = [100i16, 101, 103, 102, 98];
let mut buf = vec![0u8; svbzd_max_encoded_len(samples.len())];
let len = encode_svbzd_to_slice(&samples, &mut buf).unwrap();
assert_eq!(&buf[..len], encode_svbzd(&samples));
```

## Fused decode

`decode_svbzd_fused` collapses all three decode stages (U32Classic, unzigzag, undelta) into a single SIMD loop. This avoids intermediate buffers and is the preferred path for high-throughput BLOW5 reads:

```rust
use svb::decode_svbzd_fused;

let decoded = decode_svbzd_fused(&encoded, samples.len()).unwrap();
```

`decode_svbzd_fused_into` appends into an existing `Vec<i16>`.

## Decoding into your own buffer

`decode_svbzd_fused_to_slice` and `decode_svbzd_fused_from_to_slice` write into
a caller-supplied `&mut [i16]`. The number of samples is `out.len()`, which must
match the number that was encoded. On error the contents of `out` are
unspecified; the `_into` functions leave the `Vec` at its original length
instead.

```rust
use svb::{decode_svbzd_fused_to_slice, encode_svbzd};

let samples = [100i16, 101, 103, 102, 98];
let encoded = encode_svbzd(&samples);

let mut out = [0i16; 5];
decode_svbzd_fused_to_slice(&encoded, &mut out).unwrap();
assert_eq!(out, samples);
```

## Parallel decode with fused_from

`decode_svbzd_fused_from` accepts a caller-supplied initial carry value, enabling independent decoding of any sub-stream that starts at a known split point. This is the building block for parallel decoding:

```rust
use svb::decode_svbzd_fused_from;

// Decode second half independently, with known carry from midpoint
let half_b = decode_svbzd_fused_from(&stream_b, n - n_half, mid_carry).unwrap();
```

The `_into` variant (`decode_svbzd_fused_from_into`) appends into an existing `Vec<i16>`. `decode_svbzd_fused_from_to_slice` writes into a slice, so each half can be decoded straight into its own part of one output buffer (for example with `split_at_mut`).

## Wire format

The encoded byte layout is identical to a `U32Classic`-encoded `Vec<u32>` where each u32 is `zigzag32(samples[i].widened() - samples[i-1].widened())`. There is no additional header; the caller is responsible for tracking `n` (the number of original i16 samples).

The zigzag32 mapping is `(delta << 1) ^ (delta >> 31)`, the same convention used by slow5lib.
