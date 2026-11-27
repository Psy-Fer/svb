# SVB-ZD Pipeline

The SVB-ZD pipeline compresses 16-bit signal data into a compact byte stream. It is wire-compatible with the [slow5lib](https://github.com/hasindu2008/slow5lib) `SLOW5_COMPRESS_SVB_ZD` format used in BLOW5 files. The pipeline chains three stages:

```text
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

## Extra bytes after the frame

The decoders expect `data` to hold exactly one encoded frame (in BLOW5, the
record's signal bytes after the 4-byte sample count). If the control stream
says the frame ends before the end of `data`, they return
`DecodeError::TrailingData { used, len }`. This also catches a corrupted
control stream that claims fewer bytes than were written. See the same
section in [VBZ](vbz.md#extra-bytes-after-the-frame) for an example.

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
use svb::{decode_svbzd_fused, encode_svbzd};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];
let encoded = encode_svbzd(&samples);
let decoded = decode_svbzd_fused(&encoded, samples.len()).unwrap();
assert_eq!(decoded, samples);
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

`decode_svbzd_to_slice` does the same with the 3-pass decoder. It is about
half the speed of the fused one and allocates a temporary buffer for the
U32Classic pass.

## Parallel decode with fused_from

`decode_svbzd_fused_from` accepts a caller-supplied initial carry value, enabling independent decoding of any sub-stream that starts at a known split point. This is the building block for parallel decoding:

A stream can be split after any multiple of 4 samples. Each control byte
covers 4 samples and holds a 2-bit tag per sample; a sample's data takes
`tag + 1` bytes.

```rust
use svb::{decode_svbzd_fused_from, encode_svbzd};

let samples: Vec<i16> = (0..64).map(|i| i * 37 - 1000).collect();
let n = samples.len();
let encoded = encode_svbzd(&samples);

let n_half = 32;
let (ctrl, data) = encoded.split_at(n.div_ceil(4));
let (ctrl_a, ctrl_b) = ctrl.split_at(n_half / 4);
let data_a_len: usize = ctrl_a
    .iter()
    .map(|&c| (0..4).map(|i| ((c >> (2 * i)) & 3) as usize + 1).sum::<usize>())
    .sum();
let (data_a, data_b) = data.split_at(data_a_len);
let stream_a = [ctrl_a, data_a].concat();
let stream_b = [ctrl_b, data_b].concat();

// The second half starts from the last sample of the first. If that value is
// stored with the data, the halves can be decoded on separate threads.
let mid_carry = samples[n_half - 1] as i32;
let half_a = decode_svbzd_fused_from(&stream_a, n_half, 0).unwrap();
let half_b = decode_svbzd_fused_from(&stream_b, n - n_half, mid_carry).unwrap();
assert_eq!([half_a, half_b].concat(), samples);
```

The `_into` variant (`decode_svbzd_fused_from_into`) appends into an existing `Vec<i16>`. `decode_svbzd_fused_from_to_slice` writes into a slice, so each half can be decoded straight into its own part of one output buffer (for example with `split_at_mut`).

## Wire format

The encoded byte layout is identical to a `U32Classic`-encoded `Vec<u32>` where each u32 is `zigzag32(samples[i].widened() - samples[i-1].widened())`. There is no additional header; the caller is responsible for tracking `n` (the number of original i16 samples).

The zigzag32 mapping is `(delta << 1) ^ (delta >> 31)`, the same convention used by slow5lib.
