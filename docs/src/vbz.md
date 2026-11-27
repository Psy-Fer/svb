# VBZ Pipeline

The VBZ pipeline compresses 16-bit ADC signal data as used in Oxford Nanopore POD5 files. It chains four stages:

```text
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

## Extra bytes after the frame

The decoders expect `data` to hold exactly one encoded frame, as a POD5
signal chunk does. If the control stream says the frame ends before the end
of `data`, they return `DecodeError::TrailingData` with the position where
the frame ended (`used`) and the length of `data` (`len`). This also catches a
corrupted control stream that claims fewer bytes than were written, which
would otherwise decode to wrong samples without an error. It does not catch
corruption that leaves the length unchanged, such as a changed data byte.

```rust
use svb::{encode_vbz, decode_vbz_fused, DecodeError};

let samples = [100i16, 101, 103];
let mut data = encode_vbz(&samples);
let frame_len = data.len();
data.push(0);
match decode_vbz_fused(&data, samples.len()) {
    Err(DecodeError::TrailingData { used, len }) => {
        assert_eq!((used, len), (frame_len, frame_len + 1));
    }
    other => panic!("{other:?}"),
}
```

VBZ-K applies the same rule to each sub-stream, whose end is given by the
next sub-stream's data offset in the header. To decode a frame that sits
inside a larger buffer, slice it out first. The codec level
`Svb16::decode_to_slice` does accept extra bytes and returns the number of
bytes the frame used.

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

## Fused decode

`decode_vbz_fused` gives the same output as `decode_vbz` but does SVB16,
zigzag and delta decode in one SIMD loop instead of three passes. Use it for
POD5 reads. `decode_vbz_fused_from` starts the running sum from a given carry
value, which lets you decode separate parts of a stream independently.

```rust
use svb::{decode_vbz_fused, encode_vbz};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];
let encoded = encode_vbz(&samples);
assert_eq!(decode_vbz_fused(&encoded, samples.len()).unwrap(), samples);
```

## Decoding into your own buffer

`decode_vbz_fused_to_slice` and `decode_vbz_fused_from_to_slice` write into a
caller-supplied `&mut [i16]`. The number of samples is `out.len()`, which must
match the number that was encoded. On error the contents of `out` are
unspecified; the `_into` functions leave the `Vec` at its original length
instead.

```rust
use svb::{decode_vbz_fused_to_slice, encode_vbz};

let samples = [100i16, 101, 103, 102, 98];
let encoded = encode_vbz(&samples);

let mut out = [0i16; 5];
decode_vbz_fused_to_slice(&encoded, &mut out).unwrap();
assert_eq!(out, samples);
```

`decode_vbz_to_slice` does the same with the 3-pass decoder. It is slower than
the fused one and allocates a temporary buffer for the SVB16 pass.

`encode_vbz_to_slice` encodes into a caller-supplied `&mut [u8]` and returns
the number of bytes written. The slice must be at least
`vbz_max_encoded_len(samples.len())` bytes (`ceil(n / 8)` control bytes plus
2 bytes per sample), otherwise `EncodeError::OutputTooSmall` is returned and
the slice is left unchanged. The delta and zigzag passes still allocate
temporary buffers.

```rust
use svb::{encode_vbz, encode_vbz_to_slice, vbz_max_encoded_len};

let samples = [100i16, 101, 103, 102, 98];
let mut buf = vec![0u8; vbz_max_encoded_len(samples.len())];
let len = encode_vbz_to_slice(&samples, &mut buf).unwrap();
assert_eq!(&buf[..len], encode_vbz(&samples));
```

The experimental VBZ-K format has `encode_vbzk_to_slice` (sized with
`vbzk_max_encoded_len(n, k)`), `decode_vbzk_to_slice` and
`decode_vbzk_parallel_to_slice`. The parallel version gives each thread its
own part of `out`, so the sub-streams are not copied together afterwards.

