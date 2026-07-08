# ex-zd Pipeline

The ex-zd pipeline improves on [SVB-ZD](svbzd.md) for 16-bit signal data. It is wire-compatible with the [slow5lib](https://github.com/hasindu2008/slow5lib) `SLOW5_COMPRESS_EX_ZD` format used in BLOW5 files. The pipeline chains four stages:

```
raw i16 samples
  →  qts               (find largest right-shift q ≤ 5 that loses no low bits, apply it)
  →  zigzag-delta       (delta of differences → zigzag16 to make values unsigned, u16 domain)
  →  patched exceptions (values ≤ 255 → literal byte; values > 255 → position + residual,
                          both StreamVByte-encoded with U32Classic)
  →  zstd               (outer entropy coding; NOT part of this crate)
```

Two differences from SVB-ZD:

- **qts pre-pass.** ADC samples are frequently multiples of a power of two (the low bits carry no information). Shifting them out before delta/zigzag makes the resulting deltas smaller and more compressible. The shift is lossless and reversed on decode.
- **Patched/exception encoding instead of a per-value StreamVByte tag.** Rather than SVB-ZD's 2-bit-tag-per-value scheme, ex-zd stores most zigzag-delta values as a single literal byte and pulls the rare large values ("exceptions") out into a separate, StreamVByte-encoded side channel. This tends to compress better when most deltas are small and only occasional spikes need the full range.

Unlike [`encode_vbz`](vbz.md)/[`encode_svbzd`](svbzd.md), the ex-zd frame is self-describing: it embeds a version byte and the sample count, so `decode_exzd` takes no `n` parameter.

## High-level API

```rust
use svb::{encode_exzd, decode_exzd};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];

// Encode: i16 → qts → zigzag-delta → patched(U32Classic) bytes
let encoded = encode_exzd(&samples);

// Decode: bytes are self-describing (version + sample count embedded)
let decoded = decode_exzd(&encoded).unwrap();
assert_eq!(decoded, samples);
```

## Low-level / into variants

For zero-allocation usage or appending to an existing buffer:

```rust
use svb::{encode_exzd_into, decode_exzd_into};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];
let mut buf: Vec<u8> = Vec::new();
encode_exzd_into(&samples, &mut buf);

let mut out: Vec<i16> = Vec::new();
decode_exzd_into(&buf, &mut out).unwrap();
assert_eq!(out, samples);
```

## Composable primitives

The qts and patched/exception stages are exposed independently, matching [delta and zigzag](transforms.md):

- [`svb::quantize`] — `find_qts`, `apply_shift`, `unshift_inplace`, fixed to `i16`.
- [`svb::patched`] — `encode_into`/`decode_into` over `&[u16]`, generically useful for any patched/exception encoding scenario beyond ex-zd.

## Wire format

```
u8   version        (0)
u64  nin             (sample count, little-endian)
u8   q               (qts shift, 0..=5)
u16  zd[0]           (first zigzag-delta value, stored raw — no predecessor to patch against)
u32  nex             (exception count over the remaining nin-1 values)
  if nex > 1:
    u32  nex_pos_press_bytes
    ..   nex_pos_press_bytes    (U32Classic-encoded, off-by-one delta-encoded exception positions)
    u32  nex_press_bytes
    ..   nex_press_bytes        (U32Classic-encoded exception residuals, value - 256)
  elif nex == 1:
    u32  position
    u32  residual               (value - 256)
  ..   (nin - 1 - nex) literal bytes, one per non-exception value, in stream order
```

The off-by-one position delta trick (`pos[0]` raw, `pos[i] - pos[i-1] - 1` thereafter) is specific to this exception-position encoding and is not part of the general-purpose [`svb::delta`](transforms.md) module — it relies on positions being strictly increasing, which only holds for this use case.
