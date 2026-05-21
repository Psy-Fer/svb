# svb

Pure-Rust [StreamVByte](https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/) covering all major codec variants for `u16`, `u32`, and `u64` integers. Delta and zigzag encoding are composable layers on top. SIMD back-ends are available for x86-64 (SSSE3, AVX2) and AArch64 (NEON).

**[Documentation](https://psy-fer.github.io/svb/) | [API reference](https://docs.rs/svb)**

## Codec variants

| Variant | Element | Byte widths | Notes |
|---|---|---|---|
| `Svb16` | `u16` | 1/2 | ONT VBZ format |
| `U32Classic` | `u32` | 1/2/3/4 | Lemire / C library compatible |
| `U32Variant0124` | `u32` | 0/1/2/4 | Better compression for sparse data |
| `U64Coder1234` | `u64` | 1/2/3/4 | Values up to `u32::MAX` |
| `U64Coder1248` | `u64` | 1/2/4/8 | Full u64 range |

## Installation

```toml
[dependencies]
svb = { version = "0.1", features = ["simd-auto"] }
```

## Quick start

```rust
use svb::u32::U32Classic;

let values: Vec<u32> = vec![1, 500, 70_000, 16_000_000];
let encoded = U32Classic.encode(&values);
let decoded = U32Classic.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

For the VBZ pipeline (Oxford Nanopore POD5 signal data):

```rust
use svb::{encode_vbz, decode_vbz};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];
let encoded = encode_vbz(&samples);
let decoded = decode_vbz(&encoded, samples.len()).unwrap();
assert_eq!(decoded, samples);
```

## MSRV

1.85 (edition 2024).

## License

MIT. See [LICENSE](LICENSE). Copyright 2026 James Ferguson.
