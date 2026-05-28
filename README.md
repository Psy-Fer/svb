# svb

[![Crates.io](https://img.shields.io/crates/v/svb.svg)](https://crates.io/crates/svb)
[![docs.rs](https://docs.rs/svb/badge.svg)](https://docs.rs/svb)
[![CI](https://github.com/Psy-Fer/svb/actions/workflows/ci.yml/badge.svg)](https://github.com/Psy-Fer/svb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MSRV: 1.87](https://img.shields.io/badge/rustc-1.87+-blue.svg)](https://blog.rust-lang.org/2025/05/15/Rust-1.87.0/)

Pure-Rust [StreamVByte](https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/) covering all major codec variants for `u16`, `u32`, and `u64` integers. Delta and zigzag encoding are composable layers on top. SIMD back-ends are available for x86-64 (SSSE3, AVX2) and AArch64 (NEON).

**[Documentation](https://psy-fer.github.io/svb/) | [API reference](https://docs.rs/svb)**

**StreamVByte** stores each integer in the minimum number of bytes its value requires (1, 2, 3, or 4 bytes for a `u32`; 1 or 2 for a `u16`) and keeps the per-integer width metadata (the *control stream*) separate from the integer bytes (the *data stream*). That two-stream layout is what makes SIMD decode fast: a single shuffle instruction can unpack 4–8 values at once, once the widths are known.

**Delta encoding** replaces each value with its difference from the previous one. For sequences where adjacent values are close (sorted data, slowly-drifting measurements, oscillating signals) the differences are much smaller than the raw values. Smaller values encode to fewer bytes.

**Zigzag encoding** maps signed integers to unsigned so that small absolute values stay small: 0→0, −1→1, 1→2, −2→3, 2→4. This matters when the data has signed deltas: without zigzag, a delta of −1 would encode as 4 bytes (0xFFFFFFFF) rather than 1. With zigzag it encodes as a single byte (0x01).

The three compose naturally: delta shrinks value magnitudes, zigzag keeps the result non-negative and compact, and StreamVByte encodes each small value as efficiently as possible. See the [encoding guide](https://psy-fer.github.io/svb/encoding.html) for a full walkthrough.

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
svb = { version = "0.2", features = ["simd-auto"] }
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

## It's also pretty damn fast

Benchmarked with `simd-auto` on an Intel i7-11800H (AVX2), 8192-element slices:

| Benchmark | svb | streamvbyte64 |
|---|---|---|
| Svb16 encode | 4.91 GB/s | N/A |
| Svb16 decode | 4.51 GB/s | N/A |
| VBZ encode (delta + zigzag + SVB16) | 3.14 GB/s | N/A |
| VBZ decode (3-pass) | 1.88 GB/s | N/A |
| VBZ decode fused (single SIMD pass) | 2.77 GB/s | N/A |
| VBZ2 decode fused (2-chain, single thread) | **3.00 GB/s** | N/A |
| U32Classic decode | 4.07 GB/s | 1.67 GB/s |
| U32Classic encode | 2.08 GB/s | 1.09 GB/s |
| U64Coder1248 decode | 1.90 GB/s | 1.32 GB/s |
| U64Coder1248 encode | 1.25 GB/s | 0.73 GB/s |

VBZ is ~2.5x slower than SVB16 alone. Breaking down the pipeline (8192 i16 elements):

| Stage | encode | decode |
|---|---|---|
| delta | 11.02 GB/s | 3.75 GB/s |
| zigzag | 18.75 GB/s | 14.83 GB/s |
| SVB16 | 4.91 GB/s | 4.51 GB/s |
| **VBZ combined (3-pass)** | **3.14 GB/s** | **1.88 GB/s** |
| **VBZ fused decode** | N/A | **2.77 GB/s** |

Around **2x faster on average** than `streamvbyte64` across all variants and sizes (range: 1.4x–2.7x). Full stage-by-stage breakdowns, fused decoder analysis, and VBZ-K parallel decode numbers are in the [Performance](https://psy-fer.github.io/svb/performance.html) docs.

If you run the benchmarks on another system (especially ARM with NEON) I'd love to see the results. Run:

```sh
cargo bench --features simd-auto
```

and open an issue or drop the output in.

## Validation

Real-data parity testing is done through [pod5lib](https://crates.io/crates/pod5lib), a pure-Rust POD5 reader that uses `svb` for VBZ decompression and validates output against real Oxford Nanopore sequencing data.

## Acknowledgements

StreamVByte was invented by [Daniel Lemire](https://lemire.me), Mauel Kurz, and Robert Rupp. The `U32Classic` wire format is compatible with Lemire's [C streamvbyte library](https://github.com/lemire/streamvbyte). The u64 codec variants follow the format defined by [`streamvbyte64`](https://crates.io/crates/streamvbyte64). Benchmarks compare against `streamvbyte64 v0.2.0`.

## AI assistance

This library was developed with AI assistance (Claude). Architecture decisions, wire-compatibility validation, and algorithm choices are the author's own; AI tooling served as an accelerator over existing skill. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## MSRV

1.87 (edition 2024; SIMD intrinsics require target_feature_11, stabilised in 1.87).

## License

MIT. See [LICENSE](LICENSE). Copyright 2026 James Ferguson.
