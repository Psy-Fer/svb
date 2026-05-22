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

## It's also pretty damn fast

Benchmarked with `simd-auto` on an Intel i7-11800H (AVX2), 8192-element slices:

| Benchmark | svb | streamvbyte64 |
|---|---|---|
| Svb16 encode | 4.91 GB/s | — |
| Svb16 decode | 4.51 GB/s | — |
| VBZ encode (delta + zigzag + SVB16) | 3.14 GB/s | — |
| VBZ decode | 1.88 GB/s | — |
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

Zigzag is essentially free (pure bitwise ops, auto-vectorized). Delta encode expresses adjacent differences as two overlapping slice views, which LLVM auto-vectorizes to ~11 GB/s with no unsafe code. Delta decode uses a SIMD prefix-sum (SSE2/NEON); the serial carry dependency between 8-element blocks limits single-stream throughput to ~3.75 GB/s.

`delta::decode_2chain` decodes two independent sub-streams with interleaved SSE2 carry chains, hiding one chain's carry latency behind the other's arithmetic:

| | encode | decode |
|---|---|---|
| `delta::decode_into` | 11.02 GB/s | 3.75 GB/s |
| `delta::decode_2chain` | — | **6.25 GB/s** |

**1.65x faster** than single-stream decode. Requires one extra `i16` stored per chunk (the running delta sum at the midpoint — see `delta::mid_carry`). This is the key building block for a parallel-decode VBZ format.

Around **2x faster on average** than `streamvbyte64` across all variants and sizes (range: 1.4x–2.7x). Full numbers are in the [Performance](https://psy-fer.github.io/svb/performance.html) docs.

**Why is u32 decode so much faster than encode?** Decode is VERY parallel: one SIMD shuffle instruction unpacks four variable-width values in one shot, and each value's output position is a fixed stride, so there is nothing going on between iterations. Encode has to solve a prefix sum first: the byte offset where value N gets written depends on the cumulative widths of values 0..N-1, so even though tag computation is parallel, the scatter is sequential. SVB16 doesn't show the same gap because 1-bit tags and 1 or 2 byte widths mean the prefix sum is cheap enough to keep pace.

If you run the benchmarks on another system (especially ARM with NEON) I'd love to see the results. Run:

```sh
cargo bench --features simd-auto
```

and open an issue or drop the output in.

## MSRV

1.85 (edition 2024).

## License

MIT. See [LICENSE](LICENSE). Copyright 2026 James Ferguson.
