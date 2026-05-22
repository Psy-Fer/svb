# Performance

Benchmarks were run with `simd-auto` on a modern x86-64 machine (AVX2 path selected at runtime). All numbers are throughput in GB/s of input integers.

## Results vs streamvbyte64

## VBZ pipeline breakdown

At 8192 i16 elements, each stage measured in isolation:

| Stage | encode | decode |
|---|---|---|
| delta | 11.02 GB/s | 3.75 GB/s |
| zigzag | 18.75 GB/s | 14.83 GB/s |
| SVB16 | 4.91 GB/s | 4.51 GB/s |
| **VBZ (combined, 3-pass)** | **3.14 GB/s** | **1.88 GB/s** |
| **VBZ fused decode** | — | **2.77 GB/s** |

Zigzag is essentially free (pure bitwise ops, LLVM auto-vectorizes). Delta encode expresses adjacent differences as two overlapping slice views, which LLVM auto-vectorizes to around 11 GB/s with no unsafe code. Delta decode uses an explicit SIMD prefix-sum (SSE2/NEON); the serial carry chain between 8-element blocks limits single-stream throughput to around 3.75 GB/s — essentially the theoretical ceiling for this algorithm.

## Fused VBZ decode

`decode_vbz_fused` collapses all three decode stages into a single SIMD loop. The
SVB16 shuffle and zigzag bitwise ops (~5–6 cycles per 8-element block) execute
during the delta carry-chain stall (~8 cycles), hiding nearly all of their cost.

| | decode throughput |
|---|---|
| `decode_vbz` (3 separate passes) | 1.88 GB/s |
| `decode_vbz_fused` (single SIMD pass) | **2.77 GB/s** |

**1.47× faster** than the pipeline. The fused path reaches 74% of the delta-alone
ceiling (3.75 GB/s): SVB16 and zigzag are effectively free, and the delta carry
chain is the only remaining bottleneck.

## Delta decode: the 2-chain approach

Delta decode is a serial prefix sum — each output element depends on all previous elements. On x86_64 the SSE2 path processes 8 elements per iteration with a carry chain of ~8 cycles (extract + broadcast + add). We are already at the theoretical single-stream ceiling.

`delta::decode_2chain` breaks this by decoding two independent sub-streams simultaneously. The CPU's out-of-order engine hides one chain's carry latency behind the other's prefix-sum arithmetic, delivering **1.65× throughput**:

| | decode throughput |
|---|---|
| `delta::decode_into` (single stream) | 3.75 GB/s |
| `delta::decode_2chain` (two streams) | **6.25 GB/s** |

This requires one extra `i16` stored per chunk: the running delta sum at the midpoint (computed by `delta::mid_carry` during encode, 2 bytes overhead). Each additional sub-chunk adds another 2-byte carry value and enables one more independent decode stream.

### Path to a parallel-decode VBZ format

With K sub-chunks, all stages of the VBZ pipeline (delta, zigzag, SVB16) can be decoded independently on K cores:

| Sub-chunks | decode throughput | vs. current |
|---|---|---|
| 1 (current VBZ) | 1.88 GB/s | — |
| 2 (single-threaded 2-chain) | ~2.2 GB/s | 1.2× |
| 2 cores | ~3.8 GB/s | 2× |
| 4 cores | ~7.5 GB/s | 4× |
| 8 cores | ~15 GB/s | 8× |

The format change is: store K−1 carry values (K−1 × 2 bytes) in the chunk header and split the encoded payload into K equal sub-streams. Compression ratio is unchanged. The `svb` crate provides `decode_2chain` and `mid_carry` as the building blocks.

## Results vs streamvbyte64

| Benchmark | svb | sv64 | ratio |
|---|---|---|---|
| U32Classic decode/128 | 2.29 GB/s | 0.92 GB/s | 2.48x |
| U32Classic decode/1024 | 3.43 GB/s | 1.37 GB/s | 2.51x |
| U32Classic decode/8192 | 4.07 GB/s | 1.67 GB/s | 2.44x |
| U32Classic encode/128 | 1.64 GB/s | 0.60 GB/s | 2.74x |
| U32Classic encode/1024 | 1.98 GB/s | 1.02 GB/s | 1.94x |
| U32Classic encode/8192 | 2.08 GB/s | 1.09 GB/s | 1.90x |
| U32Variant0124 decode/128 | 2.31 GB/s | 1.00 GB/s | 2.30x |
| U32Variant0124 decode/1024 | 2.99 GB/s | 1.42 GB/s | 2.11x |
| U32Variant0124 decode/8192 | 3.87 GB/s | 1.67 GB/s | 2.32x |
| U32Variant0124 encode/128 | 1.65 GB/s | 0.63 GB/s | 2.64x |
| U32Variant0124 encode/1024 | 1.98 GB/s | 0.93 GB/s | 2.13x |
| U32Variant0124 encode/8192 | 2.07 GB/s | 1.04 GB/s | 1.98x |
| U64Coder1248 decode/128 | 1.31 GB/s | 0.94 GB/s | 1.40x |
| U64Coder1248 decode/1024 | 1.92 GB/s | 1.41 GB/s | 1.36x |
| U64Coder1248 decode/8192 | 1.90 GB/s | 1.32 GB/s | 1.44x |
| U64Coder1248 encode/128 | 0.89 GB/s | 0.49 GB/s | 1.83x |
| U64Coder1248 encode/1024 | 1.23 GB/s | 0.76 GB/s | 1.62x |
| U64Coder1248 encode/8192 | 1.25 GB/s | 0.73 GB/s | 1.72x |

`svb` is consistently 1.4x–2.7x faster than `streamvbyte64`. The u32 codecs see the largest gap; the u64 codecs are closer because 8-byte elements reduce how much SIMD parallelism is available per control byte.

## Running benchmarks

```sh
cargo bench --features simd-auto
```

Benchmarks cover all five codec variants across encode/decode and three slice sizes (128, 1024, 8192 elements). Criterion produces HTML reports in `target/criterion/`.

To run a single benchmark by name substring:

```sh
cargo bench --features simd-auto -- U32Classic/decode
```
