# Performance

Benchmarks were run with `simd-auto` on a modern x86-64 machine (AVX2 path selected at runtime). All numbers are throughput in GB/s of input integers.

## Results vs streamvbyte64

## VBZ pipeline breakdown

At 8192 i16 elements, each stage measured in isolation:

| Stage | encode | decode |
|---|---|---|
| delta | 11.02 GB/s | 3.50 GB/s |
| zigzag | 18.75 GB/s | 14.83 GB/s |
| SVB16 | 4.91 GB/s | 4.51 GB/s |
| **VBZ (combined)** | **3.14 GB/s** | **1.88 GB/s** |

Zigzag is essentially free (pure bitwise ops, LLVM auto-vectorizes). Delta encode expresses adjacent differences as two overlapping slice views, which LLVM auto-vectorizes to around 11 GB/s with no unsafe code. Delta decode uses an explicit SIMD prefix-sum (SSE2/NEON); the carry dependency limits it to around 3.5 GB/s.

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
