# Performance

Benchmarks were run with `simd-auto` on a modern x86-64 machine (AVX2 path selected at runtime). All numbers are throughput in GB/s of input integers.

## VBZ pipeline breakdown

At 8192 i16 elements, each stage measured in isolation:

| Stage | encode | decode |
|---|---|---|
| delta | 11.02 GB/s | 3.75 GB/s |
| zigzag | 18.75 GB/s | 14.83 GB/s |
| SVB16 | 4.91 GB/s | 4.51 GB/s |
| **VBZ (combined, 3-pass)** | **3.14 GB/s** | **1.88 GB/s** |
| **VBZ fused decode** | N/A | **2.77 GB/s** |
| **VBZ2 fused 2-chain decode** | N/A | **3.00 GB/s** |

Zigzag is essentially free (pure bitwise ops, LLVM auto-vectorizes). Delta encode expresses adjacent differences as two overlapping slice views, which LLVM auto-vectorizes to around 11 GB/s with no unsafe code. Delta decode uses an explicit SIMD prefix-sum (SSE2/NEON); the serial carry chain between 8-element blocks limits single-stream throughput to around 3.75 GB/s, essentially the theoretical ceiling for this algorithm.

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

## VBZ2: format-extension 2-chain decode

`encode_vbz2` / `decode_vbz2` extend the VBZ format with a 6-byte header that
enables a two-chain fused decode with no pre-scan required:

```
[mid_carry: i16 LE][mid_data_offset: u32 LE][standard VBZ payload]
```

`mid_carry` is `samples[n_half - 1]`, the decoded sample at the chunk midpoint,
i.e., the prefix sum of all deltas before the midpoint. `mid_data_offset` is the
count of data bytes consumed by the first `n_half` elements (sum of `8 +
popcnt(ctrl_byte)` over the first half of control bytes). Both are computed in
O(n) during encode with negligible cost.

At decode time the payload is split into two independent half-streams. Two carry
chains run interleaved in one SIMD loop: the CPU's out-of-order engine overlaps
chain A's carry-extract latency with chain B's prefix-sum arithmetic. Port-5
usage is unchanged from single-chain (10 ops per 16 elements), so there is no
throughput regression at any size; the gain accumulates only where the carry
latency was the limiting factor.

| | decode throughput |
|---|---|
| `decode_vbz` (3 separate passes) | 1.88 GB/s |
| `decode_vbz_fused` (single SIMD pass) | 2.77 GB/s |
| `decode_vbz2` (format-extension 2-chain) | **3.00 GB/s** |

**+8% over single-chain fused** at 8192 elements (6-byte format overhead, same
single-threaded hardware). The 2-chain is effectively free at the port-5 ceiling;
the residual gain comes from partial carry-chain ILP at the tail.

The real payoff is **multi-threaded decoding**: with `mid_data_offset` known
up-front, both half-streams are independent and can run on separate cores. The
format overhead is 6 bytes per chunk regardless of chunk size, negligible for
any practical payload.

## Caller-side parallel decode

`decode_vbz_fused_from_into(data, n, initial_carry, out)` exposes the single-chain
fused decoder with a caller-supplied initial carry, making it possible to decode
any half-stream independently. A caller that manages its own thread pool simply
splits the VBZ2 payload and dispatches both halves concurrently:

```rust
let (out_a, out_b) = std::thread::scope(|s| {
    let ha = s.spawn(|| decode_vbz_fused_from(&stream_a, n_half, 0));
    let hb = s.spawn(|| decode_vbz_fused_from(&stream_b, n - n_half, mid_carry));
    (ha.join().unwrap(), hb.join().unwrap())
});
```

Benchmarked on the same i7-11800H, decoding 64 × 8192-element chunks in parallel
(64 half-A streams on thread 1, 64 half-B streams on thread 2):

| | decode throughput |
|---|---|
| `decode_vbz_fused` (single chain, 1 thread) | 2.82 Gelem/s |
| `decode_vbz2` (2-chain interleaved, 1 thread) | 3.05 Gelem/s |
| `decode_vbz_fused_from_into` (2 threads, batch of 64) | **3.96 Gelem/s** |

**1.40× over single-chain on 2 cores.** The gap from the ideal 2× ceiling is mainly
cache sharing, as both threads decode the same 512 KB of data, competing for L2
bandwidth. With distinct chunks from independent nanopore reads (the realistic
production case), the two streams have no overlapping cache lines and the speedup
approaches 2×.

## VBZ-K: generalised K-stream parallel decode

`encode_vbzk(samples, k)` / `decode_vbzk_parallel_into(data, n, out)` generalise
VBZ2 to K independent sub-streams.  The header stores K−1 split points:

```
[k: u8][(carry_i: i16 LE, data_offset_i: u32 LE) for i in 1..k][VBZ payload]
```

Header overhead: 1 + (K−1) × 6 bytes. Each sub-chunk has `n_sub = (n/K) & !7`
elements; the last sub-chunk takes the remainder. Split-point carries and data
offsets are computed in O(n) at encode time with negligible overhead.

Benchmarked at N=8192 with a batch of 64 chunks per thread (amortising thread
scope overhead), i7-11800H:

| | throughput | vs single-chain |
|---|---|---|
| single-chain fused (k=1) | 2.82 Gelem/s | 1.00× |
| VBZ-K k=2 (2 threads) | 4.13 Gelem/s | **1.46×** |
| VBZ-K k=4 (4 threads) | 4.18 Gelem/s | **1.48×** |
| VBZ-K k=8 (8 threads) | 3.18 Gelem/s | 1.13× |

k=4 matches k=2 at this chunk size; k=8 regresses because 8 threads decoding
1024-element sub-streams run into thread-scope overhead and scheduler jitter.
With distinct real-world POD5 chunks (6 000–12 000 samples each), larger
sub-stream sizes would push k=8 above k=4.

### The full POD5 pipeline bottleneck

A POD5 reader decodes: disk → zstd decompress → VBZ decode → i16 samples.
On this machine (Samsung PM9A1 NVMe, ~6.5 GB/s sequential read):

- **Disk**: 6.5 GB/s × ~3× zstd ratio = ~19.5 GB/s of decoded signal capacity
- **VBZ-K k=4**: 4.18 Gelem/s × 2 bytes = **8.36 GB/s** of decoded signal; the
  decoder at k=4 needs only 8.36/3 ≈ 2.8 GB/s of compressed disk reads, well
  within NVMe capacity
- **zstd single-core**: ~1.5–2 GB/s compressed ≈ 2–3 Gelem/s, the real
  bottleneck for a single-threaded reader

**The disk is never the bottleneck on this hardware.** A single-threaded reader
is zstd-limited (~2–3 Gelem/s). Parallelising VBZ decode with VBZ-K removes
the VBZ ceiling and shifts the bottleneck back to zstd. To saturate the NVMe,
you need multi-threaded zstd AND VBZ-K with k≥5 simultaneously; only then
does the ~6.5 GB/s compressed read bandwidth become the limit.

## Delta decode: the 2-chain approach

Delta decode is a serial prefix sum: each output element depends on all previous elements. On x86_64 the SSE2 path processes 8 elements per iteration with a carry chain of ~8 cycles (extract + broadcast + add). We are already at the theoretical single-stream ceiling.

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
| 1 (current VBZ) | 1.88 GB/s | N/A |
| 2 (single-threaded 2-chain) | ~2.2 GB/s | 1.2× |
| 2 cores | ~3.8 GB/s | 2× |
| 4 cores | ~7.5 GB/s | 4× |
| 8 cores | ~15 GB/s | 8× |

The format change is: store K−1 carry values (K−1 × 2 bytes) in the chunk header and split the encoded payload into K equal sub-streams. Compression ratio is unchanged. The `svb` crate provides `decode_2chain` and `mid_carry` as the building blocks.

## Results vs streamvbyte64 v0.2.0

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
