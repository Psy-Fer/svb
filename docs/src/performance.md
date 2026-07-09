# Performance

Benchmarks were measured on GitHub Actions `ubuntu-latest` (Azure x86-64, AVX2) and `ubuntu-24.04-arm` (AArch64, NEON) using `cargo bench --bench decode` with `--sample-size 20`. All throughput numbers are in GB/s of input integers (Melem/s × bytes-per-element ÷ 1000).

## VBZ pipeline breakdown

At 8192 i16 elements with `simd-avx2`, each stage measured in isolation:

| Stage | encode | decode |
|---|---|---|
| delta | 29.2 GB/s | 3.70 GB/s |
| zigzag | 34.2 GB/s | 28.0 GB/s |
| SVB16 (mixed) | 9.24 GB/s | 9.42 GB/s |
| **VBZ (combined, 3-pass)** | **5.70 GB/s** | **2.42 GB/s** |
| **VBZ fused decode** | N/A | **3.68 GB/s** |
| **VBZ2 fused 2-chain decode** | N/A | **5.62 GB/s** |

Zigzag is essentially free (pure bitwise ops, LLVM auto-vectorizes). Delta encode expresses adjacent differences as two overlapping slice views, which LLVM auto-vectorizes to around 29 GB/s with no unsafe code. Delta decode uses an explicit SIMD prefix-sum (SSE2/NEON); the serial carry chain between 8-element blocks limits single-stream throughput to around 3.70 GB/s, essentially the theoretical ceiling for this algorithm.

## Fused VBZ decode

`decode_vbz_fused` collapses all three decode stages into a single SIMD loop. The
SVB16 shuffle and zigzag bitwise ops (~5–6 cycles per 8-element block) execute
during the delta carry-chain stall (~8 cycles), hiding nearly all of their cost.

| | decode throughput |
|---|---|
| `decode_vbz` (3 separate passes) | 2.42 GB/s |
| `decode_vbz_fused` (single SIMD pass) | **3.68 GB/s** |

**1.52× faster** than the pipeline. The fused path reaches 99% of the delta-alone
ceiling (3.70 GB/s): SVB16 and zigzag are effectively free, and the delta carry
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
| `decode_vbz` (3 separate passes) | 2.42 GB/s |
| `decode_vbz_fused` (single SIMD pass) | 3.68 GB/s |
| `decode_vbz2` (format-extension 2-chain) | **5.62 GB/s** |

**1.53× over single-chain fused** at 8192 elements. The 2-chain interleaves two
carry chains in one SIMD loop; the CPU's out-of-order engine overlaps chain A's
carry-extract latency with chain B's prefix-sum arithmetic, hiding most of the
serial dependency cost.

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

Decoding 64 × 8192-element chunks in parallel (64 half-A streams on thread 1,
64 half-B streams on thread 2); run locally for hardware-specific numbers:

| | decode throughput |
|---|---|
| `decode_vbz_fused` (single chain, 1 thread) | 1.84 Gelem/s |
| `decode_vbz2` (2-chain interleaved, 1 thread) | 2.81 Gelem/s |
| `decode_vbz_fused_from_into` (2 threads, batch of 64) | hardware-dependent |

Multi-threaded throughput is highly sensitive to CPU core count, L2/L3 topology,
and scheduler behaviour. The single-thread numbers above are from GitHub Actions
CI (Azure x86-64). For two-thread measurements run the `vbz2_parallel` criterion
benchmark locally: `cargo bench --features simd-avx2 --bench decode -- vbz2_parallel`.
With distinct chunks from independent nanopore reads (the realistic production case)
the two streams share no cache lines and the speedup approaches 2×.

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
scope overhead); multi-threaded results are hardware-dependent — run locally for
specific numbers:

| | throughput | vs single-chain |
|---|---|---|
| single-chain fused (k=1) | 1.84 Gelem/s | 1.00× |
| VBZ-K k=2 (2 threads) | hardware-dependent | — |
| VBZ-K k=4 (4 threads) | hardware-dependent | — |
| VBZ-K k=8 (8 threads) | hardware-dependent | — |

Multi-threaded throughput is not reliably measurable in shared CI environments.
Run `cargo bench --features simd-avx2 --bench decode -- vbzk_parallel` locally
for hardware-specific numbers.

k=4 matches k=2 at this chunk size; k=8 regresses because 8 threads decoding
1024-element sub-streams run into thread-scope overhead and scheduler jitter.
With distinct real-world POD5 chunks (6 000–12 000 samples each), larger
sub-stream sizes would push k=8 above k=4.

### The full POD5 pipeline bottleneck

A POD5 reader decodes: disk → zstd decompress → VBZ decode → i16 samples.
On a typical NVMe system (~6.5 GB/s sequential read):

- **Disk**: 6.5 GB/s × ~3× zstd ratio = ~19.5 GB/s of decoded signal capacity
- **VBZ single-chain (AVX2)**: 1.84 Gelem/s × 2 bytes = **3.68 GB/s** of decoded signal
- **VBZ-K k=4**: scales roughly linearly with cores up to the zstd bottleneck
- **zstd single-core**: ~1.5–2 GB/s compressed ≈ 2–3 Gelem/s, the real
  bottleneck for a single-threaded reader

**The disk is rarely the bottleneck.** A single-threaded reader is zstd-limited.
Parallelising VBZ decode with VBZ-K removes the VBZ ceiling and shifts the
bottleneck back to zstd. To saturate NVMe bandwidth you need multi-threaded
zstd AND VBZ-K simultaneously.

## Delta decode: the 2-chain approach

Delta decode is a serial prefix sum: each output element depends on all previous elements. On x86_64 the SSE2 path processes 8 elements per iteration with a carry chain of ~8 cycles (extract + broadcast + add). We are already at the theoretical single-stream ceiling.

`delta::decode_2chain` breaks this by decoding two independent sub-streams simultaneously. The CPU's out-of-order engine hides one chain's carry latency behind the other's prefix-sum arithmetic, delivering **1.65× throughput**:

| | decode throughput |
|---|---|
| `delta::decode_into` (single stream) | 3.70 GB/s |
| `delta::decode_2chain` (two streams) | **6.58 GB/s** |

**1.78× throughput** with two interleaved chains. This requires one extra `i16`
stored per chunk: the running delta sum at the midpoint (computed by
`delta::mid_carry` during encode, 2 bytes overhead). Each additional sub-chunk
adds another 2-byte carry value and enables one more independent decode stream.

### Path to a parallel-decode VBZ format

With K sub-chunks, all stages of the VBZ pipeline (delta, zigzag, SVB16) can be decoded independently on K cores:

| Sub-chunks | decode throughput | vs. current |
|---|---|---|
| 1 (current VBZ) | 2.42 GB/s | N/A |
| 2 (single-threaded 2-chain) | 5.62 GB/s | 2.3× |
| 2 cores | ~11 GB/s | ~4.5× |
| 4 cores | ~22 GB/s | ~9× |
| 8 cores | ~44 GB/s | ~18× |

The format change is: store K−1 carry values (K−1 × 2 bytes) in the chunk header and split the encoded payload into K equal sub-streams. Compression ratio is unchanged. The `svb` crate provides `decode_2chain` and `mid_carry` as the building blocks.

## SVB-ZD pipeline

At 8192 i16 elements, GitHub Actions CI (Azure x86-64 and AArch64):

### x86-64

| Path | Scalar | SSSE3 | SSSE3× | AVX2 | AVX2× |
|---|---:|---:|---:|---:|---:|
| `encode_svbzd` | 158 Melem/s | 1,140 Melem/s | 7.2× | 1,100 Melem/s | 6.9× |
| `decode_svbzd` (3-pass) | 105 Melem/s | 696 Melem/s | 6.6× | 722 Melem/s | 6.9× |
| `decode_svbzd_fused` | 466 Melem/s | 1,510 Melem/s | 3.2× | 1,510 Melem/s | 3.2× |

### AArch64

| Path | Scalar | NEON | NEON× |
|---|---:|---:|---:|
| `encode_svbzd` | 195 Melem/s | 551 Melem/s | 2.8× |
| `decode_svbzd` (3-pass) | 210 Melem/s | 834 Melem/s | 4.0× |
| `decode_svbzd_fused` | 564 Melem/s | 1,850 Melem/s | 3.3× |

The SIMD encode path computes zigzag-delta inline without an intermediate `Vec<u32>`
allocation. On AVX2 it processes 8 i16 values per iteration using
`_mm256_cvtepi16_epi32` + `_mm_alignr_epi8`; on NEON it uses `vmovl_s16` +
`vextq_s32`.

The fused decode collapses U32Classic decode, unzigzag, and undelta into one SIMD
loop. The 2-ctrl-byte inner loop processes 8 values per iteration. Note that
**SSSE3 ≈ AVX2 for the fused path**: the bottleneck is the serial delta carry chain,
not SIMD width — wider registers do not help once the carry chain is saturated.

### SVB-ZD vs VBZ

Both pipelines operate on i16 signal data; the choice depends on the file format
(BLOW5 vs POD5):

| Metric | VBZ | SVB-ZD |
|---|---|---|
| Codec | SVB16 (1-bit tags) | U32Classic (2-bit tags) |
| Encode (AVX2, 8192 elem) | 2,850 Melem/s | 1,100 Melem/s |
| Fused decode (AVX2, 8192 elem) | 1,840 Melem/s | 1,510 Melem/s |
| Fused decode (NEON, 8192 elem) | 2,280 Melem/s | 1,850 Melem/s |
| Wire format | ONT POD5 / VBZ | hasindu2008/slow5lib BLOW5 |

VBZ is faster because SVB16's 1-bit tags pack more tightly than U32Classic's 2-bit
tags. SVB-ZD handles values that overflow i16 after delta without truncation.

## ex-zd pipeline

ex-zd (`encode_exzd`/`decode_exzd*`) is wire-compatible with slow5lib's `SLOW5_COMPRESS_EX_ZD`: qts (quantize-trailing-shift) → zigzag-delta (u16 domain) → PFOR-style patched/exception encoding (`patched.rs`), where values `> u8::MAX` are pulled out as `(position, residual)` exceptions and everything else is stored as a literal byte. Unlike SVB-ZD's u32 widening, ex-zd keeps the zigzag-delta stream in u16, so patched/exception handling exists specifically to cover the rare values that overflow a byte after delta.

Numbers below are measured locally (AVX2, not GitHub Actions CI like the sections above), on two synthetic profiles: "ramp" (slow drift + small noise, mostly literal bytes — the common case for well-compressing signal) and "spiky" (~20% exception rate, the pathological case).

### Decode strategies

| n | `decode_exzd` (3-pass) | `decode_exzd_fused` | `ExzdDecoder::decode_into` |
|---:|---:|---:|---:|
| 128 | 1,483 Melem/s | 2,118 Melem/s | fastest for repeated small-frame decode |
| 1024 | 1,587 Melem/s | 2,319 Melem/s | (see reusable-context section) |
| 8192 | 2,339 Melem/s | 3,457 Melem/s | |

`decode_exzd_fused` collapses inverse-zigzag, the delta prefix sum, and the qts left-shift into one SIMD pass over the reconstructed zigzag-delta array (`exzd_fused.rs`) — the same fusion technique as `decode_vbz_fused`/`decode_svbzd_fused`. It's the preferred decode path for one-shot calls.

### Reusable decode context (`ExzdDecoder`)

BLOW5 files hold many thousands of individually-encoded reads, each decoded via its own call — a per-call-allocating API pays a fresh `Vec` allocation for `zd`/literal/exception scratch buffers on every read. `ExzdDecoder` reuses those buffers across calls instead:

| read length | per-call alloc | `ExzdDecoder` (reused buffers) | speedup |
|---:|---:|---:|---:|
| 64 | baseline | reused | 1.24-1.79× (varies with exception rate) |
| 512 | baseline | reused | 1.24-1.79× (varies with exception rate) |

Run `cargo bench --features simd-avx2 --bench decode -- exzd_many_small_reads` locally for exact numbers on your hardware.

### Adaptive exception-merge strategy

Reconstructing the full sample stream means merging the literal-byte run and the exception list back into one array. Two strategies exist, picked by exception density:

- **`merge_runs`**: copies each stretch of non-exception values in one `extend_from_slice` (a vectorized memcpy), interrupted only by a single `push` per exception. Wins when exceptions are rare — long runs, few interruptions.
- **`merge_walk`**: writes every output element individually through a raw pointer, no bounds/capacity checks per element — the same mechanism slow5lib's C reference (`ex_depress`) uses. Loses to `merge_runs` on long runs (a scalar per-element loop can't beat a vectorized memcpy) but wins once runs get short enough that `merge_runs`'s per-run call overhead dominates.

The crossover was found empirically (`patched::merge_density_sweep`, an `#[ignore]`d test — run with `cargo test --release --features simd-avx2 -- --ignored --nocapture merge_density_sweep`), timing both strategies directly against each other across a density sweep. It lands consistently around **14-15% exception density**, reproducible across n=128/1024/8192 — the format's own encoder-side warning threshold (~20%, "compression may not be ideal") turned out to be a reasonable starting guess but more conservative than the actual crossover. Decode dispatches on `nex * 7 >= n` (~14.3%) accordingly.

### vs. slow5lib (C reference)

Measured locally against slow5lib's compiled `ptr_compress_ex_zd`/`ex_zd_depress` on the same data profiles and machine (a small standalone C harness, not part of this repo's build — the C reference is MIT-licensed and the wire format was independently verified byte-exact via real C-encoded fixtures in `tests/parity.rs` first).

**Ramp data:**

| n | svb encode | C encode | ratio | svb fused decode | C decode | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 338 Melem/s | 21.7 Melem/s | 15.6× | 2,118 Melem/s | 787 Melem/s | 2.7× |
| 1024 | 470 Melem/s | 141 Melem/s | 3.3× | 2,319 Melem/s | 713 Melem/s | 3.3× |
| 8192 | 616 Melem/s | 440 Melem/s | 1.4× | 3,457 Melem/s | 714 Melem/s | 4.8× |

**Spiky data (~20% exceptions):**

| n | svb fused decode | C decode | ratio |
|---:|---:|---:|---:|
| 128 | 362 Melem/s | 321 Melem/s | 1.1× |
| 1024 | 445 Melem/s | 501 Melem/s | 0.9× |
| 8192 | 526 Melem/s | 575 Melem/s | 0.9× |

svb wins comfortably on the common (mostly-literal) case at every size, and on small pathological inputs. At high exception density and larger n, C's `ex_depress` remains modestly faster (~9-11%).

Encode-side comparisons for spiky data aren't included above — `bench_exzd_encode` only exercises ramp data; C's spiky encode numbers exist (from the standalone harness) but without a corresponding svb measurement they'd be an apples-to-oranges comparison.

### Sanity check against real ONT signal

All of the above (and the merge-strategy threshold tuning) used synthetic "ramp" and "spiky" generators. Since a synthetic profile can diverge from real signal in ways that make a micro-benchmark win irrelevant in practice, the same comparison was repeated against real nanopore reads — the POD5-sourced i16 arrays already used for VBZ parity/benchmark testing (`tests/vectors/parity_*.i16`, `bench_00_101988.i16`; `bench_exzd_real_reads` in `benches/decode.rs`).

**Measured exception density on real reads: 0.9-2.3%** — well below both the ~14.3% adaptive-merge threshold and the 20% "spiky" stress profile. Real nanopore signal looks far more like `vbz_i16_samples` (mostly literal bytes) than the pathological case; `merge_walk` essentially never activates on data like this. `q` (the qts shift) was 0 on every real read tested — real ADC noise fills the low bits, leaving no quantization headroom, unlike the synthetic generators.

| Read | n | exceptions | density | svb fused decode | svb `ExzdDecoder` | C decode | best ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| parity_00 | 2,885 | 44 | 1.53% | 2,378 Melem/s | 2,591 Melem/s | 824 Melem/s | 3.1× |
| parity_01 | 2,915 | 68 | 2.33% | 2,010 Melem/s | 2,450 Melem/s | 783 Melem/s | 3.1× |
| parity_02 | 2,949 | 40 | 1.36% | 2,426 Melem/s | 2,806 Melem/s | 782 Melem/s | 3.6× |
| bench_00 (largest available) | 101,988 | 944 | 0.93% | 2,807 Melem/s | 2,768 Melem/s | 552 Melem/s | 5.1× |

Encoded byte length matched the C reference exactly on every read (e.g. 2,981 / 3,053 / 3,040 / 103,643 bytes) — the wire-compat fixtures used for this check are now permanent `tests/parity.rs` tests (`exzd_c_reference_pod5_*`), not just a one-off local measurement.

**Conclusion: real data lands solidly in svb's strongest regime, not its weakest.** The 2.6-5.1× win here is consistent with (and at the largest read, better than) the synthetic ramp-profile numbers above, confirming ramp was a reasonable stand-in for real signal — if anything it understates how favorable real conditions are, since real density is even lower than what ramp implicitly produces. The ~9-11% gap chased in the section below is a genuine finding about a real (if rare) data shape, but it is not representative of what a BLOW5 reader will see on actual nanopore output.

#### Why the remaining ~9-11% gap (on pathological, not representative, data), and why it resisted closing

Two structural fixes were tried and directly disassembled to confirm they changed what they intended to change, then benchmarked:

- **Eliminating a leftover bounds check.** `merge_walk`'s exception branch read `ex_pos[j]` (guarded by `j < ex_pos.len()`, letting LLVM elide its check) and `ex_val[j]` (not guarded by anything LLVM could see, since it's a different slice with no visible length relationship) — the disassembly showed a real bounds check + conditional panic call on the `ex_val[j]` read that provably could never trigger. Switching both to `get_unchecked` (safe here: `ex_pos.len() == ex_val.len()` always, since both are decoded from `nex` items) removed the check entirely, confirmed by re-disassembling. **Measured impact: ~0%.** A branch that's essentially always-not-taken is nearly free once the branch predictor learns it — the instructions were real, but they weren't costing anything at runtime.
- **Two-phase scatter-then-walk**, matching C's `ex_depress` structure exactly at the source level: scatter exceptions into their final output positions in a dedicated loop first (no branching), then a separate walk over all `n` positions that only ever writes literals, skipping already-filled exception slots. **Measured impact: ~0%**, identical to the single combined loop.

Both were real, verifiable code shapes — matching C's structure line-for-line didn't change wall-clock time at all. That pointed at the comparison itself: recompiling the *same* C source with `clang` instead of `gcc` (isolating language from compiler backend, since `rustc` and `clang` share LLVM) made the C reference itself 15-18% faster. So "C beats Rust here" was the wrong framing — LLVM can extract more from this exact algorithm than gcc did, which means svb's own LLVM-compiled code should be able to reach it too, in principle.

Disassembling LLVM's output for the (already C-structurally-identical) two-phase `merge_walk` explained why it doesn't: LLVM 4×-unrolled the scatter loop and split the walk loop into multiple specialized variants for the Rust version, versus the single compact loop it emits for the same-shaped C. The likely driver is that Rust's `&mut` references let rustc emit stronger `noalias` guarantees than C's raw pointers ever give a compiler — which here pushes LLVM's unrolling heuristics toward a larger, more branch-heavy code shape instead of a smaller, tighter one. Chasing this further would mean fighting LLVM's cost model (e.g. via unroll-suppression hints) rather than fixing an identifiable inefficiency in the code, so the two-phase change was reverted back to the simpler single-loop form (same performance, less code) — the `get_unchecked` fix was kept since it's a correct, justified use of a real invariant regardless of its lack of measured effect.

### Approaches tried and rejected

Documented in code comments (`exzd_fused.rs`, `patched.rs`) rather than repeated here in full, so a future session doesn't re-attempt them without the context:

- **Chunked fusion** (folding `patched`'s reconstruction into the SIMD transform loop, processing fixed-size chunks instead of a full merged array) — tried at both 8-wide (SSE2) and 16-wide (AVX2); both measured slower than the simpler two-stage merge-then-transform design, at every density tested.
- **Naive raw-pointer scatter** (replacing the run-based merge outright with unconditional raw-pointer writes) — regressed 2-3× on low-exception data, because a full-array scalar walk throws away the free vectorized memcpy that `merge_runs` already gets for long literal runs. This is why the final design is *adaptive* rather than a straight replacement.
- **Position-walk via `Vec::push`** (translating C's branch structure into safe Rust without raw pointers) — regressed everywhere, including at high exception density where the raw-pointer version wins. `push`'s per-call capacity check, even with `reserve` called up front, was the dominant cost — the win was never really about the branch structure, it was about removing checks entirely.

## Results vs streamvbyte64 v0.2.0

Measured with `simd-avx2` on GitHub Actions ubuntu-latest (Azure x86-64).
`streamvbyte64` uses its own runtime detection; numbers reflect its best available path.

| Benchmark | svb | sv64 | ratio |
|---|---|---|---|
| U32Classic decode/128 | 8.68 GB/s | 3.71 GB/s | 2.34x |
| U32Classic decode/1024 | 13.6 GB/s | 4.87 GB/s | 2.79x |
| U32Classic decode/8192 | 14.1 GB/s | 4.89 GB/s | 2.88x |
| U32Classic encode/128 | 6.65 GB/s | 2.33 GB/s | 2.85x |
| U32Classic encode/1024 | 8.26 GB/s | 3.08 GB/s | 2.68x |
| U32Classic encode/8192 | 8.93 GB/s | 3.20 GB/s | 2.79x |
| U32Variant0124 decode/128 | 8.98 GB/s | 3.48 GB/s | 2.58x |
| U32Variant0124 decode/1024 | 13.8 GB/s | 4.88 GB/s | 2.83x |
| U32Variant0124 decode/8192 | 14.2 GB/s | 5.00 GB/s | 2.84x |
| U32Variant0124 encode/128 | 6.74 GB/s | 2.37 GB/s | 2.84x |
| U32Variant0124 encode/1024 | 8.32 GB/s | 2.96 GB/s | 2.81x |
| U32Variant0124 encode/8192 | 8.89 GB/s | 3.01 GB/s | 2.95x |
| U64Coder1248 decode/128 | 12.0 GB/s | 5.89 GB/s | 2.04x |
| U64Coder1248 decode/1024 | 15.0 GB/s | 8.68 GB/s | 1.73x |
| U64Coder1248 decode/8192 | 14.8 GB/s | 8.76 GB/s | 1.69x |
| U64Coder1248 encode/128 | 7.37 GB/s | 3.52 GB/s | 2.09x |
| U64Coder1248 encode/1024 | 8.73 GB/s | 4.61 GB/s | 1.89x |
| U64Coder1248 encode/8192 | 8.85 GB/s | 4.80 GB/s | 1.84x |

`svb` is consistently 1.7x–2.9x faster than `streamvbyte64`. The u32 codecs see the
largest gap (approaching 3×); the u64 codecs are closer because 8-byte elements
reduce the SIMD parallelism available per control byte.

## Running benchmarks

```sh
cargo bench --features simd-auto
```

Benchmarks cover all five codec variants across encode/decode and three slice sizes (128, 1024, 8192 elements). Criterion produces HTML reports in `target/criterion/`.

To run a single benchmark by name substring:

```sh
cargo bench --features simd-auto -- U32Classic/decode
```
