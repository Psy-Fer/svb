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

## Two carry chains in one thread (VBZ2, removed)

Versions 0.1 to 0.3 had a VBZ2 format: standard VBZ with a 6-byte header
holding the sample and data offset at the midpoint. Its decoder ran the two
halves as two interleaved carry chains in one SIMD loop, so the out-of-order
engine could overlap one chain's carry-extract latency with the other chain's
prefix sum. At 8192 elements that reached 5.62 GB/s on one thread, 1.53x the
single-chain fused decoder. VBZ2 was deprecated from the first release in
favour of VBZ-K and removed in 0.4.0. VBZ-K decodes its sub-streams with the single-chain
decoder, so the interleaving trick is not currently used anywhere; it would
apply to VBZ-K with k=2 if a kernel were written for it.

## Caller-side parallel decode

`decode_vbz_fused_from_into(data, n, initial_carry, out)` exposes the single-chain
fused decoder with a caller-supplied initial carry, so any part of a VBZ stream
can be decoded on its own once the sample before it and the position of its
data are known (see the `decode_vbz_fused_from` example). VBZ-K stores both
for each sub-stream. A caller with its own thread pool can split the stream
and decode the parts concurrently:

```rust
let (out_a, out_b) = std::thread::scope(|s| {
    let ha = s.spawn(|| decode_vbz_fused_from(&stream_a, n_half, 0));
    let hb = s.spawn(|| decode_vbz_fused_from(&stream_b, n - n_half, carry));
    (ha.join().unwrap(), hb.join().unwrap())
});
```

Multi-threaded throughput is highly sensitive to CPU core count, L2/L3 topology,
and scheduler behaviour, so it is not measured in CI. Run the `vbzk_parallel`
criterion benchmark locally for hardware-specific numbers:
`cargo bench --features simd-avx2 --bench decode -- vbzk_parallel`.
With distinct chunks from independent nanopore reads (the realistic production case)
the streams share no cache lines and the speedup approaches the thread count.

## VBZ-K: generalised K-stream parallel decode

`encode_vbzk(samples, k)` / `decode_vbzk_parallel_into(data, n, out)` split a
VBZ stream into K independent sub-streams. The header stores K−1 split points:

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
| 2 (one thread, interleaved chains; VBZ2, removed) | 5.62 GB/s | 2.3× |
| 2 cores | ~11 GB/s | ~4.5× |
| 4 cores | ~22 GB/s | ~9× |
| 8 cores | ~44 GB/s | ~18× |

The format change is: store K−1 carry values (K−1 × 2 bytes) in the chunk header and split the encoded payload into K equal sub-streams. Compression ratio is unchanged. VBZ-K is this format. `delta::decode_2chain` and `delta::mid_carry` remain as building blocks for the interleaved-chain decode.

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

Reconstructing the full sample stream means merging the literal-byte run and the exception list back into one array. Since 0.4.0 this happens in place: the literal bytes are widened straight into the front of the output, and the merge works from the back, moving each literal right to its final position and writing the exceptions into the gaps. 0.3.1 widened the literals into a separate buffer and merged forwards from it, which cost an extra buffer and copy. Two strategies exist, picked by exception density:

- **`merge_runs`**: moves each stretch of non-exception values with one `copy_within` (a vectorized memmove), interrupted only by a single write per exception. Wins when exceptions are rare: long runs, few interruptions.
- **`merge_walk`**: writes every output element individually through a raw pointer, with no bounds checks per element. slow5lib's C reference (`ex_depress`) works the same way, but forwards from a separate buffer. Loses to `merge_runs` on long runs (a scalar per-element loop can't beat a vectorized memmove) but wins once runs get short enough that `merge_runs`'s per-run call overhead dominates.

The crossover was found empirically (`patched::merge_density_sweep`, an `#[ignore]`d test — run with `cargo test --release --features simd-avx2 -- --ignored --nocapture merge_density_sweep`), timing both strategies directly against each other across a density sweep. With the 0.3.1 forward merges it landed at 14-15% exception density across n=128/1024/8192, and decode dispatched on `nex * 7 >= n`. With the in-place merges it moved to 18-20%: at 18% the two are within 5% of each other, and from 20% the walk is faster at every size. Decode now dispatches on `nex * 5 >= n` (20%), which is also the format's own encoder-side warning threshold ("compression may not be ideal").

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

## Slice API and the `Vec` wrappers

Since 0.4.0 the codec kernels write into caller-supplied slices
(`encode_to_slice`, `decode_to_slice`), and the `Vec` methods (`encode`,
`encode_into`, `decode`, `decode_into`) are wrappers around them. This covers
all five codecs.

### SIMD encode does not need padding

The C StreamVByte library adds 16 bytes of padding to its worst-case output
size, because the SIMD encoder writes a full 16-byte register per block even
when the block packs to fewer bytes. Before 0.4.0 the `svb` encoders reserved
the same 16 bytes of `Vec` capacity.

For `Svb16` the padding is not needed. Each block of 8 values is stored with
one 16-byte store at `data_pos`. The values before the block took at most 2
bytes each, so `data_pos <= 2 * i`, and the store ends at
`2 * i + 16 = 2 * (i + 8) <= 2 * n`. A store can write past the data encoded so
far, but never past `ctrl_len + 2 * n`, which is what `max_encoded_len` returns.
The AVX2 path does two such stores per 16 values and stays within the same
bound. A test (`test_util::encode_checked`) runs every encode kernel on a
slice of exactly `max_encoded_len` bytes followed by canary bytes, and fails if
any canary byte changes.

The same argument holds whenever a store covers the worst case for its block
(values per store x widest byte width >= store width), and it holds for every
kernel:

| Codec | Store | Covers | Ends at or before |
|---|---|---|---|
| `Svb16` | 16 bytes | 8 values x 2 bytes | `ctrl_len + 2 * n` |
| `U32Classic`, `U32Variant0124` | 16 bytes | 4 values x 4 bytes | `ctrl_len + 4 * n` |
| `U64Coder1234` | 16 bytes | 4 values x 4 bytes | `ctrl_len + 4 * n` |
| `U64Coder1248` | 16 bytes | 2 values x 8 bytes | `ctrl_len + 8 * n` |

The canary test covers the u32 and u64 kernels as well
(`crate::test_util::encode_checked`). An extra stray store past the encoded
data, which leaves the output unchanged, fails it.

### Cost of the `Vec` wrappers

The kernels take `&mut [T]`, so a `Vec` wrapper has to grow the vector with
`resize` before calling them, which zero-fills the new space. The kernel then
overwrites it. The old code wrote straight into spare capacity and called
`set_len`, with no zeroing. Writing into spare capacity from the new kernels
would need `&mut [MaybeUninit<T>]` kernels and `unsafe` (`set_len`, and a cast
from `&mut [T]`) outside the SIMD modules, which the crate does not allow. The
zero-fill is accepted as the cost of keeping that rule.

Measured on an Intel i7-11800H (AVX2), Rust 1.99, `simd-auto`, pinned to one
core, best of 14 timed runs. "main" is 0.3.1; "reuse" means the same `Vec` is
cleared and reused each call, so no allocation is timed.

| `Svb16` | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| decode, 0.3.1 `decode_into` (reuse) | 22.5-24.9 ns | 155-171 ns | 1219-1225 ns |
| decode, `decode_to_slice` | 22.5-27.7 ns | 163-164 ns | 1187-1220 ns |
| decode, `decode_into` (reuse) | 26.9-29.9 ns | 168-173 ns | 1250-1260 ns |
| encode, 0.3.1 `encode_into` (reuse) | 24.5-25.8 ns | 170-180 ns | 1305-1388 ns |
| encode, `encode_to_slice` | 25.9 ns | 168-169 ns | 1270 ns |
| encode, `encode_into` (reuse) | 29.8 ns | 184-185 ns | 1319-1337 ns |

Ranges are over the "small" (all 1-byte) and "large" (all 2-byte) inputs.

- The slice kernels run at the same speed as the 0.3.1 code.
- The `Vec` wrappers are about 4-5 ns slower per call. That is 15-20% at
  n=128, 2-9% at n=1024 and 2-3% at n=8192.
- The allocating `decode` builds its output with `Vec::new()` and
  `resize(n, 0)`. A first version used `vec![0; n]`, which allocates with
  `calloc`. glibc (2.35 here) serves `calloc` outside its per-thread cache, and
  in criterion that made `decode` at n=128 about 30-40 ns slower than 0.3.1
  (`Svb16` 41 to 80 ns, `U32Classic` 47 to 81 ns). `resize` allocates with
  `malloc` and then fills, which leaves only the zero-fill cost. Clippy's
  `slow_vector_initialization` lint assumes the opposite, so it is allowed on
  these functions.

### u32 and u64 codecs

Criterion, same machine, `simd-auto`, pinned to one core, runs of 0.3.1 and
0.4.0 alternated, median of 3 runs. Changes are 0.4.0 against 0.3.1.

| Benchmark | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `encode` | +4 to +10% | +2 to +4% | +3 to +10% |
| `decode_into` (reuse) | -1 to +8% | +7 to +10% | +3 to +6% |
| `decode` (allocating) | -7 to +12% | -4 to +11% | 0 to +6%, `U64Coder1234` +28% |

The `U64Coder1234` `decode` at n=8192 zero-fills 64 KB before decoding (1.97 to
2.52 us). `U64Coder1248` fills the same amount but its kernel is slower, and
there the change was within noise.

The slice functions on the same data, with a reused buffer:

| | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `U32Classic` `decode_to_slice` | 33.8 ns | 211 ns | 1.57 us |
| `U32Classic` `encode_to_slice` | 62.8 ns | 444 ns | 3.53 us |
| `U64Coder1248` `decode_to_slice` | 54.9 ns | 375 ns | 2.91 us |
| `U64Coder1248` `encode_to_slice` | 85.7 ns | 700 ns | 5.32 us |

Ratio against `streamvbyte64` 0.2.0 (both allocating a new output per call) in
the same runs, 0.4.0 with the `decode` fix:

| | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `U32Classic` decode | 2.3x | 2.3x | 2.7x |
| `U32Classic` encode | 2.2x | 1.8x | 1.8x |
| `U32Variant0124` decode | 2.4x | 2.3x | 2.7x |
| `U32Variant0124` encode | 2.2x | 1.8x | 1.8x |
| `U64Coder1248` decode | 1.5x | 1.5x | 1.5x |
| `U64Coder1248` encode | 1.9x | 1.6x | 1.6x |

In 0.3.1 the `U64Coder1248` rows were 1.4-1.7x for decode and 1.6-1.9x for
encode on this machine. These ratios are lower than the GitHub Actions table
above, which was measured on different hardware.

For the lowest per-call cost on small inputs, use the slice functions with a
buffer that is reused between calls.

### Delta and zigzag

Standalone timing program, same machine, pinned to one core, best of 9 timed
runs per process, two processes each for 0.3.1 and 0.4.0 run alternately.
`i16` data shaped like nanopore signal. "reuse" means a cleared `Vec` reused
each call.

| | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| delta `decode_into` (reuse), 0.3.1 | 22.1 ns | 245 ns | 2034 ns |
| delta `decode_into` (reuse), 0.4.0 | 24.0 ns | 271 ns | 2206 ns |
| delta `decode_to_slice` | 23.1 ns | 248 ns | 2029 ns |
| delta `decode_in_place` | 22.5 ns | 247 ns | 2027 ns |
| zigzag `decode_into` (reuse), 0.3.1 | 9.1 ns | 63.7 ns | 468 ns |
| zigzag `decode_into` (reuse), 0.4.0 | 8.4 ns | 63.3 ns | 464 ns |
| zigzag `decode_to_slice` | 8.8 ns | 63.5 ns | 461 ns |

- The delta kernels run at the same speed as 0.3.1, both into a separate
  slice and in place. Because a kernel's source and destination may be the
  same buffer, the compiler can no longer assume they do not overlap; this
  made no measurable difference.
- delta's `Vec` functions copy the deltas into the output and then decode in
  place, which costs 8-11%.
- zigzag's `Vec` functions use a plain `extend` with `decode_one`. The compiler
  vectorizes it and it matches the SSE2 kernel. An earlier version that
  zero-filled the output and called the SSE2 kernel was 10-40% slower than
  0.3.1 in criterion.
- Use `decode_vbz_fused` rather than these three steps for VBZ data.

The AVX2 delta kernels for `i32`/`u32` and `i64`/`u64` were missing
`#[target_feature(enable = "avx2")]` in 0.3.1 and earlier, so the intrinsics
were not inlined. Same program, `decode_into` with a reused `Vec`:

| | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `i32`, 0.3.1 | 719 ns | 5679 ns | 47722 ns |
| `i32`, 0.4.0 | 53.1 ns | 424 ns | 3729 ns |
| `i64`, 0.3.1 | 947 ns | 7489 ns | 61864 ns |
| `i64`, 0.4.0 | 103 ns | 827 ns | 7457 ns |

`decode_to_slice` for `i32` is 33.5 / 382 / 3231 ns at the same sizes.

### Fused SVB-ZD decode

Criterion, same machine, `simd-auto`, pinned to an idle core on an otherwise
moderately busy machine (load average 2-6 on 16 cores), runs of 0.3.1 and 0.4.0
alternated, median of 5 runs. `decode_svbzd_fused_into` with a reused `Vec`:

| | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `decode_svbzd_fused_into` (reuse), 0.3.1 | 75.0 ns | 464 ns | 3552 ns |
| `decode_svbzd_fused_into` (reuse), 0.4.0 | 78.1 ns | 484 ns | 3676 ns |
| `decode_svbzd_fused_to_slice` | 76.1 ns | 469 ns | 3581 ns |

The slice function is within 1-2% of 0.3.1, which is inside the run-to-run
spread here. The `Vec` wrapper is 3.5-4.5% slower than 0.3.1 because of the
zero-fill.

### Fused SVB-ZD encode

Standalone timing program, same machine, `simd-auto` (AVX2 kernel), pinned to
one core, best of 15 timed runs per process, three processes each for 0.3.1
and 0.4.0 run alternately. Ranges are over the three processes.

| | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `encode_svbzd_into` (reuse), 0.3.1 | 85.7-85.8 ns | 617-623 ns | 4925-4993 ns |
| `encode_svbzd_into` (reuse), 0.4.0 | 97.7-99.2 ns | 630-634 ns | 4925-4966 ns |
| `encode_svbzd_to_slice` | 80.0-82.7 ns | 607-612 ns | 4768-4805 ns |

- The slice function is 2-7% faster than 0.3.1. It skips the `reserve` and
  `set_len` calls that 0.3.1 made on every call, which may account for some of
  this; it was not looked into further.
- The `Vec` wrapper is 14-16% slower at n=128 and within 2% at the larger
  sizes. The cost is the zero-fill of `svbzd_max_encoded_len(n)` bytes.
- A first version wrote the control bytes with bounds-checked indexing
  (`ctrl[block] = cb`). That made the kernel 5-10% slower than 0.3.1 at
  n=1024 and n=8192, so the kernels write them through a raw pointer, as
  0.3.1 did.

### ex-zd and `patched`

`patched` (the exception layer under ex-zd) was converted to slices before the
ex-zd functions themselves. Criterion, same machine, `simd-auto`, pinned to one
core, runs of 0.3.1 and 0.4.0 alternated, median of 3 runs. Changes are 0.4.0
against 0.3.1; the ex-zd functions still have their 0.3.1 signatures here.
The spiky and real-read fused decode rows include a second run made after the
merge threshold moved to 20%; ranges cover both runs.

| Benchmark | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `encode_exzd` | -31% | -48% | -55% |
| `decode_exzd_fused_into` (reuse) | +3% | -3% | -4% |
| `decode_exzd_fused_into`, spiky data (reuse) | -3 to -6% | -2% | -3 to -4% |
| `decode_exzd` (3-pass) | +4% | +10% | +2% |

| Real reads (`tests/vectors`) | 2885-2949 samples | 101988 samples |
|---|---|---|
| `decode_exzd_fused` | -4 to +1% | +3 to +4% |
| `ExzdDecoder::decode_into` | -1 to -6% | -6% |
| `decode_exzd` (3-pass) | -2 to +1% | +10% |

`exzd_many_small_reads` (2000 reads decoded one after another) was 2-12%
faster, both with `ExzdDecoder` and allocating per call.

- Encode is faster because `patched::encode_into` used to push the literal
  bytes into the output one at a time and build the exception arrays in two
  temporary `Vec`s. It now sizes the output once and writes into it.
- A first version kept the 0.3.1 decode structure and zero-filled the literal
  scratch buffer and the zigzag-delta array on every call. That made decode
  4-20% slower (up to 23% on the 101988-sample read). The in-place merge
  (see "Adaptive exception-merge strategy" above) removed the literal buffer,
  and `ExzdDecoder` now only fills its zigzag-delta buffer where it grows.
- The one-shot decoders still allocate and zero-fill a new zigzag-delta array
  each call, which is the +3% on the large read. The 3-pass `decode_exzd` also
  goes through the `delta` `Vec` functions, which are 8-11% slower than 0.3.1
  (see "Delta and zigzag" above).

After the ex-zd functions were converted to slices (the `_into` functions are
now wrappers that zero-fill `out` and decode into it), same method, with the
fused and real-read rows rerun on a quiet machine:

| Benchmark | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `decode_exzd_fused_into` (reuse) | +11% | 0% | +3% |
| `decode_exzd_fused_to_slice` (one-shot) | 57.8 ns, -3% | 415 ns, -4% | 2393 ns, 0% |
| `ExzdDecoder::decode_to_slice` | 42.0 ns, -31% | 309 ns, -29% | 2177 ns, -9% |
| `decode_exzd` (3-pass) | -3% | -13% (0.3.1 varied 567-675 ns) | +3% |

The `_to_slice` percentages are against 0.3.1's `decode_exzd_fused_into` with a
reused `Vec`.

| Real reads (`tests/vectors`) | 2885-2949 samples | 101988 samples |
|---|---|---|
| `decode_exzd_fused` | +2 to +3% | +11.5% |
| `ExzdDecoder::decode_into` | 0 to +1% | +3% |
| `decode_exzd` (3-pass) | -1 to +1% | +7.5% |

`exzd_many_small_reads`: +14% with a reused `ExzdDecoder` at 64 samples per
read (about 4 ns per read), +4% allocating per call, and within 2% at 512
samples per read.

- The extra cost against the step before is the zero-fill of `out` in the
  `Vec` wrappers. On the 101988-sample read that is about 200 KB, roughly the
  4 us added to `decode_exzd_fused`.
- `ExzdDecoder::decode_to_slice` with a reused output buffer allocates nothing
  once its buffers have grown, and is the fastest way to decode many reads.

Encode after `encode_exzd_to_slice` was added, same method:

| Benchmark | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `encode_exzd` | -35% | -55% | -57% |
| `encode_exzd_into` (reused `Vec`) | -44% | -48% | -55% |
| `encode_exzd_to_slice` (reused buffer) | 164 ns | 1119 ns | 7858 ns |

`encode_exzd_to_slice` has no 0.3.1 equivalent. It runs within 3% of a reused
`encode_exzd_into`, because both spend most of their time in the qts, delta
and zigzag passes, which still allocate.

### VBZ encode and the 3-pass decoders

`encode_vbz_to_slice`, `decode_vbz_to_slice` and `decode_svbzd_to_slice` were
added last, and the matching `_into` functions became wrappers around them.
Same method as the ex-zd tables, load average 4-6 on 16 cores. Changes are
0.4.0 against 0.3.1.

| Benchmark | n=128 | n=1024 | n=8192 |
|---|---|---|---|
| `encode_vbz` | +3% | -2.5% | +2% |
| `encode_vbz_into` (reuse) | +5% | -7% | -1% |
| `encode_vbz_to_slice` (reuse) | 67.5 ns | 330 ns | 2522 ns |
| `decode_vbz` (3-pass) | -3% | 0% | +1.5% |
| `decode_vbz_into` (reuse) | -7% | -5% | +1% |
| `decode_vbz_to_slice` | 85.1 ns | 556 ns | 4041 ns |
| `decode_svbzd` (3-pass) | -9% | -15% | -16% |
| `decode_svbzd_into` (reuse) | -8% | -16% | -16% |
| `decode_svbzd_to_slice` | 153 ns | 985 ns | 7227 ns |

- `encode_vbz_into` (reuse) was 66.7 ns in 0.3.1 at n=128, so the slice
  function is about the same. The other VBZ rows are within the run-to-run
  spread.
- The 3-pass SVB-ZD decoder used to `push` each sample onto the `Vec`. It now
  writes into a slice, and the `Vec` wrapper zero-fills first, which is still
  faster than the `push` loop.
- `encode_vbzk` (k=4) is 8-16% faster than 0.3.1: 113 ns, 544 ns and 2995 ns
  against 123 ns, 602 ns and 3210-3747 ns. 0.3.1 encoded into a temporary
  `Vec` and copied it after the header; it now encodes into the output once.
  `encode_vbzk_to_slice` with a reused buffer takes 89 ns, 468 ns and 2742 ns.
  A first version allocated the output with `vec![0u8; len]`, which made
  `encode_vbzk` 16% slower than 0.3.1 at n=128; `Vec::new()` and `resize`
  fixed that, as it did for the other `Vec` wrappers.
- These are not the fast paths. `decode_vbz_fused_to_slice` is 1.3-1.6x
  faster than `decode_vbz_to_slice` (59, 346 and 3016 ns in the same run), and
  `decode_svbzd_fused_to_slice` is about 2x faster than
  `decode_svbzd_to_slice` (see "Fused SVB-ZD decode" above).

## Running benchmarks

```sh
cargo bench --features simd-auto
```

Benchmarks cover all five codec variants across encode/decode and three slice sizes (128, 1024, 8192 elements). Criterion produces HTML reports in `target/criterion/`.

To run a single benchmark by name substring:

```sh
cargo bench --features simd-auto -- U32Classic/decode
```
