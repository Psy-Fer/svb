# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Slice API for every codec (issue #4): `max_encoded_len`, `encode_to_slice` and `decode_to_slice` on `Svb16`, `U32Classic`, `U32Variant0124`, `U64Coder1234` and `U64Coder1248` encode into and decode from caller-supplied slices, so the output can live in any storage type. The `Vec` methods are now wrappers around them. `decode_to_slice` returns the number of bytes the encoded values took up and does not read past them, so the values can be followed by other data; compare the count with `data.len()` to reject extra bytes.
- `EncodeError::OutputTooSmall`, returned by `encode_to_slice` when the output slice is shorter than `max_encoded_len`.
- `decode_vbz_fused_to_slice` and `decode_vbz_fused_from_to_slice` decode VBZ into a caller-supplied `&mut [i16]`, with `n = out.len()`. VBZ-K (experimental) gains `decode_vbzk_to_slice` and `decode_vbzk_parallel_to_slice`; the parallel version has each thread decode straight into its own part of `out`.
- `decode_svbzd_fused_to_slice` and `decode_svbzd_fused_from_to_slice` do the same for SVB-ZD.
- `decode_exzd_to_slice`, `decode_exzd_fused_to_slice` and `ExzdDecoder::decode_to_slice` decode ex-zd into a caller-supplied `&mut [i16]`. `out.len()` must equal the sample count stored in the frame, which `exzd_decoded_len` reads from the header.
- `encode_vbz_to_slice` encodes VBZ into a caller-supplied `&mut [u8]` and returns the number of bytes written, with `vbz_max_encoded_len` for sizing the buffer. The delta and zigzag passes still allocate temporary buffers.
- `decode_vbz_to_slice` and `decode_svbzd_to_slice` are slice versions of the 3-pass decoders, with `n = out.len()`. `decode_vbz_into` and `decode_svbzd_into` are now wrappers around them.
- `encode_vbzk_to_slice` and `vbzk_max_encoded_len` for the experimental VBZ-K format. `encode_vbzk` is now a wrapper around them and is 8-16% faster, because it no longer encodes into a temporary `Vec` and copies it after the header.
- `encode_exzd_to_slice` encodes ex-zd into a caller-supplied `&mut [u8]` and returns the number of bytes written, with `exzd_max_encoded_len` for sizing the buffer. The bound assumes every value is an exception (about 8.5 bytes per sample), so `encode_exzd_into` still sizes its output from the actual exception count. Both still allocate temporary buffers for the qts, delta and zigzag passes.
- `DecodeError::OutputLengthMismatch`, returned by the ex-zd `_to_slice` decoders when `out.len()` does not match the frame. Code that matches exhaustively on `DecodeError` needs a new arm.
- `patched::encode_to_slice`, `patched::max_encoded_len` and `patched::decode_to_slice`. `max_encoded_len` assumes every value is an exception; `patched::encode_into` sizes its output from the actual exception count instead.
- `encode_svbzd_to_slice` encodes SVB-ZD into a caller-supplied `&mut [u8]` and returns the number of bytes written, with `svbzd_max_encoded_len` for sizing the buffer. `encode_svbzd_into` is now a wrapper around it.
- Fuzz targets `decode_vbzk`, `vbzk_roundtrip`, `patched_decode` and `patched_roundtrip`. The existing decode targets now also check that the slice, `Vec` and `_into` functions agree, and that the 3-pass and fused decoders agree; the round-trip targets check `encode_to_slice` against `max_encoded_len`.
- Slice functions for the delta and zigzag layers: `delta::encode_to_slice`, `delta::decode_to_slice` and their `_with_initial` forms, `delta::decode_in_place` / `decode_with_initial_in_place` (decode a slice of deltas without a second buffer), `delta::decode_2chain_to_slice`, `zigzag::encode_to_slice` and `zigzag::decode_to_slice`. The output must be the same length as the input; they panic otherwise.
- Builds without the `alloc` feature (`default-features = false`) now provide the slice API: every codec's `max_encoded_len`, `encode_to_slice` and `decode_to_slice`, the `delta` and `zigzag` slice functions, `decode_vbz_fused_to_slice`, `decode_vbz_fused_from_to_slice`, `decode_vbzk_to_slice`, `encode_svbzd_to_slice`, `svbzd_max_encoded_len`, `decode_svbzd_fused_to_slice` and `decode_svbzd_fused_from_to_slice`. Before, nothing but the error types was available without `alloc`. These functions never allocate in any build. Everything else still needs `alloc`.

### Changed

- `DecodeError` is now `#[non_exhaustive]`, so a `match` on it needs a `_` arm. Later releases can then add variants without breaking code.
- The pipeline decoders (VBZ, VBZ-K, SVB-ZD and ex-zd, every `Vec`, `_into`, `_to_slice` and `_from` form) now expect `data` to be exactly one frame, and return the new `DecodeError::TrailingData { used, len }` if there are bytes after it. A corrupted control stream that claims fewer bytes than were written used to decode as `Ok` with wrong samples and now gives this error. VBZ-K sub-streams and the two `patched` exception arrays inside an ex-zd frame must also use exactly the length stored for them. A VBZ-K frame for zero samples is now checked like any other, so empty input is rejected; `encode_vbzk` writes one byte for it. To decode a frame from inside a larger buffer, slice it out first. pod5-rs and slow5lib-rs already pass exactly one frame, and every read in the POD5 and BLOW5 files tested decodes unchanged.
- `encode_exzd` and `patched::encode_into` are 31-55% faster. The literal bytes were pushed into the output one at a time, and the exception positions went through two temporary `Vec`s.
- `patched` decode merges the literals and exceptions in place, working from the back of the output, instead of widening the literals into a separate buffer first. The density at which decode switches from the run-copy merge to the per-element merge moved from about 14% to 20% exceptions, re-measured with `merge_density_sweep`. ex-zd decode is 1-12% faster than 0.3.1 on most benchmarks; see "ex-zd and `patched`" in `docs/src/performance.md`.
- `decode_into` on every codec, `decode_vbz_fused_into`, `decode_vbz_fused_from_into`, `decode_vbzk_into`, `decode_vbzk_parallel_into`, `decode_svbzd_fused_into` and `decode_svbzd_fused_from_into` now leave `out` at its original length on error. Previously they kept any values decoded before the error.
- The `Vec` methods zero-fill the output before the codec writes into it, which costs about 4-5 ns per call for `Svb16` (15-20% at 128 values, 2-3% at 8192) and 2-12% for the u32 and u64 codecs in most benchmarks. The largest case is `U64Coder1234::decode` at 8192 values, which zero-fills 64 KB and is 28% slower. `decode_svbzd_fused_into` and `decode_svbzd_fused_from_into` are 3.5-4.5% slower for the same reason, and `encode_svbzd_into` is 14-16% slower at 128 samples and within 2% at 1024 and 8192. `decode_svbzd` and `decode_svbzd_into` (3-pass) are 8-16% faster, because they no longer push one sample at a time. `decode_exzd_fused_into` is 11% slower at 128 samples and on a 101988-sample read, and a reused `ExzdDecoder::decode_into` is about 4 ns per call slower; `ExzdDecoder::decode_to_slice` is 9-31% faster than 0.3.1. The slice functions have no such cost. See the "Slice API" section of `docs/src/performance.md`.
- `delta::decode`, `decode_into` and the other `Vec` decode functions copy the deltas into the output and decode them in place, which makes them about 8-11% slower than 0.3.1 for `i16`. `zigzag::decode` and `decode_into` are now a plain loop that the compiler vectorizes, measured as fast as the 0.3.1 SSE2 path.
- The SIMD encoders, including the fused SVB-ZD encoders, no longer reserve 16 bytes of padding. Their stores never go past `max_encoded_len` (`svbzd_max_encoded_len` for SVB-ZD), and a canary test checks this for every encode kernel.

### Removed

- `encode_vbz2`, `decode_vbz2` and `decode_vbz2_into`, deprecated since 0.1.0. Use VBZ-K (`encode_vbzk`, `decode_vbzk_*`) for parallel decode. The interleaved 2-chain decoder they used is gone too; `delta::decode_2chain` and `delta::mid_carry` remain.

### Fixed

- Building with `simd-auto` and without `std` (`--no-default-features --features alloc,simd-auto`) failed on x86_64, because the VBZ and SVB-ZD fused decoders called `is_x86_feature_detected!` without checking for `std`. They now fall back to scalar in that configuration, as the other decoders do. CI now tests this configuration.
- `encode_vbzk` wrote `k` into its one-byte header without checking that it fits. With `k = 256` and at least 2048 samples the header said `k = 0` and the frame could not be decoded; with `k = 257` it said `k = 1` and decoding returned the wrong samples without an error. `k` is now capped at 255, in the same way the encoder already reduces `k` to 1 when there are too few samples.
- The AVX2 decoders of `Svb16`, `U32Classic`, `U32Variant0124`, `U64Coder1234` and `U64Coder1248` returned `DataTruncated` when a valid frame was followed by more than about 64 extra bytes (96 for one `U64` kernel; the limit includes the last few bytes of the frame). The scalar, SSSE3 and NEON decoders ignore extra bytes, and so do the AVX2 decoders now. Through `Svb16` and `U32Classic` this also affected the 3-pass `decode_vbz` and `decode_svbzd` with `simd-auto` on AVX2 CPUs. The fused decoders were not affected. Present since 0.3.0; found by fuzzing the 3-pass and fused decoders against each other.
- `decode` on every codec, `decode_vbz`, `decode_vbz_fused`, `decode_vbz_fused_from`, `decode_svbzd`, `decode_svbzd_fused` and `decode_svbzd_fused_from` reserved space for `n` values before looking at the input, so a corrupted or untrusted `n` could make them try to allocate far more memory than the input could hold. They now check that the input holds at least the control stream for `n` values first, which limits the allocation to a few values per input byte. The `decode_into` functions, which already made this check in 0.3.1, still do.
- `decode_vbzk`, `decode_vbzk_into` and the parallel VBZ-K decoders panicked on a header whose data offsets pointed past the end of the data or went backwards. The offsets are now checked and these return `DecodeError::DataTruncated`.
- `DataTruncated` indices from VBZ-K decode are now relative to the whole output instead of the sub-stream.
- The AVX2 delta decode kernels (used for `i32`, `u32`, `i64` and `u64` under `simd-auto` and `simd-avx2`) were compiled without `#[target_feature(enable = "avx2")]`, so every AVX2 intrinsic was an out-of-line function call. `delta::decode` for `i32` ran at about 5.5 ns per value. With the attribute it is 8-13x faster for `i32` and `i64`. `i16`/`u16` delta, which uses SSE2, and the VBZ, SVB-ZD and ex-zd pipelines were not affected.
- `DecodeError::DataTruncated { index }` from the SSSE3, AVX2 and NEON decoders of every codec now reports the index in the whole output when the error happens in the scalar tail. Previously it reported the index within the tail.

## [0.3.1] - 2026-09-02

### Changed

- Removed the `thiserror` dependency; `DecodeError`'s `Display` and `Error` impls are now hand-written against `core::error::Error`. Error messages and public API are unchanged. The crate now has zero runtime dependencies, and `std` no longer forwards a feature to a third-party crate (`std = ["alloc"]`).

## [0.3.0] - 2026-07-09

### Added

- ex-zd pipeline: `encode_exzd` / `encode_exzd_into` i16 signal → qts (quantize-trailing-shift) → zigzag-delta (u16 domain) → PFOR-style patched/exception encoding; wire-compatible with hasindu2008/slow5lib `SLOW5_COMPRESS_EX_ZD` (BLOW5 files)
- `decode_exzd` / `decode_exzd_into` 3-pass ex-zd decode
- `decode_exzd_fused` / `decode_exzd_fused_into` single SIMD pass fusing inverse-zigzag + delta prefix sum + qts left-shift (`exzd_fused.rs`), AVX2/SSE2/NEON
- `ExzdDecoder` reusable-scratch decoder — avoids a fresh heap allocation per call when decoding many small BLOW5 reads; 1.24-1.79× faster than the per-call-allocating API for that access pattern
- `quantize` module: `find_qts` / `apply_shift` / `unshift_inplace`, the qts pre-pass specific to ex-zd
- `patched` module: PFOR-style patched/exception codec (`encode_into` / `decode_into`) over `&[u16]`, generically useful beyond ex-zd; adaptive merge strategy picks between a run-based (`extend_from_slice`) and a raw-pointer walk merge based on exception density, threshold found empirically at ~14.3% (`patched::merge_density_sweep`, an `#[ignore]`d diagnostic test)
- SIMD-accelerated literal-byte widening in `patched::widen_into` (SSE2/NEON)
- Real C-encoded and real-ONT-signal wire-compatibility fixtures in `tests/parity.rs` / `tests/vectors/` (`exzd_*`, `exzd_pod5_*`), verified byte-exact against slow5lib's `slow5_ptr_compress_solo`/`slow5_ptr_depress_solo`
- `docs/src/exzd.md` ex-zd pipeline documentation page; ex-zd performance section in `docs/src/performance.md` (including a from-scratch comparison against the slow5lib C reference on both synthetic and real nanopore data)
- `bench_exzd_*` criterion benchmarks in `benches/decode.rs`, including a real-ONT-read benchmark group (`bench_exzd_real_reads`) alongside the synthetic ones
- Fuzz targets `decode_exzd`, `exzd_roundtrip`, `decode_svbzd`, `svbzd_roundtrip` (`fuzz/fuzz_targets/`), closing the last codec variants without decode/roundtrip fuzz coverage

### Fixed

- **Malformed-input panics across every SIMD decode backend.** Adding the fuzz targets above surfaced a systemic bug: the "padded tail" loop present in every AVX2/SSE2/NEON decode path (`u16`, `u32` Classic/Variant0124, `u64` Coder1234/Coder1248, plus the fused decoders in `vbz_fused.rs` and `svbzd_fused.rs`) advanced its read position using a ctrl-byte-derived width without validating it against the real remaining bytes. Truncated or corrupted input (a mismatched `n`, or a corrupted length header reaching `decode_svbzd`/`decode_exzd`) could push the read position out of bounds, panicking instead of returning a `DecodeError`. All ~21 call sites now validate the remaining-byte bound before reading and return `DecodeError::DataTruncated` instead of panicking. Confirmed fixed by re-running the full fuzz corpus (including the pre-existing `u32_classic_decode`, `u32_variant0124_decode`, `u64_coder1234_decode`, `u64_coder1248_decode`, `decode_vbz`, `svb16_decode` targets, which had never been run to completion before and crashed on the same pattern) with zero crashes.
- **`decode_exzd` capacity-overflow panic on a crafted frame header.** The frame's `nin` (sample count) field was passed directly to `Vec::reserve` with no sanity check; a corrupted or malicious header with an implausibly large `nin` panicked with "capacity overflow" instead of returning `DecodeError::ControlStreamTooShort`. Fixed by bounding `nin` against the actual remaining buffer length before reserving.

## [0.2.0] - 2026-05-26

### Added

- SVB-ZD pipeline: `encode_svbzd` / `encode_svbzd_into` i16 signal → widen to i32 → fused zigzag-delta → U32Classic; wire-compatible with hasindu2008/slow5lib `SLOW5_COMPRESS_SVB_ZD`(BLOW5 files)
- `decode_svbzd` / `decode_svbzd_into` 3-pass SVB-ZD decode
- `decode_svbzd_fused` / `decode_svbzd_fused_into` single SIMD pass fusing U32Classic decode + unzigzag + undelta
- `decode_svbzd_fused_from` / `decode_svbzd_fused_from_into` fused decode with caller-supplied initial carry; building block for parallel decode from any split point
- SIMD fused encode for SVB-ZD on AVX2 (8 i16/iter), SSSE3 (4 i16/iter), and NEON (4 i16/iter), computes zigzag-delta inline, eliminating the intermediate `Vec<u32>` allocation; up to 5.4× faster than scalar encode on AVX2
- 2-ctrl-byte inner decode loop for SVB-ZD (SSSE3 and NEON paths), processes 8 i16 values per iteration; up to 4× faster than 3-pass scalar decode
- Benchmark workflow (`.github/workflows/bench.yml`) manual `workflow_dispatch` trigger; compares scalar / SSSE3 / AVX2 on x86-64 and scalar / NEON on AArch64; results posted as GitHub Step Summary with Melem/s and speedup ratios
- `scripts/bench_summary.py` criterion bencher-output parser used by the benchmark workflow
- `docs/src/svbzd.md` SVB-ZD pipeline documentation page covering API, wire format, and parallel decode with `fused_from` - SVB-ZD entries added to wire-compatibility table, SIMD backends page, and performance page

### Changed

- MSRV bumped from 1.85 to 1.87, required by RFC 2800 (`target_feature_11`): SIMD intrinsics are now safe to call inside `#[target_feature]` functions without explicit `unsafe {}` blocks

## [0.1.0] - 2026-05-22

### Added

- `Svb16` codec (1-bit tags, 1/2-byte widths) — wire-compatible with the ONT VBZ format
- `U32Classic` codec (2-bit tags, 1/2/3/4-byte widths) — compatible with Lemire's reference C library
- `U32Variant0124` codec (2-bit tags, 0/1/2/4-byte widths) — better compression for sparse data
- `U64Coder1234` codec (2-bit tags, 1/2/3/4-byte widths) — u64 values up to `u32::MAX`
- `U64Coder1248` codec (2-bit tags, 1/2/4/8-byte widths) — full u64 range
- `delta` module: composable delta encode/decode with `mid_carry` building block
- `zigzag` module: composable zigzag encode/decode
- VBZ pipeline (`encode_vbz`, `decode_vbz`, `decode_vbz_into`) — ONT POD5 signal codec
- `decode_vbz_fused` / `decode_vbz_fused_into` — single SIMD pass fusing SVB16 + zigzag + delta (~1.47× faster than 3-pass)
- `decode_vbz_fused_from` / `decode_vbz_fused_from_into` — fused decode with caller-supplied initial carry; building block for parallel decode
- `encode_vbzk` / `decode_vbzk` / `decode_vbzk_parallel_into` — **experimental** K-stream parallel decode format
- SIMD back-ends: AVX2 and SSSE3 (x86-64), NEON (AArch64), scalar fallback
- `simd-auto` feature for runtime CPU detection; `simd-avx2`, `simd-ssse3`, `simd-neon` for compile-time selection
- `no_std + alloc` support (all codec functionality; parallel decode requires `std`)
- Binary test vectors for cross-implementation parity checks

### Deprecated

- `encode_vbz2` / `decode_vbz2` / `decode_vbz2_into` — superseded by the generalised VBZ-K format; will be removed in a future release
