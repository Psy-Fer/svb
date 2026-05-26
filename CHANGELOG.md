# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
