# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
