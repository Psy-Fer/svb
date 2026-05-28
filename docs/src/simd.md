# SIMD Backends

`svb` provides SIMD-accelerated encode and decode for all codec variants. The scalar path is always compiled and serves as the correctness reference.

## Available backends

| Backend | Feature flag | Architecture | ISA |
|---|---|---|---|
| SSSE3 | `simd-ssse3` | x86-64 | SSE2 + SSSE3 |
| AVX2 | `simd-avx2` | x86-64 | AVX2 |
| NEON | `simd-neon` | AArch64 | NEON |
| Auto | `simd-auto` | both | runtime detection |

### simd-auto

`simd-auto` detects the best available path at runtime using `is_x86_feature_detected!` on x86-64 and unconditional NEON on AArch64. This is the recommended flag for most users.

On x86-64, `simd-auto` selects AVX2 if available, then SSSE3, then scalar. On AArch64, NEON is always selected (NEON is mandatory on AArch64).

`simd-auto` requires `std` for runtime CPU detection. In `no_std` contexts, use a compile-time flag instead.

### Compile-time flags

`simd-avx2`, `simd-ssse3`, and `simd-neon` compile in the SIMD path and assume it is available at runtime. These are appropriate when the target CPU is known:

```toml
# Cross-compile to a known AVX2 target
svb = { version = "0.2", features = ["simd-avx2"] }
```

or with `RUSTFLAGS="-C target-cpu=native"` where the build host and run host are the same.

## Pipeline coverage

SIMD paths are provided for individual codec variants and for both high-level pipelines:

- **VBZ pipeline** (`encode_vbz` / `decode_vbz_fused`): fused SVB16 + zigzag + delta in a single SIMD loop on x86-64 (SSSE3/AVX2) and AArch64 (NEON).
- **SVB-ZD pipeline** (`encode_svbzd` / `decode_svbzd_fused`): fused U32Classic + unzigzag + undelta. Encode computes zigzag-delta inline via SIMD (eliminates the intermediate `Vec<u32>` allocation), decode collapses all three stages into one SIMD loop.

## Decode throughput

With `simd-auto` on a modern x86-64 machine, decode throughput for all codec variants is in the range of **1.3–4 GB/s** depending on variant and input size. See [Performance](performance.md) for detailed numbers.
