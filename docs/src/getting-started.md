# Getting Started

## Installation

Add `svb` to your `Cargo.toml`. For most users `simd-auto` is the right choice — it detects the best available SIMD path at runtime:

```toml
[dependencies]
svb = { version = "0.1", features = ["simd-auto"] }
```

## Feature flags

| Flag | Effect |
|---|---|
| `std` (default) | Enables `std`; implies `alloc` |
| `alloc` | Enables all encode/decode APIs with no other dependencies |
| `simd-auto` | Runtime CPU detection; selects the best available SIMD path |
| `simd-avx2` | Compile-time AVX2 (asserts AVX2 is available at runtime) |
| `simd-ssse3` | Compile-time SSSE3 |
| `simd-neon` | Compile-time NEON (AArch64 only; NEON is always available there) |

The compile-time SIMD flags (`simd-avx2`, `simd-ssse3`, `simd-neon`) are intended for environments where the target CPU is known at build time, such as cross-compilation or `RUSTFLAGS="-C target-cpu=native"`. In all other cases, prefer `simd-auto`.

## Basic usage

Every codec is a zero-sized type with `encode` and `decode` methods. `encode` returns a `Vec<u8>`; `decode` takes the byte slice and the original element count.

```rust
use svb::u32::U32Classic;

let values: Vec<u32> = vec![1, 500, 70_000, 16_000_000];
let encoded = U32Classic.encode(&values);
let decoded = U32Classic.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

### Appending to an existing buffer

Every codec exposes `encode_into` and `decode_into` variants that append to a caller-supplied `Vec`, avoiding extra allocation:

```rust
use svb::u32::U32Classic;

let mut buf = Vec::new();
U32Classic.encode_into(&[1u32, 2, 3], &mut buf);
U32Classic.encode_into(&[4u32, 5, 6], &mut buf);
```

This is useful when building a larger serialised format where multiple compressed sequences are concatenated. The caller is responsible for recording the element counts needed for decode.

## Choosing a codec

- **`u16` data (e.g. ONT signal)**: use [`Svb16`](codecs/svb16.md), or the higher-level [`encode_vbz`/`decode_vbz`](vbz.md) pipeline.
- **`u32` general**: use [`U32Classic`](codecs/u32-classic.md). Wire-compatible with the Lemire C library.
- **`u32` with many zeros**: use [`U32Variant0124`](codecs/u32-variant0124.md). Zero values consume no data bytes.
- **`u64` values that fit in `u32::MAX`**: use [`U64Coder1234`](codecs/u64-coder1234.md).
- **`u64` full range**: use [`U64Coder1248`](codecs/u64-coder1248.md).

For sorted or time-series data, compose any codec with [delta encoding](transforms.md) to compress differences rather than raw values.
