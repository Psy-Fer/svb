# svb

Pure-Rust [StreamVByte](https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/) covering all major codec variants for `u16`, `u32`, and `u64` integers. Delta and zigzag encoding are composable layers on top. SIMD back-ends are available for x86-64 (SSSE3, AVX2) and AArch64 (NEON).

## Codec variants

| Type | Variant | Tag width | Byte widths | Notes |
|---|---|---|---|---|
| `u16` | `Svb16` | 1 bit | 1/2 | ONT VBZ format |
| `u32` | `U32Classic` | 2 bits | 1/2/3/4 | Lemire / C library compatible |
| `u32` | `U32Variant0124` | 2 bits | 0/1/2/4 | Better compression for sparse data |
| `u64` | `U64Coder1234` | 2 bits | 1/2/3/4 | As in `streamvbyte64` |
| `u64` | `U64Coder1248` | 2 bits | 1/2/4/8 | As in `streamvbyte64` |

## Feature flags

| Flag | Effect |
|---|---|
| `std` (default) | Enables `std`; implies `alloc` |
| `alloc` | Enables all encode/decode APIs with no other dependencies |
| `simd-auto` | Runtime CPU detection; selects the best available SIMD path |
| `simd-avx2` | Compile-time AVX2 (implies AVX2 is available at runtime) |
| `simd-ssse3` | Compile-time SSSE3 |
| `simd-neon` | Compile-time NEON (AArch64 only; NEON is always available there) |

For most users, `simd-auto` is the right choice. The compile-time flags (`simd-avx2`, `simd-ssse3`, `simd-neon`) are for environments where the target is known at build time, such as cross-compilation or `RUSTFLAGS="-C target-cpu=native"`.

```toml
[dependencies]
svb = { version = "0.1", features = ["simd-auto"] }
```

## Usage

### VBZ pipeline (Oxford Nanopore POD5 signal codec)

The VBZ codec chains delta → zigzag → SVB16. The outer zstd layer is left to the caller.

```rust
use svb::{encode_vbz, decode_vbz};

let samples: Vec<i16> = vec![100, 101, 103, 102, 98];

// Encode: i16 → delta → zigzag → SVB16 bytes
let encoded = encode_vbz(&samples);

// Decode: SVB16 bytes → zigzag → delta → i16
let decoded = decode_vbz(&encoded, samples.len()).unwrap();
assert_eq!(decoded, samples);
```

### SVB16

```rust
use svb::u16::Svb16;

let values: Vec<u16> = vec![1, 300, 0, 65000];
let encoded = Svb16.encode(&values);
let decoded = Svb16.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

### U32Classic

```rust
use svb::u32::U32Classic;

let values: Vec<u32> = vec![1, 500, 70_000, 16_000_000];
let encoded = U32Classic.encode(&values);
let decoded = U32Classic.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

### U32Variant0124: sparse data

```rust
use svb::u32::U32Variant0124;

// 0-valued elements cost 0 bytes in the data stream.
let values: Vec<u32> = vec![0, 0, 42, 0, 0, 255, 0];
let encoded = U32Variant0124.encode(&values);
let decoded = U32Variant0124.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

### U64Coder1234: values up to u32::MAX

`U64Coder1234` uses 1–4 byte widths (matching U32Classic). Values above `u32::MAX`
are silently truncated; call `check_range` first if the input may contain large values.

```rust
use svb::u64::U64Coder1234;

let values: Vec<u64> = vec![1, 500, 70_000, u32::MAX as u64];
assert_eq!(U64Coder1234.check_range(&values), None); // all fit in u32
let encoded = U64Coder1234.encode(&values);
assert_eq!(U64Coder1234.decode(&encoded, values.len()).unwrap(), values);
```

### U64Coder1248: full u64 range

`U64Coder1248` uses 1/2/4/8 byte widths, covering the full `u64` range without truncation.

```rust
use svb::u64::U64Coder1248;

let values: Vec<u64> = vec![1, 500, 1 << 32, u64::MAX];
let encoded = U64Coder1248.encode(&values);
assert_eq!(U64Coder1248.decode(&encoded, values.len()).unwrap(), values);
```

### Delta and zigzag as standalone transforms

```rust
use svb::{delta, zigzag};

// Delta-encode a sorted list, then zigzag for signed-friendly compression.
let values: Vec<i16> = vec![100, 105, 103, 110];
let deltas = delta::encode(&values);
let codes  = zigzag::encode(&deltas);
// ... encode codes with any u16 codec ...
```

### Appending to an existing buffer

Every codec exposes `encode_into` / `decode_into` variants that append to a caller-supplied `Vec`, avoiding allocation:

```rust
use svb::u32::U32Classic;

let mut buf = Vec::new();
U32Classic.encode_into(&[1u32, 2, 3], &mut buf);
U32Classic.encode_into(&[4u32, 5, 6], &mut buf);
```

## `no_std` support

`no_std` is useful whenever the Rust standard library is not available: microcontrollers and embedded targets, WebAssembly modules that need a minimal binary, or OS-level code (bootloaders, kernel modules). In those environments you still get allocations through a custom allocator, but not the full `std` runtime.

Disable the default `std` feature and enable `alloc`:

```toml
svb = { version = "0.1", default-features = false, features = ["alloc"] }
```

All encode/decode APIs are available. SIMD runtime detection (`simd-auto`) requires `std` (for `is_x86_feature_detected!`); use a compile-time SIMD flag instead if you need SIMD in a `no_std` context.

## Delta and zigzag

Both transforms are generic and work with all integer types used by the codecs.

`delta` is implemented for `i16`, `i32`, `i64`, `u32`, and `u64`. Use a signed type when the sequence is non-monotone and you intend to follow with zigzag; use an unsigned type for sorted/non-decreasing sequences where all differences are non-negative.

`zigzag` is implemented for `i16` (→`u16`), `i32` (→`u32`), and `i64` (→`u64`).

```rust
use svb::{delta, zigzag, u32::U32Classic, u64::U64Coder1248};

// Arbitrary i32 data: delta → zigzag → U32Classic
let samples: Vec<i32> = vec![-500, 200, -100, 900];
let deltas: Vec<i32> = delta::encode(&samples);
let codes: Vec<u32> = zigzag::encode(&deltas);
let encoded = U32Classic.encode(&codes);
let decoded_codes = U32Classic.decode(&encoded, codes.len()).unwrap();
let decoded_deltas: Vec<i32> = zigzag::decode(&decoded_codes);
let decoded: Vec<i32> = delta::decode(&decoded_deltas);
assert_eq!(decoded, samples);

// Sorted u64 timestamps: delta only (differences are always positive)
let timestamps: Vec<u64> = vec![1_000_000, 1_001_500, 1_003_000, 1_010_000];
let deltas: Vec<u64> = delta::encode(&timestamps);
let encoded = U64Coder1248.encode(&deltas);
let decoded_deltas = U64Coder1248.decode(&encoded, deltas.len()).unwrap();
let decoded: Vec<u64> = delta::decode(&decoded_deltas);
assert_eq!(decoded, timestamps);
```

## MSRV

The Minimum Supported Rust Version is **1.85** (edition 2024). This is the oldest Rust release guaranteed to compile this crate. Check your installed version with `rustup show`; update with `rustup update stable`.

## Performance

With `simd-auto` on a modern x86-64 machine, SVB16 and VBZ decode throughput is several GB/s. Run the included Criterion benchmarks to measure on your hardware:

```sh
cargo bench --features simd-auto
```

Benchmarks cover all five codec variants × encode/decode × three slice sizes (128, 1 024, 8 192 elements).

## License

MIT. See [LICENSE](LICENSE). Copyright 2026 James Ferguson.
