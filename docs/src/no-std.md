# no_std Support

`svb` works in `no_std` builds, with or without a global allocator. This covers microcontrollers and embedded targets, WebAssembly modules, and OS-level code such as bootloaders or kernel modules.

## Without an allocator

Disable the default features:

```toml
svb = { version = "0.4", default-features = false }
```

This gives the functions that work on caller-supplied slices:

- every codec's `max_encoded_len`, `encode_to_slice` and `decode_to_slice`;
- the `delta` and `zigzag` `_to_slice` functions, and `delta::decode_in_place`;
- `decode_vbz_fused_to_slice` and `decode_vbz_fused_from_to_slice`;
- `encode_svbzd_to_slice`, `svbzd_max_encoded_len`, `decode_svbzd_fused_to_slice` and `decode_svbzd_fused_from_to_slice`.

These functions never allocate, in any build, so they are also the ones to use when you want to control every allocation in a `std` program. The output can be a stack array, a static buffer or memory you manage yourself:

```rust
use svb::u32::U32Classic;

let values = [1u32, 300, 70_000, 5];
let mut buf = [0u8; U32Classic.max_encoded_len(4)];
let len = U32Classic.encode_to_slice(&values, &mut buf).unwrap();

let mut out = [0u32; 4];
let used = U32Classic.decode_to_slice(&buf[..len], &mut out).unwrap();
assert_eq!(used, len);
assert_eq!(out, values);
```

SVB-ZD round trip into fixed-size buffers:

```rust
let samples = [100i16, 102, 99, 99, 250, -3];
let mut buf = [0u8; svb::svbzd_max_encoded_len(6)];
let len = svb::encode_svbzd_to_slice(&samples, &mut buf).unwrap();

let mut out = [0i16; 6];
svb::decode_svbzd_fused_to_slice(&buf[..len], &mut out).unwrap();
assert_eq!(out, samples);
```

The rest of the API needs the `alloc` feature:

- the `Vec` functions (`encode`, `encode_into`, `decode`, `decode_into` and the pipeline forms that return or append to a `Vec`);
- the VBZ encoder and the 3-pass `decode_vbz_to_slice` and `decode_svbzd_to_slice`, which use temporary buffers;
- VBZ-K;
- ex-zd, `patched` and `quantize`, which need buffers whose size depends on the data.

## With an allocator

Enable `alloc` to get the whole API except the parallel VBZ-K decoder, which needs `std` for threads:

```toml
svb = { version = "0.4", default-features = false, features = ["alloc"] }
```

## SIMD in no_std

Runtime SIMD detection (`simd-auto`) requires `std` for `is_x86_feature_detected!`. Without `std` it falls back to scalar on x86_64 (on AArch64 it still uses NEON). In `no_std` builds, use a compile-time SIMD flag instead:

```toml
# no_std with compile-time NEON (AArch64 embedded target)
svb = { version = "0.4", default-features = false, features = ["simd-neon"] }
```

```toml
# no_std with compile-time AVX2, built with RUSTFLAGS="-C target-feature=+avx2"
svb = { version = "0.4", default-features = false, features = ["simd-avx2"] }
```

Add `"alloc"` to the features list in either case if you need the rest of the API.
