# no_std Support

`svb` supports `no_std` environments with a global allocator. This covers microcontrollers and embedded targets, WebAssembly modules, and OS-level code such as bootloaders or kernel modules.

## Setup

Disable the default `std` feature and enable `alloc`:

```toml
svb = { version = "0.2", default-features = false, features = ["alloc"] }
```

All encode and decode APIs are available. The [delta](transforms.md) and [zigzag](transforms.md) transforms are also fully available.

## SIMD in no_std

Runtime SIMD detection (`simd-auto`) requires `std` for `is_x86_feature_detected!`. In `no_std` contexts, use a compile-time SIMD flag instead:

```toml
# no_std with compile-time NEON (AArch64 embedded target)
svb = { version = "0.2", default-features = false, features = ["alloc", "simd-neon"] }
```

```toml
# no_std with compile-time AVX2
svb = { version = "0.2", default-features = false, features = ["alloc", "simd-avx2"] }
```
