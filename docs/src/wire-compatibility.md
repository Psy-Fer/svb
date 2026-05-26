# Wire Compatibility

`svb` is wire-compatible with the reference C implementations and the `streamvbyte64` Rust crate. This means a buffer encoded by `svb` can be decoded by the C library and vice versa.

## Compatibility table

| svb variant | Compatible with |
|---|---|
| `U32Classic` | Lemire C `streamvbyte` library, `streamvbyte64::Coder1234` |
| `U32Variant0124` | Lemire C "0124" variant, `streamvbyte64::Coder0124` |
| `U64Coder1234` | `streamvbyte64::Coder1234` (u32 values only) |
| `U64Coder1248` | `streamvbyte64::Coder1248` |
| `Svb16` | ONT `vbz_hdf_plugin` SVB16 layer |
| SVB-ZD pipeline (`encode_svbzd` / `decode_svbzd_fused`) | hasindu2008/slow5lib `SLOW5_COMPRESS_SVB_ZD` (BLOW5 files) |

## Buffer layout difference

`streamvbyte64` keeps tags and data in separate buffers. `svb` concatenates them (tags first). When exchanging data with `streamvbyte64`, split or join buffers at the control stream boundary:

```rust
// svb flat → streamvbyte64 separate buffers
fn split_flat(encoded: &[u8], n: usize) -> (&[u8], &[u8]) {
    let ctrl_len = n.div_ceil(4);
    (&encoded[..ctrl_len], &encoded[ctrl_len..])
}

// streamvbyte64 separate buffers → svb flat
fn join_flat(tags: &[u8], data: &[u8]) -> Vec<u8> {
    let mut flat = tags.to_vec();
    flat.extend_from_slice(data);
    flat
}
```

## Verification

Wire compatibility is verified in `tests/compat.rs` by round-tripping data in both directions: svb encodes and `streamvbyte64` decodes, then `streamvbyte64` encodes and svb decodes. These tests run in CI for all four compatible codec pairs.
