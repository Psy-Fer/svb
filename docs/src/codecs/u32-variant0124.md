# U32Variant0124

`U32Variant0124` is an alternative `u32` codec where zero values consume no data bytes at all. The byte-width options are 0, 1, 2, or 4 — there is no 3-byte option.

Wire-compatible with the Lemire "0124" variant and the `streamvbyte64::Coder0124`.

## Tag table

| Tag | Byte width | Value range |
|---|---|---|
| 0 | 0 | 0 (exactly) |
| 1 | 1 | 1–255 |
| 2 | 2 | 256–65535 |
| 3 | 4 | 65536–4294967295 |

Note that values in the range 65536–16777215 require 4 bytes (not 3), which is worse than `U32Classic` for that range. The benefit comes from sparse data where many values are zero.

## Example

```rust
use svb::u32::U32Variant0124;

// Zero-valued elements cost 0 bytes in the data stream.
let values: Vec<u32> = vec![0, 0, 42, 0, 0, 255, 0];
let encoded = U32Variant0124.encode(&values);
let decoded = U32Variant0124.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

## When to use

Use `U32Variant0124` when a significant fraction of values are exactly zero — for example, sparse histograms, run-length-style data, or delta-encoded sorted lists where many differences are zero. For general data with few zeros, [`U32Classic`](u32-classic.md) is typically better because it can use 3 bytes for values in the 65536–16777215 range.
