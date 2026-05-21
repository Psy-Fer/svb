# U32Classic

`U32Classic` is the original Lemire StreamVByte variant for `u32` values. Each value is stored in 1–4 bytes depending on its magnitude.

Wire-compatible with the [Lemire C library](https://github.com/lemire/streamvbyte) and the `stream-vbyte` crate.

## Tag table

| Tag | Byte width | Value range |
|---|---|---|
| 0 | 1 | 0–255 |
| 1 | 2 | 256–65535 |
| 2 | 3 | 65536–16777215 |
| 3 | 4 | 16777216–4294967295 |

## Example

```rust
use svb::u32::U32Classic;

let values: Vec<u32> = vec![1, 500, 70_000, 16_000_000];
let encoded = U32Classic.encode(&values);
let decoded = U32Classic.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

## Wire format example

Encoding `[1, 256, 65536, 0xFFFFFFFF]` produces:

```
byte 0:     0xE4        control byte (tags: 0, 1, 2, 3 packed LSB-first → 0b11_10_01_00)
bytes 1:    0x01        value 0: 1 (1 byte)
bytes 2-3:  0x00 0x01   value 1: 256 (2 bytes, little-endian)
bytes 4-6:  0x00 0x00 0x01   value 2: 65536 (3 bytes)
bytes 7-10: 0xFF 0xFF 0xFF 0xFF  value 3: 4294967295 (4 bytes)
```

## When to use

`U32Classic` is the right default for general `u32` compression and any context where wire compatibility with the C library matters. For data with many zero or small values, [`U32Variant0124`](u32-variant0124.md) compresses better.
