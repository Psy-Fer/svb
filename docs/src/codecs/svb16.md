# SVB16

`Svb16` compresses `u16` values using 1-bit tags. Each value is stored in either 1 byte (values 0–255) or 2 bytes (values 256–65535). Eight tags share one control byte.

This is the codec used in the [VBZ pipeline](../vbz.md) for Oxford Nanopore POD5 signal data.

## Tag table

| Tag | Byte width | Value range |
|---|---|---|
| 0 | 1 | 0–255 |
| 1 | 2 | 256–65535 |

## Example

```rust
use svb::u16::Svb16;

let values: Vec<u16> = vec![1, 300, 0, 65000];
let encoded = Svb16.encode(&values);
let decoded = Svb16.decode(&encoded, values.len()).unwrap();
assert_eq!(decoded, values);
```

## Control stream layout

Tags are packed 8 per byte, LSB-first. For `n` values the control stream is `ceil(n / 8)` bytes.

```
control byte k  →  tags for values 8k+0 through 8k+7
bit 0  =  tag for value 8k+0
bit 1  =  tag for value 8k+1
...
bit 7  =  tag for value 8k+7
```
