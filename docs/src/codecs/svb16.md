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

## Encoding into your own buffer

`encode_to_slice` and `decode_to_slice` encode into and decode from
caller-supplied slices. For `Svb16`, `max_encoded_len(n)` is
`ceil(n / 8) + 2 * n`. See
[Encoding into your own buffer](index.md#encoding-into-your-own-buffer) for the
rules that apply to every codec.

```rust
use svb::u16::Svb16;

let values = [1u16, 300, 0, 65000];
let mut buf = vec![0u8; Svb16.max_encoded_len(values.len())];
let len = Svb16.encode_to_slice(&values, &mut buf).unwrap();

let mut out = [0u16; 4];
Svb16.decode_to_slice(&buf[..len], &mut out).unwrap();
assert_eq!(out, values);
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
