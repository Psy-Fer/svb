# Delta and Zigzag

Delta and zigzag are independent slice transforms. They are not tied to any specific codec; you compose them with whichever codec suits your data.

## Delta encoding

Delta encoding replaces each value with the difference from the previous value. This is effective for sorted or slowly-varying data: the differences are small even when the raw values are large.

```rust
use svb::{delta, u64::U64Coder1248};

// Sorted u64 timestamps; differences are small positive numbers.
let timestamps: Vec<u64> = vec![1_000_000, 1_001_500, 1_003_000, 1_010_000];

let deltas = delta::encode(&timestamps);
let encoded = U64Coder1248.encode(&deltas);

let decoded_deltas = U64Coder1248.decode(&encoded, deltas.len()).unwrap();
let recovered = delta::decode(&decoded_deltas);
assert_eq!(recovered, timestamps);
```

`delta` is implemented for `i16`, `u16`, `i32`, `u32`, `i64` and `u64`. Use a signed type when the sequence is non-monotone and you intend to follow with zigzag; use an unsigned type for sorted data where all differences are non-negative.

### Streaming / chunked delta

For streaming use-cases where data arrives in chunks, use `encode_with_initial` and `decode_with_initial` to carry the boundary value across chunks:

```rust
use svb::delta;

let chunk_a: Vec<u32> = vec![100, 105, 110];
let chunk_b: Vec<u32> = vec![115, 120, 125];

// Encode chunk A; initial value is 0.
let deltas_a = delta::encode_with_initial(0, &chunk_a);

// Encode chunk B using the last value of chunk A as the initial value.
let deltas_b = delta::encode_with_initial(chunk_a[2], &chunk_b);

// Decode chunk A; initial value is 0.
let recovered_a = delta::decode_with_initial(0, &deltas_a);

// Decode chunk B using the last decoded value of chunk A.
let recovered_b = delta::decode_with_initial(recovered_a[2], &deltas_b);

assert_eq!(recovered_a, chunk_a);
assert_eq!(recovered_b, chunk_b);
```

## Zigzag encoding

Zigzag maps signed integers to unsigned integers so that small absolute values map to small codes. This allows signed differences from delta encoding to be compressed efficiently by any unsigned codec.

```text
0  →  0
-1 →  1
 1 →  2
-2 →  3
 2 →  4
...
```

```rust
use svb::{delta, zigzag, u32::U32Classic};

// Arbitrary i32 data: delta, then zigzag, then U32Classic.
let samples: Vec<i32> = vec![-500, 200, -100, 900];

let deltas: Vec<i32> = delta::encode(&samples);
let codes: Vec<u32> = zigzag::encode(&deltas);
let encoded = U32Classic.encode(&codes);

let decoded_codes = U32Classic.decode(&encoded, codes.len()).unwrap();
let decoded_deltas: Vec<i32> = zigzag::decode(&decoded_codes);
let recovered: Vec<i32> = delta::decode(&decoded_deltas);
assert_eq!(recovered, samples);
```

`zigzag` is implemented for `i16` (producing `u16`), `i32` (producing `u32`), and `i64` (producing `u64`).

## Writing into your own buffer

Both modules have `_to_slice` functions that write into a caller-supplied
slice instead of a `Vec`: `delta::encode_to_slice`, `delta::decode_to_slice`
(and their `_with_initial` forms), `zigzag::encode_to_slice` and
`zigzag::decode_to_slice`. The output slice must be the same length as the
input; the functions panic otherwise. Delta also has `delta::decode_in_place`
(and `decode_with_initial_in_place`), which turns a slice of deltas into the
decoded values without a second buffer.

Zigzag changes the element type (`u16` to `i16` and so on), so it has no
in-place form.

The VBZ decode pipeline written with slices only:

```rust
use svb::{delta, u16::Svb16, zigzag};

let samples: Vec<i16> = vec![500, 510, 505, 520, 300, 301];
let encoded = svb::encode_vbz(&samples);

let mut codes = [0u16; 6];
Svb16.decode_to_slice(&encoded, &mut codes).unwrap();
let mut values = [0i16; 6];
zigzag::decode_to_slice(&codes, &mut values);
delta::decode_in_place(&mut values);
assert_eq!(values, samples[..]);
```

[`decode_vbz_fused`](vbz.md) does all three steps in one pass and is faster
for this pipeline.

