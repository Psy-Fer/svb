# Delta and Zigzag

Delta and zigzag are independent slice transforms. They are not tied to any specific codec; you compose them with whichever codec suits your data.

## Delta encoding

Delta encoding replaces each value with the difference from the previous value. This is effective for sorted or slowly-varying data: the differences are small even when the raw values are large.

```rust
use svb::{delta, u64::U64Coder1248};

// Sorted u64 timestamps — differences are small positive numbers.
let timestamps: Vec<u64> = vec![1_000_000, 1_001_500, 1_003_000, 1_010_000];

let deltas = delta::encode(&timestamps);
let encoded = U64Coder1248.encode(&deltas);

let decoded_deltas = U64Coder1248.decode(&encoded, deltas.len()).unwrap();
let recovered = delta::decode(&decoded_deltas);
assert_eq!(recovered, timestamps);
```

`delta` is implemented for `i16`, `i32`, `i64`, `u32`, and `u64`. Use a signed type when the sequence is non-monotone and you intend to follow with zigzag; use an unsigned type for sorted data where all differences are non-negative.

### Streaming / chunked delta

For streaming use-cases where data arrives in chunks, use `encode_with_initial` and `decode_with_initial` to carry the boundary value across chunks:

```rust
use svb::delta;

let chunk_a: Vec<u32> = vec![100, 105, 110];
let chunk_b: Vec<u32> = vec![115, 120, 125];

// Encode chunk A; initial value is 0.
let (deltas_a, last_a) = delta::encode_with_initial(0, &chunk_a);

// Encode chunk B using the last value from chunk A as the initial.
let (deltas_b, _last_b) = delta::encode_with_initial(last_a, &chunk_b);

// Decode chunk A; initial value is 0.
let (recovered_a, last_a) = delta::decode_with_initial(0, &deltas_a);

// Decode chunk B using the boundary value.
let (recovered_b, _) = delta::decode_with_initial(last_a, &deltas_b);

assert_eq!(recovered_a, chunk_a);
assert_eq!(recovered_b, chunk_b);
```

## Zigzag encoding

Zigzag maps signed integers to unsigned integers so that small absolute values map to small codes. This allows signed differences from delta encoding to be compressed efficiently by any unsigned codec.

```
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
