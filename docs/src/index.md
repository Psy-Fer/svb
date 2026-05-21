# svb

`svb` is a pure-Rust [StreamVByte](https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/) library covering all major codec variants for `u16`, `u32`, and `u64` integers. Delta and zigzag encoding are composable layers on top. SIMD back-ends are available for x86-64 (SSSE3, AVX2) and AArch64 (NEON).

## What is StreamVByte?

StreamVByte is a family of integer compression schemes that store values in a variable number of bytes. Rather than interleaving the control information with the data, StreamVByte places all control bytes in a separate stream. This layout makes SIMD-accelerated decode practical: a batch of control bytes can be loaded and shuffled in a single instruction, determining widths for an entire group of values without branching.

```
encoded buffer layout
┌────────────────────┬─────────────────────────────────────┐
│   control stream   │            data stream               │
│  ceil(n/4) bytes   │         variable length              │
└────────────────────┴─────────────────────────────────────┘
```

Each 2-bit tag in the control stream describes the byte width of the corresponding value. Four values share one control byte. The byte widths available depend on the codec variant.

## Codec variants at a glance

| Variant | Element | Tag width | Byte widths | Best for |
|---|---|---|---|---|
| [`Svb16`](codecs/svb16.md) | `u16` | 1 bit | 1/2 | ONT VBZ signal data |
| [`U32Classic`](codecs/u32-classic.md) | `u32` | 2 bits | 1/2/3/4 | General u32, C-library compatible |
| [`U32Variant0124`](codecs/u32-variant0124.md) | `u32` | 2 bits | 0/1/2/4 | Sparse u32 (many zeros) |
| [`U64Coder1234`](codecs/u64-coder1234.md) | `u64` | 2 bits | 1/2/3/4 | u64 values that fit in u32 |
| [`U64Coder1248`](codecs/u64-coder1248.md) | `u64` | 2 bits | 1/2/4/8 | Full u64 range |

## API docs

Rustdoc API reference is published at [docs.rs/svb](https://docs.rs/svb).
