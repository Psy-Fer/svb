# Codec Variants

`svb` provides five codec variants spanning three element widths. Each is a zero-sized type implementing the same `encode`/`decode` surface.

| Variant | Element | Tag bits | Byte widths | Wire-compatible with |
|---|---|---|---|---|
| [`Svb16`](svb16.md) | `u16` | 1 | 1/2 | ONT `vbz_hdf_plugin` |
| [`U32Classic`](u32-classic.md) | `u32` | 2 | 1/2/3/4 | Lemire C library, `stream-vbyte` crate |
| [`U32Variant0124`](u32-variant0124.md) | `u32` | 2 | 0/1/2/4 | Lemire "0124" variant |
| [`U64Coder1234`](u64-coder1234.md) | `u64` | 2 | 1/2/3/4 | `streamvbyte64::Coder1234` (u32 values) |
| [`U64Coder1248`](u64-coder1248.md) | `u64` | 2 | 1/2/4/8 | `streamvbyte64::Coder1248` |

## Tag encoding

All u32 and u64 codecs pack four 2-bit tags into each control byte, LSB-first:

```
control byte n
bits 1:0  → tag for value 4n+0
bits 3:2  → tag for value 4n+1
bits 5:4  → tag for value 4n+2
bits 7:6  → tag for value 4n+3
```

`Svb16` uses 1-bit tags and packs eight tags per control byte.

## Buffer layout

All codecs use the same flat layout: control bytes first, data bytes immediately after.

```
[ ctrl[0] ctrl[1] ... ctrl[ceil(n/4)-1] | data bytes ... ]
```

The control stream length is always `ceil(n / 4)` bytes for 2-bit codecs, `ceil(n / 8)` for `Svb16`. No length prefix is stored; the caller supplies the element count to `decode`.
