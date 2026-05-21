use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use streamvbyte64::Coder as _;
use svb::{
    decode_vbz, encode_vbz,
    u16::Svb16,
    u32::{U32Classic, U32Variant0124},
    u64::{U64Coder1234, U64Coder1248},
    delta, zigzag,
};

const SIZES: &[usize] = &[128, 1024, 8192];

// ── data generators ───────────────────────────────────────────────────────────

fn svb16_data(n: usize, kind: &str) -> (Vec<u16>, Vec<u8>) {
    let values: Vec<u16> = (0..n)
        .map(|i| match kind {
            "small" => (i % 200) as u16,
            "large" => (i % 60000 + 256) as u16,
            _ => {
                if i % 2 == 0 {
                    (i % 200) as u16
                } else {
                    (i % 60000 + 256) as u16
                }
            }
        })
        .collect();
    let enc = Svb16.encode(&values);
    (values, enc)
}

// Realistic nanopore-style i16 signal: slow ramp + small noise → mostly 1-byte deltas.
fn vbz_signal(n: usize) -> (Vec<i16>, Vec<u8>) {
    let samples: Vec<i16> = (0..n)
        .map(|i| {
            let base = (i as i32 % 500 - 250) as i16;
            let noise = (i as i16).wrapping_mul(37) % 7 - 3;
            base.wrapping_add(noise)
        })
        .collect();
    let enc = encode_vbz(&samples);
    (samples, enc)
}

fn u32_data(n: usize) -> (Vec<u32>, Vec<u8>) {
    let values: Vec<u32> = (0..n)
        .map(|i| match i % 4 {
            0 => i as u32 % 256,
            1 => i as u32 % 65536,
            2 => i as u32 % 16_777_216,
            _ => i as u32,
        })
        .collect();
    let enc = U32Classic.encode(&values);
    (values, enc)
}

fn u32_0124_data(n: usize) -> (Vec<u32>, Vec<u8>) {
    // ~50% zeros (sparse) to favour the 0-byte variant
    let values: Vec<u32> = (0..n)
        .map(|i| if i % 2 == 0 { 0u32 } else { (i % 256) as u32 })
        .collect();
    let enc = U32Variant0124.encode(&values);
    (values, enc)
}

fn u64_1234_data(n: usize) -> (Vec<u64>, Vec<u8>) {
    let values: Vec<u64> = (0..n)
        .map(|i| match i % 4 {
            0 => i as u64 % 256,
            1 => i as u64 % 65536,
            2 => i as u64 % 16_777_216,
            _ => i as u64 % 0xFFFF_FFFF,
        })
        .collect();
    let enc = U64Coder1234.encode(&values);
    (values, enc)
}

fn u64_1248_data(n: usize) -> (Vec<u64>, Vec<u8>) {
    let values: Vec<u64> = (0..n)
        .map(|i| match i % 4 {
            0 => i as u64 % 256,
            1 => i as u64 % 65536,
            2 => i as u64 % 0xFFFF_FFFF,
            _ => i as u64 | 0x0100_0000_0000_0000,
        })
        .collect();
    let enc = U64Coder1248.encode(&values);
    (values, enc)
}

// ── VBZ pipeline breakdown ────────────────────────────────────────────────────
//
// Each transform measured in isolation on VBZ-style i16 signal data so the
// numbers add up to the full VBZ cost.

fn vbz_i16_samples(n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| {
            let base = (i as i32 % 500 - 250) as i16;
            let noise = (i as i16).wrapping_mul(37) % 7 - 3;
            base.wrapping_add(noise)
        })
        .collect()
}

fn bench_delta_encode_i16(c: &mut Criterion) {
    let mut g = c.benchmark_group("delta/encode_i16");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let samples = vbz_i16_samples(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &samples, |b, s| {
            b.iter(|| delta::encode(s));
        });
    }
    g.finish();
}

fn bench_delta_decode_i16(c: &mut Criterion) {
    let mut g = c.benchmark_group("delta/decode_i16");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let deltas = delta::encode(&vbz_i16_samples(n));
        g.bench_with_input(BenchmarkId::from_parameter(n), &deltas, |b, d| {
            b.iter(|| delta::decode(d));
        });
    }
    g.finish();
}

fn bench_zigzag_encode_i16(c: &mut Criterion) {
    let mut g = c.benchmark_group("zigzag/encode_i16");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let deltas = delta::encode(&vbz_i16_samples(n));
        g.bench_with_input(BenchmarkId::from_parameter(n), &deltas, |b, d| {
            b.iter(|| zigzag::encode(d));
        });
    }
    g.finish();
}

fn bench_zigzag_decode_u16(c: &mut Criterion) {
    let mut g = c.benchmark_group("zigzag/decode_u16");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let codes: Vec<u16> = zigzag::encode(&delta::encode(&vbz_i16_samples(n)));
        g.bench_with_input(BenchmarkId::from_parameter(n), &codes, |b, c_| {
            b.iter(|| zigzag::decode::<i16>(c_));
        });
    }
    g.finish();
}

// ── SVB16 ─────────────────────────────────────────────────────────────────────

fn bench_svb16_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("svb16/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        for kind in ["small", "large", "mixed"] {
            let (values, _) = svb16_data(n, kind);
            g.bench_with_input(BenchmarkId::new(kind, n), &values, |b, v| {
                b.iter(|| Svb16.encode(v));
            });
        }
    }
    g.finish();
}

fn bench_svb16_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("svb16/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        for kind in ["small", "large", "mixed"] {
            let (_, enc) = svb16_data(n, kind);
            g.bench_with_input(BenchmarkId::new(kind, n), &(enc, n), |b, (enc, n)| {
                b.iter(|| Svb16.decode(enc, *n).unwrap());
            });
        }
    }
    g.finish();
}

// ── VBZ pipeline ──────────────────────────────────────────────────────────────

fn bench_vbz_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("vbz/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (samples, _) = vbz_signal(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &samples, |b, s| {
            b.iter(|| encode_vbz(s));
        });
    }
    g.finish();
}

fn bench_vbz_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("vbz/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, enc) = vbz_signal(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| decode_vbz(enc, *n).unwrap());
        });
    }
    g.finish();
}

// ── U32 ───────────────────────────────────────────────────────────────────────

fn bench_u32_classic_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_classic/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (values, _) = u32_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, v| {
            b.iter(|| U32Classic.encode(v));
        });
    }
    g.finish();
}

fn bench_u32_classic_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_classic/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, enc) = u32_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| U32Classic.decode(enc, *n).unwrap());
        });
    }
    g.finish();
}

fn bench_u32_variant0124_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_variant0124/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (values, _) = u32_0124_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, v| {
            b.iter(|| U32Variant0124.encode(v));
        });
    }
    g.finish();
}

fn bench_u32_variant0124_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_variant0124/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, enc) = u32_0124_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| U32Variant0124.decode(enc, *n).unwrap());
        });
    }
    g.finish();
}

// ── U64 ───────────────────────────────────────────────────────────────────────

fn bench_u64_coder1234_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64_coder1234/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (values, _) = u64_1234_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, v| {
            b.iter(|| U64Coder1234.encode(v));
        });
    }
    g.finish();
}

fn bench_u64_coder1234_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64_coder1234/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, enc) = u64_1234_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| U64Coder1234.decode(enc, *n).unwrap());
        });
    }
    g.finish();
}

fn bench_u64_coder1248_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64_coder1248/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (values, _) = u64_1248_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &values, |b, v| {
            b.iter(|| U64Coder1248.encode(v));
        });
    }
    g.finish();
}

fn bench_u64_coder1248_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("u64_coder1248/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, enc) = u64_1248_data(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| U64Coder1248.decode(enc, *n).unwrap());
        });
    }
    g.finish();
}

// ── decode_into: pre-allocated output, no alloc overhead ─────────────────────
//
// Uses a single Vec that is cleared and reused across iterations.  This isolates
// pure SIMD decode throughput from malloc/free noise.

fn bench_u32_classic_decode_into(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_classic/decode_into");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, enc) = u32_data(n);
        let mut out = Vec::with_capacity(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| {
                out.clear();
                U32Classic.decode_into(enc, *n, &mut out).unwrap();
            });
        });
    }
    g.finish();
}

fn bench_u32_variant0124_decode_into(c: &mut Criterion) {
    let mut g = c.benchmark_group("u32_variant0124/decode_into");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, enc) = u32_0124_data(n);
        let mut out = Vec::with_capacity(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| {
                out.clear();
                U32Variant0124.decode_into(enc, *n, &mut out).unwrap();
            });
        });
    }
    g.finish();
}

// ── comparative: svb vs streamvbyte64 ─────────────────────────────────────────
//
// For each shared codec variant we benchmark encode and decode side-by-side
// using the same input data.  The benchmark IDs follow the pattern:
//   "svb/U32Classic/encode/<n>"
//   "streamvbyte64/U32Classic/encode/<n>"
//
// `streamvbyte64` requires len%4==0; all SIZES already satisfy this since
// 128, 1024, and 8192 are all multiples of 4.
//
// `streamvbyte64` keeps tags and data in separate buffers.  We pre-allocate
// both to max size and reuse them across iterations to avoid allocation noise.

// ── prepare sv64-style pre-split buffers from an svb-encoded blob ──────────────

/// Split an svb flat blob into owned (tags, data) for sv64 decode.
fn split_svb(encoded: &[u8], n: usize) -> (Vec<u8>, Vec<u8>) {
    let ctrl_len = n.div_ceil(4);
    (encoded[..ctrl_len].to_vec(), encoded[ctrl_len..].to_vec())
}

// ── U32Classic ↔ Coder1234 ───────────────────────────────────────────────────

fn bench_compare_u32_classic_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/U32Classic/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (values, _) = u32_data(n);

        g.bench_with_input(BenchmarkId::new("svb", n), &values, |b, v| {
            b.iter(|| U32Classic.encode(v));
        });

        g.bench_with_input(BenchmarkId::new("streamvbyte64", n), &values, |b, v| {
            let coder = streamvbyte64::Coder1234::new();
            let (tl, dl) = streamvbyte64::Coder1234::max_compressed_bytes(v.len());
            b.iter(|| {
                let mut tags = vec![0u8; tl];
                let mut data = vec![0u8; dl];
                coder.encode(v, &mut tags, &mut data)
            });
        });
    }
    g.finish();
}

fn bench_compare_u32_classic_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/U32Classic/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, svb_enc) = u32_data(n);
        let (tags, data) = split_svb(&svb_enc, n);

        g.bench_with_input(
            BenchmarkId::new("svb", n),
            &(svb_enc.clone(), n),
            |b, (enc, n)| {
                b.iter(|| U32Classic.decode(enc, *n).unwrap());
            },
        );

        g.bench_with_input(
            BenchmarkId::new("streamvbyte64", n),
            &(tags.clone(), data.clone(), n),
            |b, (tags, data, n)| {
                let coder = streamvbyte64::Coder1234::new();
                b.iter(|| {
                    let mut out = vec![0u32; *n];
                    coder.decode(tags, data, &mut out);
                    out
                });
            },
        );
    }
    g.finish();
}

// ── U32Variant0124 ↔ Coder0124 ───────────────────────────────────────────────

fn bench_compare_u32_variant0124_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/U32Variant0124/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (values, _) = u32_0124_data(n);

        g.bench_with_input(BenchmarkId::new("svb", n), &values, |b, v| {
            b.iter(|| U32Variant0124.encode(v));
        });

        g.bench_with_input(BenchmarkId::new("streamvbyte64", n), &values, |b, v| {
            let coder = streamvbyte64::Coder0124::new();
            let (tl, dl) = streamvbyte64::Coder0124::max_compressed_bytes(v.len());
            b.iter(|| {
                let mut tags = vec![0u8; tl];
                let mut data = vec![0u8; dl];
                coder.encode(v, &mut tags, &mut data)
            });
        });
    }
    g.finish();
}

fn bench_compare_u32_variant0124_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/U32Variant0124/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, svb_enc) = u32_0124_data(n);
        let (tags, data) = split_svb(&svb_enc, n);

        g.bench_with_input(
            BenchmarkId::new("svb", n),
            &(svb_enc.clone(), n),
            |b, (enc, n)| {
                b.iter(|| U32Variant0124.decode(enc, *n).unwrap());
            },
        );

        g.bench_with_input(
            BenchmarkId::new("streamvbyte64", n),
            &(tags.clone(), data.clone(), n),
            |b, (tags, data, n)| {
                let coder = streamvbyte64::Coder0124::new();
                b.iter(|| {
                    let mut out = vec![0u32; *n];
                    coder.decode(tags, data, &mut out);
                    out
                });
            },
        );
    }
    g.finish();
}

// ── U64Coder1248 ↔ Coder1248 ─────────────────────────────────────────────────

fn bench_compare_u64_coder1248_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/U64Coder1248/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (values, _) = u64_1248_data(n);

        g.bench_with_input(BenchmarkId::new("svb", n), &values, |b, v| {
            b.iter(|| U64Coder1248.encode(v));
        });

        g.bench_with_input(BenchmarkId::new("streamvbyte64", n), &values, |b, v| {
            let coder = streamvbyte64::Coder1248::new();
            let (tl, dl) = streamvbyte64::Coder1248::max_compressed_bytes(v.len());
            b.iter(|| {
                let mut tags = vec![0u8; tl];
                let mut data = vec![0u8; dl];
                coder.encode(v, &mut tags, &mut data)
            });
        });
    }
    g.finish();
}

fn bench_compare_u64_coder1248_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("compare/U64Coder1248/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let (_, svb_enc) = u64_1248_data(n);
        let (tags, data) = split_svb(&svb_enc, n);

        g.bench_with_input(
            BenchmarkId::new("svb", n),
            &(svb_enc.clone(), n),
            |b, (enc, n)| {
                b.iter(|| U64Coder1248.decode(enc, *n).unwrap());
            },
        );

        g.bench_with_input(
            BenchmarkId::new("streamvbyte64", n),
            &(tags.clone(), data.clone(), n),
            |b, (tags, data, n)| {
                let coder = streamvbyte64::Coder1248::new();
                b.iter(|| {
                    let mut out = vec![0u64; *n];
                    coder.decode(tags, data, &mut out);
                    out
                });
            },
        );
    }
    g.finish();
}

// ── registry ──────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_delta_encode_i16,
    bench_delta_decode_i16,
    bench_zigzag_encode_i16,
    bench_zigzag_decode_u16,
    bench_svb16_encode,
    bench_svb16_decode,
    bench_vbz_encode,
    bench_vbz_decode,
    bench_u32_classic_encode,
    bench_u32_classic_decode,
    bench_u32_classic_decode_into,
    bench_u32_variant0124_encode,
    bench_u32_variant0124_decode,
    bench_u32_variant0124_decode_into,
    bench_u64_coder1234_encode,
    bench_u64_coder1234_decode,
    bench_u64_coder1248_encode,
    bench_u64_coder1248_decode,
    bench_compare_u32_classic_encode,
    bench_compare_u32_classic_decode,
    bench_compare_u32_variant0124_encode,
    bench_compare_u32_variant0124_decode,
    bench_compare_u64_coder1248_encode,
    bench_compare_u64_coder1248_decode,
);
criterion_main!(benches);
