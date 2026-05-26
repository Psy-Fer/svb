use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use streamvbyte64::Coder as _;
use svb::{
    decode_svbzd, decode_svbzd_fused_into, decode_vbz, decode_vbz_fused_from_into,
    decode_vbz_fused_into, decode_vbz2_into, decode_vbzk_parallel_into, delta, encode_svbzd,
    encode_vbz, encode_vbz2, encode_vbzk,
    u16::Svb16,
    u32::{U32Classic, U32Variant0124},
    u64::{U64Coder1234, U64Coder1248},
    zigzag,
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

fn bench_delta_decode_i16_2chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta/decode_i16_2chain");
    for &n in &[128usize, 1024, 8192] {
        group.throughput(Throughput::Elements(n as u64));
        let samples = vbz_i16_samples(n);
        let deltas = delta::encode(&samples);
        let mc = delta::mid_carry(0i16, &deltas);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut out = Vec::with_capacity(n);
            b.iter(|| {
                out.clear();
                delta::decode_2chain_into(0i16, &deltas, mc, &mut out);
                black_box(&out);
            });
        });
    }
    group.finish();
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

fn bench_vbz_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("vbz_fused");
    for &n in &[128usize, 1024, 8192] {
        group.throughput(Throughput::Elements(n as u64));
        let samples = vbz_i16_samples(n);
        let encoded = encode_vbz(&samples);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut out = Vec::with_capacity(n);
            b.iter(|| {
                out.clear();
                decode_vbz_fused_into(&encoded, n, &mut out).unwrap();
                black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_vbz2_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("vbz2_fused");
    for &n in &[128usize, 1024, 8192] {
        group.throughput(Throughput::Elements(n as u64));
        let samples = vbz_i16_samples(n);
        let encoded = encode_vbz2(&samples);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let mut out = Vec::with_capacity(n);
            b.iter(|| {
                out.clear();
                decode_vbz2_into(&encoded, n, &mut out).unwrap();
                black_box(&out);
            });
        });
    }
    group.finish();
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

// ── VBZ2 parallel decode ─────────────────────────────────────────────────────
//
// Decodes two independent half-streams simultaneously using std::thread::scope.
// BATCH chunks are decoded per iteration so that thread-scope creation overhead
// is amortised: at n=8192, each half decodes in ~2.5µs; BATCH=64 gives ~160µs
// of useful work vs ~5µs of scope overhead (<3%).
//
// The two spawned threads each process BATCH half-A (or half-B) streams sequentially,
// mirroring how a real POD5 reader would assign chunks to a fixed thread pool.
// Throughput is reported as total elements (n × BATCH) per wall-clock second.

fn bench_vbz2_parallel(c: &mut Criterion) {
    use std::time::{Duration, Instant};

    const N: usize = 8192;
    const BATCH: usize = 64;

    let samples = vbz_i16_samples(N);
    let encoded = encode_vbz2(&samples);

    let mid_carry = i16::from_le_bytes([encoded[0], encoded[1]]);
    let mid_data_offset =
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]) as usize;
    let svb = &encoded[6..];
    let n_half = (N / 2) & !7;
    let ctrl_len = N.div_ceil(8);
    let ctrl_half = n_half / 8;
    let n_b = N - n_half;

    // Build the two sub-streams once; reuse them across all iterations.
    let stream_a: Vec<u8> = {
        let mut v = svb[..ctrl_half].to_vec();
        v.extend_from_slice(&svb[ctrl_len..ctrl_len + mid_data_offset]);
        v
    };
    let stream_b: Vec<u8> = {
        let mut v = svb[ctrl_half..ctrl_len].to_vec();
        v.extend_from_slice(&svb[ctrl_len + mid_data_offset..]);
        v
    };

    let mut group = c.benchmark_group("vbz2_parallel");
    group.throughput(Throughput::Elements((N * BATCH) as u64));

    group.bench_function(N.to_string(), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let t0 = Instant::now();
                std::thread::scope(|scope| {
                    scope.spawn(|| {
                        let mut out = Vec::with_capacity(n_half);
                        for _ in 0..BATCH {
                            out.clear();
                            decode_vbz_fused_from_into(&stream_a, n_half, 0, &mut out).unwrap();
                            black_box(&out);
                        }
                    });
                    scope.spawn(|| {
                        let mut out = Vec::with_capacity(n_b);
                        for _ in 0..BATCH {
                            out.clear();
                            decode_vbz_fused_from_into(&stream_b, n_b, mid_carry, &mut out)
                                .unwrap();
                            black_box(&out);
                        }
                    });
                });
                total += t0.elapsed();
            }
            total
        })
    });

    group.finish();
}

// ── VBZ-K parallel decode ─────────────────────────────────────────────────────
//
// Decodes k independent sub-streams simultaneously using std::thread::scope.
// BATCH chunks are decoded per iteration to amortise thread-scope overhead.
// Throughput is reported as total elements (n × BATCH) per wall-clock second.

fn bench_vbzk_parallel(c: &mut Criterion) {
    use std::time::{Duration, Instant};

    const N: usize = 8192;
    const BATCH: usize = 64;

    for &k in &[2usize, 4, 8] {
        let samples = vbz_i16_samples(N);
        let encoded = encode_vbzk(&samples, k);

        // Parse the VBZ-K header to extract carries and data offsets.
        let effective_k = encoded[0] as usize;
        let header_len = 1 + (effective_k - 1) * 6;
        let n_sub = (N / effective_k) & !7;
        let ctrl_len = N.div_ceil(8);
        let svb = &encoded[header_len..];
        let ctrl = &svb[..ctrl_len];
        let data_bytes = &svb[ctrl_len..];

        let mut sub_carry = vec![0i16; effective_k];
        let mut data_start = vec![0usize; effective_k + 1];
        for i in 1..effective_k {
            let off = 1 + (i - 1) * 6;
            sub_carry[i] = i16::from_le_bytes([encoded[off], encoded[off + 1]]);
            data_start[i] = u32::from_le_bytes([
                encoded[off + 2],
                encoded[off + 3],
                encoded[off + 4],
                encoded[off + 5],
            ]) as usize;
        }
        data_start[effective_k] = data_bytes.len();

        // Pre-assemble each sub-stream as a flat [ctrl | data] buffer.
        // This lets us call decode_vbz_fused_from_into (public API) in the threads.
        struct SubStream {
            flat: Vec<u8>,
            n: usize,
            carry: i16,
        }

        let sub_streams: Vec<SubStream> = (0..effective_k)
            .map(|i| {
                let sub_n = if i < effective_k - 1 {
                    n_sub
                } else {
                    N - (effective_k - 1) * n_sub
                };
                let ctrl_start = i * (n_sub / 8);
                let ctrl_end = ctrl_start + sub_n.div_ceil(8);
                let mut flat = ctrl[ctrl_start..ctrl_end].to_vec();
                flat.extend_from_slice(&data_bytes[data_start[i]..data_start[i + 1]]);
                SubStream {
                    flat,
                    n: sub_n,
                    carry: sub_carry[i],
                }
            })
            .collect();

        let mut group = c.benchmark_group("vbzk_parallel");
        group.throughput(Throughput::Elements((N * BATCH) as u64));
        group.bench_function(format!("k={k}/{N}"), |b| {
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                for _ in 0..iters {
                    let t0 = Instant::now();
                    std::thread::scope(|scope| {
                        for sub in &sub_streams {
                            scope.spawn(|| {
                                let mut out = Vec::with_capacity(sub.n);
                                for _ in 0..BATCH {
                                    out.clear();
                                    decode_vbz_fused_from_into(
                                        &sub.flat, sub.n, sub.carry, &mut out,
                                    )
                                    .unwrap();
                                    black_box(&out);
                                }
                            });
                        }
                    });
                    total += t0.elapsed();
                }
                total
            })
        });
        group.finish();
    }
}

// ── SVB-ZD pipeline ───────────────────────────────────────────────────────────
//
// Same nanopore-style i16 signal as the VBZ benchmarks for direct comparison.
// SVB-ZD uses U32Classic (2-bit tags, 1/2/3/4 bytes) instead of SVB16 (1-bit,
// 1/2 bytes), so the encoded size per element is larger and decode throughput
// differs from VBZ.

fn bench_svbzd_encode(c: &mut Criterion) {
    let mut g = c.benchmark_group("svbzd/encode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let samples = vbz_i16_samples(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &samples, |b, s| {
            b.iter(|| encode_svbzd(s));
        });
    }
    g.finish();
}

fn bench_svbzd_decode(c: &mut Criterion) {
    let mut g = c.benchmark_group("svbzd/decode");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let enc = encode_svbzd(&vbz_i16_samples(n));
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| decode_svbzd(enc, *n).unwrap());
        });
    }
    g.finish();
}

fn bench_svbzd_fused(c: &mut Criterion) {
    let mut g = c.benchmark_group("svbzd_fused");
    for &n in SIZES {
        g.throughput(Throughput::Elements(n as u64));
        let enc = encode_svbzd(&vbz_i16_samples(n));
        let mut out = Vec::with_capacity(n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &(enc, n), |b, (enc, n)| {
            b.iter(|| {
                out.clear();
                decode_svbzd_fused_into(enc, *n, &mut out).unwrap();
                black_box(&out);
            });
        });
    }
    g.finish();
}

// ── registry ──────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_delta_encode_i16,
    bench_delta_decode_i16,
    bench_delta_decode_i16_2chain,
    bench_zigzag_encode_i16,
    bench_zigzag_decode_u16,
    bench_svb16_encode,
    bench_svb16_decode,
    bench_vbz_encode,
    bench_vbz_decode,
    bench_vbz_fused,
    bench_vbz2_decode,
    bench_vbz2_parallel,
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
    bench_vbzk_parallel,
    bench_svbzd_encode,
    bench_svbzd_decode,
    bench_svbzd_fused,
);
criterion_main!(benches);
