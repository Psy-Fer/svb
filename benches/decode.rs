use criterion::{criterion_group, criterion_main, Criterion};

fn bench_decode(_c: &mut Criterion) {
    // Benchmarks will be added alongside SIMD implementations.
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
