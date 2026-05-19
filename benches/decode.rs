use criterion::{Criterion, criterion_group, criterion_main};

fn bench_decode(_c: &mut Criterion) {
    // Benchmarks will be added alongside SIMD implementations.
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
