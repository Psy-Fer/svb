# Contributing

## Running tests

```bash
cargo test                                            # scalar paths
cargo test --features simd-auto                       # all paths including SIMD cross-checks
cargo clippy --features simd-auto -- -D clippy::all  # must be clean
cargo test --no-default-features --features alloc    # no_std check
```

## Fuzz targets

```bash
cargo +nightly fuzz run decode_vbz
cargo +nightly fuzz run svb16_decode
cargo +nightly fuzz run roundtrip
```

## Rules

- No `unsafe` outside SIMD intrinsic modules; every `unsafe` block requires a `// SAFETY:` comment.
- Wire compatibility with the Lemire C `streamvbyte` library must be preserved for `U32Classic`. Changes to encode/decode logic must validate against the test vectors in `tests/vectors/`.
- All four CI platforms must pass (Linux x86-64, Linux AArch64, macOS ARM, Windows x86-64).

See [CLAUDE.md](CLAUDE.md) for full architecture details.

## AI assistance

This library was developed with AI assistance (Claude). The architecture decisions, wire-compatibility validation, and algorithm choices are the author's own; AI tooling served as an accelerator over existing skill, not a replacement for it. Real-data parity testing is done through [pod5lib](https://crates.io/crates/pod5lib), a pure-Rust POD5 reader that uses `svb` for VBZ decompression and validates against real Oxford Nanopore sequencing data.
