#!/usr/bin/env python3
"""
Parse criterion --output-format=bencher output files and write a Markdown
comparison table to stdout for use in GitHub Step Summary.

Usage:
    bench_summary.py --arch <arch> <label:file> [<label:file> ...]

The first label/file pair is treated as the scalar baseline for ratio columns.
"""
import re
import sys
import argparse
from datetime import datetime, timezone

SIZES = [128, 1024, 8192]

# Benchmark groups to display.  Each entry is (display_label, bench_id_prefix).
# The parser appends "/<n>" to each prefix to look up the per-size entry.
# N is always the last "/" component in the bench ID.

PIPELINE = [
    ("svbzd/encode",           "svbzd/encode"),
    ("svbzd/decode (3-pass)",  "svbzd/decode"),
    ("svbzd_fused",            "svbzd_fused"),
    ("vbz/encode",             "vbz/encode"),
    ("vbz/decode (3-pass)",    "vbz/decode"),
    ("vbz_fused",              "vbz_fused"),
    ("vbz2_fused (2-chain)",   "vbz2_fused"),
]

CODECS = [
    ("svb16/encode (mixed)",     "svb16/encode/mixed"),
    ("svb16/decode (mixed)",     "svb16/decode/mixed"),
    ("u32_classic/encode",       "u32_classic/encode"),
    ("u32_classic/decode",       "u32_classic/decode"),
    ("u32_variant0124/encode",   "u32_variant0124/encode"),
    ("u32_variant0124/decode",   "u32_variant0124/decode"),
    ("u64_coder1234/encode",     "u64_coder1234/encode"),
    ("u64_coder1234/decode",     "u64_coder1234/decode"),
    ("u64_coder1248/encode",     "u64_coder1248/encode"),
    ("u64_coder1248/decode",     "u64_coder1248/decode"),
]

SUBSTAGES = [
    ("delta/encode_i16",        "delta/encode_i16"),
    ("delta/decode_i16",        "delta/decode_i16"),
    ("delta/decode_2chain",     "delta/decode_i16_2chain"),
    ("zigzag/encode_i16",       "zigzag/encode_i16"),
    ("zigzag/decode_u16",       "zigzag/decode_u16"),
]


def parse_bencher(path: str) -> dict[str, int]:
    """Return {bench_id: ns_per_iter} from a criterion bencher-format file."""
    results: dict[str, int] = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r"test (.+?) \.\.\. bench:\s+([\d,]+)\s+ns/iter", line)
                if m:
                    name = m.group(1).strip()
                    ns = int(m.group(2).replace(",", ""))
                    results[name] = ns
    except FileNotFoundError:
        pass
    return results


def melem_s(bench_id: str, ns: int) -> float | None:
    """Compute Melem/s.  N is the last "/" component of the bench ID."""
    try:
        n = int(bench_id.rsplit("/", 1)[-1])
        return n * 1_000.0 / ns  # n / ns * 1e9 / 1e6
    except (ValueError, ZeroDivisionError):
        return None


def fmt_v(v: float | None) -> str:
    if v is None:
        return "—"
    if v >= 1_000:
        return f"{v / 1_000:.2f} Gelem/s"
    return f"{v:.0f} Melem/s"


def fmt_ratio(v: float | None, base: float | None) -> str:
    if v is None or not base:
        return "—"
    return f"{v / base:.2f}×"


def make_table(runs: list[tuple[str, dict]], benches: list[tuple[str, str]]) -> str:
    """Build a Markdown table comparing throughput across paths for all SIZES."""
    scalar_label = runs[0][0]
    extra_labels = [lbl for lbl, _ in runs[1:]]

    # Column headers
    headers = ["Benchmark", "n"]
    for lbl, _ in runs:
        headers.append(lbl)
        if lbl != scalar_label:
            headers.append(f"{lbl}×")
    sep = ["---"] + ["---:"] * (len(headers) - 1)

    rows: list[str] = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]

    for display, prefix in benches:
        first_row = True
        for n in SIZES:
            bid = f"{prefix}/{n}"
            vals: list[float | None] = []
            for _, results in runs:
                ns = results.get(bid)
                vals.append(melem_s(bid, ns) if ns else None)

            scalar_val = vals[0]
            cells = [display if first_row else "", str(n)]
            for i, (lbl, _) in enumerate(runs):
                cells.append(fmt_v(vals[i]))
                if lbl != scalar_label:
                    cells.append(fmt_ratio(vals[i], scalar_val))
            rows.append("| " + " | ".join(cells) + " |")
            first_row = False

    return "\n".join(rows)


def count_benches(results: dict[str, int]) -> int:
    return len(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", required=True, help="Architecture label (e.g. x86-64)")
    parser.add_argument(
        "paths", nargs="+", metavar="LABEL:FILE",
        help="label:file pairs; first is the scalar baseline"
    )
    args = parser.parse_args()

    runs: list[tuple[str, dict]] = []
    for spec in args.paths:
        label, _, path = spec.partition(":")
        if not path:
            print(f"ERROR: expected LABEL:FILE, got {spec!r}", file=sys.stderr)
            sys.exit(1)
        results = parse_bencher(path)
        runs.append((label, results))

    if not any(r for _, r in runs):
        print("No benchmark results found.", file=sys.stderr)
        sys.exit(1)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = count_benches(runs[0][1])

    print(f"## Benchmark Results — {args.arch} ({now})\n")
    print(
        f"Throughput in Melem/s (millions of elements per second). "
        f"`×` = speedup over the `{runs[0][0]}` (scalar) path. "
        f"{total} benchmarks measured.\n"
    )

    print("### Pipelines\n")
    print(make_table(runs, PIPELINE))
    print()

    print("### Codecs\n")
    print(make_table(runs, CODECS))
    print()

    print("### Sub-stages\n")
    print(make_table(runs, SUBSTAGES))
    print()


if __name__ == "__main__":
    main()
