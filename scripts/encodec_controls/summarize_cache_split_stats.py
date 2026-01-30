#!/usr/bin/env python3
"""Summarize per-drum hit-rate/velocity stats for a cache split.

This produces a CSV with one row per (item, drum) so you can quickly inspect
distributions, then decide on controllability target ranges.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _iter_manifest_npzs(cache_dir: Path, *, split: Optional[str]) -> Iterable[Path]:
    cache_dir = Path(cache_dir)
    manifests = sorted(cache_dir.glob("manifest_midigroove_*_*.jsonl"))
    if not manifests:
        raise SystemExit(f"No manifests found under {cache_dir}")
    for mp in manifests:
        name = mp.name
        if split is not None:
            want = str(split).strip().lower()
            if f"manifest_midigroove_{want}_" not in name:
                continue
        for line in mp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            npz_s = str(obj.get("npz", "") or "").strip()
            if not npz_s:
                continue
            p = Path(npz_s)
            if not p.is_absolute():
                p2 = cache_dir / p
                if p2.is_file():
                    p = p2
            if p.is_file():
                yield p


def main(argv: Optional[Sequence[str]] = None) -> None:
    from compute_grid_summary_stats import summarize_npz

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="Cache dir (contains manifest_midigroove_*.jsonl).")
    ap.add_argument("--split", type=str, default=None, help="Optional: train/validation/test.")
    ap.add_argument("--limit", type=int, default=0, help="If >0, stop after N items.")
    ap.add_argument("--hit-threshold", type=float, default=0.8)
    ap.add_argument("--peak-radius", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    n = 0
    for p in _iter_manifest_npzs(Path(args.cache), split=args.split):
        summary = summarize_npz(p, hit_threshold=float(args.hit_threshold), peak_radius=int(args.peak_radius))
        for d in summary.drums:
            rows.append(
                {
                    "npz": summary.npz,
                    "bpm": summary.bpm,
                    "beats": summary.beats,
                    "D": summary.D,
                    "T": summary.T,
                    "hit_threshold": summary.hit_threshold,
                    "peak_radius": summary.peak_radius,
                    "drum": d.name,
                    "hits": d.hits,
                    "hit_rate_per_beat": d.hit_rate_per_beat,
                    "vel_mean": d.vel_mean,
                    "vel_std": d.vel_std,
                    "vel_p10": d.vel_p10,
                    "vel_p50": d.vel_p50,
                    "vel_p90": d.vel_p90,
                }
            )
        n += 1
        if int(args.limit) > 0 and n >= int(args.limit):
            break

    if not rows:
        raise SystemExit("No rows produced (empty cache? bad split filter?)")

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote: {out_path} ({n} items, {len(rows)} rows)")


if __name__ == "__main__":
    main()
