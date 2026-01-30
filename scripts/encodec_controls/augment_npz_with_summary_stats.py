#!/usr/bin/env python3
"""Add summary-stat fields to a cache item .npz for controllability experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _require_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"numpy is required: {e}")


def _read_npz(path: Path) -> Dict[str, Any]:
    np = _require_numpy()
    with np.load(Path(path), allow_pickle=False) as d:  # type: ignore[attr-defined]
        return {str(k): np.asarray(d[k]) for k in getattr(d, "files", [])}


def _atomic_save_npz(out_path: Path, arrs: Dict[str, Any]) -> None:
    np = _require_numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    # Use a file handle to avoid numpy auto-appending ".npz" to non-.npz suffixes.
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrs)
    tmp.replace(out_path)


def _as_f32(x) -> "Any":
    np = _require_numpy()
    return np.asarray(x, dtype=np.float32)


def _as_i64(x) -> "Any":
    np = _require_numpy()
    return np.asarray(x, dtype=np.int64)


def _parse_targets(arg: Optional[str], *, D: int) -> Optional["Any"]:
    np = _require_numpy()
    if arg is None:
        return None
    s = str(arg).strip()
    if not s:
        return None
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        vals = [float(p) for p in parts]
        if len(vals) == 1:
            vals = vals * int(D)
        if len(vals) != int(D):
            raise SystemExit(f"--target expects 1 or D={D} comma-separated floats; got {len(vals)}")
        return np.asarray(vals, dtype=np.float32)
    v = float(s)
    return np.full((int(D),), float(v), dtype=np.float32)


def main(argv: Optional[Sequence[str]] = None) -> None:
    from compute_grid_summary_stats import summarize_npz

    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="Output .npz (default: <npz>.summary.npz).")
    ap.add_argument("--inplace", action="store_true", help="Overwrite --npz in place (ignores --out).")
    ap.add_argument("--hit-threshold", type=float, default=0.8)
    ap.add_argument("--peak-radius", type=int, default=2)
    ap.add_argument(
        "--target-hit-rate-per-beat",
        type=str,
        default=None,
        help="Optional requested target hit-rate(s): single float or comma-separated D floats.",
    )
    ap.add_argument(
        "--target-vel-mean",
        type=str,
        default=None,
        help="Optional requested target vel mean(s) in [0,1]: single float or comma-separated D floats.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    npz_path = Path(args.npz)
    out_path = npz_path if args.inplace else Path(args.out) if args.out is not None else npz_path.with_suffix(".summary.npz")

    summary = summarize_npz(npz_path, hit_threshold=float(args.hit_threshold), peak_radius=int(args.peak_radius))
    arrs = _read_npz(npz_path)

    D = int(summary.D)
    arrs["summary_hit_threshold"] = _as_f32(float(summary.hit_threshold))
    arrs["summary_peak_radius"] = _as_i64(int(summary.peak_radius))
    arrs["summary_beats"] = _as_f32(float(summary.beats))

    arrs["summary_hits"] = _as_i64([d.hits for d in summary.drums])
    arrs["summary_hit_rate_per_beat"] = _as_f32([d.hit_rate_per_beat for d in summary.drums])
    arrs["summary_vel_mean"] = _as_f32([d.vel_mean for d in summary.drums])
    arrs["summary_vel_std"] = _as_f32([d.vel_std for d in summary.drums])
    arrs["summary_vel_p10"] = _as_f32([d.vel_p10 for d in summary.drums])
    arrs["summary_vel_p50"] = _as_f32([d.vel_p50 for d in summary.drums])
    arrs["summary_vel_p90"] = _as_f32([d.vel_p90 for d in summary.drums])

    thr = _parse_targets(getattr(args, "target_hit_rate_per_beat", None), D=D)
    if thr is not None:
        arrs["target_hit_rate_per_beat"] = _as_f32(thr)
    tvm = _parse_targets(getattr(args, "target_vel_mean", None), D=D)
    if tvm is not None:
        arrs["target_vel_mean"] = _as_f32(tvm)

    _atomic_save_npz(out_path, arrs)
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
