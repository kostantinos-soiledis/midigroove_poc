#!/usr/bin/env python3
"""Compute summary statistics from an expressive-grid cache .npz.

This is meant to support "controllability" experiments where we condition the
renderer on summary stats derived from the grid itself:
  - per-instrument hit-rate (events per beat)
  - per-instrument velocity distribution

Works on cache items produced by `python -m midigroove_poc drumgrid train --precache ...`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _maybe_parse_semantics_channels(d: Dict[str, Any]) -> Optional[List[str]]:
    try:
        raw = d.get("semantics", None)
        if raw is None:
            return None
        s = str(raw.item() if hasattr(raw, "item") else raw)
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return None
        ch = obj.get("channels", None)
        if not isinstance(ch, list) or not ch:
            return None
        out = []
        for x in ch:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            else:
                out.append(str(x))
        return out or None
    except Exception:
        return None


def _beats_in_window(d: Dict[str, Any], *, fallback_bpm: float = 120.0) -> float:
    np = _require_numpy()
    beats_per_chunk = None
    try:
        bpc = d.get("beats_per_chunk", None)
        if bpc is not None:
            beats_per_chunk = int(np.asarray(bpc, dtype=np.int64).item())
    except Exception:
        beats_per_chunk = None
    if beats_per_chunk is not None and beats_per_chunk > 0:
        return float(beats_per_chunk)

    try:
        win_s = float(np.asarray(d.get("window_seconds", 0.0), dtype=np.float32).item())
    except Exception:
        win_s = 0.0
    try:
        bpm = float(np.asarray(d.get("bpm", fallback_bpm), dtype=np.float32).item())
    except Exception:
        bpm = float(fallback_bpm)
    if win_s > 0.0 and bpm > 1e-6:
        return float(win_s) * float(bpm) / 60.0
    return 0.0


def _nms_peaks_1d(x, *, threshold: float, radius: int) -> List[int]:
    """Return peak indices using simple non-maximum suppression."""
    np = _require_numpy()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    T = int(x.shape[0])
    if T <= 0:
        return []
    radius = int(max(0, radius))
    threshold = float(threshold)
    cand = np.where(x >= threshold)[0]
    if cand.size == 0:
        return []

    # Sort candidates by (value desc, index asc) for deterministic tie-breaking.
    vals = x[cand]
    order = np.lexsort((cand, -vals))
    cand_sorted = cand[order]

    selected: List[int] = []
    suppressed = np.zeros(T, dtype=np.bool_)
    for i in cand_sorted.tolist():
        if suppressed[int(i)]:
            continue
        selected.append(int(i))
        if radius > 0:
            j0 = max(0, int(i) - radius)
            j1 = min(T, int(i) + radius + 1)
            suppressed[j0:j1] = True
        else:
            suppressed[int(i)] = True
    selected.sort()
    return selected


def _pct(x, q: float) -> float:
    np = _require_numpy()
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, float(q)))


@dataclass(frozen=True)
class DrumSummary:
    name: str
    hits: int
    hit_rate_per_beat: float
    vel_mean: float
    vel_std: float
    vel_p10: float
    vel_p50: float
    vel_p90: float


@dataclass(frozen=True)
class GridSummary:
    npz: str
    D: int
    T: int
    bpm: float
    beats: float
    hit_threshold: float
    peak_radius: int
    drums: List[DrumSummary]


def summarize_npz(
    npz_path: Path,
    *,
    hit_threshold: float = 0.8,
    peak_radius: int = 2,
) -> GridSummary:
    np = _require_numpy()
    d = _read_npz(Path(npz_path))
    hit = np.asarray(d.get("drum_hit", None), dtype=np.float32)
    if hit.ndim != 2:
        raise ValueError(f"drum_hit must be [D,T] float32; got shape={tuple(hit.shape)} in {npz_path}")
    D, T = map(int, hit.shape)
    vel = np.asarray(d.get("drum_vel", np.zeros_like(hit)), dtype=np.float32)
    if vel.shape != (D, T):
        vel = np.broadcast_to(vel, (D, T)).copy()

    try:
        bpm = float(np.asarray(d.get("bpm", 120.0), dtype=np.float32).item())
    except Exception:
        bpm = 120.0
    beats = float(_beats_in_window(d, fallback_bpm=float(bpm)) or 0.0)
    denom_beats = beats if beats > 1e-6 else float("nan")

    ch = _maybe_parse_semantics_channels(d)
    if ch is None or len(ch) != D:
        ch = [f"drum_{i:02d}" for i in range(D)]

    drums: List[DrumSummary] = []
    for i in range(D):
        peaks = _nms_peaks_1d(hit[i], threshold=float(hit_threshold), radius=int(peak_radius))
        hits_n = int(len(peaks))
        hit_rate = float(hits_n) / float(denom_beats) if denom_beats == denom_beats else float("nan")  # noqa: E711
        v = vel[i, peaks] if hits_n > 0 else np.asarray([], dtype=np.float32)
        drums.append(
            DrumSummary(
                name=str(ch[i]),
                hits=int(hits_n),
                hit_rate_per_beat=float(hit_rate),
                vel_mean=float(np.mean(v).item()) if v.size else float("nan"),
                vel_std=float(np.std(v).item()) if v.size else float("nan"),
                vel_p10=_pct(v, 10.0),
                vel_p50=_pct(v, 50.0),
                vel_p90=_pct(v, 90.0),
            )
        )

    return GridSummary(
        npz=str(Path(npz_path)),
        D=int(D),
        T=int(T),
        bpm=float(bpm),
        beats=float(beats),
        hit_threshold=float(hit_threshold),
        peak_radius=int(peak_radius),
        drums=drums,
    )


def _write_json(path: Path, obj: object) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, summary: GridSummary) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "npz",
                "bpm",
                "beats",
                "D",
                "T",
                "hit_threshold",
                "peak_radius",
                "drum",
                "hits",
                "hit_rate_per_beat",
                "vel_mean",
                "vel_std",
                "vel_p10",
                "vel_p50",
                "vel_p90",
            ],
        )
        w.writeheader()
        base = {
            "npz": summary.npz,
            "bpm": summary.bpm,
            "beats": summary.beats,
            "D": summary.D,
            "T": summary.T,
            "hit_threshold": summary.hit_threshold,
            "peak_radius": summary.peak_radius,
        }
        for d in summary.drums:
            w.writerow(
                {
                    **base,
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True, help="Input cache item .npz (must contain drum_hit; drum_vel optional).")
    ap.add_argument("--hit-threshold", type=float, default=0.8, help="Peak threshold for drum_hit (default: 0.8).")
    ap.add_argument("--peak-radius", type=int, default=2, help="Non-max suppression radius in frames (default: 2).")
    ap.add_argument("--json-out", type=Path, default=None, help="Optional write JSON to this path.")
    ap.add_argument("--csv-out", type=Path, default=None, help="Optional write CSV (one row per drum) to this path.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    summary = summarize_npz(Path(args.npz), hit_threshold=float(args.hit_threshold), peak_radius=int(args.peak_radius))
    obj = asdict(summary)
    print(json.dumps(obj, indent=2, sort_keys=True))
    if args.json_out is not None:
        _write_json(Path(args.json_out), obj)
    if args.csv_out is not None:
        _write_csv(Path(args.csv_out), summary)


if __name__ == "__main__":
    main()
