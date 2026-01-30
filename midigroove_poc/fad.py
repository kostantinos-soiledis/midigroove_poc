"""FAD utilities built on the external `fadtk` package.

This command is optional. It is mainly useful for computing FAD / FAD∞ on
*saved* predictions produced by:

  python -m midigroove_poc eval ... --save-preds 128 --pred-include-ref --keep-preds

The main eval command can compute FAD/FAD∞ directly during evaluation via:

  python -m midigroove_poc eval ... --audio-metrics --fad-fadtk
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"`torch` is required for fadtk. Import error: {e}")


def _require_fadtk():
    try:
        import fadtk  # type: ignore

        return fadtk
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "`fadtk` is required. Install it with `pip install fadtk` in the current environment.\n"
            f"Import error: {e}"
        )


def _read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n = int(wf.getnframes())
        sw = int(wf.getsampwidth())
        ch = int(wf.getnchannels())
        data = wf.readframes(n)
    if sw != 2:
        raise RuntimeError(f"Expected 16-bit PCM WAV, got sampwidth={sw} at {path}")
    y = (np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0).reshape(-1, ch)
    y = y.mean(axis=1) if ch > 1 else y[:, 0]
    return y.astype(np.float32, copy=False), sr


def _resample_poly(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr_in = int(sr_in)
    sr_out = int(sr_out)
    if y.size == 0 or sr_in <= 0 or sr_out <= 0 or sr_in == sr_out:
        return y
    try:
        import scipy.signal  # type: ignore

        g = math.gcd(int(sr_in), int(sr_out))
        up = int(sr_out // g)
        down = int(sr_in // g)
        return scipy.signal.resample_poly(y, up=up, down=down).astype(np.float32, copy=False)
    except Exception:
        # fallback: linear (good enough for debugging)
        dur = float(y.size) / float(sr_in)
        n_out = int(round(dur * float(sr_out)))
        x_old = np.linspace(0.0, 1.0, num=y.size, endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
        return np.interp(x_new, x_old, y).astype(np.float32, copy=False)


def _fad_inf(fadtk_fad_mod: Any, *, mu_base: np.ndarray, cov_base: np.ndarray, embeds: np.ndarray, steps: int, min_n: int, max_n: int, seed: int):
    embeds = np.asarray(embeds, dtype=np.float64)
    n_total = int(embeds.shape[0])
    if n_total < 2:
        return {"fad_inf": float("nan"), "r2": float("nan"), "slope": float("nan"), "points": [], "n_total": n_total}
    max_n = min(int(max_n) if int(max_n) > 0 else n_total, n_total)
    min_n = max(2, min(int(min_n), max_n))
    steps = max(2, int(steps))
    ns = [int(x) for x in np.linspace(min_n, max_n, steps)]
    rng = np.random.default_rng(int(seed))
    points: List[Tuple[int, float]] = []
    for n in ns:
        idx = rng.integers(low=0, high=n_total, size=int(n), endpoint=False)
        sub = embeds[idx]
        mu = np.mean(sub, axis=0)
        cov = np.cov(sub, rowvar=False)
        score = float(fadtk_fad_mod.calc_frechet_distance(mu_base, cov_base, mu, cov))
        points.append((int(n), float(score)))
    xs = 1.0 / np.asarray([p[0] for p in points], dtype=np.float64)
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    slope, intercept = np.polyfit(xs, ys, 1)
    yhat = slope * xs + intercept
    denom = float(np.sum((ys - float(np.mean(ys))) ** 2))
    r2 = 1.0 - float(np.sum((ys - yhat) ** 2)) / denom if denom > 0 else float("nan")
    return {"fad_inf": float(intercept), "slope": float(slope), "r2": float(r2), "points": points, "n_total": n_total}


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Compute FAD / FAD∞ on saved eval predictions using fadtk.")
    ap.add_argument("eval_dir", type=Path, help="Evaluation output dir containing summary.json (e.g. artifacts/eval/small_one_kit).")
    ap.add_argument("--model", type=str, default="clap-laion-music", help="fadtk model name.")
    ap.add_argument("--device", type=str, default="cpu", help="Embedding device for fadtk (cpu or cuda[:idx]).")
    ap.add_argument("--inf", action="store_true", help="Also compute FAD∞ (extrapolated).")
    ap.add_argument("--inf-steps", type=int, default=15)
    ap.add_argument("--inf-min-n", type=int, default=128)
    ap.add_argument("--inf-max-n", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=None, help="Optional path to write a JSON results file (defaults to <eval_dir>/fadtk.json).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    _require_torch()
    fadtk = _require_fadtk()
    import fadtk.fad as fadtk_fad  # type: ignore
    from fadtk.model_loader import get_all_models  # type: ignore

    eval_dir = Path(args.eval_dir)
    summ = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
    pred_root = summ.get("pred_dir")
    if not pred_root:
        raise SystemExit("summary.json has no pred_dir; rerun eval with --save-preds and --pred-dir.")
    pred_root = Path(pred_root)

    models = {m.name: m for m in get_all_models()}
    if args.model not in models:
        raise SystemExit(f"Unknown fadtk model {args.model!r}. Available: {sorted(models.keys())}")
    ml = models[args.model]

    torch = _require_torch()
    d = str(args.device).strip().lower()
    if d == "" or d == "cpu":
        ml.device = torch.device("cpu")
    else:
        # Same fadtk quirk as eval.py: use torch.device('cuda') and set_device()
        # so ModelLoader.get_embedding() correctly moves tensors to CPU.
        idx = None
        if d.startswith("cuda") and ":" in d:
            try:
                idx = int(d.split(":", 1)[1])
            except Exception:
                idx = None
        if idx is not None:
            try:
                torch.cuda.set_device(int(idx))  # type: ignore[attr-defined]
            except Exception:
                pass
        ml.device = torch.device("cuda") if d.startswith("cuda") else torch.device(d)
    ml.load_model()

    out: Dict[str, Any] = {"model": args.model, "device": args.device, "systems": {}}
    systems = [k for k in (summ.get("systems", {}) or {}).keys() if not str(k).endswith(("_oracle", "_random"))]

    ref_dir = pred_root / "ref"
    if not ref_dir.is_dir():
        raise SystemExit(f"Missing ref dir: {ref_dir} (did you run with --pred-include-ref?)")
    ref_files = sorted(ref_dir.glob("*.wav"))
    if not ref_files:
        raise SystemExit(f"No ref wavs in {ref_dir}")

    # Reference embeddings (clip-level = mean over timeframes).
    ref_vecs = []
    for p in ref_files:
        y, sr = _read_wav_mono(p)
        y = _resample_poly(y, sr, int(ml.sr))
        emb = ml.get_embedding(y)
        ref_vecs.append(np.asarray(np.mean(emb, axis=0), dtype=np.float32))
    ref_mat = np.stack(ref_vecs, axis=0).astype(np.float64, copy=False)
    mu_ref = np.mean(ref_mat, axis=0)
    cov_ref = np.cov(ref_mat, rowvar=False)
    out["n_ref"] = int(ref_mat.shape[0])
    out["dim"] = int(ref_mat.shape[1])

    for sys_name in systems:
        sys_dir = pred_root / "pred" / sys_name
        files = sorted(sys_dir.glob("*.wav"))
        if not files:
            continue
        vecs = []
        for p in files:
            y, sr = _read_wav_mono(p)
            y = _resample_poly(y, sr, int(ml.sr))
            emb = ml.get_embedding(y)
            vecs.append(np.asarray(np.mean(emb, axis=0), dtype=np.float32))
        mat = np.stack(vecs, axis=0).astype(np.float64, copy=False)
        mu = np.mean(mat, axis=0)
        cov = np.cov(mat, rowvar=False)
        fad = float(fadtk_fad.calc_frechet_distance(mu_ref, cov_ref, mu, cov))
        dsys: Dict[str, Any] = {"fad": fad, "n_gen": int(mat.shape[0])}
        if bool(args.inf):
            dsys["fad_inf"] = _fad_inf(
                fadtk_fad,
                mu_base=mu_ref,
                cov_base=cov_ref,
                embeds=mat,
                steps=int(args.inf_steps),
                min_n=int(args.inf_min_n),
                max_n=int(args.inf_max_n),
                seed=0,
            )
        out["systems"][sys_name] = dsys

    out_path = Path(args.out) if args.out is not None else (eval_dir / "fadtk.json")
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
