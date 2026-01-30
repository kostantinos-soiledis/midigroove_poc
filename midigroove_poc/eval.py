"""Unified evaluation for expressivegrid->codec-token models across codecs.

Evaluates a set of systems (checkpoint + cache) on a standardized subset:
the intersection of segments present in all provided caches.

Outputs JSON + CSV summaries and (optionally) decoded WAVs for listening.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .runtime import configure_runtime

configure_runtime()


def _wrap_tqdm(
    it: Iterable[Any],
    *,
    desc: str,
    total: Optional[int] = None,
    disable: bool = False,
    leave: bool = False,
    unit: str = "it",
):
    if disable:
        return it
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm(it, desc=str(desc), total=total, leave=bool(leave), unit=str(unit))
    except Exception:
        return it


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "`torch` is required for evaluation. Install PyTorch from https://pytorch.org/get-started/locally/.\n"
            f"Import error: {e}"
        )


def _exit_with_error(msg: str) -> None:
    raise SystemExit(msg)


def _require_pretty_midi():
    try:  # pragma: no cover - optional dependency
        import pretty_midi  # type: ignore

        return pretty_midi
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"`pretty_midi` is required for MIDI-based onset metrics. Import error: {e}")


def _try_madmom():
    """Optional dependency used for stronger onset detection + evaluation utilities."""
    try:  # pragma: no cover - optional dependency
        # Compatibility shim: older madmom versions import these from `collections`,
        # but Python >= 3.10 moved them to `collections.abc`.
        import collections

        if not hasattr(collections, "MutableSequence"):
            from collections.abc import MutableMapping, MutableSequence, MutableSet  # type: ignore

            collections.MutableSequence = MutableSequence  # type: ignore[attr-defined]
            collections.MutableMapping = MutableMapping  # type: ignore[attr-defined]
            collections.MutableSet = MutableSet  # type: ignore[attr-defined]

        # Compatibility shim: older madmom versions use deprecated NumPy aliases
        # like `np.float`/`np.int` which are removed in NumPy >= 1.24.
        import numpy as _np

        if not hasattr(_np, "float"):
            _np.float = float  # type: ignore[attr-defined]
        if not hasattr(_np, "int"):
            _np.int = int  # type: ignore[attr-defined]
        if not hasattr(_np, "bool"):
            _np.bool = bool  # type: ignore[attr-defined]
        if not hasattr(_np, "object"):
            _np.object = object  # type: ignore[attr-defined]
        if not hasattr(_np, "complex"):
            _np.complex = complex  # type: ignore[attr-defined]

        import madmom  # type: ignore

        return madmom
    except Exception as e:
        # If the user explicitly requested madmom, fail loudly so we don't
        # silently fall back to a weaker detector and confuse evaluation.
        backend = str(os.environ.get("MIDIGROOVE_ONSET_BACKEND", "") or "").strip().lower()
        if backend == "madmom":  # pragma: no cover
            raise RuntimeError(
                "MIDIGROOVE_ONSET_BACKEND=madmom but `import madmom` failed.\n"
                "This usually means your notebook/kernel is using a different Python env than the one you installed into.\n"
                f"Import error: {e}"
            )
        return None


def _try_import_fadtk() -> bool:
    try:  # pragma: no cover - optional dependency
        import fadtk  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _require_fadtk():
    try:  # pragma: no cover - optional dependency
        import fadtk  # type: ignore

        return fadtk
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Requested FAD via fadtk, but `import fadtk` failed.\n"
            "Install it with `pip install fadtk` in the same environment you're running this command.\n"
            f"Import error: {e}"
        )


def _link_or_copy(src: Path, dst: Path) -> None:
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
        return
    except Exception:
        pass
    shutil.copyfile(str(src), str(dst))


def _move_legacy_manifest(*, pred_run_dir: Path, system: str) -> None:
    """If an older run left manifest.jsonl inside pred/<system>/, move it under meta/."""
    pred_run_dir = Path(pred_run_dir)
    legacy = pred_run_dir / "pred" / str(system) / "manifest.jsonl"
    if not legacy.is_file():
        return
    meta_dir = pred_run_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    dst = meta_dir / f"{system}.pred_manifest_legacy.jsonl"
    try:
        os.replace(str(legacy), str(dst))
    except Exception:
        try:
            legacy.unlink()
        except Exception:
            pass


def _clean_pred_dirs(pred_run_dir: Path) -> None:
    """Move any non-wav files out of pred/<system>/ dirs into meta/.

    Some downstream tooling naively tries to open every file in a directory.
    Keeping pred dirs wav-only avoids confusing such tools.
    """
    pred_run_dir = Path(pred_run_dir)
    pred_root = pred_run_dir / "pred"
    if not pred_root.is_dir():
        return
    meta_dir = pred_run_dir / "meta"
    for sys_dir in sorted(pred_root.iterdir()):
        if not sys_dir.is_dir():
            continue
        for p in sorted(sys_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() == ".wav":
                continue
            meta_dir.mkdir(parents=True, exist_ok=True)
            dst = meta_dir / f"{sys_dir.name}.{p.name}"
            try:
                os.replace(str(p), str(dst))
            except Exception:
                try:
                    p.unlink()
                except Exception:
                    pass

@dataclass(frozen=True)
class SystemSpec:
    name: str
    ckpt: Path
    cache: Path
    encoder_model: Optional[str] = None  # override


def _parse_system_spec(raw: str) -> SystemSpec:
    # Format: name:ckpt:cache[:encoder_model]
    parts = [p.strip() for p in str(raw).split(":") if p.strip() != ""]
    if len(parts) not in (3, 4):
        _exit_with_error("Invalid --system. Expected name:ckpt:cache[:encoder_model]")
    name, ckpt_s, cache_s = parts[0], parts[1], parts[2]
    enc = parts[3] if len(parts) == 4 else None
    return SystemSpec(name=str(name), ckpt=Path(ckpt_s), cache=Path(cache_s), encoder_model=enc)


def _read_systems_file(path: Path) -> List[str]:
    """Read a systems file containing one system spec per line.

    Format: same as --system (name:ckpt:cache[:encoder_model]), with optional
    comments starting with '#'.
    """
    path = Path(path)
    if not path.is_file():
        _exit_with_error(f"--systems-file not found: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _npz_item_key(npz_path: Path) -> Tuple[str, str, int, int, int]:
    with np.load(npz_path, allow_pickle=False) as d:
        audio_path = str(d["audio_path"].item())
        midi_path = str(d["midi_path"].item()) if "midi_path" in d else ""
        sr = int(d["sr"].item())
        start_sec = float(d["start_sec"].item())
        window_seconds = float(d["window_seconds"].item())
    start_sample = int(round(start_sec * float(sr)))
    window_samples = int(round(window_seconds * float(sr)))
    return (audio_path, midi_path, int(sr), int(start_sample), int(window_samples))


def _stable_key_str(key: Tuple[str, str, int, int, int]) -> str:
    a, m, sr, s0, n = key
    return hashlib.sha1(f"{a}|{m}|{sr}|{s0}|{n}".encode("utf-8")).hexdigest()[:16]


def _load_audio_segment(path: Path, *, start_sample: int, num_samples: int) -> Tuple[np.ndarray, int]:
    path = Path(path)
    if num_samples <= 0:
        return np.zeros((0,), dtype=np.float32), 0

    try:  # pragma: no cover - optional dependency
        import soundfile as sf  # type: ignore

        with sf.SoundFile(str(path)) as f:
            sr = int(f.samplerate)
            f.seek(max(0, int(start_sample)))
            y = f.read(frames=int(num_samples), dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=1)
        return y, sr
    except Exception:
        pass

    try:  # pragma: no cover - optional dependency
        import torchaudio  # type: ignore

        # torchaudio.load supports frame_offset/num_frames.
        wav, sr = torchaudio.load(str(path), frame_offset=int(max(0, start_sample)), num_frames=int(num_samples))
        y = wav.detach().cpu().numpy()
        if y.ndim == 2:
            y = y.mean(axis=0)
        return y.astype(np.float32, copy=False), int(sr)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load audio segment from {path}: {e}")


def _resample_linear(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr_in = int(sr_in)
    sr_out = int(sr_out)
    if y.size == 0 or sr_in <= 0 or sr_out <= 0 or sr_in == sr_out:
        return y
    dur = float(y.size) / float(sr_in)
    n_out = int(round(dur * float(sr_out)))
    if n_out <= 0:
        return np.zeros((0,), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=y.size, endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, y).astype(np.float32, copy=False)


def _resample_poly(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Higher-quality resampling using scipy.signal.resample_poly (CPU)."""
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
        return _resample_linear(y, sr_in, sr_out)


def _sha1_hex_of_iter(xs: Iterable[str]) -> str:
    h = hashlib.sha1()
    for s in xs:
        h.update(str(s).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _fadtk_device_to_torch(device: str):
    torch = _require_torch()
    d = str(device).strip().lower()
    if d == "" or d == "cpu":
        return torch.device("cpu")
    if d.startswith("cuda"):
        # fadtk's ModelLoader.get_embedding() has a strict equality check:
        #   if self.device == torch.device('cuda'): embd = embd.cpu()
        # which fails for torch.device('cuda:0') etc and crashes when converting
        # to numpy. Work around it by:
        #   (1) setting the current CUDA device when an index is given, and
        #   (2) returning torch.device('cuda') (no index) so the check passes.
        idx = None
        if ":" in d:
            try:
                idx = int(d.split(":", 1)[1])
            except Exception:
                idx = None
        if idx is not None:
            try:
                torch.cuda.set_device(int(idx))  # type: ignore[attr-defined]
            except Exception:
                pass
        return torch.device("cuda")
    raise ValueError(f"Unsupported --fadtk-device={device!r} (use cpu or cuda[:idx])")


def _fadtk_load_ref_stats(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        mu = np.asarray(d["mu"], dtype=np.float64)
        cov = np.asarray(d["cov"], dtype=np.float64)
        meta_s = d["meta"].item() if "meta" in d else "{}"
    try:
        meta = json.loads(str(meta_s))
    except Exception:
        meta = {}
    return mu, cov, meta


def _fadtk_save_ref_stats(path: Path, *, mu: np.ndarray, cov: np.ndarray, meta: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, mu=np.asarray(mu, dtype=np.float64), cov=np.asarray(cov, dtype=np.float64), meta=json.dumps(meta, sort_keys=True))


def _fadtk_fad_inf(
    fadtk_fad_mod: Any,
    *,
    mu_base: np.ndarray,
    cov_base: np.ndarray,
    embeds: np.ndarray,
    steps: int,
    min_n: int,
    max_n: int,
    seed: int,
) -> Dict[str, Any]:
    """Compute FAD∞ by regressing FAD vs 1/n on random resamples (with replacement)."""
    embeds = np.asarray(embeds, dtype=np.float64)
    n_total = int(embeds.shape[0])
    if n_total < 2:
        return {"fad_inf": float("nan"), "r2": float("nan"), "slope": float("nan"), "points": [], "n_total": n_total}

    min_n = int(min_n)
    max_n = int(max_n)
    if max_n <= 0:
        max_n = n_total
    max_n = min(max_n, n_total)
    min_n = max(2, min(min_n, max_n))
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
    return {
        "fad_inf": float(intercept),
        "slope": float(slope),
        "r2": float(r2),
        "points": [(int(n), float(v)) for (n, v) in points],
        "n_total": int(n_total),
        "min_n": int(min_n),
        "max_n": int(max_n),
        "steps": int(steps),
    }

def _rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = int(min(pred.size, ref.size))
    if n <= 0:
        return float("nan")
    d = pred[:n] - ref[:n]
    return float(np.sqrt(np.mean(d * d)))


def _mae(pred: np.ndarray, ref: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = int(min(pred.size, ref.size))
    if n <= 0:
        return float("nan")
    d = pred[:n] - ref[:n]
    return float(np.mean(np.abs(d)))


def _rms_envelope(y: np.ndarray, *, sr: int, win_ms: float = 50.0, hop_ms: float = 10.0, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    sr = int(sr)
    if y.size <= 0 or sr <= 0:
        return np.zeros((0,), dtype=np.float64)
    win_len = int(round(float(sr) * float(win_ms) / 1000.0))
    hop_len = int(round(float(sr) * float(hop_ms) / 1000.0))
    win_len = max(2, min(win_len, int(y.size)))
    hop_len = max(1, hop_len)

    if y.size < win_len:
        return np.asarray([float(np.sqrt(float(np.mean(y * y)) + float(eps)))], dtype=np.float64)

    s2 = y * y
    cs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(s2, dtype=np.float64)])
    starts = np.arange(0, int(y.size) - int(win_len) + 1, int(hop_len), dtype=np.int64)
    ends = starts + int(win_len)
    sums = cs[ends] - cs[starts]
    return np.sqrt(sums / float(win_len) + float(eps)).astype(np.float64, copy=False)


def _pearson_corr(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = int(min(a.size, b.size))
    if n <= 1:
        return float("nan")
    a = a[:n]
    b = b[:n]
    am = float(a.mean())
    bm = float(b.mean())
    da = a - am
    db = b - bm
    num = float(np.dot(da, db))
    den = float(np.linalg.norm(da) * np.linalg.norm(db)) + float(eps)
    return float(num / den)


def _envelope_rms_corr(pred: np.ndarray, ref: np.ndarray, *, sr: int) -> float:
    ep = _rms_envelope(pred, sr=int(sr))
    er = _rms_envelope(ref, sr=int(sr))
    return float(_pearson_corr(ep, er))


def _windowed_tter_db(
    y: np.ndarray,
    *,
    sr: int,
    win_ms: float = 300.0,
    hop_ms: float = 100.0,
    attack_frac: float = 0.25,
    eps: float = 1e-8,
) -> float:
    """Windowed Transient-to-Tail Energy Ratio (TTER) in dB, averaged over windows."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    sr = int(sr)
    if y.size <= 1 or sr <= 0:
        return float("nan")

    win_len = int(round(float(sr) * float(win_ms) / 1000.0))
    hop_len = int(round(float(sr) * float(hop_ms) / 1000.0))
    win_len = max(2, min(win_len, int(y.size)))
    hop_len = max(1, hop_len)

    if y.size < win_len:
        win_len = int(y.size)

    attack_frac = float(attack_frac)
    if not math.isfinite(attack_frac) or attack_frac <= 0.0:
        attack_frac = 0.25
    if attack_frac >= 1.0:
        attack_frac = 0.99
    attack_len = max(1, int(round(attack_frac * float(win_len))))
    attack_len = min(attack_len, win_len - 1)

    s2 = y * y
    cs = np.concatenate([np.zeros((1,), dtype=np.float64), np.cumsum(s2, dtype=np.float64)])
    starts = np.arange(0, int(y.size) - int(win_len) + 1, int(hop_len), dtype=np.int64)
    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int64)

    ratios: List[float] = []
    for s in starts.tolist():
        s0 = int(s)
        s1 = int(s0 + win_len)
        a0 = s0
        a1 = int(s0 + attack_len)
        t0 = a1
        t1 = s1
        e_att = float(cs[a1] - cs[a0]) + float(eps)
        e_tail = float(cs[t1] - cs[t0]) + float(eps)
        ratios.append(float(10.0 * math.log10(e_att / e_tail)))
    return float(np.mean(ratios)) if ratios else float("nan")


def _stft_mag(y: np.ndarray, *, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n_fft = int(n_fft)
    hop = int(hop)
    if y.size < n_fft or n_fft <= 0 or hop <= 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    n_frames = 1 + (y.size - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frame = y[s : s + n_fft]
        frame = frame * win
        spec = np.fft.rfft(frame, n=n_fft)
        out[i] = np.abs(spec).astype(np.float32, copy=False)
    return out


def _stft_sc(pred: np.ndarray, ref: np.ndarray, *, n_fft: int = 1024, hop: int = 256) -> float:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    n = int(min(pred.size, ref.size))
    if n <= 0:
        return float("nan")
    pred = pred[:n]
    ref = ref[:n]
    win = np.hanning(int(n_fft)).astype(np.float32)
    mp = _stft_mag(pred, n_fft=n_fft, hop=hop, win=win)
    mr = _stft_mag(ref, n_fft=n_fft, hop=hop, win=win)
    k = int(min(mp.shape[0], mr.shape[0]))
    if k <= 0:
        return float("nan")
    mp = mp[:k]
    mr = mr[:k]
    diff = mp - mr
    return float(np.linalg.norm(diff) / max(1e-9, float(np.linalg.norm(mr))))


def _mr_stft_sc(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    resolutions: Iterable[Tuple[int, int]] = ((512, 128), (1024, 256), (2048, 512)),
) -> float:
    scs: List[float] = []
    for n_fft, hop in resolutions:
        scs.append(float(_stft_sc(pred, ref, n_fft=int(n_fft), hop=int(hop))))
    scs = [float(x) for x in scs if math.isfinite(float(x))]
    return float(np.mean(scs)) if scs else float("nan")


def _onsets_from_audio(
    y: np.ndarray,
    *,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    min_separation_s: float = 0.05,
    z_thresh: float = 1.5,
    rms_gate_db: float = 35.0,
    backtrack_ms: float = 0.0,
    refine_ms: float = 0.0,
) -> np.ndarray:
    """Lightweight onset detector for percussive audio.

    Returns onset sample indices (sorted, unique-ish) at the given sr.

    Implementation notes:
      - Uses a simple spectral-flux style onset strength (positive STFT mag diffs),
        but with a few robustness tweaks to reduce common false positives when
        codec-decoded audio contains ringing/quantization artifacts:
          * frequency band selection (ignore very low/high bins)
          * log-magnitude compression
          * short smoothing of onset strength
          * non-maximum suppression (keep strongest peak per refractory window)
          * energy gate: ignore peaks whose local RMS is far below the loudest
            frame (reduces spurious triggers in silence/noise-floor)
          * optional backtracking of detected peaks to an earlier frame based on
            RMS rise (helps align detections to attacks)
          * optional sub-hop refinement that moves each onset to the maximum
            transient (|diff| peak) in a small neighborhood, which reduces hop
            quantization and better aligns detections to attacks.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr = int(sr)
    n_fft = int(n_fft)
    hop = int(hop)
    if y.size < max(4, n_fft) or sr <= 0 or n_fft <= 0 or hop <= 0:
        return np.zeros((0,), dtype=np.int64)

    # Prefer a well-tested MIR onset detector if available.
    # Fallback to the local implementation on any import/runtime issues.
    backend = str(os.environ.get("MIDIGROOVE_ONSET_BACKEND", "") or "").strip().lower()
    use_madmom = backend in ("", "madmom", "auto")
    if backend in ("native", "numpy", "local"):
        use_madmom = False
    if use_madmom:
        madmom = _try_madmom()
        if madmom is not None:
            try:  # pragma: no cover - depends on optional madmom install
                from madmom.audio.signal import Signal  # type: ignore
                from madmom.features.onsets import OnsetPeakPickingProcessor, SuperFluxProcessor  # type: ignore

                fps = float(sr) / float(hop)
                # SuperFlux: solid default for percussive onsets.
                sig = None
                try:
                    sig = Signal(y, sample_rate=sr, num_channels=1)  # type: ignore[arg-type]
                except Exception:
                    sig = Signal(y, sample_rate=sr)  # type: ignore[arg-type]

                act = SuperFluxProcessor(fps=fps)(sig)

                # Peak picking in seconds; `combine` enforces minimum separation.
                # `threshold` is relative to activation; keep conservative defaults.
                pp = OnsetPeakPickingProcessor(
                    fps=fps,
                    threshold=0.30,
                    pre_max=0.03,
                    post_max=0.03,
                    pre_avg=0.10,
                    post_avg=0.10,
                    combine=float(min_separation_s),
                    delay=0.0,
                )
                onsets_s = pp(act)
                onsets_s = np.asarray(onsets_s, dtype=np.float64).reshape(-1)
                if onsets_s.size == 0:
                    return np.zeros((0,), dtype=np.int64)
                onset_samples = np.round(onsets_s * float(sr)).astype(np.int64, copy=False)
                onset_samples = onset_samples[(onset_samples >= 0) & (onset_samples < int(y.size))]

                # Optional energy gate (computed on waveform frames aligned to hop).
                rms_gate_db = float(rms_gate_db)
                if rms_gate_db > 0.0 and onset_samples.size > 0:
                    eps = 1e-12
                    K = int(math.ceil(float(y.size) / float(hop)))
                    rms = np.zeros((K,), dtype=np.float32)
                    for fi in range(K):
                        s0 = int(fi) * int(hop)
                        s1 = min(int(y.size), s0 + int(n_fft))
                        if s1 <= s0:
                            continue
                        frame = y[s0:s1].astype(np.float32, copy=False)
                        rms[fi] = float(np.sqrt(float(np.mean(frame * frame)) + float(eps)))
                    rms_db = 20.0 * np.log10(np.maximum(rms, eps)).astype(np.float32, copy=False)
                    thr_db = float(np.max(rms_db)) - float(rms_gate_db)
                    keep = []
                    for s in onset_samples.tolist():
                        fi = int(round(float(s) / float(hop)))
                        if 0 <= fi < int(rms_db.size) and float(rms_db[fi]) >= float(thr_db):
                            keep.append(int(s))
                    onset_samples = np.asarray(keep, dtype=np.int64)

                # Optional backtracking and refinement (kept consistent with local implementation).
                bt_ms = float(backtrack_ms)
                if bt_ms > 0.0 and onset_samples.size > 0:
                    max_bt = int(round(bt_ms * 1e-3 * float(sr)))
                    max_bt = max(1, int(max_bt))
                    env = np.sqrt(np.convolve((y * y).astype(np.float32), np.ones((max(8, int(0.002 * sr)),), dtype=np.float32), mode="same"))
                    for ii in range(int(onset_samples.size)):
                        s = int(onset_samples[ii])
                        a0 = max(0, s - max_bt)
                        a1 = min(int(env.size), s + 1)
                        if a1 <= a0:
                            continue
                        seg = env[a0:a1]
                        j_min = int(np.argmin(seg))
                        thr = float(seg[j_min]) * 1.5
                        j = j_min
                        while j < seg.size and float(seg[j]) < thr:
                            j += 1
                        onset_samples[ii] = int(a0 + min(j, seg.size - 1))

                rm = float(refine_ms)
                if rm > 0.0 and onset_samples.size > 0:
                    R = int(round(rm * 1e-3 * float(sr)))
                    R = max(1, int(R))
                    dy = np.empty_like(y)
                    dy[0] = 0.0
                    dy[1:] = y[1:] - y[:-1]
                    tproxy = np.maximum(dy, 0.0).astype(np.float32, copy=False)
                    refine_frac = 0.20
                    refined = onset_samples.copy()
                    for ii in range(int(refined.size)):
                        s = int(refined[ii])
                        a0 = max(0, s - R)
                        a1 = min(int(tproxy.size), s + R + 1)
                        if a1 <= a0:
                            continue
                        seg = tproxy[a0:a1]
                        j_peak = int(np.argmax(seg))
                        peak = float(seg[j_peak])
                        if not np.isfinite(peak) or peak <= 0.0:
                            refined[ii] = int(a0 + j_peak)
                            continue
                        thr = float(refine_frac) * float(peak)
                        before = seg[: j_peak + 1]
                        mask = before >= thr
                        j_start = int(np.argmax(mask)) if bool(np.any(mask)) else int(j_peak)
                        refined[ii] = int(a0 + j_start)
                    onset_samples = refined

                if onset_samples.size <= 1:
                    return onset_samples.astype(np.int64, copy=False)
                return np.unique(onset_samples.astype(np.int64, copy=False))
            except Exception:
                # If madmom is present but fails for any reason, fall back silently.
                pass

    # Simple high-pass to emphasize transients.
    y_hp = np.empty_like(y)
    y_hp[0] = y[0]
    y_hp[1:] = y[1:] - y[:-1]

    win = np.hanning(n_fft).astype(np.float32)
    mags = _stft_mag(y_hp, n_fft=n_fft, hop=hop, win=win)  # [frames, bins]
    if mags.shape[0] < 4:
        return np.zeros((0,), dtype=np.int64)

    # Band-limit flux to reduce low-frequency rumble and very-high noise.
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0 / float(sr))
    f_lo = 80.0
    f_hi = min(8000.0, 0.49 * float(sr))
    band = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    if not np.any(band):
        band = slice(None)

    # Log-magnitude compression makes flux less sensitive to overall loudness.
    mags = np.log1p(np.maximum(0.0, mags)).astype(np.float32, copy=False)

    # Spectral flux (half-wave rectified frame-to-frame magnitude increase).
    dm = np.maximum(0.0, mags[1:] - mags[:-1])  # [frames-1, bins]
    flux = dm[:, band].sum(axis=1).astype(np.float32, copy=False)  # length ~= n_frames-1

    # Frame RMS for energy gating (computed on the original waveform, aligned to STFT frames).
    rms_gate_db = float(rms_gate_db)
    rms = None
    rms_db = None
    if rms_gate_db > 0.0 and int(mags.shape[0]) > 0:
        K = int(mags.shape[0])  # number of STFT frames in mags
        eps = 1e-12
        rms = np.zeros((K,), dtype=np.float32)
        for fi in range(K):
            s0 = int(fi) * int(hop)
            s1 = min(int(y.size), s0 + int(n_fft))
            if s1 <= s0:
                continue
            frame = y[s0:s1].astype(np.float32, copy=False)
            rms[fi] = float(np.sqrt(float(np.mean(frame * frame)) + float(eps)))
        rms_db = 20.0 * np.log10(np.maximum(rms, eps)).astype(np.float32, copy=False)
        thr_db = float(np.max(rms_db)) - float(rms_gate_db)
    else:
        thr_db = float("-inf")

    # Robust normalization.
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med)))
    scale = float(mad * 1.4826)  # approx std if normal
    z = (flux - med) / max(1e-6, scale)

    # Short smoothing (3-frame moving average) to reduce double-triggering.
    if z.size >= 5:
        z = np.convolve(z.astype(np.float32), np.ones((3,), dtype=np.float32) / 3.0, mode="same").astype(np.float32)

    # Peak picking (local maxima above threshold).
    cand: List[Tuple[int, float]] = []
    for i in range(1, int(z.size) - 1):
        zi = float(z[i])
        if not (zi > float(z[i - 1]) and zi >= float(z[i + 1])):
            continue
        if zi < float(z_thresh):
            continue
        # Energy gate: flux index i corresponds to STFT frame (i+1).
        if rms_db is not None:
            fi = int(i) + 1
            if 0 <= fi < int(rms_db.shape[0]) and float(rms_db[fi]) < float(thr_db):
                continue
        cand.append((int(i), zi))
    if not cand:
        return np.zeros((0,), dtype=np.int64)

    # Non-maximum suppression: keep the strongest peak per refractory window.
    min_dist_frames = int(round(float(min_separation_s) * float(sr) / float(hop)))
    min_dist_frames = max(1, int(min_dist_frames))
    cand.sort(key=lambda t: float(t[1]), reverse=True)
    kept: List[int] = []
    for idx, _score in cand:
        if all(abs(int(idx) - int(k)) >= int(min_dist_frames) for k in kept):
            kept.append(int(idx))
    kept.sort()

    # Convert peak indices (on flux) to frame indices in mags:
    # flux[t] corresponds to mags[t+1] - mags[t], so align to (t+1).
    onset_frames = (np.asarray(kept, dtype=np.int64) + 1).astype(np.int64)

    # Optional backtracking: move each onset earlier based on RMS rise.
    bt_ms = float(backtrack_ms)
    if bt_ms > 0.0 and rms is not None and isinstance(rms, np.ndarray) and rms.size > 0 and onset_frames.size > 0:
        max_bt_frames = int(round((bt_ms * 1e-3) * float(sr) / float(hop)))
        max_bt_frames = max(1, int(max_bt_frames))
        rise_factor = 1.5
        of = onset_frames.copy()
        for ii in range(int(of.size)):
            f = int(of[ii])
            j0 = max(0, f - max_bt_frames)
            j1 = min(int(rms.size) - 1, f)
            if j1 <= j0:
                continue
            seg = rms[j0 : j1 + 1]
            kmin = int(np.argmin(seg))
            fmin = int(j0 + kmin)
            rmin = float(rms[fmin])
            thr = float(rmin) * float(rise_factor)
            f_bt = fmin
            for jj in range(fmin, f + 1):
                if float(rms[jj]) >= thr:
                    f_bt = int(jj)
                    break
            of[ii] = int(f_bt)
        onset_frames = of

    onset_samples = onset_frames * int(hop)
    onset_samples = onset_samples[(onset_samples >= 0) & (onset_samples < int(y.size))]

    # Optional sub-hop refinement: within +/- refine_ms, move each onset to the
    # *attack start* rather than the flux peak.
    #
    # We do this by:
    #   1) finding the strongest local transient in a +/- window (peak),
    #   2) backtracking within that window to the first sample where the
    #      transient proxy crosses a fraction of that peak.
    #
    # This reduces hop quantization while avoiding a systematic late bias
    # (onset strength peaks tend to occur a few ms after the perceptual attack).
    rm = float(refine_ms)
    if rm > 0.0 and onset_samples.size > 0:
        R = int(round(rm * 1e-3 * float(sr)))
        R = max(1, int(R))
        # Transient proxy: positive time-derivative emphasizes attack rises.
        dy = np.empty_like(y)
        dy[0] = 0.0
        dy[1:] = y[1:] - y[:-1]
        tproxy = np.maximum(dy, 0.0).astype(np.float32, copy=False)
        refine_frac = 0.20  # fraction of peak used to define "attack start"
        refined = onset_samples.copy()
        for ii in range(int(refined.size)):
            s = int(refined[ii])
            a0 = max(0, s - R)
            a1 = min(int(tproxy.size), s + R + 1)
            if a1 <= a0:
                continue
            seg = tproxy[a0:a1]
            j_peak = int(np.argmax(seg))
            peak = float(seg[j_peak])
            if not np.isfinite(peak) or peak <= 0.0:
                refined[ii] = int(a0 + j_peak)
                continue

            thr = float(refine_frac) * float(peak)
            before = seg[: j_peak + 1]
            mask = before >= thr
            if bool(np.any(mask)):
                j_start = int(np.argmax(mask))  # first crossing
            else:
                j_start = int(j_peak)

            refined[ii] = int(a0 + j_start)
        onset_samples = refined

    if onset_samples.size <= 1:
        return onset_samples.astype(np.int64, copy=False)
    return np.unique(onset_samples.astype(np.int64, copy=False))


def _onsets_from_grid_npz(
    npz_path: Path,
    *,
    eval_sr: int,
    hit_thresh: float = 0.2,
    vel_thresh: float = 0.0,
    exclude_channels: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Reference onsets from cached conditioning grids.

    This is more stable than parsing MIDI at evaluation time and is exactly
    aligned to the model conditioning.

    Strategy:
      1) Prefer drum_vel (sparse, true onset frames). Optionally apply an
         energy gate via `vel_thresh` to drop extremely quiet/ghost hits that
         are often not detectable as waveform onsets.
      2) Fall back to drum_hit by taking local maxima above `hit_thresh`.

    Returns onset sample indices *relative to the window start* at eval_sr.
    """
    npz_path = Path(npz_path)
    eval_sr = int(eval_sr)
    if eval_sr <= 0:
        return np.zeros((0,), dtype=np.int64)

    channels: Optional[List[str]] = None
    try:
        with np.load(npz_path, allow_pickle=False) as d:
            fps = float(d["fps"].item()) if "fps" in d else float("nan")
            win_s = float(d["window_seconds"].item()) if "window_seconds" in d else float("nan")
            vel = np.asarray(d["drum_vel"], dtype=np.float32) if "drum_vel" in d else None
            hit = np.asarray(d["drum_hit"], dtype=np.float32) if "drum_hit" in d else None
            sem_raw = d["semantics"].item() if "semantics" in d else None
        if sem_raw is not None:
            try:
                sem = json.loads(str(sem_raw))
                ch = sem.get("channels", None)
                if isinstance(ch, list) and all(isinstance(x, str) for x in ch):
                    channels = [str(x) for x in ch]
            except Exception:
                channels = None
    except Exception:
        return np.zeros((0,), dtype=np.int64)

    # Build a lane mask from channel names, if available.
    keep_rows: Optional[np.ndarray] = None
    if exclude_channels:
        excl = {str(x).strip() for x in exclude_channels if str(x).strip()}
        if excl and channels and vel is not None and vel.ndim == 2 and len(channels) == int(vel.shape[0]):
            keep_rows = np.asarray([c not in excl for c in channels], dtype=bool)
        elif excl and channels and hit is not None and hit.ndim == 2 and len(channels) == int(hit.shape[0]):
            keep_rows = np.asarray([c not in excl for c in channels], dtype=bool)

    on_frames: np.ndarray
    T: int
    if vel is not None and isinstance(vel, np.ndarray) and vel.ndim == 2 and vel.size > 0:
        vel_use = vel[keep_rows, :] if (keep_rows is not None) else vel
        v = np.max(vel_use, axis=0).astype(np.float32, copy=False)  # [T]
        T = int(v.shape[0])
        thr_v = float(vel_thresh)
        on_frames = np.flatnonzero(v > thr_v).astype(np.int64, copy=False)
    elif hit is not None and isinstance(hit, np.ndarray) and hit.ndim == 2 and hit.size > 0:
        hit_use = hit[keep_rows, :] if (keep_rows is not None) else hit
        h = np.max(hit_use, axis=0).astype(np.float32, copy=False)  # [T]
        T = int(h.shape[0])
        thr = float(hit_thresh)
        if T < 3:
            on_frames = np.flatnonzero(h > thr).astype(np.int64, copy=False)
        else:
            mid = (h[1:-1] >= h[:-2]) & (h[1:-1] > h[2:]) & (h[1:-1] > thr)
            on_frames = (np.flatnonzero(mid) + 1).astype(np.int64, copy=False)
    else:
        return np.zeros((0,), dtype=np.int64)

    if not (fps and math.isfinite(float(fps)) and float(fps) > 1e-6):
        if win_s and math.isfinite(float(win_s)) and float(win_s) > 1e-6:
            fps = float(T) / float(win_s)
        else:
            fps = 50.0

    on_samps = np.round((on_frames.astype(np.float64) / float(fps)) * float(eval_sr)).astype(np.int64, copy=False)
    if win_s and math.isfinite(float(win_s)) and float(win_s) > 0.0:
        n = int(round(float(win_s) * float(eval_sr)))
        on_samps = on_samps[(on_samps >= 0) & (on_samps < max(1, n))]
    if on_samps.size <= 1:
        return on_samps.astype(np.int64, copy=False)
    return np.unique(on_samps.astype(np.int64, copy=False))


def _snap_ref_onsets_to_audio(
    ref_onsets: np.ndarray,
    audio_onsets: np.ndarray,
    *,
    max_shift_samps: int,
) -> np.ndarray:
    """Snap reference onsets to the nearest detected onsets in the reference audio.

    This is a calibration step to reduce systematic offsets between symbolic/grid
    timing and waveform transients. Matching is greedy and monotonic (one-to-one),
    analogous to the onset P/R/F1 matcher.

    Args:
        ref_onsets: reference onset sample indices (sorted).
        audio_onsets: onset sample indices detected on the *reference* waveform (sorted).
        max_shift_samps: maximum allowed shift (in samples) to snap ref->audio.
    Returns:
        snapped reference onsets (sorted, unique).
    """
    r = np.asarray(ref_onsets, dtype=np.int64).reshape(-1)
    a = np.asarray(audio_onsets, dtype=np.int64).reshape(-1)
    max_shift_samps = int(max(0, max_shift_samps))
    if r.size == 0 or a.size == 0 or max_shift_samps <= 0:
        return r

    out = r.copy()
    i = 0
    j = 0
    while i < int(r.size) and j < int(a.size):
        dr = int(r[i])
        da = int(a[j])
        di = da - dr
        if abs(di) <= max_shift_samps:
            out[i] = da
            i += 1
            j += 1
            continue
        if dr < (da - max_shift_samps):
            i += 1
        else:
            j += 1

    if out.size <= 1:
        return out.astype(np.int64, copy=False)
    return np.unique(out.astype(np.int64, copy=False))


def _best_constant_shift_samples(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_shift_samps: int,
    tol_samps: int,
    step_samps: int = 1,
) -> int:
    """Find a single shift (samples) to apply to `a` that best matches `b`.

    Uses greedy monotonic matching within `tol_samps`. Maximizes true positives,
    then breaks ties by smaller mean abs error of matched pairs, then smaller
    |shift|.
    """
    a = np.asarray(a, dtype=np.int64).reshape(-1)
    b = np.asarray(b, dtype=np.int64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0
    max_shift_samps = int(max(0, max_shift_samps))
    tol_samps = int(max(1, tol_samps))
    step_samps = int(max(1, step_samps))

    def _match_stats(a_shifted: np.ndarray) -> Tuple[int, float]:
        i = j = tp = 0
        errs: List[float] = []
        while i < int(a_shifted.size) and j < int(b.size):
            di = int(a_shifted[i]) - int(b[j])
            if abs(di) <= tol_samps:
                tp += 1
                errs.append(float(abs(di)))
                i += 1
                j += 1
            elif int(a_shifted[i]) < int(b[j]) - tol_samps:
                i += 1
            else:
                j += 1
        mae = float(np.mean(errs)) if errs else float("inf")
        return int(tp), float(mae)

    best_s = 0
    best_tp = -1
    best_mae = float("inf")
    for s in range(-max_shift_samps, max_shift_samps + 1, step_samps):
        a_s = a + int(s)
        a_s = a_s[a_s >= 0]
        if a_s.size == 0:
            continue
        tp, mae = _match_stats(a_s)
        if (tp > best_tp) or (tp == best_tp and mae < best_mae) or (tp == best_tp and mae == best_mae and abs(s) < abs(best_s)):
            best_tp = int(tp)
            best_mae = float(mae)
            best_s = int(s)
    return int(best_s)


_MIDI_DRUM_ONSET_CACHE: Dict[str, np.ndarray] = {}


def _drum_onset_times_from_midi(midi_path: Path) -> np.ndarray:
    """Return sorted drum note-on times (seconds) from a MIDI file."""
    midi_path = Path(midi_path)
    key = str(midi_path)
    cached = _MIDI_DRUM_ONSET_CACHE.get(key)
    if cached is not None:
        return cached
    try:  # pragma: no cover - depends on local data files
        pretty_midi = _require_pretty_midi()
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        times: List[float] = []
        for inst in pm.instruments:
            if not bool(getattr(inst, "is_drum", False)):
                continue
            for n in getattr(inst, "notes", []):
                try:
                    times.append(float(n.start))
                except Exception:
                    continue
        arr = np.asarray(sorted(times), dtype=np.float64)
    except Exception:
        arr = np.zeros((0,), dtype=np.float64)
    _MIDI_DRUM_ONSET_CACHE[key] = arr
    return arr


def _onsets_from_midi_segment(
    midi_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    sr: int,
) -> np.ndarray:
    """Reference onsets from MIDI (drum note starts) for a [start_sec, end_sec) segment."""
    sr = int(sr)
    start_sec = float(start_sec)
    end_sec = float(end_sec)
    if sr <= 0 or not math.isfinite(start_sec) or not math.isfinite(end_sec) or end_sec <= start_sec:
        return np.zeros((0,), dtype=np.int64)
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        return np.zeros((0,), dtype=np.int64)
    times = _drum_onset_times_from_midi(midi_path)
    if times.size == 0:
        return np.zeros((0,), dtype=np.int64)
    i0 = int(np.searchsorted(times, start_sec, side="left"))
    i1 = int(np.searchsorted(times, end_sec, side="left"))
    seg = times[i0:i1]
    if seg.size == 0:
        return np.zeros((0,), dtype=np.int64)
    rel = seg - float(start_sec)
    samples = np.round(rel * float(sr)).astype(np.int64, copy=False)
    n = int(round((float(end_sec) - float(start_sec)) * float(sr)))
    samples = samples[(samples >= 0) & (samples < max(1, n))]
    if samples.size <= 1:
        return samples
    # De-dup (multiple drums at exact same time).
    return np.unique(samples)


def _onset_pr_metrics(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    sr: int,
    pred_onsets: Optional[np.ndarray] = None,
    ref_onsets: Optional[np.ndarray] = None,
    midi_path: Optional[Path],
    start_sec: Optional[float],
    end_sec: Optional[float],
    tol_ms: float = 50.0,
) -> Dict[str, float]:
    """Onset precision/recall/F1.

    Reference onsets are preferably provided explicitly. If not provided, we
    take reference onsets from MIDI when available; otherwise we fall back to
    onset detection on the reference waveform.
    """
    sr = int(sr)
    tol = int(round(float(tol_ms) * 1e-3 * float(sr)))
    tol = max(1, tol)

    if pred_onsets is None:
        p = _onsets_from_audio(pred, sr=sr)
    else:
        p = np.asarray(pred_onsets, dtype=np.int64).reshape(-1)
    r: np.ndarray
    if ref_onsets is not None:
        r = np.asarray(ref_onsets, dtype=np.int64).reshape(-1)
    elif midi_path is not None and start_sec is not None and end_sec is not None:
        r = _onsets_from_midi_segment(Path(midi_path), start_sec=float(start_sec), end_sec=float(end_sec), sr=sr)
        if r.size == 0:
            # fallback if MIDI contained no drum notes in-window (or parsing failed)
            r = _onsets_from_audio(ref, sr=sr)
    else:
        r = _onsets_from_audio(ref, sr=sr)

    if p.size == 0 and r.size == 0:
        return {"onset_precision": 1.0, "onset_recall": 1.0, "onset_f1": 1.0}
    if p.size == 0 or r.size == 0:
        return {"onset_precision": 0.0, "onset_recall": 0.0, "onset_f1": 0.0}

    # Prefer madmom's evaluation utilities if available.
    madmom = _try_madmom()
    if madmom is not None:
        try:  # pragma: no cover - optional dependency
            from madmom.evaluation.onsets import OnsetEvaluation  # type: ignore

            det_t = (p.astype(np.float64) / float(sr)).tolist()
            ann_t = (r.astype(np.float64) / float(sr)).tolist()
            ev = OnsetEvaluation(det_t, ann_t, window=float(tol_ms) * 1e-3)
            return {
                "onset_precision": float(getattr(ev, "precision", 0.0)),
                "onset_recall": float(getattr(ev, "recall", 0.0)),
                "onset_f1": float(getattr(ev, "fmeasure", 0.0)),
            }
        except Exception:
            pass

    # Greedy monotonic matching fallback.
    i = 0
    j = 0
    tp = 0
    while i < int(p.size) and j < int(r.size):
        di = int(p[i]) - int(r[j])
        if abs(di) <= tol:
            tp += 1
            i += 1
            j += 1
            continue
        if int(p[i]) < int(r[j]) - tol:
            i += 1
        else:
            j += 1

    prec = float(tp) / max(1.0, float(p.size))
    rec = float(tp) / max(1.0, float(r.size))
    f1 = (2.0 * prec * rec / max(1e-12, (prec + rec))) if (prec + rec) > 0.0 else 0.0
    return {"onset_precision": float(prec), "onset_recall": float(rec), "onset_f1": float(f1)}


def _mean_std(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray([x for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _invert_int_mapping(m: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k, v in (m or {}).items():
        try:
            out[int(v)] = str(k)
        except Exception:
            continue
    return out


def _load_system_model(spec: SystemSpec, *, device: str) -> Tuple[Any, Dict[str, Any], str]:
    torch = _require_torch()
    from . import expressivegrid as eg

    ckpt = torch.load(Path(spec.ckpt), map_location=torch.device(device))
    if not isinstance(ckpt, dict):
        _exit_with_error(f"{spec.name}: unexpected ckpt format: {type(ckpt)}")
    cfg = ckpt.get("cfg", {})
    if not isinstance(cfg, dict):
        cfg = {}
    num_codebooks = int(ckpt.get("num_codebooks", 0) or 0)
    in_dim = int(ckpt.get("in_dim", 0) or 0)
    if num_codebooks <= 0 or in_dim <= 0:
        _exit_with_error(f"{spec.name}: ckpt missing num_codebooks/in_dim")
    state = ckpt.get("model", None)
    if not isinstance(state, dict):
        _exit_with_error(f"{spec.name}: ckpt missing model state_dict")
    enc = str(spec.encoder_model or cfg.get("encoder_model", "encodec") or "encodec").strip().lower()
    cfg = dict(cfg)
    cfg["in_dim"] = int(in_dim)
    cfg["num_codebooks"] = int(num_codebooks)
    if "use_kit_name" not in cfg:
        cfg["use_kit_name"] = bool("kit_name_emb.weight" in state)
    # Back-compat defaults for vocab sizing (see midigroove_poc/expressivegrid.py).
    # If ckpt already contains vocab sizing, honor it (required to load the state_dict).
    if "vocab_size" not in cfg:
        try:
            vs = eg._infer_vocab_size_from_state_dict(state, num_codebooks=int(num_codebooks))  # type: ignore[attr-defined]
            if vs is not None:
                cfg["vocab_size"] = int(vs)
        except Exception:
            pass
    if "vocab_size" in cfg and ("pad_id" not in cfg or "codebook_size" not in cfg):
        try:
            vs2 = int(cfg.get("vocab_size", 0) or 0)
            if vs2 > 1:
                cfg.setdefault("pad_id", int(vs2 - 1))
                cfg.setdefault("codebook_size", int(vs2 - 1))
        except Exception:
            pass

    codebook_size = int(cfg.get("codebook_size", eg._default_codebook_size_for_encoder(enc)))  # type: ignore[attr-defined]
    pad_id = int(cfg.get("pad_id", eg._pad_id_for_codebook(codebook_size)))  # type: ignore[attr-defined]
    vocab_size = int(cfg.get("vocab_size", eg._vocab_size_for_codebook(codebook_size)))  # type: ignore[attr-defined]
    cfg["codebook_size"] = int(codebook_size)
    cfg["pad_id"] = int(pad_id)
    cfg["vocab_size"] = int(vocab_size)

    model = eg._build_model(num_codebooks=num_codebooks, in_dim=in_dim, cfg=cfg)  # type: ignore[attr-defined]
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, cfg, enc


def _split_paths(cache_dir: Path, split: str) -> List[Path]:
    from . import expressivegrid as eg

    splits = eg._split_items_by_manifest(Path(cache_dir))  # type: ignore[attr-defined]
    split = str(split).strip().lower()
    if split == "val":
        split = "validation"
    if split not in splits:
        _exit_with_error(f"Cache {cache_dir} missing split={split!r}; found: {sorted(splits.keys())}")
    return [Path(p) for p in splits[split]]


def _index_cache(npz_paths: Iterable[Path], *, limit: Optional[int] = None) -> Dict[Tuple[str, str, int, int, int], Path]:
    out: Dict[Tuple[str, str, int, int, int], Path] = {}
    for i, p in enumerate(npz_paths):
        if limit is not None and i >= int(limit):
            break
        try:
            k = _npz_item_key(Path(p))
        except Exception:
            continue
        out[k] = Path(p)
    return out


def _write_wav(path: Path, y: np.ndarray, sr: int) -> None:
    import wave

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm.tobytes())


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate expressivegrid->token checkpoints across codecs.")
    ap.add_argument("--system", action="append", default=None, help="name:ckpt:cache[:encoder_model]")
    ap.add_argument("--systems-file", action="append", default=None, help="Path to a text file with one system spec per line.")
    ap.add_argument("--split", type=str, default="test", help="Cache split to evaluate (train|validation|test).")
    ap.add_argument("--intersection", action="store_true", help="Evaluate on the intersection of cache segments (recommended).")
    ap.add_argument("--max-items", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--device", type=str, default="cpu", help="Torch device for model forward (e.g. cpu, cuda:0).")
    ap.add_argument("--decode-device", type=str, default=None, help="Device for codec decode (defaults to --device).")
    ap.add_argument("--eval-sr", type=int, default=32000, help="Sample rate for audio metrics and saved wavs.")
    ap.add_argument("--audio-metrics", action="store_true", help="Decode tokens to audio and compute audio metrics.")
    ap.add_argument("--add-oracle", action="store_true", help="Also evaluate the codec reconstruction baseline (decode ground-truth tokens).")
    ap.add_argument("--add-random", action="store_true", help="Also evaluate a random-tokens baseline (very low-quality).")
    ap.add_argument("--save-wavs", type=int, default=0, help="Save up to N items per system as wavs.")
    ap.add_argument(
        "--save-preds",
        type=int,
        default=0,
        help=(
            "Save decoded predictions under --pred-dir/<run>/pred/<system>/. "
            "<run> defaults to the final path component of --out-dir. 0=off, -1=all, N=up to N items per system."
        ),
    )
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=Path("artifacts/pred"),
        help="Base directory for --save-preds output (files go under --pred-dir/<run>/pred/<system>/).",
    )
    ap.add_argument("--pred-include-ref", action="store_true", help="When saving preds, also save references once under --pred-dir/<run>/ref/.")
    ap.add_argument(
        "--keep-preds",
        action="store_true",
        help="Keep wav outputs produced by --save-preds. Default: if --save-preds is a positive number, delete them after metrics finish.",
    )
    ap.add_argument(
        "--fad-fadtk",
        action="store_true",
        help=(
            "Compute Frechet Audio Distance using the external `fadtk` package (recommended). "
            "Embeddings are computed directly from the decoded audio during eval (no extra WAVs written). "
            "By default this also computes FAD∞ (sample-size extrapolated) as recommended by "
            "\"Adapting FAD for generative music\"."
        ),
    )
    ap.add_argument(
        "--fadtk-per-kit",
        action="store_true",
        help=(
            "Additionally compute FAD/FAD∞ per kit (from kit_name_id/kit label in the cache) and write a CSV in the eval out-dir. "
            "This uses the same decoded-audio embeddings as --fad-fadtk and does not require saving WAVs."
        ),
    )
    ap.add_argument(
        "--fadtk-per-kit-out",
        type=Path,
        default=None,
        help="Where to write per-kit fadtk CSV (default: <out_dir>/fadtk_per_kit.csv).",
    )
    ap.add_argument(
        "--fadtk-model",
        type=str,
        default="clap-laion-music",
        help="fadtk model name (e.g. clap-laion-music, clap-laion-audio, clap-2023).",
    )
    ap.add_argument(
        "--fadtk-device",
        type=str,
        default="cpu",
        help="Device for fadtk embedding inference (e.g. cpu, cuda:0). Use cpu to avoid VRAM contention with eval/decode.",
    )
    ap.add_argument(
        "--fadtk-max-items",
        type=int,
        default=0,
        help="Optional cap on number of items used for fadtk FAD (0 = all evaluated items).",
    )
    ap.add_argument(
        "--fadtk-ref-stats",
        type=Path,
        default=None,
        help="Optional path to a cached reference stats .npz (mu/cov) to reuse across experiments.",
    )
    ap.add_argument(
        "--fadtk-ref-stats-out",
        type=Path,
        default=None,
        help="Where to write reference stats .npz (defaults to artifacts/fadtk_ref_stats/<hash>.npz).",
    )
    ap.add_argument(
        "--no-fadtk-inf",
        action="store_true",
        help="Disable FAD∞ extrapolation; only compute plain FAD.",
    )
    ap.add_argument("--fadtk-inf-steps", type=int, default=15, help="Number of points for FAD∞ regression.")
    ap.add_argument("--fadtk-inf-min-n", type=int, default=128, help="Minimum sample size for FAD∞.")
    ap.add_argument("--fadtk-inf-max-n", type=int, default=5000, help="Maximum sample size for FAD∞ (caps cost on large sets).")
    ap.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars.")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/eval"))

    args = ap.parse_args(list(argv) if argv is not None else None)
    use_tqdm = not bool(args.no_tqdm)

    raw_specs: List[str] = []
    raw_specs.extend(list(args.system or []))
    for p in (args.systems_file or []):
        raw_specs.extend(_read_systems_file(Path(p)))

    systems = [_parse_system_spec(s) for s in raw_specs]
    if len(systems) < 1:
        _exit_with_error("Provide at least one --system (or --systems-file).")

    split = str(args.split)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index caches and compute standardized keys.
    per_system_index: Dict[str, Dict[Tuple[str, str, int, int, int], Path]] = {}
    for spec in _wrap_tqdm(
        systems,
        desc="Index caches",
        total=len(systems),
        disable=not use_tqdm,
        leave=False,
        unit="sys",
    ):
        paths = _split_paths(spec.cache, split)
        per_system_index[spec.name] = _index_cache(paths)
        if not per_system_index[spec.name]:
            _exit_with_error(f"{spec.name}: empty index for split={split!r} in cache={spec.cache}")

    if bool(args.intersection) and len(systems) > 1:
        keys = set(per_system_index[systems[0].name].keys())
        for spec in systems[1:]:
            keys &= set(per_system_index[spec.name].keys())
        selected_keys = sorted(keys)
    else:
        # Union (still deterministic).
        keys_union: set[Tuple[str, str, int, int, int]] = set()
        for spec in systems:
            keys_union |= set(per_system_index[spec.name].keys())
        selected_keys = sorted(keys_union)

    if not selected_keys:
        _exit_with_error("No evaluation items found (intersection/union is empty).")

    rng = np.random.default_rng(int(args.seed))
    if int(args.max_items) > 0 and len(selected_keys) > int(args.max_items):
        idx = rng.choice(len(selected_keys), size=int(args.max_items), replace=False)
        selected_keys = [selected_keys[int(i)] for i in sorted(idx.tolist())]

    torch = _require_torch()
    device = str(args.device)
    decode_device = str(args.decode_device) if args.decode_device is not None else device
    eval_sr = int(args.eval_sr)
    save_wavs = int(args.save_wavs)
    save_preds = int(args.save_preds)
    pred_dir = Path(args.pred_dir)
    pred_include_ref = bool(args.pred_include_ref)
    keep_preds = bool(args.keep_preds)
    pred_run = str(out_dir.name).strip() or "run"
    pred_run_dir = (pred_dir / pred_run) if (save_preds != 0) else None
    pred_ref_dir = (pred_run_dir / "ref") if (pred_run_dir is not None and pred_include_ref) else None
    pred_ref_by_system_dir = (pred_run_dir / "ref_by_system") if pred_ref_dir is not None else None
    do_audio = bool(args.audio_metrics) or (save_wavs != 0) or (save_preds != 0)
    add_oracle = bool(args.add_oracle)
    add_random = bool(args.add_random)

    from . import expressivegrid as eg
    from data.codecs import decode_tokens_to_audio

    summary: Dict[str, Any] = {
        "split": split,
        "intersection": bool(args.intersection),
        "n_items": int(len(selected_keys)),
        "eval_sr": int(eval_sr),
        "pred_dir": str(pred_run_dir) if pred_run_dir is not None else None,
        "pred_ref_dir": str(pred_ref_dir) if pred_ref_dir is not None else None,
        "pred_ref_by_system_dir": str(pred_ref_by_system_dir) if pred_ref_by_system_dir is not None else None,
        "keep_preds": bool(keep_preds),
        "systems": {},
    }
    rows: List[Dict[str, Any]] = []
    saved_ref_keys: set[str] = set()  # refs under pred_ref_dir (listening)

    if pred_run_dir is not None and save_preds != 0:
        _clean_pred_dirs(pred_run_dir)

    # -------------------- Optional package FAD via fadtk --------------------
    fadtk_enabled = bool(args.fad_fadtk)
    fadtk_per_kit = bool(getattr(args, "fadtk_per_kit", False))
    fadtk_ids: Optional[set[str]] = None
    fadtk_key_shorts: List[str] = []
    fadtk_ref_mu: Optional[np.ndarray] = None
    fadtk_ref_cov: Optional[np.ndarray] = None
    fadtk_ref_stats_path: Optional[Path] = None
    fadtk_ref_meta: Dict[str, Any] = {}
    fadtk_ref_by_item: Dict[str, np.ndarray] = {}
    fadtk_clip_dur_by_item_s: Dict[str, float] = {}
    fadtk_kit_by_item: Dict[str, str] = {}
    fadtk_ref_stats_by_kit: Dict[str, Tuple[np.ndarray, np.ndarray, int]] = {}
    fadtk_per_kit_rows: List[Dict[str, Any]] = []
    fadtk_ml = None
    fadtk_fad_mod = None

    if fadtk_enabled:
        if not bool(args.audio_metrics):
            raise RuntimeError("--fad-fadtk requires --audio-metrics (needs decoded audio).")

        fadtk = _require_fadtk()
        import fadtk.fad as _fadtk_fad  # type: ignore
        from fadtk.model_loader import get_all_models  # type: ignore

        fadtk_fad_mod = _fadtk_fad
        models = {m.name: m for m in get_all_models()}
        model_name = str(args.fadtk_model)
        if model_name not in models:
            raise RuntimeError(f"Unknown fadtk model {model_name!r}. Available: {sorted(models.keys())}")
        fadtk_ml = models[model_name]

        # Force device (fadtk model loaders default to cuda if available).
        fadtk_ml.device = _fadtk_device_to_torch(str(args.fadtk_device))
        fadtk_ml.load_model()

        # Choose the set of items to include (cap by --fadtk-max-items if provided).
        fadtk_max_items = int(args.fadtk_max_items or 0)
        fadtk_keys = list(selected_keys)
        if fadtk_max_items > 0 and len(fadtk_keys) > fadtk_max_items:
            fad_rng = np.random.default_rng(int(args.seed) + 1337)
            idx = fad_rng.choice(len(fadtk_keys), size=fadtk_max_items, replace=False)
            fadtk_keys = [fadtk_keys[int(i)] for i in sorted(idx.tolist())]
        fadtk_key_shorts = [_stable_key_str(k) for k in fadtk_keys]
        fadtk_ids = set(fadtk_key_shorts)
        fadtk_sel_sha1 = _sha1_hex_of_iter(fadtk_key_shorts)

        # Reference stats caching.
        if args.fadtk_ref_stats is not None:
            fadtk_ref_stats_path = Path(args.fadtk_ref_stats)
        elif args.fadtk_ref_stats_out is not None:
            fadtk_ref_stats_path = Path(args.fadtk_ref_stats_out)
        else:
            fadtk_ref_stats_path = Path("artifacts/fadtk_ref_stats") / f"{model_name}_{split}_{len(fadtk_key_shorts)}_{fadtk_sel_sha1[:12]}.npz"

        # Per-kit FAD needs per-item reference embeddings, so we never *load* cached ref stats in that mode
        # (but we still allow writing stats for reuse in other runs).
        if (not fadtk_per_kit) and fadtk_ref_stats_path.is_file():
            mu0, cov0, meta0 = _fadtk_load_ref_stats(fadtk_ref_stats_path)
            # If the user explicitly gave a stats file, require that it matches.
            want = fadtk_sel_sha1
            got = str(meta0.get("selection_sha1", "") or "")
            if args.fadtk_ref_stats is not None and got and got != want:
                raise RuntimeError(
                    f"--fadtk-ref-stats selection mismatch: file has selection_sha1={got}, "
                    f"but this eval run selection_sha1={want}."
                )
            if args.fadtk_ref_stats is None and got and got != want:
                # Don't silently use mismatched cached stats; recompute and overwrite instead.
                pass
            else:
                fadtk_ref_mu, fadtk_ref_cov, fadtk_ref_meta = mu0, cov0, meta0

        summary["fadtk"] = {
            "backend": "fadtk",
            "model": str(model_name),
            "device": str(args.fadtk_device),
            "model_sr": int(getattr(fadtk_ml, "sr", 0)),
            "dim": int(getattr(fadtk_ml, "num_features", 0)),
            "reference": {
                "type": "eval_reference_audio",
                "split": str(split),
                "intersection": bool(args.intersection),
                "n_items_eval": int(len(selected_keys)),
                "n_items_fad": int(len(fadtk_key_shorts)),
            },
            "max_items": int(args.fadtk_max_items),
            "n_used": int(len(fadtk_key_shorts)),
            "selection_sha1": str(fadtk_sel_sha1),
            "ref_stats_path": str(fadtk_ref_stats_path),
            "ref_stats_loaded": bool(fadtk_ref_mu is not None),
            "per_kit": bool(fadtk_per_kit),
            "fad_inf_enabled": (not bool(args.no_fadtk_inf)),
            "fad_inf_steps": int(args.fadtk_inf_steps),
            "fad_inf_min_n": int(args.fadtk_inf_min_n),
            "fad_inf_max_n": int(args.fadtk_inf_max_n),
            "embed_aggregation": "mean_over_timeframes",
        }

    for spec in _wrap_tqdm(systems, desc="Systems", total=len(systems), disable=not use_tqdm, leave=True, unit="sys"):
        model, cfg, encoder_model = _load_system_model(spec, device=device)
        if pred_run_dir is not None and save_preds != 0:
            _move_legacy_manifest(pred_run_dir=pred_run_dir, system=str(spec.name))

        # Build dataset for selected paths in this cache.
        paths = [per_system_index[spec.name].get(k) for k in selected_keys]
        paths = [p for p in paths if p is not None]
        if not paths:
            continue

        vocab = eg._load_vocab_from_cache(Path(spec.cache))  # type: ignore[attr-defined]
        # Prefer explicit ckpt cfg; otherwise infer from in_dim vs D in first item.
        include_sustain = bool(cfg.get("include_sustain", False))
        include_hh_cc4 = bool(cfg.get("include_hh_cc4", False))
        try:
            if not ("include_sustain" in cfg or "include_hh_cc4" in cfg):
                with np.load(Path(paths[0]), allow_pickle=False) as d0:
                    D = int(np.asarray(d0["drum_hit"]).shape[0])
                sus, hh = eg._infer_feature_flags(in_dim=int(cfg.get("in_dim", 0) or 0), d_drum=int(D))  # type: ignore[attr-defined]
                include_sustain = bool(sus)
                include_hh_cc4 = bool(hh)
        except Exception:
            pass
        ds = eg.ExpressiveGridDataset(  # type: ignore[attr-defined]
            paths,
            vocab=vocab,
            include_sustain=include_sustain,
            include_hh_cc4=include_hh_cc4,
        )
        kit_id_to_name = _invert_int_mapping(dict(vocab.get("kit_name_to_id", {}) or {}))
        pad_id = int(cfg.get("pad_id", 2048))
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda items: eg.ExpressiveGridDataset.collate_fn(items, pad_id=pad_id),  # type: ignore[attr-defined]
            drop_last=False,
        )

        token_nll: List[float] = []
        token_acc: List[float] = []
        per_cb_acc: List[List[float]] = []
        infer_ms: List[float] = []
        audio_rmse: List[float] = []
        audio_mae: List[float] = []
        audio_mr_stft_sc: List[float] = []
        audio_env_rms_corr: List[float] = []
        audio_tter_db_mae: List[float] = []
        audio_onset_precision: List[float] = []
        audio_onset_recall: List[float] = []
        audio_onset_f1: List[float] = []

        oracle_rmse: List[float] = []
        oracle_mae: List[float] = []
        oracle_mr_stft_sc: List[float] = []
        oracle_env_rms_corr: List[float] = []
        oracle_tter_db_mae: List[float] = []
        oracle_onset_precision: List[float] = []
        oracle_onset_recall: List[float] = []
        oracle_onset_f1: List[float] = []

        random_rmse: List[float] = []
        random_mae: List[float] = []
        random_mr_stft_sc: List[float] = []
        random_env_rms_corr: List[float] = []
        random_tter_db_mae: List[float] = []
        random_onset_precision: List[float] = []
        random_onset_recall: List[float] = []
        random_onset_f1: List[float] = []

        per_kit: Dict[str, Dict[str, Any]] = {}

        def _kit_bucket(kit: str) -> Dict[str, Any]:
            b = per_kit.get(kit)
            if b is None:
                b = {
                    "token_nll": [],
                    "token_acc": [],
                    "per_cb_acc": [],
                    "infer_ms": [],
                    "rmse": [],
                    "mae": [],
                    "mr_stft_sc": [],
                    "env_rms_corr": [],
                    "tter_db_mae": [],
                    "onset_precision": [],
                    "onset_recall": [],
                    "onset_f1": [],
                }
                per_kit[str(kit)] = b
            return b

        saved = 0
        pred_saved = 0
        fadtk_pred_by_item: Dict[str, np.ndarray] = {}
        pred_manifest_fp = None
        pbar = _wrap_tqdm(
            loader,
            desc=f"Eval {spec.name}",
            total=(len(loader) if hasattr(loader, "__len__") else None),
            disable=not use_tqdm,
            leave=False,
            unit="it",
        )
        for batch_i, batch in enumerate(pbar):
            t0 = time.perf_counter()
            out = model(
                grid=batch["grid"].to(device),
                beat_pos=batch["beat_pos"].to(device),
                bpm=batch["bpm"].to(device),
                drummer_id=batch["drummer_id"].to(device),
                kit_name_id=batch["kit_name_id"].to(device) if "kit_name_id" in batch else None,
                valid_mask=batch["valid_mask"].to(device),
                return_btcv=True,
            )
            if not (isinstance(out, tuple) and len(out) == 2):
                raise RuntimeError("model did not return (logits_bctv, logits_btcv) as expected")
            logits, logits_btcv = out
            torch.cuda.synchronize() if device.startswith("cuda") else None  # type: ignore[attr-defined]
            infer_ms.append(1000.0 * (time.perf_counter() - t0))

            tgt = batch["tgt_codes"].to(device)
            tgt_btc = tgt.transpose(1, 2).contiguous()  # [B,T,C]
            loss = torch.nn.functional.cross_entropy(
                logits_btcv.reshape(-1, int(logits_btcv.shape[-1])),
                tgt_btc.view(-1),
                ignore_index=int(pad_id),
                reduction="mean",
            )
            token_nll.append(float(loss.detach().cpu().item()))

            pred = torch.argmax(logits, dim=-1)  # [B,C,T]
            mask = tgt.ne(int(pad_id))
            correct = (pred == tgt) & mask
            denom = float(mask.sum().detach().cpu().item())
            token_acc.append(float(correct.sum().detach().cpu().item()) / max(1.0, denom))
            if hasattr(pbar, "set_postfix"):
                try:
                    pbar.set_postfix({"nll": f"{token_nll[-1]:.3f}", "acc": f"{token_acc[-1]*100.0:.1f}%"}, refresh=False)
                except Exception:
                    pass

            # Per-codebook accuracy.
            C = int(tgt.shape[1])
            cb_accs: List[float] = []
            for c in range(C):
                mc = mask[:, c, :]
                dc = float(mc.sum().detach().cpu().item())
                cc = float(((pred[:, c, :] == tgt[:, c, :]) & mc).sum().detach().cpu().item())
                cb_accs.append(cc / max(1.0, dc))
            per_cb_acc.append(cb_accs)

            key = _npz_item_key(Path(ds.paths[batch_i]))
            key_short = _stable_key_str(key)
            kit_id = int(batch.get("kit_name_id", torch.zeros((1,), dtype=torch.long))[0].detach().cpu().item())
            kit_name = kit_id_to_name.get(int(kit_id), "")
            kit_label = str(kit_name).strip() if str(kit_name).strip() else (f"kit_{kit_id}" if int(kit_id) != 0 else "unknown")
            if fadtk_enabled and fadtk_ids is not None and key_short in fadtk_ids:
                fadtk_kit_by_item[key_short] = str(kit_label)
            kb = _kit_bucket(kit_label)
            kb["token_nll"].append(float(token_nll[-1]))
            kb["token_acc"].append(float(token_acc[-1]))
            kb["infer_ms"].append(float(infer_ms[-1]))
            kb["per_cb_acc"].append(list(cb_accs))
            row: Dict[str, Any] = {
                "system": spec.name,
                "item": key_short,
                "kit": kit_label,
                "kit_name_id": int(kit_id),
                "token_nll": float(token_nll[-1]),
                "token_acc": float(token_acc[-1]),
                "infer_ms": float(infer_ms[-1]),
            }
            for ci, v in enumerate(cb_accs):
                row[f"token_acc_cb{ci}"] = float(v)

            if do_audio:
                audio_path, _midi, sr_native, start_sample, window_samples = key
                ref, sr_ref = _load_audio_segment(Path(audio_path), start_sample=int(start_sample), num_samples=int(window_samples))
                if sr_ref <= 0:
                    row.update(
                        {
                            "rmse": float("nan"),
                            "mae": float("nan"),
                            "mr_stft_sc": float("nan"),
                            "env_rms_corr": float("nan"),
                            "tter_db_mae": float("nan"),
                            "onset_precision": float("nan"),
                            "onset_recall": float("nan"),
                            "onset_f1": float("nan"),
                        }
                    )
                else:
                    # Decode predicted tokens.
                    audio_pred_b1, sr_pred = decode_tokens_to_audio(
                        pred.detach().to("cpu"),
                        encoder_model=str(encoder_model),
                        device=str(decode_device),
                    )
                    y_pred = np.asarray(audio_pred_b1[0], dtype=np.float32)
                    y_ref = ref
                    y_ref = _resample_linear(y_ref, int(sr_ref), int(eval_sr))
                    y_pred = _resample_linear(y_pred, int(sr_pred), int(eval_sr))
                    n = int(min(y_ref.size, y_pred.size))
                    y_ref = y_ref[:n]
                    y_pred = y_pred[:n]

                    # fadtk package FAD: compute clip-level embeddings (mean over timeframes).
                    if fadtk_enabled and fadtk_ids is not None and key_short in fadtk_ids:
                        if fadtk_ml is None or fadtk_fad_mod is None:
                            raise RuntimeError("Internal error: fadtk not initialized.")
                        try:
                            # Track clip duration (for reporting).
                            if key_short not in fadtk_clip_dur_by_item_s:
                                fadtk_clip_dur_by_item_s[key_short] = float(n) / float(eval_sr)

                            # Reference embeddings:
                            # - Always required when per-kit FAD is enabled.
                            # - Otherwise only required if ref stats were not loaded from cache.
                            if (fadtk_ref_mu is None or fadtk_per_kit) and key_short not in fadtk_ref_by_item:
                                y_ref_f = _resample_poly(y_ref, int(eval_sr), int(getattr(fadtk_ml, "sr", eval_sr)))
                                emb_ref = fadtk_ml.get_embedding(y_ref_f)  # [T',D]
                                vec_ref = np.asarray(np.mean(emb_ref, axis=0), dtype=np.float32)
                                fadtk_ref_by_item[key_short] = vec_ref

                            y_pred_f = _resample_poly(y_pred, int(eval_sr), int(getattr(fadtk_ml, "sr", eval_sr)))
                            emb_pred = fadtk_ml.get_embedding(y_pred_f)  # [T',D]
                            vec_pred = np.asarray(np.mean(emb_pred, axis=0), dtype=np.float32)
                            fadtk_pred_by_item[key_short] = vec_pred
                        except Exception as e:
                            summary.setdefault("fadtk_embed_errors", {})[str(spec.name)] = str(e)

                    rmsev = _rmse(y_pred, y_ref)
                    maev = _mae(y_pred, y_ref)
                    mr_sc = _mr_stft_sc(y_pred, y_ref)
                    env_corr = _envelope_rms_corr(y_pred, y_ref, sr=int(eval_sr))
                    tter_pred = _windowed_tter_db(y_pred, sr=int(eval_sr))
                    tter_ref = _windowed_tter_db(y_ref, sr=int(eval_sr))
                    tter_mae = float(abs(float(tter_pred) - float(tter_ref))) if (math.isfinite(float(tter_pred)) and math.isfinite(float(tter_ref))) else float("nan")

                    # Onset evaluation:
                    #   GT = onsets from the cached symbolic conditioning (drum grid),
                    #        which reflects the intended performance timing.
                    #   Pred = onsets detected from the predicted audio waveform.
                    #
                    # No snapping/alignment is applied between GT and audio; matching
                    # uses a ±tol window (default 50ms) in _onset_pr_metrics.
                    onset_kw = {
                        "min_separation_s": 0.05,
                        "rms_gate_db": 35.0,
                        "backtrack_ms": 0.0,
                        "refine_ms": 12.0,
                    }
                    # GT = symbolic events from cached grid, with filtering:
                    #   - drop closed hi-hats (often dense/timekeeping and not reliably audible as onsets)
                    #   - drop very low-velocity hits ("minor velocities")
                    gt_onsets = _onsets_from_grid_npz(
                        Path(ds.paths[batch_i]),
                        eval_sr=int(eval_sr),
                        vel_thresh=0.30,
                        exclude_channels=[],
                    )
                    gt_onsets = gt_onsets[(gt_onsets >= 0) & (gt_onsets < int(n))]
                    pred_onsets = _onsets_from_audio(y_pred, sr=int(eval_sr), **onset_kw)
                    pred_onsets = pred_onsets[(pred_onsets >= 0) & (pred_onsets < int(n))]
                    om = _onset_pr_metrics(
                        y_pred,
                        y_ref,
                        sr=int(eval_sr),
                        pred_onsets=pred_onsets,
                        ref_onsets=gt_onsets,
                        midi_path=None,
                        start_sec=None,
                        end_sec=None,
                    )
                    audio_rmse.append(float(rmsev))
                    audio_mae.append(float(maev))
                    audio_mr_stft_sc.append(float(mr_sc))
                    audio_env_rms_corr.append(float(env_corr))
                    audio_tter_db_mae.append(float(tter_mae))
                    audio_onset_precision.append(float(om["onset_precision"]))
                    audio_onset_recall.append(float(om["onset_recall"]))
                    audio_onset_f1.append(float(om["onset_f1"]))
                    kb["rmse"].append(float(rmsev))
                    kb["mae"].append(float(maev))
                    kb["mr_stft_sc"].append(float(mr_sc))
                    kb["env_rms_corr"].append(float(env_corr))
                    kb["tter_db_mae"].append(float(tter_mae))
                    kb["onset_precision"].append(float(om["onset_precision"]))
                    kb["onset_recall"].append(float(om["onset_recall"]))
                    kb["onset_f1"].append(float(om["onset_f1"]))
                    row.update(
                        {
                            "rmse": float(rmsev),
                            "mae": float(maev),
                            "mr_stft_sc": float(mr_sc),
                            "env_rms_corr": float(env_corr),
                            "tter_db_mae": float(tter_mae),
                            "onset_precision": float(om["onset_precision"]),
                            "onset_recall": float(om["onset_recall"]),
                            "onset_f1": float(om["onset_f1"]),
                        }
                    )

                    if save_wavs > 0 and saved < save_wavs:
                        sys_dir = out_dir / "wavs" / spec.name
                        _write_wav(sys_dir / f"{key_short}_pred.wav", y_pred, int(eval_sr))
                        _write_wav(sys_dir / f"{key_short}_ref.wav", y_ref, int(eval_sr))
                        saved += 1

                    if save_preds != 0 and (save_preds < 0 or pred_saved < save_preds):
                        if pred_run_dir is None:
                            raise RuntimeError("Internal error: pred_run_dir is None but --save-preds is enabled.")
                        sys_dir = pred_run_dir / "pred" / spec.name
                        _write_wav(sys_dir / f"{key_short}.wav", y_pred, int(eval_sr))
                        if pred_ref_dir is not None:
                            ref_path = pred_ref_dir / f"{key_short}.wav"
                            if key_short not in saved_ref_keys:
                                _write_wav(ref_path, y_ref, int(eval_sr))
                                saved_ref_keys.add(key_short)
                            if pred_ref_by_system_dir is not None:
                                _link_or_copy(ref_path, pred_ref_by_system_dir / spec.name / f"{key_short}.wav")
                        if pred_manifest_fp is None:
                            meta_dir = pred_run_dir / "meta"
                            meta_dir.mkdir(parents=True, exist_ok=True)
                            pred_manifest_fp = (meta_dir / f"{spec.name}.pred_manifest.jsonl").open("a", encoding="utf-8")
                        pred_manifest_fp.write(
                            json.dumps(
                                {
                                    "item": key_short,
                                    "audio_path": str(audio_path),
                                    "sr_native": int(sr_native),
                                    "start_sample": int(start_sample),
                                    "window_samples": int(window_samples),
                                    "eval_sr": int(eval_sr),
                                    "encoder_model": str(encoder_model),
                                    "ref_wav": str((pred_ref_dir / f"{key_short}.wav") if pred_ref_dir is not None else ""),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        pred_saved += 1

                    if add_oracle:
                        audio_oracle_b1, sr_or = decode_tokens_to_audio(
                            tgt.detach().to("cpu"),
                            encoder_model=str(encoder_model),
                            device=str(decode_device),
                        )
                        y_or = _resample_linear(np.asarray(audio_oracle_b1[0], dtype=np.float32), int(sr_or), int(eval_sr))
                        n = int(min(y_ref.size, y_or.size))
                        y_or = y_or[:n]
                        y_r2 = y_ref[:n]
                        rmseo = _rmse(y_or, y_r2)
                        maeo = _mae(y_or, y_r2)
                        mrso = _mr_stft_sc(y_or, y_r2)
                        env_corr_o = _envelope_rms_corr(y_or, y_r2, sr=int(eval_sr))
                        tter_or = _windowed_tter_db(y_or, sr=int(eval_sr))
                        tter_r2 = _windowed_tter_db(y_r2, sr=int(eval_sr))
                        tter_mae_o = float(abs(float(tter_or) - float(tter_r2))) if (math.isfinite(float(tter_or)) and math.isfinite(float(tter_r2))) else float("nan")
                        onset_kw_o = {
                            "min_separation_s": 0.05,
                            "rms_gate_db": 35.0,
                            "backtrack_ms": 0.0,
                            "refine_ms": 12.0,
                        }
                        gt_onsets_o = _onsets_from_grid_npz(
                            Path(ds.paths[batch_i]),
                            eval_sr=int(eval_sr),
                            vel_thresh=0.30,
                            exclude_channels=[],
                        )
                        gt_onsets_o = gt_onsets_o[(gt_onsets_o >= 0) & (gt_onsets_o < int(n))]
                        pred_onsets_o = _onsets_from_audio(y_or, sr=int(eval_sr), **onset_kw_o)
                        pred_onsets_o = pred_onsets_o[(pred_onsets_o >= 0) & (pred_onsets_o < int(n))]
                        omo = _onset_pr_metrics(
                            y_or,
                            y_r2,
                            sr=int(eval_sr),
                            pred_onsets=pred_onsets_o,
                            ref_onsets=gt_onsets_o,
                            midi_path=None,
                            start_sec=None,
                            end_sec=None,
                        )
                        oracle_rmse.append(float(rmseo))
                        oracle_mae.append(float(maeo))
                        oracle_mr_stft_sc.append(float(mrso))
                        oracle_env_rms_corr.append(float(env_corr_o))
                        oracle_tter_db_mae.append(float(tter_mae_o))
                        oracle_onset_precision.append(float(omo["onset_precision"]))
                        oracle_onset_recall.append(float(omo["onset_recall"]))
                        oracle_onset_f1.append(float(omo["onset_f1"]))
                        if save_preds != 0 and pred_run_dir is not None:
                            # Save oracle decodes for listening/external metrics.
                            o_dir = pred_run_dir / "oracle" / spec.name
                            _write_wav(o_dir / f"{key_short}.wav", y_or, int(eval_sr))
                            if pred_ref_dir is not None:
                                ref_path = pred_ref_dir / f"{key_short}.wav"
                                if key_short not in saved_ref_keys:
                                    _write_wav(ref_path, y_ref, int(eval_sr))
                                    saved_ref_keys.add(key_short)
                                if pred_ref_by_system_dir is not None:
                                    _link_or_copy(ref_path, pred_ref_by_system_dir / spec.name / f"{key_short}.wav")

                    if add_random:
                        # Uniform random tokens in [0, pad_id-1] (avoid PAD by construction).
                        gen = torch.Generator(device="cpu").manual_seed(int(args.seed) + int(batch_i))
                        rand = torch.randint(
                            low=0,
                            high=int(pad_id),
                            size=tuple(tgt.detach().to("cpu").shape),
                            generator=gen,
                            dtype=torch.long,
                        )
                        audio_rand_b1, sr_rand = decode_tokens_to_audio(
                            rand,
                            encoder_model=str(encoder_model),
                            device=str(decode_device),
                        )
                        y_rand = _resample_linear(np.asarray(audio_rand_b1[0], dtype=np.float32), int(sr_rand), int(eval_sr))
                        n = int(min(y_ref.size, y_rand.size))
                        y_rand = y_rand[:n]
                        y_r3 = y_ref[:n]
                        rmser = _rmse(y_rand, y_r3)
                        maer = _mae(y_rand, y_r3)
                        mrsr = _mr_stft_sc(y_rand, y_r3)
                        env_corr_r = _envelope_rms_corr(y_rand, y_r3, sr=int(eval_sr))
                        tter_rand = _windowed_tter_db(y_rand, sr=int(eval_sr))
                        tter_r3 = _windowed_tter_db(y_r3, sr=int(eval_sr))
                        tter_mae_r = float(abs(float(tter_rand) - float(tter_r3))) if (math.isfinite(float(tter_rand)) and math.isfinite(float(tter_r3))) else float("nan")
                        onset_kw_r = {
                            "min_separation_s": 0.05,
                            "rms_gate_db": 35.0,
                            "backtrack_ms": 0.0,
                            "refine_ms": 12.0,
                        }
                        gt_onsets_r = _onsets_from_grid_npz(
                            Path(ds.paths[batch_i]),
                            eval_sr=int(eval_sr),
                            vel_thresh=0.30,
                            exclude_channels=[],
                        )
                        gt_onsets_r = gt_onsets_r[(gt_onsets_r >= 0) & (gt_onsets_r < int(n))]
                        pred_onsets_r = _onsets_from_audio(y_rand, sr=int(eval_sr), **onset_kw_r)
                        pred_onsets_r = pred_onsets_r[(pred_onsets_r >= 0) & (pred_onsets_r < int(n))]
                        omr = _onset_pr_metrics(
                            y_rand,
                            y_r3,
                            sr=int(eval_sr),
                            pred_onsets=pred_onsets_r,
                            ref_onsets=gt_onsets_r,
                            midi_path=None,
                            start_sec=None,
                            end_sec=None,
                        )
                        random_rmse.append(float(rmser))
                        random_mae.append(float(maer))
                        random_mr_stft_sc.append(float(mrsr))
                        random_env_rms_corr.append(float(env_corr_r))
                        random_tter_db_mae.append(float(tter_mae_r))
                        random_onset_precision.append(float(omr["onset_precision"]))
                        random_onset_recall.append(float(omr["onset_recall"]))
                        random_onset_f1.append(float(omr["onset_f1"]))

            rows.append(row)

        # Aggregate.
        cb_means: Dict[str, float] = {}
        if per_cb_acc:
            Cmax = max(len(x) for x in per_cb_acc)
            for c in range(Cmax):
                vals = [x[c] for x in per_cb_acc if c < len(x)]
                cb_means[f"token_acc_cb{c}"] = float(np.mean(vals)) if vals else float("nan")

        sys_sum: Dict[str, Any] = {
            "ckpt": str(Path(spec.ckpt)),
            "cache": str(Path(spec.cache)),
            "encoder_model": str(encoder_model),
            "n_items": int(len(per_cb_acc)),
            "token_nll": _mean_std(token_nll),
            "token_ppl": _mean_std([float(math.exp(x)) for x in token_nll]),
            "token_acc": _mean_std(token_acc),
            "infer_ms": _mean_std(infer_ms),
            **cb_means,
        }
        if do_audio:
            sys_sum.update(
                {
                    "rmse": _mean_std(audio_rmse),
                    "mae": _mean_std(audio_mae),
                    "mr_stft_sc": _mean_std(audio_mr_stft_sc),
                    "env_rms_corr": _mean_std(audio_env_rms_corr),
                    "tter_db_mae": _mean_std(audio_tter_db_mae),
                    "onset_precision": _mean_std(audio_onset_precision),
                    "onset_recall": _mean_std(audio_onset_recall),
                    "onset_f1": _mean_std(audio_onset_f1),
                }
            )

        if per_kit and len(per_kit) > 1:
            per_kit_summary: Dict[str, Any] = {}
            for kit_label, b in sorted(per_kit.items(), key=lambda kv: kv[0]):
                cb_means_k: Dict[str, float] = {}
                pcb = b.get("per_cb_acc", [])
                if pcb:
                    Cmax_k = max(len(x) for x in pcb)
                    for c in range(Cmax_k):
                        vals = [x[c] for x in pcb if c < len(x)]
                        cb_means_k[f"token_acc_cb{c}"] = float(np.mean(vals)) if vals else float("nan")

                s_k: Dict[str, Any] = {
                    "n_items": int(len(pcb)),
                    "token_nll": _mean_std(list(b.get("token_nll", []))),
                    "token_ppl": _mean_std([float(math.exp(x)) for x in list(b.get("token_nll", [])) if math.isfinite(float(x))]),
                    "token_acc": _mean_std(list(b.get("token_acc", []))),
                    "infer_ms": _mean_std(list(b.get("infer_ms", []))),
                    **cb_means_k,
                }
                if do_audio:
                    s_k.update(
                        {
                            "rmse": _mean_std(list(b.get("rmse", []))),
                            "mae": _mean_std(list(b.get("mae", []))),
                            "mr_stft_sc": _mean_std(list(b.get("mr_stft_sc", []))),
                            "env_rms_corr": _mean_std(list(b.get("env_rms_corr", []))),
                            "tter_db_mae": _mean_std(list(b.get("tter_db_mae", []))),
                            "onset_precision": _mean_std(list(b.get("onset_precision", []))),
                            "onset_recall": _mean_std(list(b.get("onset_recall", []))),
                            "onset_f1": _mean_std(list(b.get("onset_f1", []))),
                        }
                    )
                per_kit_summary[str(kit_label)] = s_k

            sys_sum["per_kit"] = per_kit_summary

        # Package-based FAD via fadtk: compute per system from embeddings collected during eval.
        if fadtk_enabled:
            try:
                if fadtk_fad_mod is None or fadtk_ml is None or fadtk_ref_stats_path is None:
                    raise RuntimeError("Internal error: fadtk not initialized.")

                model_name = str(args.fadtk_model)
                tag = f"fad_fadtk_{model_name}"

                # Finalize reference stats if not loaded from cache.
                if fadtk_ref_mu is None or fadtk_ref_cov is None:
                    ref_keys = sorted(fadtk_ref_by_item.keys())
                    if len(ref_keys) < 2:
                        raise RuntimeError(f"fadtk: not enough reference embeddings (n_ref={len(ref_keys)})")
                    ref_mat = np.stack([fadtk_ref_by_item[k] for k in ref_keys], axis=0).astype(np.float64, copy=False)
                    fadtk_ref_mu = np.mean(ref_mat, axis=0)
                    fadtk_ref_cov = np.cov(ref_mat, rowvar=False)

                    durs = list(fadtk_clip_dur_by_item_s.values())
                    dur_stats = {
                        "n": int(len(durs)),
                        "mean": float(np.mean(durs)) if durs else float("nan"),
                        "median": float(np.median(durs)) if durs else float("nan"),
                        "min": float(np.min(durs)) if durs else float("nan"),
                        "max": float(np.max(durs)) if durs else float("nan"),
                    }

                    fadtk_ref_meta = {
                        "selection_sha1": str(summary.get("fadtk", {}).get("selection_sha1", "")),
                        "model": str(model_name),
                        "dim": int(getattr(fadtk_ml, "num_features", 0)),
                        "model_sr": int(getattr(fadtk_ml, "sr", 0)),
                        "eval_sr": int(eval_sr),
                        "split": str(split),
                        "intersection": bool(args.intersection),
                        "n_ref": int(ref_mat.shape[0]),
                        "embed_aggregation": "mean_over_timeframes",
                        "clip_dur_s": dur_stats,
                    }
                    _fadtk_save_ref_stats(fadtk_ref_stats_path, mu=fadtk_ref_mu, cov=fadtk_ref_cov, meta=fadtk_ref_meta)
                    if isinstance(summary.get("fadtk"), dict):
                        summary["fadtk"]["ref_stats_loaded"] = False
                        summary["fadtk"]["ref_stats_n_ref"] = int(ref_mat.shape[0])

                pred_keys = sorted(fadtk_pred_by_item.keys())
                if len(pred_keys) < 2:
                    raise RuntimeError(f"fadtk: not enough predicted embeddings for {spec.name} (n_gen={len(pred_keys)})")
                pred_mat = np.stack([fadtk_pred_by_item[k] for k in pred_keys], axis=0).astype(np.float64, copy=False)
                mu_pred = np.mean(pred_mat, axis=0)
                cov_pred = np.cov(pred_mat, rowvar=False)
                fad_v = float(fadtk_fad_mod.calc_frechet_distance(fadtk_ref_mu, fadtk_ref_cov, mu_pred, cov_pred))

                out: Dict[str, Any] = {
                    "backend": "fadtk",
                    "fad": float(fad_v),
                    "n_ref": int(fadtk_ref_meta.get("n_ref") or (len(fadtk_ref_by_item) if fadtk_ref_by_item else pred_mat.shape[0])),
                    "n_gen": int(pred_mat.shape[0]),
                    "dim": int(pred_mat.shape[1]),
                    "model": str(model_name),
                    "embed_aggregation": "mean_over_timeframes",
                }
                if not bool(args.no_fadtk_inf):
                    out["fad_inf"] = _fadtk_fad_inf(
                        fadtk_fad_mod,
                        mu_base=np.asarray(fadtk_ref_mu, dtype=np.float64),
                        cov_base=np.asarray(fadtk_ref_cov, dtype=np.float64),
                        embeds=pred_mat,
                        steps=int(args.fadtk_inf_steps),
                        min_n=int(args.fadtk_inf_min_n),
                        max_n=int(args.fadtk_inf_max_n),
                        seed=int(args.seed) + 1009 + abs(hash(str(spec.name))) % 100000,
                    )
                sys_sum[tag] = out
            except Exception as e:
                summary.setdefault("fadtk_partial_errors", {})[str(spec.name)] = str(e)

        # Optional: per-kit FAD/FAD∞ (written to a CSV at the end).
        if fadtk_enabled and fadtk_per_kit:
            try:
                if fadtk_fad_mod is None or fadtk_ml is None:
                    raise RuntimeError("Internal error: fadtk not initialized.")

                model_name = str(args.fadtk_model)

                # Build per-kit reference stats once (shared across systems).
                if not fadtk_ref_stats_by_kit:
                    ref_by_kit: Dict[str, List[np.ndarray]] = {}
                    for k, v in fadtk_ref_by_item.items():
                        kit = str(fadtk_kit_by_item.get(k, "unknown"))
                        ref_by_kit.setdefault(kit, []).append(v)
                    for kit, vecs in ref_by_kit.items():
                        if len(vecs) < 2:
                            continue
                        mat = np.stack(vecs, axis=0).astype(np.float64, copy=False)
                        mu = np.mean(mat, axis=0)
                        cov = np.cov(mat, rowvar=False)
                        fadtk_ref_stats_by_kit[str(kit)] = (mu, cov, int(mat.shape[0]))

                # Group predicted vectors by kit for this system.
                pred_by_kit: Dict[str, List[np.ndarray]] = {}
                for k, v in fadtk_pred_by_item.items():
                    kit = str(fadtk_kit_by_item.get(k, "unknown"))
                    pred_by_kit.setdefault(kit, []).append(v)

                for kit, (mu_ref, cov_ref, n_ref) in sorted(fadtk_ref_stats_by_kit.items(), key=lambda kv: kv[0]):
                    vecs = pred_by_kit.get(str(kit), [])
                    row_pk: Dict[str, Any] = {
                        "system": str(spec.name),
                        "kit": str(kit),
                        "model": str(model_name),
                        "dim": int(getattr(fadtk_ml, "num_features", 0) or int(getattr(mu_ref, "size", 0) or 0)),
                        "n_ref": int(n_ref),
                        "n_gen": int(len(vecs)),
                        "fad": float("nan"),
                        "fad_inf": float("nan"),
                        "fad_inf_ok": False,
                        "selection_sha1": str(summary.get("fadtk", {}).get("selection_sha1", "")),
                    }
                    if len(vecs) < 2:
                        row_pk["error"] = "not enough pred embeddings"
                        fadtk_per_kit_rows.append(row_pk)
                        continue

                    pred_mat = np.stack(vecs, axis=0).astype(np.float64, copy=False)
                    mu_pred = np.mean(pred_mat, axis=0)
                    cov_pred = np.cov(pred_mat, rowvar=False)
                    row_pk["fad"] = float(fadtk_fad_mod.calc_frechet_distance(mu_ref, cov_ref, mu_pred, cov_pred))

                    if not bool(args.no_fadtk_inf):
                        fi = _fadtk_fad_inf(
                            fadtk_fad_mod,
                            mu_base=np.asarray(mu_ref, dtype=np.float64),
                            cov_base=np.asarray(cov_ref, dtype=np.float64),
                            embeds=np.asarray(pred_mat, dtype=np.float64),
                            steps=int(args.fadtk_inf_steps),
                            min_n=int(args.fadtk_inf_min_n),
                            max_n=int(args.fadtk_inf_max_n),
                            seed=int(args.seed) + 2009 + abs(hash(str(spec.name) + str(kit))) % 100000,
                        )
                        row_pk["fad_inf"] = float(fi.get("fad_inf", float("nan")))
                        row_pk["fad_inf_ok"] = bool(fi.get("ok", False))
                        if not bool(fi.get("ok", False)):
                            row_pk["error"] = str(fi.get("error", "fad_inf failed"))

                    fadtk_per_kit_rows.append(row_pk)
            except Exception as e:
                summary.setdefault("fadtk_partial_errors", {})[str(spec.name) + ":per_kit"] = str(e)
        summary["systems"][spec.name] = sys_sum

        if pred_manifest_fp is not None:
            pred_manifest_fp.close()

        if do_audio and add_oracle:
            summary["systems"][f"{spec.name}_oracle"] = {
                "ckpt": str(Path(spec.ckpt)),
                "cache": str(Path(spec.cache)),
                "encoder_model": str(encoder_model),
                "rmse": _mean_std(oracle_rmse),
                "mae": _mean_std(oracle_mae),
                "mr_stft_sc": _mean_std(oracle_mr_stft_sc),
                "env_rms_corr": _mean_std(oracle_env_rms_corr),
                "tter_db_mae": _mean_std(oracle_tter_db_mae),
                "onset_precision": _mean_std(oracle_onset_precision),
                "onset_recall": _mean_std(oracle_onset_recall),
                "onset_f1": _mean_std(oracle_onset_f1),
                "n_items": int(len(oracle_rmse)),
            }

        if do_audio and add_random:
            summary["systems"][f"{spec.name}_random"] = {
                "ckpt": str(Path(spec.ckpt)),
                "cache": str(Path(spec.cache)),
                "encoder_model": str(encoder_model),
                "rmse": _mean_std(random_rmse),
                "mae": _mean_std(random_mae),
                "mr_stft_sc": _mean_std(random_mr_stft_sc),
                "env_rms_corr": _mean_std(random_env_rms_corr),
                "tter_db_mae": _mean_std(random_tter_db_mae),
                "onset_precision": _mean_std(random_onset_precision),
                "onset_recall": _mean_std(random_onset_recall),
                "onset_f1": _mean_std(random_onset_f1),
                "n_items": int(len(random_rmse)),
            }

    # Finalize fadtk reporting.
    if fadtk_enabled and isinstance(summary.get("fadtk"), dict):
        durs = list(fadtk_clip_dur_by_item_s.values())
        summary["fadtk"]["clip_dur_s"] = {
            "n": int(len(durs)),
            "mean": float(np.mean(durs)) if durs else float("nan"),
            "median": float(np.median(durs)) if durs else float("nan"),
            "min": float(np.min(durs)) if durs else float("nan"),
            "max": float(np.max(durs)) if durs else float("nan"),
        }
        summary["fadtk"]["ref_stats_path"] = str(fadtk_ref_stats_path) if fadtk_ref_stats_path is not None else ""
        summary["fadtk"]["ref_stats_loaded"] = bool(fadtk_ref_meta) or bool(fadtk_ref_mu is not None and fadtk_ref_cov is not None)
        if fadtk_ref_mu is not None and fadtk_ref_cov is not None:
            summary["fadtk"]["ref_stats_dim"] = int(fadtk_ref_mu.size)
        summary["fadtk"]["ok"] = not bool(summary.get("fadtk_partial_errors")) and not bool(summary.get("fadtk_embed_errors"))

    # If the user only requested a small number of saved preds (e.g. --save-preds 128),
    # default to cleaning them up after metrics to avoid accumulating lots of audio on disk.
    if pred_run_dir is not None and save_preds > 0 and not keep_preds:
        try:
            shutil.rmtree(pred_run_dir / "pred")
        except Exception:
            pass
        try:
            shutil.rmtree(pred_run_dir / "oracle")
        except Exception:
            pass
        try:
            shutil.rmtree(pred_run_dir / "ref")
        except Exception:
            pass
        try:
            shutil.rmtree(pred_run_dir / "ref_by_system")
        except Exception:
            pass
        # Keep meta/ (manifests) by default; it's small and useful for debugging.
        summary["pred_wavs_cleaned"] = True

    # Write outputs.
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if rows:
        cols = sorted({k for r in rows for k in r.keys()})
        with (out_dir / "items.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Per-kit fadtk CSV (large; keep out of summary.json to avoid bloating it).
    if fadtk_enabled and fadtk_per_kit and fadtk_per_kit_rows:
        if getattr(args, "fadtk_per_kit_out", None) is not None:
            out_csv = Path(args.fadtk_per_kit_out)
            if not out_csv.is_absolute():
                out_csv = out_dir / out_csv
        else:
            out_csv = out_dir / "fadtk_per_kit.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        cols = sorted({k for r in fadtk_per_kit_rows for k in r.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in fadtk_per_kit_rows:
                w.writerow(r)
        # Add pointer + count (small) to the existing summary file.
        try:
            summary.setdefault("fadtk", {})
            if isinstance(summary.get("fadtk"), dict):
                summary["fadtk"]["per_kit_csv"] = str(out_csv)
                summary["fadtk"]["per_kit_rows"] = int(len(fadtk_per_kit_rows))
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass


if __name__ == "__main__":
    main()
