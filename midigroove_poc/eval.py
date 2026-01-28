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
) -> np.ndarray:
    """Very lightweight onset detector for percussive audio.

    Returns onset sample indices (sorted, unique-ish) at the given sr.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr = int(sr)
    n_fft = int(n_fft)
    hop = int(hop)
    if y.size < max(4, n_fft) or sr <= 0 or n_fft <= 0 or hop <= 0:
        return np.zeros((0,), dtype=np.int64)

    # Simple high-pass to emphasize transients.
    y_hp = np.empty_like(y)
    y_hp[0] = y[0]
    y_hp[1:] = y[1:] - y[:-1]

    win = np.hanning(n_fft).astype(np.float32)
    mags = _stft_mag(y_hp, n_fft=n_fft, hop=hop, win=win)  # [F, K]
    if mags.shape[0] < 3:
        return np.zeros((0,), dtype=np.int64)

    # Spectral flux (half-wave rectified frame-to-frame magnitude increase).
    dm = np.maximum(0.0, mags[1:] - mags[:-1])
    flux = dm.sum(axis=1).astype(np.float32)  # length ~= n_frames-1

    # Robust normalization.
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med)))
    scale = float(mad * 1.4826)  # approx std if normal
    z = (flux - med) / max(1e-6, scale)

    # Peak picking.
    peaks: List[int] = []
    for i in range(1, int(z.size) - 1):
        if not (z[i] > z[i - 1] and z[i] >= z[i + 1]):
            continue
        if float(z[i]) < float(z_thresh):
            continue
        peaks.append(int(i))
    if not peaks:
        return np.zeros((0,), dtype=np.int64)

    min_dist_frames = int(round(float(min_separation_s) * float(sr) / float(hop)))
    min_dist_frames = max(1, min_dist_frames)
    kept: List[int] = []
    last = -10**9
    for p in peaks:
        if p - last >= min_dist_frames:
            kept.append(int(p))
            last = int(p)

    # Convert peak indices (on flux) to frame indices in mags:
    # flux[t] corresponds to mags[t+1] - mags[t], so align to (t+1).
    onset_frames = (np.asarray(kept, dtype=np.int64) + 1).astype(np.int64)
    onset_samples = onset_frames * int(hop)
    onset_samples = onset_samples[(onset_samples >= 0) & (onset_samples < int(y.size))]
    return onset_samples.astype(np.int64, copy=False)


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
    midi_path: Optional[Path],
    start_sec: Optional[float],
    end_sec: Optional[float],
    tol_ms: float = 50.0,
) -> Dict[str, float]:
    """Onset precision/recall/F1 with greedy monotonic matching.

    Reference onsets are taken from MIDI (drum note starts) when available; if
    MIDI is missing/unreadable, we fall back to lightweight audio onsets from
    the reference waveform.
    """
    sr = int(sr)
    tol = int(round(float(tol_ms) * 1e-3 * float(sr)))
    tol = max(1, tol)

    p = _onsets_from_audio(pred, sr=sr)
    r: np.ndarray
    if midi_path is not None and start_sec is not None and end_sec is not None:
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
                    rmsev = _rmse(y_pred, y_ref)
                    maev = _mae(y_pred, y_ref)
                    mr_sc = _mr_stft_sc(y_pred, y_ref)
                    env_corr = _envelope_rms_corr(y_pred, y_ref, sr=int(eval_sr))
                    tter_pred = _windowed_tter_db(y_pred, sr=int(eval_sr))
                    tter_ref = _windowed_tter_db(y_ref, sr=int(eval_sr))
                    tter_mae = float(abs(float(tter_pred) - float(tter_ref))) if (math.isfinite(float(tter_pred)) and math.isfinite(float(tter_ref))) else float("nan")

                    midi_path = Path(_midi) if isinstance(_midi, str) and str(_midi).strip() else None
                    start_sec = (float(start_sample) / float(sr_native)) if int(sr_native) > 0 else None
                    end_sec = (float(start_sec) + (float(n) / float(eval_sr))) if start_sec is not None else None
                    om = _onset_pr_metrics(
                        y_pred,
                        y_ref,
                        sr=int(eval_sr),
                        midi_path=midi_path,
                        start_sec=start_sec,
                        end_sec=end_sec,
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
                        end_sec_o = (float(start_sec) + (float(n) / float(eval_sr))) if start_sec is not None else None
                        omo = _onset_pr_metrics(
                            y_or,
                            y_r2,
                            sr=int(eval_sr),
                            midi_path=midi_path,
                            start_sec=start_sec,
                            end_sec=end_sec_o,
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
                        end_sec_r = (float(start_sec) + (float(n) / float(eval_sr))) if start_sec is not None else None
                        omr = _onset_pr_metrics(
                            y_rand,
                            y_r3,
                            sr=int(eval_sr),
                            midi_path=midi_path,
                            start_sec=start_sec,
                            end_sec=end_sec_r,
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


if __name__ == "__main__":
    main()
