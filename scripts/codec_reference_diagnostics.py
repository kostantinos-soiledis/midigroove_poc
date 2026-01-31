#!/usr/bin/env python3
"""Codec reference-space diagnostics from cached eval windows.

Computes token-space statistics on the *ground-truth* cached codec tokens:
  - Unigram entropy / perplexity per codebook
  - Codebook usage (unique tokens used)
  - Temporal predictability (change rate, run-length stats)
  - Simple Markov (bigram) baseline NLL/PPL per codebook

Optionally (slower), it can also measure "oracle reconstruction" quality by
decoding the cached tokens back to audio and comparing to the original audio
segment, to help separate codec reconstruction quality from token learnability.

Typical usage (match the exact eval subset via items.csv):
  python scripts/codec_reference_diagnostics.py \
    --eval-items artifacts/eval/small_one_kit/items.csv \
    --cache encodec=cache/encodec_acoustic \
    --cache dac=cache/dac_acoustic \
    --cache xcodec=cache/xcodec_acoustic \
    --split test \
    --out artifacts/eval/reference_token_diagnostics.small_one_kit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _discover_manifest(cache_dir: Path, *, split: str) -> Optional[Path]:
    cache_dir = Path(cache_dir)
    patterns = [
        f"manifest_*_{split}_*.jsonl",
        f"manifest_{split}_*.jsonl",
        f"*{split}*.jsonl",
    ]
    for pat in patterns:
        hits = sorted(cache_dir.glob(pat))
        hits = [p for p in hits if p.is_file() and p.suffix.lower() == ".jsonl"]
        if hits:
            return hits[0]
    return None


def _stable_item_key_from_npz(npz_path: Path) -> str:
    with np.load(npz_path, allow_pickle=False) as d:
        audio_path = str(d["audio_path"].item())
        midi_path = str(d["midi_path"].item()) if "midi_path" in d else ""
        sr = int(d["sr"].item())
        start_sec = float(d["start_sec"].item())
        window_seconds = float(d["window_seconds"].item())
    start_sample = int(round(start_sec * float(sr)))
    window_samples = int(round(window_seconds * float(sr)))
    s = f"{audio_path}|{midi_path}|{sr}|{start_sample}|{window_samples}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _resample_linear(y: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    sr = int(sr)
    target_sr = int(target_sr)
    if sr <= 0 or target_sr <= 0 or y.size == 0 or sr == target_sr:
        return y.astype(np.float32, copy=False)
    dur = float(y.size) / float(sr)
    tgt_len = int(round(dur * float(target_sr)))
    tgt_len = max(1, int(tgt_len))
    x_old = np.linspace(0.0, 1.0, num=int(y.size), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=int(tgt_len), endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32, copy=False)


def _load_audio_segment(path: Path, *, sr: int, start_sec: float, window_seconds: float) -> np.ndarray:
    import soundfile as sf  # dependency is already in requirements.txt

    path = Path(path)
    start_sample = int(round(float(start_sec) * float(sr)))
    num_samples = int(round(float(window_seconds) * float(sr)))
    if num_samples <= 0:
        return np.zeros((0,), dtype=np.float32)
    with sf.SoundFile(str(path), "r") as f:
        if int(f.samplerate) != int(sr):
            # The cache records the "native" sample rate used for slicing.
            # If this mismatches the file, trust the file's samplerate.
            sr = int(f.samplerate)
            start_sample = int(round(float(start_sec) * float(sr)))
            num_samples = int(round(float(window_seconds) * float(sr)))
        f.seek(max(0, int(start_sample)))
        y = f.read(frames=int(num_samples), dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1).astype(np.float32, copy=False)
    return y.reshape(-1)


def _entropy_bits_from_counts(counts: Dict[int, int]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return float("nan")
    h = 0.0
    for k, v in counts.items():
        if v <= 0:
            continue
        p = float(v) / total
        h -= p * math.log(p, 2.0)
    return float(h)


def _mean_run_length(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return float("nan")
    runs = 1
    total = 0
    cur = 1
    for i in range(1, int(x.size)):
        if int(x[i]) == int(x[i - 1]):
            cur += 1
        else:
            runs += 1
            total += cur
            cur = 1
    total += cur
    return float(total) / float(runs)


def _change_rate(x: np.ndarray) -> float:
    x = np.asarray(x).reshape(-1)
    if x.size <= 1:
        return float("nan")
    diffs = (x[1:] != x[:-1]).astype(np.float32)
    return float(diffs.mean())


@dataclass
class CodebookAgg:
    counts: Dict[int, int]
    prev_counts: Dict[int, int]
    trans_counts: Dict[Tuple[int, int], int]
    n_tokens: int
    n_transitions: int
    n_changes: int
    run_len_sum: float
    run_count: int
    max_token: int


def _empty_cb() -> CodebookAgg:
    return CodebookAgg(
        counts={},
        prev_counts={},
        trans_counts={},
        n_tokens=0,
        n_transitions=0,
        n_changes=0,
        run_len_sum=0.0,
        run_count=0,
        max_token=-1,
    )


def _accum_sequence(cb: CodebookAgg, x: np.ndarray) -> None:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return

    vals, cnts = np.unique(x.astype(np.int64, copy=False), return_counts=True)
    for v, c in zip(vals.tolist(), cnts.tolist()):
        cb.counts[int(v)] = cb.counts.get(int(v), 0) + int(c)
        cb.max_token = max(int(cb.max_token), int(v))
    cb.n_tokens += int(x.size)

    if x.size >= 2:
        prev = x[:-1].astype(np.int64, copy=False)
        nxt = x[1:].astype(np.int64, copy=False)
        cb.n_transitions += int(prev.size)
        cb.n_changes += int(np.sum(prev != nxt))

        prev_vals, prev_cnts = np.unique(prev, return_counts=True)
        for v, c in zip(prev_vals.tolist(), prev_cnts.tolist()):
            cb.prev_counts[int(v)] = cb.prev_counts.get(int(v), 0) + int(c)

        pair_keys = prev.astype(np.int64) * (int(cb.max_token) + 2) + nxt.astype(np.int64)
        pk_vals, pk_cnts = np.unique(pair_keys, return_counts=True)
        base = int(cb.max_token) + 2
        for pk, c in zip(pk_vals.tolist(), pk_cnts.tolist()):
            a = int(pk) // base
            b = int(pk) % base
            cb.trans_counts[(a, b)] = cb.trans_counts.get((a, b), 0) + int(c)

    # run-length summary
    runs = 1
    cur = 1
    for i in range(1, int(x.size)):
        if int(x[i]) == int(x[i - 1]):
            cur += 1
        else:
            cb.run_count += 1
            cb.run_len_sum += float(cur)
            runs += 1
            cur = 1
    cb.run_count += 1
    cb.run_len_sum += float(cur)


def _markov_nll_bits(cb: CodebookAgg, *, alpha: float) -> float:
    alpha = float(alpha)
    if cb.n_transitions <= 0:
        return float("nan")
    V = int(cb.max_token) + 1
    if V <= 0:
        return float("nan")

    total_nll = 0.0
    for (a, b), n_ab in cb.trans_counts.items():
        n_a = cb.prev_counts.get(int(a), 0)
        p = (float(n_ab) + alpha) / (float(n_a) + alpha * float(V))
        total_nll += float(n_ab) * (-math.log(p, 2.0))
    return float(total_nll) / float(cb.n_transitions)


def _load_eval_item_ids(path: Path) -> List[str]:
    # Avoid pandas dependency for a quick one-off.
    import csv

    ids: List[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if "item" not in (r.fieldnames or []):
            raise RuntimeError(f"--eval-items missing 'item' column: {path}")
        for row in r:
            s = str(row["item"]).strip()
            if s:
                ids.append(s)
    return sorted(set(ids))


def _build_stable_id_to_npz(cache_dir: Path, *, split: str) -> Dict[str, Path]:
    cache_dir = Path(cache_dir)
    manifest = _discover_manifest(cache_dir, split=split)
    if manifest is None:
        # Fallback: scan items/*.npz (slower, but robust)
        npzs = sorted((cache_dir / "items").glob("*.npz"))
    else:
        entries = _read_jsonl(manifest)
        npzs = [Path(e["npz"]) for e in entries if isinstance(e, dict) and "npz" in e]
    out: Dict[str, Path] = {}
    for npz_path in npzs:
        npz_path = Path(npz_path)
        if not npz_path.is_file():
            continue
        sid = _stable_item_key_from_npz(npz_path)
        out[sid] = npz_path
    return out


def _decode_oracle_audio(npz_path: Path, *, encoder_model: str, device: Optional[str]) -> Tuple[np.ndarray, int, np.ndarray, int]:
    import torch

    from data.codecs import decode_tokens_to_audio

    with np.load(npz_path, allow_pickle=False) as d:
        tgt = np.asarray(d["tgt"], dtype=np.int64)
        audio_path = Path(str(d["audio_path"].item()))
        sr_ref = int(d["sr"].item())
        start_sec = float(d["start_sec"].item())
        window_seconds = float(d["window_seconds"].item())

    y_ref = _load_audio_segment(audio_path, sr=sr_ref, start_sec=start_sec, window_seconds=window_seconds)
    tokens = torch.from_numpy(tgt).to(dtype=torch.long)
    y_hat_b, sr_hat = decode_tokens_to_audio(tokens, encoder_model=str(encoder_model), device=device)
    y_hat = np.asarray(y_hat_b[0], dtype=np.float32).reshape(-1)
    return y_hat, int(sr_hat), y_ref, int(sr_ref)


def _build_probe_arrays(npz_paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X, Y) where X is [N, D] float32 and Y is [N, C] int64."""
    X_parts: List[np.ndarray] = []
    Y_parts: List[np.ndarray] = []

    for p in npz_paths:
        with np.load(p, allow_pickle=False) as d:
            tgt = np.asarray(d["tgt"], dtype=np.int64)  # [C, T]
            drum_hit = np.asarray(d["drum_hit"], dtype=np.float32)  # [D, T]
            drum_vel = np.asarray(d["drum_vel"], dtype=np.float32)  # [D, T]
            drum_sustain = np.asarray(d["drum_sustain"], dtype=np.float32)  # [D, T]
            hh_open = np.asarray(d["hh_open_cc4"], dtype=np.float32).reshape(1, -1)  # [1, T]
            beat_pos = np.asarray(d["beat_pos"], dtype=np.float32).reshape(1, -1)  # [1, T]
            bpm = float(d["bpm"].item()) if "bpm" in d else float("nan")

        if tgt.ndim != 2 or drum_hit.ndim != 2:
            continue
        T = int(min(tgt.shape[1], drum_hit.shape[1], drum_vel.shape[1], drum_sustain.shape[1], hh_open.shape[1], beat_pos.shape[1]))
        if T <= 0:
            continue

        beat_scale = float(max(1.0, float(np.max(beat_pos[:, :T]))))
        beat_pos_n = beat_pos[:, :T] / beat_scale
        log_bpm = float(math.log(max(1.0, bpm))) if math.isfinite(float(bpm)) else 0.0
        log_bpm_feat = np.full((1, T), log_bpm, dtype=np.float32)

        feats_dt = np.concatenate(
            [
                drum_hit[:, :T],
                drum_vel[:, :T],
                drum_sustain[:, :T],
                hh_open[:, :T],
                beat_pos_n[:, :T],
                log_bpm_feat[:, :T],
            ],
            axis=0,
        )  # [D, T]
        X_parts.append(feats_dt.T.astype(np.float32, copy=False))  # [T, D]
        Y_parts.append(tgt[:, :T].T.astype(np.int64, copy=False))  # [T, C]

    if not X_parts:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.int64)
    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    return X, Y


def _run_probe(
    *,
    npz_paths: Sequence[Path],
    steps: int,
    batch_size: int,
    hidden: int,
    lr: float,
    seed: int,
    device: Optional[str],
) -> dict:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        return {"error": f"probe requires torch (import failed: {e})"}

    rng = np.random.default_rng(int(seed))
    torch.manual_seed(int(seed))

    X_np, Y_np = _build_probe_arrays(npz_paths)
    if X_np.size == 0 or Y_np.size == 0:
        return {"error": "no probe data (empty X/Y)"}

    dev = torch.device(device if device else "cpu")
    X = torch.from_numpy(X_np).to(device=dev, dtype=torch.float32)
    Y = torch.from_numpy(Y_np).to(device=dev, dtype=torch.long)

    n, d = X.shape
    c = int(Y.shape[1])
    vocab_sizes: List[int] = [int(Y[:, i].max().item()) + 1 for i in range(c)]

    if hidden and int(hidden) > 0:
        trunk: nn.Module = nn.Sequential(nn.Linear(int(d), int(hidden)), nn.GELU())
        head_in = int(hidden)
    else:
        trunk = nn.Identity()
        head_in = int(d)
    heads = nn.ModuleList([nn.Linear(head_in, int(v)) for v in vocab_sizes])

    model = nn.ModuleDict({"trunk": trunk, "heads": heads}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    curve: List[dict] = []
    log_every = max(1, int(steps // 20))
    for step in range(int(steps)):
        idx = torch.from_numpy(rng.integers(0, n, size=(int(batch_size),), endpoint=False)).to(device=dev, dtype=torch.long)
        xb = X.index_select(0, idx)
        yb = Y.index_select(0, idx)

        z = model["trunk"](xb)
        losses = []
        for i, head in enumerate(model["heads"]):
            losses.append(F.cross_entropy(head(z), yb[:, i]))
        loss = torch.stack(losses).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step == 0 or (step + 1) % log_every == 0 or (step + 1) == int(steps):
            nll = float(loss.detach().cpu().item())
            curve.append({"step": int(step + 1), "nll_nats": float(nll), "ppl": float(math.exp(nll))})

    return {
        "n_frames": int(n),
        "feat_dim": int(d),
        "codebooks": int(c),
        "vocab_sizes": [int(v) for v in vocab_sizes],
        "steps": int(steps),
        "batch_size": int(batch_size),
        "hidden": int(hidden),
        "lr": float(lr),
        "seed": int(seed),
        "device": str(dev),
        "train_curve": curve,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        action="append",
        default=[],
        help="Cache spec as label=PATH (repeatable). Example: encodec=cache/encodec_acoustic",
    )
    ap.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    ap.add_argument(
        "--eval-items",
        type=str,
        default=None,
        help="Path to artifacts/eval/.../items.csv to match the exact evaluated subset (uses the 'item' stable IDs).",
    )
    ap.add_argument(
        "--intersection",
        action="store_true",
        help="If set and --eval-items is not provided, compute stats on the intersection of stable IDs across all caches.",
    )
    ap.add_argument("--markov-alpha", type=float, default=1.0, help="Additive smoothing for Markov baseline.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="Write JSON output to this path.")
    ap.add_argument(
        "--oracle-recon",
        type=int,
        default=0,
        help="If >0, decode N random cached items per cache and compute audio reconstruction metrics (slower).",
    )
    ap.add_argument("--device", type=str, default=None, help="Device for codec decode (e.g. cpu or cuda:0).")
    ap.add_argument(
        "--probe-steps",
        type=int,
        default=0,
        help="If >0, train a tiny per-frame probe on cached grid features to predict tokens (conditional-entropy proxy).",
    )
    ap.add_argument("--probe-items", type=int, default=256, help="How many cached windows to load for the probe.")
    ap.add_argument("--probe-batch", type=int, default=256, help="Probe batch size (frames).")
    ap.add_argument("--probe-hidden", type=int, default=0, help="Probe hidden width (0 => linear).")
    ap.add_argument("--probe-lr", type=float, default=3e-4, help="Probe learning rate.")

    args = ap.parse_args(list(argv) if argv is not None else None)

    caches: List[Tuple[str, Path]] = []
    for raw in args.cache:
        raw = str(raw)
        if "=" not in raw:
            raise SystemExit(f"Invalid --cache {raw!r}; expected label=PATH")
        label, p = raw.split("=", 1)
        caches.append((label.strip(), Path(p.strip())))
    if not caches:
        raise SystemExit("Provide at least one --cache label=PATH")

    random.seed(int(args.seed))

    eval_ids: Optional[List[str]] = None
    if args.eval_items:
        eval_ids = _load_eval_item_ids(Path(args.eval_items))

    per_cache_maps: Dict[str, Dict[str, Path]] = {}
    for label, cdir in caches:
        per_cache_maps[label] = _build_stable_id_to_npz(cdir, split=str(args.split))

    if eval_ids is not None:
        target_ids = set(eval_ids)
    else:
        if args.intersection and len(caches) > 1:
            keys = [set(m.keys()) for m in per_cache_maps.values()]
            target_ids = set.intersection(*keys) if keys else set()
        else:
            # default: per-cache stats over its full split
            target_ids = set()

    out: dict = {
        "split": str(args.split),
        "eval_items_path": str(args.eval_items) if args.eval_items else None,
        "intersection": bool(args.intersection) if eval_ids is None else None,
        "markov_alpha": float(args.markov_alpha),
        "probe_steps": int(args.probe_steps),
        "caches": {},
    }

    for label, cdir in caches:
        sid2npz = per_cache_maps[label]
        if eval_ids is None and not (args.intersection and len(caches) > 1):
            use_ids = sorted(sid2npz.keys())
        else:
            use_ids = sorted([sid for sid in target_ids if sid in sid2npz])

        cb_aggs: List[CodebookAgg] = []
        total_items = 0
        fps_vals: List[float] = []
        cb_vals: List[int] = []
        vocab_sizes_obs: List[int] = []

        for sid in use_ids:
            npz_path = sid2npz[sid]
            with np.load(npz_path, allow_pickle=False) as d:
                tgt = np.asarray(d["tgt"], dtype=np.int64)
                cb = int(d["cb"].item()) if "cb" in d else int(tgt.shape[0])
                fps = float(d["fps"].item()) if "fps" in d else float("nan")
            if tgt.ndim != 2:
                continue
            if tgt.shape[0] != cb:
                cb = int(tgt.shape[0])
            while len(cb_aggs) < cb:
                cb_aggs.append(_empty_cb())
            for c in range(cb):
                _accum_sequence(cb_aggs[c], tgt[c])
            total_items += 1
            fps_vals.append(float(fps))
            cb_vals.append(int(cb))
            vocab_sizes_obs.append(int(tgt.max()) + 1 if tgt.size else 0)

        cache_summary: dict = {
            "cache_dir": str(cdir),
            "num_items": int(total_items),
            "fps_mean": float(np.mean(fps_vals)) if fps_vals else float("nan"),
            "fps_std": float(np.std(fps_vals)) if fps_vals else float("nan"),
            "codebooks_mode": int(max(set(cb_vals), key=cb_vals.count)) if cb_vals else None,
            "vocab_obs_max": int(max(vocab_sizes_obs)) if vocab_sizes_obs else None,
            "codebooks": [],
        }

        total_token_count = sum(cb.n_tokens for cb in cb_aggs) or 0
        total_entropy_bits_weighted = 0.0
        total_markov_nll_bits_weighted = 0.0
        total_transitions = sum(cb.n_transitions for cb in cb_aggs) or 0
        total_changes = sum(cb.n_changes for cb in cb_aggs) or 0

        for c, cb in enumerate(cb_aggs):
            H = _entropy_bits_from_counts(cb.counts)
            ppl = float(2.0 ** H) if math.isfinite(H) else float("nan")
            V = int(cb.max_token) + 1
            unique = int(len(cb.counts))
            usage_frac = float(unique) / float(V) if V > 0 else float("nan")
            change_rate = float(cb.n_changes) / float(cb.n_transitions) if cb.n_transitions > 0 else float("nan")
            mean_run = float(cb.run_len_sum) / float(cb.run_count) if cb.run_count > 0 else float("nan")
            markov_nll_bits = _markov_nll_bits(cb, alpha=float(args.markov_alpha))
            markov_ppl = float(2.0 ** markov_nll_bits) if math.isfinite(markov_nll_bits) else float("nan")

            cache_summary["codebooks"].append(
                {
                    "codebook": int(c),
                    "tokens": int(cb.n_tokens),
                    "transitions": int(cb.n_transitions),
                    "vocab_max_token_plus1": int(V) if V > 0 else None,
                    "unique_tokens_used": int(unique),
                    "usage_frac": float(usage_frac),
                    "entropy_bits": float(H),
                    "ppl_unigram": float(ppl),
                    "change_rate": float(change_rate),
                    "mean_run_length": float(mean_run),
                    "markov_nll_bits": float(markov_nll_bits),
                    "markov_ppl": float(markov_ppl),
                }
            )

            if cb.n_tokens > 0 and math.isfinite(H):
                total_entropy_bits_weighted += float(H) * float(cb.n_tokens)
            if cb.n_transitions > 0 and math.isfinite(markov_nll_bits):
                total_markov_nll_bits_weighted += float(markov_nll_bits) * float(cb.n_transitions)

        cache_summary["overall"] = {
            "entropy_bits_token_weighted": float(total_entropy_bits_weighted) / float(total_token_count)
            if total_token_count > 0
            else float("nan"),
            "ppl_unigram_token_weighted": float(2.0 ** (float(total_entropy_bits_weighted) / float(total_token_count)))
            if total_token_count > 0
            else float("nan"),
            "markov_nll_bits_transition_weighted": float(total_markov_nll_bits_weighted) / float(total_transitions)
            if total_transitions > 0
            else float("nan"),
            "markov_ppl_transition_weighted": float(2.0 ** (float(total_markov_nll_bits_weighted) / float(total_transitions)))
            if total_transitions > 0
            else float("nan"),
            "change_rate_transition_weighted": float(total_changes) / float(total_transitions)
            if total_transitions > 0
            else float("nan"),
        }

        # Optional oracle reconstruction metrics
        if int(args.oracle_recon) > 0 and total_items > 0:
            from midigroove_poc.eval import _envelope_rms_corr, _mr_stft_sc  # reuse repo's metric defs

            n = int(min(int(args.oracle_recon), total_items))
            sample_ids = random.sample(use_ids, k=n) if n < len(use_ids) else list(use_ids)

            rmses: List[float] = []
            maes: List[float] = []
            scs: List[float] = []
            envs: List[float] = []
            lens: List[int] = []

            for sid in sample_ids:
                npz_path = sid2npz[sid]
                try:
                    y_hat, sr_hat, y_ref, sr_ref = _decode_oracle_audio(
                        npz_path, encoder_model=str(label), device=str(args.device) if args.device else None
                    )
                except Exception:
                    # allow token stats to succeed even if codec decode isn't available in the env
                    continue
                if y_hat.size == 0 or y_ref.size == 0:
                    continue
                if int(sr_hat) != int(sr_ref):
                    y_ref = _resample_linear(y_ref, int(sr_ref), int(sr_hat))
                    sr_ref = int(sr_hat)
                n0 = int(min(y_hat.size, y_ref.size))
                y_hat = y_hat[:n0]
                y_ref = y_ref[:n0]
                if n0 <= 0:
                    continue

                diff = (y_hat - y_ref).astype(np.float64)
                rmses.append(float(np.sqrt(float(np.mean(diff * diff)))))
                maes.append(float(np.mean(np.abs(diff))))
                scs.append(float(_mr_stft_sc(y_hat, y_ref)))
                envs.append(float(_envelope_rms_corr(y_hat, y_ref, sr=int(sr_ref))))
                lens.append(int(n0))

            cache_summary["oracle_recon"] = {
                "num_items_scored": int(len(rmses)),
                "rmse_mean": float(np.mean(rmses)) if rmses else float("nan"),
                "mae_mean": float(np.mean(maes)) if maes else float("nan"),
                "mr_stft_sc_mean": float(np.mean(scs)) if scs else float("nan"),
                "env_rms_corr_mean": float(np.mean(envs)) if envs else float("nan"),
                "sample_len_mean": float(np.mean(lens)) if lens else None,
            }

        if int(args.probe_steps) > 0 and total_items > 0:
            n_probe = int(min(int(args.probe_items), len(use_ids)))
            probe_ids = random.sample(use_ids, k=n_probe) if n_probe < len(use_ids) else list(use_ids)
            probe_npzs = [sid2npz[sid] for sid in probe_ids]
            cache_summary["probe"] = _run_probe(
                npz_paths=probe_npzs,
                steps=int(args.probe_steps),
                batch_size=int(args.probe_batch),
                hidden=int(args.probe_hidden),
                lr=float(args.probe_lr),
                seed=int(args.seed),
                device=str(args.device) if args.device else None,
            )

        out["caches"][label] = cache_summary

    txt = json.dumps(out, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt + "\n", encoding="utf-8")
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
