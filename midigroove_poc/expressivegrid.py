#!/usr/bin/env python3
"""Midigroove: fully expressive drum-grid (microtiming/velocities) -> codec tokens.

This script trains a Transformer encoder to predict per-frame Encodec codebook
indices from the *fully expressive* MIDI-derived conditioning grids:
  - drum_hit:     [D, T] float32  (soft-hit lane; includes microtiming)
  - drum_vel:     [D, T] float32  (velocity at onsets; zeros if unknown)
  - drum_sustain: [D, T] float32  (sustain proxy; zeros if unknown)
  - hh_open_cc4:  [T]   float32  (hi-hat openness; zeros if unknown)

Targets:
  - tgt: [C, T] int64 (Encodec token indices; typically C=4)

Expected cache format
---------------------
Use an existing Midigroove cache built by:
  `python -m midigroove_poc drumgrid train --precache --precache-only ...`

Such caches typically contain:
  items/*.npz  (each with the arrays above + bpm/drummer_id/beat_pos)
  manifest_midigroove_train_*.jsonl
  manifest_midigroove_validation_*.jsonl
  midigroove_vocab.json

CLI
---
Train:
  python -m midigroove_poc expressivegrid train \\
    --cache cache/midigroove_encodec_4beats_big_equal \\
    --device cuda:0 \\
    --save artifacts/checkpoints/midigroove_expressivegrid_to_encodec.pt

Predict+decode from an existing cache item (or a humanizer output .npz):
  python -m midigroove_poc expressivegrid predict \\
    --ckpt artifacts/checkpoints/midigroove_expressivegrid_to_encodec.pt \\
    --npz cache/midigroove_encodec_4beats_big_equal/items/<id>.npz \\
    --out out.wav
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from midigroove_poc.runtime import configure_runtime

configure_runtime()

PAD_ID = 2048  # must match dataset cache
VOCAB_SIZE = PAD_ID + 1


def _exit_with_error(msg: str) -> None:
    raise SystemExit(msg)


def _require_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"`numpy` is required (pip install numpy). Import error: {e}")


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "`torch` is required. Install PyTorch from https://pytorch.org/get-started/locally/.\n"
            f"Import error: {e}"
        )


def _require_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"`tqdm` is required (pip install tqdm). Import error: {e}")


def _is_oom(e: BaseException) -> bool:
    s = (str(e) or "").lower()
    return ("out of memory" in s) or ("cuda out of memory" in s) or ("cudaerrormemoryallocation" in s) or ("cublas" in s and "alloc" in s)


def _atomic_torch_save(path: Path, payload: object) -> None:
    torch = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    torch.save(payload, tmp)
    tmp.replace(path)


def _default_metrics_path(*, encoder_model: str) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = str(encoder_model or "encodec").strip().lower()
    return out_dir / f"expressivegrid_{kind}_{ts}.csv"


def _jsonable(obj: object) -> object:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return str(obj)


def _read_npz(path: Path) -> Dict[str, "Any"]:
    np = _require_numpy()
    d = np.load(path, allow_pickle=False)
    return {str(k): np.asarray(d[k]) for k in getattr(d, "files", [])}


def _resolve_cache_dir(cache_dir: Path) -> Path:
    """Make CLI friendlier when users omit the leading `cache/`."""
    cache_dir = Path(cache_dir)
    if cache_dir.is_dir():
        return cache_dir
    if not cache_dir.is_absolute():
        alt = Path("cache") / cache_dir
        if alt.is_dir():
            return alt
    return cache_dir


def _load_vocab_from_cache(cache_dir: Path) -> Dict[str, Any]:
    p = Path(cache_dir) / "midigroove_vocab.json"
    if not p.is_file():
        _exit_with_error(f"Missing vocab file: {p} (expected from Midigroove cache precompute).")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        _exit_with_error(f"Failed to read vocab json: {p}\n{e}")


def _split_items_by_manifest(cache_dir: Path) -> Dict[str, List[Path]]:
    """Read cache manifests and return split -> list of item .npz paths."""
    cache_dir = _resolve_cache_dir(Path(cache_dir))
    if not cache_dir.is_dir():
        _exit_with_error(f"Cache directory not found: {cache_dir}")

    manifests = sorted(cache_dir.glob("manifest_midigroove_*_*.jsonl"))
    if not manifests:
        _exit_with_error(f"No manifests found under {cache_dir} (expected manifest_midigroove_<split>_*.jsonl).")

    by_split: Dict[str, List[Path]] = {}
    rx = re.compile(r"^manifest_midigroove_(?P<split>[^_]+)_[^/]+\.jsonl$")
    for mp in manifests:
        m = rx.match(mp.name)
        if not m:
            continue
        split = str(m.group("split")).strip().lower()
        items: List[Path] = []
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
                # Manifest usually stores relative paths like "cache/<...>/items/<id>.npz".
                if not p.is_file():
                    p2 = cache_dir / npz_s
                    if p2.is_file():
                        p = p2
            if p.is_file():
                items.append(p)
        if items:
            by_split[split] = items
    return by_split



# ----------------
# Training dataset
# ----------------
@dataclass(frozen=True)
class ExpressiveCacheItem:
    grid: "Any"  # [F,T] float32
    tgt: "Any"  # [C,T] int64
    beat_pos: "Any"  # [T] int64
    bpm: float
    drummer_id: int


class ExpressiveGridDataset:
    def __init__(
        self,
        npz_paths: List[Path],
        *,
        vocab: Dict[str, Any] | None = None,
        include_sustain: bool = False,
        include_hh_cc4: bool = False,
    ) -> None:
        self.paths = list(npz_paths)
        self.vocab = dict(vocab or {})
        self.include_sustain = bool(include_sustain)
        self.include_hh_cc4 = bool(include_hh_cc4)

        v = self.vocab
        self._drummer_to_id = dict(v.get("drummer_to_id", {}) or {}) if isinstance(v.get("drummer_to_id", {}), dict) else {}

    def __len__(self) -> int:
        return int(len(self.paths))

    def _load_item(self, path: Path) -> ExpressiveCacheItem:
        np = _require_numpy()
        d = _read_npz(path)

        hit = np.asarray(d["drum_hit"], dtype=np.float32)
        if hit.ndim != 2:
            raise ValueError(f"drum_hit must be [D,T], got {hit.shape} in {path}")
        D, T = hit.shape

        # Fixed: always include hit + vel. Optional: sustain + hh_cc4.
        pieces = [hit]
        vel = np.asarray(d.get("drum_vel", np.zeros_like(hit)), dtype=np.float32)
        if vel.shape != (D, T):
            vel = np.broadcast_to(vel, (D, T)).copy()
        pieces.append(vel)
        if self.include_sustain:
            sus = np.asarray(d.get("drum_sustain", np.zeros_like(hit)), dtype=np.float32)
            if sus.shape != (D, T):
                sus = np.broadcast_to(sus, (D, T)).copy()
            pieces.append(sus)
        if self.include_hh_cc4:
            hh = np.asarray(d.get("hh_open_cc4", np.zeros((T,), dtype=np.float32)), dtype=np.float32).reshape(-1)
            if hh.shape[0] != T:
                hh = np.resize(hh, (T,)).astype(np.float32, copy=False)
            pieces.append(hh[None, :])

        grid = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)

        tgt = np.asarray(d["tgt"], dtype=np.int64)
        if tgt.ndim != 2 or tgt.shape[1] != T:
            raise ValueError(f"tgt must be [C,T={T}], got {tgt.shape} in {path}")

        beat_pos_raw = d.get("beat_pos", None)
        if beat_pos_raw is None:
            win_raw = d.get("window_seconds", None)
            try:
                win_s = float(np.asarray(win_raw, dtype=np.float32).item()) if win_raw is not None else 2.0
            except Exception:
                win_s = 2.0
            fps = float(T) / float(max(1e-6, win_s))
            bpm = float(np.asarray(d.get("bpm", 120.0), dtype=np.float32).item())
            beat_pos = _compute_beat_pos(T, fps=fps, bpm=bpm)
        else:
            beat_pos = np.asarray(beat_pos_raw, dtype=np.int64).reshape(-1)
            if beat_pos.shape[0] != T:
                beat_pos = np.resize(beat_pos, (T,)).astype(np.int64, copy=False)

        bpm = float(np.asarray(d.get("bpm", 120.0), dtype=np.float32).item())
        drummer_id = int(np.asarray(d.get("drummer_id", 0), dtype=np.int64).item())
        if drummer_id == 0 and self._drummer_to_id:
            try:
                drummer = str(np.asarray(d.get("drummer", ""), dtype=str).item())
            except Exception:
                drummer = ""
            drummer_id = int(self._drummer_to_id.get(drummer, self._drummer_to_id.get(drummer.strip(), 0)))

        return ExpressiveCacheItem(
            grid=grid,
            tgt=tgt,
            beat_pos=beat_pos,
            bpm=bpm,
            drummer_id=drummer_id,
        )

    def __getitem__(self, idx: int) -> ExpressiveCacheItem:
        return self._load_item(self.paths[int(idx)])

    @staticmethod
    def collate_fn(items: List[ExpressiveCacheItem]) -> Dict[str, "Any"]:
        torch = _require_torch()
        if not items:
            raise ValueError("empty batch")
        F = int(items[0].grid.shape[0])
        C = int(items[0].tgt.shape[0])
        Tmax = int(max(int(it.grid.shape[1]) for it in items))

        grid = torch.zeros((len(items), F, Tmax), dtype=torch.float32)
        beat_pos = torch.zeros((len(items), Tmax), dtype=torch.long)
        tgt = torch.full((len(items), C, Tmax), int(PAD_ID), dtype=torch.long)
        valid = torch.zeros((len(items), Tmax), dtype=torch.bool)

        bpm = torch.zeros((len(items),), dtype=torch.float32)
        drummer_id = torch.zeros((len(items),), dtype=torch.long)

        for i, it in enumerate(items):
            Ti = int(it.grid.shape[1])
            if int(it.grid.shape[0]) != F:
                raise ValueError("feature dim mismatch in batch")
            if int(it.tgt.shape[0]) != C:
                raise ValueError("num_codebooks mismatch in batch")
            grid[i, :, :Ti] = torch.from_numpy(it.grid)
            beat_pos[i, :Ti] = torch.from_numpy(it.beat_pos[:Ti]).to(dtype=torch.long)
            tgt[i, :, :Ti] = torch.from_numpy(it.tgt[:, :Ti]).to(dtype=torch.long)
            valid[i, :Ti] = True
            bpm[i] = float(it.bpm)
            drummer_id[i] = int(it.drummer_id)

        return {
            "grid": grid,
            "beat_pos": beat_pos,
            "tgt_codes": tgt,
            "valid_mask": valid,
            "bpm": bpm,
            "drummer_id": drummer_id,
        }


# -------------
# Model + train
# -------------

def _build_model(*, num_codebooks: int, in_dim: int, cfg: Dict[str, Any]) -> "Any":
    """Build the expressivegrid->tokens model.

    Conditioning is fixed and always includes:
      - expressive grid features: hit + vel (+ optional sustain, hh_cc4)
      - beat_pos (0..3)
      - bpm
      - drummer_id

    Codec choice (encodec/dac/xcodec) only changes the target tokens stored in the cache.
    """

    torch = _require_torch()
    import torch.nn as nn  # type: ignore

    vocab = dict(cfg.get("vocab", {}) or {})
    drummer_to_id = dict(vocab.get("drummer_to_id", {}) or {})
    drummer_vocab_size = int(max([int(v) for v in drummer_to_id.values()] + [0]) + 1)

    class ExpressiveGridToTokensModel(nn.Module):
        def __init__(
            self,
            *,
            num_codebooks: int,
            in_dim: int,
            d_model: int,
            n_layers: int,
            n_heads: int,
            max_frames: int,
            drummer_vocab_size: int,
            dropout: float,
            ff_mult: int,
        ) -> None:
            super().__init__()
            self.num_codebooks = int(num_codebooks)
            self.in_dim = int(in_dim)
            self.max_frames = int(max_frames)

            self.grid_proj = nn.Linear(self.in_dim, d_model)
            self.beat_emb = nn.Embedding(4, d_model)
            self.pos_emb = nn.Embedding(self.max_frames, d_model)

            self.drummer_emb = nn.Embedding(int(max(1, drummer_vocab_size)), d_model)
            self.bpm_proj = nn.Linear(1, d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=int(ff_mult) * int(d_model),
                dropout=float(dropout),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
            self.drop = nn.Dropout(float(dropout))

            self.head = nn.Linear(d_model, self.num_codebooks * VOCAB_SIZE)

        def forward(
            self,
            *,
            grid: "torch.Tensor",  # [B,F,T]
            beat_pos: "torch.Tensor",  # [B,T]
            bpm: "torch.Tensor",  # [B]
            drummer_id: "torch.Tensor",  # [B]
            valid_mask: Optional["torch.Tensor"] = None,  # [B,T]
        ) -> "torch.Tensor":
            if grid.dim() != 3:
                raise ValueError(f"grid must be [B,F,T], got {tuple(grid.shape)}")
            B, F, T = grid.shape
            if int(F) != int(self.in_dim):
                raise ValueError(f"expected in_dim={self.in_dim}, got {F}")
            if beat_pos.shape != (B, T):
                raise ValueError(f"beat_pos must be [B,T]={B,T}, got {tuple(beat_pos.shape)}")
            if valid_mask is not None and valid_mask.shape != (B, T):
                raise ValueError(f"valid_mask must be [B,T]={B,T}, got {tuple(valid_mask.shape)}")
            if T > self.max_frames:
                raise ValueError(f"T={T} exceeds max_frames={self.max_frames}; increase --max-frames")

            x = grid.to(dtype=torch.float32).permute(0, 2, 1).contiguous()  # [B,T,F]
            x = self.grid_proj(x)  # [B,T,d]

            beat_pos = torch.clamp(beat_pos.to(dtype=torch.long), 0, 3)
            x = x + self.beat_emb(beat_pos)

            pos = torch.arange(T, device=x.device, dtype=torch.long)[None, :]
            x = x + self.pos_emb(pos)

            bpm = bpm.to(dtype=torch.float32).view(B, 1)
            bpm = torch.log1p(torch.clamp(bpm, min=0.0))
            meta = self.bpm_proj(bpm) + self.drummer_emb(drummer_id.to(dtype=torch.long))
            x = x + meta[:, None, :]

            src_key_padding_mask = None if valid_mask is None else ~valid_mask.to(dtype=torch.bool)
            h = self.encoder(self.drop(x), src_key_padding_mask=src_key_padding_mask)  # [B,T,d]
            logits = self.head(h)  # [B,T,C*V]
            logits = logits.view(B, T, self.num_codebooks, VOCAB_SIZE).permute(0, 2, 1, 3).contiguous()
            return logits

    return ExpressiveGridToTokensModel(
        num_codebooks=int(num_codebooks),
        in_dim=int(in_dim),
        d_model=int(cfg.get("d_model", 768) or 768),
        n_layers=int(cfg.get("n_layers", 6) or 6),
        n_heads=int(cfg.get("n_heads", 8) or 8),
        max_frames=int(cfg.get("max_frames", 4096) or 4096),
        drummer_vocab_size=int(drummer_vocab_size),
        dropout=float(cfg.get("dropout", 0.1) or 0.1),
        ff_mult=int(cfg.get("ff_mult", 4) or 4),
    )


def _eval_loss(model: "Any", loader: "Any", device: "Any", *, max_batches: int) -> float:
    torch = _require_torch()
    import torch.nn.functional as F  # type: ignore

    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if int(max_batches) > 0 and i >= int(max_batches):
                break
            logits = model(
                grid=batch["grid"].to(device, non_blocking=True),
                beat_pos=batch["beat_pos"].to(device, non_blocking=True),
                bpm=batch["bpm"].to(device, non_blocking=True),
                drummer_id=batch["drummer_id"].to(device, non_blocking=True),
                valid_mask=batch["valid_mask"].to(device, non_blocking=True),
            )
            tgt = batch["tgt_codes"].to(device, non_blocking=True)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=PAD_ID)
            v = float(loss.item())
            if not math.isfinite(v):
                model.train()
                return float("inf")
            losses.append(v)
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def _train_loop(
    *,
    train_loader: "Any",
    val_loader: "Any",
    device: "Any",
    model_cfg: Dict[str, Any],
    num_codebooks: int,
    in_dim: int,
    steps: int,
    log_every: int,
    early_stop_steps: int,
    eval_max_batches: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    save_path: Path | None,
    metrics_path: Path | None,
    save_best_only: bool,
    kind: str,
    extra_ckpt_fields: Dict[str, Any] | None = None,
) -> Tuple[float, int]:
    torch = _require_torch()
    tqdm = _require_tqdm()
    import torch.nn.functional as F  # type: ignore

    model = _build_model(num_codebooks=int(num_codebooks), in_dim=int(in_dim), cfg=model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_val = float("inf")
    best_step = -1
    last_val: Optional[float] = None
    steps = int(steps)
    log_every = int(log_every)
    early_stop_steps = int(max(0, early_stop_steps))

    metrics_path = Path(metrics_path) if metrics_path is not None else None
    if metrics_path is None:
        if save_path is not None:
            metrics_path = Path(str(save_path) + ".metrics.csv")
        else:
            metrics_path = _default_metrics_path(encoder_model=str(model_cfg.get("encoder_model", "encodec") or "encodec"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_f = metrics_path.open("w", encoding="utf-8", newline="")
    metrics_w = csv.DictWriter(
        metrics_f,
        fieldnames=[
            "step",
            "train_loss",
            "val_loss",
            "best_val",
            "best_step",
            "lr",
            "elapsed_sec",
        ],
    )
    metrics_w.writeheader()
    t0 = time.time()

    pbar = tqdm(total=steps, desc="train", dynamic_ncols=True)
    it = iter(train_loader)
    step = 0
    try:
        while step < steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)

            logits = model(
                grid=batch["grid"].to(device, non_blocking=True),
                beat_pos=batch["beat_pos"].to(device, non_blocking=True),
                bpm=batch["bpm"].to(device, non_blocking=True),
                drummer_id=batch["drummer_id"].to(device, non_blocking=True),
                valid_mask=batch["valid_mask"].to(device, non_blocking=True),
            )
            tgt = batch["tgt_codes"].to(device, non_blocking=True)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=PAD_ID)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss (try lower lr or smaller model)")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip) and float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            do_eval = (log_every > 0) and (step % log_every == 0)
            if do_eval:
                last_val = _eval_loss(model, val_loader, device, max_batches=int(eval_max_batches))
                if math.isfinite(float(last_val)) and float(last_val) < float(best_val):
                    best_val = float(last_val)
                    best_step = int(step)
                    if save_path is not None:
                        payload = {
                            "kind": str(kind),
                            "model": model.state_dict(),
                            "cfg": _jsonable(model_cfg),
                            "num_codebooks": int(num_codebooks),
                            "in_dim": int(in_dim),
                            "best_val": float(best_val),
                            "best_step": int(best_step),
                        }
                        if extra_ckpt_fields:
                            payload.update(extra_ckpt_fields)
                        _atomic_torch_save(save_path, payload)

                lr_now = float(opt.param_groups[0].get("lr", 0.0)) if opt.param_groups else 0.0
                metrics_w.writerow(
                    {
                        "step": int(step),
                        "train_loss": float(loss.detach().cpu().item()),
                        "val_loss": float(last_val),
                        "best_val": float(best_val),
                        "best_step": int(best_step),
                        "lr": lr_now,
                        "elapsed_sec": float(time.time() - t0),
                    }
                )
                metrics_f.flush()

                try:
                    pbar.write(
                        f"step={step} train_loss={loss.item():.4f} val_loss={float(last_val):.4f} best_val={best_val:.4f}@{best_step}"
                    )
                except Exception:
                    pass
                pbar.set_postfix(train_loss=f"{loss.item():.4f}", val_loss=f"{float(last_val):.4f}")
            else:
                if last_val is not None and math.isfinite(float(last_val)):
                    pbar.set_postfix(train_loss=f"{loss.item():.4f}", val_loss=f"{float(last_val):.4f}")
                else:
                    pbar.set_postfix(train_loss=f"{loss.item():.4f}")

            step += 1
            pbar.update(1)

            if early_stop_steps > 0 and best_step >= 0 and (step - best_step) >= early_stop_steps:
                break
    finally:
        try:
            metrics_f.close()
        except Exception:
            pass
        try:
            pbar.write(f"wrote metrics: {metrics_path}")
        except Exception:
            pass

    return float(best_val), int(best_step)


def _choose_device(arg: Optional[str]) -> str:
    torch = _require_torch()
    if arg is not None and str(arg).strip() != "":
        return str(arg)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _seed_everything(seed: int) -> None:
    np = _require_numpy()
    torch = _require_torch()
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _infer_feature_flags(*, in_dim: int, d_drum: int) -> Tuple[bool, bool]:
    """Infer (include_sustain, include_hh_cc4) from in_dim and drum channels D."""
    in_dim = int(in_dim)
    D = int(d_drum)
    if D <= 0:
        return False, False
    base = 2 * D  # hit + vel
    if in_dim == base:
        return False, False
    if in_dim == base + 1:
        return False, True
    if in_dim == base + D:
        return True, False
    if in_dim == base + D + 1:
        return True, True
    raise ValueError(f"Cannot infer feature flags: in_dim={in_dim} is not compatible with D={D} (expected 2D, 2D+1, 3D, 3D+1).")


def cmd_train(args: argparse.Namespace) -> None:
    torch = _require_torch()
    from torch.utils.data import DataLoader  # type: ignore

    cache = _resolve_cache_dir(Path(args.cache))
    splits = _split_items_by_manifest(cache)
    if "validation" not in splits and "val" in splits:
        splits["validation"] = splits["val"]
    if "train" not in splits or "validation" not in splits:
        _exit_with_error(f"Cache must contain train and validation splits; found: {sorted(splits.keys())}")

    vocab = _load_vocab_from_cache(cache)
    train_paths = list(splits["train"])
    val_paths = list(splits["validation"])
    encoder_model = str(getattr(args, "encoder_model", None) or "encodec").strip().lower()
    include_sustain = bool(getattr(args, "include_sustain", False))
    include_hh_cc4 = bool(getattr(args, "include_hh_cc4", False))

    train_ds = ExpressiveGridDataset(
        train_paths,
        vocab=vocab,
        include_sustain=include_sustain,
        include_hh_cc4=include_hh_cc4,
    )
    val_ds = ExpressiveGridDataset(
        val_paths,
        vocab=vocab,
        include_sustain=include_sustain,
        include_hh_cc4=include_hh_cc4,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=bool(int(args.num_workers) > 0),
        collate_fn=ExpressiveGridDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=bool(int(args.num_workers) > 0),
        collate_fn=ExpressiveGridDataset.collate_fn,
        drop_last=False,
    )

    first = next(iter(train_loader))
    C = int(first["tgt_codes"].shape[1])
    in_dim = int(first["grid"].shape[1])

    device_str = _choose_device(args.device)
    device = torch.device(device_str)

    model_cfg = {
        "encoder_model": str(encoder_model),
        "d_model": int(args.d_model),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "max_frames": int(args.max_frames),
        "dropout": float(args.dropout),
        "ff_mult": int(args.ff_mult),
        "vocab": vocab,
        "include_sustain": bool(include_sustain),
        "include_hh_cc4": bool(include_hh_cc4),
    }

    _seed_everything(int(args.seed))
    best_val, best_step = _train_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        model_cfg=model_cfg,
        num_codebooks=C,
        in_dim=in_dim,
        steps=int(args.steps),
        log_every=int(args.log_every),
        early_stop_steps=int(getattr(args, "early_stop_steps", 3000)),
        eval_max_batches=int(args.eval_max_batches),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        save_path=Path(args.save) if args.save else None,
        metrics_path=Path(args.metrics_out) if getattr(args, "metrics_out", None) else None,
        save_best_only=True,
        kind="midigroove_expressivegrid_to_tokens",
        extra_ckpt_fields=None,
    )
    print(f"Done. best_val={best_val:.4f} at step={best_step}")



def cmd_predict(args: argparse.Namespace) -> None:
    torch = _require_torch()
    np = _require_numpy()

    ckpt, cfg = _load_ckpt(Path(args.ckpt), device=str(args.device))
    C = int(ckpt.get("num_codebooks", 0) or 0)
    in_dim = int(ckpt.get("in_dim", 0) or 0)
    if C <= 0 or in_dim <= 0:
        _exit_with_error("ckpt missing num_codebooks/in_dim")

    vocab = cfg.get("vocab", {})
    if not isinstance(vocab, dict):
        vocab = {}
    cfg2 = dict(cfg)
    cfg2["vocab"] = vocab
    encoder_model = str(getattr(args, "encoder_model", None) or cfg2.get("encoder_model", "encodec") or "encodec").strip().lower()

    model = _build_model(num_codebooks=C, in_dim=in_dim, cfg=cfg2).to(torch.device(str(args.device)))
    model.load_state_dict(ckpt["model"])
    model.eval()

    d = _read_npz(Path(args.npz))
    hit = np.asarray(d["drum_hit"], dtype=np.float32)
    if hit.ndim != 2:
        _exit_with_error(f"drum_hit must be [D,T], got {hit.shape}")
    D, T = hit.shape

    try:
        include_sustain, include_hh_cc4 = _infer_feature_flags(
            in_dim=int(in_dim),
            d_drum=int(D),
        )
    except Exception as e:
        _exit_with_error(str(e))
    # Prefer explicit cfg flags when present.
    if "include_sustain" in cfg2:
        include_sustain = bool(cfg2.get("include_sustain", include_sustain))
    if "include_hh_cc4" in cfg2:
        include_hh_cc4 = bool(cfg2.get("include_hh_cc4", include_hh_cc4))

    expected_in_dim = int(2 * D + (D if include_sustain else 0) + (1 if include_hh_cc4 else 0))
    if int(expected_in_dim) != int(in_dim):
        _exit_with_error(
            f"Input feature mismatch: ckpt in_dim={in_dim}, expected from npz={expected_in_dim} "
            f"(D={D}; hit+vel{' +sustain' if include_sustain else ''}{' +hh_cc4' if include_hh_cc4 else ''})"
        )

    pieces = [hit]
    vel = np.asarray(d.get("drum_vel", np.zeros_like(hit)), dtype=np.float32)
    if vel.shape != (D, T):
        vel = np.broadcast_to(vel, (D, T)).copy()
    pieces.append(vel)
    if include_sustain:
        sus = np.asarray(d.get("drum_sustain", np.zeros_like(hit)), dtype=np.float32)
        if sus.shape != (D, T):
            sus = np.broadcast_to(sus, (D, T)).copy()
        pieces.append(sus)
    if include_hh_cc4:
        hh = np.asarray(d.get("hh_open_cc4", np.zeros((T,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if hh.shape[0] != T:
            hh = np.resize(hh, (T,)).astype(np.float32, copy=False)
        pieces.append(hh[None, :])
    grid = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)

    beat_pos_raw = d.get("beat_pos", None)
    if beat_pos_raw is None:
        bpm = float(np.asarray(d.get("bpm", args.bpm), dtype=np.float32).item())
        fps = float(np.asarray(d.get("fps", args.fps), dtype=np.float32).item())
        beat_pos = _compute_beat_pos(T, fps=fps, bpm=bpm)
    else:
        beat_pos = np.asarray(beat_pos_raw, dtype=np.int64).reshape(-1)
        if beat_pos.shape[0] != T:
            beat_pos = np.resize(beat_pos, (T,)).astype(np.int64, copy=False)

    bpm = float(np.asarray(d.get("bpm", args.bpm), dtype=np.float32).item())
    drummer_id = int(np.asarray(d.get("drummer_id", 0), dtype=np.int64).item())
    if drummer_id == 0:
        dt = cfg2.get("vocab", {}).get("drummer_to_id", {}) if isinstance(cfg2.get("vocab", {}), dict) else {}
        if isinstance(dt, dict) and dt:
            try:
                drummer = str(np.asarray(d.get("drummer", ""), dtype=str).item())
            except Exception:
                drummer = ""
            drummer_id = int(dt.get(drummer, dt.get(drummer.strip(), 0)))

    dev = torch.device(str(args.device))
    with torch.inference_mode():
        logits = model(
            grid=torch.from_numpy(grid).unsqueeze(0).to(dev),
            beat_pos=torch.from_numpy(np.asarray(beat_pos, dtype=np.int64)).unsqueeze(0).to(dev),
            bpm=torch.tensor([bpm], dtype=torch.float32, device=dev),
            drummer_id=torch.tensor([drummer_id], dtype=torch.long, device=dev),
            valid_mask=None,
        )  # [1,C,T,V]
        if bool(args.sample):
            pred = _sample_from_logits(logits, temperature=float(args.temperature), topk=int(args.topk), seed=int(args.seed))
        else:
            pred = logits.argmax(dim=-1).squeeze(0).to(dtype=torch.long)

    pred = torch.where(pred == int(PAD_ID), torch.zeros_like(pred), pred)

    if args.out_tokens is not None:
        out_tokens = Path(args.out_tokens)
        out_tokens.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"codes": pred.detach().cpu()}, out_tokens)
        print(f"Wrote tokens: {out_tokens} (shape={tuple(pred.shape)})")

    if args.out is not None:
        repo_root = Path(__file__).resolve().parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from data.codecs import decode_tokens_to_audio

        audio_b1, sr = decode_tokens_to_audio(pred.detach().cpu(), encoder_model=str(encoder_model), device=str(args.decode_device or args.device))
        _write_wav(Path(args.out), audio_b1[0], int(sr))
        print(f"Wrote wav: {args.out} (sr={sr}, T={T})")


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Midigroove expressive-grid -> codec tokens.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tr = sub.add_parser(
        "train",
        help="Train expressive-grid -> codec tokens model from an existing cache (conditioning is fixed; ignores kit/style fields).",
    )
    p_tr.add_argument("--cache", type=Path, required=True)
    p_tr.add_argument("--save", type=Path, required=True)
    p_tr.add_argument("--device", type=str, default=None)
    p_tr.add_argument("--batch-size", type=int, default=24)
    p_tr.add_argument("--num-workers", type=int, default=0)
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--steps", type=int, default=200_000)
    p_tr.add_argument("--log-every", type=int, default=200, help="Run validation every N steps.")
    p_tr.add_argument("--early-stop-steps", type=int, default=3000, help="Stop after N steps without val loss improvement (0 disables).")
    p_tr.add_argument("--eval-max-batches", type=int, default=50, help="If >0, limit validation to this many batches.")
    p_tr.add_argument("--metrics-out", type=Path, default=None, help="Write train/val metrics CSV (one row per eval). Defaults next to --save.")
    p_tr.add_argument("--include-sustain", action="store_true", help="Include sustain lane as input (drum_sustain). Default: off.")
    p_tr.add_argument("--include-hh-cc4", action="store_true", help="Include hi-hat CC#4 lane as input (hh_open_cc4). Default: off.")
    p_tr.add_argument("--lr", type=float, default=6e-5)
    p_tr.add_argument("--weight-decay", type=float, default=0.0)
    p_tr.add_argument("--grad-clip", type=float, default=1.0)
    p_tr.add_argument("--d-model", type=int, default=768)
    p_tr.add_argument("--n-layers", type=int, default=6)
    p_tr.add_argument("--n-heads", type=int, default=8)
    p_tr.add_argument("--ff-mult", type=int, default=4)
    p_tr.add_argument("--dropout", type=float, default=0.1)
    p_tr.add_argument("--max-frames", type=int, default=4096)
    p_tr.add_argument(
        "--encoder-model",
        type=str,
        default="encodec",
        choices=["encodec", "dac", "xcodec"],
        help="Which codec family the target tokens come from (stored in ckpt; used for decoding in `predict`).",
    )
    p_tr.set_defaults(func=cmd_train)


    p_pr = sub.add_parser("predict", help="Predict tokens from an input expressive-grid .npz; optionally decode to wav.")
    p_pr.add_argument("--ckpt", type=Path, required=True)
    p_pr.add_argument("--npz", type=Path, required=True, help="Input .npz with at least drum_hit (+ optional drum_vel/drum_sustain/hh_open_cc4). Missing lanes default to zeros.")
    p_pr.add_argument("--device", type=str, default="cpu")
    p_pr.add_argument("--decode-device", type=str, default=None)
    p_pr.add_argument("--encoder-model", type=str, default=None, choices=["encodec", "dac", "xcodec"], help="Override ckpt cfg encoder_model for decoding.")
    p_pr.add_argument("--out", type=Path, default=None, help="Optional output wav path.")
    p_pr.add_argument("--out-tokens", type=Path, default=None, help="Optional output .pt file with {'codes': [C,T]}.")
    p_pr.add_argument("--bpm", type=float, default=120.0, help="Used only if input .npz does not include bpm.")
    p_pr.add_argument("--fps", type=float, default=50.0, help="Used only if input .npz does not include beat_pos and does include fps.")
    p_pr.add_argument("--sample", action="store_true")
    p_pr.add_argument("--temperature", type=float, default=1.0)
    p_pr.add_argument("--topk", type=int, default=0)
    p_pr.add_argument("--seed", type=int, default=0)
    p_pr.set_defaults(func=cmd_predict)

    args = ap.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
