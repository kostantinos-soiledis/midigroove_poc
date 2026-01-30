"""Midigroove: drum-grid + rhythm/meta -> codec codes (Encodec/DAC/X-Codec).

Prototype pipeline:
  - Build a fixed-length cache where input is a MIDI-derived drum grid (plus BPM
    and categorical metadata) and target is discrete codec codes for the WAV.
    Length can be fixed-seconds (e.g. 2s) or fixed-beats (e.g. 4 beats; seconds
    derived from BPM).
  - Train a small Transformer encoder to predict per-frame codebook indices.

Run (cache build):
  python -m midigroove_poc drumgrid train --precache --precache-only ...
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from pathlib import Path
from typing import Callable, Optional

from midigroove_poc.runtime import configure_runtime

configure_runtime()

try:  # allow `--help` to work in minimal envs
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore
    from torch.utils.data import DataLoader  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    DataLoader = None  # type: ignore

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _require_torch():
    if torch is None:  # pragma: no cover
        raise RuntimeError(
            "`torch` is required for training/caching. "
            "Install PyTorch from https://pytorch.org/get-started/locally/."
        )
    return torch


def _require_tqdm():
    if tqdm is None:  # pragma: no cover
        raise RuntimeError("`tqdm` is required for progress bars (pip install tqdm).")
    return tqdm

from data.codecs import DacCodesEncoder, EncodecCodesEncoder, XcodecCodesEncoder
# Note: `data.midigroove_encodec_dataset` (torch/pandas/pretty_midi/etc) is a heavy
# import. Keep it lazy so `python -m ... --help` works in minimal envs.

PAD_ID = 2048  # must match dataset
VOCAB_SIZE = PAD_ID + 1


def _steps_for_epochs(*, train_len: int, batch_size: int, epochs: float, drop_last: bool = True) -> int:
    train_len = int(train_len)
    batch_size = int(batch_size)
    epochs = float(epochs)
    if train_len <= 0 or batch_size <= 0 or not (epochs > 0 and isfinite(epochs)):
        return 0
    if drop_last:
        steps_per_epoch = max(1, train_len // batch_size)
    else:
        steps_per_epoch = max(1, (train_len + batch_size - 1) // batch_size)
    return int(max(1, round(float(epochs) * float(steps_per_epoch))))


def _atomic_torch_save(path: Path, payload: object) -> None:
    """Atomically write a torch checkpoint to `path`.

    This avoids leaving a corrupted `.pt` if the process is interrupted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # Ensure tmp from prior crashes doesn't get mistaken for a real ckpt.
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass
    _require_torch().save(payload, tmp)
    tmp.replace(path)


if nn is not None:
    class DrumGridRhythmMetaToCodesModel(nn.Module):
        def __init__(
            self,
            *,
            num_codebooks: int,
            drum_grid_channels: int,
            style_vocab_size: int,
            beat_type_vocab_size: int,
            kit_category_vocab_size: int,
            d_model: int = 512,
            n_layers: int = 6,
            n_heads: int = 8,
            ff_mult: int = 4,
            dropout: float = 0.1,
            max_frames: int = 4096,
        ) -> None:
            super().__init__()
            self.num_codebooks = int(num_codebooks)
            self.drum_grid_channels = int(drum_grid_channels)
            self.max_frames = int(max_frames)

            self.grid_proj = nn.Linear(self.drum_grid_channels, d_model)
            self.beat_emb = nn.Embedding(4, d_model)
            self.pos_emb = nn.Embedding(self.max_frames, d_model)

            self.style_emb = nn.Embedding(int(max(1, style_vocab_size)), d_model)
            self.beat_type_emb = nn.Embedding(int(max(1, beat_type_vocab_size)), d_model)
            self.kit_category_emb = nn.Embedding(int(max(1, kit_category_vocab_size)), d_model)
            self.bpm_proj = nn.Linear(1, d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ff_mult * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.drop = nn.Dropout(dropout)

            self.head = nn.Linear(d_model, self.num_codebooks * VOCAB_SIZE)

        def forward(
            self,
            *,
            drum_grid: torch.Tensor,
            beat_pos: torch.Tensor,
            bpm: torch.Tensor,
            style_id: torch.Tensor,
            beat_type_id: torch.Tensor,
            kit_category_id: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Return logits over code indices.

            drum_grid: [B, Dg, T]
            beat_pos:  [B, T] (0..3)
            bpm:       [B]
            style_id:  [B]
            beat_type_id: [B]
            kit_category_id:    [B]
            returns:   [B, C, T, VOCAB_SIZE]
            """
            if drum_grid.dim() != 3:
                raise ValueError(f"drum_grid must be [B,Dg,T], got {tuple(drum_grid.shape)}")
            B, Dg, T = drum_grid.shape
            if Dg != self.drum_grid_channels:
                raise ValueError(f"drum_grid_channels mismatch: expected {self.drum_grid_channels}, got {Dg}")
            if beat_pos.shape != (B, T):
                raise ValueError(f"beat_pos must be [B,T]={B,T}, got {tuple(beat_pos.shape)}")
            if valid_mask is not None and valid_mask.shape != (B, T):
                raise ValueError(f"valid_mask must be [B,T]={B,T}, got {tuple(valid_mask.shape)}")
            if T > self.max_frames:
                raise ValueError(f"T={T} exceeds max_frames={self.max_frames}; increase --max-frames")

            grid_bt = drum_grid.to(dtype=torch.float32).permute(0, 2, 1)  # [B,T,Dg]
            x = self.grid_proj(grid_bt)  # [B,T,d]

            beat_pos = torch.clamp(beat_pos.to(dtype=torch.long), 0, 3)
            x = x + self.beat_emb(beat_pos)

            pos = torch.arange(T, device=x.device, dtype=torch.long)[None, :]
            x = x + self.pos_emb(pos)

            bpm = bpm.to(dtype=torch.float32).view(B, 1)
            bpm = torch.log1p(torch.clamp(bpm, min=0.0))
            meta = self.bpm_proj(bpm)
            meta = meta + self.style_emb(style_id.to(dtype=torch.long))
            meta = meta + self.beat_type_emb(beat_type_id.to(dtype=torch.long))
            meta = meta + self.kit_category_emb(kit_category_id.to(dtype=torch.long))
            x = x + meta[:, None, :]
            if valid_mask is None:
                src_key_padding_mask = None
            else:
                src_key_padding_mask = ~valid_mask.to(dtype=torch.bool)
            h = self.encoder(self.drop(x), src_key_padding_mask=src_key_padding_mask)  # [B,T,d]
            logits = self.head(h)  # [B,T,C*V]
            logits = logits.view(B, T, self.num_codebooks, VOCAB_SIZE).permute(0, 2, 1, 3).contiguous()
            return logits

else:
    class DrumGridRhythmMetaToCodesModel:  # type: ignore
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            _require_torch()


@dataclass
class TrainCfg:
    train_csv: Path
    val_csv: Path
    train_split: str = "train"
    val_split: str = "validation"
    dataset_root: Path = Path(".")
    device: str = "cuda"
    encode_device: str = "cuda"
    encoder_model: str = "encodec"  # "encodec" | "dac" | "xcodec"
    xcodec_bandwidth: float = 2.0
    cache_dir: Path | None = None
    require_cache: bool = False
    encode_if_missing: bool = True
    window_seconds: float = 2.0
    hop_seconds: float = 2.0
    beats_per_chunk: int | None = None
    hop_beats: int | None = None
    use_style: bool = False
    beat_type_only: str | None = None
    kit_category_only: str | None = None
    kit_category_exclude: str | None = None
    kit_category_top_n: int | None = None
    target_train_clips: int | None = None
    target_val_clips: int | None = None
    stratify_clips: bool = False
    stratify_key_kind: str = "drummer_style_kit_category"
    stratify_mode: str = "proportional"
    stratify_seed: int = 0
    oversample_factor: float = 1.25
    include_vel: bool = True
    include_sustain: bool = True
    include_hh_cc4: bool = True
    batch_size: int = 8
    num_workers: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    steps: int = 20_000
    log_every: int = 50
    eval_max_batches: int = 50
    nonfinite_policy: str = "halt"  # halt|skip
    save_path: Path | None = None
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    max_frames: int = 4096
    save_best_only: bool = True
    metrics_out: Path | None = None


def build_encoder(cfg: TrainCfg) -> tuple[Optional[object], str]:
    _require_torch()
    kind = str(cfg.encoder_model or "encodec").strip().lower()
    if kind not in {"encodec", "dac", "xcodec"}:
        raise ValueError(f"Unsupported --encoder-model {cfg.encoder_model!r} (expected: encodec|dac|xcodec)")

    # For cache-only training, avoid loading the heavy codec model.
    if bool(cfg.require_cache) and not bool(cfg.encode_if_missing):
        return None, kind

    enc_device = torch.device(cfg.encode_device)
    if kind == "encodec":
        return EncodecCodesEncoder(device=enc_device), kind
    if kind == "dac":
        return DacCodesEncoder(device=enc_device), kind
    return XcodecCodesEncoder(device=enc_device, bandwidth=float(cfg.xcodec_bandwidth)), kind


def _load_or_build_vocab(cfg: TrainCfg) -> MidigrooveMetaVocab:
    from data.midigroove_encodec_dataset import MidigrooveMetaVocab, build_midigroove_vocab

    if cfg.cache_dir is not None:
        vocab_path = Path(cfg.cache_dir) / "midigroove_vocab.json"
        if vocab_path.is_file():
            obj = json.loads(vocab_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                # If this cache dir contains an older vocab schema, rebuild.
                # V3 adds drummer/kit_name vocab fields.
                has_new = "kit_category_to_id" in obj and int(obj.get("version", 0) or 0) >= 4
                if has_new:
                    return MidigrooveMetaVocab.from_json(obj)
    vocab = build_midigroove_vocab(train_csv=cfg.train_csv, val_csv=cfg.val_csv, use_style=bool(cfg.use_style))
    if cfg.cache_dir is not None:
        try:
            Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)
            vocab_path = Path(cfg.cache_dir) / "midigroove_vocab.json"
            vocab_path.write_text(json.dumps(vocab.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass
    return vocab


def build_datasets(cfg: TrainCfg) -> tuple[MidigrooveDrumgridMetaCodesDataset, MidigrooveDrumgridMetaCodesDataset, MidigrooveMetaVocab]:
    from data.midigroove_encodec_dataset import MidigrooveDrumgridMetaCodesDataset

    vocab = _load_or_build_vocab(cfg)
    encoder, encoder_tag = build_encoder(cfg)
    train_ds = MidigrooveDrumgridMetaCodesDataset(
        csv_path=cfg.train_csv,
        dataset_root=cfg.dataset_root,
        split=cfg.train_split,
        vocab=vocab,
        encoder=encoder,
        encoder_tag=encoder_tag,
        cache_dir=cfg.cache_dir,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        beats_per_chunk=cfg.beats_per_chunk,
        hop_beats=cfg.hop_beats,
        require_cache=cfg.require_cache,
        encode_if_missing=cfg.encode_if_missing,
        beat_type_only=cfg.beat_type_only,
        kit_category_only=cfg.kit_category_only,
        kit_category_exclude=cfg.kit_category_exclude,
        kit_category_top_n=cfg.kit_category_top_n,
        target_num_clips=cfg.target_train_clips,
        stratify=cfg.stratify_clips,
        stratify_key_kind=str(cfg.stratify_key_kind or "drummer_style_kit_category"),
        stratify_mode=cfg.stratify_mode,
        stratify_seed=cfg.stratify_seed,
        oversample_factor=cfg.oversample_factor,
        include_vel=cfg.include_vel,
        include_sustain=cfg.include_sustain,
        include_hh_cc4=cfg.include_hh_cc4,
    )
    val_ds = MidigrooveDrumgridMetaCodesDataset(
        csv_path=cfg.val_csv,
        dataset_root=cfg.dataset_root,
        split=cfg.val_split,
        vocab=vocab,
        encoder=encoder,
        encoder_tag=encoder_tag,
        cache_dir=cfg.cache_dir,
        window_seconds=cfg.window_seconds,
        hop_seconds=cfg.hop_seconds,
        beats_per_chunk=cfg.beats_per_chunk,
        hop_beats=cfg.hop_beats,
        require_cache=cfg.require_cache,
        encode_if_missing=cfg.encode_if_missing,
        beat_type_only=cfg.beat_type_only,
        kit_category_only=cfg.kit_category_only,
        kit_category_exclude=cfg.kit_category_exclude,
        kit_category_top_n=cfg.kit_category_top_n,
        target_num_clips=cfg.target_val_clips,
        stratify=cfg.stratify_clips,
        stratify_key_kind=str(cfg.stratify_key_kind or "drummer_style_kit_category"),
        stratify_mode=cfg.stratify_mode,
        stratify_seed=cfg.stratify_seed,
        oversample_factor=cfg.oversample_factor,
        include_vel=cfg.include_vel,
        include_sustain=cfg.include_sustain,
        include_hh_cc4=cfg.include_hh_cc4,
    )
    return train_ds, val_ds, vocab


def build_loaders(cfg: TrainCfg) -> tuple[DataLoader, DataLoader, MidigrooveMetaVocab]:
    from data.midigroove_encodec_dataset import MidigrooveDrumgridMetaCodesDataset

    _require_torch()
    if DataLoader is None:  # pragma: no cover
        raise RuntimeError("torch DataLoader is unavailable")
    train_ds, val_ds, vocab = build_datasets(cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=MidigrooveDrumgridMetaCodesDataset.collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=MidigrooveDrumgridMetaCodesDataset.collate_fn,
        drop_last=False,
    )
    return train_loader, val_loader, vocab


def eval_loss(model: nn.Module, val_loader: DataLoader, device: torch.device, *, max_batches: int = 50) -> float:
    _require_torch()
    if F is None:  # pragma: no cover
        raise RuntimeError("torch.nn.functional is unavailable")
    model.eval()
    losses: list[float] = []
    with torch.no_grad():  # type: ignore[union-attr]
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            drum_grid = batch["drum_grid"].to(device)
            beat_pos = batch["beat_pos"].to(device)
            bpm = batch["bpm"].to(device)
            style_id = batch["style_id"].to(device)
            beat_type_id = batch["beat_type_id"].to(device)
            kit_category_id = batch["kit_category_id"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            tgt = batch["tgt_codes"].to(device)

            logits = model(
                drum_grid=drum_grid,
                beat_pos=beat_pos,
                bpm=bpm,
                style_id=style_id,
                beat_type_id=beat_type_id,
                kit_category_id=kit_category_id,
                valid_mask=valid_mask,
            )
            loss = F.cross_entropy(  # type: ignore[union-attr]
                logits.view(-1, VOCAB_SIZE),
                tgt.view(-1),
                ignore_index=PAD_ID,
            )
            v = float(loss.item())
            if not isfinite(v):
                model.train()
                return float("inf")
            losses.append(v)
    model.train()
    return float(sum(losses) / max(1, len(losses)))


class NonFiniteLossError(RuntimeError):
    pass


def _has_nonfinite_params(model: nn.Module) -> bool:
    _require_torch()
    for p in model.parameters():
        if p is None:
            continue
        if not torch.isfinite(p).all():
            return True
    return False


def train(cfg: TrainCfg, *, on_eval: Optional[Callable[[int, float, float], None]] = None) -> float:
    _require_torch()
    _require_tqdm()
    device = torch.device(cfg.device)
    train_loader, val_loader, vocab = build_loaders(cfg)

    it = iter(train_loader)
    first = next(it)
    C = int(first["tgt_codes"].shape[1])
    Dg = int(first["drum_grid"].shape[1])

    model = DrumGridRhythmMetaToCodesModel(
        num_codebooks=C,
        drum_grid_channels=Dg,
        style_vocab_size=vocab.style_vocab_size,
        beat_type_vocab_size=vocab.beat_type_vocab_size,
        kit_category_vocab_size=vocab.kit_vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_frames=cfg.max_frames,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=float(cfg.weight_decay))

    batches = itertools.chain([first], it)
    pbar = tqdm(total=int(cfg.steps), desc="train", dynamic_ncols=True)
    step = 0
    last_val: Optional[float] = None
    best_val: float = float("inf")
    best_step: int = -1
    metrics_path = cfg.metrics_out or (Path(str(cfg.save_path) + ".metrics.csv") if cfg.save_path else _default_metrics_path(cfg))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_f = metrics_path.open("w", encoding="utf-8", newline="")
    metrics_w = csv.DictWriter(
        metrics_f,
        fieldnames=[
            "epoch",
            "steps_end",
            "train_loss_mean",
            "val_loss",
            "best_val",
            "best_step",
            "lr",
            "elapsed_sec",
        ],
    )
    metrics_w.writeheader()
    t_start = time.time()
    epoch = 0
    epoch_train_losses: list[float] = []

    try:
        while step < cfg.steps:
            for batch in batches:
                drum_grid = batch["drum_grid"].to(device)
                beat_pos = batch["beat_pos"].to(device)
                bpm = batch["bpm"].to(device)
                style_id = batch["style_id"].to(device)
                beat_type_id = batch["beat_type_id"].to(device)
                kit_category_id = batch["kit_category_id"].to(device)
                valid_mask = batch["valid_mask"].to(device)
                tgt = batch["tgt_codes"].to(device)

                logits = model(
                    drum_grid=drum_grid,
                    beat_pos=beat_pos,
                    bpm=bpm,
                    style_id=style_id,
                    beat_type_id=beat_type_id,
                    kit_category_id=kit_category_id,
                    valid_mask=valid_mask,
                )
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), tgt.view(-1), ignore_index=PAD_ID)
                if not torch.isfinite(loss):
                    if _has_nonfinite_params(model):
                        raise NonFiniteLossError("non-finite model parameters detected (training diverged)")
                    if str(cfg.nonfinite_policy).strip().lower() == "skip":
                        opt.zero_grad(set_to_none=True)
                        step += 1
                        pbar.update(1)
                        pbar.set_postfix(train_loss="nan/inf (skipped)")
                        if step >= cfg.steps:
                            break
                        continue
                    raise NonFiniteLossError("non-finite loss encountered (try lowering --lr or using --nonfinite-policy skip)")

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(cfg.grad_clip) and float(cfg.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
                opt.step()

                epoch_train_losses.append(float(loss.detach().cpu().item()))

                if cfg.log_every > 0 and step % cfg.log_every == 0:
                    last_val = eval_loss(model, val_loader, device, max_batches=int(cfg.eval_max_batches))
                    if on_eval is not None:
                        on_eval(int(step), float(loss.item()), float(last_val))
                    if cfg.save_path is not None and cfg.save_best_only and last_val < best_val:
                        best_val = float(last_val)
                        best_step = int(step)
                        _atomic_torch_save(
                            cfg.save_path,
                            {
                                "model": model.state_dict(),
                                "cfg": _jsonable(cfg.__dict__),
                                "num_codebooks": C,
                                "drum_grid_channels": Dg,
                                "vocab": vocab.to_json(),
                                "best_val": best_val,
                                "best_step": best_step,
                            },
                        )

                step += 1
                pbar.update(1)
                if last_val is None:
                    pbar.set_postfix(train_loss=f"{loss.item():.4f}")
                else:
                    pbar.set_postfix(train_loss=f"{loss.item():.4f}", val_loss=f"{last_val:.4f}")
                if step >= cfg.steps:
                    break

            # End of epoch: compute val once, write metrics row.
            val_epoch = eval_loss(model, val_loader, device, max_batches=int(cfg.eval_max_batches))
            if cfg.save_path is not None and cfg.save_best_only and float(val_epoch) < float(best_val):
                best_val = float(val_epoch)
                best_step = int(step)
                _atomic_torch_save(
                    cfg.save_path,
                    {
                        "model": model.state_dict(),
                        "cfg": _jsonable(cfg.__dict__),
                        "num_codebooks": C,
                        "drum_grid_channels": Dg,
                        "vocab": vocab.to_json(),
                        "best_val": best_val,
                        "best_step": best_step,
                    },
                )

            lr = float(opt.param_groups[0].get("lr", 0.0)) if opt.param_groups else 0.0
            train_mean = float(sum(epoch_train_losses) / max(1, len(epoch_train_losses)))
            metrics_w.writerow(
                {
                    "epoch": int(epoch),
                    "steps_end": int(step),
                    "train_loss_mean": train_mean,
                    "val_loss": float(val_epoch),
                    "best_val": float(best_val),
                    "best_step": int(best_step),
                    "lr": lr,
                    "elapsed_sec": float(time.time() - t_start),
                }
            )
            metrics_f.flush()
            epoch += 1
            epoch_train_losses = []

            if step >= cfg.steps:
                break

            batches = iter(train_loader)
    finally:
        try:
            metrics_f.close()
        except Exception:
            pass
        try:
            print(f"wrote metrics: {metrics_path}")
        except Exception:
            pass
    pbar.close()
    if cfg.save_path is not None and cfg.save_best_only and best_step >= 0:
        print(f"best checkpoint: step={best_step} val_loss={best_val:.4f} -> {cfg.save_path}")
    if best_step >= 0:
        return float(best_val)
    return float(last_val) if last_val is not None else float("inf")


def _jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _write_run_config(*, path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "models.midigroove_drumgrid_rhythm_meta_to_encodec_codes",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "args": _jsonable(vars(args)),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cfg_from_args(args: argparse.Namespace) -> TrainCfg:
    return TrainCfg(
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        train_split=str(args.train_split),
        val_split=str(args.val_split),
        dataset_root=Path(args.dataset_root),
        device=str(args.device),
        encode_device=str(args.encode_device),
        encoder_model=str(args.encoder_model),
        xcodec_bandwidth=float(args.xcodec_bandwidth),
        cache_dir=None if bool(args.no_cache) else Path(args.cache_dir),
        require_cache=bool(args.require_cache),
        encode_if_missing=not bool(args.require_cache),
        window_seconds=float(args.window_seconds),
        hop_seconds=float(args.hop_seconds),
        beats_per_chunk=int(args.beats_per_chunk) if getattr(args, "beats_per_chunk", None) else None,
        hop_beats=int(args.hop_beats) if getattr(args, "hop_beats", None) else None,
        use_style=bool(getattr(args, "use_style", False)),
        beat_type_only=str(args.beat_type_only) if getattr(args, "beat_type_only", None) else None,
        kit_category_only=str(getattr(args, "kit_category_only", None)) if getattr(args, "kit_category_only", None) else None,
        kit_category_exclude=str(getattr(args, "kit_category_exclude", None)) if getattr(args, "kit_category_exclude", None) else None,
        kit_category_top_n=int(getattr(args, "kit_category_top_n", 0) or 0) if getattr(args, "kit_category_top_n", None) else None,
        target_train_clips=int(args.target_train_clips) if getattr(args, "target_train_clips", None) else None,
        target_val_clips=int(args.target_val_clips) if getattr(args, "target_val_clips", None) else None,
        stratify_clips=bool(getattr(args, "stratify_clips", False)),
        stratify_key_kind=(
            str(getattr(args, "stratify_key_kind", "") or "").strip()
            or ("drummer_style_kit_category" if bool(getattr(args, "use_style", False)) else "drummer_kit_name")
        ),
        stratify_mode=str(getattr(args, "stratify_mode", "proportional")),
        stratify_seed=int(getattr(args, "seed", 0)),
        oversample_factor=float(getattr(args, "oversample_factor", 1.25)),
        include_vel=bool(getattr(args, "include_vel", True)),
        include_sustain=bool(getattr(args, "include_sustain", True)),
        include_hh_cc4=bool(getattr(args, "include_hh_cc4", True)),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        lr=float(args.lr),
        weight_decay=float(getattr(args, "weight_decay", 0.01)),
        grad_clip=float(getattr(args, "grad_clip", 1.0)),
        steps=int(args.steps),
        log_every=int(args.log_every),
        eval_max_batches=int(getattr(args, "eval_max_batches", 50)),
        nonfinite_policy=str(getattr(args, "nonfinite_policy", "halt")),
        save_path=Path(args.save_path) if args.save_path else None,
        d_model=int(args.d_model),
        n_layers=int(args.n_layers),
        n_heads=int(args.n_heads),
        max_frames=int(args.max_frames),
        save_best_only=bool(args.save_best_only),
        metrics_out=Path(args.metrics_out) if getattr(args, "metrics_out", None) else None,
    )


def _default_metrics_path(cfg: TrainCfg) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = str(cfg.encoder_model or "encodec").strip().lower()
    return out_dir / f"drumgrid_{kind}_{ts}.csv"



def main_train(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(
        prog="python -m midigroove_poc drumgrid train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / cache
    default_root = os.environ.get(
        "EGMD_ROOT",
        "data/e-gmd-v1.0.0/e-gmd-v1.0.0",
    )
    default_csv = os.environ.get(
        "EGMD_CSV",
        "data/e-gmd-v1.0.0/e-gmd-v1.0.0/e-gmd-v1.0.0.csv",
    )
    ap.add_argument("--train-csv", type=str, default=default_csv)
    ap.add_argument("--val-csv", type=str, default=default_csv)
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="validation")
    ap.add_argument("--test-split", type=str, default="test", help="Dataset split name for test (used only for precaching).")
    ap.add_argument(
        "--dataset-root",
        type=str,
        default=default_root,
        help="Root directory containing dataset files referenced by the CSV. Override with $EGMD_ROOT.",
    )
    ap.add_argument("--cache-dir", type=str, default="cache/encodec_big_acoustic_pop")
    ap.add_argument("--no-cache", action="store_true", help="Disable on-disk caching.")
    ap.add_argument("--require-cache", action="store_true", help="Require cache files to exist; do not encode on-the-fly.")
    ap.add_argument("--precache", action="store_true", help="Build caches before training.")
    ap.add_argument("--precache-only", action="store_true", help="Build caches then exit.")
    ap.add_argument("--no-precache-test", action="store_true", help="Do not precache the test split.")
    ap.add_argument("--window-seconds", type=float, default=2.0)
    ap.add_argument("--hop-seconds", type=float, default=2.0)
    ap.add_argument(
        "--beats-per-chunk",
        type=int,
        default=None,
        help="If set, train/cache fixed-N-beat chunks (window seconds derived from BPM).",
    )
    ap.add_argument("--hop-beats", type=int, default=None, help="Hop in beats for beat-chunk mode (defaults to --beats-per-chunk).")
    ap.add_argument("--use-style", action="store_true", help="Condition on CSV 'style' (otherwise ignore style).")

    # Dataset filtering / sampling for cache building
    ap.add_argument("--beat-type-only", type=str, default=None, help="If set, keep only rows with this beat_type (e.g. 'beat').")
    ap.add_argument(
        "--kit-category-only",
        type=str,
        default=None,
        help="Cache filter only: keep only rows whose CSV kit_name matches (e.g. 'Acoustic Kit'). Comma-separated list supported. Not used as model input.",
    )
    ap.add_argument(
        "--kit-category-exclude",
        type=str,
        default=None,
        help="Cache filter only: exclude rows whose CSV kit_name matches. Comma-separated list supported.",
    )
    ap.add_argument(
        "--kit-category-top-n",
        type=int,
        default=None,
        help="Cache filter only: keep only rows whose kit_name is in the top-N most frequent kit_names (computed within the split after beat-type filtering). Mutually exclusive with --kit-category-only.",
    )
    ap.add_argument("--target-train-clips", type=int, default=None, help="If set, cache/train on ~this many train windows.")
    ap.add_argument("--target-val-clips", type=int, default=None, help="If set, cache/train on ~this many val windows.")
    ap.add_argument("--target-test-clips", type=int, default=None, help="If set, cache on ~this many test windows (precache only).")
    ap.add_argument("--stratify-clips", action="store_true", help="Stratify clip sampling by a chosen key (see --stratify-key-kind).")
    ap.add_argument(
        "--stratify-key-kind",
        type=str,
        default=None,
        choices=["drummer_style_kit_category", "drummer_kit_name", "drummer_kit_category"],
        help="Key used for stratified clip sampling when --stratify-clips is set (default: drummer_style_kit_category if --use-style else drummer_kit_name).",
    )
    ap.add_argument("--stratify-mode", type=str, default="proportional", choices=["proportional", "uniform_keys"])
    ap.add_argument("--oversample-factor", type=float, default=1.25)
    ap.add_argument("--seed", type=int, default=0)

    # Devices / codec encoders
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--encode-device", type=str, default="cuda")
    ap.add_argument("--encoder-model", type=str, default="encodec", choices=["encodec", "dac", "xcodec"])
    ap.add_argument("--xcodec-bandwidth", type=float, default=2.0)

    # Optional conditioning inputs
    ap.add_argument("--no-vel", action="store_true", help="Disable velocity-at-onset grid conditioning.")
    ap.add_argument("--no-sustain", action="store_true", help="Disable sustain-proxy grid conditioning.")
    ap.add_argument("--no-hh-cc4", action="store_true", help="Disable hi-hat CC#4 openness lane conditioning.")

    # Training
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-max-batches", type=int, default=50)
    ap.add_argument("--nonfinite-policy", type=str, default="halt", choices=["halt", "skip"])
    ap.add_argument("--save", type=str, default=None, help="Checkpoint path (.pt). Alias of --save-path.")
    ap.add_argument("--save-path", type=str, default=None, help="Checkpoint path (.pt).")
    ap.add_argument("--save-best-only", action="store_true", default=True)
    ap.add_argument("--metrics-out", type=str, default=None, help="Write per-epoch train/val metrics CSV to this path.")

    # Model shape
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=4096)

    args = ap.parse_args(argv)
    args.include_vel = not bool(args.no_vel)
    args.include_sustain = not bool(args.no_sustain)
    args.include_hh_cc4 = not bool(args.no_hh_cc4)

    # Validate dataset paths early to avoid confusing downstream errors.
    default_root = os.environ.get(
        "EGMD_ROOT",
        "data/e-gmd-v1.0.0/e-gmd-v1.0.0",
    )
    default_csv = os.environ.get(
        "EGMD_CSV",
        "data/e-gmd-v1.0.0/e-gmd-v1.0.0/e-gmd-v1.0.0.csv",
    )

    train_csv_raw = str(getattr(args, "train_csv", "") or "").strip()
    val_csv_raw = str(getattr(args, "val_csv", "") or "").strip()
    dataset_root_raw = str(getattr(args, "dataset_root", "") or "").strip()
    train_csv = Path(default_csv if train_csv_raw == "" else train_csv_raw).expanduser()
    val_csv = Path(default_csv if val_csv_raw == "" else val_csv_raw).expanduser()
    dataset_root = Path(default_root if dataset_root_raw == "" else dataset_root_raw).expanduser()
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Missing --train-csv: {train_csv}\n"
            "Set --train-csv explicitly, or set $EGMD_CSV to your CSV path.\n"
            "If you passed --train-csv \"$EGMD_CSV\", ensure EGMD_CSV is set (not empty)."
        )
    if not val_csv.is_file():
        raise FileNotFoundError(
            f"Missing --val-csv: {val_csv}\n"
            "Set --val-csv explicitly, or set $EGMD_CSV to your CSV path.\n"
            "If you passed --val-csv \"$EGMD_CSV\", ensure EGMD_CSV is set (not empty)."
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Missing --dataset-root: {dataset_root}\n"
            "Set --dataset-root explicitly, or set $EGMD_ROOT to your dataset root directory.\n"
            "If you passed --dataset-root \"$EGMD_ROOT\", ensure EGMD_ROOT is set (not empty)."
        )

    # CUDA+fork avoidance: encoding on CUDA should not run in DataLoader workers.
    if str(args.encode_device).startswith("cuda") and int(args.num_workers) > 0:
        if bool(args.require_cache):
            # Cache-only training: safe to keep workers; encoding should not happen.
            pass
        else:
            print("note: forcing --num-workers 0 (CUDA encoding is not compatible with forked DataLoader workers).")
            args.num_workers = 0

    # Prefer --save if provided.
    save_path = args.save_path
    if args.save is not None:
        save_path = args.save
    args.save_path = save_path

    cfg = _cfg_from_args(args)
    cfg.d_model = int(args.d_model)
    cfg.n_heads = int(args.n_heads)
    cfg.n_layers = int(args.n_layers)

    if args.precache or args.precache_only:
        if cfg.cache_dir is None:
            raise ValueError("--precache requires --cache-dir (omit --no-cache).")
        precache_cfg = TrainCfg(**cfg.__dict__)
        precache_cfg.require_cache = False
        precache_cfg.encode_if_missing = True
        train_ds, val_ds, _v = build_datasets(precache_cfg)
        print(f"precaching train split into {precache_cfg.cache_dir} ...")
        print(f"precache train: {train_ds.precache(desc='cache-train')}")
        print(f"precaching val split into {precache_cfg.cache_dir} ...")
        print(f"precache val: {val_ds.precache(desc='cache-val')}")
        if not bool(args.no_precache_test):
            from data.midigroove_encodec_dataset import MidigrooveDrumgridMetaCodesDataset

            test_ds = MidigrooveDrumgridMetaCodesDataset(
                csv_path=precache_cfg.train_csv,
                dataset_root=precache_cfg.dataset_root,
                split=str(args.test_split),
                vocab=train_ds.vocab,
                encoder=train_ds.encoder,
                encoder_tag=train_ds.encoder_tag,
                cache_dir=precache_cfg.cache_dir,
                window_seconds=precache_cfg.window_seconds,
                hop_seconds=precache_cfg.hop_seconds,
                beats_per_chunk=precache_cfg.beats_per_chunk,
                hop_beats=precache_cfg.hop_beats,
                require_cache=False,
                encode_if_missing=True,
                beat_type_only=precache_cfg.beat_type_only,
                kit_category_only=precache_cfg.kit_category_only,
                kit_category_exclude=precache_cfg.kit_category_exclude,
                kit_category_top_n=precache_cfg.kit_category_top_n,
                target_num_clips=int(args.target_test_clips) if getattr(args, "target_test_clips", None) else None,
                stratify=precache_cfg.stratify_clips,
                stratify_key_kind=str(precache_cfg.stratify_key_kind or "drummer_style_kit_category"),
                stratify_mode=precache_cfg.stratify_mode,
                stratify_seed=precache_cfg.stratify_seed,
                oversample_factor=precache_cfg.oversample_factor,
                include_vel=precache_cfg.include_vel,
                include_sustain=precache_cfg.include_sustain,
                include_hh_cc4=precache_cfg.include_hh_cc4,
            )
            print(f"precaching test split into {precache_cfg.cache_dir} ...")
            print(f"precache test: {test_ds.precache(desc='cache-test')}")
        if args.precache_only:
            return

    train(cfg)


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: python -m midigroove_poc drumgrid train ...\n\n"
            "subcommands:\n"
            "  train   train a model from an on-disk cache (or build it first)\n\n"
            "help:\n"
            "  python -m midigroove_poc drumgrid train --help\n"
        )
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        return main_train(rest)
    raise SystemExit(f"unknown command {cmd!r} (expected: train)")


if __name__ == "__main__":
    main()
    if cfg.kit_category_only is not None and cfg.kit_category_top_n is not None:
        raise ValueError("Use only one of --kit-category-only or --kit-category-top-n.")
