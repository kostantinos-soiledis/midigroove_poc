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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .runtime import configure_runtime

configure_runtime()


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


def _si_sdr(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = int(min(pred.size, ref.size))
    if n <= 0:
        return float("nan")
    pred = pred[:n]
    ref = ref[:n]
    ref_zm = ref - float(ref.mean())
    pred_zm = pred - float(pred.mean())
    denom = float(np.dot(ref_zm, ref_zm)) + float(eps)
    s_target = (float(np.dot(pred_zm, ref_zm)) / denom) * ref_zm
    e_noise = pred_zm - s_target
    num = float(np.dot(s_target, s_target)) + float(eps)
    den = float(np.dot(e_noise, e_noise)) + float(eps)
    return float(10.0 * math.log10(num / den))


def _snr(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-9) -> float:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    n = int(min(pred.size, ref.size))
    if n <= 0:
        return float("nan")
    pred = pred[:n]
    ref = ref[:n]
    noise = ref - pred
    num = float(np.dot(ref, ref)) + float(eps)
    den = float(np.dot(noise, noise)) + float(eps)
    return float(10.0 * math.log10(num / den))


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


def _stft_metrics(pred: np.ndarray, ref: np.ndarray, *, n_fft: int = 1024, hop: int = 256) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32).reshape(-1)
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    n = int(min(pred.size, ref.size))
    if n <= 0:
        return {"stft_sc": float("nan"), "stft_l1": float("nan"), "lsd": float("nan")}
    pred = pred[:n]
    ref = ref[:n]
    win = np.hanning(int(n_fft)).astype(np.float32)
    mp = _stft_mag(pred, n_fft=n_fft, hop=hop, win=win)
    mr = _stft_mag(ref, n_fft=n_fft, hop=hop, win=win)
    k = int(min(mp.shape[0], mr.shape[0]))
    if k <= 0:
        return {"stft_sc": float("nan"), "stft_l1": float("nan"), "lsd": float("nan")}
    mp = mp[:k]
    mr = mr[:k]
    diff = mp - mr
    sc = float(np.linalg.norm(diff) / max(1e-9, float(np.linalg.norm(mr))))
    l1 = float(np.mean(np.abs(diff)))
    lsd = float(np.mean(np.sqrt(np.mean((np.log(mp + 1e-7) - np.log(mr + 1e-7)) ** 2, axis=1))))
    return {"stft_sc": sc, "stft_l1": l1, "lsd": lsd}


def _mean_std(xs: List[float]) -> Dict[str, float]:
    arr = np.asarray([x for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


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
    cfg = dict(cfg)
    cfg["in_dim"] = int(in_dim)
    cfg["num_codebooks"] = int(num_codebooks)
    model = eg._build_model(num_codebooks=num_codebooks, in_dim=in_dim, cfg=cfg)  # type: ignore[attr-defined]
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.eval()
    enc = str(spec.encoder_model or cfg.get("encoder_model", "encodec") or "encodec").strip().lower()
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
    ap.add_argument("--eval-sr", type=int, default=16000, help="Sample rate for audio metrics and saved wavs.")
    ap.add_argument("--audio-metrics", action="store_true", help="Decode tokens to audio and compute audio metrics.")
    ap.add_argument("--add-oracle", action="store_true", help="Also evaluate the codec reconstruction baseline (decode ground-truth tokens).")
    ap.add_argument("--add-random", action="store_true", help="Also evaluate a random-tokens baseline (very low-quality).")
    ap.add_argument("--save-wavs", type=int, default=0, help="Save up to N items per system as wavs.")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/eval"))

    args = ap.parse_args(list(argv) if argv is not None else None)

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
    for spec in systems:
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
    do_audio = bool(args.audio_metrics)
    add_oracle = bool(args.add_oracle)
    add_random = bool(args.add_random)
    save_wavs = int(args.save_wavs)

    from . import expressivegrid as eg
    from data.codecs import decode_tokens_to_audio

    summary: Dict[str, Any] = {
        "split": split,
        "intersection": bool(args.intersection),
        "n_items": int(len(selected_keys)),
        "eval_sr": int(eval_sr),
        "systems": {},
    }
    rows: List[Dict[str, Any]] = []

    for spec in systems:
        model, cfg, encoder_model = _load_system_model(spec, device=device)

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
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=eg.ExpressiveGridDataset.collate_fn,  # type: ignore[attr-defined]
            drop_last=False,
        )

        token_nll: List[float] = []
        token_acc: List[float] = []
        per_cb_acc: List[List[float]] = []
        infer_ms: List[float] = []
        audio_sisdr: List[float] = []
        audio_snr: List[float] = []
        stft_sc: List[float] = []
        stft_l1: List[float] = []
        lsd: List[float] = []

        oracle_sisdr: List[float] = []
        oracle_snr: List[float] = []
        oracle_stft_sc: List[float] = []
        oracle_stft_l1: List[float] = []
        oracle_lsd: List[float] = []

        random_sisdr: List[float] = []
        random_snr: List[float] = []
        random_stft_sc: List[float] = []
        random_stft_l1: List[float] = []
        random_lsd: List[float] = []

        saved = 0
        for batch_i, batch in enumerate(loader):
            t0 = time.perf_counter()
            logits = model(
                grid=batch["grid"].to(device),
                beat_pos=batch["beat_pos"].to(device),
                bpm=batch["bpm"].to(device),
                drummer_id=batch["drummer_id"].to(device),
                valid_mask=batch["valid_mask"].to(device),
            )
            torch.cuda.synchronize() if device.startswith("cuda") else None  # type: ignore[attr-defined]
            infer_ms.append(1000.0 * (time.perf_counter() - t0))

            tgt = batch["tgt_codes"].to(device)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, eg.VOCAB_SIZE),  # type: ignore[attr-defined]
                tgt.view(-1),
                ignore_index=eg.PAD_ID,  # type: ignore[attr-defined]
                reduction="mean",
            )
            token_nll.append(float(loss.detach().cpu().item()))

            pred = torch.argmax(logits, dim=-1)  # [B,C,T]
            mask = tgt.ne(int(eg.PAD_ID))  # type: ignore[attr-defined]
            correct = (pred == tgt) & mask
            denom = float(mask.sum().detach().cpu().item())
            token_acc.append(float(correct.sum().detach().cpu().item()) / max(1.0, denom))

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
            row: Dict[str, Any] = {
                "system": spec.name,
                "item": key_short,
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
                    row.update({"sisdr": float("nan"), "snr": float("nan"), "stft_sc": float("nan"), "stft_l1": float("nan"), "lsd": float("nan")})
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
                    s = _si_sdr(y_pred, y_ref)
                    n0 = _snr(y_pred, y_ref)
                    m = _stft_metrics(y_pred, y_ref)
                    audio_sisdr.append(float(s))
                    audio_snr.append(float(n0))
                    stft_sc.append(float(m["stft_sc"]))
                    stft_l1.append(float(m["stft_l1"]))
                    lsd.append(float(m["lsd"]))
                    row.update({"sisdr": float(s), "snr": float(n0), **m})

                    if save_wavs > 0 and saved < save_wavs:
                        sys_dir = out_dir / "wavs" / spec.name
                        _write_wav(sys_dir / f"{key_short}_pred.wav", y_pred, int(eval_sr))
                        _write_wav(sys_dir / f"{key_short}_ref.wav", y_ref, int(eval_sr))
                        saved += 1

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
                        so = _si_sdr(y_or, y_r2)
                        sn = _snr(y_or, y_r2)
                        mo = _stft_metrics(y_or, y_r2)
                        oracle_sisdr.append(float(so))
                        oracle_snr.append(float(sn))
                        oracle_stft_sc.append(float(mo["stft_sc"]))
                        oracle_stft_l1.append(float(mo["stft_l1"]))
                        oracle_lsd.append(float(mo["lsd"]))

                    if add_random:
                        # Uniform random tokens in [0, VOCAB_SIZE-2] (avoid PAD by construction).
                        gen = torch.Generator(device="cpu").manual_seed(int(args.seed) + int(batch_i))
                        rand = torch.randint(
                            low=0,
                            high=int(eg.PAD_ID),  # type: ignore[attr-defined]
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
                        sr0 = _si_sdr(y_rand, y_r3)
                        sn0 = _snr(y_rand, y_r3)
                        mr0 = _stft_metrics(y_rand, y_r3)
                        random_sisdr.append(float(sr0))
                        random_snr.append(float(sn0))
                        random_stft_sc.append(float(mr0["stft_sc"]))
                        random_stft_l1.append(float(mr0["stft_l1"]))
                        random_lsd.append(float(mr0["lsd"]))

            rows.append(row)

        # Aggregate.
        cb_means: Dict[str, float] = {}
        if per_cb_acc:
            Cmax = max(len(x) for x in per_cb_acc)
            for c in range(Cmax):
                vals = [x[c] for x in per_cb_acc if c < len(x)]
                cb_means[f"token_acc_cb{c}"] = float(np.mean(vals)) if vals else float("nan")

        sys_sum: Dict[str, Any] = {
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
                    "sisdr": _mean_std(audio_sisdr),
                    "snr": _mean_std(audio_snr),
                    "stft_sc": _mean_std(stft_sc),
                    "stft_l1": _mean_std(stft_l1),
                    "lsd": _mean_std(lsd),
                }
            )
        summary["systems"][spec.name] = sys_sum

        if do_audio and add_oracle:
            summary["systems"][f"{spec.name}_oracle"] = {
                "encoder_model": str(encoder_model),
                "n_items": int(len(oracle_sisdr)),
                "sisdr": _mean_std(oracle_sisdr),
                "snr": _mean_std(oracle_snr),
                "stft_sc": _mean_std(oracle_stft_sc),
                "stft_l1": _mean_std(oracle_stft_l1),
                "lsd": _mean_std(oracle_lsd),
            }

        if do_audio and add_random:
            summary["systems"][f"{spec.name}_random"] = {
                "encoder_model": str(encoder_model),
                "n_items": int(len(random_sisdr)),
                "sisdr": _mean_std(random_sisdr),
                "snr": _mean_std(random_snr),
                "stft_sc": _mean_std(random_stft_sc),
                "stft_l1": _mean_std(random_stft_l1),
                "lsd": _mean_std(random_lsd),
            }

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
