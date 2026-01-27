"""Midigroove / E-GMD expressive drum-grid -> codec-codes dataset.

Inputs (conditioning)
---------------------
- Expressive pad+articulation drum grid from MIDI, aligned to codec frames:
  hit, velocity-onset, sustain proxy, plus optional hi-hat CC#4 lane.
- Rhythm/meta: BPM, beat_type, style, kit_category.

Targets (supervision)
---------------------
- Discrete codec codes for the aligned WAV segment.

Caching
-------
Builds an on-disk cache of windows. Two modes:
  - fixed-seconds: `window_seconds` / `hop_seconds` (default; historically 2s)
  - fixed-beats: `beats_per_chunk` / `hop_beats` (window/hop seconds derived from BPM)

Cache entries are skipped when:
  - the corresponding WAV or MIDI is missing
  - the audio window is incomplete (<window seconds samples)
  - the MIDI-derived hit grid is empty for the window
  - the kit is a user/custom kit (category == "User")
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

try:  # optional dependency for MIDI parsing
    import pretty_midi  # type: ignore
except Exception:  # pragma: no cover
    pretty_midi = None  # type: ignore

try:  # optional dependency for quick audio duration
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torchaudio  # type: ignore
except Exception:  # pragma: no cover
    torchaudio = None  # type: ignore


PAD_ID = 2048


def load_audio(path: Path) -> Tuple["torch.Tensor", int]:
    """Load an audio file as a float tensor using soundfile or torchaudio."""

    if sf is not None:
        audio, sr = sf.read(path, always_2d=False)
        waveform = torch.from_numpy(np.asarray(audio)).float()
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=1)
        return waveform, int(sr)

    if torchaudio is None:
        raise RuntimeError("Neither soundfile nor torchaudio is available for loading audio.")

    waveform, sr = torchaudio.load(str(path))
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    return waveform, int(sr)


# -----------------------
# Stratified clip sampling
# -----------------------
def _stratified_sample_indices(
    chunks: List["ChunkSpec"],
    *,
    n: int,
    seed: int,
    key_kind: str = "drummer_style_kit_category",
    mode: str = "proportional",
) -> List[int]:
    """Return indices of a stratified sample of ChunkSpecs (no replacement).

    Strategy: proportional allocation across strata with a small min-per-stratum
    guarantee when possible, then fill remainder from the global pool.
    """
    n = int(n)
    if n <= 0 or not chunks:
        return []
    rng = np.random.default_rng(int(seed))

    def _key(spec: "ChunkSpec"):
        if key_kind == "drummer_style_kit_category":
            return (spec.drummer, spec.style, spec.kit_category)
        if key_kind == "drummer_kit_name":
            return (spec.drummer, spec.kit_name)
        if key_kind == "drummer_kit_category":
            return (spec.drummer, spec.kit_category)
        raise ValueError(f"unknown key_kind: {key_kind}")

    groups: Dict[object, List[int]] = {}
    for i, spec in enumerate(chunks):
        k = _key(spec)
        groups.setdefault(k, []).append(i)

    # Shuffle each group's indices deterministically.
    for k, idxs in groups.items():
        idxs_arr = np.asarray(idxs, dtype=np.int64)
        rng.shuffle(idxs_arr)
        groups[k] = idxs_arr.tolist()

    total = len(chunks)
    n = min(n, total)

    mode = str(mode or "proportional").strip().lower()
    if mode not in {"proportional", "uniform_keys"}:
        raise ValueError(f"unsupported stratify mode: {mode} (expected: proportional|uniform_keys)")

    # Group metadata.
    keys = list(groups.keys())
    sizes = np.asarray([len(groups[k]) for k in keys], dtype=np.int64)
    if mode == "proportional":
        weights = sizes.astype(np.float64) / float(max(1, total))
        raw = weights * float(n)
        base = np.floor(raw).astype(np.int64)
        frac = raw - base.astype(np.float64)

        # Try to give at least 1 to strata when feasible.
        if n >= len(keys):
            base = np.maximum(base, 1)
        # Clip to stratum size.
        base = np.minimum(base, sizes)

        allocated = int(base.sum())
        # If over-allocated due to min-per-stratum, reduce from largest allocations.
        if allocated > n:
            over = allocated - n
            order = np.argsort(-base)  # reduce from large strata
            j = 0
            while over > 0 and j < len(order):
                i = int(order[j])
                if base[i] > 0:
                    base[i] -= 1
                    over -= 1
                else:
                    j += 1
            allocated = int(base.sum())

        # Distribute remaining by fractional parts and available capacity.
        remaining = int(n - allocated)
        if remaining > 0:
            order = np.argsort(-frac)  # largest remainder first
            for idx in order:
                if remaining <= 0:
                    break
                idx = int(idx)
                cap = int(sizes[idx] - base[idx])
                if cap <= 0:
                    continue
                take = min(cap, remaining)
                base[idx] += take
                remaining -= take
    else:
        # Uniform across keys: allocate as evenly as possible across strata,
        # respecting per-stratum capacity; then redistribute any leftover.
        K = len(keys)
        base = np.zeros(K, dtype=np.int64)
        if K > 0:
            q = int(n // K)
            r = int(n % K)
            base[:] = q
            # Randomize which keys get the remainder.
            order = np.arange(K, dtype=np.int64)
            rng.shuffle(order)
            base[order[:r]] += 1
            base = np.minimum(base, sizes)

        remaining = int(n - int(base.sum()))
        if remaining > 0:
            # Round-robin fill from keys with remaining capacity.
            order = np.arange(K, dtype=np.int64)
            rng.shuffle(order)
            idx_ptr = 0
            guard = 0
            while remaining > 0 and guard < K * 4:
                k = int(order[idx_ptr % K])
                cap = int(sizes[k] - base[k])
                if cap > 0:
                    base[k] += 1
                    remaining -= 1
                idx_ptr += 1
                guard += 1

    selected: List[int] = []
    for k, take in zip(keys, base.tolist()):
        if take <= 0:
            continue
        selected.extend(groups[k][: int(take)])

    # Fill any gap from the leftover pool.
    if len(selected) < n:
        chosen = set(selected)
        remaining_pool = [i for i in range(total) if i not in chosen]
        if remaining_pool:
            remaining_pool = np.asarray(remaining_pool, dtype=np.int64)
            rng.shuffle(remaining_pool)
            need = int(n - len(selected))
            selected.extend(remaining_pool[:need].tolist())

    selected_arr = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_arr)
    return selected_arr.tolist()


KIT_TO_CATEGORY: Dict[str, str] = {}
USER_KIT_CATEGORY = "User"
UNKNOWN_KIT_CATEGORY = "Other/Unknown"

# Expressive "pad + articulation" grid channels (GM-first)
CHANNELS: List[str] = [
    "kick",
    "snare_head",
    "snare_xstick",
    "clap",
    "hh_closed",
    "hh_open",
    "hh_pedal",
    "tom_high",
    "tom_mid",
    "tom_low",
    "crash1",
    "crash2",
    "china",  # GM 52
    "splash",  # GM 55
    "ride_bow",
    "ride_bell",
    "perc_aux",
]
CH2I: Dict[str, int] = {c: i for i, c in enumerate(CHANNELS)}

# GM mapping fixes:
# - 52: Chinese Cymbal -> "china"
# - 55: Splash Cymbal  -> "splash"
# - 59: Ride Cymbal 2  -> ride-like, NOT bell -> map to "ride_bow"
#
# Groove MIDI Dataset note:
# Performances were recorded using a Roland TD-11 kit which uses some pitches
# that differ from GM. We canonicalize a subset of Roland/alternate pitches to
# a simplified "paper" mapping used by the original Groove dataset paper.
PAPER_PITCH_CANONICAL: Dict[int, int] = {
    # Snare variants -> snare (38)
    40: 38,  # snare rim -> snare
    37: 38,  # x-stick -> snare
    # Toms -> high/mid/low tom groups
    48: 50,  # tom1 -> high tom
    50: 50,  # tom1 rim -> high tom
    45: 47,  # tom2 -> low-mid tom
    47: 47,  # tom2 rim -> low-mid tom
    43: 43,  # floor tom -> floor tom
    58: 43,  # tom3 rim (vibraslap) -> floor tom
    # Hi-hats
    26: 46,  # open edge -> open
    22: 42,  # closed edge -> closed
    44: 42,  # pedal -> closed
    # Crashes (incl. china/splash) -> crash (49)
    55: 49,  # splash -> crash
    57: 49,  # crash2 -> crash
    52: 49,  # china -> crash
    # Ride variants -> ride (51)
    59: 51,  # ride edge -> ride
    53: 51,  # ride bell -> ride
}
GM_MAP: Dict[int, str] = {
    35: "kick",
    36: "kick",
    38: "snare_head",
    40: "snare_head",
    37: "snare_xstick",
    39: "clap",
    42: "hh_closed",
    46: "hh_open",
    44: "hh_pedal",
    50: "tom_high",
    48: "tom_mid",
    47: "tom_mid",
    45: "tom_low",
    43: "tom_low",
    41: "tom_low",
    49: "crash1",
    57: "crash2",
    52: "china",
    55: "splash",
    51: "ride_bow",
    53: "ride_bell",
    59: "ride_bow",
    56: "perc_aux",
    54: "perc_aux",
    58: "perc_aux",
}

SUSTAIN_CHANNELS = {"hh_open", "crash1", "crash2", "china", "splash", "ride_bow", "ride_bell"}
DECAY_SEC: Dict[str, float] = {
    "hh_open": 0.18,
    "crash1": 0.90,
    "crash2": 0.90,
    "china": 1.00,
    "splash": 0.45,
    "ride_bow": 0.85,
    "ride_bell": 0.65,
}


@dataclass(frozen=True)
class ChunkSpec:
    audio_path: Path
    midi_path: Path
    sr: int
    drummer: str
    bpm: float
    beat_type: str
    style: str
    kit_name: str
    kit_category: str
    start_idx: int
    start_sec: float
    window_seconds: float
    hop_seconds: float


@dataclass
class MidigrooveMetaVocab:
    """String->id vocab for categorical conditioning fields.

    ID 0 is reserved for UNK.
    """

    style_to_id: Dict[str, int]
    beat_type_to_id: Dict[str, int]
    kit_category_to_id: Dict[str, int]
    drummer_to_id: Dict[str, int]
    kit_name_to_id: Dict[str, int]

    @property
    def style_vocab_size(self) -> int:
        return int(max(self.style_to_id.values(), default=0) + 1)

    @property
    def beat_type_vocab_size(self) -> int:
        return int(max(self.beat_type_to_id.values(), default=0) + 1)

    @property
    def kit_vocab_size(self) -> int:
        return int(max(self.kit_category_to_id.values(), default=0) + 1)

    @property
    def drummer_vocab_size(self) -> int:
        return int(max(self.drummer_to_id.values(), default=0) + 1)

    @property
    def kit_name_vocab_size(self) -> int:
        return int(max(self.kit_name_to_id.values(), default=0) + 1)

    def encode_style(self, s: str) -> int:
        return int(self.style_to_id.get(str(s), 0))

    def encode_beat_type(self, s: str) -> int:
        return int(self.beat_type_to_id.get(str(s), 0))

    def encode_kit_category(self, s: str) -> int:
        return int(self.kit_category_to_id.get(str(s), 0))

    def encode_drummer(self, s: str) -> int:
        return int(self.drummer_to_id.get(str(s), 0))

    def encode_kit_name(self, s: str) -> int:
        return int(self.kit_name_to_id.get(str(s), 0))

    def to_json(self) -> Dict[str, object]:
        return {
            "version": 4,
            "unk_id": 0,
            "style_to_id": dict(self.style_to_id),
            "beat_type_to_id": dict(self.beat_type_to_id),
            "kit_category_to_id": dict(self.kit_category_to_id),
            "drummer_to_id": dict(self.drummer_to_id),
            "kit_name_to_id": dict(self.kit_name_to_id),
            "channels": list(CHANNELS),
        }

    @staticmethod
    def from_json(obj: Dict[str, object]) -> "MidigrooveMetaVocab":
        return MidigrooveMetaVocab(
            style_to_id={str(k): int(v) for k, v in (obj.get("style_to_id", {}) or {}).items()},  # type: ignore[union-attr]
            beat_type_to_id={str(k): int(v) for k, v in (obj.get("beat_type_to_id", {}) or {}).items()},  # type: ignore[union-attr]
            kit_category_to_id={str(k): int(v) for k, v in (obj.get("kit_category_to_id", {}) or {}).items()},  # type: ignore[union-attr]
            drummer_to_id={str(k): int(v) for k, v in (obj.get("drummer_to_id", {}) or {}).items()},  # type: ignore[union-attr]
            kit_name_to_id={str(k): int(v) for k, v in (obj.get("kit_name_to_id", {}) or {}).items()},  # type: ignore[union-attr]
        )


def build_midigroove_vocab(*, train_csv: Path, val_csv: Path, use_style: bool = True) -> MidigrooveMetaVocab:
    import pandas as pd

    def _uniq(df, col: str) -> List[str]:
        if col not in df.columns:
            return []
        vals = [str(x) for x in df[col].astype(str).fillna("").tolist()]
        vals = [v for v in vals if v]
        return sorted(set(vals))

    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)

    styles: List[str]
    if bool(use_style):
        styles = sorted(set(_uniq(df_tr, "style") + _uniq(df_va, "style")))
    else:
        styles = []
    beat_types = sorted(set(_uniq(df_tr, "beat_type") + _uniq(df_va, "beat_type")))
    kit_names = sorted(set(_uniq(df_tr, "kit_name") + _uniq(df_va, "kit_name")))
    drummers = sorted(set(_uniq(df_tr, "drummer") + _uniq(df_va, "drummer")))
    kit_names_clean = [k for k in kit_names if str(k).strip() != ""]
    categories = list(kit_names_clean)

    def _mk(vals: List[str]) -> Dict[str, int]:
        # 0 reserved for UNK
        return {v: i + 1 for i, v in enumerate(vals)}

    return MidigrooveMetaVocab(
        style_to_id=_mk(styles),
        beat_type_to_id=_mk(beat_types),
        kit_category_to_id=_mk(categories),
        drummer_to_id=_mk(drummers),
        kit_name_to_id=_mk(sorted(set(kit_names_clean))),
    )


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _safe_relpath(x: object) -> str:
    """Convert a CSV cell to a relative path string; treat NaN/None as empty."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(float(x)):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


def _norm_text(s: str) -> str:
    return "".join(ch for ch in str(s).strip().lower() if ch.isalnum())


def _audio_duration_sec(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    if sf is not None:
        try:
            info = sf.info(str(path))
            dur = float(getattr(info, "duration", 0.0))
            return dur if dur > 0 else None
        except Exception:
            pass
    if torchaudio is not None:
        try:
            info = torchaudio.info(str(path))
            num_frames = int(getattr(info, "num_frames", 0))
            sr = int(getattr(info, "sample_rate", 0))
            if num_frames > 0 and sr > 0:
                return float(num_frames) / float(sr)
        except Exception:
            pass
    return None


def _audio_info(path: Path) -> Tuple[Optional[float], Optional[int]]:
    if not path.is_file():
        return None, None
    if sf is not None:
        try:
            info = sf.info(str(path))
            dur = float(getattr(info, "duration", 0.0))
            sr = int(getattr(info, "samplerate", 0))
            return (dur if dur > 0 else None), (sr if sr > 0 else None)
        except Exception:
            pass
    if torchaudio is not None:
        try:
            info = torchaudio.info(str(path))
            num_frames = int(getattr(info, "num_frames", 0))
            sr = int(getattr(info, "sample_rate", 0))
            dur = (float(num_frames) / float(sr)) if (num_frames > 0 and sr > 0) else None
            return dur, (sr if sr > 0 else None)
        except Exception:
            pass
    return _audio_duration_sec(path), None


def _load_audio_segment(path: Path, *, start_sample: int, num_samples: int) -> Tuple[np.ndarray, int]:
    """Load a mono float32 segment [num_samples] starting at start_sample."""
    if num_samples <= 0:
        return np.zeros((0,), dtype=np.float32), 0
    if sf is not None:
        audio, sr = sf.read(str(path), start=int(max(0, start_sample)), frames=int(num_samples), always_2d=False, dtype="float32")
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1).astype(np.float32, copy=False)
        return y, int(sr)
    if torchaudio is not None:
        wav, sr = torchaudio.load(str(path), frame_offset=int(max(0, start_sample)), num_frames=int(num_samples))
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
        return wav.detach().cpu().numpy().astype(np.float32, copy=False), int(sr)
    # Fallback: load entire audio then slice (slow).
    waveform, sr = load_audio(path)
    sr_i = int(sr)
    start = int(max(0, start_sample))
    end = int(min(start + int(num_samples), int(waveform.numel())))
    seg = waveform[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
    return seg, sr_i


def _compute_beat_pos(n_frames: int, *, fps: float, bpm: float) -> np.ndarray:
    if not (bpm and math.isfinite(bpm) and bpm > 1e-6):
        bpm = 120.0
    frames_per_beat = fps * (60.0 / bpm)
    if frames_per_beat <= 0:
        frames_per_beat = max(1.0, fps / 2.0)
    t = np.arange(n_frames, dtype=np.float32)
    beat_index = t / frames_per_beat
    pos = np.floor(np.mod(beat_index, 4.0)).astype(np.int64)
    return np.clip(pos, 0, 3)


def _gaussian_bump(hit_row: np.ndarray, center: int, amp: float, *, width: int = 2, sigma: float = 1.0) -> None:
    T = int(hit_row.shape[0])
    for d in range(-int(width), int(width) + 1):
        j = int(center + d)
        if 0 <= j < T:
            w = float(np.exp(-0.5 * (float(d) / max(1e-6, float(sigma))) ** 2))
            hit_row[j] = max(float(hit_row[j]), float(amp) * w)


def _add_fixed_decay(sustain_row: np.ndarray, i: int, amp: float, *, fps: float, decay_sec: float) -> None:
    L = int(round(float(decay_sec) * float(fps)))
    if L <= 0:
        return
    i = int(i)
    j1 = min(int(sustain_row.shape[0]), i + L)
    if j1 <= i:
        return
    t = np.arange(j1 - i, dtype=np.float32) / float(max(1e-6, fps))
    curve = np.exp(-t / max(1e-6, float(decay_sec) / 3.0)).astype(np.float32)
    sustain_row[i:j1] = np.maximum(sustain_row[i:j1], float(amp) * curve)


def _build_expressive_grid_from_events(
    notes: List[Tuple[int, float, float, int]],
    *,
    start_sec: float,
    dur_sec: float,
    fps: float,
    T: int,
    soft_hits: bool,
    sustain_mode: str,
    enable_choke: bool,
    include_vel: bool,
    include_sustain: bool,
) -> Dict[str, np.ndarray]:
    D = len(CHANNELS)
    hit = np.zeros((D, T), dtype=np.float32)
    vel = np.zeros((D, T), dtype=np.float32)
    sustain = np.zeros((D, T), dtype=np.float32)
    choke = np.zeros((D, T), dtype=np.float32)

    t0 = float(start_sec)
    t1 = t0 + float(dur_sec)
    fps_f = float(max(1e-6, fps))

    for pitch, n_start, n_end, n_vel in notes:
        ns = float(n_start)
        ne = float(n_end)
        if ne <= t0 or ns >= t1:
            continue

        pitch_i = int(PAPER_PITCH_CANONICAL.get(int(pitch), int(pitch)))
        ch = GM_MAP.get(pitch_i, "perc_aux")
        r = CH2I.get(ch)
        if r is None:
            continue

        v = float(n_vel)
        v = 0.0 if not np.isfinite(v) else max(0.0, min(127.0, v))
        v01 = float(v / 127.0)

        rel = max(0.0, min(ns - t0, dur_sec))
        i = int(np.clip(int(round(rel * fps_f)), 0, T - 1))

        if soft_hits:
            _gaussian_bump(hit[r], i, 1.0, width=2, sigma=1.0)
        else:
            hit[r, i] = 1.0
        if include_vel:
            vel[r, i] = max(float(vel[r, i]), v01)

        if include_sustain and ch in SUSTAIN_CHANNELS:
            if sustain_mode == "note_dur":
                rel_end = max(rel, min(ne - t0, dur_sec))
                j0 = i
                j1 = int(np.clip(int(np.ceil(rel_end * fps_f)), j0 + 1, T))
                sustain[r, j0:j1] = np.maximum(sustain[r, j0:j1], v01)
            else:
                _add_fixed_decay(
                    sustain[r],
                    i,
                    v01,
                    fps=fps_f,
                    decay_sec=float(DECAY_SEC.get(ch, 0.6)),
                )

            if enable_choke:
                if (ne - ns) < 0.08:
                    choke[r, i] = 1.0

    return {"hit": hit, "vel": vel, "sustain": sustain, "choke": choke}


def _sample_cc4_lane(
    cc4_events: List[Tuple[float, int]],
    *,
    start_sec: float,
    dur_sec: float,
    fps: float,
    T: int,
    polarity: str,
    hit_grid: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if not cc4_events:
        return None
    t0 = float(start_sec)
    t1 = t0 + float(dur_sec)
    events = [(float(t), int(v)) for (t, v) in cc4_events if t0 <= float(t) < t1]
    if not events:
        return None
    events.sort(key=lambda x: x[0])

    times = t0 + (np.arange(T, dtype=np.float32) / float(max(1e-6, fps)))
    out = np.zeros(T, dtype=np.float32)
    cur = 0.0
    j = 0
    for i, t in enumerate(times):
        while j < len(events) and events[j][0] <= float(t):
            cur = float(events[j][1]) / 127.0
            j += 1
        out[i] = cur

    pol = str(polarity or "auto")
    if pol == "invert":
        out = 1.0 - out
    elif pol == "auto" and hit_grid is not None:
        ci = CH2I.get("hh_closed", -1)
        oi = CH2I.get("hh_open", -1)
        if ci >= 0 and oi >= 0:
            closed_frames = np.where(hit_grid[ci] > 0.2)[0]
            open_frames = np.where(hit_grid[oi] > 0.2)[0]
            if closed_frames.size >= 3 and open_frames.size >= 3:
                if float(np.mean(out[open_frames])) < float(np.mean(out[closed_frames])):
                    out = 1.0 - out
    return out


def _stable_cache_key(
    spec: ChunkSpec,
    *,
    window_seconds: float,
    hop_seconds: float,
    encoder_tag: str,
    cache_tag: str,
) -> str:
    # Use relative-ish strings for stability across mounts.
    parts = [
        spec.audio_path.as_posix(),
        spec.midi_path.as_posix(),
        spec.kit_name,
        spec.kit_category,
        spec.drummer,
        f"start_idx={spec.start_idx}",
        f"win={float(window_seconds):.6f}",
        f"hop={float(hop_seconds):.6f}",
        f"enc={encoder_tag}",
        f"cfg={cache_tag}",
        "grid=v2",
    ]
    h = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return h[:20]


class MidigrooveDrumgridMetaCodesDataset(torch.utils.data.Dataset):
    """Dataset of cached 2s chunks for midigroove.

    Each item returns:
      - drum_grid:  [Dg, T] float32 (hit+vel+sustain+hh_cc4 concatenated)
      - drum_hit:   [D, T] float32
      - drum_vel:   [D, T] float32
      - drum_sustain: [D, T] float32
      - hh_open_cc4: [T] float32 (zeros when missing)
      - beat_pos:   [T] long (0..3)
      - tgt_codes:  [C, T] long
      - bpm:        scalar float32
      - style_id:   scalar long
      - beat_type_id: scalar long
      - kit_category_id: scalar long
      - valid_mask: [T] bool
    """

    def __init__(
        self,
        *,
        csv_path: Path,
        dataset_root: Path,
        split: str,
        vocab: MidigrooveMetaVocab,
        encoder: Optional[object],
        encoder_tag: str = "encodec",
        cache_dir: Optional[Path],
        window_seconds: float = 2.0,
        hop_seconds: float = 2.0,
        beats_per_chunk: Optional[int] = None,
        hop_beats: Optional[int] = None,
        require_cache: bool = False,
        encode_if_missing: bool = True,
        beat_type_only: Optional[str] = None,
        kit_category_only: Optional[str] = None,
        kit_category_exclude: Optional[str] = None,
        kit_category_top_n: Optional[int] = None,
        target_num_clips: Optional[int] = None,
        stratify: bool = False,
        stratify_key_kind: str = "drummer_style_kit_category",
        stratify_mode: str = "proportional",
        stratify_seed: int = 0,
        oversample_factor: float = 1.25,
        include_vel: bool = True,
        include_sustain: bool = True,
        include_hh_cc4: bool = True,
        max_items: Optional[int] = None,
        skip_errors: bool = True,
    ) -> None:
        import pandas as pd

        self.csv_path = Path(csv_path)
        self.dataset_root = Path(dataset_root)
        self.split = str(split)
        self.vocab = vocab
        self.encoder = encoder
        self.encoder_tag = str(encoder_tag or "encodec").strip().lower()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.beats_per_chunk = int(beats_per_chunk) if (beats_per_chunk is not None and int(beats_per_chunk) > 0) else None
        if hop_beats is None and self.beats_per_chunk is not None:
            hop_beats = int(self.beats_per_chunk)
        self.hop_beats = int(hop_beats) if (hop_beats is not None and int(hop_beats) > 0) else None
        self.require_cache = bool(require_cache)
        self.encode_if_missing = bool(encode_if_missing)
        self.beat_type_only = str(beat_type_only) if beat_type_only else None
        self.kit_category_only = str(kit_category_only) if kit_category_only else None
        self.kit_category_exclude = str(kit_category_exclude) if kit_category_exclude else None
        self._kit_name_only_norms: set[str] = set()
        if self.kit_category_only:
            self._kit_name_only_norms = {_norm_text(x) for x in str(self.kit_category_only).split(",") if _norm_text(x)}
        self._kit_name_exclude_norms: set[str] = set()
        if self.kit_category_exclude:
            self._kit_name_exclude_norms = {_norm_text(x) for x in str(self.kit_category_exclude).split(",") if _norm_text(x)}
        self.kit_category_top_n = int(kit_category_top_n) if (kit_category_top_n is not None and int(kit_category_top_n) > 0) else None
        self._kit_category_top_set: set[str] = set()
        if self.kit_category_top_n is not None and self._kit_name_only_norms:
            raise ValueError("Use only one of kit_category_only or kit_category_top_n.")
        self.target_num_clips = int(target_num_clips) if (target_num_clips is not None and int(target_num_clips) > 0) else None
        self.stratify = bool(stratify)
        self.stratify_key_kind = str(stratify_key_kind or "drummer_style_kit_category").strip()
        self.stratify_mode = str(stratify_mode or "proportional").strip().lower()
        self.stratify_seed = int(stratify_seed)
        self.oversample_factor = float(oversample_factor) if (oversample_factor and float(oversample_factor) > 1.0) else 1.0
        self.max_items = int(max_items) if (max_items is not None and int(max_items) > 0) else None
        self.skip_errors = bool(skip_errors)
        self.include_vel = bool(include_vel)
        self.include_sustain = bool(include_sustain)
        self.include_hh_cc4 = bool(include_hh_cc4)
        # Expressive grid knobs (fixed for now; keep deterministic for caching).
        self.soft_hits = True
        self.sustain_mode = "fixed_decay"
        self.enable_choke = False
        self.hh_cc4_polarity = "auto"

        # Include a config tag in the manifest name so different ablations /
        # sampling regimes don't clobber each other when sharing cache_dir.
        tag_parts = [f"enc={self.encoder_tag}"]
        if self.beats_per_chunk is not None:
            tag_parts.extend([f"beats={int(self.beats_per_chunk)}", f"hopbeats={int(self.hop_beats or self.beats_per_chunk)}"])
        else:
            tag_parts.extend([f"win={self.window_seconds:.3f}", f"hop={self.hop_seconds:.3f}"])
        tag_parts.extend(
            [
                f"vel={int(self.include_vel)}",
                f"sus={int(self.include_sustain)}",
                f"hh={int(self.include_hh_cc4)}",
                f"beat={self.beat_type_only or 'any'}",
                f"kitcat={self.kit_category_only or ('top' + str(self.kit_category_top_n)) if self.kit_category_top_n is not None else 'any'}",
                f"kitx={('1' if self.kit_category_exclude else '0')}",
                f"strat={int(self.stratify)}",
                f"key={self.stratify_key_kind}",
                f"mode={self.stratify_mode}",
                f"n={self.target_num_clips or 'all'}",
                f"seed={self.stratify_seed}",
            ]
        )
        self._manifest_tag = hashlib.sha1("|".join(tag_parts).encode("utf-8")).hexdigest()[:10]
        self._cache_tag = self._manifest_tag
        self._manifest_path = (
            self.cache_dir / f"manifest_midigroove_{self.split}_{self._manifest_tag}.jsonl" if self.cache_dir else None
        )
        self._items: List[Dict[str, object]] = []
        self._chunks: List[ChunkSpec] = []
        self._midi_cache: Dict[str, Dict[str, object]] = {}

        # If we're in cache-only mode, prefer using an existing manifest (exact
        # tag if present; otherwise fall back to the most recent manifest for
        # this split under cache_dir). This ensures training can use a
        # pre-cached fixed-size subset without needing to regenerate the full
        # chunk list.
        if self.require_cache:
            manifest_to_use: Optional[Path] = None
            if self._manifest_path is not None and self._manifest_path.is_file():
                manifest_to_use = self._manifest_path
            elif self.cache_dir is not None and self.cache_dir.is_dir():
                cands = sorted(
                    self.cache_dir.glob(f"manifest_midigroove_{self.split}_*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if cands:
                    manifest_to_use = cands[0]
                    self._manifest_path = manifest_to_use
            if manifest_to_use is None:
                raise FileNotFoundError(
                    f"No cache manifest found for split={self.split!r} under {self.cache_dir}. "
                    "Run with --precache --precache-only first (same --cache-dir)."
                )

            self._items = list(self._read_manifest(manifest_to_use))
            if self.max_items is not None:
                self._items = self._items[: self.max_items]
        else:
            df = pd.read_csv(self.csv_path)
            if "split" in df.columns:
                df = df[df["split"].astype(str) == str(self.split)]

            # Optional: keep only the top-N kit names (by row count) for this split,
            # after applying beat_type filter.
            if self.kit_category_top_n is not None:
                df2 = df
                if self.beat_type_only is not None and "beat_type" in df2.columns:
                    df2 = df2[df2["beat_type"].astype(str) == str(self.beat_type_only)]
                if "kit_name" in df2.columns:
                    kit_names = [str(x) for x in df2["kit_name"].astype(str).fillna("").tolist()]
                    kit_names = [k for k in kit_names if k.strip() != ""]
                    if kit_names:
                        from collections import Counter

                        top = [k for k, _ in Counter(kit_names).most_common(int(self.kit_category_top_n))]
                        self._kit_category_top_set = set(top)

            rows = df.to_dict(orient="records")
            chunks: List[ChunkSpec] = []
            scan_total = 0
            scan_has_paths = 0
            scan_audio_exists = 0
            scan_midi_exists = 0
            scan_audio_info_ok = 0
            scan_beat_match = 0
            scan_kit_match = 0
            try:  # optional progress bar for the (potentially slow) chunk enumeration
                from tqdm.auto import tqdm  # type: ignore
            except Exception:  # pragma: no cover
                tqdm = None  # type: ignore

            row_iter = rows if tqdm is None else tqdm(rows, desc=f"scan_chunks[{self.split}]", dynamic_ncols=True)
            for row in row_iter:
                scan_total += 1
                audio_rel = _safe_relpath(row.get("audio_filename", ""))
                midi_rel = _safe_relpath(row.get("midi_filename", ""))
                if not audio_rel or not midi_rel:
                    continue
                scan_has_paths += 1
                audio_path = (self.dataset_root / audio_rel).resolve()
                midi_path = (self.dataset_root / midi_rel).resolve()
                if not audio_path.is_file():
                    # Requirement: discard rows where WAV is missing.
                    continue
                scan_audio_exists += 1
                if not midi_path.is_file():
                    # We need MIDI to build the drum grid.
                    continue
                scan_midi_exists += 1

                dur, sr0 = _audio_info(audio_path)
                if dur is None or sr0 is None:
                    continue
                scan_audio_info_ok += 1
                sr0_i = int(sr0)

                bpm = _safe_float(row.get("bpm", 0.0), default=0.0)
                beat_type = str(row.get("beat_type", "") or "")
                if self.beat_type_only is not None and beat_type != self.beat_type_only:
                    continue
                scan_beat_match += 1
                style = str(row.get("style", "") or "")
                kit_name = str(row.get("kit_name", "") or "")
                kit_category = kit_name  # "category" is the raw kit name
                if self._kit_name_only_norms:
                    if _norm_text(kit_name) not in self._kit_name_only_norms:
                        continue
                if self._kit_name_exclude_norms:
                    if _norm_text(kit_name) in self._kit_name_exclude_norms:
                        continue
                if self._kit_category_top_set:
                    if kit_name not in self._kit_category_top_set:
                        continue
                scan_kit_match += 1
                drummer = str(row.get("drummer", "") or "")

                bpm_eff = float(bpm) if (bpm and math.isfinite(float(bpm)) and float(bpm) > 1e-6) else 120.0
                if self.beats_per_chunk is not None:
                    win_sec = float(self.beats_per_chunk) * (60.0 / bpm_eff)
                    hop_beats_eff = float(self.hop_beats or self.beats_per_chunk)
                    hop_sec = hop_beats_eff * (60.0 / bpm_eff)
                else:
                    win_sec = float(self.window_seconds)
                    hop_sec = float(self.hop_seconds)

                if win_sec <= 0 or hop_sec <= 0:
                    continue
                if dur < win_sec:
                    continue
                n_chunks = int(math.floor((dur - win_sec) / max(hop_sec, 1e-6))) + 1
                if n_chunks <= 0:
                    continue

                for start_idx in range(n_chunks):
                    start_sec = float(start_idx) * float(hop_sec)
                    chunks.append(
                        ChunkSpec(
                            audio_path=audio_path,
                            midi_path=midi_path,
                            sr=sr0_i,
                            drummer=drummer,
                            bpm=bpm,
                            beat_type=beat_type,
                            style=style,
                            kit_name=kit_name,
                            kit_category=kit_category,
                            start_idx=int(start_idx),
                            start_sec=float(start_sec),
                            window_seconds=float(win_sec),
                            hop_seconds=float(hop_sec),
                        )
                    )
                if tqdm is not None:
                    try:
                        row_iter.set_postfix(chunks=len(chunks))  # type: ignore[attr-defined]
                    except Exception:
                        pass

            if self.max_items is not None:
                chunks = chunks[: self.max_items]
            # Optional: reduce to a target number of clips, stratified by
            # (drummer, style, kit_category).
            if self.target_num_clips is not None:
                desired = int(self.target_num_clips)
                buf = int(math.ceil(float(desired) * float(self.oversample_factor)))
                cand_n = min(len(chunks), max(desired, buf))
                if self.stratify:
                    idxs = _stratified_sample_indices(
                        chunks,
                        n=cand_n,
                        seed=self.stratify_seed,
                        key_kind=str(self.stratify_key_kind),
                        mode=self.stratify_mode,
                    )
                else:
                    rng = np.random.default_rng(int(self.stratify_seed))
                    idxs = rng.permutation(len(chunks)).tolist()[:cand_n]
                self._chunks = [chunks[i] for i in idxs]
            else:
                self._chunks = chunks
            if not self._chunks:
                print(
                    "warning: no cache chunks generated. "
                    f"split={self.split!r} beat_type_only={self.beat_type_only!r} "
                    f"kit_name_only={self.kit_category_only!r} kit_name_exclude={self.kit_category_exclude!r} kit_name_top_n={self.kit_category_top_n!r} "
                    f"dataset_root={str(self.dataset_root)!r} "
                    f"scan_total={scan_total} has_paths={scan_has_paths} audio_exists={scan_audio_exists} "
                    f"midi_exists={scan_midi_exists} audio_info_ok={scan_audio_info_ok} "
                    f"beat_match={scan_beat_match} kit_match={scan_kit_match}",
                    file=sys.stderr,
                )

    def _get_midi_payload(self, midi_path: Path) -> Dict[str, object]:
        key = midi_path.as_posix()
        cached = self._midi_cache.get(key)
        if cached is not None:
            return cached

        payload: Dict[str, object] = {"notes": [], "cc4": []}
        if pretty_midi is None or not midi_path.is_file():
            self._midi_cache[key] = payload
            return payload
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            notes = [
                (int(n.pitch), float(n.start), float(n.end), int(getattr(n, "velocity", 0)))
                for inst in pm.instruments
                if inst.is_drum
                for n in inst.notes
            ]
            cc4 = [
                (float(cc.time), int(cc.value))
                for inst in pm.instruments
                if inst.is_drum
                for cc in getattr(inst, "control_changes", [])
                if int(getattr(cc, "number", -1)) == 4
            ]
            payload = {"notes": notes, "cc4": cc4}
        except Exception:
            payload = {"notes": [], "cc4": []}

        self._midi_cache[key] = payload
        # Bound memory on huge datasets.
        if len(self._midi_cache) > 4096:
            try:
                first_key = next(iter(self._midi_cache.keys()))
                self._midi_cache.pop(first_key, None)
            except Exception:
                pass
        return payload

    @staticmethod
    def _read_manifest(path: Path) -> Iterator[Dict[str, object]]:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "npz" in obj:
                    yield obj
            except Exception:
                continue

    def __len__(self) -> int:
        if self._items:
            return int(len(self._items))
        return int(len(self._chunks))

    def _cache_item_path(self, key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        p = self.cache_dir / "items"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{key}.npz"

    def _load_npz(self, path: Path) -> Dict[str, np.ndarray]:
        npz = np.load(path, allow_pickle=False)
        return {k: np.asarray(npz[k]) for k in npz.files}

    def _encode_and_cache(self, spec: ChunkSpec) -> Optional[Dict[str, object]]:
        if self.encoder is None:
            raise RuntimeError("encoder is required to build cache entries")
        key = _stable_cache_key(
            spec,
            window_seconds=float(spec.window_seconds),
            hop_seconds=float(spec.hop_seconds),
            encoder_tag=self.encoder_tag,
            cache_tag=self._cache_tag,
        )
        cache_path = self._cache_item_path(key)
        if cache_path is not None and cache_path.is_file():
            return {"key": key, "npz": str(cache_path)}

        # Load only the requested window (avoid decoding the whole file for each chunk).
        # We rely on the file's native sample rate for correct alignment.
        # Compute start/length in samples from the native sample rate.
        sr_i = int(spec.sr)
        start_s = int(round(float(spec.start_sec) * sr_i))
        win_s = int(round(float(spec.window_seconds) * sr_i))
        end_s = start_s + win_s
        if win_s <= 0:
            return None
        chunk, sr_i2 = _load_audio_segment(spec.audio_path, start_sample=start_s, num_samples=win_s)
        if sr_i2 != sr_i:
            sr_i = int(sr_i2)
        if chunk.size <= 0:
            return None
        if int(chunk.shape[0]) < int(win_s):
            # Incomplete last window; keep fixed-length chunks faithful by skipping.
            return None

        # Encode codes.
        codes, cb = self.encoder.encode_waveform(chunk, sr_i, seconds_per_chunk=float(spec.window_seconds))  # type: ignore[attr-defined]
        if codes is None:
            return None
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 2:
            return None
        C, T = int(codes.shape[0]), int(codes.shape[1])
        if C <= 0 or T <= 0:
            return None

        fps = float(T) / max(float(spec.window_seconds), 1e-6)
        beat_pos = _compute_beat_pos(T, fps=fps, bpm=float(spec.bpm))

        # Align expressive MIDI conditioning to codec frames.
        payload = self._get_midi_payload(spec.midi_path)
        notes = list(payload.get("notes", []))  # type: ignore[list-item]
        cc4 = list(payload.get("cc4", []))  # type: ignore[list-item]
        grids = _build_expressive_grid_from_events(
            notes,  # type: ignore[arg-type]
            start_sec=float(spec.start_sec),
            dur_sec=float(spec.window_seconds),
            fps=float(fps),
            T=int(T),
            soft_hits=bool(self.soft_hits),
            sustain_mode=str(self.sustain_mode),
            enable_choke=bool(self.enable_choke),
            include_vel=bool(self.include_vel),
            include_sustain=bool(self.include_sustain),
        )
        drum_hit = grids["hit"]
        if float(np.sum(drum_hit)) <= 0.0:
            return None
        drum_vel = grids["vel"]
        drum_sustain = grids["sustain"]
        if self.include_hh_cc4:
            hh_lane = _sample_cc4_lane(
                cc4,  # type: ignore[arg-type]
                start_sec=float(spec.start_sec),
                dur_sec=float(spec.window_seconds),
                fps=float(fps),
                T=int(T),
                polarity=str(self.hh_cc4_polarity),
                hit_grid=drum_hit,
            )
            if hh_lane is None:
                hh_lane = np.zeros(int(T), dtype=np.float32)
        else:
            hh_lane = np.zeros(int(T), dtype=np.float32)

        style_id = int(self.vocab.encode_style(spec.style))
        beat_type_id = int(self.vocab.encode_beat_type(spec.beat_type))
        kit_category_id = int(self.vocab.encode_kit_category(spec.kit_category))
        drummer_id = int(self.vocab.encode_drummer(spec.drummer))
        kit_name_id = int(self.vocab.encode_kit_name(spec.kit_name))

        if cache_path is not None:
            semantics = json.dumps(
                {
                    "dataset": "midigroove",
                    "version": 3,
                    "window_seconds": float(spec.window_seconds),
                    "hop_seconds": float(spec.hop_seconds),
                    "beats_per_chunk": int(self.beats_per_chunk or 0),
                    "hop_beats": int(self.hop_beats or 0),
                    "encoder": self.encoder_tag,
                    "xcodec_bandwidth": float(getattr(self.encoder, "bandwidth", 0.0) or 0.0) if self.encoder_tag == "xcodec" else 0.0,
                    "grid": "pad_articulation_v1",
                    "channels": list(CHANNELS),
                    "soft_hits": bool(self.soft_hits),
                    "sustain_mode": str(self.sustain_mode),
                    "enable_choke": bool(self.enable_choke),
                    "hh_cc4_polarity": str(self.hh_cc4_polarity),
                    "include_vel": bool(self.include_vel),
                    "include_sustain": bool(self.include_sustain),
                    "include_hh_cc4": bool(self.include_hh_cc4),
                    "kit_category_top_n": int(self.kit_category_top_n or 0),
                    "kit_category_top_set": sorted(self._kit_category_top_set) if self._kit_category_top_set else [],
                },
                sort_keys=True,
            )
            np.savez_compressed(
                cache_path,
                semantics=np.asarray(semantics),
                audio_path=np.asarray(spec.audio_path.as_posix()),
                midi_path=np.asarray(spec.midi_path.as_posix()),
                split=np.asarray(str(self.split)),
                sr=np.asarray(int(sr_i), dtype=np.int64),
                cb=np.asarray(int(cb)),
                fps=np.asarray(float(fps), dtype=np.float32),
                start_sec=np.asarray(float(spec.start_sec), dtype=np.float32),
                window_seconds=np.asarray(float(spec.window_seconds), dtype=np.float32),
                hop_seconds=np.asarray(float(spec.hop_seconds), dtype=np.float32),
                beats_per_chunk=np.asarray(int(self.beats_per_chunk or 0), dtype=np.int64),
                hop_beats=np.asarray(int(self.hop_beats or 0), dtype=np.int64),
                bpm=np.asarray(float(spec.bpm), dtype=np.float32),
                drummer=np.asarray(str(spec.drummer)),
                style=np.asarray(str(spec.style)),
                beat_type=np.asarray(str(spec.beat_type)),
                kit_name=np.asarray(str(spec.kit_name)),
                kit_category=np.asarray(str(spec.kit_category)),
                style_id=np.asarray(int(style_id), dtype=np.int64),
                beat_type_id=np.asarray(int(beat_type_id), dtype=np.int64),
                kit_category_id=np.asarray(int(kit_category_id), dtype=np.int64),
                drummer_id=np.asarray(int(drummer_id), dtype=np.int64),
                kit_name_id=np.asarray(int(kit_name_id), dtype=np.int64),
                beat_pos=beat_pos.astype(np.int64, copy=False),
                drum_hit=drum_hit.astype(np.float32, copy=False),
                drum_vel=drum_vel.astype(np.float32, copy=False),
                drum_sustain=drum_sustain.astype(np.float32, copy=False),
                hh_open_cc4=hh_lane.astype(np.float32, copy=False),
                tgt=codes.astype(np.int64, copy=False),
            )

        return {"key": key, "npz": str(cache_path) if cache_path is not None else ""}

    def precache(self, *, desc: str = "precache") -> Dict[str, int]:
        if self.cache_dir is None:
            raise ValueError("precache requires cache_dir")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self._manifest_path is None:
            raise RuntimeError("manifest_path missing")

        ok = 0
        skipped = 0
        errors = 0

        # Write vocab alongside the cache so cache-only runs can reconstruct embedding sizes.
        vocab_path = self.cache_dir / "midigroove_vocab.json"
        try:
            vocab_path.write_text(json.dumps(self.vocab.to_json(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception:
            pass

        from tqdm.auto import tqdm

        manifest_lines: List[str] = []
        for spec in tqdm(self._chunks, desc=desc, dynamic_ncols=True):
            try:
                rec = self._encode_and_cache(spec)
                if rec is None:
                    skipped += 1
                    continue
                manifest_lines.append(json.dumps(rec, sort_keys=True))
                ok += 1
                if self.target_num_clips is not None and ok >= int(self.target_num_clips):
                    break
            except Exception:
                if self.skip_errors:
                    errors += 1
                    continue
                raise

        # Atomically overwrite manifest for this split.
        tmp = self._manifest_path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
        tmp.replace(self._manifest_path)
        return {"ok": int(ok), "skipped": int(skipped), "errors": int(errors)}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._items:
            rec = self._items[int(idx)]
            npz_path = Path(str(rec["npz"]))
            arrs = self._load_npz(npz_path)
        else:
            spec = self._chunks[int(idx)]
            key = _stable_cache_key(
                spec,
                window_seconds=float(spec.window_seconds),
                hop_seconds=float(spec.hop_seconds),
                encoder_tag=self.encoder_tag,
                cache_tag=self._cache_tag,
            )
            cache_path = self._cache_item_path(key)
            if cache_path is not None and cache_path.is_file():
                arrs = self._load_npz(cache_path)
            else:
                if self.require_cache and not self.encode_if_missing:
                    raise FileNotFoundError(f"cache missing for {spec.audio_path} start={spec.start_sec:.3f}s")
                if not self.encode_if_missing:
                    raise FileNotFoundError("cache missing and encode_if_missing=False")
                rec = self._encode_and_cache(spec)
                if rec is None:
                    raise RuntimeError("failed to encode/cache item (possibly empty grid or incomplete window)")
                cache_path2 = Path(str(rec["npz"]))
                arrs = self._load_npz(cache_path2)

        # Backward/forward cache compatibility: support older field names.
        if "tgt" not in arrs and "tgt_codes" in arrs:
            arrs["tgt"] = arrs["tgt_codes"]
        if "drum_hit" not in arrs and "drum_grid" in arrs:
            # Older caches stored only a single grid; treat it as hit.
            arrs["drum_hit"] = arrs["drum_grid"]

        tgt = torch.from_numpy(np.asarray(arrs["tgt"], dtype=np.int64))  # [C,T]
        drum_hit = torch.from_numpy(np.asarray(arrs["drum_hit"], dtype=np.float32))  # [D,T]
        T = int(tgt.shape[-1])
        D = int(drum_hit.shape[0])
        # Backward/forward cache compatibility: missing optional fields -> zeros.
        if "drum_vel" in arrs:
            drum_vel = torch.from_numpy(np.asarray(arrs["drum_vel"], dtype=np.float32))
        else:
            drum_vel = torch.zeros((D, T), dtype=torch.float32)
        if "drum_sustain" in arrs:
            drum_sustain = torch.from_numpy(np.asarray(arrs["drum_sustain"], dtype=np.float32))
        else:
            drum_sustain = torch.zeros((D, T), dtype=torch.float32)
        if "hh_open_cc4" in arrs:
            hh_open_cc4 = torch.from_numpy(np.asarray(arrs["hh_open_cc4"], dtype=np.float32)).reshape(-1)
        else:
            hh_open_cc4 = torch.zeros((T,), dtype=torch.float32)
        # Apply feature toggles (optional conditioning inputs).
        if not bool(self.include_vel):
            drum_vel = torch.zeros_like(drum_vel)
        if not bool(self.include_sustain):
            drum_sustain = torch.zeros_like(drum_sustain)
        if not bool(self.include_hh_cc4):
            hh_open_cc4 = torch.zeros_like(hh_open_cc4)
        beat_pos = torch.from_numpy(np.asarray(arrs["beat_pos"], dtype=np.int64))  # [T]
        bpm = torch.tensor(float(np.asarray(arrs["bpm"]).reshape(-1)[0]), dtype=torch.float32)
        style_id = torch.tensor(int(np.asarray(arrs["style_id"]).reshape(-1)[0]), dtype=torch.long)
        beat_type_id = torch.tensor(int(np.asarray(arrs["beat_type_id"]).reshape(-1)[0]), dtype=torch.long)
        if "kit_category_id" in arrs:
            kit_category_id = torch.tensor(int(np.asarray(arrs["kit_category_id"]).reshape(-1)[0]), dtype=torch.long)
        else:
            # Older caches may have kit_id/kit_name; default to UNK id.
            kit_category_id = torch.tensor(0, dtype=torch.long)
        valid_mask = torch.ones(T, dtype=torch.bool)
        drum_grid = torch.cat(
            [
                drum_hit,
                *( [drum_vel] if bool(self.include_vel) else [] ),
                *( [drum_sustain] if bool(self.include_sustain) else [] ),
                *( [hh_open_cc4[None, :]] if bool(self.include_hh_cc4) else [] ),
            ],
            dim=0,
        )
        return {
            "tgt_codes": tgt,
            "drum_grid": drum_grid,
            "drum_hit": drum_hit,
            "drum_vel": drum_vel,
            "drum_sustain": drum_sustain,
            "hh_open_cc4": hh_open_cc4,
            "beat_pos": beat_pos,
            "bpm": bpm,
            "style_id": style_id,
            "beat_type_id": beat_type_id,
            "kit_category_id": kit_category_id,
            "valid_mask": valid_mask,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("empty batch")
        C = int(batch[0]["tgt_codes"].shape[0])
        Dg = int(batch[0]["drum_grid"].shape[0])
        D = int(batch[0]["drum_hit"].shape[0])
        T_max = int(max(int(b["tgt_codes"].shape[-1]) for b in batch))

        tgt = torch.full((len(batch), C, T_max), PAD_ID, dtype=torch.long)
        drum_grid = torch.zeros((len(batch), Dg, T_max), dtype=torch.float32)
        drum_hit = torch.zeros((len(batch), D, T_max), dtype=torch.float32)
        drum_vel = torch.zeros((len(batch), D, T_max), dtype=torch.float32)
        drum_sustain = torch.zeros((len(batch), D, T_max), dtype=torch.float32)
        hh_open_cc4 = torch.zeros((len(batch), T_max), dtype=torch.float32)
        beat_pos = torch.zeros((len(batch), T_max), dtype=torch.long)
        valid_mask = torch.zeros((len(batch), T_max), dtype=torch.bool)
        bpm = torch.zeros((len(batch),), dtype=torch.float32)
        style_id = torch.zeros((len(batch),), dtype=torch.long)
        beat_type_id = torch.zeros((len(batch),), dtype=torch.long)
        kit_category_id = torch.zeros((len(batch),), dtype=torch.long)

        for i, b in enumerate(batch):
            t = int(b["tgt_codes"].shape[-1])
            tgt[i, :, :t] = b["tgt_codes"]
            drum_grid[i, :, :t] = b["drum_grid"]
            drum_hit[i, :, :t] = b["drum_hit"]
            drum_vel[i, :, :t] = b["drum_vel"]
            drum_sustain[i, :, :t] = b["drum_sustain"]
            hh_open_cc4[i, :t] = b["hh_open_cc4"]
            beat_pos[i, :t] = b["beat_pos"]
            valid_mask[i, :t] = True
            bpm[i] = b["bpm"]
            style_id[i] = b["style_id"]
            beat_type_id[i] = b["beat_type_id"]
            kit_category_id[i] = b["kit_category_id"]

        return {
            "tgt_codes": tgt,
            "drum_grid": drum_grid,
            "drum_hit": drum_hit,
            "drum_vel": drum_vel,
            "drum_sustain": drum_sustain,
            "hh_open_cc4": hh_open_cc4,
            "beat_pos": beat_pos,
            "valid_mask": valid_mask,
            "bpm": bpm,
            "style_id": style_id,
            "beat_type_id": beat_type_id,
            "kit_category_id": kit_category_id,
        }
