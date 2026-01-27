"""Audio codec helpers (Encodec / DAC / X-Codec).

This repo uses discrete codec "codes" (token indices) as a training target.
The wrappers here standardize:
  - encoding: waveform -> integer codes shaped [C, T]
  - decoding: integer codes -> waveform for listening UIs
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np

# Environment hygiene: avoid importing TF/JAX via Transformers.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:  # allow `python -m ... --help` to work without torch installed
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _require_torch():
    if torch is None:  # pragma: no cover
        raise RuntimeError(
            "`torch` is required for codec encode/decode helpers. "
            "Install PyTorch from https://pytorch.org/get-started/locally/."
        )
    return torch

try:  # pragma: no cover - optional dependency
    from transformers import AutoProcessor  # type: ignore
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import DacModel  # type: ignore
except Exception:  # pragma: no cover
    DacModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import AutoFeatureExtractor  # type: ignore
except Exception:  # pragma: no cover
    AutoFeatureExtractor = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import XcodecModel  # type: ignore
except Exception:  # pragma: no cover
    XcodecModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import AutoModel  # type: ignore
except Exception:  # pragma: no cover
    AutoModel = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from transformers import EncodecModel  # type: ignore
except Exception:  # pragma: no cover
    EncodecModel = None  # type: ignore


class DacCodesEncoder:
    """Thin wrapper around HF DAC for robust chunked encoding to codes.

    Produces integer codes shaped [C, T] for mono audio.
    """

    def __init__(self, device: torch.device, model_name: str = "descript/dac_44khz") -> None:
        torch = _require_torch()
        if DacModel is None or AutoProcessor is None:  # pragma: no cover
            raise ImportError("Transformers with DacModel is required for DAC code encoding.")
        self.model = DacModel.from_pretrained(model_name).to(device).eval()
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)  # type: ignore[call-arg]
        except TypeError:  # pragma: no cover
            self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        try:
            self.num_codebooks: Optional[int] = int(getattr(self.model.config, "num_codebooks", 0)) or None
        except Exception:  # pragma: no cover
            self.num_codebooks = None

    def encode_waveform(self, y: np.ndarray, sr: int, *, seconds_per_chunk: float = 2.0) -> Tuple[np.ndarray, int]:
        torch = _require_torch()
        with torch.no_grad():
            if y.ndim != 1:
                y = y.mean(axis=1).astype(np.float32)
            else:
                y = y.astype(np.float32)
            target_sr = 44100
            if sr != target_sr:
                dur = y.shape[0] / float(sr)
                tgt_len = int(round(dur * target_sr))
                x_old = np.linspace(0.0, 1.0, num=y.shape[0], endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=tgt_len, endpoint=False)
                y = np.interp(x_new, x_old, y).astype(np.float32)
                sr = target_sr

            hop = int(max(1, seconds_per_chunk * sr))
            pieces: list[np.ndarray] = []
            cb_out: Optional[int] = None
            for start in range(0, len(y), hop):
                chunk = y[start : min(len(y), start + hop)]
                if chunk.size == 0:
                    continue
                inputs = self.processor(raw_audio=chunk, sampling_rate=sr, return_tensors="pt")
                inputs["input_values"] = inputs["input_values"].to(self.device)
                enc = self.model.encode(inputs["input_values"])  # type: ignore[attr-defined]
                codes = enc.audio_codes
                if codes.dim() != 3:
                    raise RuntimeError(f"Unexpected DAC audio_codes shape: {tuple(codes.shape)}")
                _, a1, a2 = codes.shape
                arr2d = codes.squeeze(0).detach().cpu().numpy()
                target_cb = self.num_codebooks or cb_out or 9
                if a1 == target_cb:
                    arr = arr2d
                elif a2 == target_cb:
                    arr = arr2d.T
                else:
                    if a1 <= 32 and a2 > 32:
                        arr = arr2d
                    elif a2 <= 32 and a1 > 32:
                        arr = arr2d.T
                    else:
                        arr = arr2d

                if cb_out is None:
                    cb_out = arr.shape[0]
                else:
                    if arr.shape[0] != cb_out and arr.shape[1] == cb_out:
                        arr = arr.T
                    if arr.shape[0] != cb_out:
                        if arr.shape[0] > cb_out:
                            arr = arr[:cb_out]
                        else:
                            reps = int(math.ceil(cb_out / float(arr.shape[0])))
                            arr = np.tile(arr, (reps, 1))[:cb_out]
                pieces.append(arr)

            if not pieces:
                raise RuntimeError("DAC encoding produced no chunks")
            codes_cat = np.concatenate(pieces, axis=1)
            cb = codes_cat.shape[0] if cb_out is None else cb_out
            return codes_cat.astype(np.int64, copy=False), int(cb)


class XcodecCodesEncoder:
    """Wrapper around HF X-Codec that produces codes shaped [C, T]."""

    def __init__(
        self,
        device: torch.device,
        model_id: str = "hf-audio/xcodec-hubert-general",
        bandwidth: float = 2.0,
    ) -> None:
        _require_torch()
        if XcodecModel is None or AutoFeatureExtractor is None:  # pragma: no cover
            raise ImportError("Transformers with XcodecModel is required for X-Codec code encoding.")
        self.device = device
        self.model = XcodecModel.from_pretrained(
            model_id,
            device_map=device if device.type == "cuda" else None,
        ).to(device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.target_sr = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        self.bandwidth = float(bandwidth)
        try:
            self.num_codebooks: Optional[int] = int(getattr(self.model.config, "num_codebooks", 0)) or None
        except Exception:  # pragma: no cover
            self.num_codebooks = None

    def encode_waveform(self, y: np.ndarray, sr: int, *, seconds_per_chunk: float = 2.0) -> Tuple[np.ndarray, int]:
        torch = _require_torch()
        with torch.no_grad():
            if y.ndim != 1:
                y = y.mean(axis=1).astype(np.float32)
            else:
                y = y.astype(np.float32)

            if sr != self.target_sr:
                dur = y.shape[0] / float(sr)
                tgt_len = int(round(dur * self.target_sr))
                if tgt_len <= 0:
                    raise RuntimeError("X-Codec encoder received empty audio after resampling")
                x_old = np.linspace(0.0, 1.0, num=y.shape[0], endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=tgt_len, endpoint=False)
                y = np.interp(x_new, x_old, y).astype(np.float32)
                sr = self.target_sr

            inputs = self.feature_extractor(raw_audio=y, sampling_rate=self.target_sr, return_tensors="pt").to(self.device)
            audio = inputs["input_values"]

            codes = self.model.encode(audio, bandwidth=self.bandwidth, return_dict=False)
            if isinstance(codes, (tuple, list)):
                codes = codes[0]
            if not isinstance(codes, torch.Tensor):
                raise RuntimeError(f"Unexpected X-Codec encode() output type: {type(codes)}")
            if codes.dim() == 3:
                arr2d = codes.squeeze(0).detach().cpu().numpy()
            elif codes.dim() == 2:
                arr2d = codes.detach().cpu().numpy()
            else:
                raise RuntimeError(f"Unexpected X-Codec codes shape: {tuple(codes.shape)}")
            if arr2d.ndim != 2:
                raise RuntimeError(f"Expected 2D codes array, got shape {arr2d.shape}")
            a1, a2 = arr2d.shape
            target_cb = self.num_codebooks or a1
            if a1 == target_cb:
                arr = arr2d
            elif a2 == target_cb:
                arr = arr2d.T
            else:
                if a1 <= 32 and a2 > 32:
                    arr = arr2d
                elif a2 <= 32 and a1 > 32:
                    arr = arr2d.T
                else:
                    arr = arr2d
            cb = int(arr.shape[0])
            return arr.astype(np.int64, copy=False), cb


class EncodecCodesEncoder:
    """Wrapper around HF Encodec that produces codes shaped [C, T]."""

    def __init__(self, device: torch.device, model_id: str = "facebook/encodec_32khz") -> None:
        _require_torch()
        if AutoModel is None or AutoFeatureExtractor is None:  # pragma: no cover
            raise ImportError("Transformers with Encodec/AutoFeatureExtractor are required for Encodec encoding.")
        self.device = device
        self.model = AutoModel.from_pretrained(
            model_id,
            device_map=device if device.type == "cuda" else None,
        ).to(device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        try:
            self.target_sr: int = int(getattr(self.feature_extractor, "sampling_rate", 32000))
        except Exception:  # pragma: no cover
            self.target_sr = 32000
        try:
            self.num_codebooks: Optional[int] = int(getattr(self.model.config, "num_codebooks", 0)) or None
        except Exception:  # pragma: no cover
            self.num_codebooks = None

    def encode_waveform(self, y: np.ndarray, sr: int, *, seconds_per_chunk: float = 2.0) -> Tuple[np.ndarray, int]:
        torch = _require_torch()
        with torch.no_grad():
            if y.ndim != 1:
                y = y.mean(axis=1).astype(np.float32)
            else:
                y = y.astype(np.float32)

            if sr != self.target_sr:
                dur = y.shape[0] / float(sr)
                tgt_len = int(round(dur * self.target_sr))
                if tgt_len <= 0:
                    raise RuntimeError("Encodec encoder received empty audio after resampling")
                x_old = np.linspace(0.0, 1.0, num=y.shape[0], endpoint=False)
                x_new = np.linspace(0.0, 1.0, num=tgt_len, endpoint=False)
                y = np.interp(x_new, x_old, y).astype(np.float32)
                sr = self.target_sr

            inputs = self.feature_extractor(raw_audio=y, sampling_rate=self.target_sr, return_tensors="pt").to(self.device)
            audio = inputs["input_values"]
            outputs = self.model.encode(audio)
            codes = getattr(outputs, "audio_codes", outputs)
            if isinstance(codes, (tuple, list)):
                codes = codes[0]
            if not isinstance(codes, torch.Tensor):
                raise RuntimeError(f"Unexpected Encodec encode() output type: {type(codes)}")
            if codes.dim() == 4:
                if codes.size(0) != 1 or codes.size(1) != 1:
                    raise RuntimeError(f"Unexpected Encodec 4D codes shape (B,C,Q,T): {tuple(codes.shape)}")
                arr2d = codes[0, 0].detach().cpu().numpy()
            elif codes.dim() == 3:
                arr2d = codes.squeeze(0).detach().cpu().numpy()
            elif codes.dim() == 2:
                arr2d = codes.detach().cpu().numpy()
            else:
                raise RuntimeError(f"Unexpected Encodec codes shape: {tuple(codes.shape)}")
            if arr2d.ndim != 2:
                raise RuntimeError(f"Expected 2D codes array, got shape {arr2d.shape}")

            a1, a2 = arr2d.shape
            target_cb = self.num_codebooks or a1
            if a1 == target_cb:
                arr = arr2d
            elif a2 == target_cb:
                arr = arr2d.T
            else:
                if a1 <= 64 and a2 > 64:
                    arr = arr2d
                elif a2 <= 64 and a1 > 64:
                    arr = arr2d.T
                else:
                    arr = arr2d
            try:
                codebook_size = int(getattr(self.model.config, "codebook_size", 2048))
            except Exception:  # pragma: no cover
                codebook_size = 2048
            arr = np.clip(arr, 0, max(0, codebook_size - 1)).astype(np.int64, copy=False)
            cb = int(arr.shape[0])
            return arr, cb


def decode_tokens_to_audio(
    tokens_bct: "torch.Tensor",
    *,
    encoder_model: str = "dac",
    device: "torch.device | str | None" = None,
) -> Tuple[np.ndarray, int]:
    """Decode tokens to a numpy waveform batch and return (audio, sample_rate).

    Args:
        tokens_bct: [B, C, T] or [C, T] integer codes.
        encoder_model: 'encodec'|'dac'|'xcodec'
        device: device to run the decoder on (e.g. 'cuda:0' or 'cpu').
    Returns:
        audio: [B, T] float32
        sample_rate: int
    """
    torch = _require_torch()
    kind = (encoder_model or "dac").strip().lower()
    dev = torch.device("cpu" if device is None else device)

    with torch.no_grad():
        if tokens_bct.dim() == 2:
            codes_bct = tokens_bct.unsqueeze(0)
        elif tokens_bct.dim() == 3:
            codes_bct = tokens_bct
        else:
            raise RuntimeError(f"decode_tokens_to_audio expected 2D or 3D tensor, got shape {tuple(tokens_bct.shape)}")
        codes_bct = codes_bct.to(device=dev, dtype=torch.long)

        if kind == "dac":
            if DacModel is None:  # pragma: no cover
                raise ImportError("Transformers with DacModel is required for encoder_model='dac'.")
            cache = getattr(decode_tokens_to_audio, "_dac_cache", None)
            if cache is None:
                cache = {}
                setattr(decode_tokens_to_audio, "_dac_cache", cache)
            key = ("dac", str(dev))
            model = cache.get(key)
            if model is None:
                model = DacModel.from_pretrained("descript/dac_44khz").to(dev).eval()
                cache[key] = model

            try:
                expected_codebooks = int(getattr(model.config, "num_codebooks", codes_bct.shape[1]))
            except Exception:
                expected_codebooks = int(codes_bct.shape[1])
            if int(codes_bct.shape[1]) != int(expected_codebooks) and int(expected_codebooks) > 0:
                if int(codes_bct.shape[1]) > int(expected_codebooks):
                    codes_bct = codes_bct[:, : int(expected_codebooks), :]
                else:
                    pad = int(expected_codebooks) - int(codes_bct.shape[1])
                    codes_bct = torch.nn.functional.pad(codes_bct, (0, 0, 0, pad))

            try:
                codebook_size = int(getattr(model.config, "codebook_size", 1024))
            except Exception:
                codebook_size = 1024
            codes_bct = codes_bct.clamp(min=0, max=max(0, codebook_size - 1))
            codes_btc = codes_bct.permute(0, 2, 1).contiguous()  # [B, T, C]

            def _extract_audio_values(out: object) -> "torch.Tensor":
                if isinstance(out, (tuple, list)) and out:
                    out = out[0]
                v = getattr(out, "audio_values", out)
                if not isinstance(v, torch.Tensor):
                    raise RuntimeError(f"Unexpected DAC decode output type: {type(v)}")
                return v

            def _try_decode(codes: torch.Tensor) -> "torch.Tensor | None":
                try:
                    return _extract_audio_values(model.decode(audio_codes=codes))  # type: ignore[call-arg]
                except Exception:
                    pass
                try:
                    return _extract_audio_values(model.decode(codes))  # type: ignore[call-arg]
                except Exception:
                    pass
                try:
                    return _extract_audio_values(model.decode(codebook_indices=codes))  # type: ignore[call-arg]
                except Exception:
                    pass
                try:
                    return _extract_audio_values(model(audio_codes=codes))  # type: ignore[call-arg]
                except Exception:
                    pass
                return None

            audio_values = None
            for candidate in (codes_btc, codes_bct):
                audio_values = _try_decode(candidate)
                if audio_values is not None:
                    break
            if audio_values is None:
                raise RuntimeError(
                    "DAC decode failed. Try upgrading/downgrading `transformers` and ensure the checkpoint "
                    "`descript/dac_44khz` loads correctly."
                )

            audio_np = audio_values.detach().to("cpu").numpy()
            if audio_np.ndim == 3:
                audio_np = audio_np[:, 0, :]
            elif audio_np.ndim != 2:
                audio_np = audio_np.reshape(audio_np.shape[0], -1)
            try:
                sample_rate = int(getattr(model.config, "sampling_rate", 44100))
            except Exception:
                sample_rate = 44100
            return audio_np.astype(np.float32, copy=False), sample_rate

        if kind == "xcodec":
            if XcodecModel is None:  # pragma: no cover
                raise ImportError("Transformers with XcodecModel is required for encoder_model='xcodec'.")
            cache = getattr(decode_tokens_to_audio, "_xcodec_cache", None)
            if cache is None:
                cache = {}
                setattr(decode_tokens_to_audio, "_xcodec_cache", cache)
            key = ("xcodec", str(dev))
            model = cache.get(key)
            if model is None:
                model = XcodecModel.from_pretrained("hf-audio/xcodec-hubert-general").to(dev).eval()
                cache[key] = model

            try:
                expected_codebooks = int(getattr(model.config, "num_codebooks", codes_bct.shape[1]))
            except Exception:
                expected_codebooks = int(codes_bct.shape[1])
            if int(codes_bct.shape[1]) != int(expected_codebooks) and int(expected_codebooks) > 0:
                if int(codes_bct.shape[1]) > int(expected_codebooks):
                    codes_bct = codes_bct[:, : int(expected_codebooks), :]
                else:
                    pad = int(expected_codebooks) - int(codes_bct.shape[1])
                    codes_bct = torch.nn.functional.pad(codes_bct, (0, 0, 0, pad))

            try:
                codebook_size = int(getattr(model.config, "codebook_size", 1024))
            except Exception:
                codebook_size = 1024
            codes_bct = codes_bct.clamp(min=0, max=max(0, codebook_size - 1))

            # X-Codec decode API differs across transformers versions; try a few variants.
            codes_btc = codes_bct.permute(0, 2, 1).contiguous()  # [B,T,C]

            def _extract_audio_values(out: object) -> "torch.Tensor":
                if isinstance(out, (tuple, list)) and out:
                    out = out[0]
                v = getattr(out, "audio_values", out)
                if not isinstance(v, torch.Tensor):
                    raise RuntimeError(f"Unexpected X-Codec decode output type: {type(v)}")
                return v

            audio_values = None
            last_err: Optional[BaseException] = None
            for codes in (codes_btc, codes_bct):
                for kwargs in (
                    {"return_dict": False},
                    {"return_dict": False, "audio_codes": codes},
                ):
                    try:
                        if "audio_codes" in kwargs:
                            out = model.decode(**kwargs)  # type: ignore[call-arg]
                        else:
                            out = model.decode(codes, return_dict=False)  # type: ignore[call-arg]
                        audio_values = _extract_audio_values(out)
                        last_err = None
                        break
                    except Exception as e:
                        last_err = e
                        audio_values = None
                if audio_values is not None:
                    break
            if audio_values is None:
                raise RuntimeError(f"X-Codec decode failed. Last error: {last_err}")

            audio_np = audio_values.detach().to("cpu").numpy()
            if audio_np.ndim == 3:
                audio_np = audio_np[:, 0, :]
            elif audio_np.ndim != 2:
                audio_np = audio_np.reshape(audio_np.shape[0], -1)
            try:
                sample_rate = int(getattr(model.config, "sampling_rate", 16000))
            except Exception:
                sample_rate = 16000
            return audio_np.astype(np.float32, copy=False), sample_rate

        if kind == "encodec":
            if EncodecModel is None:  # pragma: no cover
                raise ImportError("Transformers with EncodecModel is required for encoder_model='encodec'.")
            cache = getattr(decode_tokens_to_audio, "_encodec_cache", None)
            if cache is None:
                cache = {}
                setattr(decode_tokens_to_audio, "_encodec_cache", cache)
            key = ("encodec", str(dev))
            model = cache.get(key)
            if model is None:
                model = EncodecModel.from_pretrained("facebook/encodec_32khz").to(dev).eval()
                cache[key] = model

            try:
                expected_codebooks = int(
                    getattr(model.config, "num_quantizers", getattr(model.config, "num_codebooks", codes_bct.shape[1]))
                )
            except Exception:
                expected_codebooks = int(codes_bct.shape[1])
            if int(codes_bct.shape[1]) != int(expected_codebooks) and int(expected_codebooks) > 0:
                if int(codes_bct.shape[1]) > int(expected_codebooks):
                    codes_bct = codes_bct[:, : int(expected_codebooks), :]
                else:
                    pad = int(expected_codebooks) - int(codes_bct.shape[1])
                    codes_bct = torch.nn.functional.pad(codes_bct, (0, 0, 0, pad))

            try:
                codebook_size = int(getattr(model.config, "codebook_size", 2048))
            except Exception:
                codebook_size = 2048
            codes_bct = codes_bct.clamp(min=0, max=max(0, codebook_size - 1)).to(dev)

            # HF Encodec expects [B, channels, Q, T] (channels is typically 1).
            audio_codes = codes_bct.unsqueeze(1)  # [B, 1, Q, T]
            batch_size = int(audio_codes.size(0))
            audio_scales_candidates = [
                [None],
                [torch.ones(batch_size, 1, device=dev)],
            ]
            last_err: Optional[BaseException] = None
            audio_values = None
            for audio_scales in audio_scales_candidates:
                try:
                    audio_values = model.decode(
                        audio_codes=audio_codes,
                        audio_scales=audio_scales,  # type: ignore[arg-type]
                        return_dict=False,
                        last_frame_pad_length=0,
                    )[0]
                    last_err = None
                    break
                except RuntimeError as e:
                    last_err = e
                except TypeError as e:
                    last_err = e
            if audio_values is None:
                msg = (str(last_err) if last_err is not None else "").lower()
                if dev.type == "cuda" and ("cudnn" in msg or "cuda error" in msg or "cublas" in msg):
                    return decode_tokens_to_audio(tokens_bct, encoder_model="encodec", device="cpu")
                raise RuntimeError(f"Encodec decode failed (tried audio_scales=[None] and ones). Last error: {last_err}")

            audio_np = audio_values.detach().to("cpu").numpy()
            if audio_np.ndim == 3:
                audio_np = audio_np[:, 0, :]
            elif audio_np.ndim != 2:
                audio_np = audio_np.reshape(audio_np.shape[0], -1)
            try:
                sample_rate = int(getattr(model.config, "sampling_rate", 32000))
            except Exception:
                sample_rate = 32000
            return audio_np.astype(np.float32, copy=False), sample_rate

        raise ValueError(f"Unsupported encoder_model: {encoder_model!r} (expected: 'encodec'|'dac'|'xcodec')")
