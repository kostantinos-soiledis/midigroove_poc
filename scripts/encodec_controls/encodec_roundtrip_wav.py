#!/usr/bin/env python3
"""EnCodec encode->decode roundtrip smoke test.

This is useful to verify your local env can:
  - load HF EnCodec (facebook/encodec_32khz)
  - encode waveform -> discrete codes
  - decode codes -> waveform
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence


def _require_soundfile():
    try:
        import soundfile as sf  # type: ignore

        return sf
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"soundfile is required: {e}")


def _require_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"numpy is required: {e}")


def _require_torch():
    try:
        import torch  # type: ignore

        return torch
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "PyTorch is required for EnCodec roundtrip in this repo.\n"
            f"Install torch/torchaudio then re-run.\n\n{type(e).__name__}: {e}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True, help="Input wav path.")
    ap.add_argument("--out-wav", type=Path, default=None, help="Optional write reconstructed wav.")
    ap.add_argument("--device", type=str, default="cpu", help="torch device (cpu/cuda:0).")
    ap.add_argument("--seconds", type=float, default=2.0, help="Trim input to this many seconds (0 disables).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    np = _require_numpy()
    sf = _require_soundfile()
    torch = _require_torch()

    from data.codecs import EncodecCodesEncoder, decode_tokens_to_audio

    y, sr = sf.read(str(args.wav), dtype="float32", always_2d=False)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1).astype(np.float32, copy=False)
    if float(args.seconds) > 0.0 and sr > 0:
        y = y[: int(round(float(args.seconds) * float(sr)))]

    dev = torch.device(str(args.device))
    enc = EncodecCodesEncoder(dev)
    codes_ct, cb = enc.encode_waveform(y, int(sr))

    # decode_tokens_to_audio expects [B,C,T]
    tokens_bct = torch.from_numpy(codes_ct[None, :, :].astype("int64", copy=False))
    yhat_b, sr_hat = decode_tokens_to_audio(tokens_bct, encoder_model="encodec", device=str(args.device))
    yhat = np.asarray(yhat_b[0], dtype=np.float32)

    if int(sr_hat) != int(enc.target_sr):
        # This shouldn't happen, but avoid silently mis-reporting metrics.
        raise SystemExit(f"Unexpected sample-rate mismatch: decoded_sr={sr_hat}, encoder_target_sr={enc.target_sr}")

    # Align lengths.
    n = int(min(y.shape[0], yhat.shape[0]))
    y0 = y[:n]
    y1 = yhat[:n]
    err = y0 - y1
    mse = float(np.mean(err * err))
    sig = float(np.mean(y0 * y0))
    snr = float("inf") if mse <= 1e-12 else 10.0 * math.log10(max(1e-12, sig) / max(1e-12, mse))

    print(f"encoded codes: shape={codes_ct.shape} (C={cb})")
    print(f"decoded: sr={sr_hat}, samples={yhat.shape[0]}")
    print(f"mse={mse:.6e}  snr_db={snr:.2f}")

    if args.out_wav is not None:
        out = Path(args.out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), yhat, int(sr_hat))
        print(f"wrote: {out}")


if __name__ == "__main__":
    main()

