# midigroove_poc

Proof-of-concept pipeline for generating drum audio from *expressive* symbolic drum performances by predicting discrete neural audio codec tokens (EnCodec / DAC / X-Codec) from a MIDI-derived conditioning grid.

At a high level:
1) Encode paired drum audio into discrete codec codes.
2) Extract an expressive drum grid from the paired MIDI (micro-timing, velocity, drum IDs, optional kit/style metadata).
3) Train a Transformer to predict codec codes from the expressive grid.
4) Decode predicted codes back to audio for evaluation and listening.

Artifacts (checkpoints, eval outputs) live under `artifacts/`. Dataset caches live under `cache/`.

## Setup

Install Python deps:

```bash
python -m pip install -r requirements.txt
```

You will also need working PyTorch + torchaudio/soundfile, and access to the Hugging Face codec models referenced by this repo (via `transformers`).

## Dataset

Point the code at your Expanded Groove MIDI (E‑GMD) CSV and dataset root:

```bash
export EGMD_CSV="/path/to/e-gmd-v1.0.0.csv"
export EGMD_ROOT="/path/to/e-gmd-v1.0.0"
```

You can also pass `--train-csv/--val-csv/--dataset-root` explicitly to the CLI.

## CLI

The repo exposes a unified CLI:

```bash
python -m midigroove_poc --help
python -m midigroove_poc drumgrid train --help
python -m midigroove_poc expressivegrid train --help
python -m midigroove_poc eval --help
```

## Typical usage

### 1) Build caches (audio→codec codes + aligned expressive grids)

Example (single kit, EnCodec):

```bash
python -m midigroove_poc drumgrid train \
  --cache-dir cache/encodec_acoustic \
  --encoder-model encodec \
  --precache --precache-only \
  --beat-type-only beat \
  --kit-category-only "Acoustic Kit" \
  --encode-device cuda:0 --num-workers 0 --seed 0 \
  --beats-per-chunk 4 --hop-beats 4
```

Notes:
- `--encoder-model` is one of: `encodec`, `dac`, `xcodec`.
- For X-Codec you can set `--xcodec-bandwidth` (e.g. `2.0`).
- If `--encode-device` is CUDA, keep `--num-workers 0` (CUDA encoding is not fork-safe).

### 2) Train (expressivegrid→codec token model)

```bash
python -m midigroove_poc expressivegrid train \
  --cache cache/encodec_acoustic \
  --device cuda:0 \
  --save artifacts/checkpoints/expressivegrid_to_encodec.pt \
  --encoder-model encodec \
  --steps 200000 --log-every 300
```

### 3) Evaluate (objective metrics + saved preds)

```bash
python -m midigroove_poc eval \
  --split test --intersection --max-items 0 \
  --device cuda:0 --decode-device cuda:0 \
  --audio-metrics --add-oracle \
  --pred-dir artifacts/pred --pred-include-ref --save-preds 128 \
  --system encodec:artifacts/checkpoints/expressivegrid_to_encodec.pt:cache/encodec_acoustic:encodec \
  --out-dir artifacts/eval/example_run
```

## Reproducible command list

`notes.txt` is the scratchpad of full end-to-end commands used for the paper-style runs (single-kit vs all-kits, small vs big models, and multi-codec eval).
