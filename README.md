# Midigroove POC: expressive drum grid → codec tokens

Proof-of-concept pipeline for predicting discrete audio codec token sequences (Encodec / DAC / X-Codec) from MIDI-derived *expressive* drum conditioning grids (E-GMD / Midigroove).

The workflow is:
1) Build a cache of fixed-length windows where the **input** is a MIDI-derived expressive drum grid + `bpm` + `drummer_id` + `beat_pos`, and the **target** is discrete codec token indices for the aligned WAV.
2) Train a Transformer encoder to predict codec tokens from this conditioning (kit category is only for optional cache filtering). By default the model uses `drum_hit`+`drum_vel` (+ `bpm`/`drummer_id`/`beat_pos`); you can optionally add `drum_sustain` and `hh_open_cc4`.
3) Evaluate multiple checkpoints with a unified script across caches/codecs.

## Repo Layout

- `data/`: dataset + codec encode/decode helpers
- `midigroove_poc/`: unified CLI + cache/train/eval modules
- `notes.txt`: personal scratchpad commands (kept functional)

## Setup

Recommended: Python 3.10+.

1) Install PyTorch (GPU optional, follow the official instructions): https://pytorch.org/get-started/locally/
2) Install the remaining deps:

```bash
pip install -r requirements.txt
```

## Data

This repo expects the E-GMD / Midigroove dataset on disk (WAV + MIDI + metadata CSV). The dataset is not included.

Set:

```bash
export EGMD_CSV="/media/maindisk/ksoil/data/e-gmd-v1.0.0/e-gmd-v1.0.0/e-gmd-v1.0.0.csv"
export EGMD_ROOT="/media/maindisk/ksoil/data/e-gmd-v1.0.0/e-gmd-v1.0.0"
```

## Build A Cache (Codec Tokens)

Encodec example (precache-only, E-GMD big dataset, Acoustic/Pop kits only):

```bash
python -m midigroove_poc drumgrid train \
  --train-csv "$EGMD_CSV" \
  --val-csv "$EGMD_CSV" \
  --train-split train \
  --val-split validation \
  --test-split test \
  --dataset-root "$EGMD_ROOT" \
  --cache-dir cache/encodec_big_acoustic_pop \
  --precache --precache-only \
  --beat-type-only beat \
  --kit-category-only "Acoustic/Pop" \
  --stratify-clips \
  --encode-device cuda:0 \
  --num-workers 0 \
  --seed 0 --beats-per-chunk 4 --hop-beats 4
```

## Train (Expressive Grid → Tokens)

```bash
python -m midigroove_poc expressivegrid train \
  --cache cache/encodec_big_acoustic_pop \
  --device cuda:0 \
  --save artifacts/checkpoints/midigroove_expressivegrid_to_encodec.pt \
  --encoder-model encodec \
  --steps 200000 --log-every 200 --early-stop-steps 3000
```

## Evaluation

Evaluate multiple systems on a standardized subset (intersection of cache segments):

```bash
python -m midigroove_poc eval \
  --intersection --split test --max-items 256 --eval-sr 16000 --audio-metrics \
  --out-dir artifacts/eval/std \
  --system encodec:artifacts/checkpoints/expgrid_encodec.pt:cache/midigroove_encodec_4beats_std:encodec \
  --system dac:artifacts/checkpoints/expgrid_dac.pt:cache/midigroove_dac_4beats_std:dac \
  --system xcodec:artifacts/checkpoints/expgrid_xcodec.pt:cache/midigroove_xcodec_4beats_std:xcodec
```

Outputs: `artifacts/eval/std/summary.json` and `artifacts/eval/std/items.csv` (plus optional WAVs).

You can also load systems from a file:

```bash
python -m midigroove_poc eval --systems-file artifacts/exp/systems.txt --intersection --split test --audio-metrics
```

## Experiments (DAC / X-Codec)

The repo includes a ready-to-run experiment matrix (small/medium/large architectures) for DAC and X-Codec:

- `scripts/experiments_dac_xcodec.sh`
- `experiments/README.md`

## Notes

See `notes.txt` for additional command variants and historical run logs.
