# Experiments

This folder contains reproducible experiment entrypoints.

## DAC + X-Codec matrix (small/medium/large)

Use `scripts/experiments_dac_xcodec.sh`:

- Run via bash (from repo root):
  - `bash scripts/experiments_dac_xcodec.sh cache`
  - `bash scripts/experiments_dac_xcodec.sh train`
  - `bash scripts/experiments_dac_xcodec.sh eval`
  - `bash scripts/experiments_dac_xcodec.sh all`
  - `bash scripts/experiments_dac_xcodec.sh screen` (runs `all` inside a detached GNU screen session)
- Or make it executable:
  - `chmod +x scripts/experiments_dac_xcodec.sh`
  - `./scripts/experiments_dac_xcodec.sh cache`

- Build caches (DAC + X-Codec):
  - `ENCODE_DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh cache`
- Train 3 model sizes per codec:
  - `DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh train`
- Evaluate all 6 systems on common ground (intersection across caches):
  - `DEVICE=cuda:2 DECODE_DEVICE=cuda:2 bash scripts/experiments_dac_xcodec.sh eval`

You must set dataset env vars first:

- `EGMD_CSV` – path to the E-GMD CSV
- `EGMD_ROOT` – dataset root directory
