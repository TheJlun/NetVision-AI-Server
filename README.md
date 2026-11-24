# NetVision AI Server (dataset export build)

This is a compatible receiver scaffold, not a trained detection model. It stores incoming samples with `label: unlabeled` and provides export routes.

## Routes
- `/` service info
- `/health` health status
- `/count` sample count
- `/dataset-info` file metadata
- `/export` download JSON Lines dataset
- `/export.csv` download CSV for spreadsheet review

## Environment
- `NETVISION_API_KEY` (default `NetVision`)
- `DATA_DIR` (default `dataset`)

**Important:** Render free instances have ephemeral storage; files may be lost after restart/deploy. Download exports regularly. Do not treat unlabeled data as training-ready or as evidence of cheating. This server returns a neutral `probability: 0.0` and does not classify players.
