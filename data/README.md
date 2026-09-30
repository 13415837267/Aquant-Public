# Data

`candidates.json` is the current generated release artifact.

Each trading day also creates one immutable snapshot under `data/history/YYYY-MM-DD.json`. This is the incremental history layer used for later monitoring, attribution and backtesting.

The ingestion job runs on trading days at 18:00 Beijing time, checks the market calendar, builds the current candidate snapshot, writes that day's history file, and commits only generated data.

v0.1 uses the AKShare Eastmoney A-share spot interface. Credentials, if a future provider requires them, belong in GitHub Actions secrets or Vercel environment variables and never in source control.
