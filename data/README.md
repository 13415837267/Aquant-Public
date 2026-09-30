# Data

`candidates.json` is a generated release artifact.

The ingestion job runs on trading days at 18:00 Beijing time, checks the market calendar, builds the current candidate snapshot, and commits only when the generated file actually changes.

v0.1 uses the AKShare Eastmoney A-share spot interface. Credentials, if a future provider requires them, belong in GitHub Actions secrets or Vercel environment variables and never in source control.
