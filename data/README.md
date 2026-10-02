# Data

`candidates.json` is the current generated release artifact.

Historical market data is stored as one gzip-compressed CSV per trading day:

`data/history/YYYY/YYYY-MM-DD.csv.gz`

The historical layer may retain Shanghai/Shenzhen/Beijing stock rows. It does not use the current active strategy universe as a historical filter, so later-delisted rows can remain available for research. Production candidate generation is restricted to Shanghai/Shenzhen main-board A shares.

Current strategy generation reads the latest 61 stored trading-day files, derives a 60-trading-day momentum series from the provider `pct_chg` field, and excludes stocks without a complete momentum window.

Daily ingestion runs through GitHub Actions. Provider credentials belong in GitHub Actions secrets and never in source control.
