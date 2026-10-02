# Data

`candidates.json` is the current generated release artifact.

Historical market data is stored as one gzip-compressed CSV per trading day:

`data/history/YYYY/YYYY-MM-DD.csv.gz`

The collector can retain Shanghai/Shenzhen/Beijing stock rows without using the current active strategy universe as a historical filter. The existing historical files are the previously built Shanghai/Shenzhen main-board dataset; future rebuilds use the broader database stock scope. Production candidate generation remains restricted to Shanghai/Shenzhen main-board A shares.

Current strategy generation reads the latest 61 stored trading-day files, derives a 60-trading-day momentum series from the provider `pct_chg` field, and excludes stocks without a complete momentum window.

Daily ingestion runs through GitHub Actions. Provider credentials belong in GitHub Actions secrets and never in source control.
