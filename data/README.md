# Data

`candidates.json` is the current generated release artifact.

Historical market data is stored as one gzip-compressed CSV per trading day:

`data/history/YYYY/YYYY-MM-DD.csv.gz`

The historical layer is scoped to Shanghai/Shenzhen main-board A shares. Board scope is determined by stock code rather than the current listing status, so later-delisted main-board stocks can remain available for historical research.

Current strategy generation reads the latest 61 stored trading-day files, derives a 60-trading-day momentum series from the provider `pct_chg` field, and excludes stocks without a complete momentum window.

Daily ingestion runs through GitHub Actions. Provider credentials belong in GitHub Actions secrets and never in source control.
