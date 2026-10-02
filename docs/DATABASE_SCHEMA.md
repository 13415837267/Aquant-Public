# Quant Database Schema

The public repository is the only cloud build/runtime surface. The database is built incrementally with Git-tracked checkpoints so an interrupted job can resume without re-downloading completed units.

## Core layers

### Daily market data
Path: `data/history/YYYY/YYYY-MM-DD.csv.gz`

One row per source stock and trading date. The daily collector keeps Shanghai/Shenzhen/Beijing stock rows in the database; candidate generation applies the narrower Shanghai/Shenzhen main-board scope later:

- OHLCV and turnover amount
- percentage change and price change
- adjustment factor
- average price
- upper/lower limit
- turnover rate
- amplitude
- paused flag
- ST flag

The historical daily layer is intentionally broader than the production candidate universe. It does not use the current active `list_status='L'` universe as a row filter, so later ST/delisted and non-target-board stock rows can remain available for research. The production candidate layer filters to Shanghai/Shenzhen main-board A shares.

### Daily valuation
Path: `data/fundamentals/valuation/YYYY-MM-DD_YYYY-MM-DD.csv.gz`

Historical daily valuation snapshots, including the provider's valuation fields such as PE/PB/PS, market capitalization, circulating capitalization and turnover-related fields. The files retain `symbol` and `date` as stable keys.

### Quarterly fundamentals
Paths:

- `data/fundamentals/indicator/YYYYqN.csv.gz`
- `data/fundamentals/income/YYYYqN.csv.gz`
- `data/fundamentals/balance/YYYYqN.csv.gz`
- `data/fundamentals/cash_flow/YYYYqN.csv.gz`

Raw provider fields are retained. Standardized fields `report_date` and `pub_date` are added. `pub_date` is the publication date and is the key field for point-in-time joins that avoid look-ahead bias.

## Checkpoint files

- `data/history/_BACKFILL_STATE.json`
- `data/history/_BACKFILL_COMPLETE`
- `data/history/_ZZSHARE_VALIDATION.json`
- `data/fundamentals/_FUNDAMENTALS_STATE.json`
- `data/fundamentals/_FUNDAMENTALS_COMPLETE`
- `data/fundamentals/_FUNDAMENTALS_VALIDATION.json`

Every completed daily file and fundamentals batch updates a state file and creates a Git checkpoint commit. Re-running the workflow skips completed units.

## Strategy universe

`data/universe.json` is the current active, non-ST main-board strategy universe used for current stock names and candidate context. It is not the historical database filter. The database can retain non-target-board rows.

## Look-ahead protection

Do not join quarterly financial values using `report_date` alone. Use `pub_date <= trade_date` and the latest available publication for each security. For price-based momentum and volatility, use the provider `pct_chg` return series rather than applying a future-adjusted factor to historical close prices. The `factor` field remains stored for research and future adjustment work.
