# Quant Database Schema

The public repository is the only cloud build/runtime surface. The database is built incrementally with Git-tracked checkpoints so an interrupted job can resume without re-downloading completed units.

## Core layers

### Daily market data
Path: `data/history/YYYY/YYYY-MM-DD.csv.gz`

One row per stock and trading date. The daily collector uses zzshare's full-field market endpoint and keeps:

- OHLCV and turnover amount
- percentage change and price change
- adjustment factor
- average price
- upper/lower limit
- turnover rate
- amplitude
- paused flag
- ST flag

This is the raw full-market layer. Strategy universe exclusion of ST/delisted securities is kept separately in `data/universe.json` so historical state is not silently rewritten into survivorship-biased data.

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

`data/universe.json` is the current active, non-ST strategy universe. It is not used as a filter for the raw daily layer; historical data may contain securities that later became ST or were later delisted. Strategy code should use historical `is_st` and lifecycle/status data when constructing a tradable universe.

## Look-ahead protection

Do not join quarterly financial values using `report_date` alone. Use `pub_date <= trade_date` and the latest available publication for each security. This convention is encoded in the stored schema so a later PIT feature layer can be generated deterministically.
