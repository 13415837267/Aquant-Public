# Architecture

## Production flow

```text
zzshare
   |
   v
Historical database (Aquant-Public)
   |
   +--> daily incremental data
   |
   +--> historical/fundamental research data
   |
   v
scripts/update_candidates.py
   |
   +--> reads the latest database window
   |
   +--> loads Aquant-Private/main at runtime
   |
   v
data/candidates.json
   |
   v
Next.js / GitHub Pages
```

## Repository responsibilities

- **Aquant-Private**: the single source of truth for production strategy code, factors, parameters, and strategy version.
- **Aquant-Public**: production data, database collection, quality checks, candidate generation, backtesting infrastructure, Actions, and Pages.
- Public does **not** maintain an independent strategy implementation.

## Database

Daily market and valuation data are persisted as:

`data/history/YYYY/YYYY-MM-DD.csv.gz`

Quarterly financial data are persisted under:

`data/fundamentals/{indicator,income,balance,cash_flow}/`

Candidate generation reads the persisted database rather than rebuilding five years of history from the provider on every run.

## Strategy loading

At candidate-generation time, Public checks out `Aquant-Private/main` and loads:

- `strategy/model.py`
- `strategy/version.py`

The generated `data/candidates.json` records both `strategy_version` and `strategy_commit` for reproducibility.

## Runtime boundary

The current production boundary stops at candidate generation. Broker/OMS execution remains a separate future service.
