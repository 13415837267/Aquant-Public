# Architecture

## Production flow

```text
zzshare
   |
   v
Shanghai/Shenzhen main-board historical database
   |
   +--> daily incremental data
   |
   +--> historical/fundamental research data
   |
   v
scripts/update_candidates.py
   |
   +--> hard eligibility filters
   +--> cross-sectional factor scoring from Aquant-Private/main
   +--> candidate admission policy
   |
   v
data/candidates.json
   |
   v
scripts/portfolio.py
   |
   +--> deterministic inverse-volatility allocation
   +--> 5% single-name cap / 5% cash buffer
   |
   v
data/portfolio.json
   |
   v
scripts/execution_plan.py
   |
   +--> lot/T+1/cash/turnover checks
   +--> next-open recheck gate
   |
   v
data/execution_plan.json
   |
   +--> historical backtest / constrained execution research
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

`scripts/pit_fundamentals.py` provides the canonical point-in-time read path: only publications with `pub_date <= trade_date` can enter a historical information set, and the latest available publication is selected per security.

Candidate generation reads the persisted database rather than rebuilding five years of history from the provider on every run.

## Strategy loading

At candidate-generation time, Public checks out `Aquant-Private/main` and loads:

- `strategy/model.py`
- `strategy/version.py`

The generated `data/candidates.json` records both `strategy_version` and `strategy_commit` for reproducibility.

## Runtime boundary

Production computation covers data collection, candidate generation, portfolio construction, execution-plan preparation, and historical execution-constrained research. Broker/OMS submission remains a separate external service; `data/execution_plan.json` is an auditable plan and not a broker fill.
