"""Walk-forward / out-of-sample evaluation for the fixed Private strategy.

The current Private model is not trained by this script. Each fold therefore
measures genuine out-of-sample calendar performance of the frozen strategy
after a rolling historical window, while preserving the 60-session feature
warm-up before the OOS start.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import backtest as base

ROOT = base.ROOT
OUT_FILE = ROOT / "data" / "backtest" / "walk_forward.json"


def trading_dates() -> list[str]:
    return [p.name[:10] for p in base.history_files()]


def prior_calendar_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year - years)
    except ValueError:
        return d.replace(year=d.year - years, day=28)


def build_folds(dates: list[str], train_years: int, test_years: int) -> list[dict]:
    if not dates:
        raise ValueError("history is empty")
    first = date.fromisoformat(dates[0])
    last = date.fromisoformat(dates[-1])
    folds = []
    oos_start = prior_calendar_years(first, -0)  # replaced below
    # Start with the first complete train_years block available in history.
    cursor = first.replace(year=first.year + train_years)
    while cursor <= last:
        oos_end = min(cursor.replace(year=cursor.year + test_years) - timedelta(days=1), last)
        if oos_end < cursor:
            break
        train_start = prior_calendar_years(cursor, train_years)
        # Include the immediately preceding history for the 60-session feature warm-up.
        warmup_idx = max(0, dates.index(cursor) - 60) if cursor.isoformat() in dates else 0
        warmup_start = dates[warmup_idx]
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_start": train_start.isoformat(),
                "train_end": (cursor - timedelta(days=1)).isoformat(),
                "oos_start": cursor.isoformat(),
                "oos_end": oos_end.isoformat(),
                "warmup_start": warmup_start,
            }
        )
        cursor = cursor.replace(year=cursor.year + test_years)
    return folds


def filter_oos(payload: dict, oos_start: str, oos_end: str) -> pd.DataFrame:
    daily = pd.DataFrame(payload.get("daily", []))
    if daily.empty:
        return daily
    daily["date"] = daily["date"].astype(str)
    return daily[(daily["date"] >= oos_start) & (daily["date"] <= oos_end)].reset_index(drop=True)


def run_fold(fold: dict, top_n: int, cost_bps: float, slippage_bps: float) -> dict:
    payload = base.run_backtest(
        start=fold["warmup_start"],
        end=fold["oos_end"],
        top_n=top_n,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
    )
    oos = filter_oos(payload, fold["oos_start"], fold["oos_end"])
    if oos.empty:
        raise ValueError(f"fold {fold['fold']} has no OOS observations")

    overall = base.metrics(oos)
    annual = base.period_metrics(oos, "Y")
    rolling = base.rolling_252d_metrics(oos)
    return {
        **fold,
        "oos_trading_days": len(oos),
        "overall": overall,
        "annual": annual,
        "rolling_252d": rolling,
        "strategy_version": payload["strategy_version"],
        "strategy_commit": payload["strategy_commit"],
        "future_function": payload["future_function"],
    }


def validate(result: dict) -> None:
    if result["status"] != "ready":
        raise ValueError("walk-forward result is not ready")
    commits = {x["strategy_commit"] for x in result["folds"]}
    versions = {x["strategy_version"] for x in result["folds"]}
    if len(commits) != 1 or len(versions) != 1:
        raise ValueError("strategy version/commit changed across folds")
    if commits.pop() != result["strategy_commit"]:
        raise ValueError("top-level strategy commit mismatch")
    if any(not x["future_function"] for x in result["folds"]) is False:
        raise ValueError("PIT audit failed")
    for fold in result["folds"]:
        metrics = fold["overall"]
        for key in ("total_return_pct", "annualized_volatility_pct", "max_drawdown_pct", "win_rate_pct"):
            if not np.isfinite(float(metrics[key])):
                raise ValueError(f"non-finite OOS metric: {key}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run fixed-strategy walk-forward OOS analysis")
    p.add_argument("--start", default="2015-01-05")
    p.add_argument("--end", default="2026-09-29")
    p.add_argument("--train-years", type=int, default=3)
    p.add_argument("--test-years", type=int, default=1)
    p.add_argument("--top-n", type=int, default=30)
    p.add_argument("--cost-bps", type=float, default=3)
    p.add_argument("--slippage-bps", type=float, default=2)
    p.add_argument("--output", default=str(OUT_FILE))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dates = [x for x in trading_dates() if args.start <= x <= args.end]
    folds = build_folds(dates, args.train_years, args.test_years)
    if not folds:
        raise SystemExit("no complete walk-forward folds")

    # run_backtest writes latest.json on every fold; keep the final research
    # artifact separate and do not treat it as the production baseline.
    results = []
    for fold in folds:
        print(
            f"[walk-forward] fold={fold['fold']} "
            f"train={fold['train_start']}..{fold['train_end']} "
            f"oos={fold['oos_start']}..{fold['oos_end']}"
        )
        results.append(run_fold(fold, args.top_n, args.cost_bps, args.slippage_bps))

    result = {
        "schema_version": 1,
        "status": "ready",
        "method": "rolling_calendar_walk_forward_fixed_strategy",
        "train_years": args.train_years,
        "test_years": args.test_years,
        "top_n": args.top_n,
        "transaction_cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": results[0]["strategy_version"],
        "strategy_commit": results[0]["strategy_commit"],
        "start": args.start,
        "end": args.end,
        "fold_count": len(results),
        "folds": results,
        "audit": {
            "fixed_strategy_no_retraining": True,
            "oos_only_metrics": True,
            "60_session_warmup_before_oos": True,
            "current_universe_not_used_for_history": True,
            "future_adjusted_factor_not_used": True,
        },
    }
    validate(result)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"fold_count": len(results), "strategy_commit": result["strategy_commit"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
