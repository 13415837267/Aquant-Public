"""Walk-forward / out-of-sample evaluation for the current Private strategy.

The current Private model is not trained by this script. Each fold therefore
measures genuine out-of-sample calendar performance of the current strategy
after a rolling historical window, while preserving the 126-session feature
warm-up before the OOS start.
"""
from __future__ import annotations

import argparse
import json
import sys
from bisect import bisect_left, bisect_right
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    if train_years <= 0 or test_years <= 0:
        raise ValueError("train_years and test_years must be positive")

    date_values = [date.fromisoformat(x) for x in dates]
    first = date_values[0]
    last = date_values[-1]

    def first_on_or_after(target: date) -> int | None:
        idx = bisect_left(date_values, target)
        return idx if idx < len(date_values) else None

    def last_on_or_before(target: date) -> int | None:
        idx = bisect_right(date_values, target) - 1
        return idx if idx >= 0 else None

    folds = []
    cursor_calendar = first.replace(year=first.year + train_years)

    while cursor_calendar <= last:
        oos_start_idx = first_on_or_after(cursor_calendar)
        if oos_start_idx is None:
            break

        oos_start = date_values[oos_start_idx]
        oos_end_calendar = cursor_calendar.replace(
            year=cursor_calendar.year + test_years
        ) - timedelta(days=1)
        oos_end_idx = last_on_or_before(oos_end_calendar)
        if oos_end_idx is None or oos_end_idx < oos_start_idx:
            break

        oos_end = date_values[oos_end_idx]
        train_start = prior_calendar_years(oos_start, train_years)
        warmup_idx = max(0, oos_start_idx - 126)

        folds.append(
            {
                "fold": len(folds) + 1,
                "train_start": train_start.isoformat(),
                "train_end": (oos_start - timedelta(days=1)).isoformat(),
                "oos_start": oos_start.isoformat(),
                "oos_end": oos_end.isoformat(),
                "warmup_start": dates[warmup_idx],
            }
        )

        cursor_calendar = cursor_calendar.replace(
            year=cursor_calendar.year + test_years
        )

    return folds


def filter_oos(payload: dict, oos_start: str, oos_end: str) -> pd.DataFrame:
    daily = pd.DataFrame(payload.get("daily", []))
    if daily.empty:
        return daily
    daily["date"] = daily["date"].astype(str)
    return daily[(daily["date"] >= oos_start) & (daily["date"] <= oos_end)].reset_index(drop=True)


def run_fold(fold: dict, payload: dict) -> dict:
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
    if not all(x["future_function"] is False for x in result["folds"]):
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
    p.add_argument("--top-n", type=int, default=20)
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
    full_payload = base.run_backtest(
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
    )
    results = []
    aggregate_frames = []
    for fold in folds:
        print(
            f"[walk-forward] fold={fold['fold']} "
            f"train={fold['train_start']}..{fold['train_end']} "
            f"oos={fold['oos_start']}..{fold['oos_end']}"
        )
        results.append(run_fold(fold, full_payload))
        aggregate_frames.append(filter_oos(full_payload, fold["oos_start"], fold["oos_end"]))

    aggregate_oos = pd.concat(aggregate_frames, ignore_index=True)
    aggregate_metrics = base.metrics(aggregate_oos)
    result = {
        "schema_version": 1,
        "status": "ready",
        "future_function": False,
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
        "aggregate_oos": {
            **aggregate_metrics,
            "start_date": aggregate_oos["date"].min() if not aggregate_oos.empty else None,
            "end_date": aggregate_oos["date"].max() if not aggregate_oos.empty else None,
            "positive_fold_count": sum(
                1 for x in results if float(x["overall"]["total_return_pct"]) > 0
            ),
            "negative_fold_count": sum(
                1 for x in results if float(x["overall"]["total_return_pct"]) <= 0
            ),
        },
        "folds": results,
        "audit": {
            "fixed_strategy_no_retraining": True,
            "oos_only_metrics": True,
            "126_session_warmup_before_oos": True,
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
