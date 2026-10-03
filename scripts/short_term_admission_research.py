"""Walk-forward research of stricter short-term candidate admission policies.

The private 2.0.1 scorer is loaded from AQUANT_PRIVATE_STRATEGY_PATH and is
never modified. Only admission rules are varied. OOS is held out from policy
selection and checked against the same production gate used by system_audit.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, managed_trade, read_daily

POLICIES = {
    "current": {
        "description": "existing production admission",
        "min_breadth_pct": 35.0, "min_market_median_pct": -0.30, "min_score": 62.0,
    },
    "strict_market": {
        "description": "stronger breadth and market median gate",
        "min_breadth_pct": 45.0, "min_market_median_pct": 0.0, "min_score": 70.0,
    },
    "risk_on": {
        "description": "risk-on breadth plus positive median return",
        "min_breadth_pct": 55.0, "min_market_median_pct": 0.20, "min_score": 70.0,
    },
    "high_score": {
        "description": "current market gate with higher score floor",
        "min_breadth_pct": 35.0, "min_market_median_pct": -0.30, "min_score": 80.0,
    },
    "high_score_strict_market": {
        "description": "higher score floor plus stronger market gate",
        "min_breadth_pct": 45.0, "min_market_median_pct": 0.0, "min_score": 80.0,
    },
    "score90": {
        "description": "current market gate with very high score floor",
        "min_breadth_pct": 35.0, "min_market_median_pct": -0.30, "min_score": 90.0,
    },
    "anti_chase": {
        "description": "higher score floor plus anti-chase filter",
        "min_breadth_pct": 35.0, "min_market_median_pct": -0.30, "min_score": 70.0,
        "max_change_pct": 7.0, "max_return_5d_pct": 15.0,
    },
    "pullback_selective": {
        "description": "strong market plus tighter pullback filter",
        "min_breadth_pct": 45.0, "min_market_median_pct": 0.0, "min_score": 70.0,
        "max_change_pct": 5.0, "max_return_5d_pct": 10.0,
    },
}

GATE = {
    "forward_3d_mean_return_pct": 0.10,
    "forward_5d_mean_return_pct": 0.10,
    "forward_5d_positive_rate_pct": 50.0,
    "managed_trade_mean_return_pct": 0.10,
    "managed_trade_win_rate_pct": 45.0,
    "max_managed_trade_drawdown_pct": -40.0,
}


def load_model():
    root = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", "").strip()
    if not root:
        raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required")
    sys.path.insert(0, str(Path(root).resolve()))
    return importlib.import_module("strategy.model")


def stats(rows, cost_bps, slippage_bps):
    if not rows:
        return {"samples": 0, "win_rate_pct": None, "mean_return_pct": None,
                "median_return_pct": None, "max_drawdown_pct": None,
                "mean_holding_days": None}
    df = pd.DataFrame(rows)
    gross = pd.to_numeric(df["gross_return_pct"], errors="coerce")
    net = gross - 2 * (cost_bps + slippage_bps) / 100.0
    equity = (1.0 + net / 100.0).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "samples": int(len(net)),
        "win_rate_pct": float((net > 0).mean() * 100.0),
        "mean_return_pct": float(net.mean()),
        "median_return_pct": float(net.median()),
        "max_drawdown_pct": float(dd.min() * 100.0),
        "mean_holding_days": float(pd.to_numeric(df["holding_days"]).mean()),
    }


def forward_return(future_days, symbol, horizon):
    row = future_days[0].loc[future_days[0]["symbol"].eq(symbol)]
    if row.empty or pd.isna(row.iloc[0].get("open")):
        return None
    entry = float(row.iloc[0]["open"])
    idx = horizon - 1
    if idx >= len(future_days):
        return None
    target = future_days[idx].loc[future_days[idx]["symbol"].eq(symbol)]
    if target.empty or pd.isna(target.iloc[0].get("close")):
        return None
    close = float(target.iloc[0]["close"])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(close):
        return None
    return (close / entry - 1.0) * 100.0


def run(args):
    model = load_model()
    files = history_files()
    dates = [p.name[:10] for p in files]
    for d in (args.start, args.train_end, args.test_start, args.test_end):
        if d not in dates:
            raise ValueError(f"{d} must be a trading date")
    start_i = dates.index(args.start)
    train_end_i = dates.index(args.train_end)
    test_start_i = dates.index(args.test_start)
    test_end_i = dates.index(args.test_end)
    if test_start_i <= train_end_i:
        raise ValueError("test period must start after train period")
    max_future = 5
    if test_end_i + max_future >= len(files):
        raise ValueError("test_end needs five future trading sessions")

    begin = max(0, start_i - 20)
    active = files[begin:test_end_i + max_future + 1]
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(active[i])
        return cache[i]

    state = FeatureState()
    buckets = {
        policy: {
            "train": {"signal_days": 0, "candidate_days": 0, "forward_3d": [], "forward_5d": [], "trades": []},
            "oos": {"signal_days": 0, "candidate_days": 0, "forward_3d": [], "forward_5d": [], "trades": []},
        }
        for policy in POLICIES
    }
    rel_train_end = train_end_i - begin

    for i in range(len(active) - max_future):
        signal_date = active[i].name[:10]
        # Always advance the feature state through every trading day so the
        # OOS window retains the full uninterrupted lookback history.
        frame = state.build(get(i))

        period = None
        if args.start <= signal_date <= args.train_end and i + max_future <= rel_train_end:
            period = "train"
        elif args.test_start <= signal_date <= args.test_end:
            period = "oos"
        if period is None:
            continue

        for policy_name, bucket_set in buckets.items():
            bucket_set[period]["signal_days"] += 1

        if frame.empty:
            continue

        scored = model.score_universe(frame)
        future = [get(i + j) for j in range(1, max_future + 1)]
        breadth = float(scored["market_breadth_pct"].iloc[0])
        median_ret = float(scored["market_median_return_pct"].iloc[0])

        for policy_name, policy in POLICIES.items():
            data = buckets[policy_name][period]
            keep = (
                scored["score"].notna()
                & scored["score"].ge(policy["min_score"])
                & (breadth >= policy["min_breadth_pct"])
                & (median_ret >= policy["min_market_median_pct"])
            )
            if "max_change_pct" in policy:
                keep &= pd.to_numeric(scored["change_pct"], errors="coerce").le(policy["max_change_pct"])
            if "max_return_5d_pct" in policy:
                keep &= pd.to_numeric(scored["return_5d_pct"], errors="coerce").le(policy["max_return_5d_pct"])

            selected = scored.loc[keep].head(3)
            if selected.empty:
                continue

            data["candidate_days"] += 1
            for symbol in selected["symbol"].astype(str).str.zfill(6):
                r3 = forward_return(future, symbol, 3)
                r5 = forward_return(future, symbol, 5)
                if r3 is not None: data["forward_3d"].append(r3)
                if r5 is not None: data["forward_5d"].append(r5)
                trade = managed_trade(symbol, future)
                if trade:
                    trade.update({"signal_date": signal_date, "symbol": symbol})
                    data["trades"].append(trade)

    report = []
    for policy_name, policy in POLICIES.items():
        item = {"policy": policy_name, "description": policy["description"], "parameters": policy}
        for period in ("train", "oos"):
            data = buckets[policy_name][period]
            r3 = np.asarray(data["forward_3d"], dtype=float)
            r5 = np.asarray(data["forward_5d"], dtype=float)
            trade = stats(data["trades"], args.cost_bps, args.slippage_bps)
            out = {
                "signal_days": data["signal_days"],
                "candidate_days": data["candidate_days"],
                "candidate_day_rate_pct": (
                    data["candidate_days"] / data["signal_days"] * 100.0
                    if data["signal_days"] else 0.0
                ),
                "forward_3d": {
                    "samples": int(len(r3)),
                    "mean_return_pct": float(r3.mean()) if len(r3) else None,
                    "positive_rate_pct": float((r3 > 0).mean() * 100.0) if len(r3) else None,
                },
                "forward_5d": {
                    "samples": int(len(r5)),
                    "mean_return_pct": float(r5.mean()) if len(r5) else None,
                    "positive_rate_pct": float((r5 > 0).mean() * 100.0) if len(r5) else None,
                },
                "managed_trade": trade,
            }
            out["production_gate_passed"] = bool(
                out["forward_3d"]["mean_return_pct"] is not None
                and out["forward_5d"]["mean_return_pct"] is not None
                and trade["mean_return_pct"] is not None
                and trade["win_rate_pct"] is not None
                and trade["max_drawdown_pct"] is not None
                and out["forward_3d"]["mean_return_pct"] >= GATE["forward_3d_mean_return_pct"]
                and out["forward_5d"]["mean_return_pct"] >= GATE["forward_5d_mean_return_pct"]
                and out["forward_5d"]["positive_rate_pct"] >= GATE["forward_5d_positive_rate_pct"]
                and trade["mean_return_pct"] >= GATE["managed_trade_mean_return_pct"]
                and trade["win_rate_pct"] >= GATE["managed_trade_win_rate_pct"]
                and trade["max_drawdown_pct"] >= GATE["max_managed_trade_drawdown_pct"]
            )
            item[period] = out
        report.append(item)

    out = {
        "schema_version": 1,
        "status": "ready",
        "method": "short_term_admission_selectivity_walk_forward",
        "start": args.start, "train_end": args.train_end,
        "test_start": args.test_start, "test_end": args.test_end,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": args.strategy_version,
        "strategy_commit": args.strategy_commit,
        "future_function": False,
        "entry": "T+1_open",
        "max_holding_sessions": 5,
        "cost_bps": args.cost_bps, "slippage_bps": args.slippage_bps,
        "production_gate": GATE,
        "policies": report,
        "selection_rule": "OOS is a holdout diagnostic; no OOS result is used in policy selection",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ready", "policies": len(report),
        "oos_passes": sum(1 for r in report if r["oos"]["production_gate_passed"]),
        "output": str(path)
    }, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-05")
    ap.add_argument("--train-end", default="2024-12-20")
    ap.add_argument("--test-start", default="2025-01-02")
    ap.add_argument("--test-end", default="2026-09-21")
    ap.add_argument("--strategy-version", required=True)
    ap.add_argument("--strategy-commit", required=True)
    ap.add_argument("--cost-bps", type=float, default=3.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--output", default=str(ROOT / "data/backtest/short_term_admission_research.json"))
    run(ap.parse_args())
