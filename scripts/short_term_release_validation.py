"""Release validation with an untouched final 2026 holdout.

The strategy is loaded from Aquant-Private/main. No parameter selection is
performed here. The final_holdout period is evaluated only after the strategy
version is fixed in the private repository.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, managed_trade, read_daily

GATE = {
    "forward_3d_mean_return_pct": 0.10,
    "forward_5d_mean_return_pct": 0.10,
    "forward_5d_positive_rate_pct": 50.0,
    "managed_trade_mean_return_pct": 0.10,
    "managed_trade_win_rate_pct": 45.0,
    "max_managed_trade_drawdown_pct": -40.0,
}


def load_strategy():
    root = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", "").strip()
    if not root:
        raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required")
    root = str(Path(root).resolve())
    sys.path.insert(0, root)
    model = importlib.import_module("strategy.model")
    version = importlib.import_module("strategy.version")
    commit = os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT", "").strip()
    if not commit:
        commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    return model, str(version.STRATEGY_VERSION), commit


def forward_return(future_days, symbol, horizon):
    if not future_days:
        return None
    entry_row = future_days[0].loc[future_days[0]["symbol"].eq(symbol)]
    if entry_row.empty or pd.isna(entry_row.iloc[0].get("open")):
        return None
    entry = float(entry_row.iloc[0]["open"])
    target_index = horizon - 1
    if target_index >= len(future_days):
        return None
    target_row = future_days[target_index].loc[future_days[target_index]["symbol"].eq(symbol)]
    if target_row.empty or pd.isna(target_row.iloc[0].get("close")):
        return None
    close = float(target_row.iloc[0]["close"])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(close):
        return None
    return (close / entry - 1.0) * 100.0


def summarize(data, cost_bps, slippage_bps):
    r3 = np.asarray(data["forward_3d"], dtype=float)
    r5 = np.asarray(data["forward_5d"], dtype=float)
    trades = data["trades"]
    gross = pd.to_numeric(pd.Series([x["gross_return_pct"] for x in trades]), errors="coerce")
    net = gross - 2 * (cost_bps + slippage_bps) / 100.0 if len(gross) else gross
    if len(net):
        eq = (1.0 + net / 100.0).cumprod()
        dd = eq / eq.cummax() - 1.0
        trade_stats = {
            "samples": int(len(net)),
            "win_rate_pct": float((net > 0).mean() * 100.0),
            "mean_return_pct": float(net.mean()),
            "median_return_pct": float(net.median()),
            "max_drawdown_pct": float(dd.min() * 100.0),
            "mean_holding_days": float(pd.to_numeric(pd.Series([x["holding_days"] for x in trades])).mean()),
        }
    else:
        trade_stats = {
            "samples": 0, "win_rate_pct": None, "mean_return_pct": None,
            "median_return_pct": None, "max_drawdown_pct": None, "mean_holding_days": None,
        }
    result = {
        "signal_days": int(data["signal_days"]),
        "candidate_days": int(data["candidate_days"]),
        "candidate_day_rate_pct": float(data["candidate_days"] / data["signal_days"] * 100.0) if data["signal_days"] else 0.0,
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
        "managed_trade": trade_stats,
    }
    result["production_gate_passed"] = bool(
        result["forward_3d"]["mean_return_pct"] is not None
        and result["forward_5d"]["mean_return_pct"] is not None
        and trade_stats["mean_return_pct"] is not None
        and trade_stats["win_rate_pct"] is not None
        and trade_stats["max_drawdown_pct"] is not None
        and result["forward_3d"]["mean_return_pct"] >= GATE["forward_3d_mean_return_pct"]
        and result["forward_5d"]["mean_return_pct"] >= GATE["forward_5d_mean_return_pct"]
        and result["forward_5d"]["positive_rate_pct"] >= GATE["forward_5d_positive_rate_pct"]
        and trade_stats["mean_return_pct"] >= GATE["managed_trade_mean_return_pct"]
        and trade_stats["win_rate_pct"] >= GATE["managed_trade_win_rate_pct"]
        and trade_stats["max_drawdown_pct"] >= GATE["max_managed_trade_drawdown_pct"]
    )
    return result


def run(args):
    model, version, commit = load_strategy()
    files = history_files()
    dates = [p.name[:10] for p in files]
    windows = {
        "development": (args.development_start, args.development_end),
        "validation": (args.validation_start, args.validation_end),
        "final_holdout": (args.final_start, args.final_end),
    }
    indices = {}
    for name, (start, end) in windows.items():
        if start not in dates or end not in dates:
            raise ValueError(f"{name} dates must be trading dates")
        indices[name] = (dates.index(start), dates.index(end))
        if indices[name][1] + 5 >= len(files):
            raise ValueError(f"{name} end needs five future sessions")

    begin = max(0, indices["development"][0] - 20)
    end_i = indices["final_holdout"][1]
    active = files[begin : end_i + 6]
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(active[i])
        return cache[i]

    state = FeatureState()
    buckets = {name: {"signal_days": 0, "candidate_days": 0, "forward_3d": [], "forward_5d": [], "trades": []} for name in windows}

    for i in range(len(active) - 5):
        signal_date = active[i].name[:10]
        frame = state.build(get(i))
        period = None
        for name, (start, end) in windows.items():
            if start <= signal_date <= end:
                # Require all five future sessions to remain inside the same window.
                end_rel = indices[name][1] - begin
                if i + 5 <= end_rel:
                    period = name
                break
        if period is None or frame.empty:
            continue

        scored = model.score_universe(frame)
        selected = model.admit_candidates(scored)
        bucket = buckets[period]
        bucket["signal_days"] += 1
        if selected.empty:
            continue

        bucket["candidate_days"] += 1
        future = [get(i + j) for j in range(1, 6)]
        for symbol in selected["symbol"].astype(str).str.zfill(6):
            r3 = forward_return(future, symbol, 3)
            r5 = forward_return(future, symbol, 5)
            if r3 is not None:
                bucket["forward_3d"].append(r3)
            if r5 is not None:
                bucket["forward_5d"].append(r5)
            trade = managed_trade(symbol, future)
            if trade:
                bucket["trades"].append(trade)

    output = {
        "schema_version": 1,
        "status": "ready",
        "method": "short_term_release_validation",
        "strategy_source": "Aquant-Private/main",
        "strategy_version": version,
        "strategy_commit": commit,
        "future_function": False,
        "entry": "T+1_open",
        "max_holding_sessions": 5,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "production_gate": GATE,
        "selection_rule": "strategy version and admission parameters are fixed before final_holdout evaluation",
    }
    for name in windows:
        output[name] = summarize(buckets[name], args.cost_bps, args.slippage_bps)
    output["release_gate"] = output["final_holdout"]["production_gate_passed"]
    output["release_gate_scope"] = "final_holdout_only"
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ready",
        "strategy_version": version,
        "strategy_commit": commit,
        "release_gate": output["release_gate"],
        "output": str(path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-start", default="2018-01-05")
    ap.add_argument("--development-end", default="2024-12-20")
    ap.add_argument("--validation-start", default="2025-01-02")
    ap.add_argument("--validation-end", default="2025-12-31")
    ap.add_argument("--final-start", default="2026-01-05")
    ap.add_argument("--final-end", default="2026-09-21")
    ap.add_argument("--cost-bps", type=float, default=3.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--output", default=str(ROOT / "data/backtest/short_term_release_validation.json"))
    run(ap.parse_args())
