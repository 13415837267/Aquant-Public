"""Research-only next-generation short-term signal study.

All formulas are fixed before evaluation. This script never modifies the
private production strategy. It uses the same T_close -> T+1_open contract,
the same main-board universe, and the same managed-trade model.

Research hypotheses are motivated by recent evidence on Chinese A-shares:
conditional reversal under high turnover/T+1 pressure, medium-horizon
overnight persistence, and regime dependence on cross-sectional dispersion.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import (
    FeatureState,
    history_files,
    managed_trade,
    read_daily,
)

TOP_N = (1, 2, 3)
GATE = {
    "forward_3d_mean_return_pct": 0.10,
    "forward_5d_mean_return_pct": 0.10,
    "forward_5d_positive_rate_pct": 50.0,
    "managed_trade_mean_return_pct": 0.10,
    "managed_trade_win_rate_pct": 45.0,
    "max_managed_trade_drawdown_pct": -40.0,
}

VARIANTS = {
    "t1_turnover_reversal": "High-turnover decline reversal with safety and liquidity controls.",
    "overnight_continuation": "Medium-horizon overnight persistence with extreme-jump penalty.",
    "dispersion_reversal": "Reversal-heavy regime under high cross-sectional dispersion.",
    "overnight_reversal_mix": "Balanced overnight persistence plus conditional T+1 reversal.",
    "event_defensive": "Momentum/pullback mix with stronger limit-up and chase penalties.",
    "hybrid_t1_regime": "Regime-conditioned combination of turnover reversal, overnight persistence, and trend.",
}


def numeric(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return (
        series.replace([np.inf, -np.inf], np.nan)
        .rank(pct=True, ascending=ascending, method="average")
        .fillna(0.5)
    )


def score_variants(frame: pd.DataFrame, dispersion_z: float) -> dict[str, pd.Series]:
    r1 = numeric(frame, "return_1d_pct")
    r3 = numeric(frame, "return_3d_pct")
    r5 = numeric(frame, "return_5d_pct")
    r10 = numeric(frame, "return_10d_pct")
    o1 = numeric(frame, "overnight_1d_pct")
    o3 = numeric(frame, "overnight_3d_pct")
    o5 = numeric(frame, "overnight_5d_pct")
    vol10 = numeric(frame, "volatility_10d_pct")
    volume_ratio = numeric(frame, "volume_ratio_5d")
    turnover = numeric(frame, "turnover_pct")
    strength = numeric(frame, "close_strength")
    intraday = numeric(frame, "intraday_return_pct")
    limit_count = numeric(frame, "limit_up_5d_count")

    r1_rank = rank(r1)
    r3_rank = rank(r3)
    r5_rank = rank(r5)
    trend = 0.25 * rank(r10) + 0.25 * r5_rank + 0.20 * r3_rank + 0.30 * r1_rank
    reversal = 0.55 * rank(-r1) + 0.30 * rank(-r3) + 0.15 * rank(-r5)
    high_turnover = 0.60 * rank(turnover) + 0.40 * rank(volume_ratio)
    turnover_reversal = (
        0.45 * rank(-r1)
        + 0.25 * rank(-r3)
        + 0.15 * high_turnover
        + 0.15 * rank(-r5)
    )
    overnight = 0.20 * rank(o1) + 0.40 * rank(o3) + 0.40 * rank(o5)
    overnight_abs = rank(o1.abs(), ascending=False)
    intraday_reversal = rank(-intraday)
    safety = rank(vol10, ascending=False)
    liquidity = rank(np.log1p(numeric(frame, "amount_20d").clip(lower=0)))
    price = 0.70 * rank(strength) + 0.30 * rank(numeric(frame, "change_pct"))

    reversal_component = (
        0.55 * turnover_reversal
        + 0.15 * intraday_reversal
        + 0.10 * safety
        + 0.10 * liquidity
        + 0.10 * overnight
    )

    base_defensive = (
        0.23 * trend
        + 0.17 * overnight
        + 0.16 * high_turnover
        + 0.12 * price
        + 0.18 * liquidity
        + 0.14 * safety
    )

    t1_reversal = (
        0.38 * turnover_reversal
        + 0.17 * overnight
        + 0.12 * price
        + 0.15 * liquidity
        + 0.18 * safety
    )

    overnight_cont = (
        0.28 * trend
        + 0.28 * overnight
        + 0.14 * price
        + 0.14 * high_turnover
        + 0.16 * safety
        - 0.07 * overnight_abs
    )

    dispersion_rev = (
        0.18 * trend
        + 0.12 * overnight
        + 0.28 * reversal
        + 0.12 * intraday_reversal
        + 0.12 * high_turnover
        + 0.18 * safety
    )

    overnight_mix = (
        0.24 * trend
        + 0.24 * overnight
        + 0.20 * turnover_reversal
        + 0.10 * intraday_reversal
        + 0.10 * liquidity
        + 0.12 * safety
    )

    event_defensive = (
        base_defensive
        - 0.06 * rank(limit_count, ascending=False)
        - 0.04 * rank(intraday.clip(lower=0), ascending=False)
    )

    trend_regime = base_defensive
    high_dispersion_regime = (
        0.20 * trend
        + 0.10 * overnight
        + 0.34 * turnover_reversal
        + 0.14 * intraday_reversal
        + 0.10 * liquidity
        + 0.12 * safety
    )
    hybrid = high_dispersion_regime if dispersion_z >= 0.75 else trend_regime

    return {
        "t1_turnover_reversal": (100 * t1_reversal).clip(0, 100),
        "overnight_continuation": (100 * overnight_cont).clip(0, 100),
        "dispersion_reversal": (100 * dispersion_rev).clip(0, 100),
        "overnight_reversal_mix": (100 * overnight_mix).clip(0, 100),
        "event_defensive": (100 * event_defensive).clip(0, 100),
        "hybrid_t1_regime": (100 * hybrid).clip(0, 100),
    }


def forward_return(future_days: list[pd.DataFrame], symbol: str, horizon: int):
    if len(future_days) < horizon:
        return None
    entry_row = future_days[0].loc[future_days[0]["symbol"].eq(symbol)]
    target_row = future_days[horizon - 1].loc[future_days[horizon - 1]["symbol"].eq(symbol)]
    if entry_row.empty or target_row.empty:
        return None
    entry = pd.to_numeric(entry_row.iloc[0].get("open"), errors="coerce")
    close = pd.to_numeric(target_row.iloc[0].get("close"), errors="coerce")
    if pd.isna(entry) or pd.isna(close) or entry <= 0:
        return None
    return float((close / entry - 1.0) * 100.0)


def trade_stats(rows, cost_bps, slippage_bps):
    if not rows:
        return {"samples": 0, "win_rate_pct": None, "mean_return_pct": None,
                "median_return_pct": None, "max_drawdown_pct": None,
                "mean_holding_days": None}
    df = pd.DataFrame(rows)
    gross = pd.to_numeric(df["gross_return_pct"], errors="coerce")
    net = gross - 2 * (cost_bps + slippage_bps) / 100.0
    equity = (1 + net / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "samples": int(net.notna().sum()),
        "win_rate_pct": float((net > 0).mean() * 100.0),
        "mean_return_pct": float(net.mean()),
        "median_return_pct": float(net.median()),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "mean_holding_days": float(pd.to_numeric(df["holding_days"], errors="coerce").mean()),
    }


def forward_stats(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"samples": 0, "mean_return_pct": None, "median_return_pct": None, "positive_rate_pct": None}
    return {
        "samples": int(len(arr)),
        "mean_return_pct": float(arr.mean()),
        "median_return_pct": float(np.median(arr)),
        "positive_rate_pct": float((arr > 0).mean() * 100.0),
    }


def summarize(bucket, cost_bps, slippage_bps):
    out = {
        "signal_days": bucket["signal_days"],
        "candidate_days": bucket["candidate_days"],
        "candidate_day_rate_pct": bucket["candidate_days"] / bucket["signal_days"] * 100.0 if bucket["signal_days"] else 0.0,
        "top_n": {},
    }
    for n in TOP_N:
        row = {}
        for h in (1, 3, 5):
            row[f"forward_{h}d"] = forward_stats(bucket["forward"][h][n])
        row["managed_trade"] = trade_stats(bucket["trades"][n], cost_bps, slippage_bps)
        out["top_n"][str(n)] = row
    return out


def gate_passes(summary, n):
    row = summary["top_n"][str(n)]
    f3, f5, tr = row["forward_3d"], row["forward_5d"], row["managed_trade"]
    checks = [
        f3["mean_return_pct"] is not None,
        f5["mean_return_pct"] is not None,
        f5["positive_rate_pct"] is not None,
        tr["mean_return_pct"] is not None,
        tr["win_rate_pct"] is not None,
        tr["max_drawdown_pct"] is not None,
        f3.get("mean_return_pct", -999) >= GATE["forward_3d_mean_return_pct"],
        f5.get("mean_return_pct", -999) >= GATE["forward_5d_mean_return_pct"],
        f5.get("positive_rate_pct", -999) >= GATE["forward_5d_positive_rate_pct"],
        tr.get("mean_return_pct", -999) >= GATE["managed_trade_mean_return_pct"],
        tr.get("win_rate_pct", -999) >= GATE["managed_trade_win_rate_pct"],
        tr.get("max_drawdown_pct", -999) >= GATE["max_managed_trade_drawdown_pct"],
    ]
    return bool(all(checks))


def run(args):
    files = history_files()
    dates = [p.name[:10] for p in files]
    bounds = [args.development_start, args.development_end, args.validation_start,
              args.validation_end, args.final_start, args.final_end]
    if any(x not in dates for x in bounds):
        raise ValueError("all research boundaries must be trading dates")
    windows = {
        "development": (dates.index(args.development_start), dates.index(args.development_end)),
        "validation": (dates.index(args.validation_start), dates.index(args.validation_end)),
        "final_holdout": (dates.index(args.final_start), dates.index(args.final_end)),
    }
    begin = max(0, windows["development"][0] - 20)
    end_i = windows["final_holdout"][1]
    active = files[begin:end_i + 6]
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(active[i])
        return cache[i]

    state = FeatureState()
    dispersion_history = deque(maxlen=60)
    buckets = {
        name: {
            "signal_days": 0, "candidate_days": 0,
            "forward": {h: {n: [] for n in TOP_N} for h in (1, 3, 5)},
            "trades": {n: [] for n in TOP_N},
        }
        for name in VARIANTS
    }

    for i in range(len(active) - 5):
        signal_date = active[i].name[:10]
        frame = state.build(get(i))
        if frame.empty:
            continue

        current_dispersion = float(numeric(frame, "change_pct").std(ddof=1))
        if len(dispersion_history) >= 20:
            hist = np.asarray(dispersion_history, dtype=float)
            hist_std = float(hist.std(ddof=1))
            dispersion_z = (current_dispersion - float(hist.mean())) / hist_std if hist_std > 1e-9 else 0.0
        else:
            dispersion_z = 0.0
        dispersion_history.append(current_dispersion)

        period = None
        absolute_i = begin + i
        for name, (s, e) in windows.items():
            if s <= absolute_i <= e - 5:
                period = name
                break
        if period is None:
            continue

        future = [get(i + j) for j in range(1, 6)]
        scores = score_variants(frame, dispersion_z)
        for name, score in scores.items():
            bucket = buckets[name][period] if period in buckets[name] else None
            if bucket is None:
                continue
            tmp = frame.copy()
            tmp["_score"] = score
            breadth = float(tmp["market_breadth_pct"].iloc[0])
            median_ret = float(tmp["market_median_return_pct"].iloc[0])
            bucket["signal_days"] += 1
            if breadth < 35.0 or median_ret < -0.30:
                continue

            selected = tmp.loc[tmp["_score"].notna() & (tmp["_score"] >= 70.0)].sort_values(
                ["_score", "volume_ratio_5d", "amount", "symbol"],
                ascending=[False, False, False, True],
                kind="mergesort",
            ).head(3)
            if selected.empty:
                continue

            bucket["candidate_days"] += 1
            for n in TOP_N:
                for symbol in selected.head(n)["symbol"].astype(str).str.zfill(6):
                    for h in (1, 3, 5):
                        value = forward_return(future, symbol, h)
                        if value is not None:
                            bucket["forward"][h][n].append(value)
                    trade = managed_trade(symbol, future)
                    if trade:
                        bucket["trades"][n].append(trade)

    report = []
    for name in VARIANTS:
        item = {
            "variant": name,
            "description": VARIANTS[name],
            "development": summarize(buckets[name]["development"], args.cost_bps, args.slippage_bps),
            "validation": summarize(buckets[name]["validation"], args.cost_bps, args.slippage_bps),
            "final_holdout": summarize(buckets[name]["final_holdout"], args.cost_bps, args.slippage_bps),
        }
        item["gate"] = {
            str(n): {
                "development": gate_passes(item["development"], n),
                "validation": gate_passes(item["validation"], n),
                "final_holdout": gate_passes(item["final_holdout"], n),
            }
            for n in TOP_N
        }
        report.append(item)

    out = {
        "schema_version": 1,
        "status": "ready",
        "method": "short_term_nextgen_research",
        "future_function": False,
        "entry": "T+1_open",
        "max_holding_sessions": 5,
        "development": ["2018-01-05", "2024-12-20"],
        "validation": ["2025-01-02", "2025-12-31"],
        "final_holdout": ["2026-01-05", "2026-09-21"],
        "threshold_policy": {"market_breadth_min_pct": 35.0, "market_median_min_pct": -0.30, "score_min": 70.0, "max_candidates": 3},
        "production_gate": GATE,
        "selection_rule": "All formulas, score floor, market gates, and top-N evaluations are fixed before final_holdout evaluation. final_holdout is not used for model selection.",
        "variants": report,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready", "variants": len(report), "output": str(path)}, ensure_ascii=False))


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
    ap.add_argument("--output", default=str(ROOT / "data/backtest/short_term_nextgen_research.json"))
    run(ap.parse_args())
