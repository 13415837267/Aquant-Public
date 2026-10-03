"""Research-only factor ablation for the frozen Private strategy.

The Private strategy remains the production source; this module only tests nearby weight variants. This module reconstructs the canonical
factor score from the Private model outputs, then evaluates research-only
weight variants under the same PIT data and T+1 execution model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backtest as base

EXPECTED_WEIGHTS = {
    "momentum": 0.35,
    "liquidity": 0.15,
    "value": 0.30,
    "safety": 0.20,
}
VARIANTS = ["baseline", "drop_momentum", "drop_liquidity", "drop_value",
            "drop_safety", "equal_weight"]


def variant_weights(name: str, base_weights: dict[str, float]) -> dict[str, float]:
    if name == "baseline":
        return dict(base_weights)
    if name == "equal_weight":
        return {k: 1.0 / len(base_weights) for k in base_weights}
    if not name.startswith("drop_"):
        raise ValueError(f"unknown variant: {name}")
    drop = name.removeprefix("drop_")
    if drop not in base_weights:
        raise ValueError(f"unknown factor: {drop}")
    remaining = 1.0 - base_weights[drop]
    return {k: v / remaining for k, v in base_weights.items() if k != drop}


def score_variant(
    scored: pd.DataFrame,
    weights: dict[str, float],
    penalty: pd.Series,
) -> pd.DataFrame:
    out = scored.copy()
    weighted = sum(
        float(weights[key]) * pd.to_numeric(out[f"f_{key}"], errors="coerce").fillna(0.5)
        for key in weights
    ) * 100.0
    out["score"] = (weighted + penalty).clip(0.0, 100.0)
    return out.sort_values(
        ["score", "amount", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def evaluate(
    *,
    start: str,
    end: str,
    top_n: int,
    cost_bps: float,
    slippage_bps: float,
) -> dict:
    strategy_model, strategy_version, strategy_commit = base.load_strategy()
    base_weights = dict(getattr(strategy_model, "WEIGHTS", {}))
    if base_weights != EXPECTED_WEIGHTS:
        raise ValueError(f"unexpected strategy weights: {base_weights}")

    files = base.history_files()
    selected_files, _ = base.iter_selected_dates(files, start, end)
    if len(selected_files) < 2:
        raise ValueError("factor ablation needs at least two historical sessions")

    state = base.RollingFeatureState()
    current = base.read_daily(selected_files[0])
    variant_daily: dict[str, list[dict]] = {name: [] for name in VARIANTS}
    prev_target: dict[str, dict[str, float]] = {name: {} for name in VARIANTS}
    current_scored_rows = 0

    for i in range(len(selected_files) - 1):
        next_day = base.read_daily(selected_files[i + 1])
        signal_date = current["date"].iloc[0]
        frame = base.build_strategy_frame(current, state)

        if frame.empty:
            scored_base = frame.copy()
            penalty = pd.Series(index=frame.index, dtype=float)
        else:
            scored_base = strategy_model.score_universe(frame).copy()
            raw_score = 100.0 * sum(
                float(base_weights[key])
                * pd.to_numeric(scored_base[f"f_{key}"], errors="coerce").fillna(0.5)
                for key in base_weights
            )
            penalty = pd.to_numeric(scored_base["score"], errors="coerce") - raw_score
            current_scored_rows += int(len(scored_base))

        for name in VARIANTS:
            weights = variant_weights(name, base_weights)
            targets = (
                score_variant(scored_base, weights, penalty).head(top_n).reset_index(drop=True)
                if not frame.empty
                else frame
            )
            symbols = targets["symbol"].astype(str).str.zfill(6).tolist() if not targets.empty else []
            target_weights = base.normalize_weights(symbols)
            gross_return, executed, missing = base.next_session_return(
                targets, next_day, target_weights
            )
            actual_target = {
                symbol: target_weights[symbol]
                for symbol in executed
                if symbol in target_weights
            }
            turn = base.turnover(prev_target[name], actual_target)
            total_cost = turn * (cost_bps + slippage_bps) / 10000.0
            net_return = gross_return - total_cost
            variant_daily[name].append(
                {
                    "date": next_day["date"].iloc[0],
                    "signal_date": signal_date,
                    "gross_return": float(gross_return),
                    "turnover": float(turn),
                    "net_return": float(net_return),
                    "target_count": int(len(targets)),
                    "executed_count": int(len(executed)),
                    "missing_execution_count": int(len(missing)),
                }
            )
            prev_target[name] = actual_target
        current = next_day

    results = {}
    for name, rows in variant_daily.items():
        daily = pd.DataFrame(rows)
        if daily.empty:
            raise ValueError(f"variant has no rows: {name}")
        active = daily.index[daily["target_count"] > 0].tolist()
        performance = daily.iloc[int(active[0]):].reset_index(drop=True) if active else daily
        m = base.metrics(performance)
        results[name] = {
            "variant": name,
            "weights": variant_weights(name, base_weights),
            **{k: m[k] for k in (
                "trading_days", "total_return_pct", "annualized_return_pct",
                "annualized_volatility_pct", "sharpe", "max_drawdown_pct",
                "win_rate_pct", "average_turnover_pct", "total_turnover_pct",
            )},
            "strategy_version": strategy_version,
            "strategy_commit": strategy_commit,
            "future_function": False,
        }

    baseline = results["baseline"]
    for item in results.values():
        item["delta_vs_baseline"] = {
            "total_return_pct": item["total_return_pct"] - baseline["total_return_pct"],
            "annualized_return_pct": item["annualized_return_pct"] - baseline["annualized_return_pct"],
            "sharpe": (
                item["sharpe"] - baseline["sharpe"]
                if item["sharpe"] is not None and baseline["sharpe"] is not None else None
            ),
            "max_drawdown_pct": item["max_drawdown_pct"] - baseline["max_drawdown_pct"],
            "average_turnover_pct": item["average_turnover_pct"] - baseline["average_turnover_pct"],
        }

    if len(results) != len(VARIANTS):
        raise ValueError("factor ablation variant count mismatch")
    if any(item["strategy_commit"] != strategy_commit or item["future_function"] for item in results.values()):
        raise ValueError("factor ablation strategy/PIT audit failed")

    return {
        "schema_version": 1,
        "status": "ready",
        "method": "research_only_factor_ablation",
        "start": start,
        "end": end,
        "top_n": top_n,
        "transaction_cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "baseline_weights": base_weights,
        "results": [results[name] for name in VARIANTS],
        "audit": {
            "private_strategy_modified": False,
            "future_adjusted_factor_not_used": True,
            "future_function": False,
            "production_filter_reused": True,
            "current_names_not_used_for_history": True,
            "execution_model": "next_open_to_close",
            "factor_score_penalties_reused": True,
            "scored_rows": current_scored_rows,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--end", default="2026-09-29")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=3.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--output", default=str(ROOT / "data" / "backtest" / "factor_ablation.json"))
    args = ap.parse_args()

    result = evaluate(
        start=args.start, end=args.end, top_n=args.top_n,
        cost_bps=args.cost_bps, slippage_bps=args.slippage_bps,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "variants": len(result["results"]), "strategy_commit": result["strategy_commit"]}))


if __name__ == "__main__":
    main()
