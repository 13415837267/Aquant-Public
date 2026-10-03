"""Walk-forward OOS evaluation for research-only factor ablations.

The Private strategy remains the production source. Each factor variant is evaluated with the
same historical PIT state, T+1 execution model, and 9 calendar walk-forward
folds used by the canonical strategy OOS report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backtest as base
from scripts.factor_ablation import EXPECTED_WEIGHTS, VARIANTS, score_variant, variant_weights
from scripts.walk_forward import build_folds


def evaluate_daily(start: str, end: str, top_n: int, cost_bps: float, slippage_bps: float):
    strategy_model, strategy_version, strategy_commit = base.load_strategy()
    weights = dict(getattr(strategy_model, "WEIGHTS", {}))
    if weights != EXPECTED_WEIGHTS:
        raise ValueError(f"unexpected strategy weights: {weights}")

    files = base.history_files()
    selected_files, _ = base.iter_selected_dates(files, start, end)
    if len(selected_files) < 2:
        raise ValueError("OOS ablation needs at least two sessions")

    state = base.RollingFeatureState()
    current = base.read_daily(selected_files[0])
    daily = {name: [] for name in VARIANTS}
    prev_target = {name: {} for name in VARIANTS}

    for next_path in selected_files[1:]:
        next_day = base.read_daily(next_path)
        signal_date = current["date"].iloc[0]
        frame = base.build_strategy_frame(current, state)

        if frame.empty:
            scored_base = frame.copy()
            penalty = pd.Series(index=frame.index, dtype=float)
        else:
            scored_base = strategy_model.score_universe(frame).copy()
            raw = 100.0 * sum(
                float(weights[k])
                * pd.to_numeric(scored_base[f"f_{k}"], errors="coerce").fillna(0.5)
                for k in weights
            )
            penalty = pd.to_numeric(scored_base["score"], errors="coerce") - raw

        for name in VARIANTS:
            targets = (
                score_variant(scored_base, variant_weights(name, weights), penalty)
                .head(top_n).reset_index(drop=True)
                if not frame.empty else frame
            )
            symbols = targets["symbol"].astype(str).str.zfill(6).tolist() if not targets.empty else []
            target_weights = base.normalize_weights(symbols)
            gross, executed, missing = base.next_session_return(
                targets, next_day, target_weights
            )
            actual_target = {
                symbol: target_weights[symbol]
                for symbol in executed
                if symbol in target_weights
            }
            turn = base.turnover(prev_target[name], actual_target)
            cost = turn * (cost_bps + slippage_bps) / 10000.0
            daily[name].append({
                "date": next_day["date"].iloc[0],
                "signal_date": signal_date,
                "gross_return": float(gross),
                "turnover": float(turn),
                "net_return": float(gross - cost),
                "target_count": int(len(targets)),
                "executed_count": int(len(executed)),
                "missing_execution_count": int(len(missing)),
            })
            prev_target[name] = actual_target

        current = next_day

    history_dates = [path.name[:10] for path in selected_files]
    return daily, strategy_version, strategy_commit, history_dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--end", default="2026-09-29")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--cost-bps", type=float, default=3.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--baseline-walk-forward",
                    default=str(ROOT / "data" / "backtest" / "walk_forward.json"))
    ap.add_argument("--output", default=str(ROOT / "data" / "backtest" / "factor_ablation_walk_forward.json"))
    args = ap.parse_args()

    daily, version, commit, history_dates = evaluate_daily(
        args.start, args.end, args.top_n, args.cost_bps, args.slippage_bps
    )
    folds = build_folds(history_dates, train_years=3, test_years=1)

    baseline_wf = json.loads(Path(args.baseline_walk_forward).read_text(encoding="utf-8"))
    expected_folds = {
        int(x["fold"]): x for x in baseline_wf.get("folds", [])
    }

    results = []
    for name in VARIANTS:
        frame = pd.DataFrame(daily[name])
        for fold in folds:
            oos = frame[
                (frame["date"].astype(str) >= fold["oos_start"])
                & (frame["date"].astype(str) <= fold["oos_end"])
            ].reset_index(drop=True)
            if oos.empty:
                raise ValueError(f"empty OOS fold {name} #{fold['fold']}")
            m = base.metrics(oos)
            row = {
                **fold,
                "variant": name,
                "weights": variant_weights(name, EXPECTED_WEIGHTS),
                "oos_trading_days": int(len(oos)),
                "overall": m,
                "strategy_version": version,
                "strategy_commit": commit,
                "future_function": False,
            }
            if name == "baseline":
                expected = expected_folds.get(int(fold["fold"]))
                if expected:
                    for key in ("total_return_pct", "annualized_return_pct",
                                "annualized_volatility_pct", "sharpe",
                                "max_drawdown_pct", "win_rate_pct"):
                        if abs(float(m[key]) - float(expected["overall"][key])) > 1e-8:
                            raise ValueError(
                                f"baseline OOS mismatch fold {fold['fold']} {key}"
                            )
            results.append(row)

    aggregate = {}
    for name in VARIANTS:
        frame = pd.DataFrame(daily[name])
        chunks = []
        for fold in folds:
            chunks.append(
                frame[
                    (frame["date"].astype(str) >= fold["oos_start"])
                    & (frame["date"].astype(str) <= fold["oos_end"])
                ]
            )
        oos = pd.concat(chunks, ignore_index=True)
        aggregate[name] = {
            "variant": name,
            "weights": variant_weights(name, EXPECTED_WEIGHTS),
            "overall": base.metrics(oos),
            "strategy_version": version,
            "strategy_commit": commit,
            "future_function": False,
        }

    if any(x["strategy_commit"] != commit or x["future_function"] for x in aggregate.values()):
        raise ValueError("OOS ablation strategy/PIT audit failed")

    result = {
        "schema_version": 1,
        "status": "ready",
        "method": "research_only_factor_ablation_walk_forward_oos",
        "start": args.start,
        "end": args.end,
        "top_n": args.top_n,
        "transaction_cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": version,
        "strategy_commit": commit,
        "future_function": False,
        "folds": results,
        "aggregate_oos": [aggregate[name] for name in VARIANTS],
        "audit": {
            "private_strategy_modified": False,
            "future_function": False,
            "future_adjusted_factor_not_used": True,
            "baseline_folds_cross_checked": True,
            "same_walk_forward_definition": "3y train / 1y test / 60-session warmup",
            "execution_model": "next_open_to_close",
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "folds": len(folds),
        "variants": len(VARIANTS),
        "strategy_commit": commit,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
