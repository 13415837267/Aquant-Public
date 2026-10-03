"""Market-regime and factor diagnostics for the frozen Private strategy."""
# Public diagnostics layer; the Private strategy remains the source of truth.
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import backtest as base

OUT_FILE = ROOT / "data" / "backtest" / "regime_factor_diagnostics.json"
EXPECTED_WEIGHTS = {
    "momentum": 0.35,
    "liquidity": 0.15,
    "value": 0.30,
    "safety": 0.20,
}


def spearman_corr(x: pd.Series, y: pd.Series) -> float | None:
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return None
    xr = frame["x"].rank(method="average")
    yr = frame["y"].rank(method="average")
    if xr.std(ddof=1) <= 0 or yr.std(ddof=1) <= 0:
        return None
    return float(xr.corr(yr))


def classify_trend(cumulative_return_pct: float | None) -> str:
    if cumulative_return_pct is None or not np.isfinite(cumulative_return_pct):
        return "unknown"
    if cumulative_return_pct >= 10.0:
        return "bull"
    if cumulative_return_pct <= -10.0:
        return "bear"
    return "neutral"


def classify_volatility(vol_annualized_pct: float | None, rolling_median: float | None) -> str:
    if (
        vol_annualized_pct is None
        or rolling_median is None
        or not np.isfinite(vol_annualized_pct)
        or not np.isfinite(rolling_median)
        or rolling_median <= 0
    ):
        return "unknown"
    return "high_vol" if vol_annualized_pct > 1.5 * rolling_median else "normal_vol"


def prepare_baseline(path: Path, top_n: int) -> tuple[dict, dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ready":
        raise ValueError("baseline backtest is not ready")
    if payload.get("future_function") is not False:
        raise ValueError("baseline future_function audit failed")
    if int(payload.get("top_n", -1)) != top_n:
        raise ValueError("baseline top_n does not match diagnostics top_n")

    audit = payload.get("audit", {})
    if not audit.get("current_names_not_used_for_history", False):
        raise ValueError("baseline current-name audit missing")

    selection = {
        row["signal_date"]: row
        for row in payload.get("selection_audit", [])
        if row.get("candidate_count", 0) > 0
    }
    daily = {
        row["execution_date"] if "execution_date" in row else row["date"]: row
        for row in payload.get("selection_audit", [])
    }
    return payload, selection, daily


def run_diagnostics(
    *,
    start: str,
    end: str,
    top_n: int,
    baseline_path: Path,
) -> dict:
    baseline, selection_map, _ = prepare_baseline(baseline_path, top_n)
    strategy_model, strategy_version, strategy_commit = base.load_strategy()

    weights = dict(getattr(strategy_model, "WEIGHTS", {}))
    if weights != EXPECTED_WEIGHTS:
        raise ValueError(f"unexpected strategy weights: {weights}")

    files = base.history_files()
    selected_files, _ = base.iter_selected_dates(files, start, end)
    if len(selected_files) < 2:
        raise ValueError("diagnostics needs at least two historical sessions")

    state = base.RollingFeatureState()
    current = base.read_daily(selected_files[0])

    daily_diag: list[dict] = []
    factor_selected: dict[str, list[float]] = defaultdict(list)
    factor_universe: dict[str, list[float]] = defaultdict(list)
    factor_contrib: dict[str, list[float]] = defaultdict(list)
    factor_ic: dict[str, list[float]] = defaultdict(list)
    factor_ic_by_year: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    market_daily: list[dict] = []
    score_audit_max_abs_error = 0.0
    score_audit_rows = 0

    for i in range(len(selected_files) - 1):
        next_day = base.read_daily(selected_files[i + 1])
        signal_date = current["date"].iloc[0]

        frame = base.build_strategy_frame(current, state)
        if not frame.empty:
            scored = strategy_model.score_universe(frame)
            raw_score = 100 * sum(
                float(weights[key]) * pd.to_numeric(scored[f"f_{key}"], errors="coerce").fillna(0.5)
                for key in weights
            )
            penalty = raw_score - pd.to_numeric(scored["score"], errors="coerce")
            scored = scored.copy()
            scored["raw_score"] = raw_score
            scored["penalty"] = penalty

            change = pd.to_numeric(frame["change_pct"], errors="coerce")
            market_mean = float(change.mean()) if change.notna().any() else None
            breadth = float((change > 0).mean() * 100.0) if change.notna().any() else None
            market_daily.append(
                {
                    "date": signal_date,
                    "market_mean_return_pct": market_mean,
                    "breadth_pct": breadth,
                    "universe_size": int(len(frame)),
                }
            )

            execution = next_day[["symbol", "open", "close"]].copy()
            execution["symbol"] = execution["symbol"].astype(str).str.zfill(6)
            execution["open"] = pd.to_numeric(execution["open"], errors="coerce")
            execution["close"] = pd.to_numeric(execution["close"], errors="coerce")
            execution["forward_return"] = execution["close"] / execution["open"] - 1.0
            forward = execution[["symbol", "forward_return"]]
            merged = scored.merge(forward, on="symbol", how="left")
            merged = merged.replace([np.inf, -np.inf], np.nan)

            # Cross-sectional one-day signal quality.
            year_key = signal_date[:4]
            for key in list(weights) + ["score"]:
                ic = spearman_corr(merged[key if key == "score" else f"f_{key}"], merged["forward_return"])
                if ic is not None:
                    factor_ic[key].append(ic)
                    factor_ic_by_year[key][year_key].append(ic)

            selected_info = selection_map.get(signal_date)
            if selected_info:
                symbols = [str(x).zfill(6) for x in selected_info.get("symbols", [])]
                expected_scores = [float(x) for x in selected_info.get("scores", [])]
                selected_rows = scored[scored["symbol"].astype(str).str.zfill(6).isin(symbols)].copy()
                selected_rows["symbol"] = selected_rows["symbol"].astype(str).str.zfill(6)
                selected_rows = selected_rows.set_index("symbol").reindex(symbols).reset_index()
                actual_scores = pd.to_numeric(selected_rows["score"], errors="coerce").to_numpy()
                for expected, actual in zip(expected_scores, actual_scores):
                    if np.isfinite(actual):
                        score_audit_max_abs_error = max(score_audit_max_abs_error, abs(expected - actual))
                        score_audit_rows += 1

                for key in weights:
                    selected_values = pd.to_numeric(selected_rows[f"f_{key}"], errors="coerce")
                    universe_values = pd.to_numeric(scored[f"f_{key}"], errors="coerce")
                    contrib = weights[key] * selected_values * 100.0
                    factor_selected[key].extend(selected_values.dropna().tolist())
                    factor_universe[key].extend(universe_values.dropna().tolist())
                    factor_contrib[key].extend(contrib.dropna().tolist())

        current = next_day

    market = pd.DataFrame(market_daily)
    market["market_mean_return_pct"] = pd.to_numeric(market["market_mean_return_pct"], errors="coerce")
    market["breadth_pct"] = pd.to_numeric(market["breadth_pct"], errors="coerce")
    market["market_20d_return_pct"] = (
        (1.0 + market["market_mean_return_pct"].fillna(0.0) / 100.0)
        .rolling(20)
        .apply(np.prod, raw=True)
        .sub(1.0)
        .mul(100.0)
    )
    market["market_60d_return_pct"] = (
        (1.0 + market["market_mean_return_pct"].fillna(0.0) / 100.0)
        .rolling(60)
        .apply(np.prod, raw=True)
        .sub(1.0)
        .mul(100.0)
    )
    market["market_20d_vol_pct"] = (
        market["market_mean_return_pct"].rolling(20).std(ddof=1) * np.sqrt(252.0)
    )
    market["vol_252d_median_pct"] = market["market_20d_vol_pct"].rolling(252).median()
    market["trend_regime"] = market["market_60d_return_pct"].map(classify_trend)
    market["volatility_regime"] = [
        classify_volatility(a, b)
        for a, b in zip(market["market_20d_vol_pct"], market["vol_252d_median_pct"])
    ]

    baseline_daily = pd.DataFrame(baseline.get("daily", []))
    baseline_daily = baseline_daily[baseline_daily["date"] >= baseline.get("trade_start", "")].copy()
    baseline_daily["net_return"] = pd.to_numeric(baseline_daily["net_return"], errors="coerce")
    baseline_daily = baseline_daily.merge(
        market[["date", "trend_regime", "volatility_regime", "market_60d_return_pct", "market_20d_vol_pct", "breadth_pct"]],
        left_on="signal_date",
        right_on="date",
        how="left",
        suffixes=("", "_market"),
    )

    regime_rows = []
    regime_coverage = {}
    for regime_key in ["trend_regime", "volatility_regime"]:
        known_mask = (
            baseline_daily[regime_key].notna()
            & ~baseline_daily[regime_key].astype(str).eq("unknown")
        )
        regime_coverage[regime_key] = {
            "known_sessions": int(known_mask.sum()),
            "unknown_or_unavailable_sessions": int((~known_mask).sum()),
        }
        for regime, group in baseline_daily.loc[known_mask].groupby(regime_key):
            returns = group["net_return"].dropna()
            if len(returns) == 0:
                continue
            equity = (1.0 + returns).cumprod()
            regime_rows.append(
                {
                    "dimension": regime_key,
                    "regime": str(regime),
                    "sessions": int(len(returns)),
                    "total_return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
                    "mean_daily_return_pct": float(returns.mean() * 100.0),
                    "win_rate_pct": float((returns > 0).mean() * 100.0),
                    "max_drawdown_pct": float((equity / equity.cummax() - 1.0).min() * 100.0),
                }
            )

    selected_exposure = {
        key: {
            "selected_mean_rank": float(np.mean(factor_selected[key])) if factor_selected[key] else None,
            "universe_mean_rank": float(np.mean(factor_universe[key])) if factor_universe[key] else None,
            "selected_mean_weighted_contribution_pct": float(np.mean(factor_contrib[key])) if factor_contrib[key] else None,
        }
        for key in weights
    }

    factor_ic_summary = {}
    for key, values in factor_ic.items():
        arr = np.asarray(values, dtype=float)
        factor_ic_summary[key] = {
            "sessions": int(len(arr)),
            "mean_ic": float(np.mean(arr)) if len(arr) else None,
            "median_ic": float(np.median(arr)) if len(arr) else None,
            "ic_positive_rate_pct": float((arr > 0).mean() * 100.0) if len(arr) else None,
            "yearly_mean_ic": {
                year: float(np.mean(vals))
                for year, vals in sorted(factor_ic_by_year[key].items())
                if vals
            },
        }

    overall_market = {
        "sessions": int(len(market)),
        "average_breadth_pct": float(market["breadth_pct"].mean()),
        "bull_sessions": int((market["trend_regime"] == "bull").sum()),
        "neutral_sessions": int((market["trend_regime"] == "neutral").sum()),
        "bear_sessions": int((market["trend_regime"] == "bear").sum()),
        "high_vol_sessions": int((market["volatility_regime"] == "high_vol").sum()),
        "normal_vol_sessions": int((market["volatility_regime"] == "normal_vol").sum()),
    }

    result = {
        "schema_version": 1,
        "status": "ready",
        "method": "frozen_strategy_regime_factor_diagnostics",
        "start": start,
        "end": end,
        "top_n": top_n,
        "strategy_source": "Aquant-Private/main",
        "strategy_version": strategy_version,
        "strategy_commit": strategy_commit,
        "strategy_weights": weights,
        "overall_market": overall_market,
        "regime_performance": regime_rows,
        "regime_coverage": regime_coverage,
        "factor_exposure": selected_exposure,
        "factor_ic": factor_ic_summary,
        "audit": {
            "future_function": False,
            "baseline_strategy_commit": baseline.get("strategy_commit"),
            "baseline_score_reconstruction_max_abs_error": score_audit_max_abs_error,
            "baseline_score_reconstruction_rows": score_audit_rows,
            "baseline_selection_sessions": int(len(selection_map)),
            "factor_ic_keys": sorted(factor_ic_summary),
            "future_adjusted_factor_not_used": True,
            "current_names_not_used_for_history": True,
            "production_filter_reused": True,
            "market_regime_proxy": "main-board eligible universe equal-weight daily pct_chg breadth",
            "trend_regime_rule": "60-session proxy return >= +10% bull, <= -10% bear, otherwise neutral",
            "volatility_regime_rule": "20-session annualized volatility > 1.5x trailing 252-session median = high_vol",
        },
    }
    if result["audit"]["baseline_strategy_commit"] != strategy_commit:
        raise ValueError("strategy commit mismatch against baseline")
    if score_audit_rows == 0 or score_audit_max_abs_error > 1e-6:
        raise ValueError("baseline score reconstruction audit failed")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--end", default="2026-09-29")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--baseline", default=str(ROOT / "data" / "backtest" / "latest.json"))
    ap.add_argument("--output", default=str(OUT_FILE))
    args = ap.parse_args()

    result = run_diagnostics(
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        baseline_path=Path(args.baseline),
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "strategy_commit": result["strategy_commit"],
                "score_reconstruction_max_abs_error": result["audit"]["baseline_score_reconstruction_max_abs_error"],
                "regime_rows": len(result["regime_performance"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
