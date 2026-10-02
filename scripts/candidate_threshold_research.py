from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backtest import RollingFeatureState
from scripts.market_scope import is_main_board_symbol

DATA = ROOT / "data"
HISTORY = DATA / "history"
OUTPUT = DATA / "research" / "candidate_thresholds.json"

START_DEFAULT = "2015-01-05"
END_DEFAULT = "2026-09-29"
THRESHOLDS = tuple(range(60, 91, 2))
PERCENTILES = (1, 2, 3, 5, 10, 15, 20)


@dataclass
class DailyObservation:
    date: str
    eligible_count: int
    forward_universe_return: float
    scored: pd.DataFrame


def load_private_strategy():
    strategy_path = os.environ.get("AQUANT_PRIVATE_STRATEGY_PATH", "").strip()
    if not strategy_path:
        raise RuntimeError("AQUANT_PRIVATE_STRATEGY_PATH is required")
    root = Path(strategy_path).resolve()
    sys.path.insert(0, str(root))
    model = importlib.import_module("strategy.model")
    version = importlib.import_module("strategy.version")
    commit = os.environ.get("AQUANT_PRIVATE_STRATEGY_COMMIT", "").strip()
    if not commit:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    return model, str(getattr(version, "STRATEGY_VERSION", "unknown")), commit


def history_files() -> list[Path]:
    files = sorted(HISTORY.glob("????/*.csv.gz"), key=lambda p: p.name[:10])
    if not files:
        raise RuntimeError("no historical files")
    return files


def read_day(path: Path) -> pd.DataFrame:
    cols = [
        "symbol", "date", "open", "close", "volume", "amount", "pct_chg",
        "turnover_pct", "is_paused", "is_st", "pe_ratio", "pb_ratio",
    ]
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        frame = pd.read_csv(fh, usecols=cols)
    frame = frame.loc[frame["symbol"].map(is_main_board_symbol)].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.extract(r"(\d{6})")[0].str.zfill(6)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for col in cols[2:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if frame["date"].nunique() != 1 or frame["date"].iloc[0] != path.name[:10]:
        raise RuntimeError(f"date/file mismatch: {path}")
    if frame.duplicated("symbol").any():
        raise RuntimeError(f"duplicate symbols: {path}")
    return frame


def feature_frame(day: pd.DataFrame, state: RollingFeatureState) -> pd.DataFrame:
    symbols = day["symbol"].to_numpy(dtype=object)
    daily_ret = day["pct_chg"].to_numpy(dtype=float) / 100.0
    volumes = day["volume"].to_numpy(dtype=float)
    ret60, vol, vol_ratio = state.update_and_features(symbols, daily_ret, volumes)

    close = day["close"].to_numpy(dtype=float)
    amount = day["amount"].to_numpy(dtype=float)
    paused = day["is_paused"].fillna(0).to_numpy(dtype=float)
    st = day["is_st"].fillna(0).to_numpy(dtype=float)

    eligible = (
        (paused == 0)
        & (st == 0)
        & np.isfinite(close)
        & (close > 2.0)
        & np.isfinite(amount)
        & (amount >= 2e7)
        & np.isfinite(ret60)
    )
    if not eligible.any():
        return pd.DataFrame()

    frame = pd.DataFrame({
        "symbol": day.loc[eligible, "symbol"].to_numpy(),
        "name": day.loc[eligible, "symbol"].astype(str).to_numpy(),
        "close": close[eligible],
        "amount": amount[eligible],
        "turnover_pct": day.loc[eligible, "turnover_pct"].to_numpy(dtype=float),
        "change_pct": day.loc[eligible, "pct_chg"].to_numpy(dtype=float),
        "pe": day.loc[eligible, "pe_ratio"].to_numpy(dtype=float),
        "pb": day.loc[eligible, "pb_ratio"].to_numpy(dtype=float),
        "ret_60d": ret60[eligible],
        "momentum_60d": ret60[eligible] * 100.0,
        "volatility_proxy": vol[eligible],
        "volume_ratio": vol_ratio[eligible],
    })
    return frame.reset_index(drop=True)


def next_returns(execution: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(execution["close"], errors="coerce")
    open_px = pd.to_numeric(execution.get("open", np.nan), errors="coerce")
    if open_px is None or isinstance(open_px, float):
        return pd.Series(index=execution.index, dtype=float)
    return close.div(open_px).sub(1.0).replace([np.inf, -np.inf], np.nan)


def split_of(date: str) -> str:
    if date <= "2022-12-30":
        return "train"
    if date <= "2024-12-31":
        return "validation"
    return "oos"


def empty_accumulator():
    return {
        "days": 0,
        "candidate_observations": 0,
        "candidate_count_sum": 0.0,
        "candidate_count_sq_sum": 0.0,
        "zero_candidate_days": 0,
        "positive_candidate_days": 0,
        "daily_returns": [],
        "daily_excess_returns": [],
        "candidate_hits": 0,
    }


def update_accumulator(acc, candidate_returns, universe_return):
    count = int(candidate_returns.size)
    acc["days"] += 1
    acc["candidate_observations"] += count
    acc["candidate_count_sum"] += count
    acc["candidate_count_sq_sum"] += count * count
    if count == 0:
        acc["zero_candidate_days"] += 1
        acc["daily_returns"].append(np.nan)
        acc["daily_excess_returns"].append(np.nan)
        return
    daily_ret = float(np.nanmean(candidate_returns))
    acc["positive_candidate_days"] += int(daily_ret > 0)
    acc["daily_returns"].append(daily_ret)
    acc["daily_excess_returns"].append(daily_ret - universe_return)
    acc["candidate_hits"] += int(np.nansum(candidate_returns > 0))


def summarize(acc):
    returns = pd.Series(acc["daily_returns"], dtype=float).dropna()
    excess = pd.Series(acc["daily_excess_returns"], dtype=float).dropna()
    avg_count = acc["candidate_count_sum"] / acc["days"] if acc["days"] else 0.0
    count_var = (
        acc["candidate_count_sq_sum"] / acc["days"] - avg_count * avg_count
        if acc["days"] else 0.0
    )
    daily_mean = float(returns.mean()) if not returns.empty else None
    annualized = float((1 + daily_mean) ** 252 - 1) if daily_mean is not None and daily_mean > -1 else None
    hit_rate = (
        acc["candidate_hits"] / acc["candidate_observations"]
        if acc["candidate_observations"] else None
    )
    return {
        "days": int(acc["days"]),
        "candidate_observations": int(acc["candidate_observations"]),
        "avg_candidate_count": round(float(avg_count), 4),
        "candidate_count_std": round(float(max(count_var, 0.0) ** 0.5), 4),
        "zero_candidate_day_pct": round(100.0 * acc["zero_candidate_days"] / acc["days"], 4) if acc["days"] else None,
        "positive_daily_mean_pct": round(100.0 * acc["positive_candidate_days"] / acc["days"], 4) if acc["days"] else None,
        "candidate_mean_forward_return_pct": round(100.0 * daily_mean, 6) if daily_mean is not None else None,
        "candidate_annualized_from_mean_daily_pct": round(100.0 * annualized, 4) if annualized is not None else None,
        "candidate_hit_rate_pct": round(100.0 * hit_rate, 4) if hit_rate is not None else None,
        "excess_vs_eligible_mean_return_pct": round(100.0 * float(excess.mean()), 6) if not excess.empty else None,
        "median_daily_return_pct": round(100.0 * float(returns.median()), 6) if not returns.empty else None,
        "p10_daily_return_pct": round(100.0 * float(returns.quantile(0.10)), 6) if not returns.empty else None,
        "p90_daily_return_pct": round(100.0 * float(returns.quantile(0.90)), 6) if not returns.empty else None,
    }


def run(start: str, end: str) -> dict:
    model, version, commit = load_private_strategy()
    files = history_files()
    start_i = next((i for i,p in enumerate(files) if p.name[:10] == start), None)
    end_i = next((i for i,p in enumerate(files) if p.name[:10] == end), None)
    if start_i is None or end_i is None:
        raise ValueError("start/end must be trading dates present in history")
    if start_i > end_i:
        raise ValueError("start must be <= end")

    state = RollingFeatureState()
    pending = None
    score_acc = {
        split: {threshold: empty_accumulator() for threshold in THRESHOLDS}
        for split in ("train", "validation", "oos")
    }
    pct_acc = {
        split: {pct: empty_accumulator() for pct in PERCENTILES}
        for split in ("train", "validation", "oos")
    }

    selected_dates = files[start_i:end_i + 1]
    warmup_files = files[max(0, start_i - 60):start_i]
    replay_files = warmup_files + selected_dates
    total = len(selected_dates)

    for idx, path in enumerate(replay_files):
        day = read_day(path)
        frame = feature_frame(day, state)

        if pending is not None:
            previous_date, previous_frame, previous_split = pending
            execution = day.set_index("symbol")
            returns = execution["close"].div(execution["open"]).sub(1.0)
            eligible_symbols = previous_frame["symbol"].astype(str).tolist()
            eligible_returns = returns.reindex(eligible_symbols).dropna()
            universe_return = float(eligible_returns.mean()) if not eligible_returns.empty else 0.0

            for threshold in THRESHOLDS:
                selected = previous_frame.loc[previous_frame["score"] >= threshold, "symbol"]
                cr = returns.reindex(selected).dropna().to_numpy(dtype=float)
                update_accumulator(score_acc[previous_split][threshold], cr, universe_return)

            ranking = previous_frame["score"].rank(method="first", ascending=False, pct=True)
            for pct in PERCENTILES:
                cutoff = pct / 100.0
                selected = previous_frame.loc[ranking <= cutoff, "symbol"]
                cr = returns.reindex(selected).dropna().to_numpy(dtype=float)
                update_accumulator(pct_acc[previous_split][pct], cr, universe_return)

        if path.name[:10] >= start and path.name[:10] <= end and not frame.empty:
            scored = model.score_universe(frame).reset_index(drop=True)
            pending = (path.name[:10], scored, split_of(path.name[:10]))
        elif path.name[:10] >= start and path.name[:10] <= end:
            pending = (path.name[:10], pd.DataFrame(columns=["symbol", "score"]), split_of(path.name[:10]))
        if idx % 250 == 0:
            print(f"[threshold-research] {idx}/{len(replay_files)} signal_date={path.name[:10]}")

    # The last signal date has no next-session observation.
    return {
        "schema_version": 1,
        "status": "ready",
        "method": "historical_score_threshold_forward_return",
        "signal_definition": "EOD T cross-section after production hard eligibility filters; score from Aquant-Private/main",
        "execution_definition": "T+1 open-to-close stock return; candidate-pool return is equal-weight mean across admitted names",
        "hard_eligibility": {
            "main_board_only": True,
            "not_st": True,
            "not_paused": True,
            "close_gt": 2.0,
            "amount_gte": 20000000.0,
            "momentum_window_days": 60,
        },
        "splits": {
            "train": "2015-01-05 through 2022-12-30",
            "validation": "2023-01-03 through 2024-12-31",
            "oos": "2025-01-02 through 2026-09-29",
        },
        "score_thresholds": {
            str(t): {split: summarize(score_acc[split][t]) for split in ("train", "validation", "oos")}
            for t in THRESHOLDS
        },
        "percentile_reference": {
            str(p): {split: summarize(pct_acc[split][p]) for split in ("train", "validation", "oos")}
            for p in PERCENTILES
        },
        "strategy_source": "Aquant-Private/main",
        "strategy_version": version,
        "strategy_commit": commit,
        "future_function": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Research candidate score admission thresholds")
    parser.add_argument("--start", default=START_DEFAULT)
    parser.add_argument("--end", default=END_DEFAULT)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    payload = run(args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "strategy_version": payload["strategy_version"], "strategy_commit": payload["strategy_commit"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
