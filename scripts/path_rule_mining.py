"""Path-aware conditional rule mining for the 1% short-term objective.

Research only. Rules use information available at signal close T and are
selected on train, ranked on validation, then evaluated once on final holdout.
A win requires net +1% before a gross -3% stop within five sessions.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily

OUT_DIR = ROOT / "data" / "backtest"

STOCK_FEATURES = [
    "return_1d_pct", "return_3d_pct", "return_5d_pct", "return_10d_pct",
    "return_20d_pct", "overnight_1d_pct", "overnight_3d_pct",
    "overnight_5d_pct", "overnight_10d_pct", "volume_ratio_5d",
    "amount_20d", "volatility_10d_pct", "close_strength",
    "intraday_return_pct", "limit_up_5d_count", "turnover_pct", "change_pct",
]
MARKET_FEATURES = ["market_breadth_pct", "market_median_return_pct"]

TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"

NET_WIN_THRESHOLD_PCT = 1.0
STOP_LOSS_PCT = 3.0
ROUND_TRIP_COST_BPS = 10.0
MAX_FORWARD_SESSIONS = 5
ENTRY_LIMIT_UP_BLOCK = True

RANK_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
BREADTH_THRESHOLDS = (40.0, 45.0, 50.0, 55.0, 60.0, 65.0)
MEDIAN_THRESHOLDS = (-1.0, -0.5, 0.0, 0.5, 1.0)

MIN_TRAIN_RULE_SAMPLES = 10_000
MIN_VALIDATION_RULE_SAMPLES = 5_000
MAX_ATOMIC_CANDIDATES = 16
MAX_PAIR_CANDIDATES = 24
TOP_OUTPUT_RULES = 20


def percentile_rank(series):
    x = pd.to_numeric(series, errors="coerce")
    return x.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=np.float64)


def make_features(frame):
    cols = [percentile_rank(frame[name]) for name in STOCK_FEATURES]
    breadth = np.clip(
        pd.to_numeric(frame["market_breadth_pct"], errors="coerce")
        .fillna(50.0).to_numpy(dtype=float),
        0.0, 100.0,
    )
    median_ret = pd.to_numeric(
        frame["market_median_return_pct"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype=float)
    return np.column_stack(cols + [breadth, median_ret])


def feature_defs():
    defs = []
    for name in STOCK_FEATURES:
        for threshold in RANK_THRESHOLDS:
            defs.append({"feature": name, "op": ">=", "threshold": threshold, "scale": "rank"})
            defs.append({"feature": name, "op": "<=", "threshold": threshold, "scale": "rank"})
    for threshold in BREADTH_THRESHOLDS:
        defs.append({"feature": "market_breadth_pct", "op": ">=", "threshold": threshold, "scale": "pct"})
        defs.append({"feature": "market_breadth_pct", "op": "<=", "threshold": threshold, "scale": "pct"})
    for threshold in MEDIAN_THRESHOLDS:
        defs.append({"feature": "market_median_return_pct", "op": ">=", "threshold": threshold, "scale": "pct"})
        defs.append({"feature": "market_median_return_pct", "op": "<=", "threshold": threshold, "scale": "pct"})
    return defs


RULE_DEFS = feature_defs()


def rule_label(rule):
    return f'{rule["feature"]}{rule["op"]}{rule["threshold"]:g}'


def wilson_lower_bound(wins, samples, z=1.96):
    if samples <= 0:
        return 0.0
    p = wins / samples
    denom = 1.0 + z * z / samples
    center = p + z * z / (2.0 * samples)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * samples)) / samples)
    return (center - spread) / denom


def path_targets(symbols, future_days):
    n = len(symbols)
    empty = (
        np.zeros(n, dtype=bool),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.zeros(n, dtype=bool),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
        np.full(n, np.nan),
    )
    if len(future_days) < MAX_FORWARD_SESSIONS:
        return empty

    keys = pd.Index(pd.Series(symbols, dtype="string").astype(str).str.zfill(6))
    first = future_days[0].set_index("symbol")
    entry = pd.to_numeric(first["open"], errors="coerce").reindex(keys).to_numpy(dtype=float)
    if ENTRY_LIMIT_UP_BLOCK and "high_limit" in first.columns:
        high_limit = pd.to_numeric(first["high_limit"], errors="coerce").reindex(keys).to_numpy(dtype=float)
        executable = ~np.isfinite(high_limit) | (entry < high_limit * (1.0 - 1e-6))
    else:
        executable = np.ones(n, dtype=bool)

    target_gross = NET_WIN_THRESHOLD_PCT + ROUND_TRIP_COST_BPS / 100.0
    target = entry * (1.0 + target_gross / 100.0)
    stop = entry * (1.0 - STOP_LOSS_PCT / 100.0)

    opens, highs, lows, closes = [], [], [], []
    for day in future_days[:MAX_FORWARD_SESSIONS]:
        idx = day.set_index("symbol")
        opens.append(pd.to_numeric(idx["open"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        highs.append(pd.to_numeric(idx["high"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        lows.append(pd.to_numeric(idx["low"], errors="coerce").reindex(keys).to_numpy(dtype=float))
        closes.append(pd.to_numeric(idx["close"], errors="coerce").reindex(keys).to_numpy(dtype=float))

    opens = np.column_stack(opens)
    highs = np.column_stack(highs)
    lows = np.column_stack(lows)
    closes = np.column_stack(closes)

    complete = (
        np.isfinite(entry) & (entry > 0) & executable
        & np.isfinite(opens).all(axis=1)
        & np.isfinite(highs).all(axis=1)
        & np.isfinite(lows).all(axis=1)
        & np.isfinite(closes).all(axis=1)
    )

    path_win = np.zeros(n, dtype=bool)
    best_net = np.full(n, np.nan)
    close_net = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    target_day = np.full(n, np.nan)
    stop_day = np.full(n, np.nan)

    for row in np.flatnonzero(complete):
        best_net[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        close_net[row] = closes[row, -1] / entry[row] * 100.0 - 100.0 - ROUND_TRIP_COST_BPS / 100.0
        mae[row] = np.min(lows[row]) / entry[row] * 100.0 - 100.0
        mfe[row] = np.max(highs[row]) / entry[row] * 100.0 - 100.0

        stopped = False
        for d in range(MAX_FORWARD_SESSIONS):
            day_open, day_high, day_low = opens[row, d], highs[row, d], lows[row, d]
            if day_open <= stop[row] or day_low <= stop[row]:
                stop_day[row] = d + 1
                stopped = True
                break
            if day_high >= target[row]:
                target_day[row] = d + 1
                break
        path_win[row] = (target_day[row] > 0) and not stopped

    return path_win, best_net, close_net, complete, mae, mfe, target_day, stop_day


def atomic_masks(x):
    masks = np.empty((len(x), len(RULE_DEFS)), dtype=np.uint8)
    for idx, rule in enumerate(RULE_DEFS):
        col = STOCK_FEATURES.index(rule["feature"]) if rule["feature"] in STOCK_FEATURES else len(STOCK_FEATURES) + (0 if rule["feature"] == "market_breadth_pct" else 1)
        values = x[:, col]
        threshold = rule["threshold"]
        masks[:, idx] = (values >= threshold if rule["op"] == ">=" else values <= threshold).astype(np.uint8)
    return masks


def accumulate_atomic(stats, masks, y):
    stats["samples"] += masks.sum(axis=0, dtype=np.int64)
    stats["wins"] += (masks * y[:, None]).sum(axis=0, dtype=np.int64)


def evaluate_selected(masks, y, best, close, mae, mfe, target_day, stop_day, selected_indices):
    rows = []
    for idx in selected_indices:
        mask = masks[:, idx].astype(bool)
        n = int(mask.sum())
        w = int(y[mask].sum())
        rows.append({
            "rule_type": "atomic",
            "rule_indices": [int(idx)],
            "rules": [RULE_DEFS[idx]],
            "rule_label": rule_label(RULE_DEFS[idx]),
            "samples": n,
            "wins": w,
            "path_win_1pct_rate_pct": float(w / n * 100.0) if n else None,
            "wilson_lower_pct": float(wilson_lower_bound(w, n) * 100.0) if n else None,
            "mean_best_return_pct": float(best[mask].mean()) if n else None,
            "mean_5d_close_return_pct": float(close[mask].mean()) if n else None,
            "mean_mae_pct": float(mae[mask].mean()) if n else None,
            "mean_mfe_pct": float(mfe[mask].mean()) if n else None,
            "target_day_mean": float(target_day[mask][target_day[mask] > 0].mean()) if np.any(mask & (target_day > 0)) else None,
            "stop_day_mean": float(stop_day[mask][stop_day[mask] > 0].mean()) if np.any(mask & (stop_day > 0)) else None,
        })
    return rows


def evaluate_pairs(masks, y, best, close, mae, mfe, target_day, stop_day, pair_indices):
    rows = []
    for left, right in pair_indices:
        mask = (masks[:, left] & masks[:, right]).astype(bool)
        n = int(mask.sum())
        if n == 0:
            continue
        w = int(y[mask].sum())
        rows.append({
            "rule_type": "pair",
            "rule_indices": [int(left), int(right)],
            "rules": [RULE_DEFS[left], RULE_DEFS[right]],
            "rule_label": f'{rule_label(RULE_DEFS[left])} AND {rule_label(RULE_DEFS[right])}',
            "samples": n,
            "wins": w,
            "path_win_1pct_rate_pct": float(w / n * 100.0),
            "wilson_lower_pct": float(wilson_lower_bound(w, n) * 100.0),
            "mean_best_return_pct": float(best[mask].mean()),
            "mean_5d_close_return_pct": float(close[mask].mean()),
            "mean_mae_pct": float(mae[mask].mean()),
            "mean_mfe_pct": float(mfe[mask].mean()),
            "target_day_mean": float(target_day[mask][target_day[mask] > 0].mean()) if np.any(mask & (target_day > 0)) else None,
            "stop_day_mean": float(stop_day[mask][stop_day[mask] > 0].mean()) if np.any(mask & (stop_day > 0)) else None,
        })
    return rows


def collect_split(files, start, end, state, collect_masks=None):
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start), dates.index(end)
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[i]

    atomic_stats = None
    stored = []
    processed = 0
    samples = 0
    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start or date > end:
            continue
        if i + MAX_FORWARD_SESSIONS >= len(files) or frame.empty:
            continue
        futures = [get(i + j) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        x_full = make_features(frame)
        symbols = frame["symbol"].astype(str).str.zfill(6).tolist()
        y, best, close, complete, mae, mfe, target_day, stop_day = path_targets(symbols, futures)
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue
        x = x_full[keep]
        yy, bb, cc = y[keep], best[keep], close[keep]
        mm, ff, td, sd = mae[keep], mfe[keep], target_day[keep], stop_day[keep]
        masks = atomic_masks(x)
        if collect_masks is not None:
            collect_masks.append((date, masks, yy, bb, cc, mm, ff, td, sd))
        if atomic_stats is None:
            atomic_stats = {"samples": np.zeros(len(RULE_DEFS), dtype=np.int64), "wins": np.zeros(len(RULE_DEFS), dtype=np.int64)}
        accumulate_atomic(atomic_stats, masks, yy)
        stored.append((date, masks, yy, bb, cc, mm, ff, td, sd))
        samples += len(yy)
        processed += 1
        if processed % 50 == 0:
            print(f"[条件挖掘] {start}-{end} 已处理{processed}日，样本{samples}", flush=True)
    return stored, processed, samples, atomic_stats


def summarize_atomic(stats):
    rows = []
    if stats is None:
        return rows
    for idx, rule in enumerate(RULE_DEFS):
        n, w = int(stats["samples"][idx]), int(stats["wins"][idx])
        if n < MIN_TRAIN_RULE_SAMPLES:
            continue
        rows.append({
            "rule_type": "atomic",
            "rule_index": idx,
            "rule": rule,
            "rule_label": rule_label(rule),
            "samples": n,
            "wins": w,
            "path_win_1pct_rate_pct": w / n * 100.0,
            "wilson_lower_pct": wilson_lower_bound(w, n) * 100.0,
        })
    rows.sort(key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]), reverse=True)
    return rows


def pair_stats_from_masks(stored, selected):
    stats = {(a, b): [0, 0] for a, b in selected}
    for _, masks, y, *_ in stored:
        if not selected:
            continue
        a = masks[:, [p[0] for p in selected]]
        b = masks[:, [p[1] for p in selected]]
        hits = a & b
        counts = hits.sum(axis=0, dtype=np.int64)
        wins = (hits * y[:, None]).sum(axis=0, dtype=np.int64)
        for pos, pair in enumerate(selected):
            stats[pair][0] += int(counts[pos])
            stats[pair][1] += int(wins[pos])
    return stats


def rank_pair_candidates(stored, atomic_indices):
    pairs = [(left, right) for pos, left in enumerate(atomic_indices) for right in atomic_indices[pos + 1:]]
    raw = pair_stats_from_masks(stored, pairs)
    rows = []
    for pair, (n, w) in raw.items():
        if n < MIN_TRAIN_RULE_SAMPLES:
            continue
        rows.append({
            "left": pair[0], "right": pair[1], "samples": n, "wins": w,
            "path_win_1pct_rate_pct": w / n * 100.0,
            "wilson_lower_pct": wilson_lower_bound(w, n) * 100.0,
            "rule_label": f'{rule_label(RULE_DEFS[pair[0]])} AND {rule_label(RULE_DEFS[pair[1]])}',
        })
    rows.sort(key=lambda r: (r["wilson_lower_pct"], r["path_win_1pct_rate_pct"], r["samples"]), reverse=True)
    return rows


def subset_rows(stored, y, selected_atomic, selected_pairs):
    return evaluate_selected(stored[0][1], y, stored[0][3], stored[0][4], stored[0][5], stored[0][6], stored[0][7], stored[0][8], selected_atomic)  # pragma: no cover


def flatten_eval(stored, selected_atomic, selected_pairs):
    rows = []
    for item in stored:
        date, masks, y, best, close, mae, mfe, target_day, stop_day = item
        rows.extend(evaluate_selected(masks, y, best, close, mae, mfe, target_day, stop_day, selected_atomic))
        rows.extend(evaluate_pairs(masks, y, best, close, mae, mfe, target_day, stop_day, selected_pairs))
    by_key = {}
    for row in rows:
        key = tuple(row["rule_indices"])
        if key not in by_key:
            by_key[key] = row.copy()
            by_key[key]["samples"] = 0
            by_key[key]["wins"] = 0
            for field in ("mean_best_return_pct", "mean_5d_close_return_pct", "mean_mae_pct", "mean_mfe_pct"):
                by_key[key][field] = 0.0
            by_key[key]["target_day_sum"] = 0.0
            by_key[key]["target_day_count"] = 0
            by_key[key]["stop_day_sum"] = 0.0
            by_key[key]["stop_day_count"] = 0
        dst = by_key[key]
        n = row["samples"]
        dst["samples"] += n
        dst["wins"] += row["wins"]
        for field in ("mean_best_return_pct", "mean_5d_close_return_pct", "mean_mae_pct", "mean_mfe_pct"):
            dst[field] += (row[field] or 0.0) * n
        if row["target_day_mean"] is not None:
            dst["target_day_sum"] += row["target_day_mean"] * n
            dst["target_day_count"] += n
        if row["stop_day_mean"] is not None:
            dst["stop_day_sum"] += row["stop_day_mean"] * n
            dst["stop_day_count"] += n
    out = []
    for row in by_key.values():
        n = row["samples"]
        row["path_win_1pct_rate_pct"] = row["wins"] / n * 100.0 if n else None
        row["wilson_lower_pct"] = wilson_lower_bound(row["wins"], n) * 100.0 if n else None
        for field in ("mean_best_return_pct", "mean_5d_close_return_pct", "mean_mae_pct", "mean_mfe_pct"):
            row[field] = row[field] / n if n else None
        row["target_day_mean"] = row["target_day_sum"] / row["target_day_count"] if row["target_day_count"] else None
        row["stop_day_mean"] = row["stop_day_sum"] / row["stop_day_count"] if row["stop_day_count"] else None
        for field in ("target_day_sum", "target_day_count", "stop_day_sum", "stop_day_count"):
            row.pop(field, None)
        out.append(row)
    out.sort(key=lambda r: (r["path_win_1pct_rate_pct"] or -1, r["wilson_lower_pct"] or -1, r["samples"]), reverse=True)
    return out


def split_summary(stored):
    if not stored:
        return {"samples": 0, "path_win_1pct_rate_pct": None, "mean_best_return_pct": None, "mean_5d_close_return_pct": None}
    y = np.concatenate([x[2] for x in stored])
    best = np.concatenate([x[3] for x in stored])
    close = np.concatenate([x[4] for x in stored])
    return {
        "samples": int(len(y)),
        "path_win_1pct_rate_pct": float(y.mean() * 100.0),
        "mean_best_return_pct": float(best.mean()),
        "mean_5d_close_return_pct": float(close.mean()),
    }


def choose_validation_rules(train_atomic, train_pairs, validation_rows):
    candidates = {}
    for row in train_atomic[:MAX_ATOMIC_CANDIDATES]:
        candidates[("atomic", tuple(row["rule_indices"]) if "rule_indices" in row else (row["rule_index"],))] = row
    for row in train_pairs[:MAX_PAIR_CANDIDATES]:
        candidates[("pair", tuple(row["rule_indices"]) if "rule_indices" in row else (row["left"], row["right"]))] = row

    ranked = [r for r in validation_rows if r["samples"] >= MIN_VALIDATION_RULE_SAMPLES]
    ranked.sort(key=lambda r: (r["path_win_1pct_rate_pct"] or -1, r["wilson_lower_pct"] or -1, r["samples"]), reverse=True)
    return ranked[:TOP_OUTPUT_RULES]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-05")
    ap.add_argument("--final-end", default=FINAL_END)
    ap.add_argument("--output", default=str(OUT_DIR / "path_rule_mining_latest.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.start not in dates or args.final_end not in dates:
        raise ValueError("训练区间必须落在历史数据范围内")

    train_state = FeatureState()
    train_stored, train_days, train_samples, train_atomic_stats = collect_split(
        files, args.start, TRAIN_END, train_state
    )
    train_atomic = summarize_atomic(train_atomic_stats)
    top_atomic = train_atomic[:MAX_ATOMIC_CANDIDATES]
    atomic_indices = [row["rule_index"] for row in top_atomic]

    train_pairs = rank_pair_candidates(train_stored, atomic_indices)
    top_pairs = train_pairs[:MAX_PAIR_CANDIDATES]
    pair_indices = [(row["left"], row["right"]) for row in top_pairs]

    validation_state = FeatureState()
    validation_stored, validation_days, validation_samples, _ = collect_split(
        files, VALIDATION_START, VALIDATION_END, validation_state
    )
    validation_rows = flatten_eval(validation_stored, atomic_indices, pair_indices)
    selected = choose_validation_rules(top_atomic, top_pairs, validation_rows)
    selected_indices = [(r["rule_type"], tuple(r["rule_indices"])) for r in selected]

    final_state = FeatureState()
    final_stored, final_days, final_samples, _ = collect_split(
        files, FINAL_START, args.final_end, final_state
    )
    final_atomic = [idx for typ, idxs in selected_indices if typ == "atomic" for idx in idxs]
    final_pairs = [idxs for typ, idxs in selected_indices if typ == "pair"]
    final_rows = flatten_eval(final_stored, final_atomic, final_pairs)

    train_selected_rows = []
    if selected_indices:
        train_atomic_sel = [idxs[0] for typ, idxs in selected_indices if typ == "atomic"]
        train_pairs_sel = [idxs for typ, idxs in selected_indices if typ == "pair"]
        train_selected_rows = flatten_eval(train_stored, train_atomic_sel, train_pairs_sel)

    validation_selected = selected
    final_map = {tuple(r["rule_indices"]): r for r in final_rows}
    train_map = {tuple(r["rule_indices"]): r for r in train_selected_rows}
    selected_output = []
    for row in validation_selected:
        key = tuple(row["rule_indices"])
        selected_output.append({
            "rule_type": row["rule_type"],
            "rule_indices": list(key),
            "rules": row["rules"],
            "rule_label": row["rule_label"],
            "train": train_map.get(key),
            "validation": row,
            "final": final_map.get(key),
        })

    result = {
        "schema_version": 1,
        "status": "research_only",
        "method": "path_aware_conditional_rule_mining",
        "objective": "net_profit_at_least_1pct_before_3pct_stop_within_5_sessions",
        "entry_rule": "T_plus_1_open; limit_up_entry_blocked",
        "data_start": args.start,
        "data_end": args.final_end,
        "splits": {
            "train": [args.start, TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final": [FINAL_START, args.final_end],
        },
        "parameters": {
            "net_win_threshold_pct": NET_WIN_THRESHOLD_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "max_forward_sessions": MAX_FORWARD_SESSIONS,
            "min_train_rule_samples": MIN_TRAIN_RULE_SAMPLES,
            "min_validation_rule_samples": MIN_VALIDATION_RULE_SAMPLES,
            "max_atomic_candidates": MAX_ATOMIC_CANDIDATES,
            "max_pair_candidates": MAX_PAIR_CANDIDATES,
            "rank_thresholds": list(RANK_THRESHOLDS),
            "breadth_thresholds": list(BREADTH_THRESHOLDS),
            "median_thresholds": list(MEDIAN_THRESHOLDS),
        },
        "base": {
            "train": split_summary(train_stored),
            "validation": split_summary(validation_stored),
            "final": split_summary(final_stored),
        },
        "train_top_atomic": train_atomic[:TOP_OUTPUT_RULES],
        "train_top_pairs": top_pairs[:TOP_OUTPUT_RULES],
        "selected_rules": selected_output,
        "audit": {
            "no_future_features": True,
            "path_label_target_first": True,
            "same_day_stop_first": True,
            "entry_is_T_plus_1_open": True,
            "limit_up_entry_blocked": ENTRY_LIMIT_UP_BLOCK,
            "final_holdout_used_only_after_validation_selection": True,
            "formal_production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "final_samples": final_samples,
        "selected_rules": len(selected_output),
        "top_validation_rule": selected_output[0]["validation"]["rule_label"] if selected_output else None,
        "top_validation_rate": selected_output[0]["validation"]["path_win_1pct_rate_pct"] if selected_output else None,
        "top_final_rate": selected_output[0]["final"]["path_win_1pct_rate_pct"] if selected_output and selected_output[0]["final"] else None,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
