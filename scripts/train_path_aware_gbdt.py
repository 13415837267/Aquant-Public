"""云端路径感知梯度提升模型研究。

仅用于研究，不直接修改生产策略。训练标签严格遵守 T+1 开盘入场、T+2 起退出、
先触发 -3% 止损则失败、5 个交易日内净收益达到 +1% 才算成功。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.short_term_research import FeatureState, history_files, read_daily
from scripts.train_short_term_model import FEATURES, build_targets, make_features

TRAIN_END = "2022-12-30"
VALIDATION_START = "2023-01-03"
VALIDATION_END = "2024-12-31"
FINAL_START = "2025-01-02"
FINAL_END = "2026-09-30"
MAX_FORWARD_SESSIONS = 5
NET_WIN_THRESHOLD_PCT = 1.0
SEED = 42
THRESHOLDS = tuple(np.arange(0.50, 0.991, 0.01))
MIN_OPERATING_SAMPLES = 1000
MIN_OPERATING_SAMPLE_SHARE_PCT = 1.0


def add_training_samples(buffers, x, y):
    for cls in (0, 1):
        rows = x[y == cls]
        if len(rows):
            buffers[cls].extend(rows)


def collect_training(files, dates, train_end):
    buffers = {0: [], 1: []}
    state = FeatureState()
    start_i, end_i = dates.index(dates[0]), dates.index(train_end)
    processed = samples = 0
    for i in range(start_i, end_i + 1):
        date = dates[i]
        frame = state.build(read_daily(files[i]))
        if date < dates[0] or date > train_end or frame.empty or i + MAX_FORWARD_SESSIONS >= len(files):
            continue
        future = [read_daily(files[i + j]) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        labels, _, _, complete = build_targets(frame["symbol"].astype(str).str.zfill(6).tolist(), future)
        keep = np.flatnonzero(complete)
        if len(keep):
            add_training_samples(buffers, make_features(frame)[keep], labels[keep].astype(np.int8))
            samples += len(keep)
        processed += 1
        if processed % 25 == 0:
            print(f"[路径感知训练] 已处理{processed}日，原始样本{samples}，保留训练样本{len(buffers[0])+len(buffers[1])}", flush=True)
    x = np.asarray(buffers[0] + buffers[1], dtype=np.float32)
    y = np.asarray([0] * len(buffers[0]) + [1] * len(buffers[1]), dtype=np.int8)
    return x, y, processed, samples


def collect_eval(files, dates, start, end, model):
    state = FeatureState()
    metrics = []
    start_i, end_i = dates.index(start), dates.index(end)
    processed = samples = 0
    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(read_daily(files[i]))
        if date < start or date > end or frame.empty or i + MAX_FORWARD_SESSIONS >= len(files):
            continue
        future = [read_daily(files[i + j]) for j in range(1, MAX_FORWARD_SESSIONS + 1)]
        labels, _, _, complete = build_targets(frame["symbol"].astype(str).str.zfill(6).tolist(), future)
        keep = np.flatnonzero(complete)
        if len(keep):
            x = make_features(frame)[keep]
            y = labels[keep].astype(np.int8)
            p = model.predict_proba(x)[:, 1]
            metrics.append((date, p, y))
            samples += len(y)
        processed += 1
        if processed % 25 == 0:
            print(f"[路径感知评估] {start}-{end} 已处理{processed}日，样本{samples}", flush=True)
    return metrics, processed, samples


def rows_for(metrics):
    if not metrics:
        return []
    ps = np.concatenate([m[1] for m in metrics])
    ys = np.concatenate([m[2] for m in metrics])
    total = len(ys)
    out = []
    for threshold in THRESHOLDS:
        mask = ps >= threshold
        n = int(mask.sum())
        wins = int(ys[mask].sum())
        out.append({
            "probability_threshold": round(float(threshold), 2),
            "samples": n,
            "sample_share_pct": float(n / total * 100.0) if total else 0.0,
            "wins": wins,
            "hit_1pct_rate_pct": float(wins / n * 100.0) if n else None,
        })
    return out


def select(rows):
    eligible = [
        r for r in rows
        if r["samples"] >= MIN_OPERATING_SAMPLES
        and r["sample_share_pct"] >= MIN_OPERATING_SAMPLE_SHARE_PCT
    ]
    return max(eligible, key=lambda r: (r["hit_1pct_rate_pct"], r["samples"])) if eligible else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "data/backtest/path_aware_gbdt_research.json"))
    ap.add_argument("--train-end", default=TRAIN_END)
    ap.add_argument("--fixed-threshold", type=float, default=None)
    args = ap.parse_args()

    started = time.time()
    files = history_files()
    dates = [p.name[:10] for p in files]
    if args.train_end not in dates:
        raise SystemExit(f"训练截止日不在交易日数据中: {args.train_end}")
    x, y, train_days, train_samples = collect_training(files, dates, args.train_end)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(x, y)

    validation, validation_days, validation_samples = collect_eval(files, dates, VALIDATION_START, VALIDATION_END, model)
    final, final_days, final_samples = collect_eval(files, dates, FINAL_START, FINAL_END, model)

    validation_rows = rows_for(validation)
    final_rows = rows_for(final)
    if args.fixed_threshold is not None:
        selected = next(
            (r for r in validation_rows if r["probability_threshold"] == round(args.fixed_threshold, 2)),
            None,
        )
        if selected is None:
            raise SystemExit(f"固定阈值必须落在研究阈值网格内: {args.fixed_threshold}")
        threshold_for_final = selected["probability_threshold"]
    else:
        selected = select(validation_rows)
        threshold_for_final = selected["probability_threshold"] if selected else None
    selected_final = None
    if threshold_for_final is not None:
        selected_final = next(
            (r for r in final_rows if r["probability_threshold"] == threshold_for_final),
            None,
        )

    result = {
        "schema_version": 2,
        "status": "research_only",
        "method": "hist_gradient_boosting_path_aware",
        "objective": "net_profit_at_least_1pct_before_3pct_stop_within_5_sessions",
        "features": FEATURES,
        "parameters": {
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
            "seed": SEED,
            "full_training_samples": True,
            "train_end": args.train_end,
            "fixed_threshold": args.fixed_threshold,
            "min_operating_samples": MIN_OPERATING_SAMPLES,
            "min_operating_sample_share_pct": MIN_OPERATING_SAMPLE_SHARE_PCT,
        },
        "train": {"processed_days": train_days, "raw_samples": train_samples, "samples": int(len(y)), "positive_rate_pct": float(y.mean() * 100.0)},
        "validation": {"processed_days": validation_days, "samples": validation_samples, "thresholds": validation_rows},
        "final": {"processed_days": final_days, "samples": final_samples, "thresholds": final_rows},
        "selected_validation_operating_point": selected,
        "selected_final_operating_point": selected_final,
        "audit": {
            "no_future_features": True,
            "entry_is_T_plus_1_open": True,
            "exit_starts_T_plus_2": True,
            "strict_stop_first_managed_label": True,
            "final_holdout_used_once_after_validation_selection": args.fixed_threshold is None,
            "threshold_selection_source": "previous_baseline" if args.fixed_threshold is not None else "validation_selection",
            "minimum_operating_sample_guard": True,
            "production_changed": False,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "selected_validation_operating_point": selected,
        "selected_final_operating_point": selected_final,
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
