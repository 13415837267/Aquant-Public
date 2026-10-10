"""第一版正式候选模型的固定权重读写与校验。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.high_precision_profit_mining import NonlinearModel
from scripts.selection_factor_catalog import BASE_STOCK_FACTORS, transform_market_feature
from scripts.short_term_research import history_files, read_daily
from scripts.train_short_term_model import build_targets

MODEL_SCHEMA_VERSION = 1
MODEL_TYPE = "numpy_two_layer_mlp"
DEFAULT_MODEL_PATH = Path("data/models/production_v1.json")
DEFAULT_TRAINING_START = "2019-01-02"
DEFAULT_TRAINING_END = "2026-09-30"
DEFAULT_MODEL_CODE_COMMIT = "3df9ef34ee23a86d6f844ceb9e96bf4b4f387592"
DEFAULT_MODEL_VERSION = "近期训练窗口研究版"
DEFAULT_MAX_FORWARD_SESSIONS = 5
DEFAULT_SEED = 42
DEFAULT_HIDDEN = 32
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_L2 = 0.001
PRODUCTION_V1_MARKET_FEATURES = ("market_breadth_pct", "market_median_return_pct")
PRODUCTION_V1_FEATURES = tuple(BASE_STOCK_FACTORS) + PRODUCTION_V1_MARKET_FEATURES


def _percentile_rank(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True).fillna(0.5).to_numpy(dtype=np.float64)


def make_production_features(frame):
    """按第一版正式模型固定的17个个股因子和2个市场因子构造输入。"""
    if frame.empty:
        return np.empty((0, len(PRODUCTION_V1_FEATURES)), dtype=np.float64)
    missing = [name for name in PRODUCTION_V1_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(f"第一版正式模型缺少输入特征：{missing}")
    columns = [_percentile_rank(frame[name]) for name in BASE_STOCK_FACTORS]
    columns.extend(transform_market_feature(frame, name) for name in PRODUCTION_V1_MARKET_FEATURES)
    return np.column_stack(columns)


def train_fixed_baseline(files, start_date: str, end_date: str) -> tuple[NonlinearModel, int, int]:
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start_date), dates.index(end_date)
    model = NonlinearModel(
        len(PRODUCTION_V1_FEATURES),
        hidden=DEFAULT_HIDDEN,
        learning_rate=DEFAULT_LEARNING_RATE,
        l2=DEFAULT_L2,
        seed=DEFAULT_SEED,
    )
    state = __import__("scripts.short_term_research", fromlist=["FeatureState"]).FeatureState()
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = read_daily(files[i])
        return cache[i]

    days = 0
    samples = 0
    for i in range(max(0, start_i - 20), end_i + 1):
        date = dates[i]
        frame = state.build(get(i))
        if date < start_date or date > end_date or frame.empty:
            continue
        if i + DEFAULT_MAX_FORWARD_SESSIONS >= len(files):
            continue
        futures = [get(i + j) for j in range(1, DEFAULT_MAX_FORWARD_SESSIONS + 1)]
        x = make_production_features(frame)
        labels, _, _, complete = build_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), futures
        )
        keep = complete.nonzero()[0]
        if len(keep) == 0:
            continue
        model.update(x[keep], labels[keep].astype(float))
        days += 1
        samples += len(keep)
    return model, days, samples


def model_payload(model: NonlinearModel, training_start: str, training_end: str, source_commit: str = DEFAULT_MODEL_CODE_COMMIT) -> dict:
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "status": "active_baseline",
        "model_type": MODEL_TYPE,
        "strategy_version": DEFAULT_MODEL_VERSION,
        "model_code_commit": source_commit,
        "training_start": training_start,
        "training_end": training_end,
        "data_cutoff": training_end,
        "seed": DEFAULT_SEED,
        "hidden_units": int(model.hidden),
        "learning_rate": float(model.learning_rate),
        "l2": float(model.l2),
        "features": list(PRODUCTION_V1_FEATURES),
        "input_weights": model.w1.tolist(),
        "hidden_bias": model.b1.tolist(),
        "output_weights": model.w2.tolist(),
        "intercept": float(model.b2),
        "candidate_policy": "top_2_by_recent_window_model_probability_threshold_0.60",
        "audit": {
            "fixed_model_for_daily_inference": True,
            "daily_retraining": False,
            "historical_replay_uses_signal_date_data_cutoff": True,
        },
    }


def write_model(path: Path, model: NonlinearModel, training_start: str, training_end: str, source_commit: str = DEFAULT_MODEL_CODE_COMMIT) -> None:
    payload = model_payload(model, training_start, training_end, source_commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model(path: Path, expected_version: str | None = None, expected_commit: str | None = None) -> tuple[NonlinearModel, dict]:
    if not path.exists():
        raise FileNotFoundError(f"正式模型文件不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "active_baseline":
        raise RuntimeError("正式模型状态不是 active_baseline")
    if payload.get("model_type") != MODEL_TYPE:
        raise RuntimeError("正式模型类型不一致")
    if expected_version and payload.get("strategy_version") != expected_version:
        raise RuntimeError("正式模型版本与生产基准不一致")
    if expected_commit and payload.get("model_code_commit") != expected_commit:
        raise RuntimeError("正式模型代码提交号与生产基准不一致")
    if payload.get("features") != list(PRODUCTION_V1_FEATURES):
        raise RuntimeError("正式模型特征空间与第一版固定特征定义不一致")

    hidden_units = int(payload["hidden_units"])
    input_weights = np.asarray(payload["input_weights"], dtype=float)
    hidden_bias = np.asarray(payload["hidden_bias"], dtype=float)
    output_weights = np.asarray(payload["output_weights"], dtype=float)
    if (
        input_weights.shape != (len(PRODUCTION_V1_FEATURES), hidden_units)
        or hidden_bias.shape != (hidden_units,)
        or output_weights.shape != (hidden_units,)
    ):
        raise RuntimeError("正式模型权重维度与第一版固定特征定义不一致")

    model = NonlinearModel(
        len(PRODUCTION_V1_FEATURES),
        hidden=hidden_units,
        learning_rate=float(payload["learning_rate"]),
        l2=float(payload["l2"]),
        seed=int(payload.get("seed", DEFAULT_SEED)),
    )
    model.w1 = input_weights
    model.b1 = hidden_bias
    model.w2 = output_weights
    model.b2 = float(payload["intercept"])
    return model, payload
