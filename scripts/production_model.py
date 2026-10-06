"""第一版正式候选模型的固定权重读写与校验。"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.high_precision_profit_mining import NonlinearModel
from scripts.short_term_research import history_files, read_daily
from scripts.train_short_term_model import FEATURES, build_targets, make_features

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


def train_fixed_baseline(files, start_date: str, end_date: str) -> tuple[NonlinearModel, int, int]:
    dates = [p.name[:10] for p in files]
    start_i, end_i = dates.index(start_date), dates.index(end_date)
    model = NonlinearModel(
        len(FEATURES),
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
        x = make_features(frame)
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
        "features": FEATURES,
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
    if payload.get("features") != FEATURES:
        raise RuntimeError("正式模型特征空间与当前代码不一致")

    model = NonlinearModel(
        len(FEATURES),
        hidden=int(payload["hidden_units"]),
        learning_rate=float(payload["learning_rate"]),
        l2=float(payload["l2"]),
        seed=int(payload.get("seed", DEFAULT_SEED)),
    )
    model.w1 = __import__("numpy").array(payload["input_weights"], dtype=float)
    model.b1 = __import__("numpy").array(payload["hidden_bias"], dtype=float)
    model.w2 = __import__("numpy").array(payload["output_weights"], dtype=float)
    model.b2 = float(payload["intercept"])
    return model, payload
