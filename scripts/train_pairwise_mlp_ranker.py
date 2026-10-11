"""在固定的短线交易标签与特征上训练非线性成对排序挑战模型。

研究目标是与同结构的非线性分类模型比较排序损失的价值；不调整概率阈值，不修改正式模型。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.high_precision_profit_mining import (
    FEATURES,
    MAX_FORWARD_SESSIONS,
    NET_WIN_THRESHOLD_PCT,
    RESEARCH_CONFIG,
    build_targets,
    make_features,
)
from scripts.short_term_research import (
    FEATURE_WARMUP_SESSIONS,
    FeatureState,
    history_files,
    label_window_precedes_boundary,
    read_daily,
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


class PairwiseMLPRankingModel:
    """使用两层感知机和逐交易日正负样本对的逻辑排序损失。"""

    def __init__(
        self,
        n_features: int,
        hidden_units: int = 32,
        learning_rate: float = 0.03,
        l2: float = 0.001,
        max_positives_per_day: int = 32,
        max_negatives_per_day: int = 96,
        seed: int = 42,
    ):
        if n_features < 1 or hidden_units < 1:
            raise ValueError("特征数和隐藏层宽度必须为正数")
        if max_positives_per_day < 1 or max_negatives_per_day < 1:
            raise ValueError("每日正负样本上限必须为正数")
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_positives_per_day = max_positives_per_day
        self.max_negatives_per_day = max_negatives_per_day
        # 初始化和采样使用独立随机数流，令同种子下的初始权重、样本对抽样均可复现。
        init_rng = np.random.default_rng(seed)
        self.w1 = init_rng.normal(
            0.0, np.sqrt(2.0 / n_features), (n_features, hidden_units)
        )
        self.b1 = np.zeros(hidden_units, dtype=np.float64)
        self.w2 = init_rng.normal(
            0.0, np.sqrt(2.0 / hidden_units), hidden_units
        )
        self.b2 = 0.0
        self.rng = np.random.default_rng(seed)
        self.pair_updates = 0
        self.pairs_seen = 0

    def score(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.w1.shape[0]:
            raise ValueError("非线性成对排序模型的输入特征数量不匹配")
        hidden = np.tanh(np.clip(matrix @ self.w1 + self.b1, -8.0, 8.0))
        return hidden @ self.w2 + self.b2

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        # Sigmoid 单调，不改变排序关系；下游固定按排名选股，不使用概率阈值。
        return sigmoid(self.score(matrix))

    def update(self, matrix: np.ndarray, labels: np.ndarray) -> int:
        matrix = np.asarray(matrix, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.w1.shape[0]:
            raise ValueError("训练输入特征数量不匹配")
        if labels.shape != (matrix.shape[0],):
            raise ValueError("训练标签数量与样本数量不匹配")

        positive_idx = np.flatnonzero(labels > 0.5)
        negative_idx = np.flatnonzero(labels <= 0.5)
        if len(positive_idx) == 0 or len(negative_idx) == 0:
            return 0
        if len(positive_idx) > self.max_positives_per_day:
            positive_idx = self.rng.choice(
                positive_idx, size=self.max_positives_per_day, replace=False
            )
        if len(negative_idx) > self.max_negatives_per_day:
            negative_idx = self.rng.choice(
                negative_idx, size=self.max_negatives_per_day, replace=False
            )

        positive = matrix[positive_idx]
        negative = matrix[negative_idx]
        positive_hidden = np.tanh(
            np.clip(positive @ self.w1 + self.b1, -8.0, 8.0)
        )
        negative_hidden = np.tanh(
            np.clip(negative @ self.w1 + self.b1, -8.0, 8.0)
        )
        positive_score = positive_hidden @ self.w2 + self.b2
        negative_score = negative_hidden @ self.w2 + self.b2
        pair_weights = sigmoid(
            -(positive_score[:, None] - negative_score[None, :])
        )
        pair_count = int(pair_weights.size)
        if pair_count == 0:
            return 0

        # 对所有正负样本对求平均梯度，避免显式构造“样本对×特征×隐藏层”张量。
        positive_score_grad = -pair_weights.sum(axis=1) / pair_count
        negative_score_grad = pair_weights.sum(axis=0) / pair_count
        grad_w2 = (
            positive_hidden.T @ positive_score_grad
            + negative_hidden.T @ negative_score_grad
            + self.l2 * self.w2
        )
        grad_b2 = float(positive_score_grad.sum() + negative_score_grad.sum())

        positive_hidden_grad = positive_score_grad[:, None] * self.w2[None, :]
        negative_hidden_grad = negative_score_grad[:, None] * self.w2[None, :]
        positive_pre_grad = positive_hidden_grad * (1.0 - positive_hidden**2)
        negative_pre_grad = negative_hidden_grad * (1.0 - negative_hidden**2)
        grad_w1 = (
            positive.T @ positive_pre_grad
            + negative.T @ negative_pre_grad
            + self.l2 * self.w1
        )
        grad_b1 = positive_pre_grad.sum(axis=0) + negative_pre_grad.sum(axis=0)

        grad_norm = float(
            np.sqrt(
                np.sum(grad_w1 * grad_w1)
                + np.sum(grad_b1 * grad_b1)
                + np.sum(grad_w2 * grad_w2)
                + grad_b2 * grad_b2
            )
        )
        if grad_norm > 5.0:
            scale = 5.0 / grad_norm
            grad_w1 *= scale
            grad_b1 *= scale
            grad_w2 *= scale
            grad_b2 *= scale

        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.pair_updates += 1
        self.pairs_seen += pair_count
        return pair_count


def load_source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到分类基准研究产物：{path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("status") != "research_only":
        raise ValueError("只允许读取研究模型产物")
    if source.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("研究训练不得改变正式生产模型")
    linear = source.get("pairwise_ranker", {})
    if linear.get("audit", {}).get("threshold_tuning_used") is not False:
        raise ValueError("分类基准产物缺少可复核的线性成对排序对照")
    features = source.get("model", {}).get("features", [])
    if not features or features != linear.get("features"):
        raise ValueError("分类模型与线性排序模型的特征清单不一致")
    if features != FEATURES:
        raise ValueError("研究产物特征清单与当前训练代码不一致")
    return source


def train(source: dict, model: PairwiseMLPRankingModel, files: list[Path]) -> dict:
    dates = [path.name[:10] for path in files]
    train_start, train_end = source["splits"]["train"]
    validation_start = source["splits"]["validation"][0]
    if train_start not in dates or train_end not in dates or validation_start not in dates:
        raise ValueError("训练或验证边界不在现有历史行情覆盖内")
    start_index = dates.index(train_start)
    end_index = dates.index(train_end)
    validation_index = dates.index(validation_start)
    cache: dict[int, object] = {}

    def get_daily(index: int):
        if index not in cache:
            cache[index] = read_daily(files[index])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[index]

    state = FeatureState()
    processed_days = 0
    sample_count = 0
    positive_count = 0
    start_time = time.time()

    for index in range(max(0, start_index - FEATURE_WARMUP_SESSIONS), end_index + 1):
        current_date = dates[index]
        frame = state.build(get_daily(index))
        if current_date < train_start or current_date > train_end:
            continue
        if frame.empty or index + MAX_FORWARD_SESSIONS >= len(files):
            continue
        if not label_window_precedes_boundary(
            index, MAX_FORWARD_SESSIONS, validation_index
        ):
            continue

        future_days = [
            get_daily(index + offset)
            for offset in range(1, MAX_FORWARD_SESSIONS + 1)
        ]
        all_features = make_features(frame)
        labels, _, _, complete = build_targets(
            frame["symbol"].astype(str).str.zfill(6).tolist(), future_days
        )
        keep = np.flatnonzero(complete)
        if len(keep) == 0:
            continue
        features = all_features[keep]
        labels = labels[keep].astype(np.float64)
        if features.shape[1] != model.w1.shape[0]:
            raise ValueError("实际训练特征维度与锁定特征清单不一致")

        model.update(features, labels)
        processed_days += 1
        sample_count += int(len(labels))
        positive_count += int(labels.sum())
        if processed_days % 50 == 0:
            print(
                f"[非线性成对排序] 已处理交易日={processed_days}，"
                f"样本={sample_count}，正样本={positive_count}",
                flush=True,
            )

    if processed_days == 0 or sample_count == 0:
        raise RuntimeError("训练窗口没有生成有效样本")
    return {
        "start": train_start,
        "end": train_end,
        "processed_days": processed_days,
        "samples": sample_count,
        "positive_samples": positive_count,
        "positive_rate_pct": positive_count / sample_count * 100.0,
        "pair_updates": model.pair_updates,
        "pairs_seen": model.pairs_seen,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="训练非线性成对排序研究模型")
    parser.add_argument(
        "--model-result",
        default="data/backtest/high_precision_profit_mining_research_latest.json",
    )
    parser.add_argument(
        "--output",
        default="data/backtest/pairwise_mlp_ranker_research_latest.json",
    )
    args = parser.parse_args()
    input_path = (ROOT / args.model_result).resolve()
    output_path = (ROOT / args.output).resolve()
    source = load_source(input_path)

    baseline_parameters = source.get("parameters", {})
    linear_parameters = source["pairwise_ranker"].get("parameters", {})
    features = source["model"]["features"]
    hidden_units = int(baseline_parameters["hidden_units"])
    learning_rate = float(baseline_parameters["learning_rate"])
    l2 = float(baseline_parameters["l2"])
    max_positives = int(linear_parameters["max_positives_per_day"])
    max_negatives = int(linear_parameters["max_negatives_per_day"])
    seed = int(linear_parameters["random_seed"])

    model = PairwiseMLPRankingModel(
        n_features=len(features),
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        l2=l2,
        max_positives_per_day=max_positives,
        max_negatives_per_day=max_negatives,
        seed=seed,
    )
    files = history_files()
    training = train(source, model, files)
    public_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "schema_version": 1,
        "status": "research_only",
        "method": "daily_cross_sectional_pairwise_mlp",
        "source_model_result": input_path.relative_to(ROOT).as_posix(),
        "source_public_commit": public_commit,
        "features": features,
        "parameters": {
            "hidden_units": hidden_units,
            "learning_rate": learning_rate,
            "l2": l2,
            "max_positives_per_day": max_positives,
            "max_negatives_per_day": max_negatives,
            "random_seed": seed,
            "loss": "average_pairwise_logistic_loss",
        },
        "training": training,
        "model": {
            "input_weights": model.w1.tolist(),
            "hidden_bias": model.b1.tolist(),
            "output_weights": model.w2.tolist(),
            "output_bias": float(model.b2),
        },
        "audit": {
            "same_feature_list_as_classification_baseline": True,
            "same_mlp_architecture_and_initialization_as_classification_baseline": True,
            "same_pair_sampling_limits_and_seed_as_linear_ranker": True,
            "same_binary_labels_as_linear_ranker": True,
            "labels_use_T_plus_1_open_entry_and_T_plus_2_earliest_exit": True,
            "stop_loss_precedes_target_if_both_hit_same_day": True,
            "training_label_windows_end_before_validation": True,
            "validation_used_for_training": False,
            "final_holdout_used_for_training_or_selection": False,
            "probability_threshold_tuning_used": False,
            "formal_production_changed": False,
        },
        "target_definition": {
            "net_profit_threshold_pct": NET_WIN_THRESHOLD_PCT,
            "stop_loss_pct": float(RESEARCH_CONFIG["止损幅度百分比"]),
            "round_trip_cost_bps": float(RESEARCH_CONFIG["往返交易成本基点"]),
            "max_forward_sessions": MAX_FORWARD_SESSIONS,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method": payload["method"],
                "training": payload["training"],
                "parameters": payload["parameters"],
                "audit": payload["audit"],
                "formal_production_changed": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
