"""直接学习可执行交易净收益的研究模型；不接触正式生产模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compound_portfolio_backtest import 解析退出
from scripts.selection_factor_catalog import load_research_config
from scripts.short_term_research import (
    FEATURE_WARMUP_SESSIONS,
    FeatureState,
    history_files,
    label_window_precedes_boundary,
    executable_entry_mask,
    read_daily,
)
from scripts.train_short_term_model import FEATURES, build_targets, make_features

CONFIG = load_research_config()
NET_TARGET_PCT = float(CONFIG["短线目标净收益百分比"])
STOP_LOSS_PCT = float(CONFIG["止损幅度百分比"])
ROUND_TRIP_COST_BPS = float(CONFIG["往返交易成本基点"])
MAX_FORWARD_SESSIONS = int(CONFIG["最大前瞻交易日数"])
TARGET_SCALE_PCT = 5.0
HIDDEN_UNITS = 32
LEARNING_RATE = 0.01
L2 = 0.001
RANDOM_SEED = 42


def _normalized_index(day: pd.DataFrame) -> pd.DataFrame:
    """生成稳定的六位证券代码索引，兼容历史数据中的带交易所前缀代码。"""
    if day.empty or "symbol" not in day.columns:
        return pd.DataFrame(index=pd.Index([], name="_symbol_key"))
    normalized = day.copy()
    normalized["_symbol_key"] = (
        normalized["symbol"].astype(str).str.extract(r"(\d{6})")[0].fillna("").str.zfill(6)
    )
    normalized = normalized.loc[normalized["_symbol_key"].ne("")]
    return normalized.drop_duplicates("_symbol_key", keep="last").set_index("_symbol_key")


def simulate_executable_net_return(
    entry_price: float,
    bars: list[dict],
    target_net_pct: float = NET_TARGET_PCT,
    stop_loss_pct: float = STOP_LOSS_PCT,
    cost_bps: float = ROUND_TRIP_COST_BPS,
) -> float | None:
    """严格按 T+1 开盘入场、T+2 起止盈止损、跳空按开盘成交的规则计算净收益。"""
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    if len(bars) != MAX_FORWARD_SESSIONS:
        return None

    for held_day, bar in enumerate(bars):
        # T+1 当日只入场，不能把入场日内触及止盈/止损算作可卖出结果。
        if held_day == 0:
            continue
        values = [bar.get(name, float("nan")) for name in ("high", "low", "close")]
        if not all(np.isfinite(value) and value > 0 for value in values):
            return None
        exit_reason, net_return, _ = 解析退出(
            float(entry_price),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
            held_day,
            float(target_net_pct),
            float(stop_loss_pct),
            float(cost_bps),
            开盘价=bar.get("open"),
        )
        if exit_reason is not None:
            return float(net_return)
    return None


def build_executable_net_returns(
    symbols: list[str],
    future_days: list[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    """给训练样本生成真实止盈/止损/到期路径净收益；不把未来行情当输入特征。"""
    _, _, _, complete = build_targets(symbols, future_days)
    returns = np.full(len(symbols), np.nan, dtype=np.float64)
    if len(future_days) != MAX_FORWARD_SESSIONS:
        raise ValueError("未来行情长度与配置中的最大持有交易日不一致")

    # 训练标签必须与组合回测使用相同的T+1开盘可执行门禁。
    # 开盘涨停、停牌或涨停价缺失的样本不能作为“可买入交易”训练标签。
    executable_entries = executable_entry_mask(symbols, future_days[0])
    indexed_days = [_normalized_index(day) for day in future_days]
    normalized_symbols = []
    for symbol in symbols:
        digits = "".join(character for character in str(symbol) if character.isdigit())
        normalized_symbols.append(digits[-6:].zfill(6) if digits else "")
    entry_day = indexed_days[0]

    for index in np.flatnonzero(complete & executable_entries):
        symbol = normalized_symbols[index]
        if symbol not in entry_day.index:
            continue
        entry_value = entry_day.loc[symbol].get("open", float("nan"))
        try:
            entry_price = float(entry_value)
        except (TypeError, ValueError):
            continue
        bars = []
        valid_path = True
        for day in indexed_days:
            if symbol not in day.index:
                valid_path = False
                break
            row = day.loc[symbol]
            bar = {}
            for name in ("open", "high", "low", "close"):
                try:
                    bar[name] = float(row.get(name, float("nan")))
                except (TypeError, ValueError):
                    bar[name] = float("nan")
            bars.append(bar)
        if not valid_path:
            continue
        result = simulate_executable_net_return(entry_price, bars)
        if result is not None and np.isfinite(result):
            returns[index] = result

    usable = complete & executable_entries & np.isfinite(returns)
    return returns, usable


class ExecutableReturnMLPRegressor:
    """小型两层感知机，回归交易规则实际产生的净收益百分比。"""

    def __init__(
        self,
        n_features: int,
        hidden_units: int = HIDDEN_UNITS,
        learning_rate: float = LEARNING_RATE,
        l2: float = L2,
        seed: int = RANDOM_SEED,
    ):
        if n_features < 1 or hidden_units < 1:
            raise ValueError("特征数和隐藏层宽度必须为正数")
        self.hidden_units = int(hidden_units)
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(
            0.0, np.sqrt(2.0 / n_features), (n_features, self.hidden_units)
        )
        self.b1 = np.zeros(self.hidden_units, dtype=np.float64)
        # 初始输出权重较小，避免在未经训练时将预测净收益推到边界。
        self.w2 = rng.normal(0.0, 0.01, self.hidden_units)
        self.b2 = 0.0
        self.updates = 0

    def _forward(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.w1.shape[0]:
            raise ValueError("净收益回归模型的输入特征数量不匹配")
        hidden = np.tanh(np.clip(matrix @ self.w1 + self.b1, -8.0, 8.0))
        scaled_return = np.tanh(np.clip(hidden @ self.w2 + self.b2, -8.0, 8.0))
        return hidden, scaled_return

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        _, scaled_return = self._forward(matrix)
        return scaled_return * TARGET_SCALE_PCT

    def update(self, matrix: np.ndarray, realized_net_returns: np.ndarray) -> float:
        matrix = np.asarray(matrix, dtype=np.float64)
        realized_net_returns = np.asarray(realized_net_returns, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.w1.shape[0]:
            raise ValueError("净收益回归模型的训练特征数量不匹配")
        if realized_net_returns.shape != (matrix.shape[0],):
            raise ValueError("净收益标签数量与样本数量不匹配")
        if len(realized_net_returns) == 0:
            return 0.0
        if not np.isfinite(matrix).all() or not np.isfinite(realized_net_returns).all():
            raise ValueError("净收益回归训练输入包含非有限值")

        target = np.clip(realized_net_returns, -TARGET_SCALE_PCT, TARGET_SCALE_PCT) / TARGET_SCALE_PCT
        hidden, predicted = self._forward(matrix)
        error = predicted - target
        loss = float(np.mean(error * error))
        # 对回归输出使用均方误差梯度，训练目标是实际净收益，不是二元命中标签。
        delta_output = (2.0 / len(target)) * error * (1.0 - predicted * predicted)
        grad_w2 = hidden.T @ delta_output + self.l2 * self.w2
        grad_b2 = float(delta_output.sum())
        delta_hidden = (delta_output[:, None] * self.w2[None, :]) * (1.0 - hidden * hidden)
        grad_w1 = matrix.T @ delta_hidden + self.l2 * self.w1
        grad_b1 = delta_hidden.sum(axis=0)

        grad_norm = float(np.sqrt(
            np.sum(grad_w1 * grad_w1)
            + np.sum(grad_b1 * grad_b1)
            + np.sum(grad_w2 * grad_w2)
            + grad_b2 * grad_b2
        ))
        if grad_norm > 5.0:
            factor = 5.0 / grad_norm
            grad_w1 *= factor
            grad_b1 *= factor
            grad_w2 *= factor
            grad_b2 *= factor

        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.updates += 1
        return loss


def load_source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"找不到分类基准研究产物：{path}")
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("status") != "research_only":
        raise ValueError("只允许读取研究模型产物")
    if source.get("audit", {}).get("formal_production_changed") is not False:
        raise ValueError("净收益回归研究不得改变正式生产模型")
    features = source.get("model", {}).get("features", [])
    if not features or features != FEATURES:
        raise ValueError("分类基准的特征清单与净收益回归训练代码不一致")
    if not all(key in source.get("splits", {}) for key in ("train", "validation", "final")):
        raise ValueError("分类基准缺少严格时间切分元数据")
    return source


def _checkpoint_signature(source: dict, files: list[Path]) -> str:
    """断点兼容签名覆盖训练代码、标签/特征依赖、配置及本次历史数据内容。"""
    dependency_paths = [
        ROOT / "scripts/train_executable_return_ranker.py",
        ROOT / "scripts/high_precision_profit_mining.py",
        ROOT / "scripts/train_short_term_model.py",
        ROOT / "scripts/short_term_research.py",
        ROOT / "scripts/market_scope.py",
        ROOT / "scripts/selection_factor_catalog.py",
        ROOT / "scripts/compound_portfolio_backtest.py",
        ROOT / "config/选股条件研究配置.json",
    ]
    digest = hashlib.sha256()
    for path in dependency_paths:
        if not path.is_file():
            raise FileNotFoundError(f"断点签名依赖缺失：{path}")
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)

    # 对历史行情做内容指纹；若同名数据被改写，旧断点不得混用。
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)

    metadata = {
        "signature_version": 2,
        "splits": source["splits"],
        "features": source["model"]["features"],
        "net_target_pct": NET_TARGET_PCT,
        "stop_loss_pct": STOP_LOSS_PCT,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "max_forward_sessions": MAX_FORWARD_SESSIONS,
        "target_scale_pct": TARGET_SCALE_PCT,
        "hidden_units": HIDDEN_UNITS,
        "learning_rate": LEARNING_RATE,
        "l2": L2,
        "seed": RANDOM_SEED,
        "history_dates": [files[0].name, files[-1].name, len(files)],
        "dependency_and_data_sha256": digest.hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def train(
    source: dict,
    model: ExecutableReturnMLPRegressor,
    files: list[Path],
    checkpoint_path: Path | None = None,
    checkpoint_every_days: int = 100,
) -> dict:
    dates = [path.name[:10] for path in files]
    train_start, train_end = source["splits"]["train"]
    validation_start = source["splits"]["validation"][0]
    if any(value not in dates for value in (train_start, train_end, validation_start)):
        raise ValueError("训练或验证边界不在现有历史行情覆盖内")
    if checkpoint_every_days < 1:
        raise ValueError("断点保存周期必须是正整数")

    start_index = dates.index(train_start)
    end_index = dates.index(train_end)
    validation_index = dates.index(validation_start)
    initial_index = max(0, start_index - FEATURE_WARMUP_SESSIONS)
    signature = _checkpoint_signature(source, files)
    cache: dict[int, pd.DataFrame] = {}

    def get_daily(index: int) -> pd.DataFrame:
        if index not in cache:
            cache[index] = read_daily(files[index])
        while len(cache) > MAX_FORWARD_SESSIONS + 1:
            del cache[next(iter(cache))]
        return cache[index]

    state = FeatureState()
    next_index = initial_index
    processed_days = 0
    sample_count = 0
    clip_count = 0
    return_sum = 0.0
    positive_count = 0
    strict_target_count = 0
    daily_losses: list[float] = []
    accumulated_seconds = 0.0
    checkpoint_was_complete = False
    resumed_from_checkpoint = False

    if checkpoint_path is not None and checkpoint_path.is_file():
        saved = None
        try:
            with checkpoint_path.open("rb") as handle:
                saved = pickle.load(handle)
        except (OSError, pickle.UnpicklingError, EOFError) as error:
            print(
                f"[执行净收益回归] 断点损坏，已作废并从训练起点重算：{error}",
                flush=True,
            )
            checkpoint_path.unlink(missing_ok=True)

        if saved is not None and (
            not isinstance(saved, dict) or saved.get("signature") != signature
        ):
            print(
                "[执行净收益回归] 断点与当前代码、配置或历史数据不兼容，已作废并重新训练",
                flush=True,
            )
            checkpoint_path.unlink(missing_ok=True)
            saved = None

        if saved is not None:
            saved_model = saved["model"]
            model.w1 = np.asarray(saved_model["w1"], dtype=np.float64)
            model.b1 = np.asarray(saved_model["b1"], dtype=np.float64)
            model.w2 = np.asarray(saved_model["w2"], dtype=np.float64)
            model.b2 = float(saved_model["b2"])
            model.updates = int(saved_model["updates"])
            if model.w1.shape != (len(source["model"]["features"]), model.hidden_units):
                raise RuntimeError("研究断点中的模型维度与当前特征不匹配")
            state = saved["feature_state"]
            next_index = int(saved["next_index"])
            processed_days = int(saved["processed_days"])
            sample_count = int(saved["sample_count"])
            clip_count = int(saved["clip_count"])
            return_sum = float(saved["return_sum"])
            positive_count = int(saved["positive_count"])
            strict_target_count = int(saved["strict_target_count"])
            daily_losses = [float(value) for value in saved["daily_losses"]][-50:]
            accumulated_seconds = float(saved["accumulated_seconds"])
            checkpoint_was_complete = bool(saved.get("complete", False))
            resumed_from_checkpoint = True
            if next_index < initial_index or next_index > end_index + 1:
                raise RuntimeError("研究断点的下一个交易日位置超出训练窗口")
            print(
                f"[执行净收益回归] 已恢复断点：完成训练日={processed_days}，"
                f"样本={sample_count}，下一个位置={next_index}，完整断点={checkpoint_was_complete}",
                flush=True,
            )

    start_time = time.time()

    def save_checkpoint(next_day_index: int, complete: bool = False) -> None:
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "signature": signature,
            "complete": complete,
            "next_index": next_day_index,
            "processed_days": processed_days,
            "sample_count": sample_count,
            "clip_count": clip_count,
            "return_sum": return_sum,
            "positive_count": positive_count,
            "strict_target_count": strict_target_count,
            "daily_losses": daily_losses[-50:],
            "accumulated_seconds": accumulated_seconds + (time.time() - start_time),
            "feature_state": state,
            "model": {
                "w1": model.w1,
                "b1": model.b1,
                "w2": model.w2,
                "b2": model.b2,
                "updates": model.updates,
            },
        }
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
        temporary_path.replace(checkpoint_path)

    if not checkpoint_was_complete:
        for index in range(next_index, end_index + 1):
            frame = state.build(get_daily(index))
            signal_date = dates[index]
            if signal_date < train_start or signal_date > train_end:
                continue
            if frame.empty or index + MAX_FORWARD_SESSIONS >= len(files):
                continue
            if not label_window_precedes_boundary(index, MAX_FORWARD_SESSIONS, validation_index):
                continue

            future_days = [
                get_daily(index + offset)
                for offset in range(1, MAX_FORWARD_SESSIONS + 1)
            ]
            symbols = frame["symbol"].astype(str).str.zfill(6).tolist()
            all_features = make_features(frame)
            net_returns, usable = build_executable_net_returns(symbols, future_days)
            keep = np.flatnonzero(usable)
            if len(keep) == 0:
                continue

            features = all_features[keep]
            targets = net_returns[keep]
            loss = model.update(features, targets)
            daily_losses.append(loss)
            daily_losses = daily_losses[-50:]
            processed_days += 1
            sample_count += int(len(targets))
            clip_count += int(np.count_nonzero(np.abs(targets) > TARGET_SCALE_PCT))
            return_sum += float(np.sum(targets))
            positive_count += int(np.count_nonzero(targets > 0.0))
            strict_target_count += int(np.count_nonzero(targets >= NET_TARGET_PCT))

            if processed_days % checkpoint_every_days == 0:
                save_checkpoint(index + 1)
                print(
                    f"[执行净收益回归] 已保存断点：训练日={processed_days}，样本={sample_count}，"
                    f"最近{len(daily_losses)}日均方损失={float(np.mean(daily_losses)):.5f}",
                    flush=True,
                )

    if processed_days == 0 or sample_count == 0:
        raise RuntimeError("净收益回归训练窗口没有生成有效样本")

    total_elapsed_seconds = accumulated_seconds + (time.time() - start_time)
    save_checkpoint(end_index + 1, complete=True)
    return {
        "start": train_start,
        "end": train_end,
        "processed_days": processed_days,
        "samples": sample_count,
        "daily_model_updates": model.updates,
        "realized_net_return_mean_pct": return_sum / sample_count,
        "realized_positive_rate_pct": positive_count / sample_count * 100.0,
        "strict_target_hit_rate_pct": strict_target_count / sample_count * 100.0,
        "target_scale_pct": TARGET_SCALE_PCT,
        "target_clipped_samples": clip_count,
        "mean_last_50_daily_losses": float(np.mean(daily_losses)) if daily_losses else None,
        "elapsed_seconds": round(total_elapsed_seconds, 2),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "checkpoint_signature": signature,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="训练直接优化可执行交易净收益的研究模型")
    parser.add_argument(
        "--model-result",
        default="data/backtest/high_precision_profit_mining_research_latest.json",
    )
    parser.add_argument(
        "--output",
        default="data/backtest/executable_return_ranker_research_latest.json",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="data/research/checkpoints/executable_return_ranker.pkl",
    )
    parser.add_argument(
        "--checkpoint-every-days",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    input_path = Path(args.model_result).resolve()
    output_path = Path(args.output).resolve()
    source = load_source(input_path)
    features = source["model"]["features"]
    baseline_parameters = source.get("parameters", {})

    model = ExecutableReturnMLPRegressor(
        n_features=len(features),
        hidden_units=HIDDEN_UNITS,
        learning_rate=LEARNING_RATE,
        l2=L2,
        seed=RANDOM_SEED,
    )
    training = train(
        source,
        model,
        history_files(),
        checkpoint_path=Path(args.checkpoint_path).resolve(),
        checkpoint_every_days=args.checkpoint_every_days,
    )
    public_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "schema_version": 1,
        "status": "research_only",
        "method": "two_layer_mlp_regression_on_realized_executable_net_return",
        "source_model_result": input_path.relative_to(ROOT).as_posix(),
        "source_public_commit": public_commit,
        "features": features,
        "splits": source["splits"],
        "parameters": {
            "hidden_units": HIDDEN_UNITS,
            "learning_rate": LEARNING_RATE,
            "l2": L2,
            "random_seed": RANDOM_SEED,
            "target_scale_pct": TARGET_SCALE_PCT,
            "loss": "mean_squared_error_on_clipped_realized_net_return",
            "net_target_pct": NET_TARGET_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "max_forward_sessions": MAX_FORWARD_SESSIONS,
        },
        "training": training,
        "model": {
            "input_weights": model.w1.tolist(),
            "hidden_bias": model.b1.tolist(),
            "output_weights": model.w2.tolist(),
            "output_bias": float(model.b2),
        },
        "audit": {
            "training_target_is_realized_executable_net_return_not_binary_hit": True,
            "training_exit_uses_T_plus_1_entry_and_T_plus_2_earliest_exit": True,
            "same_day_stop_precedes_target": True,
            "gap_fill_uses_open_price_when_worse_than_stop_or_target_price": True,
            "training_label_windows_end_before_validation": True,
            "validation_used_for_training": False,
            "final_holdout_used_for_training_or_selection": False,
            "no_probability_threshold_grid_search": True,
            "checkpoint_signature_covers_training_code_config_and_history_data": True,
            "training_checkpoint_saved_atomically_and_resumable": True,
            "formal_production_changed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": payload["status"],
        "method": payload["method"],
        "training": training,
        "audit": payload["audit"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
