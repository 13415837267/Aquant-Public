"""在GitHub云端统一执行条件挖掘与模型对比，并保存可恢复的进度清单。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.research_checkpoint_runner import canonical_hash, file_sha256, history_data_fingerprint
from scripts.selection_factor_catalog import active_market_factors, active_stock_factors, load_research_config

OUT_DIR = ROOT / "data" / "backtest"
STATE_DIR = ROOT / "data" / "research" / "all_conditions"
MANIFEST_PATH = OUT_DIR / "all_conditions_research_manifest_latest.json"
TIMEZONE = ZoneInfo("Asia/Shanghai")
START_DATE = "2015-01-05"
FINAL_DATE = "2026-09-30"

STAGE_DEPENDENCIES = {
    "single_and_double_condition_rules": ["scripts/path_rule_mining.py", "scripts/short_term_research.py", "scripts/market_scope.py", "scripts/selection_factor_catalog.py", "config/选股条件研究配置.json"],
    "market_regime_three_condition_rules": ["scripts/path_regime_rule_mining.py", "scripts/short_term_research.py", "scripts/market_scope.py", "scripts/selection_factor_catalog.py", "config/选股条件研究配置.json"],
    "private_strategy_score_thresholds": ["scripts/strict_path_strategy_score_mining.py", "scripts/short_term_research.py", "scripts/market_scope.py", "scripts/selection_factor_catalog.py", "config/选股条件研究配置.json", "scripts/train_short_term_model.py"],
    "short_term_one_percent_model": ["scripts/train_short_term_model.py", "scripts/short_term_research.py", "scripts/market_scope.py", "scripts/selection_factor_catalog.py", "config/选股条件研究配置.json"],
    "executable_three_percent_path_model": ["scripts/train_executable_path_model.py", "scripts/train_short_term_model.py", "scripts/short_term_research.py", "scripts/market_scope.py", "scripts/selection_factor_catalog.py", "config/选股条件研究配置.json"],
}


def stage_fingerprint(stage: dict, private_commit: str, history_digest: str, requirements_digest: str) -> str:
    key = stage["key"]
    dependencies = STAGE_DEPENDENCIES[key]
    return canonical_hash({
        "schema_version": 2,
        "stage_key": key,
        "command": stage["command"][1:],
        "dependencies": {name: file_sha256(ROOT / name) for name in dependencies},
        "private_strategy_commit": private_commit if key == "private_strategy_score_thresholds" else None,
        "history_data_fingerprint": history_digest,
        "requirements_sha256": requirements_digest,
        "period": {"start": START_DATE, "final_end": FINAL_DATE},
    })


def stage_resume_reason(previous: dict, expected_fingerprint: str, output: Path) -> str | None:
    if previous.get("status") != "success":
        return "没有已成功阶段记录"
    if previous.get("fingerprint") != expected_fingerprint:
        return "本阶段代码、数据或参数已变化"
    if not output.is_file():
        return "结果文件不存在"
    expected_digest = previous.get("output_sha256")
    if not isinstance(expected_digest, str) or file_sha256(output) != expected_digest:
        return "结果文件摘要不匹配"
    try:
        json.loads(output.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "结果文件不是有效JSON"
    return None


def summarize_stage_status(results: dict, stage_keys: list[str]) -> tuple[str, list[str], list[str]]:
    successful = [key for key in stage_keys if results.get(key, {}).get("status") == "success"]
    failed = [key for key in stage_keys if results.get(key, {}).get("status") == "failed"]
    if failed:
        status = "failed"
    elif len(successful) == len(stage_keys):
        status = "completed"
    else:
        status = "running"
    return status, successful, failed


def current_time() -> str:
    return datetime.now(TIMEZONE).isoformat(timespec="seconds")


def git_value(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sanitize(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(v) for v in value]
    return value


def append_log(message: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with (STATE_DIR / "运行日志.txt").open("a", encoding="utf-8") as fh:
        fh.write(f"{current_time()} {message}\n")
    print(f"{current_time()} {message}", flush=True)


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {"result_file": str(path.relative_to(ROOT)), "status": "结果文件不存在"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "result_file": str(path.relative_to(ROOT)),
            "status": "结果解析失败",
            "error": f"{type(exc).__name__}: {exc}",
        }

    summary = {
        "result_file": str(path.relative_to(ROOT)),
        "status": data.get("status"),
        "method": data.get("method"),
        "data_start": data.get("data_start", data.get("start")),
        "data_end": data.get("data_end", data.get("end")),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "audit": data.get("audit"),
    }
    for key in (
        "train", "validation", "final", "base",
        "selected_validation_operating_point", "selected_final_operating_point",
        "selected_rules", "validation_top_rules", "final_qualified_rules_ge_80pct",
        "validation_qualified_rules_ge_80pct", "final",
    ):
        if key not in data:
            continue
        value = data[key]
        if key in ("selected_rules", "validation_top_rules", "final_qualified_rules_ge_80pct",
                   "validation_qualified_rules_ge_80pct"):
            summary[key] = {
                "count": len(value) if isinstance(value, list) else None,
                "top": value[:5] if isinstance(value, list) else value,
            }
        elif key in ("train", "validation", "final", "base") and isinstance(value, dict):
            summary[key] = {
                subkey: subvalue
                for subkey, subvalue in value.items()
                if any(word in subkey.lower() for word in (
                    "sample", "rate", "win", "return", "threshold", "drawdown",
                    "trade", "qualified", "positive", "mean", "selected", "days",
                ))
            }
        else:
            summary[key] = value
    return sanitize(summary)


def build_stages(private_path: Path, private_commit: str) -> list[dict]:
    py = sys.executable
    common = ["--start", START_DATE, "--final-end", FINAL_DATE]
    strategy_env = {
        "AQUANT_PRIVATE_STRATEGY_PATH": str(private_path),
        "AQUANT_PRIVATE_STRATEGY_COMMIT": private_commit,
    }
    return [
        {
            "key": "single_and_double_condition_rules",
            "label": "单因子与双条件组合挖掘",
            "command": [py, "scripts/path_rule_mining.py", *common,
                        "--output", "data/backtest/path_rule_mining_research_latest.json"],
            "output": OUT_DIR / "path_rule_mining_research_latest.json",
            "env": {},
        },
        {
            "key": "market_regime_three_condition_rules",
            "label": "市场状态与三条件组合挖掘",
            "command": [py, "scripts/path_regime_rule_mining.py", *common,
                        "--output", "data/backtest/path_regime_rule_mining_research_latest.json"],
            "output": OUT_DIR / "path_regime_rule_mining_research_latest.json",
            "env": {},
        },
        {
            "key": "private_strategy_score_thresholds",
            "label": "正式策略评分阈值挖掘",
            "command": [py, "scripts/strict_path_strategy_score_mining.py", *common,
                        "--output", "data/backtest/strict_path_strategy_score_mining_research_latest.json"],
            "output": OUT_DIR / "strict_path_strategy_score_mining_research_latest.json",
            "env": strategy_env,
        },
        {
            "key": "short_term_one_percent_model",
            "label": "短线净收益百分之一模型训练",
            "command": [py, "scripts/train_short_term_model.py", *common,
                        "--output", "data/backtest/feature_training_research_latest.json"],
            "output": OUT_DIR / "feature_training_research_latest.json",
            "env": {},
        },
        {
            "key": "executable_three_percent_path_model",
            "label": "可执行净收益百分之三路径模型训练",
            "command": [py, "scripts/train_executable_path_model.py", *common,
                        "--output", "data/backtest/executable_path_profit_training_research_latest.json"],
            "output": OUT_DIR / "executable_path_profit_training_research_latest.json",
            "env": {},
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="运行全部条件挖掘和候选模型研究")
    parser.add_argument("--resume-manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--stage-key", choices=tuple(STAGE_DEPENDENCIES), default=None,
                        help="只运行指定阶段，以便在云端每阶段持久化检查点")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    public_commit = git_value("rev-parse", "HEAD")
    private_path = ROOT / "private-strategy"
    private_commit = ""
    if private_path.exists():
        private_commit = git_value("-C", str(private_path), "rev-parse", "HEAD")
    history_digest = history_data_fingerprint(ROOT)
    requirements_digest = file_sha256(ROOT / "requirements.txt")
    compatibility_key = canonical_hash({
        "schema_version": 2,
        "period": {"start": START_DATE, "final_end": FINAL_DATE},
        "history_data_fingerprint": history_digest,
        "requirements_sha256": requirements_digest,
    })

    manifest_path = Path(args.resume_manifest)
    previous = {}
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    same_research = (
        previous.get("schema_version") == 2
        and previous.get("compatibility_key") == compatibility_key
    )
    results = previous.get("stages", {}) if same_research else {}
    manifest = {
        "schema_version": 2,
        "status": "running",
        "method": "all_condition_and_model_research",
        "started_at": current_time(),
        "updated_at": current_time(),
        "timezone": "Asia/Shanghai",
        "public_commit": public_commit,
        "private_strategy_commit": private_commit or None,
        "research_key": compatibility_key,
        "compatibility_key": compatibility_key,
        "input_fingerprints": {
            "history_data": history_digest,
            "requirements": requirements_digest,
        },
        "period": {"start": START_DATE, "final_end": FINAL_DATE},
        "resume_source_matched": same_research,
        "factor_configuration": {"active_stock_factors": active_stock_factors(), "active_market_factors": active_market_factors(), "settings": load_research_config()},
        "stages": results,
        "formal_production_changed": False,
    }
    write_json(MANIFEST_PATH, manifest)
    append_log(f"研究启动；公开提交={public_commit}；私有策略提交={private_commit or '未检出'}")

    failures = []
    all_stages = build_stages(private_path, private_commit)
    all_stage_keys = [stage["key"] for stage in all_stages]
    stages_to_run = (
        [stage for stage in all_stages if stage["key"] == args.stage_key]
        if args.stage_key else all_stages
    )
    for stage in stages_to_run:
        key = stage["key"]
        output = stage["output"]
        old = results.get(key, {})
        fingerprint = stage_fingerprint(stage, private_commit, history_digest, requirements_digest)
        if same_research:
            reuse_reason = stage_resume_reason(old, fingerprint, output)
            if reuse_reason is None:
                append_log(f"断点恢复：复用已验证阶段“{stage['label']}”")
                continue
            if old:
                append_log(f"检查点不可复用：{stage['label']}；原因={reuse_reason}")

        append_log(f"阶段开始：{stage['label']}")
        env = os.environ.copy()
        env.update(stage["env"])
        started = time.monotonic()
        try:
            with subprocess.Popen(
                stage["command"],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as proc:
                log_path = STATE_DIR / f"{key}.日志.txt"
                with log_path.open("w", encoding="utf-8") as log_file:
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log_file.write(line)
                        log_file.flush()
                    return_code = proc.wait()

            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, stage["command"])
            if not output.exists():
                raise FileNotFoundError(f"研究脚本未生成结果：{output}")
            results[key] = {
                "status": "success",
                "label": stage["label"],
                "started_at": current_time(),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "fingerprint": fingerprint,
                "output_sha256": file_sha256(output),
                "summary": load_summary(output),
            }
            append_log(f"阶段完成：{stage['label']}；耗时={results[key]['elapsed_seconds']}秒")
        except Exception as exc:
            results[key] = {
                "status": "failed",
                "label": stage["label"],
                "failed_at": current_time(),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(key)
            append_log(f"阶段失败：{stage['label']}；错误={type(exc).__name__}: {exc}")
        manifest["stages"] = results
        manifest["updated_at"] = current_time()
        manifest["completed_stage_count"] = sum(v.get("status") == "success" for v in results.values())
        manifest["failed_stage_count"] = sum(v.get("status") == "failed" for v in results.values())
        manifest["last_completed_or_failed_stage"] = key
        write_json(MANIFEST_PATH, sanitize(manifest))

    status, successful_stage_keys, failed_stage_keys = summarize_stage_status(results, all_stage_keys)
    manifest["status"] = status
    if status in ("failed", "completed"):
        manifest["finished_at"] = current_time()
    else:
        manifest.pop("finished_at", None)
    manifest["completed_stage_count"] = len(successful_stage_keys)
    manifest["failed_stage_count"] = len(failed_stage_keys)
    manifest["failed_stages"] = failed_stage_keys
    manifest["stages"] = results
    manifest["metrics_summary"] = {
        key: value.get("summary", {"status": value.get("status"), "error": value.get("error")})
        for key, value in results.items()
    }
    manifest = sanitize(manifest)
    write_json(MANIFEST_PATH, manifest)

    append_log(f"阶段调用结束；总研究状态={manifest['status']}；失败阶段={failed_stage_keys}")
    print(json.dumps({
        "status": manifest["status"],
        "successful_stages": len(successful_stage_keys),
        "failed_stages": failed_stage_keys,
        "executed_stage": args.stage_key or "全部阶段",
        "manifest": str(MANIFEST_PATH),
        "formal_production_changed": False,
    }, ensure_ascii=False), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
