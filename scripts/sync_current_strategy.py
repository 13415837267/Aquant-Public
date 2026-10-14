"""每次候选运行动态读取 Aquant-Private/main 最新策略，并同步本轮审计元数据。"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
MAX_CANDIDATES_HARD_LIMIT = 100


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取有效 JSON：{path.relative_to(ROOT)}") from exc


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_latest_strategy(private_root: Path) -> dict:
    if not private_root.is_dir():
        raise RuntimeError(f"找不到最新私有策略目录：{private_root}")
    version_file = private_root / "strategy" / "version.py"
    config_file = private_root / "config" / "strategy.yaml"
    if not version_file.is_file() or not config_file.is_file():
        raise RuntimeError("最新私有策略缺少版本文件或配置文件。")

    version_text = version_file.read_text(encoding="utf-8")
    declared_version = None
    for line in version_text.splitlines():
        if line.startswith("STRATEGY_VERSION") and "=" in line:
            declared_version = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if not declared_version:
        raise RuntimeError("无法从最新私有策略读取 STRATEGY_VERSION。")

    config_text = config_file.read_text(encoding="utf-8")
    configured_version = None
    for line in config_text.splitlines():
        if line.strip().startswith("version:"):
            configured_version = line.split(":", 1)[1].strip().strip('"').strip("'")
            break
    if configured_version != declared_version:
        raise RuntimeError("最新私有策略的代码版本与策略配置版本不一致。")

    commit = subprocess.check_output(
        ["git", "-C", str(private_root), "rev-parse", "HEAD"], text=True
    ).strip()
    root_text = str(private_root.resolve())
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    version_module = importlib.import_module("strategy.version")
    model_module = importlib.import_module("strategy.model")
    precision_module = importlib.import_module("strategy.precision_config")
    version = str(version_module.STRATEGY_VERSION)
    if version != declared_version:
        raise RuntimeError("最新私有策略版本解析结果不一致。")

    max_candidates = int(precision_module.MAX_CANDIDATES)
    if not 1 <= max_candidates <= MAX_CANDIDATES_HARD_LIMIT:
        raise RuntimeError(f"最新策略最大候选数超出系统安全范围：{max_candidates}。")
    return {
        "version": version,
        "commit": commit,
        "candidate_admission_policy": str(model_module.ADMISSION_POLICY),
        "candidate_count_max": max_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="动态同步 Private/main 最新策略版本，不固定历史提交。")
    parser.add_argument("--private-strategy", type=Path, required=True)
    args = parser.parse_args()

    strategy = load_latest_strategy(args.private_strategy.resolve())
    manifest_path = ROOT / "config" / "system_version.json"
    trial_path = ROOT / "data" / "research_trial_status.json"
    manifest = read_json(manifest_path)
    trial = read_json(trial_path)
    previous_version = str(manifest.get("strategy_version") or "")
    previous_commit = str(manifest.get("strategy_commit") or "")
    changed = previous_version != strategy["version"] or previous_commit != strategy["commit"]

    manifest.update({
        "system_version": strategy["version"],
        "strategy_version": strategy["version"],
        "lifecycle_status": "research",
        "release_status": "not_released",
        "strategy_source_repo": "13415837267/Aquant-Private",
        "strategy_source_branch": "main",
        "strategy_commit": strategy["commit"],
        "version_policy": "每次候选运行动态读取 Aquant-Private/main 最新提交；版本号和提交号仅记录本次快照来源，不固定历史策略。",
        "current_model": "Aquant-Private/strategy/model.py",
        "candidate_admission_policy": strategy["candidate_admission_policy"],
        "candidate_count_max": strategy["candidate_count_max"],
        "production_release": False,
        "automatic_ordering": False,
        "note": (
            f"本轮候选运行使用 Aquant-Private/main 最新策略版本 {strategy['version']}，"
            f"来源提交 {strategy['commit']}。此提交号仅用于本轮候选、计划与审计追溯；"
            "后续运行将重新读取 main 最新提交。研究候选不代表正式发布。"
        ),
    })

    trial_changed = (
        str(trial.get("strategy_version") or "") != strategy["version"]
        or str(trial.get("strategy_commit") or "") != strategy["commit"]
    )
    now_date = datetime.now(TIMEZONE).date().isoformat()
    trial.update({
        "system_version": strategy["version"],
        "strategy_version": strategy["version"],
        "strategy_source": "Aquant-Private/main",
        "strategy_commit": strategy["commit"],
        "release_status": "not_released",
        "production_release": False,
        "timezone": "Asia/Shanghai",
    })
    if changed or trial_changed:
        trial["status"] = "research_pending"
        trial["as_of"] = now_date
        trial["note"] = (
            f"已自动跟随 Aquant-Private/main 最新策略版本 {strategy['version']}，"
            f"本轮来源提交为 {strategy['commit']}。候选生产立即使用当前策略，"
            "研究未完成不阻断候选生成；旧提交的验证证据不会转移给新提交，"
            "正式发布门禁与自动下单关闭状态保持不变。"
        )

    write_json(manifest_path, manifest)
    write_json(trial_path, trial)
    print(json.dumps({
        "status": "synced",
        "strategy_version": strategy["version"],
        "strategy_commit": strategy["commit"],
        "candidate_admission_policy": strategy["candidate_admission_policy"],
        "candidate_count_max": strategy["candidate_count_max"],
        "strategy_changed_since_previous_snapshot": changed,
        "research_trial_reset": bool(changed or trial_changed),
        "production_release": False,
        "automatic_ordering": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
