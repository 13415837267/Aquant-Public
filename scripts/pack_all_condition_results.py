"""将全部条件挖掘原始结果打包到既有研究工件，方便完整复核。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "backtest"
MANIFEST = OUT_DIR / "all_conditions_research_manifest_latest.json"
TARGET = OUT_DIR / "high_precision_profit_mining_research_latest.json"

RESULTS = {
    "single_and_double_condition_rules": OUT_DIR / "path_rule_mining_research_latest.json",
    "market_regime_three_condition_rules": OUT_DIR / "path_regime_rule_mining_research_latest.json",
    "private_strategy_score_thresholds": OUT_DIR / "strict_path_strategy_score_mining_research_latest.json",
    "short_term_one_percent_model": OUT_DIR / "feature_training_research_latest.json",
    "executable_three_percent_path_model": OUT_DIR / "executable_path_profit_training_research_latest.json",
    "compound_portfolio_simulation": OUT_DIR / "high_precision_compound_portfolio_latest.json",
}

def sanitize(value):
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value

def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(f"缺少高精度研究结果：{TARGET}")
    payload = json.loads(TARGET.read_text(encoding="utf-8"))
    manifest = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_results = {}
    for name, path in RESULTS.items():
        if not path.exists():
            raw_results[name] = {"status": "结果文件不存在"}
            continue
        try:
            raw_results[name] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raw_results[name] = {"status": "结果解析失败", "error": f"{type(exc).__name__}: {exc}"}
    payload["all_conditions_research"] = sanitize(manifest)
    payload["all_conditions_full_results"] = sanitize(raw_results)
    TARGET.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ready",
        "included_result_families": list(raw_results),
        "full_result_count": len(raw_results),
        "manifest_status": manifest.get("status"),
        "artifact": str(TARGET.relative_to(ROOT)),
    }, ensure_ascii=False))

if __name__ == "__main__":
    main()
