"""确保所有工作流只引用已核验使用 Node.js 24 的官方 Actions 版本。"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# 以下版本的 action.yml 已核验为 Node.js 24；工件下载已升级到 v8，避免
# v7 日志中 @actions/artifact 旧依赖触发 Buffer() 弃用警告。
APPROVED_NODE24_ACTIONS = {
    "actions/checkout": "v7",
    "actions/setup-python": "v7",
    "actions/setup-node": "v7",
    "actions/cache/restore": "v5",
    "actions/cache/save": "v5",
    "actions/download-artifact": "v8",
    "actions/upload-artifact": "v7",
    "actions/upload-pages-artifact": "v5",
    "actions/deploy-pages": "v5",
}
USES_PATTERN = re.compile(r"^\s*uses:\s*([^\s#]+)", re.MULTILINE)


def test_all_workflow_actions_use_audited_node24_versions():
    problems = []
    files = sorted(
        path for path in WORKFLOWS.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yml", ".yaml"}
    )
    assert files, "没有找到GitHub Actions工作流文件"

    for path in files:
        content = path.read_text(encoding="utf-8")
        for action_ref in USES_PATTERN.findall(content):
            if action_ref.startswith("./"):
                continue
            if "@" not in action_ref:
                problems.append(f"{path.relative_to(ROOT)}：未固定版本 {action_ref}")
                continue
            action_name, version = action_ref.rsplit("@", 1)
            expected = APPROVED_NODE24_ACTIONS.get(action_name)
            if expected is None:
                problems.append(f"{path.relative_to(ROOT)}：未审计的Action {action_ref}")
            elif version != expected:
                problems.append(
                    f"{path.relative_to(ROOT)}：{action_name} 使用 {version}，"
                    f"应使用已审计版本 {expected}"
                )

    assert not problems, "工作流Action运行时审计失败：\n" + "\n".join(problems)
