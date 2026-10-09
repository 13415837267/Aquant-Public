"""为云端研究阶段提供可校验的跨运行检查点。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
CHECKPOINT_DIR = ROOT / "data" / "research" / "workflow_stages"
EVENT_LOG = CHECKPOINT_DIR / "运行审计日志.jsonl"


def canonical_hash(value: object) -> str:
    content = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def history_data_fingerprint(root: Path = ROOT) -> str:
    """使用Git对象指纹识别历史行情；工作区有数据变动时改用实际文件摘要。"""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-s", "--", "data/history", "data/universe.json"],
        cwd=root,
        text=True,
    ).splitlines()
    changes = subprocess.check_output(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "data/history", "data/universe.json",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    worktree_inputs = {}
    for line in changes:
        if not line:
            continue
        relative = line[3:].strip().strip('"')
        path = root / relative
        if path.is_file():
            worktree_inputs[relative] = file_sha256(path)
        else:
            worktree_inputs[relative] = "已删除或不可读取"
    return canonical_hash({
        "tracked_git_objects": tracked,
        "worktree_changes": worktree_inputs,
    })


def checkpoint_reuse_reason(
    previous: object, expected_fingerprint: str, output_path: Path
) -> str | None:
    """返回不可复用原因；返回None仅表示检查点及其结果文件均可复用。"""
    if not isinstance(previous, dict) or previous.get("status") != "success":
        return "不存在已成功的检查点"
    if previous.get("fingerprint") != expected_fingerprint:
        return "代码、数据或参数指纹已变化"
    if not output_path.is_file():
        return "结果文件不存在"
    expected_digest = previous.get("output_sha256")
    if not isinstance(expected_digest, str) or file_sha256(output_path) != expected_digest:
        return "结果文件摘要不匹配"
    try:
        json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "结果文件不是有效JSON"
    return None


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def log_event(key: str, event: str, **fields) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "时间": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "阶段": key,
        "事件": event,
        **fields,
    }
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[{event}] {key} {json.dumps(fields, ensure_ascii=False)}", flush=True)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="带校验的云端研究阶段执行器")
    parser.add_argument("--key", required=True, help="稳定的阶段标识")
    parser.add_argument("--output", required=True, help="阶段结果JSON路径")
    parser.add_argument("--dependencies", nargs="+", required=True, help="影响结果的代码文件")
    parser.add_argument("--inputs", nargs="*", default=[], help="影响本阶段的既有结果或输入文件")
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True, help="实际执行命令及参数")
    args = parser.parse_args()

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("--command后必须提供实际命令")

    output = (ROOT / args.output).resolve()
    try:
        output_relative = output.relative_to(ROOT).as_posix()
        dependency_hashes = {}
        for item in args.dependencies:
            dependency = (ROOT / item).resolve()
            dependency_relative = dependency.relative_to(ROOT).as_posix()
            if not dependency.is_file():
                raise FileNotFoundError(f"依赖代码不存在：{dependency_relative}")
            dependency_hashes[dependency_relative] = file_sha256(dependency)
        input_hashes = {}
        for item in args.inputs:
            input_path = (ROOT / item).resolve()
            input_relative = input_path.relative_to(ROOT).as_posix()
            if not input_path.is_file():
                raise FileNotFoundError(f"阶段输入不存在：{input_relative}")
            input_hashes[input_relative] = file_sha256(input_path)
        requirements = ROOT / "requirements.txt"
        public_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        fingerprint = canonical_hash({
            "schema_version": 1,
            "key": args.key,
            "output": output_relative,
            "command": command,
            "dependencies": dependency_hashes,
            "inputs": input_hashes,
            "requirements_sha256": file_sha256(requirements),
            "history_data_fingerprint": history_data_fingerprint(ROOT),
            "python_major_minor": list(sys.version_info[:2]),
        })
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        log_event(args.key, "阶段预检失败", error=f"{type(exc).__name__}: {exc}")
        return 2

    checkpoint_path = CHECKPOINT_DIR / f"{args.key}.json"
    previous = load_json(checkpoint_path)
    reason = checkpoint_reuse_reason(previous, fingerprint, output)
    if reason is None:
        log_event(
            args.key,
            "断点恢复成功",
            output=output_relative,
            output_sha256=previous.get("output_sha256"),
            original_success_at=previous.get("success_at"),
        )
        return 0

    log_event(args.key, "阶段开始", output=output_relative, resume_reason=reason)
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            log_event(
                args.key, "阶段失败", return_code=completed.returncode,
                note="保留旧检查点；下次启动仍会校验代码、参数和结果摘要",
            )
            return completed.returncode or 1
        if not output.is_file():
            raise FileNotFoundError(f"阶段结束但未生成结果：{output_relative}")
        try:
            json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"阶段结果不是有效JSON：{output_relative}") from exc
        output_digest = file_sha256(output)
        checkpoint = {
            "schema_version": 1,
            "status": "success",
            "key": args.key,
            "fingerprint": fingerprint,
            "output": output_relative,
            "output_sha256": output_digest,
            "success_at": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
            "public_commit": public_commit,
            "command": command,
            "dependencies": dependency_hashes,
            "inputs": input_hashes,
        }
        write_json_atomic(checkpoint_path, checkpoint)
        log_event(
            args.key, "阶段成功并保存检查点",
            output=output_relative, output_sha256=output_digest,
        )
        return 0
    except Exception as exc:
        log_event(args.key, "阶段失败", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
