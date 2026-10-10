"""为云端研究阶段提供可校验的跨运行检查点。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
RESEARCH_DATA_CUTOFF = "2026-09-30"
CHECKPOINT_DIR = ROOT / "data" / "research" / "workflow_stages"
EVENT_LOG = CHECKPOINT_DIR / "运行审计日志.jsonl"
_HISTORY_DAILY_PATH = re.compile(r"^data/history/(?:\d{4}/)?(\d{4}-\d{2}-\d{2})\.csv\.gz$")
_HISTORY_METADATA_FILES = {
    "data/history/_BACKFILL_STATE.json",
    "data/history/_BACKFILL_COMPLETE",
    "data/history/_ZZSHARE_VALIDATION.json",
}


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


def _research_daily_date(relative_path: str) -> date | None:
    match = _HISTORY_DAILY_PATH.fullmatch(relative_path)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def history_data_fingerprint(root: Path = ROOT, cutoff: str = RESEARCH_DATA_CUTOFF) -> str:
    """只对研究截止日内的历史日线建指纹，忽略截止日后的增量行情。"""
    cutoff_date = date.fromisoformat(cutoff)
    tracked_lines = subprocess.check_output(
        ["git", "ls-files", "-s", "--", "data/history"],
        cwd=root,
        text=True,
    ).splitlines()
    tracked = []
    for line in tracked_lines:
        metadata, separator, relative = line.partition("\t")
        if not separator:
            continue
        file_date = _research_daily_date(relative)
        if file_date is not None and file_date <= cutoff_date:
            tracked.append(line)

    changes = subprocess.check_output(
        [
            "git", "status", "--porcelain", "--untracked-files=all", "--",
            "data/history",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    worktree_inputs = {}
    for line in changes:
        if not line:
            continue
        relative = line[3:].strip().strip('"')
        file_date = _research_daily_date(relative)
        if file_date is None or file_date > cutoff_date:
            continue
        path = root / relative
        if path.is_file():
            worktree_inputs[relative] = file_sha256(path)
        else:
            worktree_inputs[relative] = "已删除或不可读取"
    return canonical_hash({
        "cutoff_date": cutoff,
        "tracked_git_objects": tracked,
        "worktree_changes": worktree_inputs,
    })


def _ensure_commit_available(commit_sha: str, root: Path = ROOT) -> bool:
    if not commit_sha or not re.fullmatch(r"[0-9a-f]{40}", commit_sha):
        return False
    check = subprocess.run(
        ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if check.returncode == 0:
        return True
    fetch = subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", commit_sha],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if fetch.returncode != 0:
        return False
    return subprocess.run(
        ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def history_data_fingerprint_at_commit(commit_sha: str, root: Path = ROOT) -> str:
    """重建旧版检查点使用的完整历史行情指纹，以验证旧检查点来源。"""
    if not _ensure_commit_available(commit_sha, root):
        raise ValueError(f"旧检查点提交不可读取：{commit_sha}")
    lines = subprocess.check_output(
        ["git", "ls-tree", "-r", "-s", "--full-tree", commit_sha, "--", "data/history", "data/universe.json"],
        cwd=root,
        text=True,
    ).splitlines()
    tracked = []
    for line in lines:
        metadata, relative = line.split("\t", 1)
        mode, object_type, blob_sha, size = metadata.split()
        tracked.append(f"{mode} {blob_sha} 0\t{relative}")
    return canonical_hash({
        "tracked_git_objects": tracked,
        "worktree_changes": {},
    })


def history_changes_only_after_cutoff(
    previous_commit: str,
    cutoff: str = RESEARCH_DATA_CUTOFF,
    root: Path = ROOT,
) -> bool:
    """只允许研究截止日后日线追加，以及采集维护元数据和证券列表更新。"""
    if not _ensure_commit_available(previous_commit, root):
        return False
    try:
        changed = subprocess.check_output(
            [
                "git", "diff", "--name-status", "--no-renames",
                previous_commit, "HEAD", "--",
                "data/history", "data/universe.json",
            ],
            cwd=root,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError:
        return False

    cutoff_date = date.fromisoformat(cutoff)
    for line in changed:
        fields = line.split("\t")
        if len(fields) != 2:
            return False
        _status, relative = fields
        if relative == "data/universe.json" or relative in _HISTORY_METADATA_FILES:
            continue
        file_date = _research_daily_date(relative)
        if file_date is None or file_date <= cutoff_date:
            return False
    return True


def files_unchanged_since_commit(
    previous_commit: str, paths: list[str], root: Path = ROOT
) -> bool:
    if not _ensure_commit_available(previous_commit, root):
        return False
    if not paths:
        return True
    result = subprocess.run(
        ["git", "diff", "--quiet", previous_commit, "HEAD", "--", *paths],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0



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


def compatible_legacy_checkpoint_reason(
    previous: object,
    key: str,
    command: list[str],
    dependency_hashes: dict,
    input_hashes: dict,
    output_path: Path,
    cutoff: str = RESEARCH_DATA_CUTOFF,
    root: Path = ROOT,
) -> str | None:
    """旧检查点只有在阶段命令、代码依赖、输入、结果及截止日前数据均核验通过时才可复用。"""
    if not isinstance(previous, dict) or previous.get("status") != "success":
        return "不存在成功的旧版检查点"
    if previous.get("key") != key or previous.get("command") != command:
        return "检查阶段标识或执行命令已变化"
    if previous.get("dependencies") != dependency_hashes:
        return "阶段代码依赖已变化"
    if previous.get("inputs") != input_hashes:
        return "阶段输入结果已变化"
    if not output_path.is_file():
        return "旧检查点结果文件不存在"
    expected_digest = previous.get("output_sha256")
    if not isinstance(expected_digest, str) or file_sha256(output_path) != expected_digest:
        return "旧检查点结果摘要不匹配"
    try:
        json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "旧检查点结果不是有效JSON"

    previous_commit = previous.get("public_commit")
    if not isinstance(previous_commit, str) or not _ensure_commit_available(previous_commit, root):
        return "旧检查点对应的公开提交不可读取"
    try:
        previous_requirements = subprocess.check_output(
            ["git", "show", f"{previous_commit}:requirements.txt"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return "无法验证旧检查点依赖版本"
    previous_requirements_sha = hashlib.sha256(previous_requirements).hexdigest()
    if previous_requirements_sha != file_sha256(root / "requirements.txt"):
        return "Python依赖清单已变化"

    try:
        old_history_fingerprint = history_data_fingerprint_at_commit(previous_commit, root)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return "无法重建旧检查点历史数据指纹"
    reconstructed = canonical_hash({
        "schema_version": 1,
        "key": key,
        "output": previous.get("output"),
        "command": command,
        "dependencies": dependency_hashes,
        "inputs": input_hashes,
        "requirements_sha256": previous_requirements_sha,
        "history_data_fingerprint": old_history_fingerprint,
        "python_major_minor": list(sys.version_info[:2]),
    })
    if reconstructed != previous.get("fingerprint"):
        return "旧检查点原始指纹无法核实"
    if not history_changes_only_after_cutoff(previous_commit, cutoff, root):
        return "研究截止日前的历史数据或关键数据文件发生变化"
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

    legacy_reason = compatible_legacy_checkpoint_reason(
        previous,
        key=args.key,
        command=command,
        dependency_hashes=dependency_hashes,
        input_hashes=input_hashes,
        output_path=output,
        cutoff=RESEARCH_DATA_CUTOFF,
        root=ROOT,
    )
    if legacy_reason is None:
        checkpoint = dict(previous)
        checkpoint["fingerprint"] = fingerprint
        checkpoint["public_commit"] = public_commit
        checkpoint["兼容恢复说明"] = (
            f"已核实研究截止日 {RESEARCH_DATA_CUTOFF} 内数据、代码依赖、参数及结果未变化"
        )
        checkpoint["兼容恢复时间"] = datetime.now(TIMEZONE).isoformat(timespec="seconds")
        write_json_atomic(checkpoint_path, checkpoint)
        log_event(
            args.key,
            "断点兼容恢复成功",
            output=output_relative,
            cutoff=RESEARCH_DATA_CUTOFF,
            output_sha256=previous.get("output_sha256"),
        )
        return 0

    log_event(args.key, "旧检查点不兼容", reason=legacy_reason)
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
