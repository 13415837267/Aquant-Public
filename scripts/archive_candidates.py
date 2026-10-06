from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "candidates.json"
HISTORY_ROOT = ROOT / "data" / "candidates_history"


def archive_candidates(payload: dict, root: Path = HISTORY_ROOT) -> Path:
    if not isinstance(payload, dict):
        raise RuntimeError("candidate snapshot must be an object")
    as_of = str(payload.get("as_of") or "")
    if len(as_of) < 10:
        raise RuntimeError("candidate snapshot has no valid as_of")
    try:
        day = date.fromisoformat(as_of[:10])
    except ValueError as exc:
        raise RuntimeError(f"invalid candidate as_of: {as_of!r}") from exc

    if payload.get("status") != "ready":
        raise RuntimeError("only ready candidate snapshots can be archived")

    destination = root / f"{day.year:04d}" / f"{day.isoformat()}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if destination.exists():
        try:
            existing_payload = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"candidate history is unreadable: {destination}") from exc
        if existing_payload == payload:
            return destination
        version = str(payload.get("strategy_version") or "未知版本").replace("/", "_")
        commit = str(payload.get("strategy_commit") or "未知提交")[:10]
        versioned = root / f"{day.year:04d}" / f"{day.isoformat()}__{version}__{commit}.json"
        if versioned.exists():
            existing_versioned = json.loads(versioned.read_text(encoding="utf-8"))
            if existing_versioned == payload:
                return versioned
            digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:10]
            versioned = root / f"{day.year:04d}" / f"{day.isoformat()}__{version}__{commit}__{digest}.json"
            if versioned.exists():
                existing_hashed = json.loads(versioned.read_text(encoding="utf-8"))
                if existing_hashed != payload:
                    raise RuntimeError(f"candidate history content collision: {versioned}")
                return versioned
        versioned.write_text(serialized, encoding="utf-8")
        return versioned

    destination.write_text(serialized, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--history-root", type=Path, default=HISTORY_ROOT)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    destination = archive_candidates(payload, args.history_root)
    print(json.dumps({
        "status": "archived",
        "path": str(destination.relative_to(ROOT)),
        "as_of": payload["as_of"],
        "candidate_count": len(payload.get("candidates", [])),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
