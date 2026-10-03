from __future__ import annotations

import argparse
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
        existing = destination.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(f"candidate history is immutable and differs: {destination}")
        return destination

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
