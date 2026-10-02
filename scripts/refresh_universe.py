from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_history import api_client, load_universe


def main() -> None:
    universe = load_universe(api_client())
    print(f"MAIN_BOARD_UNIVERSE_READY symbols={len(universe)}")


if __name__ == "__main__":
    main()
