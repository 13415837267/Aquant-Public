from __future__ import annotations

from scripts.backfill_history import api_client, load_universe


def main() -> None:
    universe = load_universe(api_client())
    print(f"MAIN_BOARD_UNIVERSE_READY symbols={len(universe)}")


if __name__ == "__main__":
    main()
