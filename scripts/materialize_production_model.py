"""一次性在GitHub云端物化第一版正式模型权重。"""
from __future__ import annotations

import argparse
from pathlib import Path

from scripts.production_model import (
    DEFAULT_MODEL_CODE_COMMIT,
    DEFAULT_TRAINING_END,
    DEFAULT_TRAINING_START,
    DEFAULT_MODEL_PATH,
    train_fixed_baseline,
    write_model,
)
from scripts.short_term_research import history_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_TRAINING_START)
    ap.add_argument("--end", default=DEFAULT_TRAINING_END)
    ap.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    args = ap.parse_args()

    files = history_files()
    model, days, samples = train_fixed_baseline(files, args.start, args.end)
    if samples < 5000:
        raise RuntimeError(f"模型训练样本不足: {samples}")
    write_model(Path(args.output), model, args.start, args.end, DEFAULT_MODEL_CODE_COMMIT)
    print({
        "status": "active_baseline",
        "training_start": args.start,
        "training_end": args.end,
        "processed_days": days,
        "samples": samples,
        "output": args.output,
    })


if __name__ == "__main__":
    main()
