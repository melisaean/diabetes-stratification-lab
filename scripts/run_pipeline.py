"""CLI entry point for the stratification pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is on path for dev runs
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from personas.config import Settings
from personas.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the diabetes stratification pipeline")
    parser.add_argument("--device", default=None, help="torch device (mps, cpu, cuda)")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    settings = Settings()
    if args.epochs is not None:
        settings = settings.model_copy(update={"epochs": args.epochs})
    if args.seed is not None:
        settings = settings.model_copy(update={"seed": args.seed})

    run_pipeline(settings, device=args.device)


if __name__ == "__main__":
    main()