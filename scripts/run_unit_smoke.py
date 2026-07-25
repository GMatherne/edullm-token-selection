#!/usr/bin/env python3
"""Run unit tests plus a paired full-vs-REL TinyLM plumbing smoke (train-only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "token_selection/configs/run_smoke.yaml"


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    run([sys.executable, "-m", "pytest", "token_selection/tests", "-q"])
    run([sys.executable, "-m", "token_selection.scripts.build_smoke_tokens", "--config", str(CFG)])
    run([sys.executable, "-m", "token_selection.scripts.freeze_order", "--config", str(CFG)])
    for method in ("full", "rel_ema"):
        run(
            [
                sys.executable,
                "-m",
                "token_selection.scripts.train_method",
                "--config",
                str(CFG),
                "--method",
                method,
                "--mode",
                "local",
            ]
        )
    print("Paired full-vs-REL unit + local train-only smoke completed.")


if __name__ == "__main__":
    main()
