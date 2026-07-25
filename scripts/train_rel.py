#!/usr/bin/env python3
"""Backward-compatible alias: prefer ``train_method --method rel_ema``."""

from __future__ import annotations

import sys

from token_selection.scripts.train_method import main

if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv.extend(["--method", "rel_ema"])
    main()
