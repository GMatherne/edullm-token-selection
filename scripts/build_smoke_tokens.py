#!/usr/bin/env python3
"""Build synthetic fixed-sequence-length token arrays for local packing smoke.

Writes the same on-disk contract the real pre-tokenized corpus must satisfy (see
``experiment_contract.TOKEN_MANIFEST_SCHEMA``): raw headerless shards plus a manifest
listing them. Keeping the smoke on the production format means the shard layout is
exercised locally instead of first being discovered on the training box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as script or module
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.olmo_ext.token_io import (  # noqa: E402
    MASK_DTYPE,
    TOKEN_DTYPE,
    dtype_name,
    write_token_array,
)
from token_selection.scripts import load_config, resolve_output_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_smoke.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    smoke = cfg["smoke"]
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    n_docs = int(smoke["n_docs"])
    doc_len = int(smoke["doc_len"])
    seq_len = int(smoke["sequence_length"])
    vocab = int(smoke["vocab_size"])
    assert doc_len == seq_len, "smoke builder packs 1 doc = 1 sequence for simplicity"

    tokens_dir = out / "tokens"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    # Shards from an earlier build would read as stray files the manifest does not list.
    for stale in (*tokens_dir.glob("tokens_*.npy"), *tokens_dir.glob("labels_mask_*.npy")):
        stale.unlink()

    # ids in [1, vocab-1]; 0 reserved as pad-like
    flat = rng.integers(1, vocab, size=(n_docs * seq_len,), dtype=np.uint32)
    shard_name = "tokens_0000.npy"
    n_tokens = write_token_array(tokens_dir / shard_name, flat, dtype=TOKEN_DTYPE)

    # All-True label mask (online selection overrides during training)
    mask_name = "labels_mask_0000.npy"
    write_token_array(
        tokens_dir / mask_name, np.ones_like(flat, dtype=MASK_DTYPE), dtype=MASK_DTYPE
    )

    meta = {
        "n_tokens": n_tokens,
        "dtype": dtype_name(TOKEN_DTYPE),
        "sequence_length": seq_len,
        "vocab_size": vocab,
        "n_docs": n_docs,
        "shards": [
            {"path": shard_name, "label_mask_path": mask_name, "n_tokens": n_tokens},
        ],
        "synthetic": True,
        "seed": seed,
    }
    (tokens_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {tokens_dir / shard_name} ({n_tokens} tokens) and mask -> {tokens_dir}")


if __name__ == "__main__":
    main()
