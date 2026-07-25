"""End-to-end regression for the paired frozen-data TinyLM pipeline (train-only)."""

from __future__ import annotations

import json

import numpy as np

from token_selection.olmo_ext.token_io import TOKEN_DTYPE, dtype_name, write_token_array
from token_selection.scripts.experiment_contract import build_order_contract
from token_selection.scripts.train_method import run_local_smoke


def _cfg() -> dict:
    return {
        "run_id": "paired-test",
        "seed": 17,
        "k": 0.5,
        "t0_frac": 0.0,
        "alpha_start": 0.99,
        "alpha_end": 0.98,
        "model": {"init_mode": "scratch", "init_seed": 17, "load_path": None},
        "smoke": {
            "sequence_length": 8,
            "vocab_size": 32,
            "train_steps": 3,
            "batch_size": 2,
            "lr": 1e-3,
        },
        "train": {"global_batch_size": 16, "data_loader_seed": 17},
        "eval": {
            "metrics_filename": "metrics.json",
        },
    }


def test_full_and_rel_share_frozen_data_and_emit_train_metrics(tmp_path):
    cfg = _cfg()
    out = tmp_path / "smoke"
    tokens_dir = out / "tokens"
    order_dir = out / "order"
    tokens_dir.mkdir(parents=True)
    order_dir.mkdir()
    token_values = (np.arange(8 * 8, dtype=np.uint32) % 31) + 1
    n_tokens = write_token_array(tokens_dir / "tokens_0000.npy", token_values)
    token_manifest = {
        "n_tokens": n_tokens,
        "dtype": dtype_name(TOKEN_DTYPE),
        "shards": [{"path": "tokens_0000.npy", "n_tokens": n_tokens}],
    }
    (tokens_dir / "manifest.json").write_text(json.dumps(token_manifest), encoding="utf-8")
    np.save(order_dir / "sequence_permutation.npy", np.arange(8, dtype=np.int64))
    (order_dir / "manifest.json").write_text(
        json.dumps(
            {
                "order_contract": build_order_contract(
                    cfg, output_dir=out, token_manifest=token_manifest
                )
            }
        ),
        encoding="utf-8",
    )

    full_path = run_local_smoke(cfg, out, "full")
    rel_path = run_local_smoke(cfg, out, "rel_ema")
    full = json.loads(full_path.read_text(encoding="utf-8"))
    rel = json.loads(rel_path.read_text(encoding="utf-8"))

    assert full["experiment"] == rel["experiment"]
    assert "validation" not in full
    assert "validation" not in rel
    assert full["compute"]["forward_tokens_train"] == rel["compute"]["forward_tokens_train"]
    assert rel["compute"]["forward_tokens_history"] > 0
    assert len(full["train_loss_curve"]) == len(rel["train_loss_curve"]) > 0
    assert (out / "checkpoints" / "full" / "last.pt").exists()
    assert (out / "checkpoints" / "rel_ema" / "last.pt").exists()
