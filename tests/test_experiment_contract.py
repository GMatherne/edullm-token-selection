"""Tests for fail-closed scratch and deterministic-order contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from token_selection.olmo_ext.token_io import write_token_array
from token_selection.scripts import resolve_tokens_s3
from token_selection.scripts.experiment_contract import (
    build_order_contract,
    manifest_train_paths,
    validate_order_contract,
    validate_scratch_config,
    validate_token_manifest,
)


def _config() -> dict:
    return {
        "seed": 42,
        "model": {"init_mode": "scratch", "init_seed": 42, "load_path": None},
        "train": {"data_loader_seed": 42, "global_batch_size": 128},
        "data": {"sequence_length": 16},
    }


def test_scratch_contract_rejects_checkpoint_or_seed_drift():
    cfg = _config()
    validate_scratch_config(cfg)

    cfg["model"]["load_path"] = "s3://checkpoint"
    with pytest.raises(ValueError, match="checkpoint"):
        validate_scratch_config(cfg)

    cfg = _config()
    cfg["train"]["data_loader_seed"] = 7
    with pytest.raises(ValueError, match="data_loader_seed"):
        validate_scratch_config(cfg)


def test_order_contract_binds_token_manifest_and_loader_settings(tmp_path):
    out = tmp_path / "out"
    tokens = out / "tokens"
    tokens.mkdir(parents=True)
    manifest = {"n_tokens": 128, "sequence_length": 16}
    (tokens / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    cfg = _config()

    contract = build_order_contract(cfg, output_dir=out, token_manifest=manifest)
    validate_order_contract(cfg, output_dir=out, contract=contract)

    changed = _config()
    changed["train"]["global_batch_size"] = 256
    with pytest.raises(ValueError, match="Order contract mismatch"):
        validate_order_contract(changed, output_dir=out, contract=contract)


def test_manifest_train_paths_refuses_stray_and_missing(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    write_token_array(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))
    (tokens / "manifest.json").write_text(
        json.dumps(
            {
                "n_tokens": 16,
                "shards": [{"source": "a", "path": "tokens_0000.npy", "n_tokens": 16}],
            }
        ),
        encoding="utf-8",
    )
    assert manifest_train_paths(tokens) == [str(tokens / "tokens_0000.npy")]

    # A stray shard not in the manifest must be refused (it would silently train).
    write_token_array(tokens / "tokens_0001.npy", np.arange(16, dtype=np.uint32))
    with pytest.raises(ValueError, match="Unlisted token shard"):
        manifest_train_paths(tokens)

    # A manifest that lists a missing file must also be refused.
    (tokens / "tokens_0001.npy").unlink()
    (tokens / "manifest.json").write_text(
        json.dumps(
            {
                "n_tokens": 32,
                "shards": [
                    {"source": "a", "path": "tokens_0000.npy", "n_tokens": 16},
                    {"source": "b", "path": "tokens_0002.npy", "n_tokens": 16},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absent from"):
        manifest_train_paths(tokens)


def _write_manifest(tokens, manifest) -> None:
    (tokens / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_manifest_rejects_np_save_shards(tmp_path):
    """A .npy header would be read as tokens and shift every sequence boundary."""
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    np.save(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))
    _write_manifest(
        tokens,
        {"n_tokens": 16, "shards": [{"path": "tokens_0000.npy", "n_tokens": 16}]},
    )
    with pytest.raises(ValueError, match="np.save"):
        validate_token_manifest(tokens)


def test_manifest_rejects_token_count_mismatch(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    write_token_array(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))
    _write_manifest(
        tokens,
        {"n_tokens": 99, "shards": [{"path": "tokens_0000.npy"}]},
    )
    with pytest.raises(ValueError, match="n_tokens=99"):
        validate_token_manifest(tokens)


def test_manifest_requires_n_tokens_and_shards(tmp_path):
    tokens = tmp_path / "tokens"
    tokens.mkdir(parents=True)
    write_token_array(tokens / "tokens_0000.npy", np.arange(16, dtype=np.uint32))

    _write_manifest(tokens, {"n_tokens": 16, "paths": ["tokens_0000.npy"]})
    with pytest.raises(ValueError, match="lists no shards"):
        validate_token_manifest(tokens)

    _write_manifest(tokens, {"shards": [{"path": "tokens_0000.npy"}]})
    with pytest.raises(ValueError, match="positive integer 'n_tokens'"):
        validate_token_manifest(tokens)


def test_tokens_uri_placeholder_is_refused():
    with pytest.raises(ValueError, match="placeholder"):
        resolve_tokens_s3({"data": {"tokens_s3": "s3://REPLACE_ME/tokens"}})
    assert resolve_tokens_s3({"data": {"tokens_s3": "s3://real-bucket/tokens/"}}) == (
        "s3://real-bucket/tokens"
    )
