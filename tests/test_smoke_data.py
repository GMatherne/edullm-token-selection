"""Tests for the train-only smoke corpus helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from token_selection.olmo_ext.smoke_data import (
    iter_train_batches,
    load_fixed_sequence_corpus,
)
from token_selection.olmo_ext.token_io import TOKEN_DTYPE, dtype_name, write_token_array


def _write_shard(tokens_dir, values) -> None:
    """Write the raw headerless shard + manifest the production loader expects."""
    n_tokens = write_token_array(tokens_dir / "tokens_0000.npy", values)
    (tokens_dir / "manifest.json").write_text(
        json.dumps(
            {
                "n_tokens": n_tokens,
                "dtype": dtype_name(TOKEN_DTYPE),
                "shards": [{"path": "tokens_0000.npy", "n_tokens": n_tokens}],
            }
        ),
        encoding="utf-8",
    )


def test_load_fixed_sequence_corpus_uses_full_stream(tmp_path):
    cfg = {
        "smoke": {"sequence_length": 16},
    }
    tokens_dir = tmp_path / "tokens"
    order_dir = tmp_path / "order"
    tokens_dir.mkdir()
    order_dir.mkdir()
    values = np.arange(4 * 16, dtype=np.uint32)
    _write_shard(tokens_dir, values)
    np.save(order_dir / "sequence_permutation.npy", np.array([2, 0, 3, 1], dtype=np.int64))

    corpus = load_fixed_sequence_corpus(cfg, tmp_path)
    assert corpus.sequences.shape == (4, 16)
    assert corpus.order_ids == (2, 0, 3, 1)

    batches = list(iter_train_batches(corpus, batch_size=2, steps=2))
    assert len(batches) == 2
    assert batches[0].shape == (2, 16)
    # First batch follows the permutation head: ids 2, 0
    assert int(batches[0][0, 0]) == int(values.reshape(4, 16)[2, 0])


def test_corpus_rejects_invalid_permutation(tmp_path):
    cfg = {"smoke": {"sequence_length": 8}}
    tokens_dir = tmp_path / "tokens"
    order_dir = tmp_path / "order"
    tokens_dir.mkdir()
    order_dir.mkdir()
    values = np.arange(2 * 8, dtype=np.uint32)
    _write_shard(tokens_dir, values)
    np.save(order_dir / "sequence_permutation.npy", np.array([0, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="permutation"):
        load_fixed_sequence_corpus(cfg, tmp_path)
