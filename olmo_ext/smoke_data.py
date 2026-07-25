"""Shared fixed-sequence smoke corpus helpers (train-only; no held-out split)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor

from .token_io import TOKEN_DTYPE, dtype_from_name, read_token_array


@dataclass(frozen=True)
class FixedSequenceCorpus:
    sequences: Tensor
    order_ids: Tuple[int, ...]


def _smoke_sequence_length(cfg: Mapping[str, object]) -> int:
    smoke = cfg.get("smoke") or {}
    if not isinstance(smoke, Mapping):
        raise ValueError("local smoke requires a smoke config")
    return int(smoke["sequence_length"])


def load_fixed_sequence_corpus(cfg: Mapping[str, object], output_dir: Path) -> FixedSequenceCorpus:
    """Load the local fixed sequence corpus and its frozen permutation (full stream)."""
    seq_len = _smoke_sequence_length(cfg)
    tokens_dir = Path(output_dir) / "tokens"
    manifest_path = tokens_dir / "manifest.json"
    order_path = Path(output_dir) / "order" / "sequence_permutation.npy"
    if not manifest_path.exists() or not order_path.exists():
        raise ValueError("Missing frozen smoke tokens/order; run build_smoke_tokens and freeze_order")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dtype = dtype_from_name(manifest["dtype"]) if manifest.get("dtype") else TOKEN_DTYPE
    shards = [str(shard["path"]) for shard in manifest.get("shards") or []]
    if not shards:
        raise ValueError(f"Token manifest lists no shards: {manifest_path}")
    pieces = [read_token_array(tokens_dir / name, dtype=dtype) for name in sorted(shards)]
    flat = np.concatenate(pieces)
    n_sequences = min(int(manifest["n_tokens"]) // seq_len, flat.size // seq_len)
    if n_sequences < 1:
        raise ValueError("Need at least one fixed sequence for smoke training")
    packed = flat[: n_sequences * seq_len].reshape(n_sequences, seq_len)
    # torch has no uint32 bridge, so widen before handing the buffer over.
    sequences = torch.from_numpy(packed.astype(np.int64, copy=False))

    permutation = np.load(order_path).astype(np.int64, copy=False)
    if permutation.ndim != 1 or permutation.size != n_sequences:
        raise ValueError("Frozen smoke permutation does not match the token sequence count")
    if sorted(permutation.tolist()) != list(range(n_sequences)):
        raise ValueError("Frozen smoke permutation is not a valid sequence permutation")

    return FixedSequenceCorpus(
        sequences=sequences,
        order_ids=tuple(int(i) for i in permutation),
    )


def iter_train_batches(
    corpus: FixedSequenceCorpus,
    *,
    batch_size: int,
    steps: int,
) -> Iterator[Tensor]:
    """Yield the identical frozen cyclic training stream for each arm."""
    if batch_size <= 0 or steps <= 0:
        raise ValueError("batch_size and steps must be positive")
    ids = np.asarray(corpus.order_ids, dtype=np.int64)
    positions = np.arange(steps * batch_size, dtype=np.int64) % ids.size
    for start in range(0, positions.size, batch_size):
        batch_ids = ids[positions[start : start + batch_size]]
        yield corpus.sequences[torch.from_numpy(batch_ids)]
