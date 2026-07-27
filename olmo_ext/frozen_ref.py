"""Frozen reference-model weight shadow for RHO-1 excess-loss scoring.

Stores a rank-local copy of reference parameters and temporarily swaps them into
the training module for a no-grad scoring forward (same FSDP-safe pattern as
:class:`~token_selection.olmo_ext.ema.EMAHistory.swap_to`). Unlike the EMA
history, these weights are loaded once and never updated.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from .ema import _copy_into_param_, _local_tensor


class FrozenReference:
    """Immutable parameter shadow used as L_ref in ``excess = L_curr − L_ref``."""

    STATE_VERSION = 1

    def __init__(self, named_params: Iterable[Tuple[str, Tensor]]):
        self._shadow: Dict[str, Tensor] = {}
        for name, p in named_params:
            self._shadow[name] = _local_tensor(p).detach().clone()
        if not self._shadow:
            raise ValueError("FrozenReference requires at least one parameter")

    @classmethod
    def from_module(cls, module: nn.Module) -> "FrozenReference":
        """Snapshot ``module`` parameters (typically a loaded reference checkpoint)."""
        return cls(((n, p) for n, p in module.named_parameters()))

    @classmethod
    def from_state_dict(
        cls,
        module: nn.Module,
        state_dict: Mapping[str, Tensor],
    ) -> "FrozenReference":
        """Allocate shadow storage from ``module`` shapes, then copy ``state_dict`` in.

        ``module`` supplies the live (possibly FSDP-sharded) parameter layout; values
        come from ``state_dict``. Keys must match ``module.named_parameters()``.
        """
        ref = cls(((n, p) for n, p in module.named_parameters()))
        ref.load_weights(state_dict)
        return ref

    @property
    def shadow(self) -> Mapping[str, Tensor]:
        return self._shadow

    def load_weights(self, state_dict: Mapping[str, Tensor]) -> None:
        """Overwrite shadow tensors from a flat parameter state dict."""
        missing = set(self._shadow) - set(state_dict)
        # Allow extra keys (buffers, non-trainable) but refuse missing params.
        if missing:
            raise KeyError(
                f"reference state_dict missing parameters: {sorted(missing)[:8]}"
                + ("…" if len(missing) > 8 else "")
            )
        for name in self._shadow:
            value = state_dict[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"reference weight {name!r} must be a Tensor")
            local = _local_tensor(value)
            if local.shape != self._shadow[name].shape:
                raise ValueError(
                    f"reference weight {name!r} shape {tuple(local.shape)} does not "
                    f"match training shard shape {tuple(self._shadow[name].shape)}"
                )
            self._shadow[name].copy_(local.detach())

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "shadow": {k: v.detach().clone() for k, v in self._shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        version = state.get("version")
        if version != self.STATE_VERSION:
            raise ValueError(
                f"unsupported FrozenReference state version {version!r}; "
                f"expected {self.STATE_VERSION}"
            )
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("FrozenReference state is missing its shadow weights")
        self.load_weights(shadow)  # type: ignore[arg-type]

    @torch.no_grad()
    def copy_to(self, module: nn.Module) -> None:
        for name, p in module.named_parameters():
            if name in self._shadow:
                _copy_into_param_(p, self._shadow[name])

    @contextlib.contextmanager
    def swap_to(self, module: nn.Module):
        """Temporarily load reference weights into ``module`` for one no-grad forward."""
        saved: Dict[str, Tensor] = {}
        try:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in self._shadow:
                        saved[name] = _local_tensor(p).detach().clone()
                        _copy_into_param_(p, self._shadow[name])
            yield module
        finally:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    if name in saved:
                        _copy_into_param_(p, saved[name])

    def named_params(self) -> Iterator[Tuple[str, Tensor]]:
        yield from self._shadow.items()
