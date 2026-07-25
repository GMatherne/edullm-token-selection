"""Loss masks for full-token CPT and REL (retrospective excess loss) scoring.

Polarity matches OLMo-core ``label_mask``: ``True`` = token contributes to loss.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import Tensor

MethodName = Literal["full", "rel_ema"]


def top_k_mask(scores: Tensor, k: float, *, valid: Optional[Tensor] = None) -> Tensor:
    """Keep the top ``k`` fraction of positions by ``scores`` (higher = more important).

    Selection is *per sequence*: for 2-D ``[B, T]`` scores, each row keeps its own
    top-``k`` fraction of *valid* positions independently, matching ssToken's per-sample
    top-rho selection (Eq. 10). A 1-D tensor is treated as a single sequence, so the
    old global behaviour is preserved for that case. Leading dims of an N-D tensor are
    treated as independent rows and selection runs along the last (sequence) dim.
    """
    if scores.numel() == 0:
        return scores.bool()

    k = float(min(max(k, 1e-8), 1.0))
    orig_shape = scores.shape
    scores2d = scores.reshape(1, -1) if scores.dim() == 1 else scores.reshape(-1, orig_shape[-1])
    rows, cols = scores2d.shape

    if valid is None:
        valid2d = torch.ones_like(scores2d, dtype=torch.bool)
    else:
        valid2d = valid.reshape(scores2d.shape).to(dtype=torch.bool)

    # Per-row keep count over valid positions; >=1 where any valid, 0 if the row is empty.
    n_valid = valid2d.sum(dim=1)
    n_keep = torch.clamp((n_valid.to(torch.float32) * k).round().long(), min=1)
    n_keep = torch.minimum(n_keep, n_valid)

    masked = scores2d.masked_fill(~valid2d, float("-inf"))
    order = masked.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(cols, device=scores.device).expand(rows, cols)
    ranks.scatter_(1, order, ar)
    keep = (ranks < n_keep.unsqueeze(1)) & valid2d
    return keep.reshape(orig_shape)


def normalize_rel_per_row(
    rel: Tensor,
    *,
    valid: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """Per-sequence min-max normalize REL into ``[0, 1]`` (ssToken Eq. 8).

    Monotonic within each row, so for *pure* REL top-k this does not change the
    selection; it exists to make REL commensurable when blending with other signals
    (e.g. an attention score) in future conditions. Invalid positions map to 0.
    """
    orig_shape = rel.shape
    r = rel.reshape(1, -1) if rel.dim() == 1 else rel.reshape(-1, orig_shape[-1])
    if valid is None:
        v = torch.ones_like(r, dtype=torch.bool)
    else:
        v = valid.reshape(r.shape).to(dtype=torch.bool)

    row_max = r.masked_fill(~v, float("-inf")).max(dim=1, keepdim=True).values
    row_min = r.masked_fill(~v, float("inf")).min(dim=1, keepdim=True).values
    denom = (row_max - row_min).clamp_min(eps)
    out = (r - row_min) / denom
    out = out.masked_fill(~v, 0.0)
    return out.reshape(orig_shape)


def warmup_mask(shape_ref: Tensor, *, valid: Optional[Tensor] = None) -> Tensor:
    """All valid positions contribute to loss (full baseline + REL warmup)."""
    if valid is not None:
        return valid.to(dtype=torch.bool)
    return torch.ones_like(shape_ref, dtype=torch.bool)


def full_mask(shape_ref: Tensor, *, valid: Optional[Tensor] = None) -> Tensor:
    """Full-token baseline: every valid position contributes to loss."""
    return warmup_mask(shape_ref, valid=valid)


def rel_ema_mask(
    current_loss: Tensor,
    history_loss: Tensor,
    k: float,
    *,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """Keep tokens with highest ``REL = L_hist − L_curr``."""
    return top_k_mask(history_loss - current_loss, k, valid=valid)


def build_mask(
    *,
    method: MethodName = "rel_ema",
    k: float,
    current_loss: Optional[Tensor] = None,
    history_loss: Optional[Tensor] = None,
    shape_ref: Optional[Tensor] = None,
    valid: Optional[Tensor] = None,
    warmup: bool = False,
) -> Tensor:
    """Build loss mask for ``full`` or ``rel_ema``."""
    if method == "full" or warmup:
        ref = shape_ref if shape_ref is not None else current_loss
        if ref is None:
            raise ValueError("full/warmup mask requires shape_ref or current_loss")
        return full_mask(ref, valid=valid)

    if method != "rel_ema":
        raise ValueError(f"Unknown method {method!r}; expected 'full' or 'rel_ema'")

    if current_loss is None or history_loss is None:
        raise ValueError("REL mask requires current_loss and history_loss")
    return rel_ema_mask(current_loss, history_loss, k, valid=valid)
