#!/usr/bin/env python3
"""Train ``full``, ``rel_ema``, ``rho_excess``, or ``middle_ppl`` on an identical local stream.

The local path is a plumbing check: methods start from the same TinyLM init
recipe and consume the same frozen full training stream (no held-out carve-out).
Production training is delegated to ``train_olmo_template``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from token_selection.olmo_ext.frozen_ref import FrozenReference
from token_selection.olmo_ext.metrics import (
    MetricLogger,
    StepMetrics,
    empty_metrics_payload,
)
from token_selection.olmo_ext.smoke_data import iter_train_batches, load_fixed_sequence_corpus
from token_selection.olmo_ext.train_module import (
    TokenSelectConfig,
    TokenSelectLoop,
    has_olmo_core,
    make_ts_config,
)
from token_selection.scripts import derive_steps, load_config, resolve_output_dir
from token_selection.scripts.experiment_contract import validate_scratch_config

MethodName = Literal["full", "rel_ema", "rho_excess", "middle_ppl"]
_SELECTING = ("rel_ema", "rho_excess", "middle_ppl")


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.out(self.embed(input_ids))


def _stable_id(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_local_smoke(cfg: dict, out: Path, method: MethodName) -> Path:
    smoke = cfg["smoke"]
    seed = int(cfg["seed"])
    torch.manual_seed(seed)

    vocab = int(smoke["vocab_size"])
    seq_len = int(smoke["sequence_length"])
    steps = int(smoke["train_steps"])
    bs = int(smoke["batch_size"])
    lr = float(smoke["lr"])
    corpus = load_fixed_sequence_corpus(cfg, out)
    order_manifest = json.loads((out / "order" / "manifest.json").read_text(encoding="utf-8"))
    order_id = str(order_manifest["order_contract"]["contract_sha256"])

    t0_frac = float(cfg.get("t0_frac", 0.02))
    t0_steps = max(1, int(round(steps * t0_frac))) if method in _SELECTING else 0

    frozen_ref = None
    if method == "rho_excess":
        # Distinct in-memory twin so excess loss is not identically zero.
        torch.manual_seed(seed + 1)
        ref_model = TinyLM(vocab)
        frozen_ref = FrozenReference.from_module(ref_model)
        torch.manual_seed(seed)

    model = TinyLM(vocab)
    ts_cfg = TokenSelectConfig(
        method=method,
        k=float(cfg["k"]),
        t0_steps=t0_steps,
        total_steps=steps,
        alpha_start=float(cfg["alpha_start"]),
        alpha_end=float(cfg["alpha_end"]),
        seed=seed,
    )
    loop = TokenSelectLoop(model, ts_cfg, frozen_ref=frozen_ref)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ckpt_dir = out / "checkpoints" / method
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_filename = str((cfg.get("eval") or {}).get("metrics_filename", "metrics.json"))
    metrics_path = out / "metrics" / method / metrics_filename
    if metrics_path.exists():
        metrics_path.unlink()
    jsonl = metrics_path.with_suffix(".jsonl")
    if jsonl.exists():
        jsonl.unlink()

    n_params = int(sum(p.numel() for p in model.parameters()))
    payload = empty_metrics_payload(
        run_id=cfg["run_id"],
        method=method,
        seed=seed,
        k=float(cfg["k"]),
        t0_tokens=t0_steps * bs * seq_len,
        alpha_start=float(cfg["alpha_start"]),
        alpha_end=float(cfg["alpha_end"]),
        n_params=n_params,
        experiment_id=str(cfg["run_id"]),
        order_id=order_id,
        init_id=_stable_id(
            {
                "model": cfg["model"],
                "seed": seed,
                "vocab_size": vocab,
                "d_model": 64,
            }
        ),
    )
    logger = MetricLogger(metrics_path, payload)

    t0 = time.time()
    for step, input_ids in enumerate(iter_train_batches(corpus, batch_size=bs, steps=steps)):
        out_step = loop.train_step(input_ids)
        opt.zero_grad(set_to_none=True)
        out_step["loss"].backward()
        opt.step()
        loop.state.tokens_seen += int(input_ids.numel())
        loop.optim_step_done()

        cmp = out_step["compute"]
        logger.log_step(
            StepMetrics(
                step=step,
                tokens_seen=loop.state.tokens_seen,
                k=float(cfg["k"]),
                alpha=out_step["alpha"],
                warmup=out_step["warmup"],
                selected_frac=out_step["selected_frac"],
                mean_rel_kept=out_step["mean_score_kept"],
                mean_rel_dropped=out_step["mean_score_dropped"],
                train_loss=float(out_step["loss"].detach().item()),
                wall_time_s=time.time() - t0,
                selected_tokens=int(cmp["selected_tokens"]),
                forward_tokens_train=int(cmp["forward_tokens_train"]),
                forward_tokens_history=int(cmp["forward_tokens_history"]),
                forward_tokens_current=int(cmp["forward_tokens_current"]),
                fwd_passes_train=int(cmp["fwd_passes_train"]),
                fwd_passes_history=int(cmp["fwd_passes_history"]),
                fwd_passes_current=int(cmp["fwd_passes_current"]),
                method=method,
            )
        )
        logger.add_compute(
            train_tokens=int(cmp["forward_tokens_train"]),
            scoring_overhead_tokens=int(out_step["scoring_tokens"]),
            selected_tokens=int(cmp["selected_tokens"]),
            forward_tokens_train=int(cmp["forward_tokens_train"]),
            forward_tokens_history=int(cmp["forward_tokens_history"]),
            forward_tokens_current=int(cmp["forward_tokens_current"]),
            fwd_passes_train=int(cmp["fwd_passes_train"]),
            fwd_passes_history=int(cmp["fwd_passes_history"]),
            fwd_passes_current=int(cmp["fwd_passes_current"]),
        )

    logger.payload["compute"]["wall_time_s"] = time.time() - t0
    logger.flush()
    ts_state = loop.state.state_dict() if hasattr(loop.state, "state_dict") else None
    torch.save(
        {
            "model": model.state_dict(),
            "ts_cfg": ts_cfg.__dict__,
            "ts_state": ts_state,
            "optimizer": opt.state_dict(),
            "rng_state": torch.get_rng_state(),
        },
        ckpt_dir / "last.pt",
    )
    print(f"{method} smoke: wrote {metrics_path} and {ckpt_dir / 'last.pt'}")
    return metrics_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "token_selection/configs/run_smoke.yaml")
    ap.add_argument(
        "--method",
        choices=["full", "rel_ema", "rho_excess", "middle_ppl"],
        default=None,
    )
    ap.add_argument("--mode", choices=["auto", "local", "olmo"], default="auto")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve_output_dir(cfg, ROOT)
    methods = cfg.get("methods") or []
    if args.method:
        method: MethodName = args.method  # type: ignore[assignment]
    elif "middle_ppl" in methods and len(methods) == 1:
        method = "middle_ppl"
    elif "rho_excess" in methods:
        method = "rho_excess"
    else:
        method = "rel_ema"
    validate_scratch_config(cfg, method=method)

    allowed = cfg.get("methods") or ["full", "rel_ema", "rho_excess", "middle_ppl"]
    if method not in allowed:
        raise SystemExit(f"method {method!r} not in config methods {allowed}")

    mode = args.mode
    if mode == "auto":
        mode = "local"

    if mode == "local":
        if "smoke" not in cfg:
            raise SystemExit("Local mode requires a smoke: block in the config")
        run_local_smoke(cfg, out, method)
        return

    total_steps, t0_steps = derive_steps(cfg)
    ts_cfg = make_ts_config(cfg, method=method, total_steps=total_steps, t0_steps=t0_steps)
    print(
        json.dumps(
            {
                "status": "olmo_core_required",
                "method": method,
                "has_olmo_core": has_olmo_core(),
                "total_steps": total_steps,
                "t0_steps": t0_steps,
                "ts_cfg": ts_cfg.__dict__,
                "hint": "Use token_selection.scripts.train_olmo_template --method "
                f"{method} --launch (requires the pinned olmo_core checkout).",
                "run_id": cfg.get("run_id"),
            },
            indent=2,
        )
    )
    if not has_olmo_core():
        raise SystemExit(2)


if __name__ == "__main__":
    main()
