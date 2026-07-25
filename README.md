# REL+EMA from-scratch pretraining

Train **OLMo-2 370M from scratch** with REL+EMA token selection on a ~10B
pre-tokenized corpus.

**This pass:** run training and keep checkpoints (plus train metrics). You want
the model now; evaluation / benchmarks happen later on those checkpoints.

**Target hardware:** a **single NVIDIA B200**. Config knobs in
`configs/run_10b.yaml` (`rank_microbatch_size`, `num_workers`, ephemeral
checkpoint interval) are set for that box.

| Method | Loss mask | History |
|--------|-----------|---------|
| `rel_ema` | top-k by `REL = L_hist − L_curr` after warmup | EMA of weights |

Schedule defaults: `k=0.6`, `T0=2%` of steps, EMA `α: 0.99 → 0.98`.

Tokens are already tokenized in S3 (`data.tokens_s3` in `run_10b.yaml`). Download
them, derive the token manifest, freeze the loader-order contract locally, then
train. Init, seed, batch shape, LR warmup, and `max_tokens` are pinned in that config.

## Layout

```
token_selection/
  configs/run_10b.yaml      # REL 10B experiment (B200)
  configs/run_smoke.yaml    # TinyLM plumbing only
  olmo_ext/                 # EMA, masks, TokenSelectLoop / TrainModule
  scripts/
    build_smoke_tokens.py     # reference implementation of the token input contract
    build_token_manifest.py   # derive tokens/manifest.json from the corpus sidecars
    freeze_order.py           # local order contract from token manifest
    sync_artifacts.py         # download tokens; upload metrics/checkpoints
    train_method.py           # TinyLM smoke arm
    train_olmo_template.py    # production entry point
    validate_experiment.py    # preflight
    run_unit_smoke.py
    resolve_checkpoint.py     # inert here: CPT-only, refuses scratch configs
    compare_runs.py           # inert here: exits until a shared eval protocol exists
  tests/
  data/                     # local only; real artifacts on S3
```

## The corpus

`data.tokens_s3` points at `s3://edullm-dataset-regmix/regmix-10b/tokenized`, which holds
the RegMix 10B mix already tokenized with **`allenai/dolma2-tokenizer`** (EOS `100257`):
one shard per domain, a JSON sidecar beside each one, and an index of all seven.

```
tokenized/
  paths.txt                              # one shard path per line, rooted at the corpus
  dclm/dclm.npy                          # raw headerless uint32 tokens
  dclm/dclm.json                         # {"tokenizer", "eos_token_id", "docs",
  arxiv/arxiv.npy                        #  "tokens_content", "tokens_with_eos",
  arxiv/arxiv.json                       #  "bytes", "dtype", ...}
  starcoder/…  pes2o/…  open-web-math/…  algebraic-stack/…  wiki/…
```

| Domain | Tokens | Domain | Tokens |
|---|---:|---|---:|
| `dclm` | 3,752,801,841 | `algebraic-stack` | 615,239,017 |
| `arxiv` | 2,500,162,905 | `wiki` | 156,360,805 |
| `starcoder` | 1,406,986,385 | | |
| `pes2o` | 938,157,310 | **total** | **10,004,807,041** |
| `open-web-math` | 635,098,778 | | |

Shards are **raw headerless arrays** despite the `.npy` suffix — every file's byte size is
exactly `tokens_with_eos × 4`, with no 128-byte header. That matters because OLMo-core
reads them with `np.frombuffer` over a byte range and derives the sequence count from the
byte size, so a header would be read as tokens and shift every sequence boundary.

There is **no `manifest.json` in the bucket**, and the rest of the pipeline needs one
because the order contract fingerprints it to pin the training set. So step 2 below
derives it from the sidecars. That step is also where the corpus gets verified: shards on
disk must match `paths.txt`, each file's size must match its sidecar's `bytes`, and all
shards must agree on dtype, tokenizer, and EOS id. Re-deriving is byte-for-byte
reproducible, so it does not disturb an already-frozen order contract.

Two consequences worth knowing:

- **`data.tokenizer` must be the corpus tokenizer.** It sets `padded_vocab_size()` and
  therefore the embedding table, so a mismatch is an out-of-range index rather than a bad
  score. Preflight compares the two and refuses to launch if they differ.
- **`max_tokens` must fit one epoch.** `NumpyFSLDataset` truncates *each* shard to a whole
  number of sequences, so the corpus serves 4,885,156 sequences = 10,004,799,488 tokens
  per epoch, slightly under its token total. The 10B budget is 99.95% of that; a larger
  one would silently wrap into a replaying second epoch, which preflight rejects.

`scripts/build_smoke_tokens.py` writes the same raw format at TinyLM scale.

## Train on one B200

```bash
cd Capstone
pip install -r token_selection/requirements.txt
# Pin the only supported framework revision on the machine:
git clone https://github.com/edu-llm/OLMo-core /opt/OLMo-core
git -C /opt/OLMo-core checkout 99e0009ed67679c90da970ec5ba439c9459e3757
pip install -e /opt/OLMo-core

# 0) Local plumbing check (not a scientific result)
python -m token_selection.scripts.run_unit_smoke

# 1) Download the pre-tokenized corpus (~37 GiB, 7 shards + sidecars)
python -m token_selection.scripts.sync_artifacts \
  --config token_selection/configs/run_10b.yaml --direction download --what tokens

# 2) Derive tokens/manifest.json from the sidecars, verifying the corpus
python -m token_selection.scripts.build_token_manifest \
  --config token_selection/configs/run_10b.yaml

# 3) Freeze the seeded loader-order contract (local)
python -m token_selection.scripts.freeze_order \
  --config token_selection/configs/run_10b.yaml

# 4) Fail-closed scratch/tokenizer/budget/order/revision preflight
python -m token_selection.scripts.validate_experiment \
  --config token_selection/configs/run_10b.yaml --olmo-root /opt/OLMo-core

# 5) Launch on physical GPU 7 only (shared 8×B200 node). The wrapper sets
# CUDA_VISIBLE_DEVICES=7 before Python starts; --launch also refuses if GPU 7 is busy.
./token_selection/scripts/launch_gpu7.sh token_selection/configs/run_10b.yaml rel_ema
# Short memory smoke (same microbatch, 64 steps):
./token_selection/scripts/launch_gpu7.sh token_selection/configs/run_10b_smoke.yaml rel_ema
# Restart after a crash / wall-clock timeout (guarded resume):
./token_selection/scripts/launch_gpu7.sh token_selection/configs/run_10b.yaml rel_ema --resume

# 6) Publish this run's outputs
python -m token_selection.scripts.sync_artifacts \
  --config token_selection/configs/run_10b.yaml --direction upload --what metrics
python -m token_selection.scripts.sync_artifacts \
  --config token_selection/configs/run_10b.yaml --direction upload --what checkpoints
```

`total_steps = max_tokens // global_batch_size` (~2384); `t0_steps =
round(0.02 * total_steps)` (~48). `model.load_path` is forbidden in this config.
A non-empty checkpoint directory is rejected on a fresh scratch launch; the dataset
cache is kept outside that directory so a failed build does not block relaunching.

**Checkpoints.** Permanent saves every `checkpoint_every_steps`, plus
`checkpoint_milestone_steps` (around the REL warmup boundary), plus the step-0
scratch init. Keep them all (`checkpoint_keep_last: null`) for later evaluation.
Ephemeral checkpoints back guarded `--resume`.

**B200 notes.** `global_batch_size` is the scientific batch (do not change for
hardware). `rank_microbatch_size` only controls how that per-step batch is split on the
single GPU, and it is sized by the **logits tensor, not the parameters**: REL scoring
needs full unsharded logits, so this train module cannot use a fused linear+CE and
instead materializes `[seqs, 2048, ~100k vocab]`, about 0.41 GB per sequence in bf16.

At the default 65536 tokens (32 sequences) that is ~13 GB per copy and roughly 26 GB of
logits live, with activations of the same order — real margin against 180 GB of HBM.
Two things keep logits from doubling: the scoring logits are released before the train
forward allocates its own, and the per-token cross-entropy is computed **once** and reused
for both the REL score and the objective. The previous 128-sequence setting was expected
to OOM once activations were counted alongside logits.

**This is the one unmeasured knob.** If it OOMs it will do so at step 48, the first
REL-active step, where the history forward appears — so you find out in minutes, not
hours. Halve `rank_microbatch_size` and relaunch with `--resume`; the knob is
deliberately excluded from the run fingerprint so the identity guard allows it, and the
gradient is unchanged by the split because top-k selection is per sequence and the loss
divisor is the batch-level selected-token count. Allocation sizes are constant per step,
so if step 48 survives, a later OOM is unlikely. Raise the microbatch only after a smoke
reports `torch.cuda.max_memory_allocated()`.

REL still costs a second forward per micro-batch, and masking loss does not shrink the
backward pass, so a REL step is ~1.33x a full-token step: ~29.4 EFLOPs for the run
against ~22.2 for a full-token baseline, or ~3.6 h at B200 BF16 peak before utilization.

## Stack

- **Data:** `s3://edullm-dataset-regmix/regmix-10b/tokenized` (read-only), 7 domain shards
- **Outputs:** metrics to `s3://edullm-dataset-olmo/<prefix>/metrics`, checkpoints to
  `s3://edullm-checkpoints/<prefix>/` (both `us-east-1`, same region as the corpus)
- **Training:** [edu-llm/OLMo-core](https://github.com/edu-llm/OLMo-core) + `TokenSelectTrainModule` (`method=rel_ema`)
- **AWS:** Intern OIDC → `sbsandbox` only
- **GPU:** single B200

## REL + EMA schedule

1. Steps `[0, T0)`: loss on all valid tokens; update the EMA after every optimizer step.
2. After `T0`: keep top k% by `REL = L_hist − L_curr`, per sequence.
3. After each optim step, accumulate `s ← α s + (1−α) θ` and `c ← α c + (1−α)`, both from
   zero, with α decaying `alpha_start → alpha_end`. The history model is `θ_hist = s / c`.

**Why the EMA is bias-corrected.** Dividing by `c` makes `θ_hist` an exact convex
combination of the *observed* weights θ₁..θ_t, so the random initialization θ₀ never
appears in it. The textbook form — seed the shadow with θ₀ and blend in place — leaves θ₀
as the single heaviest ingredient for the first ~1/(1−α) steps, which is fatal here: a
weight-space blend of a random init and a trained model is not an older model but an
off-manifold point whose per-token loss is nearly flat. That reduces `REL = L_hist − L_curr`
to `constant − L_curr`, so top-k would keep the **easiest** tokens, inverting the method.
At `α = 0.999` the uncorrected shadow would still be 95% init at T0 and 51% init at step 500.

Warmup does not substitute for this. It controls when REL is *read*, not what the history
contains, so a 48-step warmup against a 1000-step memory removes almost nothing. Warmup
earns its keep for a different reason: it keeps gradients on all tokens while the current
model is still untrained.

With the init excluded, α is just a tuning knob for how far back the reference looks —
~100 steps at `0.99`, tightening to ~50. Sanity check on the real run: `mean_rel_kept`
should sit above `mean_rel_dropped` from the first selecting step onward.

## Local smoke

```bash
python -m token_selection.scripts.run_unit_smoke
```

TinyLM plumbing only: frozen stream, train metrics, checkpoint write. Not evidence
about the 10B run.

## Metrics

`RawComputeCallback` writes rank-zero schema-v2 **train** metrics to
`metrics/rel_ema/metrics.json` (plus a `.jsonl` of the same rows): compute counters,
and one row per optimizer step with batch CE, `selected_frac`, `alpha`, and the mean REL
score of kept vs dropped tokens. Those last two are the only online evidence that
selection is working — if the kept mean stops sitting above the dropped mean, REL is not
separating tokens. OLMo-core's own metric stream carries the usual loss/throughput
logging alongside this ledger.

On `--resume`, rows written after the last checkpoint are dropped before logging
continues, since those steps get replayed.

Held-out evaluation is deliberately not part of this pass. Use the checkpoints under
`checkpoints/rel_ema/` when you evaluate later; `compare_runs.py` intentionally exits
until a shared test-loss protocol exists.
