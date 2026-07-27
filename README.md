# Token selection

Train **OLMo-2 370M from scratch** on a shared ~10B RegMix corpus with online
token selection. This pass writes **checkpoints + train metrics** only;
held-out eval comes later.

## Methods (what the code actually supports)

| Method | After warmup | Scorer | Status |
|--------|--------------|--------|--------|
| `full` | loss on all valid tokens | none | implemented (smoke); no dedicated 10B config yet |
| `rel_ema` | keep top `k` by `REL = L_hist − L_curr` | bias-corrected EMA of train weights | **10B run finished**; artifacts under its S3 prefix / local `output_dir` |
| `rho_excess` | keep top `k` by `excess = L_curr − L_ref` | frozen reference checkpoint | code + config present; **not launch-ready** until `reference.load_path` and a GPU are set |
| `middle_ppl` | keep middle `k` by `L_curr` (CE ≈ log-PPL) | none (reuses train-forward CE) | code + config present; not launched |

Defaults for selecting methods: `k=0.6`, `t0_frac=0.02` (~48 of ~2384 steps).
REL also decays EMA `α: 0.99 → 0.98`.

Not implemented yet (P1 leftovers): random-k, raw-loss (top by CE),
attention, learnability, dynamic reference. Add them as new method plugs +
separate configs — do not retarget an existing arm’s YAML (especially not
REL’s finished `run_id` / prefix).

## Arm isolation

One shared spine (`TokenSelectTrainModule` + scorers). Arms stay separate by
**config identity**, not git branches:

| Arm | Config | `run_id` | `output_dir` | S3 `prefix` |
|-----|--------|----------|--------------|-------------|
| REL (done) | `configs/run_10b.yaml` | `rel-ema-10b-scratch-v1` | `token_selection/data/10b` | `token-selection/rel-ema-10b-scratch-v1` |
| RHO | `configs/run_rho_10b.yaml` | `rho-excess-10b-scratch-v1` | `token_selection/data/rho_10b` | `token-selection/rho-excess-10b-scratch-v1` |
| middle_ppl | `configs/run_middle_ppl_10b.yaml` | `middle-ppl-10b-scratch-v1` | `token_selection/data/middle_ppl_10b` | `token-selection/middle-ppl-10b-scratch-v1` |

Also present: `run_5b.yaml` (earlier REL segment / resume history),
`run_10b_smoke.yaml` (short OLMo memory smoke), `run_smoke.yaml` /
`run_rho_smoke.yaml` / `run_middle_ppl_smoke.yaml` (TinyLM plumbing).

**GPU pins are host-specific**, not part of scientific identity. Before
`--launch`, set `train.cuda_visible_devices` to an idle index on the machine
you are using (and the same value in `CUDA_VISIBLE_DEVICES`).

Do not reuse another arm’s `run_id`, `output_dir`, or S3 `prefix`. Sync/upload
always use **that config’s** prefix.

## Layout

```
token_selection/
  configs/           # one YAML per scientific run (or smoke)
  olmo_ext/          # EMA, FrozenReference, scorers, TrainModule, metrics
  scripts/           # sync, manifest, freeze_order, validate, train, smoke
  tests/
  data/              # local only (gitignored); real artifacts live on S3
```

## Corpus

Shared read-only input: `s3://edullm-dataset-regmix/regmix-10b/tokenized`
(`allenai/dolma2-tokenizer`, EOS `100257`). Seven domain shards are **raw
headerless** `uint32` arrays (despite `.npy`); each has a JSON sidecar. There is
**no** `manifest.json` in the bucket — `build_token_manifest` derives it locally
and verifies sizes/tokenizer/EOS.

Usable one-epoch budget after per-shard sequence truncation: **10,004,799,488**
tokens. Config `max_tokens: 10_000_000_000` fits; preflight rejects a budget that
would wrap into a second epoch. `data.tokenizer` must match the corpus or the
embedding table is wrong.

## How to run a 10B arm

```bash
cd Capstone
pip install -r token_selection/requirements.txt
# Pin OLMo-core (required revision is in each YAML under olmo_core.revision):
#   git clone https://github.com/edu-llm/OLMo-core /opt/OLMo-core
#   git -C /opt/OLMo-core checkout <revision>
#   pip install -e /opt/OLMo-core

CFG=token_selection/configs/run_middle_ppl_10b.yaml   # one config per arm; never retarget a finished run_id
METHOD=middle_ppl                                     # must be listed in that config's methods:

python -m token_selection.scripts.run_unit_smoke          # TinyLM plumbing
python -m token_selection.scripts.sync_artifacts \
  --config "$CFG" --direction download --what tokens
python -m token_selection.scripts.build_token_manifest --config "$CFG"
python -m token_selection.scripts.freeze_order --config "$CFG"
python -m token_selection.scripts.validate_experiment \
  --config "$CFG" --olmo-root /opt/OLMo-core

# Launch: set train.cuda_visible_devices in the YAML to an idle GPU on this host,
# then pin the same index in the environment before Python starts.
CUDA_VISIBLE_DEVICES=<gpu> python -m torch.distributed.run --standalone \
  --nproc_per_node=1 -m token_selection.scripts.train_olmo_template \
  --config "$CFG" --method "$METHOD" --olmo-root /opt/OLMo-core --launch

# Resume after crash / timeout (fingerprint-guarded):
CUDA_VISIBLE_DEVICES=<gpu> python -m torch.distributed.run --standalone \
  --nproc_per_node=1 -m token_selection.scripts.train_olmo_template \
  --config "$CFG" --method "$METHOD" --olmo-root /opt/OLMo-core --launch --resume

python -m token_selection.scripts.sync_artifacts \
  --config "$CFG" --direction upload --what metrics
python -m token_selection.scripts.sync_artifacts \
  --config "$CFG" --direction upload --what checkpoints
```

Fresh scratch refuses a non-empty checkpoint dir; dataset cache lives **outside**
that dir so a failed build can relaunch. `model.load_path` must stay null
(scratch only). Extending `max_tokens` upward on `--resume` is allowed; changing
seed, arch, order, `k`, reference path/bytes, etc. is not.

**Hardware knob:** `rank_microbatch_size` (default 65536 tokens) is not in the
run fingerprint. Selecting methods that need a second scoring forward (REL/RHO)
and full logits: if the first selecting step OOMs, halve it and `--resume`.
`middle_ppl` has no second forward, so it is lighter at the same microbatch.

## REL (`rel_ema`)

1. Steps `[0, T0)`: full-token loss; EMA updates every optim step.
2. After `T0`: per-sequence top-`k` by `REL = L_hist − L_curr`.
3. EMA is **bias-corrected** (`θ_hist = s / c` from zero). Without that, early
   history is dominated by the random init and REL collapses toward “keep easy
   tokens.” Online check: `mean_rel_kept` should sit above `mean_rel_dropped`
   once selection starts.

## RHO (`rho_excess`)

Same corpus contract / `k` / `t0_frac` as REL. After warmup, keep top-`k` by
`excess = L_curr − L_ref`. Reference is a weight shadow (`FrozenReference.swap_to`),
loaded once from a **local** `reference.load_path` (`.pt`/`.pth` or a dir with
`model.pt`); `s3://` URIs are refused. Resume fingerprints path **and** file
bytes. Preflight refuses a missing path.

Until `reference.load_path` and `train.cuda_visible_devices` are set,
`run_rho_10b.yaml` will not pass validate/launch. TinyLM smoke uses an
in-memory twin (`run_rho_smoke.yaml`) and does not need a checkpoint.

```bash
python -m token_selection.scripts.train_method \
  --config token_selection/configs/run_rho_smoke.yaml --method rho_excess
```

## Middle perplexity (`middle_ppl`)

Same corpus contract / `k` / `t0_frac` as REL/RHO. After warmup, per sequence
keep the middle `k` of valid tokens by current-model CE (`L_curr` ≈
log-perplexity): drop the easiest and hardest `(1−k)/2` each. No EMA and no
reference — the score is the train-forward CE already computed for the loss.
Permanent checkpoints every 250 steps (plus T0 milestone).

```bash
python -m token_selection.scripts.train_method \
  --config token_selection/configs/run_middle_ppl_smoke.yaml --method middle_ppl
```

## Metrics

Rank-zero schema-v2 train ledger under `metrics/<method>/`: compute counters plus
per-step CE, `selected_frac`, `alpha`, and `mean_rel_kept` /
`mean_rel_dropped` (name is historical; RHO writes excess means and middle_ppl
writes CE means into the same fields). On `--resume`, post-checkpoint rows are
truncated before logging continues. `compare_runs.py` exits until a shared eval
protocol exists.

## Stack

- **Tokens (RO):** `s3://edullm-dataset-regmix/regmix-10b/tokenized`
- **Outputs:** `s3://edullm-dataset-olmo/<prefix>/metrics`,
  `s3://edullm-checkpoints/<prefix>/` (`sbsandbox`, `us-east-1`)
- **Train:** [edu-llm/OLMo-core](https://github.com/edu-llm/OLMo-core) @ pinned
  revision + `TokenSelectTrainModule`
- **GPU:** single B200 per run
