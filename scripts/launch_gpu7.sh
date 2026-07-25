#!/usr/bin/env bash
# Launch the REL experiment (or smoke) on physical GPU 7 only.
#
# CUDA_VISIBLE_DEVICES is set *before* Python starts so torch/OLMo-core cannot
# open any other GPU, even if something imports CUDA at module load time.
# train_olmo_template.py --launch re-checks the pin and refuses a busy device.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REL_ROOT="${REL_ROOT:-/mnt/nvme/rel-test}"
VENV="${REL_ROOT}/venv"
OLMO_ROOT="${REL_ROOT}/OLMo-core"
CONFIG="${1:-token_selection/configs/run_10b_smoke.yaml}"
METHOD="${2:-rel_ema}"

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Missing venv at ${VENV}; provision /mnt/nvme/rel-test first." >&2
  exit 1
fi

# Refuse to start if physical GPU 7 is already holding memory (someone else's job).
USED_MIB="$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "${USED_MIB}" ]]; then
  echo "nvidia-smi could not read GPU 7; refusing to launch." >&2
  exit 1
fi
if (( USED_MIB > 256 )); then
  echo "GPU 7 is not idle (${USED_MIB} MiB used); refusing to launch." >&2
  nvidia-smi -i 7
  exit 1
fi

echo "Launching on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (physical GPU 7 idle: ${USED_MIB} MiB)"
cd "${ROOT}"
exec "${VENV}/bin/torchrun" --standalone --nproc_per_node=1 \
  -m token_selection.scripts.train_olmo_template \
  --config "${CONFIG}" \
  --method "${METHOD}" \
  --olmo-root "${OLMO_ROOT}" \
  --launch \
  "${@:3}"
