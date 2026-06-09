#!/usr/bin/env bash
# Run from shallowMerge (repo root): bash lerobot/bt/dm0/run_dm0_openloop_eval.sh
set -euo pipefail

MERGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$MERGE_ROOT"

# HF `datasets` must be able to create its cache when reading meta/episodes parquet.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${MERGE_ROOT}/.hf_datasets_parquet_cache}"
# Broken env: e.g. HF_DATASETS_CACHE or a parent points at an existing non-directory (often a stray file).
if [[ -e "$HF_DATASETS_CACHE" && ! -d "$HF_DATASETS_CACHE" ]]; then
  export HF_DATASETS_CACHE="${MERGE_ROOT}/.hf_datasets_parquet_cache"
fi
mkdir -p "$HF_DATASETS_CACHE"
# Do not mkdir HF_HOME here: if HF_HOME is a file or a bad symlink, mkdir fails with "File exists".
# Python + --dataset-root set HF_DATASETS_CACHE under the dataset when unset; that is enough for parquet.
if [[ -n "${HF_HOME:-}" && -e "$HF_HOME" && ! -d "$HF_HOME" ]]; then
  unset HF_HOME
fi

# python lerobot/bt/dm0/dm0_openloop_eval.py \
#   --dataset-repo local/r1_pro_chassis_v3_test \
#   --dataset-root /mnt/r/share/zwy/datasets/r1_pro_test_data_v3 \
#   --checkpoint "${MERGE_ROOT}/lerobot/outputs/bt/dm0/full_train-20260509_094559_2383629/checkpoints/022500/pretrained_model"

python lerobot/bt/dm0/dm0_openloop_policy.py \
  --checkpoint "${MERGE_ROOT}/lerobot/outputs/bt/dm0_aloha/full_train-20260520_082155_3489023_428317392/checkpoints/010000/pretrained_model" \
  --dataset-repo local/r1_pro_chassis_v3_test \
  --dataset-root /mnt/r/share/zwy/datasets/r1_pro_test_data_v3 \
  --episode-index 0 \
  --steps 1000 \
  --action-horizon 50 \
  --ylim-min -1.8 \
  --ylim-max 1.8 \
  --save-plot dm0_openloop_aloha_10000.png