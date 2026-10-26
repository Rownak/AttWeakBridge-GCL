#!/usr/bin/env bash
# Train the Sprint 9 ablation matrix (dev_docs/sprints/sprint_9.md, Phase 9.2) for one dataset. Settings are fixed in
# configs/experiments/unsup_u{1,2}.yaml (100 epochs, k 5, Adam 1e-3); nothing here is tuned and nothing reads gold.
#
# Usage: scripts/run_unsup_matrix.sh <dataset> <priority>...   e.g.  scripts/run_unsup_matrix.sh ncit_doid_unsup 1 2 3
#   priority 1: u1, u2            priority 2: u1-i, u2-i            priority 3: u2-rand, u2-hard, u2-mut
# Per seed: one training_data run per sampler variant, then the gcl_train rows that use it (model id = row + seed).
set -euo pipefail

DATASET=$1
shift
PRIORITIES=" $* "
SEEDS=(42 43 44)

td() {  # td <seed> <experiment> [--set ...]: build pseudo-pairs
  local seed=$1 exp=$2
  shift 2
  uv run cga run training_data --dataset "$DATASET" --experiment "$exp" --set "seed=$seed" "$@"
}

gcl() {  # gcl <seed> <experiment> <row> <sampler_id> [--set ...]: train one row
  local seed=$1 exp=$2 row=$3 sampler=$4
  shift 4
  uv run cga run gcl_train --dataset "$DATASET" --experiment "$exp" --set "seed=$seed" \
    --set "gcl_train.model.model_id=$row-s$seed" --set "gcl_train.inputs.training_data_sampler_id=$sampler" "$@"
}

has() { [[ "$PRIORITIES" == *" $1 "* ]]; }

for seed in "${SEEDS[@]}"; do
  if has 1 || has 2; then
    td "$seed" unsup_u2
    has 1 && gcl "$seed" unsup_u1 u1 rank_topk-k5 && gcl "$seed" unsup_u2 u2 rank_topk-k5
    if has 2; then
      gcl "$seed" unsup_u1 u1-i rank_topk-k5 --set gcl_train.loss.name=infonce
      gcl "$seed" unsup_u2 u2-i rank_topk-k5 --set gcl_train.loss.name=infonce
    fi
  fi
  if has 3; then
    td "$seed" unsup_u2 --set training_data.sampler.negatives=random
    gcl "$seed" unsup_u2 u2-rand rank_topk-k5-negrandom
    td "$seed" unsup_u2 --set training_data.sampler.negatives=hard
    gcl "$seed" unsup_u2 u2-hard rank_topk-k5-neghard-off10
    td "$seed" unsup_u2 --set training_data.sampler.positives=mutual
    gcl "$seed" unsup_u2 u2-mut rank_topk-k5-posmutual
  fi
done
