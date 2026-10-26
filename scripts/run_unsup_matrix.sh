#!/usr/bin/env bash
# Train the Sprint 9 ablation matrix (dev_docs/sprints/sprint_9.md, Phases 9.2 and 9.3) for one dataset. Settings are fixed in
# configs/experiments/unsup_u{1,2}.yaml (100 epochs, k 5, Adam 1e-3); nothing here is tuned and nothing reads gold.
#
# Usage: [TEXT_EMBEDDER=<embedder_id> PREFIX=<tag->] scripts/run_unsup_matrix.sh <dataset> <priority>...
#   scripts/run_unsup_matrix.sh ncit_doid_unsup 1 2 3                                              # Phase 9.2 (frozen BERT)
#   TEXT_EMBEDDER=hf_mean_pool-sapbert PREFIX=sap- scripts/run_unsup_matrix.sh ncit_doid_unsup 1 2 3   # Phase 9.3 (SapBERT)
#   priority 1: u1, u2            priority 2: u1-i, u2-i            priority 3: u2-rand, u2-hard, u2-mut
# Per seed: one training_data run per sampler variant, then the gcl_train rows that use it. Model id = PREFIX + row + seed
# (`sap-u2-hard-s43`); with a PREFIX the sampler id is also prefixed, so a second text encoder never moves the first one's LATEST.
# Without TEXT_EMBEDDER / PREFIX the commands are exactly those of Phase 9.2.
set -euo pipefail

DATASET=$1
shift
PRIORITIES=" $* "
SEEDS=(42 43 44)
PREFIX=${PREFIX:-}
TEXT_EMBEDDER=${TEXT_EMBEDDER:-}

td_args=()
[[ -n "$TEXT_EMBEDDER" ]] && td_args+=(--set "training_data.inputs.text_embeddings.embedder_id=$TEXT_EMBEDDER")

sampler_id() {  # sampler_id <default id>: the id training_data writes to and gcl_train reads from
  echo "$PREFIX$1"
}

td() {  # td <seed> <default sampler id> [--set ...]: build pseudo-pairs
  local seed=$1 sid=$2
  shift 2
  local extra=()
  [[ -n "$PREFIX" ]] && extra+=(--set "training_data.sampler.sampler_id=$(sampler_id "$sid")")
  uv run cga run training_data --dataset "$DATASET" --experiment unsup_u2 --set "seed=$seed" "${td_args[@]}" "${extra[@]}" "$@"
}

gcl() {  # gcl <seed> <experiment> <row> <default sampler id> [--set ...]: train one row
  local seed=$1 exp=$2 row=$3 sid=$4
  shift 4
  uv run cga run gcl_train --dataset "$DATASET" --experiment "$exp" --set "seed=$seed" \
    --set "gcl_train.model.model_id=$PREFIX$row-s$seed" \
    --set "gcl_train.inputs.training_data_sampler_id=$(sampler_id "$sid")" "$@"
}

has() { [[ "$PRIORITIES" == *" $1 "* ]]; }

for seed in "${SEEDS[@]}"; do
  if has 1 || has 2; then
    td "$seed" rank_topk-k5
    if has 1; then
      gcl "$seed" unsup_u1 u1 rank_topk-k5
      gcl "$seed" unsup_u2 u2 rank_topk-k5
    fi
    if has 2; then
      gcl "$seed" unsup_u1 u1-i rank_topk-k5 --set gcl_train.loss.name=infonce
      gcl "$seed" unsup_u2 u2-i rank_topk-k5 --set gcl_train.loss.name=infonce
    fi
  fi
  if has 3; then
    td "$seed" rank_topk-k5-negrandom --set training_data.sampler.negatives=random
    gcl "$seed" unsup_u2 u2-rand rank_topk-k5-negrandom
    td "$seed" rank_topk-k5-neghard-off10 --set training_data.sampler.negatives=hard
    gcl "$seed" unsup_u2 u2-hard rank_topk-k5-neghard-off10
    td "$seed" rank_topk-k5-posmutual --set training_data.sampler.positives=mutual
    gcl "$seed" unsup_u2 u2-mut rank_topk-k5-posmutual
  fi
done
