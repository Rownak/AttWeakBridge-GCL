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
#
# Phase 9.4 (DeepWalk is the only encoder input, `gcl_train.inputs.text_features=graph`); needs the 9.3 `sap-rank_topk-k5` sampler:
#   PREFIX=dw- SAMPLER_PREFIX=sap- scripts/run_unsup_matrix.sh ncit_doid_unsup 4        # priority 4: dw-u1, dw-u1-i, dw-u2, dw-u2-i
#   TEXT_EMBEDDER=deepwalk-deepwalk PREFIX=dw- scripts/run_unsup_matrix.sh ncit_doid_unsup 5   # priority 5: dw-u2-own (DeepWalk's own pairs)
# Phase 9.5 (SapBERT residual with GCN branches, InfoNCE, SapBERT pairs of the seed):
#   PREFIX=res- SAMPLER_PREFIX=sap- scripts/run_unsup_matrix.sh ncit_doid_unsup 6            # priority 6: res-r1, res-r2, res-r3
# Phase 10.3 (ATT&CK-CWE, SecureBERT2 residual GCN without the frozen block, InfoNCE, cross-graph SecureBERT2 pairs of the seed):
#   TEXT_EMBEDDER=hf_mean_pool-SecureBERT2 PREFIX=ent- SAMPLER_PREFIX=sb2- scripts/run_unsup_matrix.sh attack_cwe_ent 7
#   priority 7: ent-g-sb2 (sampler id sb2-rank_topk-k5)
# Phase 10.5 (the 10.3 row with SecRoBERTa as the text encoder):
#   TEXT_EMBEDDER=hf_mean_pool-pt_SecRoBERTa PREFIX=ent- SAMPLER_PREFIX=srb- scripts/run_unsup_matrix.sh attack_cwe_ent 8
#   priority 8: ent-g-srb (sampler id srb-rank_topk-k5)
# SAMPLER_PREFIX (default PREFIX) names the sampler lineage the rows read, so model ids and pseudo-pair provenance are set apart.
set -euo pipefail

DATASET=$1
shift
PRIORITIES=" $* "
SEEDS=(42 43 44)
PREFIX=${PREFIX:-}
TEXT_EMBEDDER=${TEXT_EMBEDDER:-}
SAMPLER_PREFIX=${SAMPLER_PREFIX-$PREFIX}

td_args=()
[[ -n "$TEXT_EMBEDDER" ]] && td_args+=(--set "training_data.inputs.text_embeddings.embedder_id=$TEXT_EMBEDDER")

sampler_id() {  # sampler_id <default id>: the id training_data writes to and gcl_train reads from
  echo "$SAMPLER_PREFIX$1"
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

pinned_run() {  # pinned_run <seed> <default sampler id>: the newest training_data run of that lineage built with this seed
  uv run python - "$DATASET" "$(sampler_id "$2")" "$1" <<'PYEOF2'
import json, sys
from pathlib import Path
dataset, sid, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
runs = sorted(Path("artifacts/training_data", dataset, sid).glob("2*/manifest.json"))
hits = [r.parent.name for r in runs if json.loads(r.read_text(encoding="utf-8")).get("seed") == seed]
if not hits:
    sys.exit(f"no {sid} run built with seed {seed}")
print(hits[-1])
PYEOF2
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
  if has 4; then
    for row in u1 u1-i u2 u2-i; do
      exp=unsup_u2
      [[ "$row" == u1* ]] && exp=unsup_u1
      extra=()
      [[ "$row" == *-i ]] && extra+=(--set gcl_train.loss.name=infonce)
      # Each seed reads the pair set the 9.3 run of that seed used, not the lineage's LATEST (the last seed).
      gcl "$seed" "$exp" "$row" rank_topk-k5 --set gcl_train.inputs.text_features=graph         --set "gcl_train.inputs.training_data_run=$(pinned_run "$seed" rank_topk-k5)" "${extra[@]}"
    done
  fi
  if has 6; then
    res=(--set gcl_train.model.encoder=residual_adapter --set gcl_train.loss.name=infonce
         --set "gcl_train.inputs.training_data_run=$(pinned_run "$seed" rank_topk-k5)")
    gcl "$seed" unsup_u1 r1 rank_topk-k5 "${res[@]}" --set gcl_train.model.adapter=gcn
    gcl "$seed" unsup_u1 r2 rank_topk-k5 "${res[@]}" --set gcl_train.model.adapter=none --set gcl_train.model.graph_branch=true
    gcl "$seed" unsup_u1 r3 rank_topk-k5 "${res[@]}" --set gcl_train.model.adapter=gcn --set gcl_train.model.graph_branch=true
  fi
  if has 7; then
    td "$seed" rank_topk-k5
    # Sprint 5 A12 module: residual_adapter + gcn, no frozen block (z = h); each seed reads its own pair set.
    gcl "$seed" unsup_u1 g-sb2 rank_topk-k5 \
      --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=gcn \
      --set gcl_train.model.residual_floor=false --set gcl_train.loss.name=infonce \
      --set "gcl_train.inputs.training_data_run=$(pinned_run "$seed" rank_topk-k5)"
  fi
  if has 8; then
    td "$seed" rank_topk-k5
    # Same module as priority 7; only the text store (TEXT_EMBEDDER) and the sampler lineage differ.
    gcl "$seed" unsup_u1 g-srb rank_topk-k5 \
      --set gcl_train.model.encoder=residual_adapter --set gcl_train.model.adapter=gcn \
      --set gcl_train.model.residual_floor=false --set gcl_train.loss.name=infonce \
      --set "gcl_train.inputs.training_data_run=$(pinned_run "$seed" rank_topk-k5)"
  fi
  if has 5; then
    td "$seed" rank_topk-k5
    gcl "$seed" unsup_u2 u2-own rank_topk-k5 --set gcl_train.inputs.text_features=graph
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
