#!/bin/bash

# Get the directory of the current script
AWEB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export the BASE_DIR as an environment variable
export AWEB_DIR

# model_ids=(3)
# for model_id in "${model_ids[@]}"; do
#     echo "Running model_id=$model_id"
#     python ./LLM_embeddings/1_llm_fine_tune.py --model_id $model_id
#     sleep 2
# done

model_ids2=(6)
for model_id in "${model_ids2[@]}"; do
    echo "Running model_id=$model_id"
    python ./LLM_embeddings/2_extract_embeddings.py --model_id $model_id
    sleep 2
done

# model_ids=(4 5 6 7)

# for model_id in "${model_ids[@]}"; do
#     echo "Running model_id=$model_id"
#     python ./GCL/get_graph_features/1_text_hop_emb_model_batch.py --model_id $model_id
#     sleep 2
# done

# python ./GCL/get_graph_features/2_node2vec_deepwalk.py
