#!/bin/bash

# Define the model_ids
model_ids=(3)

# # Loop over each model_id
# for model_id in "${model_ids[@]}"; do
#     echo "Running model_id=$model_id"
#     python 1_llm_fine_tune.py --model_id $model_id
#     sleep 2
# done

# Define the model_ids
model_ids2=(6)

# Loop over each model_id
for model_id in "${model_ids2[@]}"; do
    echo "Running model_id=$model_id"
    python create_embeddings.py --model_id $model_id
    sleep 2
done

