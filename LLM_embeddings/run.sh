#!/bin/bash

# Define the model_ids
# models = ["jackaduma/SecRoBERTa", "ehsanaghaei/SecureBERT", "gpt2-xl", "gpt2"]
# model_names = ["SecRoBERTa", "SecureBERT", "gpt2-xl", "gpt2"]

model_ids=(0 3)

# Loop over each model_id
for model_id in "${model_ids[@]}"; do
    echo "Running model_id=$model_id"
    python 1_llm_fine_tune.py --model_id $model_id
    sleep 2
done

# Define the model_ids
# models = ["pt_SecRoBERTa", "SecRoBERTa", "pt_SecureBERT", "SecureBERT", "pt_gpt2-xl", "gpt2-xl", "pt_bert"]

model_ids2=(1 5)

# Loop over each model_id
for model_id in "${model_ids2[@]}"; do
    echo "Running model_id=$model_id"
    python 2_extract_embeddings.py --model_id $model_id
    sleep 2
done

