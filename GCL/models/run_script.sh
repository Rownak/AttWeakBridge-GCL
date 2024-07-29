#!/bin/bash
# Define the model_ids, gnn_models, and samples
# model_names=("pt_SecRoBERTa" "SecRoBERTa" "pt_SecureBERT" "SecureBERT" "pt_gpt2-xl" "gpt2-xl")
# gnn_models=("GCN" "GAT")
# features=("text_hop" "node2vec" "deepwalk")
model_names=("gpt2-xl")
gnn_models=("GCN" "GAT")
features=("text_hop" "node2vec" "deepwalk")
samples=(3)

# Loop over each model_id, gnn_model, and sample
for model_name in "${model_names[@]}"; do
    for gnn_model in "${gnn_models[@]}"; do
        for sample in "${samples[@]}"; do
            for feature in "${features[@]}"; do
                echo "Running model_name=$model_name with gnn_model=$gnn_model and sample=$sample and feature=$feature"
    
                python GNN_dual.py --model_name $model_name --gnn_model $gnn_model --sample $sample --feature $feature
                # Wait for 10 seconds between runs to allow GPU memory to clear
                sleep 2
            done
        done
    done
done
