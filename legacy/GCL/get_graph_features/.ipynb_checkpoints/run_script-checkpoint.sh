#!/bin/bash
export AWEB_DIR
# # # Define the model_ids
# model_ids=(4 5 6 7)

# # Loop over each model_id
# for model_id in "${model_ids[@]}"; do
#     echo "Running model_id=$model_id"
#     python 1_text_hop_emb_model.py --model_id $model_id

#     # Wait for 2 seconds between runs to allow GPU memory to clear
#     sleep 2
# done

python 2_node2vec_deepwalk.py