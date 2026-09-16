import numpy as np
import os
import json
import pickle
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import config

models = ["pt_SecRoBERTa","SecRoBERTa","pt_SecureBERT","SecureBERT","pt_gpt2-xl","gpt2-xl", "pt_gpt2","gpt2"]
result_dir = config.OUTPUT_DIR
embeddings_dir = config.EMBEDDING_DIR
data_dir = config.DATA_DIR
model_name = models[0]
text_emb_dir = embeddings_dir+model_name+"/"
output_dir = result_dir+"gcl_data/"+model_name
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Assuming predefined weights are stored in a numpy array named 'predefined_embeddings1_weights'
# text_embeddings = np.load(embeddings_dir+'deepwalk.npy')
text_embeddings = np.load(text_emb_dir+'text_embeddings.npy')
with open(data_dir+"doc_id_to_emb_id.json") as f:
    doc_id_to_emb_id = json.load(f)
with open(data_dir+"emb_id_to_doc_id.json") as f:
    emb_id_to_doc_id = json.load(f)
with open(data_dir+'attack_weak_range.json') as fp:
    attack_weak_range = json.load(fp)

print("text_embedding shape",text_embeddings.shape)
print("id_to_pos len",len(doc_id_to_emb_id))
print("pos_to_id len",len(emb_id_to_doc_id))


attack_range = attack_weak_range['attack']
weak_range = attack_weak_range['cwe']
total_nodes = weak_range[1]
print(attack_range)
print(weak_range)
print(total_nodes)


from sklearn.metrics.pairwise import cosine_similarity
# Compute the cosine similarity matrix
text_sim_matrix = cosine_similarity(text_embeddings)
np.save(output_dir+"/text_sim_matrix.npy", text_sim_matrix)
# np.save(output_dir+"/deepwalk_sim_matrix.npy", text_sim_matrix)

# Verify the shape of the resulting cosine similarity matrix
print(text_sim_matrix.shape)  # This should print (1136, 1136)




# Initialize an empty list to store the filtered results
attack_sorted_cos_sim = []
weak_sorted_cos_sim = []
# Iterate over the size in the cosine similarity matrix
for i in range(total_nodes):
    # Get the current row
    row = text_sim_matrix[i]
    
    # Get the sorted indices in descending order
    sorted_indices = np.argsort(-row)
    
    # Get the sorted cosine similarities
    sorted_similarities = row[sorted_indices]
    
    # Combine the indices and similarities
    sorted_row = list(zip(sorted_indices, sorted_similarities))
    if(i<attack_range[1]):
        # Filter the sorted row to keep only the pairs with indices from   0 to 202
        filtered_sorted_row = [pair for pair in sorted_row if attack_range[0] <= pair[0] < attack_range[1]]
        # Append the filtered sorted row to the results
        attack_sorted_cos_sim.append(filtered_sorted_row)
    else:
        # Filter the sorted row to keep only the pairs with indices from 203 to 1135
        filtered_sorted_row = [pair for pair in sorted_row if weak_range[0] <= pair[0] < weak_range[1]]
        # Append the filtered sorted row to the results
        weak_sorted_cos_sim.append(filtered_sorted_row)

with open(output_dir+'/attack_text.pkl', 'wb') as file:
    pickle.dump(attack_sorted_cos_sim, file)

with open(output_dir+'/weak_text.pkl', 'wb') as file:
    pickle.dump(weak_sorted_cos_sim, file)

print(output_dir)