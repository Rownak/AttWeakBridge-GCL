import numpy as np
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
# Optimized
from collections import defaultdict
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import config
import pickle
from scipy.stats import percentileofscore




# models = ["pt_SecRoBERTa","SecRoBERTa","pt_SecureBERT","SecureBERT","pt_gpt2-xl","gpt2-xl"]
models = ["pt_SecRoBERTa"]

result_dir = config.OUTPUT_DIR
embeddings_dir = config.EMBEDDING_DIR
data_dir = config.DATA_DIR
model_name = models[0]
top_percentage = 0.03
bottom_percentage = 0.03
cwe_per_attack = 30
attack_per_cwe = 30
positive_threshold = 0.65
negative_threshold = 0.45
text_emb_dir = embeddings_dir+model_name+"/"
output_dir = result_dir+"gcl_data/"+model_name+"/"
# Assuming predefined weights are stored in a numpy array named 'predefined_embeddings1_weights'
text_embeddings = np.load(text_emb_dir+'text_embeddings.npy')

deepwalk_embeddings = np.load(embeddings_dir+'deepwalk.npy')
with open(data_dir+"doc_id_to_emb_id.json") as f:
    doc_id_to_emb_id = json.load(f)
with open(data_dir+"emb_id_to_doc_id.json") as f:
    emb_id_to_doc_id = json.load(f)
with open(data_dir+'attack_weak_range.json') as fp:
    attack_weak_range = json.load(fp)
with open(output_dir+'/attack_text.pkl','rb') as f:
    attack_sim = pickle.load(f)
with open(output_dir+'/weak_text.pkl','rb') as f:
    weak_sim = pickle.load(f)


attack_range = attack_weak_range['attack']
weak_range = attack_weak_range['cwe']


# Generate embeddings for both sets of nodes
attack_nodes = list(range(attack_range[0],attack_range[1]))
weakness_nodes = list(range(weak_range[0],weak_range[1]))
attack_embeddings = text_embeddings[attack_nodes]
weakness_embeddings = text_embeddings[weakness_nodes]

deepwalk_attack_embeddings = deepwalk_embeddings[attack_nodes]
deepwalk_weakness_embeddings = deepwalk_embeddings[weakness_nodes]


# Compute cosine similarity between all pairs of cwe nodes in text embeddings
text_weak_attack_matrix = cosine_similarity(weakness_embeddings, attack_embeddings)
# Compute cosine similarity between all pairs of attack nodes in text embeddings
text_attack_weak_matrix = cosine_similarity(attack_embeddings, weakness_embeddings)

# Compute cosine similarity between all pairs of attack nodes in deepwalk embeddings
deepwalk_attack_matrix = cosine_similarity(deepwalk_attack_embeddings, deepwalk_attack_embeddings)
# Compute cosine similarity between all pairs of cwe nodes in deepwalk embeddings
deepwalk_weak_matrix = cosine_similarity(deepwalk_weakness_embeddings, deepwalk_weakness_embeddings)


# anchor, positive-text, negative-text, positive-graph, negative-graph
import random
def get_sample_10(weak_attack_matrix,attack_weak_matrix, graph_weak_matrix, graph_attack_matrix,  sample_no=10):
    anchor_pos_neg_triple = []
    for i, attack_node in enumerate(attack_nodes):
        # Get the similarity scores for the current attack node
        sim_scores = attack_weak_matrix[i]
        sim_scores2 = graph_attack_matrix[i]
        # Get the indices of the cwe_per_attack most similar and cwe_per_attack least similar nodes
        # This one is weak text positive negative
        top5_sim_indices = np.argsort(sim_scores)[-cwe_per_attack:]  # Most similar
        bottom5_sim_indices = np.argsort(sim_scores)[:cwe_per_attack]  # Least similar

        # Here 2 means cwe to attack graph positive negative
        top5_sim_indices2 = np.argsort(sim_scores2)[-attack_per_cwe:]  # Most similar
        bottom5_sim_indices2 = np.argsort(sim_scores2)[:attack_per_cwe]  # Least similar
        
        # Select the most similar positive nodes
        # Select the least similar negative nodes
        for j in range(len(top5_sim_indices)):
            idx1 = top5_sim_indices[j]
            idx2 = bottom5_sim_indices[j]
            idx3 = top5_sim_indices2[j]
            idx4 = bottom5_sim_indices2[j]
            anchor_pos_neg_triple.append((attack_node, weakness_nodes[idx1], weakness_nodes[idx2], attack_nodes[idx3], attack_nodes[idx4] ))


    
    for i, weak_node in enumerate(weakness_nodes):
        # Get the similarity scores for the current attack node
        sim_scores = weak_attack_matrix[i]
        sim_scores2 = graph_weak_matrix[i]
        # Get the indices of the 5 most similar and 5 least similar nodes
        top5_sim_indices = np.argsort(sim_scores)[-attack_per_cwe:]  # Most similar
        bottom5_sim_indices = np.argsort(sim_scores)[:attack_per_cwe]  # Least similar

        # Here 2 means attack graph positive negative
        top5_sim_indices2 = np.argsort(sim_scores2)[-attack_per_cwe:]  # Most similar
        bottom5_sim_indices2 = np.argsort(sim_scores2)[:attack_per_cwe]  # Least similar
        
        # Select the most similar positive nodes
        # Select the least similar negative nodes
        for j in range(len(top5_sim_indices)):
            idx1 = top5_sim_indices[j]
            idx2 = bottom5_sim_indices[j]
            idx3 = top5_sim_indices2[j]
            idx4 = bottom5_sim_indices2[j]
            anchor_pos_neg_triple.append((weak_node, attack_nodes[idx1], attack_nodes[idx2],weakness_nodes[idx3], weakness_nodes[idx4]))
    anchor_set = set()
    positive_set=set()
    negative_set=set()   
    positive_set2=set()
    negative_set2=set()
    
    for a,p,n,pg,ng in anchor_pos_neg_triple:
        anchor_set.add(a)
        positive_set.add(p)
        negative_set.add(n)
        positive_set2.add(pg)
        negative_set2.add(ng)
        if(a>=weak_range[1] or p>=weak_range[1] or n>=weak_range[1]):
            print(a," ",p," ", n)
    print("# anchor nodes:",len(anchor_set))
    print("# text-postive nodes:",len(positive_set))
    print("# text-negative nodes :",len(negative_set))
    
    print("# text-not selected pos :",len(anchor_set-positive_set))
    print("# text-not selected neg  :",len(anchor_set-negative_set))
    print("# text-not selected intersect :",len((anchor_set-positive_set)&(anchor_set-negative_set)))

    
    print("# graph-postive nodes:",len(positive_set2))
    print("# graph-negative nodes :",len(negative_set2))

    print("# graph-not selected pos :",len(anchor_set-positive_set2))
    print("# graph-not selected neg  :",len(anchor_set-negative_set2))
    print("# graph-not selected intersect :",len((anchor_set-positive_set2)&(anchor_set-negative_set2)))
    
    print("# examples: ", len(anchor_pos_neg_triple))
    random.shuffle(anchor_pos_neg_triple)
    save_file = output_dir+'anchor_pos_neg_triple_{}.pkl'.format(sample_no)
    with open(save_file, 'wb') as f:
        print("Save at:", save_file)
        pickle.dump(anchor_pos_neg_triple, f)
    return anchor_pos_neg_triple

def get_top_bottom_pairs_row(i,sim_values, top_percentage,bottom_percentage, isAttack):
    top_threshold = np.percentile(sim_values, 100 - top_percentage * 100)
    bottom_threshold = np.percentile(sim_values, bottom_percentage * 100)
    # print("top_threshold",top_threshold)
    # print("bottom_threshold",bottom_threshold)
    top_pairs = []
    bottom_pairs = []
    for j in range(len(sim_values)):
        if(isAttack):
            
            neighbor_id = weakness_nodes[j]
        else:
            
            neighbor_id = attack_nodes[j]
        if sim_values[j] >= top_threshold:
            top_pairs.append((neighbor_id, sim_values[j]))
        elif sim_values[j] <= bottom_threshold:
            bottom_pairs.append((neighbor_id, sim_values[j]))
    return top_pairs, bottom_pairs


from collections import defaultdict
import random
def get_sample_11(weak_attack_matrix,attack_weak_matrix, graph_weak_matrix, graph_attack_matrix,  sample_no=11):
    top_percentage=0.04
    bottom_percentage=0.04
    anchor_pos_neg_triple = []
    for i, attack_node in enumerate(attack_nodes):
        # Get the index pairs for the top and bottom 4%
        anchor_pos_pair1, anchor_neg_pair1 = get_top_bottom_pairs_row(i,attack_weak_matrix[i], top_percentage, bottom_percentage, True)
        anchor_pos_pair2, anchor_neg_pair2 = get_top_bottom_pairs_row(i,graph_attack_matrix[i], top_percentage, bottom_percentage, False)
        print(len(anchor_pos_pair1))
        print(len(anchor_pos_pair2))
        #Sort pairs
        anchor_pos_pair1.sort(reverse=True, key=lambda x: x[1])
        anchor_neg_pair1.sort(key=lambda x: x[1])
        anchor_pos_pair2.sort(reverse=True, key=lambda x: x[1])
        anchor_neg_pair2.sort(key=lambda x: x[1])

        for j in range(len(anchor_pos_pair1)):
            #text better
            pos1,_ = anchor_pos_pair1[j]
            neg1,_ = anchor_neg_pair1[j]
            #graph poor
            randi = random.randrange(0, len(anchor_pos_pair2))
            pos2,_ = anchor_pos_pair2[randi]
            neg2,_ = anchor_neg_pair2[randi]
            anchor_pos_neg_triple.append((attack_node, pos1,neg1,pos2,neg2 ))
        
    print("Attack")
    print(len(anchor_pos_neg_triple))
    for i, weak_node in enumerate(weakness_nodes):
        # Get the index pairs for the top and bottom 25%
        anchor_pos_pair1, anchor_neg_pair1 = get_top_bottom_pairs_row(i,weak_attack_matrix[i], top_percentage+0.12, bottom_percentage+0.12, False)
        anchor_pos_pair2, anchor_neg_pair2 = get_top_bottom_pairs_row(i,graph_weak_matrix[i], top_percentage, bottom_percentage, True)
        #Sort pairs
        anchor_pos_pair1.sort(reverse=True, key=lambda x: x[1])
        anchor_neg_pair1.sort(key=lambda x: x[1])
        anchor_pos_pair2.sort(reverse=True, key=lambda x: x[1])
        anchor_neg_pair2.sort(key=lambda x: x[1])
        # print("Weak")
        # print(len(anchor_pos_pair1))
        # print(len(anchor_neg_pair1))

        # Make triplets (anchor, positive, negative)
        for j in range(len(anchor_pos_pair2)):
            #text poor
            randi = random.randrange(0, len(anchor_pos_pair1))
            pos1,_ = anchor_pos_pair1[randi]
            neg1,_ = anchor_neg_pair1[randi]
            #graph better
            pos2,_ = anchor_pos_pair2[j]
            neg2,_ = anchor_neg_pair2[j]
            anchor_pos_neg_triple.append((weak_node, pos1,neg1,pos2,neg2 ))

    anchor_set = set()
    positive_set=set()
    negative_set=set()   
    positive_set2=set()
    negative_set2=set()
    
    for a,p,n,pg,ng in anchor_pos_neg_triple:
        anchor_set.add(a)
        positive_set.add(p)
        negative_set.add(n)
        positive_set2.add(pg)
        negative_set2.add(ng)
        if(a>=weak_range[1] or p>=weak_range[1] or n>=weak_range[1]):
            print(a," ",p," ", n)
    print("# anchor nodes:",len(anchor_set))
    print("# text-postive nodes:",len(positive_set))
    print("# text-negative nodes :",len(negative_set))
    
    print("# text-not selected pos :",len(anchor_set-positive_set))
    print("# text-not selected neg  :",len(anchor_set-negative_set))
    print("# text-not selected intersect :",len((anchor_set-positive_set)&(anchor_set-negative_set)))

    
    print("# graph-postive nodes:",len(positive_set2))
    print("# graph-negative nodes :",len(negative_set2))

    print("# graph-not selected pos :",len(anchor_set-positive_set2))
    print("# graph-not selected neg  :",len(anchor_set-negative_set2))
    print("# graph-not selected intersect :",len((anchor_set-positive_set2)&(anchor_set-negative_set2)))
    
    print("# examples: ", len(anchor_pos_neg_triple))
    random.shuffle(anchor_pos_neg_triple)
    save_file = output_dir+'anchor_pos_neg_triple_{}.pkl'.format(sample_no)
    with open(save_file, 'wb') as f:
        print("Save at:", save_file)
        pickle.dump(anchor_pos_neg_triple, f)
    return anchor_pos_neg_triple

anchor_pos_neg_triple = get_sample_10(text_weak_attack_matrix,text_attack_weak_matrix,deepwalk_weak_matrix,deepwalk_attack_matrix, sample_no=10)
# anchor_pos_neg_triple = get_sample_11(text_weak_attack_matrix,text_attack_weak_matrix,deepwalk_weak_matrix,deepwalk_attack_matrix, sample_no=11)