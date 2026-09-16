# AttWeakBridge focuses on bringing nodes with similar text and graph properties 
# within the same neighborhood. In this experiment, we investigate how effectively 
# AttWeakBridge brings such nodes together. Since the experiment is focused on the 
# neighborhood, we use precision@k, where k is the number of nearest neighbors. The 
# precision is computed with respect to either graph neighborhood using shortest path 
# distance, or text distance using BERT embeddings. 

import csv
from collections import defaultdict
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics.pairwise import cosine_similarity
import sys
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
import config

def convert_keys_to_int(d):
    new_dict = {}
    for k, v in d.items():
        new_key = int(k) if k.isdigit() else k
        if isinstance(v, dict):
            new_dict[new_key] = convert_keys_to_int(v)
        else:
            new_dict[new_key] = v
    return new_dict

result_dir = config.OUTPUT_DIR
embeddings_dir = config.EMBEDDING_DIR
data_dir = config.DATA_DIR
print(result_dir)
with open(data_dir+'attack_weak_range.json') as fp:
    attack_weak_range = json.load(fp)

attack_size=attack_weak_range['attack'][1]
weakness_size=attack_weak_range['cwe'][1]-attack_size

from collections import defaultdict
sample = 10
dual = "dual3"
loss_func = "triplet"
plot_dir = result_dir+"plots/sample_{}/{}/".format(sample,loss_func)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
if not os.path.exists(plot_dir+"src/"):
    os.makedirs(plot_dir+"src/")
models = ["gpt2-xl", "gpt2-xl"]
gnn_models = ["GCN","GAT"]
features = ["text", "text_hop", "node2vec", "deepwalk"]
# gnn_models = ["GAN_embeddings","GCN_embeddings"]
graph_embeddings = {}
graph_embeddings["node2vec"]=np.load(embeddings_dir+"node2vec.npy")
graph_embeddings["deepwalk"]=np.load(embeddings_dir+"deepwalk.npy")
gnn_embeddings = defaultdict(list)
text_hop_embeddings = defaultdict(list)
text_hop_embeddings["bert"].append(np.load(embeddings_dir+"pt_bert/text_embeddings.npy"))
count=0
for model in models:
    text_emb_dir = embeddings_dir+model+"/"
    gcl_data_dir = result_dir+"gcl_data/"+model+"/"
    text_hop_embeddings["text"].append(np.load(text_emb_dir+"text_embeddings.npy"))
    text_hop_embeddings["hop_text"].append(np.load(text_emb_dir+"text_hop_embeddings.npy"))
    

    gnn_hop = []
    gnn_node2vec = []
    gnn_deepwalk = []
    gnn_text = []
    for j in range(len(gnn_models)):

        if(count==0):
            dual = ""
            # gnn_embeddings2=np.load(gcl_data_dir+"sample_{}/{}/{}/text_hop_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            # gnn_hop.append(gnn_embeddings2)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/node2vec_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            gnn_node2vec.append(gnn_embeddings3)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/deepwalk_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            gnn_deepwalk.append(gnn_embeddings3)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/text_gm_1.0.npy".format(sample,gnn_models[j],loss_func))
            gnn_text.append(gnn_embeddings3)
        else:
            dual = "dual3"
            # gnn_embeddings2=np.load(gcl_data_dir+"sample_{}/{}/{}/text_hop_{}_gm_1.0.npy".format(sample,gnn_models[j],loss_func,dual))
            # gnn_hop.append(gnn_embeddings2)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/text_node2vec_{}_gm_1.0.npy".format(sample,gnn_models[j],loss_func,dual))
            gnn_node2vec.append(gnn_embeddings3)
            gnn_embeddings3 = np.load(gcl_data_dir+"sample_{}/{}/{}/text_deepwalk_{}_gm_1.0.npy".format(sample,gnn_models[j],loss_func,dual))
            gnn_deepwalk.append(gnn_embeddings3)

        
    
        # gnn_embeddings2.append(np.load(dir_name+"{}/sample_{}/hop_gm_1.0.npy".format(gnn_models[j],sample)))
        # gnn_emb_matrix2.append(cosine_similarity(gnn_embeddings2[-1]))
        # gnn_embeddings2.append(np.load(dir_name+"{}/sample_{}/node2vec_gm_1.0.npy".format(gnn_models[j],sample)))
        # gnn_emb_matrix2.append(cosine_similarity(gnn_embeddings2[-1]))
    # gnn_embeddings["hop_text"].append(gnn_hop)
    gnn_embeddings["node2vec"].append(gnn_node2vec)
    gnn_embeddings["deepwalk"].append(gnn_deepwalk)
    if(count==0):
        gnn_embeddings["text"].append(gnn_text)
    count+=1

from scipy.stats import spearmanr
def calc_intersect(base, cmp_emb, ctype, isAverage, isSrc):
    # Loop over each k from 10 to 1136
    average_hit_at_k = {}
    avg_src_at_k = {}
    if(ctype=="attack_graph"):
        size = attack_size
        cmp_matrix = cosine_similarity(cmp_emb[:attack_size],cmp_emb[:attack_size])
        base_matrix = base
    elif(ctype=="weak_graph"):
        size = weakness_size
        cmp_matrix =  cosine_similarity(cmp_emb[attack_size:],cmp_emb[attack_size:])
        base_matrix = base
    if(ctype=="attack_text"):
        size = attack_size
        cmp_matrix = cosine_similarity(cmp_emb[:attack_size],cmp_emb[:attack_size])
        base_matrix = cosine_similarity(base[:attack_size],base[:attack_size])
    elif(ctype=="weak_text"):
        size = weakness_size
        cmp_matrix =  cosine_similarity(cmp_emb[attack_size:],cmp_emb[attack_size:])
        base_matrix = cosine_similarity(base[attack_size:],base[attack_size:])
    elif(ctype=="text"):
        size = attack_size+weakness_size
        cmp_matrix =  cosine_similarity(cmp_emb)
        base_matrix = cosine_similarity(base)
    for k in range(5, 30,5):
        hit_at_k = []
        src_at_k = []
        for i in range(size):
            # Get the indices of the top k values in the row for both matrices

            top_base = np.argsort(base_matrix[i])[-k:]
            top_k = np.argsort(cmp_matrix[i])[-k:]
            
            # Calculate the hit@k (number of common indices in top k)
            #hit_count = len(set(top_bron) & set(top_k))
            if(isSrc==1):
                spearman_corr, _ = spearmanr(top_base, top_k)
                src_at_k.append(spearman_corr)
            else:
                if(isAverage):
                    hit_count = len(set(top_base) & set(top_k))/k
                    
                else:
                    hit_count = len(set(top_base) & set(top_k))
                hit_at_k.append(hit_count)
            # Append the result to the list
            
            
        # Calculate the average hit@k for the current k
        # print(hit_at_k)
        if(isSrc==1):
            avg_src_at_k[k] = np.mean(src_at_k)
        else:
            average_hit_at_k[k] = np.mean(hit_at_k)
    if(isSrc==1):    
        return avg_src_at_k
    else:
        return average_hit_at_k

def list_subtract(list1,list2):
    return [x-y for x,y in zip(list2,list1)]

list_subtract([1,2,3,4],[2,3,9,5])

colors = list(mcolors.TABLEAU_COLORS)
print(colors)


def plot_hit_k_graph_gnn(base, ctype, isAverage, isSrc):
    isGraph = True
    dual = "s_d"
    models = ["Single encoder","Dual encoder"]
    if(ctype!="text" and ctype.split("_")[1]=="text"):
        isGraph = False
    # Initialize a dictionary to store average hit@k for each k
    markers = ['o', 's', '+', 'v', '<', '>']
    colors = ['tab:pink', 'tab:gray', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:blue', 'tab:orange', 'tab:olive', 'tab:cyan']
    colors.pop(3)
    plt.figure(figsize=(3, 3))
    # Loop over all the models
    #models = ["pretrained_SecBert", "SecBert", "pretrained_SecureBert",  "SecureBert", "pretrained_Gpt2","Gpt2"]
    #models = ["Gpt2"]
    ks = list(range(5, 30,5))
    if(ctype.split("_")[0]=="attack"):
        size = attack_size
    else:
        size = weakness_size
    hit_differ = {}
    hits2 = list(calc_intersect(base, graph_embeddings["node2vec"], ctype, isAverage, isSrc).values())
    hits3 = list(calc_intersect(base, graph_embeddings["deepwalk"], ctype, isAverage, isSrc).values())
    
    if(isGraph):
        base_hit = hits3
    else:
        base_hit = list(calc_intersect(base, text_hop_embeddings["text"][0], ctype, isAverage, isSrc).values())
    hit_differ["node2vec"] = [hits2,list_subtract(hits2,base_hit)]
    hit_differ["deepwalk"] = [hits3,list_subtract(hits3,base_hit)]
    
    # plt.plot(ks, hits2, label="node2vec", marker='*', color = 'tab:olive')
    plt.plot(ks, hits3, label="DeepWalk Graph emb.", marker='s', color = 'tab:cyan')
    all_hits = []
    for i in range(0,len(models),1):
        # Define the cosine similarity matrices
        model = models[i]

        hits7 = list(calc_intersect(base, text_hop_embeddings["text"][i], ctype, isAverage, isSrc).values())
        # hits8 = list(calc_intersect(base, text_hop_embeddings["hop_text"][i], ctype, isAverage, isSrc).values())
        name = model+"_"
        # hit_differ[name+"text"]=[hits7,list_subtract(hits7,base_hit)]
        # hit_differ[name+"hop_text"]=[hits8,list_subtract(hits8,base_hit)]
        if(i==0):
            cl1 = "tab:green"
            plt.plot(ks, hits7, label="Fine-tuned Gpt-2 Text emb.", marker='.', color = cl1)
        else:
            cl1 = "tab:purple"
        
        # plt.plot(ks, hits8, label=models[i]+" Hop emb", marker='v', color = cl1)
        
        for j in range(len(gnn_models)):
            name = model+"_"+gnn_models[j]+"_"
            # hits4 = list(calc_intersect(base, gnn_embeddings["hop_text"][i][j], ctype, isAverage, isSrc).values())
            hits5 = list(calc_intersect(base, gnn_embeddings["node2vec"][i][j], ctype, isAverage, isSrc).values())            # if(j==2 or j==5):
            hits6 = list(calc_intersect(base, gnn_embeddings["deepwalk"][i][j], ctype, isAverage, isSrc).values()) 
            # hit_differ[name+"hop_text"]=[hits4,list_subtract(hits4,base_hit)]
            hit_differ[name+"node2vec"]=[hits5,list_subtract(hits5,base_hit)]
            hit_differ[name+"deepwalk"]=[hits6,list_subtract(hits6,base_hit)]
            if(i==0):
                hits9 = list(calc_intersect(base, gnn_embeddings["text"][i][j], ctype, isAverage, isSrc).values()) 
                hit_differ[name+"text"]=[hits9,list_subtract(hits9,base_hit)]
            # plt.plot(ks, hits4, label=models[i]+" "+gnn_models[j] +" hop", marker="s", color = colors[j+i*5])
            # plt.plot(ks, hits5, label=models[i]+" "+gnn_models[j]+ " node2vec", marker="*", color = colors[j+i*5])
                plt.plot(ks, hits6, label=models[i]+" "+gnn_models[j]+ " DeepWalk", marker="s", color = colors[j+i*5])
            if(i==1):
                plt.plot(ks, hits6, label=models[i]+" "+gnn_models[j], marker="s", color = colors[j+i*5])
            # if(dual==""):
            if(i==0):
                plt.plot(ks, hits9, label=models[i]+" "+gnn_models[j]+ " Text emb.", marker="o", color = colors[j+i*5])
        

        


    plt.xlabel('K')
    plt.ylabel('Average Precision@k')
    plt.xticks(ks)
    plt.grid(True)
    plt.legend(fontsize="12",bbox_to_anchor=(1.1, 1.05))
    #plt.title('GNN Semi Supervised - Text - Hop Embeddings')
    
    if(isSrc==1):
        evl_method = "SRC"
        plot_dir2 = plot_dir+"src/"
    else:
        evl_method = "hit@k"
        plot_dir2 = plot_dir
    if(ctype.split("_")[0]=="attack"):
        if(isAverage):
            #plt.title('GNN-{} Avg Hit@k Attack - {}'.format(dual,ctype))
            #title = "avg_bron_hit@k_gnn_sample_{}_th_{}_{}.png".format(sample, threshold[0], threshold[1])
            title = "e1_{}_hit@k_gnn_{}_sample_{}.png".format(ctype,dual,sample)
        else:
            #plt.title('GNN-{} Hit@k Attack - {}'.format(dual,ctype))
            #title = "avg_bron_hit@k_gnn_sample_{}_th_{}_{}.png".format(sample, threshold[0], threshold[1])
            title = "e1_{}_hit@k_gnn_{}_sample_{}.png".format(ctype,dual,sample)
    elif(ctype.split("_")[0]=="weak"):
        if(isAverage):
            #plt.title('GNN-{} Avg Hit@k CWE to {}'.format(dual,ctype))
            #title = "avg_bron_hit@k_gnn_sample_{}_th_{}_{}.png".format(sample, threshold[0], threshold[1])
            title = "e1_{}_hit@k_gnn_{}_sample_{}.png".format(ctype,dual,sample)
        else:
            #plt.title('GNN-{} Hit@k CWE to {}'.format(dual,ctype))
            #title = "avg_bron_hit@k_gnn_sample_{}_th_{}_{}.png".format(sample, threshold[0], threshold[1])
            title = "e1_{}_hit@k_gnn_{}_sample_{}.png".format(ctype,dual,sample)
    elif(ctype=="text"):
        if(isAverage):
            #plt.title('GNN-{} Avg Hit@k Bert Text'.format(dual))
            #title = "avg_bron_hit@k_gnn_sample_{}_th_{}_{}.png".format(sample, threshold[0], threshold[1])
            title = "e1_{}_hit@k_gnn_{}_sample_{}.png".format(ctype,dual,sample)
        else:
            #plt.title('GNN-{} Hit@k Bert Text'.format(dual))
            #title = "avg_bron_hit@k_gnn_sample_{}_th_{}_{}.png".format(sample, threshold[0], threshold[1])
            title = "e1_{}_hit@k_gnn_{}_sample_{}.png".format(ctype,dual,sample)
    print(plot_dir2)
    plt.savefig(plot_dir2+title, dpi=300, bbox_inches='tight')
    print("save file name: ", plot_dir2+title)
    plt.show()
    return hit_differ


attack_text_hit_differ= plot_hit_k_graph_gnn(text_hop_embeddings["bert"][0], "attack_text", True,0)

weak_text_hit_differ=weak_text_hit_differ = plot_hit_k_graph_gnn(text_hop_embeddings["bert"][0], "weak_text", True,0)

all_text_hit_differ=plot_hit_k_graph_gnn(text_hop_embeddings["bert"][0], "text", True, 0)

weakness_hop_sim_matrix = np.load(data_dir+'weakness_hop_sim_matrix.npy')
attack_hop_sim_matrix = np.load(data_dir+'attack_hop_sim_matrix.npy')

attack_graph_hit_differ = plot_hit_k_graph_gnn(attack_hop_sim_matrix, "attack_graph", True, 0)

weak_graph_hit_differ = plot_hit_k_graph_gnn(weakness_hop_sim_matrix, "weak_graph", True, 0)