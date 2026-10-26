import networkx as nx 
import random
import json
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.insert(0, project_root)
import config
import os

blacklist = []
data_dir = config.DATA_DIR
cwe_dir = config.CWE_DIR
# Load the JSON data for attack graph
with open(data_dir+'attack_graph.json', 'r') as file:
    a_graph_data = json.load(file)
print("len of links: ",len(a_graph_data['links']))
if not os.path.exists(data_dir+"/plots/"):
    os.makedirs(data_dir+"/plots/")
with open(data_dir+"attack_info.json") as f:
    attack_info = json.load(f)
with open(cwe_dir+config.CWE_DATASET+"_info.json") as f:
    cwe_info = json.load(f)
# Create dictionary to store the id_pos and pos_id mapping
doc_id_to_emb_id = {}
emb_id_to_doc_id = {}
attack_weak_range = {}
emb_id_to_tid = {}

# Create a new graph
a_graph = nx.Graph()
# Add nodes
for node in a_graph_data['nodes']:
    a_graph.add_node(node['id'], **node)
# Add edges
for i in range(len(a_graph_data['links'])):
    source = a_graph_data['links'][i]['source']
    target = a_graph_data['links'][i]['target']
    if(attack_info[source]['Description'].strip()=='' or attack_info[target]['Description'].strip()==''):
        if(attack_info[source]['Description'].strip()==''):
            print("Empty Source: ", source)
        else:
            print("Empty Target: ", target)
        continue
    if(source not in doc_id_to_emb_id and source not in blacklist):
        pos = len(doc_id_to_emb_id.keys())
        doc_id_to_emb_id[source] = pos
        emb_id_to_doc_id[pos]=source
        emb_id_to_tid[pos] = attack_info[source]['t_id']
    if(target not in doc_id_to_emb_id and target not in blacklist):
        pos = len(doc_id_to_emb_id.keys())
        doc_id_to_emb_id[target] = pos
        emb_id_to_doc_id[pos]=target
        emb_id_to_tid[pos] = attack_info[target]['t_id']
    a_graph.add_edge(source, target)
    #a_graph.add_edge(target, source, weight=1.0-a_graph_data['links'][i]['similarity'])
    # G_all_sim.add_edge(source, target)

a_graph.number_of_edges(), a_graph.number_of_nodes()

attack_size=len(doc_id_to_emb_id)
attack_weak_range['attack']=(0,attack_size)
attack_size

# Create a new graph for weakness data
with open(cwe_dir+config.CWE_DATASET+'_graph.json') as f:
    w_graph_data = json.load(f)
w_graph = nx.Graph()
# Add nodes
for node in w_graph_data['nodes']:
    w_graph.add_node(node['ID'], **node)

# Add edges
for i in range(len(w_graph_data['links'])):
    source = w_graph_data['links'][i]['source']
    target = w_graph_data['links'][i]['target']
    if((cwe_info[source]['Description'].strip()=='' and cwe_info[source]['Extended_Description'].strip()=='') 
       or (cwe_info[target]['Description'].strip()=='' and cwe_info[target]['Extended_Description'].strip()=='')):
        if(cwe_info[source]['Description'].strip()==''):
            print("Empty Source: ", source)
        else:
            print("Empty Target: ", target)
        continue
    if(source not in doc_id_to_emb_id):
        pos = len(doc_id_to_emb_id.keys())
        doc_id_to_emb_id[source] = pos
        emb_id_to_doc_id[pos]=source
        emb_id_to_tid[pos] =source
    if(target not in doc_id_to_emb_id):
        pos = len(doc_id_to_emb_id.keys())
        doc_id_to_emb_id[target] = pos
        emb_id_to_doc_id[pos]=target
        emb_id_to_tid[pos] =target
    w_graph.add_edge(source, target)
    #w_graph.add_edge(target, source, weight=1.0-link['similarity'])
    # G_all_sim.add_edge(source, target)

weak_size = len(doc_id_to_emb_id)-attack_size
attack_weak_range['cwe']=(attack_size,len(doc_id_to_emb_id))
attack_weak_range['cwe']
attack_weak_range['n_nodes']=len(doc_id_to_emb_id)

with open(data_dir+'emb_id_to_tid.json','w') as fp:
    json.dump(emb_id_to_tid,fp)

with open(data_dir+'doc_id_to_emb_id.json','w') as fp:
    json.dump(doc_id_to_emb_id,fp)
with open(data_dir+'emb_id_to_doc_id.json','w') as fp:
    json.dump(emb_id_to_doc_id,fp)
with open(data_dir+'attack_weak_range.json','w') as fp:
    json.dump(attack_weak_range,fp)

# Get all connections and shortest path according to hop distance

def get_shortest_path(graph_data, graph):
    all_links = []
    count = 0
    w_dict = {}
    for source in graph.nodes:
        
        for target in graph.nodes: #len(graph_data['links']
            
            if(source==target or (source,target) in w_dict or (target,source) in w_dict):
                continue

            # following line give shortest path according to the weight
            # dist2 = nx.shortest_path_length(graph,source,target2, weight='weight')
            # Following line gives shortes path according to hop distance
            try:
                dist = nx.shortest_path_length(graph,source,target)

                w_dict[(source,target)] = dist
                #w_dict[(target,source)] = dist
            except:
                count+=1
                #print("No path between "+source+" ",target )
    print("No path: ", count)
    return w_dict

attack_w_dict = get_shortest_path(a_graph_data,a_graph)
weakness_w_dict = get_shortest_path(w_graph_data, w_graph)

def create_training(weight_dict):
    pair_data = []
    missing_nodes = set()
    mx_dist = max(list(weight_dict.values()))
    print("max distance ", mx_dist)
    for key in weight_dict.keys():
        if(key[0] in doc_id_to_emb_id):
            source_id = doc_id_to_emb_id[key[0]]
            if(key[1] in doc_id_to_emb_id):
                target_id = doc_id_to_emb_id[key[1]]
                pair_data.append([source_id,target_id, weight_dict[key]/float(mx_dist)])
            else:
                missing_nodes.add(key[1])
        else:
            missing_nodes.add(key[0])
    print(missing_nodes)
    return pair_data, missing_nodes

all_attack_dist, missing_attack = create_training(attack_w_dict)
all_weakness_dist, missing_weakness = create_training(weakness_w_dict)

training_data = all_attack_dist+all_weakness_dist
len(training_data)

import numpy as np
np.save(data_dir+'hop_training_data.npy', np.array(training_data)) # save
#new_num_arr = np.load('data.npy') # load

def get_hop_sim_matrix(all_dist, size, isAttack):
    hop_sim_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for n1,n2,dist in all_dist:
        if(not isAttack):
            n1=n1-attack_size
            n2=n2-attack_size
        # print("n1",n1)
        # print("n2",n2)
        
        hop_sim_matrix[n1][n2]=1.0001-dist
        hop_sim_matrix[n2][n1]=1.0001-dist
    return hop_sim_matrix

weakness_hop_sim_matrix = get_hop_sim_matrix(all_weakness_dist, weak_size, False)
attack_hop_sim_matrix = get_hop_sim_matrix(all_attack_dist, attack_size, True)

count=0
for i in range(len(attack_hop_sim_matrix)):
    for j in range(len(attack_hop_sim_matrix[0])):
        if(attack_hop_sim_matrix[i][j]==0):
            count+=1
print(count)

np.save(data_dir+'weakness_hop_sim_matrix.npy', np.array(weakness_hop_sim_matrix)) # save
np.save(data_dir+'attack_hop_sim_matrix.npy', np.array(attack_hop_sim_matrix)) # save