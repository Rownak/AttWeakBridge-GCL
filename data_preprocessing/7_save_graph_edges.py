import numpy as np
import json
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.insert(0, project_root)
import config


data_dir = config.DATA_DIR
cwe_dir = config.CWE_DIR
with open(data_dir+'doc_id_to_emb_id.json') as f:
    doc_id_to_emb_id = json.load(f)
with open(data_dir+'emb_id_to_doc_id.json') as f:
    emb_id_to_doc_id = json.load(f)
with open(data_dir+'attack_graph.json') as f:
    attack_graph = json.load(f)
with open(cwe_dir+config.CWE_DATASET+'_graph.json') as f:
    cwe_graph = json.load(f)

combined_edges = []

attack_graph['links'][0]

for entry in attack_graph['links']:
    if(entry['source'] in doc_id_to_emb_id and entry['target'] in doc_id_to_emb_id):
        combined_edges.append((doc_id_to_emb_id[entry['source']], doc_id_to_emb_id[entry['target']]))

for entry in cwe_graph['links']:
    if(entry['source'] in doc_id_to_emb_id and entry['target'] in doc_id_to_emb_id):
        combined_edges.append((doc_id_to_emb_id[entry['source']], doc_id_to_emb_id[entry['target']]))

with open(data_dir+'graph_edges.json', 'w') as f:
    json.dump(combined_edges, f)

# Find the maximum of the first elements
max_first = max(pair[0] for pair in combined_edges)

# Find the maximum of the second elements
max_second = max(pair[1] for pair in combined_edges)

print("Max of first elements:", max_first)
print("Max of second elements:", max_second)