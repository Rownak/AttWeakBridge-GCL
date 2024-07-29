import os

# import sys
# #base_dir = os.environ['AWEB_DIR']
# sys.path.append("../../")
# import config

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
DATASETS = ['ics_attack','enterprise_attack', 'mobile_attack']
ATTACK_DATASET = 'ics_attack'
CWE_DATASET = 'cwec_v4.12'
# 0 - Select Only Description
# 1 - Select Full Metadata
description_selection = 0
LLM_FT_EPOCH = 10
GCL_EPOCH = 200
    
# Do not change
DATA_DIR = ROOT_PATH+"/datasets/{}/".format(ATTACK_DATASET)
CWE_DIR = ROOT_PATH+"/datasets/cwe/"
OUTPUT_DIR = ROOT_PATH+"/model_outputs/{}/".format(ATTACK_DATASET)
EMBEDDING_DIR = OUTPUT_DIR+"embeddings/"
DESCRIPTION_FILE = DATA_DIR+"doc_id_to_desc_only.json"
if(description_selection==1):
    DESCRIPTION_FILE = DATA_DIR+"doc_id_to_full_metadata.json"
assert ATTACK_DATASET in DATASETS, 'Dataset {} not in available datasets ({})'.format(ATTACK_DATASET, DATASETS)


