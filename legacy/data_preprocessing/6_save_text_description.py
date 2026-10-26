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


with open(data_dir+"attack_info.json") as f:
    attack_info = json.load(f)
with open(cwe_dir+config.CWE_DATASET+"_info.json") as f:
    cwe_info = json.load(f)
with open(data_dir+"doc_id_to_emb_id.json") as f:
    doc_id_to_emb_id = json.load(f)
with open(data_dir+"emb_id_to_doc_id.json") as f:
    emb_id_to_doc_id = json.load(f)

doc_id_to_desc={}
def save_description(include_all_metadata):
    for doc_id in doc_id_to_emb_id:
        text_desc = ""
        print(doc_id)
        if(doc_id in attack_info):
            text_desc+=attack_info[doc_id]['Description']
            if(include_all_metadata):
                text_desc+=". "+attack_info[doc_id]['x_mitre_detection']
        elif(doc_id in cwe_info):
            Name = cwe_info[doc_id]['Name']
            Description = cwe_info[doc_id]['Description']
            Extended_Description = cwe_info[doc_id]['Extended_Description']
            Modes_Of_Introduction = cwe_info[doc_id]['Modes_Of_Introduction']
            Common_Consequences = cwe_info[doc_id]['Common_Consequences']
            Detection_Methods = cwe_info[doc_id]['Detection_Methods']
            Potential_Mitigations = cwe_info[doc_id]['Potential_Mitigations']
            Demonstrative_Examples = cwe_info[doc_id]['Demonstrative_Examples']
            
            text_desc+=Name+". " if (Name!='N/A') else ""
            text_desc+=Description+" " if (Description!='N/A') else ""
            text_desc+=Extended_Description+" " if (Extended_Description!='N/A') else ""
            if(include_all_metadata):
                text_desc+=Modes_Of_Introduction+" " if (Modes_Of_Introduction!='N/A') else ""
                text_desc+=Common_Consequences+" " if (Common_Consequences!='N/A') else ""
                text_desc+=Detection_Methods+" " if (Detection_Methods!='N/A') else ""
                text_desc+=Potential_Mitigations+" " if (Potential_Mitigations!='N/A') else ""
                text_desc+=Demonstrative_Examples+" " if (Demonstrative_Examples!='N/A') else ""
        if(len(text_desc)<10):
            print(doc_id," : ",text_desc)
        doc_id_to_desc[doc_id]=text_desc
    return doc_id_to_desc


include_all_metadata = False
doc_id_to_desc=save_description(False)

with open(data_dir+"doc_id_to_desc_only.json",'w') as f:
    json.dump(doc_id_to_desc,f)

with open(data_dir+"doc_id_to_desc_only.json") as f:
    doc_id_to_desc_1 = json.load(f)

include_all_metadata = True
doc_id_to_desc=save_description(True)

with open(data_dir+"doc_id_to_full_metadata.json",'w') as f:
    json.dump(doc_id_to_desc,f)