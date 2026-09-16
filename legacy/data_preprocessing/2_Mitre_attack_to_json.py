import json
import os
import numpy as np
from collections import Counter
from collections import defaultdict
import sys
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.insert(0, project_root)
import config

data_dir = config.DATA_DIR
print("data dir: ", data_dir)
with open(data_dir+'attack.json', encoding='utf-8') as f:
    data = json.load(f)

# Get the list of objects
objects = data['objects']
# Extract Attack information
attack_info = defaultdict(list)

count=0
count2=0
for i in range(0, len(objects)):
    obj=objects[i]
    i=i+1

    description=""
    name=""
    x_mitre_detection = ""
    t_id = ""
    if 'name' in obj:
        if(len(obj['name'])>0):
            name = obj['name']
    if 'description' in obj:
        if len(obj['description'])>0:
            try:
                description = obj['description']
                count=count+1
            except Exception as e:
                print("Exception:", e)        
                print("i", i)
                print("Len:", len(obj['description']))
                print(obj['description'])
                break
    if 'x_mitre_detection' in obj:
        if len(obj['x_mitre_detection'])>0:
            try:
                x_mitre_detection = obj['x_mitre_detection']
                count2=count2+1
            except Exception as e:
                print("Exception:", e)        
                print("i", i)
                print("Len:", len(obj['x_mitre_detection']))
                print(obj['x_mitre_detection'])
                break
    if 'external_references' in obj:
        if len(obj['external_references'])>0:
            if 'external_id' in obj['external_references'][0]:
                try:
                    t_id = obj['external_references'][0]["external_id"]
                    count2=count2+1
                except Exception as e:
                    print("Exception:", e)        
                    print("i", i)
                    print("Len:", len(obj['external_references']))
                    print(obj['external_references'])
                    break
        
    if 'id' in obj:
        attack_info[obj['id']] = {
            'Name': name,
            'Description': description.strip(),
            'x_mitre_detection': x_mitre_detection.strip(),
            't_id':t_id
        }
        
    if (i%50==0):
        print("Procesed", i,"th object: id:", obj['id'])   

    #if (i==2):
    #    break
print(count, "objects have valid text descriptions.")
print(count2, "objects have valid text x_mitre_detection.")

with open(data_dir+'attack_info.json', 'w') as f:
    json.dump(attack_info,f)