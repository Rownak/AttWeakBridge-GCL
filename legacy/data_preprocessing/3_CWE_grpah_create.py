import json
import xml.etree.ElementTree as ET
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.insert(0, project_root)

import config

cwe_dir = config.CWE_DIR
with open(cwe_dir+config.CWE_DATASET+'_info.json') as fp:
    weakness_info = json.load(fp)

len(weakness_info)


# construct nodes and links
nodesDict = {}
links = []

e=0
#for e in range(0, 5):
count_link = 0
for key_id in weakness_info:
    entry=weakness_info[key_id]
    
    if "Related_Weaknesses" in entry:
        if isinstance(entry["Related_Weaknesses"], list):
            print(key_id," len of related: ", len(entry["Related_Weaknesses"]))
            count_link+=len(entry["Related_Weaknesses"])
            for j in range(0, len(entry["Related_Weaknesses"])):
                #print(j,": ", entry["Related_Weaknesses"])    
                if "Nature" in entry["Related_Weaknesses"][j]:
                    if entry["Related_Weaknesses"][j]["Nature"]=="ChildOf":
                        source_ref=entry["Related_Weaknesses"][j]["CWE_ID"]
                        links.append({'source': source_ref, 'target': key_id})
                        nodesDict[source_ref]=weakness_info[source_ref]
                        nodesDict[key_id]=weakness_info[key_id]
                        nodesDict[key_id]['ID']=key_id
        
        else:
            if entry["Related_Weaknesses"]["Nature"]=="ChildOf":
                source_ref=entry["Related_Weaknesses"]["CWE_ID"]
                nodesDict[source_ref]=weakness_info[source_ref]
                nodesDict[key_id]=weakness_info[key_id]
                nodesDict[key_id]['ID']=key_id
                links.append({'source': source_ref, 'target': key_id})
        
                
nodes= list(nodesDict.values())

graph = {'nodes': nodes, 'links': links}

with open(cwe_dir+config.CWE_DATASET+'_graph.json', 'w') as f:
    json.dump(graph, f)