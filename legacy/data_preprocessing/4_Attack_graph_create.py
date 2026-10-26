import json
import sys
import os
# Add project root to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.insert(0, project_root)
import config

# Load JSON file
data_dir = config.DATA_DIR
print("data dir: ",data_dir)
with open(data_dir+'attack.json', encoding='utf-8') as f:
    data = json.load(f)

data_keys = data.keys()
print(data_keys)

print("Number of objects: ", len(data['objects']))

from collections import Counter

# Get the list of objects
objects = data['objects']

# Get the 'type' of each object
types = [obj['type'] for obj in objects]

# Count the occurrences of each type
type_counts = Counter(types)

# print(type_counts)
# Get the maximum length of the type names
max_length = max(len(type) for type in type_counts.keys())

# Print the counts in two columns
for type, count in type_counts.items():
    print(f'{type.ljust(max_length)}: {count}')

def getObjectsOfSpecificType(objects, theType='relationship'):
    objOfThisType=[]
    for i in range(0, len(objects)):
        if 'type' in objects[i]:
            if objects[i]['type']==theType:
                objOfThisType.append(objects[i])
    return objOfThisType

def idVSobjects(objects):
    idVSObjects={}
    for i in range(0, len(objects)):
        if 'id' in objects[i]:
            idVSObjects[objects[i]['id']]=objects[i]
    return idVSObjects

relationObjects = getObjectsOfSpecificType(objects, 'relationship')
idVsObj=idVSobjects(objects)


from sklearn.metrics.pairwise import cosine_similarity

# construct nodes and links
nodes = []
links = []
count=0
# add all unique source and target refs as nodes
for relation in relationObjects:
    source_ref = relation['source_ref']
    target_ref = relation['target_ref']
    if(source_ref not in idVsObj.keys() or target_ref not in idVsObj.keys()):
        count+=1
        continue
    if source_ref not in nodes:
        nodes.append(source_ref)
    if target_ref not in nodes:
        nodes.append(target_ref)
    links.append({'source': source_ref, 'target': target_ref})

# write graph.json file
print("missing relationship: ", count)
for i in range (0, len(nodes)):
    nodeID=nodes[i]
    
    nodes[i]=idVsObj[nodeID]
    #print(nodes[i])


#graph = {'nodes': [{'id': node, 'group': 1} for node in nodes], 'links': links}
graph = {'nodes': nodes, 'links': links}

with open(data_dir+'attack_graph.json', 'w') as f:
    json.dump(graph, f)