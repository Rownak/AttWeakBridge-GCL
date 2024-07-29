import os
import json
import numpy as np
import torch
import networkx as nx
from gensim.models import Word2Vec
from torch_geometric.data import Data
from node2vec import Node2Vec
import os
import sys
base_dir = os.environ['AWEB_DIR']
sys.path.append(base_dir)
import config


# Set CUDA device and ensure the correct device is used
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Paths
data_dir = config.DATA_DIR
embeddings_dir = config.EMBEDDING_DIR

# Load data
def load_data(data_dir):
    with open(os.path.join(data_dir, 'attack_weak_range.json')) as fp:
        attack_weak_range = json.load(fp)
    with open(os.path.join(data_dir, 'graph_edges.json')) as fp:
        edges_json = json.load(fp)
    return attack_weak_range, edges_json

# Generate random walks for DeepWalk
def generate_random_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        np.random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if neighbors:
                    next_node = np.random.choice(neighbors)
                    walk.append(next_node)
                else:
                    break
            walks.append(walk)
    return walks

# Create and save node2vec embeddings
def create_node2vec_embeddings(G, dimensions, walk_length, num_walks, window, min_count, batch_words, embeddings_dir):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=4)
    model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
    embeddings = np.array([model.wv[str(n)] for n in G.nodes()])
    np.save(os.path.join(embeddings_dir, 'node2vec.npy'), embeddings)

# Create and save DeepWalk embeddings
def create_deepwalk_embeddings(G, num_walks, walk_length, vector_size, window, min_count, sg, workers, epochs, embeddings_dir):
    walks = generate_random_walks(G, num_walks, walk_length)
    walks = [[str(node) for node in walk] for walk in walks]
    model = Word2Vec(sentences=walks, vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers, epochs=epochs)
    node_ids = list(G.nodes())
    embeddings_array = np.array([model.wv[str(node)] for node in node_ids])
    np.save(os.path.join(embeddings_dir, 'deepwalk.npy'), embeddings_array)

def main():
    # Load data
    attack_weak_range, edges_json = load_data(data_dir)

    attack_range = attack_weak_range['attack']
    weak_range = attack_weak_range['cwe']
    n_nodes = attack_weak_range['n_nodes']
    node_list = list(range(0, n_nodes))
    edge_list = [(int(e[0]), int(e[1])) for e in edges_json]

    # Create graph
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Create and save node2vec embeddings
    create_node2vec_embeddings(G, dimensions=64, walk_length=30, num_walks=100, window=10, min_count=1, batch_words=4, embeddings_dir=embeddings_dir)

    # Create and save DeepWalk embeddings
    create_deepwalk_embeddings(G, num_walks=200, walk_length=30, vector_size=256, window=5, min_count=0, sg=1, workers=4, epochs=10,embeddings_dir=embeddings_dir)

if __name__ == "__main__":
    main()