import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import pickle
import argparse
import gc
import sys
import math

sys.path.append("../../")
import config

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, hid_dim):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hid_dim, heads=8)
        self.gat2 = GATConv(hid_dim * 8, out_channels, heads=1)
    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize the embeddings to unit vectors
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negative_norm = F.normalize(negative, p=2, dim=1)
        # Compute the cosine similarity
        cos_sim1 = F.cosine_similarity(anchor_norm, positive_norm, dim=1)
        cos_sim2 = F.cosine_similarity(anchor_norm, negative_norm, dim=1)
        # Cosine distance is 1 - cosine similarity
        pos_dist = 1 - cos_sim1
        neg_dist = 1 - cos_sim2

        # Compute the loss
        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()
        return loss
        
class ModelRunner:
    def __init__(self, model_name, gnn_model, sample, feature):
        self.gnn_model = gnn_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = ["pt_SecRoBERTa","SecRoBERTa","pt_SecureBERT","SecureBERT","pt_gpt2-xl","gpt2-xl"]
        self.model_name = model_name
        self.feature_name = feature
        self.result_dir = config.OUTPUT_DIR
        self.embeddings_dir = config.EMBEDDING_DIR
        self.data_dir = config.DATA_DIR
        self.text_emb_dir = self.embeddings_dir + self.model_name + "/"
        self.gcl_data_dir = self.result_dir + "gcl_data/" + self.model_name + "/"
        self.sample = sample
        self.margin1 = 1.0
        self.gnn_dir = self.gcl_data_dir + 'sample_{}/{}/'.format(self.sample, self.gnn_model)
        

    def load_data(self):
        if(self.feature_name=="text"):
            self.feature_2 = np.load(self.text_emb_dir + 'text_embeddings.npy')
            print("Features :",self.text_emb_dir + 'text_hop_embeddings.npy' )
            self.out_file = '/text_gm_{}.npy'.format(self.margin1)
        elif(self.feature_name=="text_hop"):
            self.feature_2 = np.load(self.text_emb_dir + 'text_hop_embeddings.npy')
            print("Features :",self.text_emb_dir + 'text_hop_embeddings.npy' )
            self.out_file = '/text_hop_gm_{}.npy'.format(self.margin1)
        else:
            self.feature_2 = np.load(self.embeddings_dir+'{}.npy'.format(self.feature_name))
            self.out_file = '/{}_gm_{}.npy'.format(self.feature_name,self.margin1)
            print("Features :",self.embeddings_dir+'{}.npy'.format(self.feature_name))
            print(self.feature_2.shape)
        
        # with open(self.data_dir + 'doc_id_to_emb_id.json') as f:
        #     self.doc_id_to_emb_id = json.load(f)
        # with open(self.data_dir + 'emb_id_to_doc_id.json') as f:
        #     self.emb_id_to_doc_id = json.load(f)
        with open(self.data_dir + 'graph_edges.json') as fp:
            self.edges_json = json.load(fp)
        with open(self.data_dir + 'attack_weak_range.json') as fp:
            self.attack_weak_range = json.load(fp)

        with open(self.gcl_data_dir + 'anchor_pos_neg_triple_{}.pkl'.format(self.sample), 'rb') as f:
            self.anchor_pos_neg_triple = pickle.load(f)
        if not os.path.exists(self.gnn_dir):
            os.makedirs(self.gnn_dir)
        self.anchor_nodes = []
        self.positive_nodes = []
        self.negative_nodes = []
        self.node_list = list(range(0, self.attack_weak_range['n_nodes']))
        self.edge_list = [(int(e[0]), int(e[1])) for e in self.edges_json]
        self.full_edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous().to(self.device)
        
        self.node_feature_2 = torch.tensor(self.feature_2, dtype=torch.float).to(self.device)
        for a,p,n in self.anchor_pos_neg_triple:
            self.anchor_nodes.append(a)
            self.positive_nodes.append(p)
            self.negative_nodes.append(n)
            

    def run(self):
        print(f'Using device: {self.device}')
        self.load_data()
        print("Model: ", self.model_name)
        print("Sample: ", self.sample)
        print("Gnn Model: ", self.gnn_model)
        in_channels = self.feature_2.shape[1]
        out_channels = 128
        hidden_dim = 128

        if self.gnn_model == "GCN":
            model = GCN(in_channels=in_channels, out_channels=out_channels, hid_dim=128).to(self.device)
        else:
            model = GAT(in_channels=in_channels, out_channels=out_channels, hid_dim=128).to(self.device)

        print(model)
        contrastive_loss1 = ContrastiveLoss(margin=self.margin1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = config.GCL_EPOCH
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            gnn_model = model(self.node_feature_2,self.full_edge_index)
            anchor_output = gnn_model[self.anchor_nodes]
            positive_output = gnn_model[self.positive_nodes]
            negative_output = gnn_model[self.negative_nodes]
            loss = contrastive_loss1(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()
            #scheduler.step()
            if(epoch%20==0):
                print(f'Epoch {epoch}, Loss: {loss.item()}')

        final_node_embeddings = self.extract_embeddings(model, self.node_feature_2, self.full_edge_index)

        # Print the final embeddings
        print("final embeddings:", final_node_embeddings.shape)
        node_embeddings_np = final_node_embeddings.detach().cpu().numpy()
        np.save(self.gnn_dir + self.out_file, np.array(node_embeddings_np))
        torch.cuda.empty_cache()
        gc.collect()
    def extract_embeddings(self, model, node_feature_2, edge_index):
        model.eval()
        with torch.no_grad():
            embeddings = model(node_feature_2, edge_index)
        return embeddings

def main(model_name, gnn_model, sample, feature):
    runner = ModelRunner(model_name, gnn_model, sample, feature)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified model_name and gnn_model")
    parser.add_argument("--model_name", type=str, required=True, choices=["pt_SecRoBERTa", "SecRoBERTa", "pt_SecureBERT", "SecureBERT", "pt_gpt2-xl", "gpt2-xl"], help="Model Name")
    parser.add_argument("--gnn_model", type=str, required=True, choices=["GCN", "GAT"], help="GNN Model")
    parser.add_argument("--sample", type=int, required=True, help="Sample number")
    parser.add_argument("--feature", type=str, required=True, choices=["text", "text_hop", "node2vec", "deepwalk"], help="feature")
    args = parser.parse_args()

    main(args.model_name, args.gnn_model, args.sample, args.feature)