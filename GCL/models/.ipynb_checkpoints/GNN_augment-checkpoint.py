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

import torch_geometric.transforms as T

class EdgeDropout(T.BaseTransform):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, edge_index):
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > self.p
        edge_index = edge_index[:, mask]
        return edge_index

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

def contrastive_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    pos_mask = torch.eye(batch_size, dtype=torch.float32).to(z1.device)
    neg_mask = 1 - pos_mask

    pos_score = torch.exp(torch.sum(z1 * z2, dim=1) / temperature)
    neg_score = torch.sum(torch.exp(torch.mm(z1, z2.t()) / temperature), dim=1) - pos_score

    loss = -torch.log(pos_score / neg_score).mean()
    return loss

class ModelRunner:
    def __init__(self, model_name, gnn_model, sample, feature, loss_func):
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
        self.loss_func = loss_func
        self.gnn_dir = self.gcl_data_dir + 'sample_{}/{}/{}'.format(self.sample, self.gnn_model, self.loss_func)
        self.aug = "augment4"
        

    def load_data(self):
        if(self.feature_name=="text"):
            self.feature_2 = np.load(self.text_emb_dir + 'text_embeddings.npy')
            print("Features :",self.text_emb_dir + 'text_embeddings.npy' )
            self.out_file = '/text_{}.npy'.format(self.aug)
        elif(self.feature_name=="text_hop"):
            self.feature_2 = np.load(self.text_emb_dir + 'text_hop_embeddings.npy')
            print("Features :",self.text_emb_dir + 'text_hop_embeddings.npy' )
            self.out_file = '/text_hop_{}.npy'.format(self.aug)
        else:
            self.feature_2 = np.load(self.embeddings_dir+'{}.npy'.format(self.feature_name))
            self.out_file = '/{}_{}.npy'.format(self.feature_name,self.aug)
            print("Features :",self.embeddings_dir+'{}.npy'.format(self.feature_name))
            print(self.feature_2.shape)
        
        # with open(self.data_dir + 'doc_id_to_emb_id.json') as f:
        #     self.doc_id_to_emb_id = json.load(f)
        # with open(self.data_dir + 'emb_id_to_doc_id.json') as f:
        #     self.emb_id_to_doc_id = json.load(f)

        with open(self.data_dir + 'graph_edges.json') as fp:
            self.edges_json = json.load(fp)
        with open(self.gcl_data_dir + 'graph_edges_sample_{}.json'.format(self.sample)) as fp:
            self.edges_json1 = json.load(fp)
        with open(self.data_dir + 'attack_weak_range.json') as fp:
            self.attack_weak_range = json.load(fp)

        if not os.path.exists(self.gnn_dir):
            os.makedirs(self.gnn_dir)

        self.node_list = list(range(0, self.attack_weak_range['n_nodes']))
        self.edge_list = [(int(e[0]), int(e[1])) for e in self.edges_json]
        self.full_edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous().to(self.device)

        self.edge_list1 = [(int(e[0]), int(e[1])) for e in self.edges_json1]
        self.full_edge_index1 = torch.tensor(self.edge_list1, dtype=torch.long).t().contiguous().to(self.device)
        
        self.node_feature_2 = torch.tensor(self.feature_2, dtype=torch.float).to(self.device)

            

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
        augment = EdgeDropout(p=0.2)
        print(model)
        # if(self.loss_func=="triplet"):
        #     contrastive_loss1 = contrastive_loss(margin=self.margin1).to(self.device)
        # elif(self.loss_func=="JSD"):
        #     contrastive_loss1 = JSDContrastiveLoss(margin=self.margin1).to(self.device)
        # elif(self.loss_func=="infoNCE"):
        #     contrastive_loss1 = InfoNCELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = config.GCL_EPOCH
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Augment1
            # augmented_data1 = augment(self.full_edge_index1)
            # augmented_data2 = augment(self.full_edge_index1)
            # z1 = model(self.node_feature_2,augmented_data1)
            # z2 = model(self.node_feature_2,augmented_data2)
            # Augment2
            # z1 = model(self.node_feature_2,self.full_edge_index)
            # z2 = model(self.node_feature_2,self.full_edge_index1)

            # Augment3
            # augmented_data1 = augment(self.full_edge_index)
            # augmented_data2 = augment(self.full_edge_index)
            # z1 = model(self.node_feature_2,augmented_data1)
            # z2 = model(self.node_feature_2,augmented_data2)

            # Augment4
            augmented_data1 = augment(self.full_edge_index)
            augmented_data2 = augment(self.full_edge_index1)
            z1 = model(self.node_feature_2,augmented_data1)
            z2 = model(self.node_feature_2,augmented_data2)
        
            if self.loss_func == "triplet" or self.loss_func == "JSD":
                loss = contrastive_loss(z1, z2)

            elif self.loss_func == "infoNCE":
                loss = contrastive_loss(z1, z2)
        
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

def main(model_name, gnn_model, sample, feature, loss_func):
    runner = ModelRunner(model_name, gnn_model, sample, feature, loss_func)
    runner.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified model_name and gnn_model")
    parser.add_argument("--model_name", type=str, required=True, choices=["pt_SecRoBERTa", "SecRoBERTa", "pt_SecureBERT", "SecureBERT", "pt_gpt2-xl", "gpt2-xl"], help="Model Name")
    parser.add_argument("--gnn_model", type=str, required=True, choices=["GCN", "GAT"], help="GNN Model")
    parser.add_argument("--sample", type=int, required=True, help="Sample number")
    parser.add_argument("--feature", type=str, required=True, choices=["text", "text_hop", "node2vec", "deepwalk"], help="feature")
    parser.add_argument("--loss", type=str, required=True, choices=["infoNCE", "triplet", "JSD", "BYOL"], help="Loss")
    args = parser.parse_args()

    main(args.model_name, args.gnn_model, args.sample, args.feature, args.loss)