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
import random
sys.path.append("../../")
import config

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

        # Compute the triplet loss
        loss = torch.relu(pos_dist - neg_dist + self.margin).mean()
        return loss




def kl_divergence(p, q):
    return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)), dim=1)

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))

class JSDContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(JSDContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize the embeddings to unit vectors
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negative_norm = F.normalize(negative, p=2, dim=1)

        # Convert embeddings to probability distributions
        anchor_dist = F.log_softmax(anchor_norm, dim=1).exp()
        positive_dist = F.log_softmax(positive_norm, dim=1).exp()
        negative_dist = F.log_softmax(negative_norm, dim=1).exp()

        # Compute JSD between anchor-positive and anchor-negative
        jsd_pos = jensen_shannon_divergence(anchor_dist, positive_dist)
        jsd_neg = jensen_shannon_divergence(anchor_dist, negative_dist)

        # Compute the contrastive loss using JSD
        loss = torch.relu(jsd_pos - jsd_neg).mean()
        return loss
# class InfoNCELoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super(InfoNCELoss, self).__init__()
#         self.temperature = temperature

#     def forward(self, anchor, positive, negatives):
#         # print(anchor.shape)
#         # print(positive.shape)
#         # print(negatives.shape)
#         # Normalize the embeddings to unit vectors
#         anchor_norm = F.normalize(anchor, p=2, dim=1)
#         positive_norm = F.normalize(positive, p=2, dim=1)
#         negatives_norm = F.normalize(negatives, p=2, dim=2)

#         # Compute the positive logit
#         positive_logit = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature

#         # Compute the negative logits
#         negative_logits = torch.bmm(anchor_norm.unsqueeze(1), negatives_norm.transpose(1, 2)).squeeze(1) / self.temperature

#         # Concatenate positive logit and negative logits
#         logits = torch.cat([positive_logit.unsqueeze(1), negative_logits], dim=1)

#         # Create labels: 0 for the positive sample
#         labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)

#         # Compute loss
#         loss = F.cross_entropy(logits, labels)

#         return loss   

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor_emb, positive_emb, negative_nodes_emb):
        # anchor_emb = [batch_size, emb_size]
        # positive_emb = [batch_size, emb_size]
        # negative_nodes_emb = [batch_size, n_negatives, emb_size]

        batch_size, emb_size = anchor_emb.shape
        n_negatives = negative_nodes_emb.shape[1]

        # Calculate similarities
        pos_sim = F.cosine_similarity(anchor_emb, positive_emb) / self.temperature
        pos_sim = pos_sim.unsqueeze(-1)  # [batch_size, 1]
        
        neg_sim = torch.bmm(negative_nodes_emb, anchor_emb.unsqueeze(2)).squeeze(2) / self.temperature  # [batch_size, n_negatives]

        # Concatenate positive and negative similarities
        logits = torch.cat((pos_sim, neg_sim), dim=1)  # [batch_size, 1 + n_negatives]
        
        # Labels: first element is the positive example
        labels = torch.zeros(batch_size, dtype=torch.long).to(anchor_emb.device)

        # Cross-Entropy Loss
        loss = F.cross_entropy(logits, labels)

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

        self.node_list = list(range(0, self.attack_weak_range['n_nodes']))
        self.edge_list = [(int(e[0]), int(e[1])) for e in self.edges_json]
        self.full_edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous().to(self.device)
        
        self.node_feature_2 = torch.tensor(self.feature_2, dtype=torch.float).to(self.device)

        self.anchor_nodes = []
        self.positive_nodes = []
        self.negative_nodes = []
        # self.negative_nodes_infoNCE = []
        # self.window_size = 10  # number of negative samples per positive sample
        # random.shuffle(self.anchor_pos_neg_triple)
        for a, p, n in self.anchor_pos_neg_triple:
            self.anchor_nodes.append(a)
            self.positive_nodes.append(p)
            self.negative_nodes.append(n)
            
            # # Sliding window to create negative samples for infoNCE
            # start_index = max(0, n - self.window_size // 2)
            # end_index = min(len(self.node_list), n + self.window_size // 2 + 1)
            # neg_samples = self.node_list[start_index:end_index]
            
            # if len(neg_samples) < self.window_size:
            #     neg_samples.extend(self.node_list[:self.window_size - len(neg_samples)])
            
            # self.negative_nodes_infoNCE.append(neg_samples[:self.window_size])
            

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
        if(self.loss_func=="triplet"):
            contrastive_loss1 = ContrastiveLoss(margin=self.margin1).to(self.device)
        elif(self.loss_func=="JSD"):
            contrastive_loss1 = JSDContrastiveLoss(margin=self.margin1).to(self.device)
        elif(self.loss_func=="infoNCE"):
            contrastive_loss1 = InfoNCELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = config.GCL_EPOCH
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            gnn_model = model(self.node_feature_2,self.full_edge_index)
            anchor_output = gnn_model[self.anchor_nodes]
            positive_output = gnn_model[self.positive_nodes]
            if self.loss_func == "triplet" or self.loss_func == "JSD":
                negative_output = gnn_model[self.negative_nodes]
                loss = contrastive_loss1(anchor_output, positive_output, negative_output)

            elif self.loss_func == "infoNCE":
                # negative_output = gnn_model[self.negative_nodes_infoNCE].reshape(len(self.anchor_nodes), self.window_size, -1)
                negative_output = []
                for neg_nodes in self.negative_nodes:
                    x = gnn_model[neg_nodes]
                    negative_output.append(x)
                negative_output = torch.stack(negative_output)
                loss = contrastive_loss1(anchor_output, positive_output,negative_output )
        
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