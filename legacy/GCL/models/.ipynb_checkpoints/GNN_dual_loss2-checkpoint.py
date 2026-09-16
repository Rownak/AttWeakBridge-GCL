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
import time
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

class DualEncoder(nn.Module):
    def __init__(self, graph_model1, graph_model2, text_dim, feature_2_dim,out_channels, hidden_dim):
        super(DualEncoder, self).__init__()
        self.graph_model1 = graph_model1
        self.graph_model2 = graph_model2
        self.fc1 = nn.Linear(text_dim, out_channels)
        self.fc2 = nn.Linear(feature_2_dim, feature_2_dim)
        self.fc3 = nn.Linear(2*out_channels+2*feature_2_dim, hidden_dim)
        # print("fc2:", 4*out_channels,"x", hidden_dim)

    def forward(self, text_emb, feature_2, edge_index):
        graph_emb1 = self.graph_model1(text_emb, edge_index)
        graph_emb2 = self.graph_model2(feature_2, edge_index)
        # print("graph_emb1: ", graph_emb1.shape)
        # print("graph_emb2: ", graph_emb2.shape)
        text_emb = self.fc1(text_emb)
        # print("text shape: ",text_emb.shape)
        feature_2 = self.fc2(feature_2)
        # print("feat shape: ",feature_2.shape)
        combined_emb = torch.cat([text_emb,graph_emb1,feature_2, graph_emb2], dim=1)
        # print("concat shape: ",combined_emb.shape)
        combined_emb = self.fc3(combined_emb)
        return combined_emb

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
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        # print(anchor.shape)
        # print(positive.shape)
        # print(negatives.shape)
        # Normalize the embeddings to unit vectors
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negatives_norm = F.normalize(negatives, p=2, dim=2)

        # Compute the positive logit
        positive_logit = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature

        # Compute the negative logits
        negative_logits = torch.bmm(anchor_norm.unsqueeze(1), negatives_norm.transpose(1, 2)).squeeze(1) / self.temperature

        # Concatenate positive logit and negative logits
        logits = torch.cat([positive_logit.unsqueeze(1), negative_logits], dim=1)

        # Create labels: 0 for the positive sample
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        return loss   

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
        self.text_embeddings = np.load(self.text_emb_dir + 'text_embeddings.npy')
        if(self.feature_name=="text_hop"):
            self.feature_2 = np.load(self.text_emb_dir + 'text_hop_embeddings.npy')
            print("Features :",self.text_emb_dir + 'text_hop_embeddings.npy' )
            self.out_file = '/text_hop_dual3_gm_{}.npy'.format(self.margin1)
        else:
            self.feature_2 = np.load(self.embeddings_dir+'{}.npy'.format(self.feature_name))
            self.out_file = '/text_{}_dual3_gm_{}.npy'.format(self.feature_name,self.margin1)
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
        self.text_positive_nodes = []
        self.text_negative_nodes = []
        self.graph_positive_nodes = []
        self.graph_negative_nodes = []
        self.node_list = list(range(0, self.attack_weak_range['n_nodes']))
        self.edge_list = [(int(e[0]), int(e[1])) for e in self.edges_json]
        self.full_edge_index = torch.tensor(self.edge_list, dtype=torch.long).t().contiguous().to(self.device)
        self.node_text_embeddings = torch.tensor(self.text_embeddings, dtype=torch.float).to(self.device)
        self.node_feature_2 = torch.tensor(self.feature_2, dtype=torch.float).to(self.device)
        for a,pt,nt,pg,ng in self.anchor_pos_neg_triple:
            self.anchor_nodes.append(a)
            self.text_positive_nodes.append(pt)
            self.text_negative_nodes.append(nt)
            self.graph_positive_nodes.append(pg)
            self.graph_negative_nodes.append(ng)
            

    def run(self):
        print(f'Using device: {self.device}')
        self.load_data()
        print("Model: ", self.model_name)
        print("Sample: ", self.sample)
        print("Gnn Model: ", self.gnn_model)
        in_channels = self.text_embeddings.shape[1]
        feature_2_dim = self.feature_2.shape[1]
        out_channels = 128
        hidden_dim = 128
        text_dim = in_channels

        if self.gnn_model == "GCN":
            graph_model1 = GCN(in_channels=in_channels, out_channels=out_channels, hid_dim=128).to(self.device)
            graph_model2 = GCN(in_channels=feature_2_dim, out_channels=feature_2_dim, hid_dim=feature_2_dim).to(self.device)
        else:
            graph_model1 = GAT(in_channels=in_channels, out_channels=out_channels, hid_dim=128).to(self.device)
            graph_model2 = GAT(in_channels=feature_2_dim, out_channels=feature_2_dim, hid_dim=feature_2_dim).to(self.device)

        print(graph_model2)
        model = DualEncoder(graph_model1, graph_model2, text_dim=text_dim, feature_2_dim=feature_2_dim,out_channels=out_channels, hidden_dim=hidden_dim).to(self.device)
        if(self.loss_func=="triplet"):
            contrastive_loss1 = ContrastiveLoss(margin=self.margin1).to(self.device)
        elif(self.loss_func=="JSD"):
            contrastive_loss1 = JSDContrastiveLoss(margin=self.margin1).to(self.device)
        elif(self.loss_func=="infoNCE"):
            contrastive_loss1 = InfoNCELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        num_epochs = config.GCL_EPOCH
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            gnn_model = model(self.node_text_embeddings,self.node_feature_2, self.full_edge_index)
            anchor_output = gnn_model[self.anchor_nodes]
            text_positive_output = gnn_model[self.text_positive_nodes]
            
            graph_positive_output = gnn_model[self.graph_positive_nodes]
            
            if self.loss_func == "triplet" or self.loss_func == "JSD":
                text_negative_output = gnn_model[self.text_negative_nodes]
                graph_negative_output = gnn_model[self.graph_negative_nodes]
                loss1 = contrastive_loss1(anchor_output, text_positive_output, text_negative_output)
                loss2 = contrastive_loss1(anchor_output, graph_positive_output, graph_negative_output)

            # elif self.loss_func == "infoNCE":
            #     # negative_output = gnn_model[self.negative_nodes_infoNCE].reshape(len(self.anchor_nodes), self.window_size, -1)
            #     text_negative_output = []
            #     for neg_nodes in self.text_negative_nodes:
            #         x = gnn_model[neg_nodes]
            #         text_negative_output.append(x)
            #     text_negative_output = torch.stack(text_negative_output)
            #     loss = contrastive_loss1(anchor_output, text_positive_output,text_negative_output )
            alpha = 0.5
            loss = alpha*loss1+(1-alpha)*loss2
            loss.backward()
            optimizer.step()
            #scheduler.step()
            if(epoch%20==0):
                print(f'Epoch {epoch},Loss1: {loss1.item()},Loss2: {loss2.item()} Loss: {loss.item()}')

        final_node_embeddings = self.extract_embeddings(model, self.node_text_embeddings, self.node_feature_2, self.full_edge_index)

        # Print the final embeddings
        print("final embeddings:", final_node_embeddings.shape)
        node_embeddings_np = final_node_embeddings.detach().cpu().numpy()
        np.save(self.gnn_dir + self.out_file, np.array(node_embeddings_np))
        torch.cuda.empty_cache()
        gc.collect()
    def extract_embeddings(self, model, node_embeddings, node_feature_2, edge_index):
        model.eval()
        with torch.no_grad():
            embeddings = model(node_embeddings, node_feature_2, edge_index)
        return embeddings

def main(model_name, gnn_model, sample, feature, loss_func):
    runner = ModelRunner(model_name, gnn_model, sample, feature, loss_func)
    runner.run()

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run the model with specified model_name and gnn_model")
    parser.add_argument("--model_name", type=str, required=True, choices=["pt_SecRoBERTa", "SecRoBERTa", "pt_SecureBERT", "SecureBERT", "pt_gpt2-xl", "gpt2-xl"], help="Model Name")
    parser.add_argument("--gnn_model", type=str, required=True, choices=["GCN", "GAT"], help="GNN Model")
    parser.add_argument("--sample", type=int, required=True,choices=[10,11], help="Only works on sample number 10,11")
    parser.add_argument("--feature", type=str, required=True, choices=["text_hop", "node2vec", "deepwalk"], help="feature")
    parser.add_argument("--loss", type=str, required=True, choices=["triplet"], help="Only works on triplet Loss")
    args = parser.parse_args()

    main(args.model_name, args.gnn_model, args.sample, args.feature, args.loss)
    print("--- %s seconds ---" % (time.time() - start_time))