import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
import torch.optim as optim
import random
import os
import sys
import math
base_dir = os.environ['AWEB_DIR']
sys.path.append(base_dir)
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def main(model_id):
    dataset_name = config.ATTACK_DATASET
    data_dir = config.DATA_DIR
    models = ["pt_SecRoBERTa", "SecRoBERTa", "pt_SecureBERT", "SecureBERT", "pt_gpt2-xl", "gpt2-xl"]
    no_epoch = config.LLM_FT_EPOCH
    model_name = models[model_id]
    
    text_emb_dir = os.path.join(config.EMBEDDING_DIR, model_name) 
    with open(os.path.join(data_dir, 'doc_id_to_emb_id.json')) as f:
        doc_id_to_emb_id = json.load(f)
    with open(os.path.join(data_dir, 'emb_id_to_doc_id.json')) as f:
        emb_id_to_doc_id = json.load(f)

    training_data = np.load(os.path.join(data_dir, 'hop_training_data.npy'))
    text_embeddings = np.load(os.path.join(text_emb_dir, 'text_embeddings.npy'))
    weights = [x for _, _, x in training_data]
    pairs = [[int(x), int(y)] for x, y, _ in training_data]

    def shuffle_two_arrays(array1, array2):
        combined = list(zip(array1, array2))
        random.shuffle(combined)
        shuffled_array1, shuffled_array2 = zip(*combined)
        return list(shuffled_array1), list(shuffled_array2)

    spairs, sweights = shuffle_two_arrays(pairs, weights)
    X_training = np.array(spairs)
    Y_training = np.array(sweights).reshape(-1, 1).squeeze()
    
    X1_training = X_training.T[0]
    X2_training = X_training.T[1]

    TOTAL_OBJECTS = len(doc_id_to_emb_id)
    EMBEDDING_DIM_1 = len(text_embeddings[0])
    HIDDEN_DIM = 256
    EMBEDDING_DIM_2 = 64
    training_epochs = 200
    learning_rate = 0.001
    SCALE_FACTOR = 1
    BATCH_SIZE = 2**(int(math.log2(len(X1_training)//3)))  # Define a batch size

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.text_embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(text_embeddings), freeze=True)
            self.hidden_layers = nn.Sequential(
                nn.Linear(EMBEDDING_DIM_1, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, EMBEDDING_DIM_2),
                nn.ReLU()
            )

        def forward(self, X1, X2):
            obj1_text_embedding = self.text_embedding_layer(X1)
            obj2_text_embedding = self.text_embedding_layer(X2)
            hidden_output = self.hidden_layers(self.text_embedding_layer.weight)
            obj1_model1_embedding = hidden_output[X1]
            obj2_model1_embedding = hidden_output[X2]
            obj1_model1_embedding_norm = F.normalize(obj1_model1_embedding, p=2, dim=1)
            obj2_model1_embedding_norm = F.normalize(obj2_model1_embedding, p=2, dim=1)
            obj1_text_embedding_norm = F.normalize(obj1_text_embedding, p=2, dim=1)
            obj2_text_embedding_norm = F.normalize(obj2_text_embedding, p=2, dim=1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            hop_dist_predict = SCALE_FACTOR * (1 - cos(obj1_model1_embedding_norm, obj2_model1_embedding_norm))
            text_dist_predict = SCALE_FACTOR * (1 - cos(obj1_text_embedding_norm, obj2_text_embedding_norm))
            return hop_dist_predict, text_dist_predict

    model = NeuralNetwork().to(device)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    X1 = torch.tensor(X1_training, dtype=torch.long).to(device)
    X2 = torch.tensor(X2_training, dtype=torch.long).to(device)
    Y = torch.tensor(Y_training, dtype=torch.float32).to(device)

    dataset = TensorDataset(X1, X2, Y)
    print("data size:", X1.shape)
    print("BATCH_SIZE: ",BATCH_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Num of Batches: ",  len(dataloader))
    for epoch in range(training_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        for batch_X1, batch_X2, batch_Y in dataloader:
            optimizer.zero_grad()
            hop_dist_predict, text_dist_predict = model(batch_X1, batch_X2)
            loss1 = criterion(batch_Y, hop_dist_predict)
            loss2 = criterion(hop_dist_predict, text_dist_predict)
            alpha = 0.5
            cost = (alpha) * loss1 + (1 - alpha) * loss2
            cost.backward()
            optimizer.step()
            epoch_loss += cost.item()
            batch_count+=1
        if epoch % 20 == 0:
            print(f"Epoch: {epoch} - Training Cost: {epoch_loss/batch_count}")

    embeddings1_val = model.cpu().text_embedding_layer.weight.data.numpy()
    embeddings2_val = model.cpu().hidden_layers(model.text_embedding_layer.weight).detach().numpy()
    np.save(os.path.join(text_emb_dir, "text_hop_embeddings.npy"), np.array(embeddings2_val), allow_pickle=True)
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the model with specified model_id")
    parser.add_argument("--model_id", type=int, required=True, help="Model ID")
    args = parser.parse_args()
    main(args.model_id)