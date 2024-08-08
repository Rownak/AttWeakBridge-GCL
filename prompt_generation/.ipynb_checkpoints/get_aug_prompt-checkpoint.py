import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import sys
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('../')
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model_name = "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).to(device)
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.eval()

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def generate_graph_emb(org_emb, encoder, decoder):
    checkpoint = torch.load(config.ENCODER_PATH)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    
    new_text_embeddings = torch.tensor(org_emb, dtype=torch.float32).to(device)
    with torch.no_grad():
        encoded_text = encoder(new_text_embeddings.to(device))
        generated_graph_embeddings = decoder(encoded_text)
    return generated_graph_embeddings.cpu().detach().numpy()  # Move tensor to CPU before converting to numpy

def retrieve_embeddings(input_text, chunk_size=110):
    tokens = tokenizer.tokenize(input_text)
    chunk_embeddings = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        input_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
        input_tensors = torch.tensor([input_ids]).to(device)
        with torch.no_grad():
            outputs = model(input_tensors, output_hidden_states=True)
        last_layer_embeddings = outputs.hidden_states[-1]
        mean_pooled = last_layer_embeddings.mean(dim=1)
        chunk_embeddings.append(mean_pooled)
    all_embeddings = torch.cat(chunk_embeddings, dim=0)
    return all_embeddings.mean(dim=0).squeeze()

def get_top_attack_weakness(prompt_graph_embeddings, graph_embeddings, top_k1, top_k2):
    attack_size = 203
    weakness_size = 933
    # Ensure prompt_graph_embeddings is a CPU tensor if it's used with NumPy
    # prompt_graph_embeddings = prompt_graph_embeddings.cpu().reshape(1, -1)  # Move to CPU and reshape
    print(prompt_graph_embeddings.shape)
    #graph_embeddings = graph_embeddings.cpu()  # Move to CPU if not already
    cos_sim_attack = cosine_similarity(prompt_graph_embeddings, graph_embeddings[:attack_size]).reshape(-1)
    print("Cos sim: ", cos_sim_attack.shape)
    cos_sim_weak = cosine_similarity(prompt_graph_embeddings, graph_embeddings[attack_size:]).reshape(-1)
    top_attack = np.argsort(cos_sim_attack)[::-1][:top_k1]
    top_weak = np.argsort(cos_sim_weak)[::-1][:top_k2]
    attack_pairs = list(zip(cos_sim_attack[top_attack], top_attack))
    weak_pairs = list(zip(cos_sim_weak[top_weak], top_weak + attack_size))
    return attack_pairs, weak_pairs
    return None,None

# def generate_prompt_context(prompt, related_attack, related_weakness, doc_id_to_desc, emb_id_to_doc_id):
#     attack_text = " ".join([doc_id_to_desc[emb_id_to_doc_id[str(pos)]] for _, pos in related_attack])
#     weakness_text = " ".join([doc_id_to_desc[emb_id_to_doc_id[str(pos)]] for _, pos in related_weakness])
#     return f"{prompt}\nRelated Att@ck Description:\n{attack_text}\nRelated Weakness Description:\n{weakness_text}"

def generate_prompt_context(prompt, related_attack, related_weakness, doc_id_to_desc, emb_id_to_doc_id, emb_id_to_tid):
    attack_text = "\n\n".join([f"ATTACK ID: {emb_id_to_tid[str(pos)]}: {doc_id_to_desc[emb_id_to_doc_id[str(pos)]]}" for _, pos in related_attack])
    weakness_text = "\n\n".join([f"CWE ID: {emb_id_to_doc_id[str(pos)]}: {doc_id_to_desc[emb_id_to_doc_id[str(pos)]]}" for _, pos in related_weakness])
    return f"Given Prompt: {prompt}\n\n\nRelated Att@ck Description:\n\n{attack_text}\n\n\nRelated Weakness Description:\n\n{weakness_text}"


def main(prompt):
    prompt_embeddings = retrieve_embeddings(prompt)
    print(prompt_embeddings.shape)
    graph_embeddings = np.load(config.OUTPUT_DIR + "gcl_data/pt_gpt2-xl/sample_10/GAT/triplet/text_deepwalk_dual3_gm_1.0.npy")
    print(graph_embeddings.shape)

    text_embedding_dim = prompt_embeddings.shape[0]  # Example text embedding dimension
    hidden_dim = 128
    encoder = Encoder(text_embedding_dim, hidden_dim).to(device)
    decoder = Decoder(hidden_dim, 128).to(device)
    prompt_graph_embeddings = generate_graph_emb(prompt_embeddings,encoder,decoder)
    # print("prompt_graph_embeddings shape:",prompt_graph_embeddings.shape)
    related_attack, related_weakness = get_top_attack_weakness(prompt_graph_embeddings.reshape(1, -1), graph_embeddings, 10, 10)
    with open(config.DESCRIPTION_FILE) as fp:
        doc_id_to_desc = json.load(fp)
    with open(config.DATA_DIR + 'emb_id_to_doc_id.json') as fp:
        emb_id_to_doc_id = json.load(fp)
    with open(config.DATA_DIR + 'emb_id_to_tid.json') as fp:
        emb_id_to_tid = json.load(fp)
    augmented_prompt = generate_prompt_context(prompt, related_attack, related_weakness, doc_id_to_desc, emb_id_to_doc_id,emb_id_to_tid)
    return augmented_prompt

if __name__ == "__main__":
    prompt = sys.argv[1]  # Taking prompt as command-line argument
    print(main(prompt))