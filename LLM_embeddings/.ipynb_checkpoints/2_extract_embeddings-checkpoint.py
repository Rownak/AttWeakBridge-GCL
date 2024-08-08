import torch
import os
import json
import numpy as np
import argparse
from transformers import RobertaTokenizer, RobertaForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
import sys
base_dir = os.environ['AWEB_DIR']
sys.path.append(base_dir)
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def getModel(model_name, isPretrained, model_path):
    if not isPretrained:
        if model_name == "gpt2-xl" or model_name == "gpt2":
            print("Load finetuned GPT2 model:")
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        elif model_name == "SecureBERT":
            print("Load finetuned SecureBERT model:")
            model = RobertaForMaskedLM.from_pretrained(model_path)
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        elif model_name == "SecRoBERTa":
            print("Load finetuned SecRoBERTa model:")
            model = AutoModelForMaskedLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        if model_name == "pt_SecureBERT":
            print("Load pretrained_SecureBert model:")
            tokenizer = RobertaTokenizer.from_pretrained("ehsanaghaei/SecureBERT")
            model = RobertaForMaskedLM.from_pretrained("ehsanaghaei/SecureBERT")
        elif model_name == "pt_SecRoBERTa":
            print("Load pretrained_SecBert model:")
            tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecRoBERTa")
            model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecRoBERTa")
        elif model_name == "pt_gpt2-xl":
            print("Load pretrained gpt2 model:")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
            model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        elif model_name == "pt_gpt2":
            print("Load pretrained gpt2 model:")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
        elif model_name == "pt_bert":
            print("Load pretrained BERT model:")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

    
def retrieveEmbeddings(input_text, tokenizer, model,model_name, device):
    # Tokenize the input text
    tokens = tokenizer.tokenize(input_text)
    
    # Initialize a list to store embeddings from each chunk
    chunk_embeddings = []
    
    # Process text in chunks that fit within the model's limit
    if(model_name=="pt_bert"):
        chunk_size = 512
    else:
        chunk_size = model.config.n_positions  # GPT-2's max sequence length
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        input_ids = tokenizer.convert_tokens_to_ids(chunk_tokens)
        input_tensors = torch.tensor([input_ids]).to(device)
        
        # Forward pass, get hidden states for the chunk
        with torch.no_grad():
            outputs = model(input_tensors, output_hidden_states=True)
        
        # Extract the hidden states
        hidden_states = outputs.hidden_states
        last_layer_embeddings = hidden_states[-1]
        
        # Mean pool the embeddings of the last layer across the sequence length dimension
        mean_pooled = last_layer_embeddings.mean(dim=1)
        chunk_embeddings.append(mean_pooled)
    # Concatenate embeddings from all chunks along the batch dimension
    # and then take the mean across the concatenated dimension to get a single embedding
    all_embeddings = torch.cat(chunk_embeddings, dim=0)
    aggregated_embedding = all_embeddings.mean(dim=0)
    
    return aggregated_embedding.squeeze()


def main(model_id):
    dataset_name = config.ATTACK_DATASET
    data_dir = config.DATA_DIR
    #model_output_dir = os.path.join(base_dir, "model_outputs",dataset_name,"llm_finetuned_models" )
    model_output_dir = config.OUTPUT_DIR+"llm_finetuned_models"
    models = ["pt_SecRoBERTa", "SecRoBERTa", "pt_SecureBERT", "SecureBERT", "pt_gpt2-xl", "gpt2-xl", "pt_bert"]
    model_name = models[model_id]
    no_epoch = config.LLM_FT_EPOCH
    isPretrained = (model_name.split("_")[0] == "pt")
    model_path = os.path.join(model_output_dir, model_name, "epoch_{}".format(no_epoch))
    embeddings_path = config.EMBEDDING_DIR+model_name
    print(data_dir)
    print(model_output_dir)
    print(embeddings_path)
    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)

    with open(config.DESCRIPTION_FILE) as f:
        doc_id_to_desc = json.load(f)
    with open(os.path.join(data_dir, 'doc_id_to_emb_id.json')) as f:
        doc_id_to_emb_id = json.load(f)
    with open(os.path.join(data_dir, 'emb_id_to_doc_id.json')) as f:
        emb_id_to_doc_id = json.load(f)

    model, tokenizer = getModel(model_name, isPretrained, model_path)
    text_embeddings = [None for _ in range(len(emb_id_to_doc_id))]
    
    count = 0
    i = 0
    for doc_id in doc_id_to_desc:
        text_data = doc_id_to_desc[doc_id]
        try:
            embedding = retrieveEmbeddings(text_data, tokenizer, model,model_name, device)
            count += 1
        except Exception as e:
            print("Exception:", e)
            print("i", i)
            print("Len:", len(text_data))
            print(text_data)
            break
        
        text_embeddings[int(doc_id_to_emb_id[doc_id])] = embedding.detach().cpu().numpy()
        
        if i % 100 == 0:
            print("Processed", i, "th object: id:", doc_id)
        i += 1

    print("Processing of", len(text_embeddings), "objects complete.")
    print(count, "objects have valid text descriptions.")
    text_embeddings_np = np.array(text_embeddings)
    np.save(os.path.join(embeddings_path, "text_embeddings.npy"), text_embeddings_np)
    torch.cuda.empty_cache()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with specified model_id")
    parser.add_argument("--model_id", type=int, required=True, help="Model ID")
    args = parser.parse_args()
    main(args.model_id)