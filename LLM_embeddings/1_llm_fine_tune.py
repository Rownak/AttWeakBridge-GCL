import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, DataCollatorWithPadding, TrainingArguments, Trainer
import sys
base_dir = os.environ['AWEB_DIR']
sys.path.append(base_dir)
import config


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Get the BASE_DIR environment variable
data_dir = config.DATA_DIR
print("data_dir: ", data_dir)
output_dir = config.OUTPUT_DIR
print("output_dir: ", output_dir)
llm_dir = output_dir+"llm_finetuned_models/"
print("llm_dir: ", llm_dir)
models = ["jackaduma/SecRoBERTa", "ehsanaghaei/SecureBERT", "gpt2-xl", "gpt2"]
model_names = ["SecRoBERTa", "SecureBERT", "gpt2-xl", "gpt2"]
n_epoch = config.LLM_FT_EPOCH

class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for text in texts:
            tokenized_text = tokenizer.encode(text)
            for i in range(0, len(tokenized_text), max_length):
                chunk = tokenized_text[i:i + max_length]
                self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        chunk = self.examples[idx]
        tokenized_inputs = self.tokenizer.prepare_for_model(
            chunk,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        for key in tokenized_inputs:
            tokenized_inputs[key] = tokenized_inputs[key].squeeze(0).to(device)
        return tokenized_inputs

def main(model_id):
    if not os.path.exists(llm_dir+model_names[model_id]):
        os.makedirs(llm_dir+model_names[model_id])

    with open(config.DESCRIPTION_FILE) as f:
        doc_id_to_desc = json.load(f)
    print("Number of Nodes with Description: ", len(doc_id_to_desc))

    text_data = [desc for desc in doc_id_to_desc.values()]

    model_name = models[model_id]
    if "gpt2" in model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    model.to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = 512  # Define your max length
    dataset = CustomDataset(tokenizer, text_data, max_length)

    if "gpt2" in model_name:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=n_epoch,
        learning_rate=1e-4,
        output_dir=os.path.join(llm_dir, model_names[model_id], 'results'),
        logging_dir=os.path.join(llm_dir , model_names[model_id] , 'logs'),
        logging_steps=100,
        load_best_model_at_end=False,
        evaluation_strategy="no",
        remove_unused_columns=False,
        push_to_hub=False,
        save_strategy="no",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    torch.cuda.empty_cache()
    trainer.train()
    trainer.save_model()
    model.save_pretrained(os.path.join(llm_dir , model_names[model_id],'epoch_{}'.format(n_epoch)))
    tokenizer.save_pretrained(os.path.join(llm_dir , model_names[model_id],'epoch_{}'.format(n_epoch)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune LLM model")
    parser.add_argument("--model_id", type=int, required=True, help="Model ID to fine-tune")
    args = parser.parse_args()
    main(args.model_id)