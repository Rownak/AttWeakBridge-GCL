# AttWeakBridge-GCL

Bridge Cybersecurity Attack Patterns and Weaknesses by Graph Contrastive Learning

## Overview

AttWeakBridge-GCL is a framework that bridges cybersecurity attack patterns (MITRE ATT&CK) and weaknesses (CWE) using Graph Contrastive Learning (GCL). The project leverages fine-tuned language models and graph neural networks to learn meaningful embeddings that capture relationships between attack patterns and weaknesses.

## Project Structure

```
AttWeakBridge-GCL/
├── datasets/                    # Dataset storage (CWE and MITRE ATT&CK)
│   ├── cwec_v4.12/             # CWE dataset
│   ├── enterprise_attack/       # MITRE ATT&CK Enterprise
│   ├── ics_attack/             # MITRE ATT&CK ICS
│   └── mobile_attack/          # MITRE ATT&CK Mobile
├── data_preprocessing/          # Data preprocessing notebooks
├── LLM_embeddings/             # LLM fine-tuning and embedding generation
├── GCL/                        # Graph Contrastive Learning components
│   ├── get_graph_features/     # Graph feature extraction
│   ├── models/                 # GCL model training
│   └── gcl_analysis/           # Analysis and visualization
├── embedding_generator/         # Embedding generation utilities
├── prompt_generation/          # Prompt generation for LLMs
├── gradio_app/                 # Web interface
├── config.py                   # Configuration file
└── run.sh                      # Main execution script
```

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- NetworkX
- DGL (Deep Graph Library)
- Jupyter Notebook
- Additional dependencies in requirements.txt (if available)

## Configuration

Edit `config.py` to configure:
- `ATTACK_DATASET`: Choose from `'ics_attack'`, `'enterprise_attack'`, or `'mobile_attack'`
- `CWE_DATASET`: CWE version (default: `'cwec_v4.12'`)
- `description_selection`: 0 for description only, 1 for full metadata
- `LLM_FT_EPOCH`: LLM fine-tuning epochs (default: 10)
- `GCL_EPOCH`: GCL training epochs (default: 400)

## Dataset Setup

Before running the pipeline, download the required datasets:

- **MITRE ATT&CK Data**: Download from [https://attack.mitre.org/resources/attack-data-and-tools/](https://attack.mitre.org/resources/attack-data-and-tools/)
  - Enterprise ATT&CK
  - ICS ATT&CK
  - Mobile ATT&CK
- **CWE Data**: Download CWE XML data and place in `datasets/cwec_v4.12/`

Place the downloaded datasets in their respective folders under `datasets/`.

## Execution Pipeline

### Step 1: Data Preprocessing

Run all notebooks in the `data_preprocessing/` folder in sequential order:

```bash
cd data_preprocessing
```

Execute the following notebooks in order:

1. **1_CWE_xml_to_json.ipynb** - Convert CWE XML data to JSON format
2. **2_Mitre_attack_to_json.ipynb** - Convert MITRE ATT&CK data to JSON format
3. **3_CWE_graph_create.ipynb** - Create CWE knowledge graph
4. **4_Attack_graph_create.ipynb** - Create ATT&CK knowledge graph
5. **5_Calculate_hop_distances.ipynb** - Calculate hop distances between nodes
6. **6_save_text_description.ipynb** - Extract and save text descriptions
7. **7_save_graph_edges.ipynb** - Save graph edge information

**Note**: Ensure all required dataset files are present in the `datasets/` folder before running these notebooks.

### Step 2: LLM Fine-tuning and Embedding Generation

From the root directory, run:

```bash
bash run.sh
```

This script performs the following operations:
1. **LLM Fine-tuning**: Fine-tunes security-focused language models (SecRoBERTa, SecureBERT, GPT-2-XL)
2. **Embedding Extraction**: Generates text embeddings from fine-tuned models
3. **Graph Feature Generation**: Creates text-hop embeddings combining textual and graph structural information
4. **Optional**: Node2Vec and DeepWalk embeddings (uncomment if needed)

### Step 3: GCL Model Training

Navigate to the GCL models directory and run the training script:

```bash
cd GCL/models
bash run_dual_loss.sh
```

This script trains GCL models with various configurations:
- **Models**: Pre-trained and fine-tuned versions of SecRoBERTa, SecureBERT, GPT-2-XL
- **GNN Architectures**: GCN (Graph Convolutional Network), GAT (Graph Attention Network)
- **Features**: text_hop, node2vec, deepwalk
- **Loss Functions**: Triplet loss, InfoNCE, JSD, BYOL

Training outputs are saved in `model_outputs/`.

### Step 4: Analysis and Visualization

Navigate to the `GCL/gcl_analysis/` folder to run various analysis notebooks:

```bash
cd GCL/gcl_analysis
```

Available analysis notebooks:

- **1_bron_data.ipynb** - BRON dataset analysis
- **2_plots_stat_sig.ipynb** - Statistical significance plots
- **hit@k_paper_exp.ipynb** - Hit@K metric evaluation
- **precision@k_*.ipynb** - Precision@K metrics for different configurations
- **rand_index.ipynb** - Rand Index clustering evaluation
- **silhouette_score.ipynb** - Silhouette score analysis
- **top_pairs_text_graph.ipynb** - Top similar pairs analysis
- **top_similar.ipynb** - Top similar entities visualization

## Model Outputs

All model outputs, embeddings, and results are saved in:
```
model_outputs/<dataset_name>/
├── embeddings/          # Generated embeddings
├── models/             # Trained model checkpoints
└── results/            # Evaluation results
```

## Additional Components

### Embedding Generator
The `embedding_generator/` folder contains alternative embedding generation methods:
- Variational Autoencoder (VAE)
- Encoder-Decoder
- Conditional GAN (cGAN)

### Gradio App
A web interface for interacting with the trained models (located in `gradio_app/`).

## Citation

If you use this code in your research, please cite the relevant paper.


## Contact

Ahnaf Farhan
rownak.utep@gmail.com
