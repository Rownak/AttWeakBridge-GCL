# AttWeakBridge-GCL — Existing Architecture

Bridge Cybersecurity Attack Patterns and Weaknesses by Graph Contrastive Learning.

**Scope of this document.** This is a factual description of the project *as it currently exists on
disk* (branch `v0.0.2_notebook_to_py`). It describes the current structure, inputs, execution order,
data flow, intermediate artifacts, and final outputs, intended as a reference for planning a future
refactor. It deliberately contains no recommendations, fixes, or proposed changes. Where behavior
could not be determined from the code, it is labeled **[UNCERTAIN]**.

---

## 1. What the project does

The pipeline learns a joint embedding space over two cybersecurity knowledge bases:

- **MITRE ATT&CK** attack-pattern objects (STIX JSON bundle), and
- **MITRE CWE** weakness entries (XML catalog).

Each entity becomes a node with (a) a text description and (b) a position in a graph derived from
the source data's own relationships. A transformer language model produces a text embedding per node;
a random-walk method (DeepWalk / node2vec) produces a structural embedding per node. A dual-encoder
GNN is then trained with a contrastive objective over anchor/positive/negative tuples drawn from both
the text-similarity and graph-similarity views. The output is one 128-dimensional embedding per node,
in a space where ATT&CK and CWE nodes are directly comparable.

---

## 2. Current repository state

### 2.1 Notebook-to-script migration

Most original implementation lived in Jupyter notebooks. On the current branch the notebook contents
were copied into `.py` files and the notebooks deleted from the working tree. The git index still
lists the `.ipynb` files (they show as deleted in `git status`), and the new `.py` files are largely
untracked. Consequences visible in the code:

- The scripts execute top-to-bottom at import time, in notebook cell order. Only
  `LLM_embeddings/*.py` and `GCL/models/GNN_dual_loss2.py` have a `main()` / `__main__` guard and
  argument parsing; the rest run everything as module-level statements.
- Several scripts retain notebook residue: unused imports, exploratory `print` statements, unused
  intermediate variables (e.g. bare expressions like `a_graph.number_of_edges(), a_graph.number_of_nodes()`),
  and dead/commented-out cells.
- Every script bootstraps config the same way — walking up parent directories to insert the project
  root on `sys.path`, then `import config`.

### 2.2 Directories present in the working tree

```
AttWeakBridge-GCL/
├── config.py                       # Single global config module (all paths derived here)
├── pyproject.toml / uv.lock        # uv-managed dependency lock (Python 3.13)
├── .python-version                 # 3.13
├── run.sh                          # Root driver (currently: embedding extraction only)
├── data_preprocessing/             # Stage 1 — 7 scripts, source data → graph + text corpus
├── LLM_embeddings/                 # Stage 2 — LLM fine-tuning + text embedding extraction
├── GCL/
│   ├── get_graph_features/         # Stage 3 — node2vec / DeepWalk structural embeddings
│   ├── data_preprocessing/         # Stage 4 — similarity matrices + contrastive sample generation
│   └── models/                     # Stage 5 — dual-encoder GCL training
├── gradio_app/app.py               # Demo web UI (currently non-runnable, see §8)
├── datasets/                       # Inputs + stage-1 artifacts (git-ignored)
├── model_outputs/                  # All model artifacts (git-ignored)
└── dev_docs/                       # Sprint plans and execution logs
```

### 2.3 Directories referenced but absent from the working tree

These appear in the git history and in older documentation but **do not exist on disk**:

| Path | Referenced by | Status |
|---|---|---|
| `prompt_generation/` | `gradio_app/app.py` (`from prompt_generation import get_aug_prompt`) | Missing — the Gradio app cannot import |
| `embedding_generator/` | `config.ENCODER_PATH` points at `embedding_generator/data/encoder_decoder_model.pth` | Missing — `ENCODER_PATH` is defined but not read by any script in the working tree |
| `GCL/gcl_analysis/`, `GCL/text_hop_analysis/` | Analysis/plot notebooks for the paper's result tables | Missing — deleted with the notebooks, not ported to `.py` |
| `scripts/check_env.py` | `dev_docs/execution_logs/sprint0.md` | Missing from the working tree |
| `README.md` | `pyproject.toml` `readme` field | Missing from the working tree |

`GCL/get_graph_features/1_text_hop_emb_model.py` and `1_text_hop_emb_model_batch.py` (which produced
the `text_hop_embeddings.npy` feature) are also **absent from the working tree**, although
`run.sh` still contains a commented-out invocation of the batch variant and `GNN_dual_loss2.py`
still accepts `--feature text_hop` and would load `text_hop_embeddings.npy`. That feature branch is
therefore currently unreachable end-to-end.

---

## 3. Configuration

All path and hyperparameter configuration is centralized in [config.py](config.py). It is a plain
module of module-level constants — there is no config file, environment overlay, or CLI override for
these values; changing a run means editing the file.

**User-set values:**

| Name | Current value | Meaning |
|---|---|---|
| `ATTACK_DATASET` | `'ics_attack'` | One of `ics_attack`, `enterprise_attack`, `mobile_attack`; asserted at import |
| `CWE_DATASET` | `'cwec_v4.12'` | CWE catalog version; also the XML/JSON filename stem |
| `description_selection` | `0` | `0` = description-only text; `1` = full metadata text |
| `LLM_FT_EPOCH` | `10` | LLM fine-tuning epochs; also names the checkpoint directory `epoch_{N}` |
| `GCL_EPOCH` | `400` | GCL training epochs |

**Derived paths** (all absolute, rooted at `config.py`'s own directory):

| Name | Value |
|---|---|
| `DATA_DIR` | `datasets/{ATTACK_DATASET}/` |
| `CWE_DIR` | `datasets/cwe/` |
| `OUTPUT_DIR` | `model_outputs/{ATTACK_DATASET}/` |
| `EMBEDDING_DIR` | `model_outputs/{ATTACK_DATASET}/embeddings/` |
| `DESCRIPTION_FILE` | `DATA_DIR + doc_id_to_desc_only.json`, or `doc_id_to_full_metadata.json` when `description_selection == 1` |
| `ENCODER_PATH` | `embedding_generator/data/encoder_decoder_model.pth` (unused; target directory missing) |

Note that `CWE_DIR` is **not** parameterized by `ATTACK_DATASET` — the CWE catalog is shared across
all three ATT&CK datasets, but the node-ID mapping built in stage 1 *is* per-ATT&CK-dataset.

**Other configuration is hardcoded at the call site**, not in `config.py`: the model name lists
(repeated with differing contents in five separate files), sampling thresholds in
`3_pos_neg_sample.py`, walk parameters in `2_node2vec_deepwalk.py`, and the GNN dimensions,
learning rate, margin, and loss-blend `alpha` in `GNN_dual_loss2.py`.

---

## 4. Inputs

Two source files must be present before anything runs. Both are git-ignored and must be downloaded
manually.

| Input | Location | Source |
|---|---|---|
| ATT&CK STIX bundle | `datasets/{ATTACK_DATASET}/attack.json` | https://github.com/mitre-attack/attack-stix-data |
| CWE catalog XML | `datasets/cwe/{CWE_DATASET}.xml` | MITRE CWE XML download |

The `datasets/` directory on disk also contains several sibling copies —
`cwe_1/`, `ics_attack_1/`, `ics_attack_2/` — holding prior builds of the same artifacts. **[UNCERTAIN]**
Nothing in the code reads these paths; they appear to be manual snapshots of earlier runs, but their
provenance and which run each corresponds to cannot be determined from the code.

---

## 5. Execution sequence

Five stages, strictly ordered. Each stage consumes the previous stage's artifacts; there is no
orchestrator, dependency check, or idempotency guard — ordering is enforced only by the operator
running the scripts in order. Every script writes its outputs unconditionally, overwriting.

### Stage 1 — Source data → graph and text corpus (`data_preprocessing/`)

Run in numeric order. All seven read `config` and write into `DATA_DIR` / `CWE_DIR`.

**1. [1_CWE_xml_to_json.py](data_preprocessing/1_CWE_xml_to_json.py)**
Parses the CWE XML with `xml.etree.ElementTree` under the CWE-7 and XHTML namespaces. For each
`Weakness` element it flattens `Description`, `Extended_Description`, `Modes_Of_Introduction`,
`Common_Consequences`, `Detection_Methods`, `Potential_Mitigations`, and `Demonstrative_Examples`
into single strings (joining nested `xhtml:p` children), and keeps the raw `Related_Weaknesses` list.
→ `datasets/cwe/cwec_v4.12_info.json` (keyed by CWE ID)

**2. [2_Mitre_attack_to_json.py](data_preprocessing/2_Mitre_attack_to_json.py)**
Walks `attack.json`'s `objects` array; for every object with an `id`, extracts `Name`, `Description`,
`x_mitre_detection`, and `t_id` (the first `external_references[0].external_id`).
→ `datasets/ics_attack/attack_info.json` (keyed by STIX object id)

**3. [3_CWE_grpah_create.py](data_preprocessing/3_CWE_grpah_create.py)** *(filename contains the typo "grpah")*
Builds the CWE graph from `Related_Weaknesses` entries whose `Nature == "ChildOf"`, emitting an edge
`{source: parent CWE, target: child CWE}`. Only nodes touched by such an edge are included.
→ `datasets/cwe/cwec_v4.12_graph.json` — `{nodes: [...], links: [...]}`

**4. [4_Attack_graph_create.py](data_preprocessing/4_Attack_graph_create.py)**
Builds the ATT&CK graph from STIX objects of `type == "relationship"`, using `source_ref`/`target_ref`.
Relationships whose endpoints are not present in the bundle are skipped and counted. Nodes are the
full STIX objects.
→ `datasets/ics_attack/attack_graph.json` — `{nodes: [...], links: [...]}`

**5. [5_Calculate_hop_distances.py](data_preprocessing/5_Calculate_hop_distances.py)** — *the ID-assignment step; everything downstream depends on it*
Loads both graphs into `networkx.Graph` objects. Walking the edge lists in order, it assigns each
document a contiguous integer **embedding id**, skipping any node whose description text is empty
(for CWE: empty `Description` *and* empty `Extended_Description`). ATT&CK nodes are numbered first,
then CWE nodes — this is what makes `attack_weak_range` a pair of contiguous slices. It then computes
all-pairs shortest-path hop distance within each graph separately (never across the two), normalizes
by the per-graph maximum, and converts to a similarity as `1.0001 - normalized_distance`.

Outputs, all in `datasets/ics_attack/`:

| File | Content (current `ics_attack` run) |
|---|---|
| `doc_id_to_emb_id.json` | doc id → integer index |
| `emb_id_to_doc_id.json` | inverse mapping |
| `emb_id_to_tid.json` | index → ATT&CK technique id (`t_id`); for CWE nodes the value is the CWE id itself |
| `attack_weak_range.json` | `{"attack": [0, 203], "cwe": [203, 1136], "n_nodes": 1136}` |
| `attack_hop_sim_matrix.npy` | `(203, 203)` float |
| `weakness_hop_sim_matrix.npy` | `(933, 933)` float |
| `hop_training_data.npy` | `(455281, 3)` — `[source_emb_id, target_emb_id, normalized_hop_distance]` |

Note the two hop-similarity matrices are indexed *locally* (CWE indices are rebased by subtracting
the attack block size), whereas `hop_training_data.npy` uses global embedding ids. **[UNCERTAIN]**
No script remaining in the working tree consumes these three hop artifacts — their only known
consumer was the missing `1_text_hop_emb_model*.py`.

**6. [6_save_text_description.py](data_preprocessing/6_save_text_description.py)**
For each doc id in the mapping, assembles the model input text. ATT&CK nodes use `Description`
(plus `x_mitre_detection` in the full-metadata variant); CWE nodes use `Name + Description +
Extended_Description` (plus the five remaining metadata fields in the full variant). Both variants
are always written on every run, regardless of `description_selection` — that setting only selects
which file `config.DESCRIPTION_FILE` points at downstream.
→ `doc_id_to_desc_only.json`, `doc_id_to_full_metadata.json`

**7. [7_save_graph_edges.py](data_preprocessing/7_save_graph_edges.py)**
Translates both graphs' edges into embedding-id space and concatenates them into one flat list,
keeping only edges whose endpoints both survived step 5's filtering. This is the single edge list
the GNN trains on; because stages 3 and 4 never create cross-domain edges, the graph consists of two
disconnected components (ATT&CK and CWE).
→ `datasets/ics_attack/graph_edges.json` — 2078 `[src, dst]` pairs in the current run

### Stage 2 — Text embeddings (`LLM_embeddings/`)

**[1_llm_fine_tune.py](LLM_embeddings/1_llm_fine_tune.py)** — `--model_id {0..3}` selecting
`["jackaduma/SecRoBERTa", "ehsanaghaei/SecureBERT", "gpt2-xl", "gpt2"]`.
Continues pretraining the selected model on the corpus in `config.DESCRIPTION_FILE` via HuggingFace
`Trainer`. BERT-family models use masked-LM (`mlm_probability=0.15`); GPT-2 models use causal LM with
`DataCollatorWithPadding`. Fixed hyperparameters: `max_length=512`, batch size 2, gradient accumulation
4, lr `1e-4`, `fp16=True`, `save_strategy="no"`. Labels are set to a clone of `input_ids`.
→ `model_outputs/{dataset}/llm_finetuned_models/{model_name}/epoch_{LLM_FT_EPOCH}/` (plus a
`results/` directory from `trainer.save_model()`)

**[2_extract_embeddings.py](LLM_embeddings/2_extract_embeddings.py)** — `--model_id {0..6}` selecting
`["pt_SecRoBERTa", "SecRoBERTa", "pt_SecureBERT", "SecureBERT", "pt_gpt2-xl", "gpt2-xl", "pt_bert"]`.
The `pt_` prefix means *pretrained* (loaded from the HuggingFace hub); an unprefixed name loads the
fine-tuned checkpoint from stage 2's output directory. Per document: tokenize, split into fixed-size
chunks (512, or `model.config.n_positions` for GPT-2-XL), run a forward pass with
`output_hidden_states=True`, mean-pool the **last hidden layer** over the sequence dimension, then
mean across chunks. Embeddings are written into a dense array indexed by `doc_id_to_emb_id`.
→ `model_outputs/{dataset}/embeddings/{model_name}/text_embeddings.npy` — `(1136, 768)` for
`pt_SecRoBERTa` in the current run.

Note: chunking iterates the token list in `chunk_size` steps *without* reserving room for special
tokens, and the ids are fed to the model directly via `convert_tokens_to_ids` (no
`build_inputs_with_special_tokens`), so no CLS/SEP/BOS tokens are added.

**Drivers.** [run.sh](run.sh) (repo root) exports `AWEB_DIR` and, as currently written, runs *only*
`2_extract_embeddings.py --model_id 6` (`pt_bert`); the fine-tuning loop and the `text_hop`/node2vec
blocks are commented out. [LLM_embeddings/run.sh](LLM_embeddings/run.sh) is the fuller variant,
looping fine-tuning then extraction over model ids `0 1`, but it invokes the scripts by bare filename
so it must be run from inside `LLM_embeddings/`.

### Stage 3 — Structural graph embeddings (`GCL/get_graph_features/`)

**[2_node2vec_deepwalk.py](GCL/get_graph_features/2_node2vec_deepwalk.py)**
Builds a `networkx.Graph` from `graph_edges.json` (note: from the edge list only, via
`add_edges_from` — isolated nodes in `node_list` are never added), then trains two `gensim.Word2Vec`
skip-gram models over random walks:

- **node2vec**: 2nd-order biased walks (Grover & Leskovec `p`/`q`), `dimensions=128`,
  `walk_length=30`, `num_walks=100`, `window=10`, `p=q=1.0` as called. The biased-walk implementation
  is hand-rolled in this file; per `dev_docs/execution_logs/sprint0.md` it replaced the PyPI
  `node2vec` package, which pinned `numpy<2.0`.
- **DeepWalk**: uniform walks, `num_walks=200`, `walk_length=30`, `vector_size=128`, `window=5`,
  `epochs=10`.

Rows are emitted in `G.nodes()` iteration order.
→ `model_outputs/{dataset}/embeddings/node2vec.npy`, `deepwalk.npy` — both `(1136, 128)`

[run_script.sh](GCL/get_graph_features/run_script.sh) invokes the script by bare filename, so it must
be run from inside `GCL/get_graph_features/`.

### Stage 4 — Similarity analysis and contrastive sampling (`GCL/data_preprocessing/`)

All three scripts hardcode `model_name = models[0]`; the active model is changed by editing the list
at the top of each file (they are not CLI-parameterized, and the lists differ between files —
`1_text_similarities.py` has eight entries, the other two have one).

**1. [1_text_similarities.py](GCL/data_preprocessing/1_text_similarities.py)**
Computes the full cosine-similarity matrix over `text_embeddings.npy`, then for each node produces
its similarity list sorted descending, **filtered to its own domain** (an ATT&CK row keeps only
ATT&CK neighbors; a CWE row keeps only CWE neighbors).
→ `gcl_data/{model}/text_sim_matrix.npy` `(1136, 1136)`, `attack_text.pkl` (203 rows × 203),
`weak_text.pkl` (933 rows × 933)

**2. [2_analysis_text_sim.py](GCL/data_preprocessing/2_analysis_text_sim.py)**
Diagnostic only — plots a cosine-similarity histogram (bins of 0.05) for each domain.
→ `gcl_data/{model}/{CWE|ATTACK}_graph_deepwalk_sim_histogram.png`

**3. [3_pos_neg_sample.py](GCL/data_preprocessing/3_pos_neg_sample.py)** — *builds the training set*
Loads `text_embeddings.npy` and `deepwalk.npy` and computes four cosine matrices: cross-domain
attack↔weakness in **text** space, and within-domain attack↔attack / weakness↔weakness in **DeepWalk**
space. It then emits 5-tuples `(anchor, text_positive, text_negative, graph_positive, graph_negative)`
where the text pair is cross-domain and the graph pair is within-domain. Two sampling strategies run
unconditionally on every invocation:

| | `get_sample_10` → `anchor_pos_neg_triple_10.pkl` | `get_sample_11` → `anchor_pos_neg_triple_11.pkl` |
|---|---|---|
| Selection | top-*k* / bottom-*k* by `argsort`, `k = 30` | percentile thresholds, top/bottom 4% (18%/18% for the weakness text view) |
| Pairing | positives and negatives zipped by rank index `j` | one view zipped by rank, the other sampled at a uniform random index |
| Size (current run) | 34,080 tuples | 43,168 tuples |

Both shuffle before saving. `random` is used without a seed, so sample 11 is not reproducible across
runs. Several module-level constants (`top_percentage`, `bottom_percentage`, `positive_threshold`,
`negative_threshold`) are defined but shadowed or unused — `get_sample_11` redefines the percentages
locally and the two thresholds are never read.
→ `model_outputs/{dataset}/gcl_data/{model}/anchor_pos_neg_triple_{10,11}.pkl`

### Stage 5 — GCL training (`GCL/models/`)

**[GNN_dual_loss2.py](GCL/models/GNN_dual_loss2.py)** — the only model script in the working tree.
Fully CLI-parameterized:

```
--model_name {pt_SecRoBERTa,SecRoBERTa,pt_SecureBERT,SecureBERT,pt_gpt2-xl,gpt2-xl}
--gnn_model  {GCN,GAT}
--sample     {10,11}
--feature    {text_hop,node2vec,deepwalk}
--loss       {triplet}
```

Architecture (`DualEncoder`): two independent GNN towers over the *same* edge index — tower 1 over
the text embeddings, tower 2 over the second feature. Each tower is 2 layers (`GCNConv×2`, or
`GATConv` with 8 heads then 1 head). In parallel, the raw text embedding passes through `fc1`
(→128) and the raw second feature through `fc2` (→ same dim). All four are concatenated
(`2*out_channels + 2*feature_2_dim`) and projected by `fc3` to the 128-d output.

Training: full-batch (all nodes, all tuples, every epoch), Adam `lr=0.001`, `GCL_EPOCH` (400) epochs.
Loss is `alpha * loss_text + (1 - alpha) * loss_graph` with `alpha = 0.5`, where each term is the
selected contrastive loss applied to the (anchor, positive, negative) triple of that view.

Three loss classes are implemented — `ContrastiveLoss` (cosine-distance triplet margin, `margin=1.0`),
`JSDContrastiveLoss`, and `InfoNCELoss` — but `--loss` only accepts `triplet`, and the infoNCE
branch in the training loop is commented out. BYOL, mentioned in older documentation, is not present
in this file.

→ `model_outputs/{dataset}/gcl_data/{model}/sample_{N}/{GCN|GAT}/{loss}/text_{feature}_dual3_gm_1.0.npy`
— `(1136, 128)`. For `--feature text_hop` the filename is `text_hop_dual3_gm_1.0.npy`.

[run_dual_loss.sh](GCL/models/run_dual_loss.sh) sweeps the configured lists — currently
`model_names=(pt_SecRoBERTa)`, `gnn_models=(GCN GAT)`, `features=(node2vec deepwalk)`, `samples=(10)`,
`loss=triplet`, i.e. four runs. It invokes `GNN_dual_loss2.py` by bare filename, so it must be run
from inside `GCL/models/`.

---

## 6. Data flow summary

```
attack.json ──► 2_Mitre_attack_to_json ──► attack_info.json ──┐
            └─► 4_Attack_graph_create  ──► attack_graph.json ─┤
                                                              ├─► 5_Calculate_hop_distances
cwec_v4.12.xml ─► 1_CWE_xml_to_json ──► cwec_..._info.json ───┤    │
                └► 3_CWE_grpah_create ─► cwec_..._graph.json ─┘    │
                                                                   ├─► doc_id_to_emb_id / emb_id_to_doc_id
                                                                   ├─► attack_weak_range.json
                                                                   └─► hop_*.npy  [no live consumer]
                                                 │
       ┌─────────────────────────────────────────┴──────────────────────────────┐
       ▼                                                                        ▼
6_save_text_description                                                7_save_graph_edges
       │  doc_id_to_desc_only.json                                              │ graph_edges.json
       ▼                                                                        │
1_llm_fine_tune ──► llm_finetuned_models/{model}/epoch_10/                      │
       ▼                                                                        ▼
2_extract_embeddings ──► embeddings/{model}/text_embeddings.npy    2_node2vec_deepwalk
       │                          │                                      │  deepwalk.npy
       │                          │                                      │  node2vec.npy
       ▼                          └──────────────┬───────────────────────┘
1_text_similarities                              ▼
  attack_text.pkl, weak_text.pkl        3_pos_neg_sample ──► anchor_pos_neg_triple_{10,11}.pkl
       │                                         │
       ▼                                         │
2_analysis_text_sim (plots)                      │
                                                 ▼
    text_embeddings.npy + {deepwalk|node2vec}.npy + graph_edges.json + tuples
                                                 ▼
                                         GNN_dual_loss2.py
                                                 ▼
                    sample_{N}/{GCN|GAT}/triplet/text_{feature}_dual3_gm_1.0.npy   ← final output
```

The key coupling across every stage is the **embedding id**: stage 1 step 5 fixes a single contiguous
integer index per document, and every `.npy` array thereafter — text embeddings, deepwalk/node2vec,
similarity matrices, the sample tuples, and the final output — is positionally indexed by it. There
is no id column carried alongside the arrays; alignment is by row position only.

---

## 7. Artifact inventory (current `ics_attack` run on disk)

**`datasets/ics_attack/`** — 203 ATT&CK nodes, 933 CWE nodes, 1136 total, 2078 edges.

| File | Shape / size | Producer |
|---|---|---|
| `attack.json` | input | manual download |
| `attack_info.json` | dict | stage 1.2 |
| `attack_graph.json` | `{nodes, links}` | stage 1.4 |
| `doc_id_to_emb_id.json`, `emb_id_to_doc_id.json`, `emb_id_to_tid.json` | 1136 entries | stage 1.5 |
| `attack_weak_range.json` | `{"attack":[0,203],"cwe":[203,1136],"n_nodes":1136}` | stage 1.5 |
| `attack_hop_sim_matrix.npy` | `(203, 203)` | stage 1.5 |
| `weakness_hop_sim_matrix.npy` | `(933, 933)` | stage 1.5 |
| `hop_training_data.npy` | `(455281, 3)` | stage 1.5 |
| `doc_id_to_desc_only.json`, `doc_id_to_full_metadata.json` | 1136 entries | stage 1.6 |
| `graph_edges.json` | 2078 pairs | stage 1.7 |

**`datasets/cwe/`** — `cwec_v4.12.xml` (input), `cwec_v4.12_info.json` (1.1), `cwec_v4.12_graph.json` (1.3).

**`model_outputs/ics_attack/`**

| Path | Shape | Producer |
|---|---|---|
| `llm_finetuned_models/SecRoBERTa/epoch_10/` | HF checkpoint | stage 2.1 |
| `llm_finetuned_models/SecRoBERTa/results/` | HF checkpoint + `training_args.bin` | stage 2.1 |
| `embeddings/pt_SecRoBERTa/text_embeddings.npy` | `(1136, 768)` | stage 2.2 |
| `embeddings/SecRoBERTa/text_embeddings.npy` | `(1181, 768)` | stage 2.2 |
| `embeddings/deepwalk.npy`, `node2vec.npy` | `(1136, 128)` | stage 3 |
| `gcl_data/pt_SecRoBERTa/text_sim_matrix.npy` | `(1136, 1136)` | stage 4.1 |
| `gcl_data/pt_SecRoBERTa/attack_text.pkl` | 203 rows × 203 | stage 4.1 |
| `gcl_data/pt_SecRoBERTa/weak_text.pkl` | 933 rows × 933 | stage 4.1 |
| `gcl_data/pt_SecRoBERTa/*_graph_deepwalk_sim_histogram.png` | 2 plots | stage 4.2 |
| `gcl_data/pt_SecRoBERTa/anchor_pos_neg_triple_10.pkl` | 34,080 tuples | stage 4.3 |
| `gcl_data/pt_SecRoBERTa/anchor_pos_neg_triple_11.pkl` | 43,168 tuples | stage 4.3 |
| `gcl_data/pt_SecRoBERTa/sample_10/GCN/triplet/text_deepwalk_dual3_gm_1.0.npy` | `(1136, 128)` | stage 5 |
| `gcl_data/SecRoBERTa/text_sim_matrix.npy` | `(1136, 1136)` | stage 4.1 |
| `gcl_data/gpt2-xl/` | empty directory | — |

Two artifacts on disk are inconsistent with the current stage-1 build: `embeddings/SecRoBERTa/text_embeddings.npy`
has 1181 rows against the current 1136-node mapping, and `gcl_data/SecRoBERTa/text_sim_matrix.npy` has
1136. **[UNCERTAIN]** The 1181-row array is presumably left over from an earlier dataset build; the
code has no versioning or provenance metadata that would confirm which run produced it.

Only one stage-5 output exists (`GCN` + `deepwalk` + `sample_10`), although `run_dual_loss.sh` is
configured for four combinations — so the current on-disk state reflects a partial sweep.

---

## 8. Auxiliary components

**[gradio_app/app.py](gradio_app/app.py)** — a two-column Gradio Blocks UI: the user enters a query
and an OpenAI API key, and the app shows the model's answer to the raw prompt beside its answer to an
augmented prompt built by `prompt_generation.get_aug_prompt.main(query)`. It calls `gpt-3.5-turbo`
via the `openai` client. In its current state it is **not runnable**: `prompt_generation/` is absent
from the working tree, `gradio` and `openai` are not in `pyproject.toml`, and `demo.launch()` is
hardcoded to `server_name="129.108.18.37", server_port=7860, share=True` (a specific host address).
It is also launched at module import, not under a `__main__` guard.

**Analysis / evaluation.** The scripts that produced the paper's reported metrics — precision@k,
hit@k, silhouette score, Rand index, t-SNE plots, BRON-based evaluation — lived in
`GCL/gcl_analysis/` and `GCL/text_hop_analysis/` as notebooks and are **not present** in the working
tree. There is consequently no evaluation step in the current runnable pipeline; stage 5 ends at the
embedding array.

---

## 9. Environment

- Python 3.13 (`.python-version`), dependencies locked with `uv` (`pyproject.toml` + `uv.lock`);
  scripts are run as `uv run python <script>`.
- Declared dependencies: `torch>=2.4` (via the explicit `pytorch-cu124` index), `transformers[torch]`,
  `accelerate`, `networkx`, `torch-geometric`, `numpy>=2.0`, `scikit-learn`, `gensim`, `matplotlib`;
  `jupyter` in the dev group.
- GPU: stages 2, 3, and 5 set `CUDA_VISIBLE_DEVICES` and select `cuda` when available, falling back
  to CPU. Stage 2 sets it to `"0"` unconditionally; stage 3 uses `setdefault`, so it is overridable.
  Per `dev_docs/execution_logs/sprint0.md`, the original experiments ran on A6000 hardware and the
  current development machine is an RTX 4090 Laptop (16GB).
- `datasets/` and `model_outputs/` are git-ignored, so no pipeline data or model artifact is under
  version control.

---

## 10. Cross-cutting properties of the current code

Observations relevant to planning a refactor, stated as facts about the present code:

- **No orchestration.** Five stages, ~13 scripts, run by hand or by three shell scripts that are not
  chained. Three of those shell scripts invoke Python by bare filename and therefore only work from
  their own directory, while the root `run.sh` uses repo-relative paths.
- **Global mutable config.** Every script imports the same `config` module; changing dataset, model,
  or epoch count means editing source. Stage-4 scripts additionally require editing a hardcoded
  `models` list, and those lists are duplicated with different contents across five files.
- **Positional coupling.** All inter-stage contracts are unlabeled `.npy` arrays and `.pkl` lists
  aligned by row index against a mapping produced in stage 1.5. Nothing validates that an array's row
  count matches the current `n_nodes` (the 1181-vs-1136 artifact above went undetected).
- **No determinism controls.** `random` and `numpy.random` are used unseeded in stage 3 (walks) and
  stage 4.3 (sampling); no torch seed is set in stage 5.
- **No error handling or validation.** Missing upstream files surface as raw `FileNotFoundError`; a
  stage will happily consume a stale artifact from a previous run. Two `try/except` blocks in stages
  1.2 and 2.2 `break` out of their loops on any exception, silently producing a truncated result.
- **No tests.** No test directory or harness exists in the working tree.
- **Dead and unreachable paths.** `ENCODER_PATH`; the hop-similarity artifacts; the `text_hop`
  feature option; the `JSD`/`infoNCE`/BYOL loss paths; `get_sample_11` output (nothing on disk
  consumes `sample_11`); the `positive_threshold`/`negative_threshold` constants.
- **Provenance.** No run is tagged with the config that produced it. Output paths encode
  `{dataset}/{model}/sample_{N}/{gnn}/{loss}/text_{feature}_...`, but not epoch count, margin,
  `description_selection`, or the code revision.
