# Design: making graph structure add value over frozen SapBERT without gold alignments

Context: Sprint 9 ([plan](../sprints/sprint_9.md), [9.3 log](../execution_logs/sprint_9_3.md)). Question: what is the smallest set of
changes most likely to make NCIT/DOID structure **improve** on frozen SapBERT under the unsupervised protocol (no gold pair in
training), on a short timeline?

## 0. Answer in brief

- **Structure already helps, but only as a training-free residual.** The parameter-free directional neighbour-mean adapter
  from Sprint 5.1 (**A1**: `z = [x̂ ; √λ · normalize(h)]`, `h = [mean SapBERT of parents ; mean SapBERT of children]`, λ = 0.25)
  beats frozen SapBERT on every test metric that was measured: global MRR **0.4922 vs 0.4475**, F1 (greedy, no filter)
  **0.4601 vs 0.4136** (P 0.5125 vs 0.4663, R 0.4174 vs 0.3716). It has no trained weights and reads no gold pairs. Its only
  link to gold is that λ and "directional" were picked on 211 val pairs in Sprint 5.1, where the optimum was flat
  (λ 0.1–0.5 within noise).
- **The Sprint 9 matrix loses to SapBERT for two identifiable reasons, not because structure is useless.**
  1. **Replacement instead of residual.** U2 passes SapBERT through a randomly initialised 128-d fusion with no path back to the
     original vector. U2-I (SapBERT + GCN + InfoNCE) lands almost exactly on the *untrained* GCN smoothing probe of Sprint 4:
     Hits@1 0.274 vs 0.260, Hits@10 0.697 vs 0.709, MRR 0.409 vs 0.412. Training learned "SapBERT, smoothed", which trades rank-1
     precision for top-10 recall.
  2. **The pseudo-pairs carry no information beyond SapBERT.** The positives are SapBERT's own top-5, so the text loss asks the
     model to reproduce SapBERT's ranking, including its errors. Even with the residual floor (U1) the best case is "SapBERT,
     slightly perturbed" (MRR 0.4276).
- **Recommended plan (about 3 working days):**
  1. Bring A1 into the unsupervised table, with the full λ curve and a label-free λ rule. About 0.5 day, no training.
  2. Score-level fusion of A1 (or SapBERT) with the existing unsupervised GCN U2-I, at fixed weights. About 0.5 day, no training.
  3. Pseudo-seed self-training of the residual GCN adapter (the A11 recipe) from high-confidence mutual pairs. About 1.5 days.
     This is the experiment that answers "is gold supervision necessary?".
- **Outlook.** A gain of about +0.04–0.05 MRR / F1 over SapBERT is realistic without labels (it is already measured for A1).
  **A large gain is unlikely without changing the supervision signal.** Sprint 6.3 shows that 25% of the gold train pairs (≈ 293)
  only bring the trained residual GCN to A1's level on val (0.456 vs 0.4555), and that the big jump (fusion A19, test MRR
  0.6385, +0.19) needs all 1,406 gold pairs. Changing the evaluation setup is not recommended as a way to "win": the train-label
  filter and the candidate pools raise every row, SapBERT included.

---

## 1. Diagnosis from existing results

| Evidence (source) | What it shows |
|---|---|
| Untrained GCN smoothing of SapBERT: Hits@1 0.260, Hits@10 0.709, MRR 0.412 (Sprint 4 probe, [design](sapbert_residual_graph_adapter.md) §2) | Averaging a node with its neighbours blurs the fine distinctions that decide rank 1 but adds recall. |
| SapBERT + 0.25 · 1-hop neighbour mean: test MRR 0.469 (same probe) | The same information, *added beside* SapBERT, helps on every metric. |
| A1, directional (parents and children kept apart), λ 0.25: test MRR 0.4922, F1 0.4601 ([5.5](../execution_logs/sprint_5_5.md), [8.2](../execution_logs/sprint_8_2.md)) | Directional aggregation adds ≈ +0.03 val MRR over undirected (Sprint 5.1). Still no training and no gold pairs in the vectors. |
| U2-I: Hits@1 0.274, Hits@10 0.697, MRR 0.409 ([9.3](../execution_logs/sprint_9_3.md)) | Matches the untrained smoothing probe: the trained GCN converged to a smoothing operator. |
| U1 (residual floor, MLP, triplet): MRR 0.4276 | A floor prevents collapse, but self-generated positives cannot teach anything new. |
| U2-mut > U2-rand > U2-hard ≈ U2 (MRR 0.276 / 0.232 / 0.199 / 0.190) | Cleaner positives matter more than harder negatives; none reaches U0 without a residual. |
| Anchor-agreement probe, *gold* anchors: ratio 1.24 for gold vs top wrong candidate (gate 1.5) ([6.2](../execution_logs/sprint_6_2.md)) | Post-retrieval structural re-ranking discriminates weakly even with gold anchors; with pseudo-anchors it will be weaker. |
| Scaling curve, gold pairs: A11 val MRR 0.456 at 25%, 0.494 at 100%; A0 val 0.4125, A1 val 0.4555 ([6.3](../execution_logs/sprint_6_3.md)) | A few hundred gold seeds buy roughly what the training-free residual already gives; large gains need all the seeds. |

One property of this dataset limits every unsupervised option. **NCIT and DOID share no edges.** Message passing only moves
information *within* each ontology, and cross-graph DeepWalk cosine is at chance (MRR 0.0007, Sprint 4). Structure can only help by
making each node's vector more informative about its own context. It cannot connect the two graphs. That is exactly what A1 does,
and why a residual form works while a replacement form does not.

---

## 2. Recommended changes

### R1. Training-free directional residual (A1) under the unsupervised protocol

**Idea.** `z_n = [x̂_n ; √λ · normalize([mean x̂ over parents(n) ; mean x̂ over children(n)])]`, so that
`cos(z_s, z_t) ∝ cos_SapBERT + λ · cos_context` (up to the shared normalisation). SapBERT stays intact as the first block. The graph
term only re-orders candidates whose own text scores are close, and λ = 0 is exactly SapBERT. Directionality keeps parents and
children apart, so a node is not averaged toward its own granularity confusers. This is the "residual / graph regularisation instead of
replacement" option in its simplest form, with no trained weights to overfit pseudo-labels. APPNP and Correct & Smooth make the same
argument for keeping a node's own features in propagation.

**Needs.** Graph and text only (no pseudo-pairs, no gold). It is already implemented as `residual_adapter` with
`adapter: neighbor_mean`, `directional: true`; `gcl_train` skips the optimiser when there are 0 parameters.

**Choosing λ without gold.** Report the **whole curve**, so nothing is selected on test. Pick the headline value by a label-free
rule fixed before the run: the λ that maximises the number of **mutual nearest-neighbour pairs** between NCIT and DOID. Unsupervised
bilingual lexicon induction uses this kind of criterion for model selection (Conneau et al. 2018). Whether it tracks MRR here is
itself a result: the curve shows it.

**Minimal ablation.**

| run | purpose |
|---|---|
| λ ∈ {0, 0.1, 0.25, 0.5, 1.0} × {undirected, directional} | λ = 0 must reproduce U0 exactly; the curve shows the flat optimum and where smoothing starts to cost Hits@1 |
| λ from the mutual-NN rule | the headline row, chosen without gold |
| A1 with `edge_rewire` (degree-preserving random edges, already implemented) | shows the gain comes from the real hierarchy |
| A1 on frozen BERT | whether the residual helps a weak encoder too (expected: little, because neighbour means of BERT are also poor) |

**Expected.** Test MRR ≈ 0.49 and F1 ≈ 0.46 at λ 0.25, directional: the Sprint 5/8 numbers, since the vectors are deterministic and
the universe is the same 24,221 nodes. That is a check, not a new finding. The new part is the label-free λ and the rewired control.

### R2. Score-level fusion with the unsupervised GCN, at fixed weights

**Idea.** `score = cos_base + w · cos_U2-I`, using the existing `fusion` method kind, with base = SapBERT or A1. U2-I has the best
Hits@10 of all rows (0.697) and a different error profile from SapBERT. Sprint 4 found that a GCN which loses on its own can still
add to SapBERT as a score term (SapBERT + 0.25 · GCN ≈ 0.49 MRR, supervised GCN). This is the "post-retrieval / late fusion versus
embedding-level fusion" option: the GCN can no longer overwrite SapBERT.

**Needs.** Nothing new: the U0, A1 and U2-I artifacts exist. No training and no gold.

**Minimal ablation.** w ∈ {0, 0.25, 0.5}, all reported (the 6.1 fusion used 0.5), for base ∈ {SapBERT, A1}. Add the control
"base + w · U2 (triplet)" to show that the member's quality matters, and the 3-seed ensemble of U2-I as the member, as in A17/A18.

**Expected.** Uncertain. U2-I is close to a smoothing operator, and A1 already contains a better-behaved version of that signal, so
fusion with A1 may add little. Fusion with plain SapBERT will likely land near A1. Run it because it costs half a day and tells us
whether the trained GCN holds anything beyond smoothing.

### R3. Pseudo-seed self-training of the residual GCN adapter (the supervised A11 recipe, with pseudo-pairs)

**Idea.** Bootstrapping, as in BootEA and in VecMap's self-learning: replace the gold train pairs of the A11 recipe (residual GCN
adapter with SapBERT floor, full-softmax InfoNCE, 50 epochs) with **pseudo-seeds** chosen for reliability, not quantity.

- Take mutual nearest neighbours under the **A1** score, not raw SapBERT. A1 is the stronger first stage, and it already uses
  structure, so the seeds include structurally consistent pairs.
- Keep the top q% by margin to the second-best candidate, with q fixed in advance (for example 10%, a few hundred pairs, the size
  at which gold seeds matched A1).
- Enforce one-to-one.

The floor keeps the output at SapBERT when the adapter learns nothing. The full softmax supplies the hard negatives automatically: all
other DOID nodes are candidates. Pseudo-seed pairs are masked as "other positives" in place of gold pairs. This fixes both failure
causes of Section 1 at once: a residual architecture, and positives that differ from SapBERT's raw top-k.

**Needs.** Pseudo-pairs only. Code: an option for `infonce_full` to take its positive pairs and mask from a pseudo-seed file instead
of `alignments.parquet`, and a guard rule that accepts it under `protocol: unsupervised` (≈ 1 day with tests). Optionally one or two
refresh rounds: re-mine the seeds with the trained model and retrain.

**Minimal ablation.**

| run | purpose |
|---|---|
| pseudo-seeds top 10% vs top 25% by margin | the reliability-versus-quantity trade-off |
| same number of seeds drawn at random from A1's mutual pairs | separates "seed reliability" from "any seeds" |
| with floor vs without floor | confirms the residual is what prevents the U2 collapse |
| round 1 vs round 2 | whether self-training compounds or drifts |
| reference (already measured, not re-run): the gold-seed scaling curve at a matched size (≈ 300 pairs, val 0.456) | puts pseudo-seeds against gold seeds of the same size |

**Expected.** At best close to A1 plus a few points. Seed precision cannot be measured without gold. If it is well below that of gold
seeds, the gold-seed curve says the result will sit at or below A1. This experiment is still the most informative of the three,
because it is the cleanest test of whether the supervision signal is the limiting factor.

### Optional add-on: CSLS hubness correction at scoring time

`CSLS(s, t) = 2 cos(s, t) − r_T(s) − r_S(t)`, where `r` is the mean cosine to the 10 nearest cross-graph neighbours (Conneau et al.
2018). It penalises "hub" DOID targets that are near many NCIT sources. This is label-free and training-free. It is cheap as an
evaluator scoring option, about half a day including tests. There is no repo evidence yet on how hub-prone SapBERT is on DOID;
apply it to the best of R1–R3 only, as a single extra row.

---

## 3. Options considered and not recommended now

| Option | Reason |
|---|---|
| Per-node gated fusion of SapBERT / GCN / DeepWalk | A softmax gate can drive SapBERT's weight to 0 for any node (no floor). DeepWalk is not comparable across the disconnected graphs. Unsupervised gates have no signal to learn from. A bounded per-node confidence on the *graph residual only* is the safe version, but it is a refinement of R1/R3, not a first step ([design](sapbert_residual_graph_adapter.md) §4). |
| Graph-regularisation or similarity-preservation losses (keep SapBERT geometry, add smoothness) | Without labels the best they can do is stay near SapBERT plus smoothing, which A1 already does exactly and without training. Worth considering only inside R3, as a regulariser on the adapter. |
| More hard-negative mining under pseudo-pairs | U2-hard gave no gain. With pseudo-positives, hard negatives near the top of SapBERT's ranking are often *true* matches (false negatives). The full softmax of R3 already supplies hard negatives, with pseudo-seeds masked. |
| Relation-aware GNNs (R-GCN), graph transformers | There is effectively one relation type (`subClassOf`, plus NCIT's intersection parents). Directionality, the only useful distinction, is already in A1 / the relational adapter. Too little signal to train attention without labels, and no cross-graph edges to attend over. |
| Anchor-based structural re-ranking (LogMap / BERTMap-style extension) | The Sprint 6.2 probe failed its gate with *gold* anchors (ratio 1.24 < 1.5). Pseudo-anchors would be weaker. Keep it as a mapping-repair step for a supervised system, not for this question. |
| Changing the evaluation (candidate pools, train-label filter, tuned threshold) | It changes absolute numbers for every row, SapBERT included, and the filter reads gold train labels. It does not make structure more useful. Keep the 9.2 evaluation fixed for comparability. |

---

## 4. Ranked recommendation

| Rank | Experiment | Supervision | Cost | Expected outcome |
|---|---|---|---|---|
| 1 | **R1: A1 under the unsupervised protocol** (λ curve, label-free λ, rewired control) | none (graph + text) | ≈ 0.5 day, no training | Beats SapBERT by ≈ +0.045 MRR and +0.047 F1. Already measured, so this is a protocol-clean reproduction plus the controls the paper needs. |
| 2 | **R3: pseudo-seed residual GCN adapter** from A1 mutual pairs (+ random-seed and no-floor controls) | pseudo-pairs | ≈ 1.5 days incl. code, 3 seeds | At or slightly above A1. It is the decisive test of whether supervision is the bottleneck. |
| 3 | **R2: score fusion** of SapBERT / A1 with U2-I, w ∈ {0.25, 0.5} | none (uses existing models) | ≈ 0.5 day | Small or no gain over A1. Cheap evidence on whether the trained GCN holds anything beyond smoothing. |

**Bottom line.** Graph structure can beat frozen SapBERT without gold, but through a *training-free residual*, not through a trained
fusion on SapBERT's own pseudo-pairs. The expected margin is a few points (≈ +0.045 MRR). Exceeding SapBERT by a large margin, like the
+0.19 MRR of the supervised fusion A19, is **unlikely without changing the supervision signal**: either gold seeds (a few hundred
already match A1, all 1,406 give the large jump) or an external alignment resource. Note that SapBERT's UMLS pretraining already
plays that role partly; this needs to be disclosed with any SapBERT row. Neither is allowed in the current unsupervised protocol.
The evaluation setup should stay as it is.

---

## 5. Protocol notes for running these

- R1–R3 must use only `ncit_doid_unsup` artifacts, with `held_out_policy: none` training data where training data is needed.
  A1 needs a `training_data` run only because `gcl_train` loads one; the sampler output is unused when there are 0 parameters.
- λ = 0.25 / directional came from a gold-val selection in Sprint 5.1. Report it with that disclosure, or use the label-free rule
  as the headline and show 0.25 on the curve.
- Fix every grid (λ, w, q, rounds) before the run and report all points: one test read per configuration, as in 9.2/9.3.
- R3 changes what `infonce_full` reads, so the guard needs a reviewed exception for a pseudo-seed file. It must stay refused for
  `alignments.parquet`.

## References

- Liu et al., *Self-Alignment Pretraining for Biomedical Entity Representations* (SapBERT), NAACL 2021.
- Klicpera, Bojchevski, Günnemann, *Predict then Propagate: Graph Neural Networks meet Personalized PageRank* (APPNP), ICLR 2019.
- Huang et al., *Combining Label Propagation and Simple Models Out-performs Graph Neural Networks* (Correct & Smooth), ICLR 2021.
- Li, Han, Wu, *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning* (over-smoothing), AAAI 2018.
- Conneau et al., *Word Translation Without Parallel Data* (CSLS, unsupervised model selection), ICLR 2018.
- Artetxe, Labaka, Agirre, *A robust self-learning method for fully unsupervised cross-lingual mappings of word embeddings*, ACL 2018.
- Sun et al., *Bootstrapping Entity Alignment with Knowledge Graph Embedding* (BootEA), IJCAI 2018.
- Liu et al., *SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs*, WWW 2022.
- He et al., *BERTMap: A BERT-based Ontology Alignment System*, AAAI 2022; He et al., *Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching* (Bio-ML), ISWC 2022.
- Jiménez-Ruiz, Cuenca Grau, *LogMap: Logic-based and Scalable Ontology Matching*, ISWC 2011.
- Repo: [sapbert_residual_graph_adapter.md](sapbert_residual_graph_adapter.md), execution logs of Sprints 5.1, 5.5, 6.2, 6.3, 8.2, 9.2, 9.3.
