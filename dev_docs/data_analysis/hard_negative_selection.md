# Hard-negative selection for the supervised `ncit_doid` GCN

**Context:** Sprint 4 (Phases 4.4–4.5): the supervised GCN beats the unsupervised GCN but reaches only about half of
SapBERT's held-out MRR (0.223–0.227 vs 0.448, `global` mode). This note measures *why*, then compares ways to choose
harder negatives, plus the non-negative changes that are probably also needed to beat SapBERT.

All numbers below come from the Sprint 4 artifacts:
- ingest `20261014T185110Z-c44babf2`;
- SapBERT store `20261009T182919Z-95011340` (`label_synonyms_definition`, mean-pooled);
- `gcn-supervised` `20261014T195655Z-fcc54097`;
- supervised examples `20261014T195623Z-d97cb67f`.

They were computed in memory, with no new artifacts. "Train pairs" means the 1,383 usable gold train pairs, and the
"candidate pool" is the 5,195 non-held-out DOID nodes the sampler can draw from.

---

## 1. Diagnosis: the current negatives carry almost no ranking signal

| Measurement | Value | Meaning |
|---|---|---|
| SapBERT cos(anchor, gold positive), mean | 0.645 | |
| SapBERT cos(anchor, current text negative), mean | **0.054** | The negatives are ~0.6 cosine below the positive, so they are trivially separable. |
| Distinct DOID nodes used as negatives | 520 of 6,915 rows | The same far-away nodes recur (for example "childhood kidney neoplasm" 380×). |
| GCN cos(anchor, pos) / cos(anchor, neg) after training | 0.848 / −0.375 | The model fits the triplets easily. |
| Triplets with non-zero loss after training | **4.8%** | 95% of the training signal is gone. |
| SapBERT median cos of the **1st** most-similar non-gold DOID | **0.670** | The real competitors are *more* similar than the gold positive (0.655). |
| Train pairs where a non-gold DOID outranks the gold target (SapBERT) | **791 / 1,383** (57%) | The hard negatives exist and are plentiful. |

**The key test is whether the GCN fits its own training pairs.** Retrieval on the *train* pairs (`global` mode, all
DOID):

| method | train MRR | train Hits@1 | test MRR |
|---|---|---|---|
| SapBERT | 0.435 | 0.322 | 0.448 |
| GCN supervised | **0.267** | 0.166 | 0.223 |

The GCN is worse than SapBERT **even on the pairs it was trained on**. This is under-fitting the ranking task, not
over-fitting. The loss teaches "the gold target is closer than five random far-away nodes", which says almost nothing
about the thousands of close competitors that decide the rank. The near-zero triplet loss confirms it.

**What the competitors look like.** These are SapBERT's top non-gold DOID nodes that rank above the gold target
(train pairs, random sample):

| NCIT anchor | gold DOID | SapBERT's top confuser | cos gold / confuser |
|---|---|---|---|
| Nephritis | nephritis | chronic pyelonephritis | 0.341 / 0.636 |
| Heart Disorder | heart disease | heart cancer | 0.285 / 0.672 |
| Benign Prostatic Hyperplasia | prostatic hypertrophy | prostate cancer | 0.487 / 0.692 |
| Labyrinthitis | otitis interna | viral labyrinthitis | 0.791 / 0.844 |
| Cystadenofibroma | cystadenofibroma | rete ovarii cystadenofibroma | 0.596 / 0.743 |
| Scabies | scabies | contagious pustular dermatitis | 0.566 / 0.693 |

Two patterns stand out:

1. **Granularity confusers.** These are a parent, child or sibling concept, such as "nephritis" vs "chronic
   pyelonephritis". In the DOID graph, the top confuser is within 1 hop of the gold target for 108 of the 791 cases,
   2 hops for 144, 3 hops for 85, and further away for 454. Within the SapBERT top-5 negatives, 21% are within 2 hops
   of the gold target.
2. **Lexically similar but structurally distant confusers** (57% are more than 3 hops away). These are the cases
   where the **graph view could disambiguate** what text alone cannot, which is the argument for a GCL model beating
   SapBERT at all.

**False-negative risk is low but not zero.** Only 1 train anchor has a non-gold DOID node with exactly its label.
Close confusers are mostly real non-equivalences (parent/child), but the OAEI gold set may not be exhaustive, so some
filtering is still advisable (Section 3, F).

**The input text weakens the bar.** Exact-label pairs score only 0.34–0.57 cosine ("Nephritis"/"nephritis" 0.341,
"Scabies"/"scabies" 0.566). The store embeds `label + synonyms + definition` with mean pooling, while SapBERT is
designed for **short names with `[CLS]` pooling**. Definitions differ across the two ontologies and dilute the name
signal. This affects both the baseline and the GCN's input (Section 4, P2).

---

## 2. Methods for choosing hard negatives

Each method plugs into the existing `supervised` sampler (`training_data/samplers.py`) unless stated otherwise. Each
keeps the Sprint 4 invariants: gold targets of every split are excluded, held-out nodes cannot be candidates, and one
example row is written per negative.

### H1. Static SapBERT top-k ("hardest")
- **How:** take the **top**-k of the anchor's `text_g1_g2` row instead of the bottom-k, using the same
  `_topk_bottomk_indices(row, k, exclude_indices=gold_cols)` call. This is a one-word change.
- **Pros:** trivial to build, deterministic, and targets exactly the competitors measured above (2,223 distinct
  negatives for top-5, vs 520 now).
- **Cons:**
  - the highest false-negative risk;
  - with only hardest negatives, triplet training can collapse or stall;
  - the negatives are fixed to SapBERT's view, so the model may only learn to undo SapBERT's specific mistakes.

### H2. Rank-band / semi-hard sampling (recommended first)
- **How:** skip the top `r` ranks (for example `r=2`), then sample `n` negatives from ranks `r+1 … R` (for example
  `R=50`) with the seeded RNG. Alternatively, define a similarity band relative to the positive, such as
  `cos(a, p) − 0.3 < cos(a, n) < cos(a, p) + 0.05`.
- **Pros:**
  - still hard: the median cos at rank 10 is 0.569 and at rank 50 is 0.511, versus 0.054 now;
  - lower false-negative risk than H1;
  - the band gives more diversity across epochs if re-sampled.
- **Cons:** two new parameters (`r`, `R`).
- **Config shape:** `negative_strategy: band`, `band: [3, 50]`, with `negatives_per_positive` kept as is.

### H3. Mixed hard + easy (curriculum)
- **How:** for example 3 negatives from H2 and 2 random negatives (not bottom-k). Optionally start mostly easy and
  increase the hard share over epochs.
- **Pros:** the easy negatives keep the global geometry stable while the hard ones fix local rank; this is a
  standard remedy for the collapse risk of H1.
- **Cons:** a curriculum needs per-epoch example changes in `gcl_train`, unless it is approximated with two
  `training_data` runs.

### H4. Structure-aware (hierarchy) negatives
- **How:** use DOID nodes within 1–2 hops of the **gold target** (parents, children, siblings), excluding gold and
  held-out nodes. The median is 4 such nodes per anchor (mean 9.9), and 76 anchors have none (fall back to H2).
- **Pros:**
  - teaches *granularity* ("nephritis" ≠ "chronic pyelonephritis"), the confuser type SapBERT is worst at;
  - uses graph structure, which the GCN can exploit and SapBERT cannot.
- **Cons:**
  - parent/child pairs are the ones most likely to be partial matches in other benchmarks;
  - the hierarchy is thinned by held-out test targets.
- **Tip:** combine with H2 (for example 2 hierarchy + 3 band).

### H5. Lexical-overlap negatives (mimic the OAEI candidate generation)
- **How:** rank DOID nodes by token or idf overlap with the anchor's label/synonyms (BM25 or a simple idf-weighted
  Jaccard), and take the top non-gold ones.
- **Pros:**
  - matches how the `test.cands` pools were built, so it trains for exactly the `candidates`-mode distribution;
  - independent of SapBERT, so it covers SapBERT's blind spots.
- **Cons:** needs a small tokenizer/idf helper, which is new code.
- **Constraint:** **never use `test.cands.tsv` itself** for training. Those pools belong to test pairs, and their
  endpoints are held out.

### H6. Dynamic, model-mined negatives (ANCE-style)
- **How:** every `m` epochs (for example 10), recompute the current GCN embeddings and re-mine each anchor's top-k
  non-gold DOID nodes in **model** space, then continue training on them.
- **Pros:**
  - always targets the model's own current mistakes, so the loss never goes to zero the way it does now;
  - training is full-batch, so all embeddings are already computed every epoch and the extra cost is one
    5,195-column similarity per anchor.
- **Cons:**
  - needs a `gcl_train` change: the negatives are mined inside the training loop, so `examples.parquet` would only
    hold positives and graph pairs;
  - non-stationary, so it needs H7's false-negative filter and a warm-up on H2 negatives.

### H7. Many negatives at once: full-softmax InfoNCE over DOID
- **How:** for each anchor, compute softmax cross-entropy of the gold target against **all** non-held-out DOID nodes
  (5,195), with the other gold targets masked out. That is up to 1,383 × 5,195 logits, which fits easily in
  full-batch training.
  - A lighter variant needs no code at all: the existing registered `infonce` loss (in-batch negatives) with H2
    negatives in `examples.parquet`.
- **Pros:**
  - it directly optimizes the quantity being evaluated (the rank among all DOID nodes);
  - every hard negative is included automatically, with no mining parameters;
  - the temperature controls hardness.
- **Cons:**
  - the full version is a new loss and a trainer change, because negatives no longer come from
    `examples.parquet`;
  - needs a false-negative mask.
- **Expected effect:** in retrieval and entity linking this is typically the strongest single change, and it
  subsumes H1/H6.

### F. False-negative safeguards (apply with any of H1–H7)
- **Already in place:** exclude gold targets of every split.
- **Exclude exact or normalized-label matches** and shared synonyms (cheap and safe; 1 case found).
- **Similarity ceiling:** drop candidates with `cos(a, n) > cos(a, p) + δ` (for example δ=0.1), or above an absolute
  value (for example 0.95).
- **Skip the top `r` ranks** (H2).
- Optionally, down-weight hierarchy-adjacent negatives rather than dropping them.

### Loss settings that must change with hard negatives
- **Triplet margin.** The current margin is `1.0` in cosine *distance*. That suits random negatives, but for hard
  negatives (cos gap of about 0–0.2) it is so large that almost every triplet stays active and gradients are
  dominated by the hardest, noisiest pairs. Use **0.1–0.3**, or switch to InfoNCE with a temperature of about
  0.05–0.1.
- **Monitoring.** Track the fraction of active triplets and a small *validation* MRR (Section 4, P3), not just the
  loss, which is misleading once negatives get harder.

---

## 3. Comparison

| Method | Hardness | False-negative risk | Code change | Cost | Expected impact |
|---|---|---|---|---|---|
| H1 static top-k | very high | high | 1 line (sampler) | none | medium; unstable alone |
| **H2 rank band** | high | medium | small (sampler + 2 params) | none | **medium–high** |
| H3 hard + easy | mixed | medium | small (sampler) | none | stabilizes H1/H2 |
| H4 hierarchy | targeted | medium (parent/child) | small (sampler, uses `graph_2` edges) | none | high on granularity errors |
| H5 lexical | targeted (`candidates` mode) | medium | new idf/BM25 helper | low | high for `candidates` mode |
| H6 model-mined | adaptive | medium (needs F) | `gcl_train` loop change | +~10% train time | high |
| **H7 full-softmax InfoNCE** | all negatives | low with mask | new loss + trainer | low (full batch) | **highest** |

---

## 4. Not about negatives, but probably required to beat SapBERT

Hard negatives fix the *signal*. They do not fix two structural handicaps the diagnosis exposed.

- **P1. Start from SapBERT, don't relearn it.** The `DualEncoder` maps `[fc1(text), GCN(text), fc2(struct),
  GCN(struct)]` through a randomly initialized `fc3` to 128 dimensions, so the model must rebuild SapBERT's similarity
  from 1,383 pairs, and it fails to (train MRR 0.267 < 0.435). Options, in order of effort:
  - **(a)** Fuse scores at evaluation time: `score = cos_SapBERT + λ · cos_GCN`. No training change, and λ is chosen
    on validation. This tests whether the GCN carries *complementary* (structural) signal.
  - **(b)** Use a residual output, `z = W·SapBERT + Δ(GNN)` with `W` initialized to (a projection of) the identity,
    so training starts at SapBERT's ranking and only learns corrections.
  - **(c)** Keep 768 dimensions or a higher output width.
- **P2. Fix the text input first.** Re-embed with SapBERT on `label` (or `label + synonyms`) using `[CLS]` pooling,
  which is SapBERT's intended use, and make **that** the baseline. It is likely to raise the SapBERT bar
  substantially (exact-label pairs currently score as low as 0.34). Beating a weakened baseline would not be a
  meaningful result.
- **P3. Add a validation split.** Hold out about 15% of the gold *train* pairs (as held-out nodes, like test) for model
  selection: band limits, margin or temperature, λ, epochs. Never tune on the 3,280 test pairs.

---

## 5. Suggested experiment order

Each step is one or a few runs, scored with `gold_alignment` (`global` + `candidates`) on validation, and on test only
for the final pick.

| # | Experiment | Change | Question answered |
|---|---|---|---|
| E0 | SapBERT `label` / `label+synonyms`, `[CLS]` | new embed run | What is the true text bar? (P2) |
| E1 | Score fusion `cos_SapBERT + λ·cos_GCN` (current model) | evaluator option | Does the GCN add complementary signal? (P1a) |
| E2 | H2 band negatives (`[3, 50]`, 5 per positive), triplet margin 0.2 | sampler + config | Do hard negatives fix the under-fitting? |
| E3 | E2 + H4 hierarchy (2 of the 5) | sampler | Does structure-aware hardness help granularity errors? |
| E4 | H7 full-softmax InfoNCE with the F mask | new loss | Best achievable with the current architecture |
| E5 | Best of E2–E4 + residual SapBERT output (P1b) | model change | Can the GCN beat SapBERT on its own? |
| E6 | E5 + H6 dynamic re-mining | trainer change | Remaining gains from adaptive negatives |

**Success criterion:** held-out test MRR above the **E0** SapBERT bar in both modes, with the gain coming from pairs
where SapBERT's top confuser is structurally far from the gold target (Section 1, pattern 2). A per-pair breakdown by
confuser hop distance would show whether the graph view is really doing the work.

---

## 6. Caveats

- **Sample size.** All diagnostics use train pairs (1,383), and the confuser examples are a random sample.
- **Hop distances.** These are measured on the DOID graph *with held-out test targets removed*, so true hierarchy
  neighbourhoods are somewhat larger.
- **Transductive training.** Test nodes are in the message-passing graph but never in examples. Hard negatives are
  drawn only from non-held-out nodes, so this does not change.
- **Untested gains.** The expected-impact ratings are judgement calls based on the measurements here and on
  common retrieval practice. None of H1–H7 has been run on this dataset yet.
