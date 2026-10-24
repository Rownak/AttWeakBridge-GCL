# Sprint 6: Closing the global-ranking gap with fusion, anchor re-ranking and more supervision

## Objective

Sprint 5 beat SapBERT on held-out NCIT–DOID gold pairs (`gold_alignment`) with a frozen-SapBERT residual adapter. Its best
trained variants still sit well below the target of about **MRR 0.62 and Hits@10 0.86** in global ranking. This sprint tries
three cheap, evidence-backed levers, each a separate phase with its own go/stop decision, without changing the generic
design (frozen SapBERT, graph structure, no ontology-specific code):

| Phase | Lever | Why it might help | Evidence it rests on |
|---|---|---|---|
| 6.1 | Score-level fusion of A11 and A12, and seed ensembles | A11 rarely breaks SapBERT-correct queries; A12 fixes the most. Ensembling also averages seed noise. | Sprint 5.5/5.6: A11 breaks 98 and fixes 281; A12 breaks 412 and fixes 627 (test, global) |
| 6.2 | Anchor-based re-ranker over the top-K (design §8) | Uses the only cross-graph structural signal not yet used: known gold pairs near the query | Design §2: 85.9% of test pairs have a train anchor within 2 hops; A12 leaves about 620 of 901 "wrong, near" queries unfixed |
| 6.3 | More supervision for the final model (train + val), plus a data-scaling curve | The train/val gap (train MRR 0.99, val 0.52) points at data, not λ | Sprint 5.2–5.5 overfitting; val is 211 pairs (+18% over the 1,174 used for training) |

**Open question for the developer (answer before reading the results against the target).** The 0.62 / 0.86 figures may come
from candidate ranking (the dataset's own 101-candidate lists) or from global ranking. The Sprint 5 candidate-mode test
numbers of A12 (MRR 0.621, Hits@10 0.903) and A15 (0.607, 0.905) already reach that level. In global mode the best is A12 at
0.536 / 0.773. Every table in this sprint reports both modes. Whether the published model uses the same test pairs is also
unverified, so treat the target as a reference point, not a like-for-like bar.

### Bars to beat (test, 3,280 pairs, mean over 3 seeds where trained; from Sprints 4 and 5)

| Bar | global MRR | global Hits@10 | candidates MRR | candidates Hits@10 |
|---|---|---|---|---|
| A0 SapBERT | 0.4475 | 0.6537 | 0.5193 | 0.7784 |
| A3 MLP, no edges (Sprint 5 selection) | 0.4957 | 0.7295 | 0.5687 | 0.8450 |
| A11 GCN + SapBERT floor | 0.5234 | 0.7708 | 0.5852 | 0.8710 |
| A12 GCN, no floor | 0.5361 | 0.7725 | 0.6209 | 0.9034 |
| A14 GAT + SapBERT floor | 0.5279 | 0.7774 | 0.5906 | 0.8732 |
| A15 GAT, no floor | 0.5289 | 0.7849 | 0.6065 | 0.9047 |
| Reference target (protocol unknown) | about 0.62 | about 0.86 | about 0.62 | about 0.86 |

Background: [sprint_5.md](sprint_5.md), [execution_logs/sprint_5_4.md](../execution_logs/sprint_5_4.md) to
[sprint_5_6.md](../execution_logs/sprint_5_6.md), [design_docs/sapbert_residual_graph_adapter.md](../design_docs/sapbert_residual_graph_adapter.md)
(§8 re-ranker), [design_docs/learning_lambda_or_fusion.md](../design_docs/learning_lambda_or_fusion.md).

## What already exists (reuse, do not rebuild)

| Piece | Where | Use in Sprint 6 |
|---|---|---|
| `ncit_doid_srga` lineage: ingest, three embedding stores, `training_data` run, val/test splits | Sprint 5.0 | Unchanged for 6.1 and 6.2 |
| The Sprint 5 trained models, 3 seeds each: A3, A11, A12, A14, A15 (first stages) and A4, A5, A6, A13, A16 (controls), plus the A1 grid | `models/gcl/ncit_doid_srga/` | The first stage for 6.1 and 6.2: **no new first-stage training until 6.3** |
| `gold_alignment` evaluator, `gcl_run` and `embedding_store` method kinds | `pipelines/evaluate/` | 6.1 adds one method kind; 6.2 adds one evaluator kind |
| `infonce_full` trainer path, `residual_floor`, `edge_rewire` | `gcl_train/` | 6.3 retrains with the frozen recipes |
| Analysis scripts: `srga_paired_ranks.py`, `srga_stratified.py`, `srga_pool_diagnostic.py` | `scripts/` | Reused for every phase |

## Fixed choices

| Choice | Value | Reason |
|---|---|---|
| Seeds | 42 / 43 / 44 for every trained model (as in Sprint 5) | Continuity. Ensembles use all three. |
| Hyperparameters of the first stage | Frozen as in Sprint 5: A11 and A14 at lambda 0.25 and 50 epochs, A12 at 200 epochs, A15 at 100 epochs, tau 0.05 | No re-tuning on test, and no new val selections of first-stage models |
| Model selection | On `val` (211 pairs, global mode only). Prefer the simpler configuration when two are within noise (a paired bootstrap CI that includes 0). | Val is small; Phase 5.4 showed it cannot resolve 0.01–0.02 MRR, so a CI over queries is reported with every val comparison |
| Test discipline | Each phase lists below **exactly which groups may be read on test**. Pass only those groups to `evaluate` and to every script run with `--split test`. | In Phases 5.5 and 5.6 a diagnostic script printed an unplanned test number (A13, A16). Both were disclosed, but the rule is now explicit. |
| Hits@K | `ks: [1, 5, 10]`, both modes on test, global only on val | Same as Sprint 5 |
| New numbering | A17 onwards; A0–A16 are the Sprint 5 variants | Continuity |

## Conventions for every phase

- Google-style docstrings, tests in `tests/unit/`, `uv run pytest tests/unit -q` green.
- Each phase ends with `dev_docs/execution_logs/sprint_6_N.md`:
  - architecture or method description and equations;
  - results tables (val, then test where allowed) with run ids;
  - the go/stop decision and caveats.
- Commit prefix `[Sprint-6.N]`.
- Every new option is **default-off**, so earlier runs, configs and the Sprint 4 and 5 artifacts are unchanged.
- Small floats in `--set` are written `0.0001`, not `1e-4`.

## Decision gates (summary)

| After | Go if | Else |
|---|---|---|
| 6.1 | The best fusion beats the best single member on val MRR with a paired bootstrap CI (1,000 resamples over queries) that excludes 0 | Report the null result. The best single variant (A12, or A11 for safety) stays the Sprint 6 first stage. |
| 6.2 probe | Anchor agreement fires for the gold target at least 1.5 times as often as for the first-stage top wrong candidate, on at least 20% of val queries | Stop 6.2 before any fitting and record the probe. |
| 6.2 fit | A21 beats the no-anchor control A23 on cross-validated val MRR (CI excludes 0), **and** A24 (shuffled anchors) is below A21 | Stop and report: re-ranking helps only as learned score fusion, or not at all. |
| 6.3 | Full-data models beat their train-only counterparts on test by more than the larger seed std | Report that data is not the binding limit at this size; keep the train-only models as final. |

---

## Phase 6.1: Score-level fusion of A11 and A12, and seed ensembles (A17–A20)

**Capability:** a `fusion` method kind that combines trained exports at the score level with no new training, so a seed
ensemble and a heterogeneous fusion are scored by the unchanged `gold_alignment` evaluator and every script.

**Method.** Each member's vectors are L2-normalised, scaled by the square root of its weight, and concatenated:

```
z_n = [ sqrt(w_1) · ẑ_n^(1) ; sqrt(w_2) · ẑ_n^(2) ; ... ]          ẑ^(i) = z^(i) / ||z^(i)||
z_a · z_d = Σ_i w_i · cos(z_a^(i), z_d^(i)),     ||z_n||² = Σ_i w_i
=>  cos(z_a, z_d) = Σ_i w_i · cos_i(a, d) / Σ_i w_i                (exactly the weighted mean of the members' cosines)
```

- [x] `evaluate/methods.py`: `MethodSpec` accepts `kind: fusion` with `members: [{kind, model_id | embedder_id, ref, weight}]`
      (weights `>= 0`, not all zero; zero-weight members are dropped).
  - The loader resolves every member through the existing path (so each is recorded in `manifest.inputs`).
  - Members are restricted to the nodes they all share; an empty intersection raises a `ConfigError`.
  - The fused matrix is an in-memory `AlignedMatrix`. `method_run_id` records the member run ids joined with `+`.
- [x] Variants (all val first, 3-seed members):
  - **A17** seed ensemble of A11 (equal weights over seeds 42/43/44).
  - **A18** seed ensemble of A12 (same).
  - **A19** fusion `w · A18 + (1 − w) · A17`, `w ∈ {0, 0.25, 0.5, 0.75, 1}` (`w = 0` equals A17 and `w = 1` equals A18: the floor check).
  - **A20** grand ensemble: A11, A12, A14 and A15, three seeds each, equal weights (12 members).
  - Experiment `configs/experiments/ncit_doid_srga_fusion.yaml` (A0, A11, A12, A17–A20; `gold_val` only) and the fusion
    grid for A19 in a second file.
- [x] Analysis (`scripts/`): a paired bootstrap over queries (1,000 resamples) for the MRR difference between two variants,
      printed with the paired wins/losses of `srga_paired_ranks.py`. Compare each of A17–A20 against its best single member
      (single members use the mean over seeds) and against A0.
- [x] Selection on val (rule above): the best of A17–A20 by val MRR, preferring fewer members inside the CI.
- [x] Test, once: **the selected fusion, A17 and A18.** The Sprint 5 single-variant rows already exist and are reused.
  - Run `scripts/srga_stratified.py` and `scripts/srga_pool_diagnostic.py` for the selected fusion.
- [x] Tests (`test_methods.py`):
  - the fused cosine equals the weighted mean of member cosines on a 3-member fixture;
  - one member equals itself, zero weights drop a member, and a zero-sum or negative weight raises;
  - mismatched node sets use the intersection and an empty intersection raises;
  - `method_run_id` joins the member run ids; the `embedding_store` and `gcl_run` paths are unchanged.

**Training objective / negatives:** none (no training).
**Ablation:**
- A17 and A18 vs a single seed (does averaging seeds help?).
- A19 vs A17 and A18 (is A11 + A12 complementary?).
- A20 vs A19 (do the GAT variants add anything to the GCN ones?).

**Acceptance:**
- a val table (MRR, Hits@1/5/10) and paired counts with CIs for A0, A11, A12, A17–A20, and the A19 weight curve;
- a test table for the selected fusion, A17 and A18 in both modes;
- the stratified fixed/broken report and the held-out-pool diagnostic for the selected fusion;
- the rate of broken SapBERT-correct queries, since A11 and A12 differ most there.

**Decision:** gate 6.1.

---

## Phase 6.2: Anchor-based re-ranker over the first-stage top-K (A21–A24)

**Capability:** a second stage that reorders the first stage's top-K candidates with **anchor-consistency features**
that no node embedding can express. NCIT and DOID are disconnected, so structure is comparable across them only through
known gold pairs (anchors). This is design §8. The first stage is the Phase 6.1 selection (or A12 if 6.1 fails).

**Anchors.** Gold **train** pairs only (1,174 usable). Val and test pairs are never anchors, for fitting and for evaluation alike,
so no query's own pair or any held-out pair informs its features. Hops are taken in each ontology's own graph over the ingest
edges, stored child to parent, so parents and children are distinguished.

**Features** for a query source `n` and candidate `d` (`P`, `C` = parents and children; `anchor(x)` = x's train partner, if any):

| Group | Feature |
|---|---|
| First-stage scores | `cos` from SapBERT (A0), from A11, from A12, the first-stage score, and its gap to the top-1 score |
| Anchor agreement (structural) | parent agreement `|{a in P(n) : anchor(a) in P(d)}|`; child agreement `|{a in C(n) : anchor(a) in C(d)}|`; 2-hop agreement: share of anchors within 2 hops of `n` whose partner is within 2 hops of `d` |
| Granularity mismatch | some `a in P(n)` or `C(n)` has `anchor(a) = d` (the neighbour of `n` is already matched to `d`, so `d` is probably a parent or child of `n`'s true match, not the match) |
| Exclusivity | `d` is already the train partner of a different source |

The exclusivity group is **separated and ablated** (A22). The split is built so that test targets are almost never train targets
(23 of 1,406 train pairs touch a test endpoint), so an "already aligned, so not the answer" feature can exploit the split and not just the
data. It is a legitimate one-to-one constraint in a real matching task, but it must not be what carries the result.

**Model.** A listwise linear model on at most about 12 standardised features, no new dependency (`scikit-learn` is already installed
but a few lines of `torch` or `numpy` suffice):

```
s_i = w · f_i                         i over the top-K candidates of the query (gold included only if it is in the top-K)
L   = − mean_queries log softmax(s)[gold]  +  l2 · ||w||²
```

Candidates beyond rank K keep their first-stage order after the re-ranked top-K, and a query whose gold target is outside the
top-K is unchanged. The weights are fitted on **val** (211 queries). The first-stage models never trained on val, so their val
scores are unbiased (train-pair scores are not, which is why train pairs are not used for fitting).

- [x] `evaluate/rerank.py`: feature extraction, listwise fit and apply. The evaluator resources gain the **directed** edge list
      (additive; the existing undirected `res.graph` has no parent/child distinction).
- [x] `scripts/srga_rerank_fit.py`: 5-fold cross-validation grouped by source node over val, to choose `K in {10, 20, 50}` and
      `l2 in {0.01, 0.1, 1}`, then a final fit on all val; it writes `artifacts/rerankers/<dataset>/<name>.json` (weights, feature
      names and standardisation, K, l2, first-stage run ids) and prints the weights.
- [x] New evaluator kind `gold_alignment_rerank` (same metrics and modes as `gold_alignment`): params name the first-stage
      method, the re-ranker file and the split; it emits rows for the re-ranked method.
- [x] **Probe first**, before any fitting (analysis only, val and test coverage):
  - what share of queries has an anchor within 1 and 2 hops (the design measured 85.9% and 63.7% at 2 hops on test);
  - among queries with the gold target in the top-K, how often agreement fires for the gold target against the first-stage top
    wrong candidate.
  - Apply the probe gate above.
- [ ] Variants, all selected by CV on val and fitted on val:  *(not run: probe gate STOP, see sprint_6_2.md)*
  - **A21** full re-ranker (all feature groups).
  - **A22** A21 without the exclusivity group.
  - **A23** control: first-stage scores only, no anchors (learned score fusion).
  - **A24** control: A21 with the anchors' DOID partners shuffled among anchors (same endpoints, correspondences destroyed).
- [ ] Test, once: **A21, A22 and A23.** A24 is val only.  *(not run: probe gate STOP, see sprint_6_2.md)*
  - Compare against the first stage and the Sprint 5 bars.
  - Run `scripts/srga_stratified.py`, with attention to the "SapBERT wrong, confuser near" stratum.
- [x] Tests (`test_rerank.py`):
  - the feature values on a hand-built 2-graph fixture with known anchors (parent agreement, child agreement, granularity mismatch, exclusivity);
  - anchors exclude val and test pairs, and a query's own pair is never an anchor;
  - the re-ranked order beyond K is unchanged and a gold-outside-top-K query is unchanged;
  - the listwise fit recovers a planted weight on synthetic data, and the apply step reorders as the weights dictate;
  - the shuffled-anchor control keeps the endpoints and breaks the correspondence.

**Training objective / negatives:** the first stage is untouched. The re-ranker sees the top-K of the first stage as its negatives (hard
negatives by construction).
**Ablation:**
- **A21 vs A23:** do anchors add beyond learned score fusion? This is the question of the phase.
- **A21 vs A22:** is the gain structural, or is it the exclusivity prior?
- **A21 vs A24:** is it the true correspondence, or just the feature distribution?

**Acceptance:**
- the probe table;
- the cross-validated val table of A21–A24 with a paired bootstrap CI for A21 vs A23 and A21 vs A24;
- the fitted weights (a handful of interpretable numbers);
- a test table for A21, A22 and A23, with the stratified breakdown;
- the cost: wins, losses and breaks against the first stage.

**Decision:** probe gate, then fit gate (above).

---

## Phase 6.3: More supervision for the final model, and a data-scaling curve (A25)

**Capability:** retrain the frozen recipes on **train + val** (all 1,406 gold train pairs) in a new lineage, and measure how
performance scales with the number of training pairs. The test pairs and the embedding stores are unchanged, so the
comparison against the train-only models is paired on identical queries.

- [x] **Scaling curve (val, no new lineage).** `gcl_train.train.pair_fraction` and `pair_seed` (default `1.0`; default-off):
  - sample that fraction of distinct source nodes among the positive pairs and keep all of their pairs;
  - the unsampled pairs stay in the gold mask, so they are never scored as false negatives; their endpoints remain ordinary
    candidates;
  - sizes `{25%, 50%, 75%, 100%}` for the **A11 and A12 recipes**, 3 seeds each (24 runs, a few minutes);
  - report val MRR and Hits@10 against pair count (mean ± std) and a straight-line fit on log(pairs), which extrapolates the
    expected gain from +18% data.
- [x] **Full-data lineage `ncit_doid_srga_full`:**
  - a dataset config like `ncit_doid_srga` with no `val_split` (so all 1,406 train pairs train and the test split is unchanged), under its
    own id so every Sprint 4 and 5 artifact and `LATEST` pointer is untouched (the reasoning of Phases 4.0 and 5.0);
  - copy the three embedding stores' `vectors.npy`, `index.parquet`, `status.parquet`, `manifest.json`, `stats.json` and `LATEST`
    **without the `shards/` directory** (about 24,000 small files; the Sprint 5.0 copy of them took about 30 minutes) and re-verify `node_id` order and `text_sha256`;
  - ingest, then `training_data` (supervised sampler).
- [x] **A25 retraining** with the frozen recipes and 3 seeds: A11, A12 (and A3, as the Sprint 5 selection).
      If Phase 6.1 selected a fusion, build the same fusion from the full-data members.
      There is no val readout (val is now training data) and no new selection.
- [x] Test, once: **A25 for A3, A11, A12 and the selected fusion, if any.**
  - Compare against the train-only counterparts with per-query paired wins and losses (extend `srga_paired_ranks.py` so a group can name a
    different lineage).
  - A0 test must equal the Sprint 5 value exactly (stores and test pairs unchanged).
- [x] If Phase 6.2 passed, apply its val-fitted weights to the full-data first stage as one extra row, flagged as slightly  *(not applicable: Phase 6.2 stopped at its probe gate, so there are no weights)*
      shifted: the weights were learned on scores from train-only models.
- [x] Tests:
  - `pair_fraction` is deterministic for a seed, moves whole sources, leaves the gold mask intact and defaults to the old behaviour;
  - an `infonce_full` run with `pair_fraction < 1` completes and uses fewer positive pairs;
  - the full-data config loads and has no `val_split`.

**Training objective / negatives:** unchanged from Sprints 5.2–5.6 (full-softmax InfoNCE over every non-held-out candidate, gold mask of any split).
**Ablation:** A25 vs the train-only models (+18% pairs) and against the scaling curve's prediction.

**Acceptance:**
- the scaling table and fitted slope;
- a test table of A25 against the train-only counterparts, with paired counts and the per-seed spread;
- the A0 test numbers identical to Sprint 5;
- the data-size conclusion stated plainly.

**Decision:** gate 6.3.

---

## Out of scope (separate sprint if Sprint 6 does not reach the target)

- **Adapting the text encoder** (LoRA or a light fine-tune of SapBERT on ontology relations). It is likely the largest remaining lever for global ranking, but it
  changes the frozen-text design and the cost profile.
- **A learned lambda or a per-node confidence gate** (design A8). Sprint 5 found no support for it (the lambda curve is flat from 0.25 to 1, and a gate over node features cannot see the candidate list).
- **DeepWalk as an adapter input**, directional graph adapters beyond A4, and the Sprint 5 controls.
- **K-fold or cross-fitting over train pairs** as a replacement for the small val split. It would give more evaluation data and unbiased train-pair scores for the re-ranker, but it needs a per-fold
  ingest and a cross-fit loop.
- **Weak supervision from ontology synonyms**, pseudo-labelling or self-training.
- **Changes to the ingest adapters, the embedding pipeline, or the `gold_alignment` evaluator itself.** 6.2 adds a *new* evaluator kind and reads, but does not change, the existing one.

## Caveats for the write-up

- **Val is small.** Fusion weights and the re-ranker are chosen on 211 pairs. Every val comparison carries a paired bootstrap CI; the coarse grids above are deliberate.
- **The re-ranker is fitted on val.** Val therefore has no clean readout for A21 beyond cross-validation, and the single test readout is the real check. A fit on cross-fitted train-pair
  scores (see out of scope) would be larger and cleaner.
- **Ensembling and fusion add no information to the training data.** They reduce variance and combine complementary errors. A gain much below the gap to the target is expected, and a null
  result would still be informative about where the gap comes from.
- **Training and re-ranking are transductive** where they use the graph: val and test nodes take part in message passing and in anchor lookups as features only, never in a loss term or as anchors.
- **The published reference model differs in kind** (a BERT fine-tuned on ontology relations). Matching its numbers is a target, not a claim of equivalence.
