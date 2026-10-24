# Sprint 6, Phase 6.2: Anchor-based re-ranker over the first-stage top-K (A21–A24)

Plan: [sprint_6.md](../sprints/sprint_6.md). Previous phase: [6.1](sprint_6_1.md). Lineage `ncit_doid_srga`; val = 211 pairs, test = 3,280.
First stage: **A19** (0.5 · seed ensemble of A12 + 0.5 · seed ensemble of A11; Phase 6.1).

## 0. Outcome in brief

- **Probe gate: STOP.** The plan's rule is "anchor agreement fires for the gold target at least 1.5 times as often as for the first-stage top wrong candidate, on at least 20% of val queries". On val (K = 20, 188 of 211 queries have the gold target in the top-K):
  - agreement of any kind (parent, child or 2-hop) fires for the gold target on 55.9% of those queries and for the top wrong candidate on 45.2%: **ratio 1.24 (< 1.5)**, although it fires on 49.8% of val queries;
  - restricted to parent or child agreement the ratio is 2.06, but it fires for the gold target on only **16.6% of val queries (< 20%)**.
  - Neither reading passes both conditions.
- **So no re-ranker was fitted and nothing was read on test.** A21–A24 have no val or test numbers. The plan says to stop 6.2 before any fitting and record the probe, and that is what was done. The first stage for Phase 6.3 stays **A19**.
- **The machinery is built, tested and ready** (module, evaluator kind, fit and probe scripts, test config) in case the developer decides to override the gate. A zero-weight re-ranker run through the real pipeline on val reproduces A19's val numbers exactly (MRR 0.6278, Hits@1 0.5213), so the evaluator path is consistent with `gold_alignment`.
- The probe also shows the worry the plan raised about the exclusivity feature: it fires for the top wrong candidate on **25.5%** of val queries and for the gold target on **0%**. That is a property of the split (test and val targets are almost never train targets), not of the data, so any re-ranker that uses it would gain from the split.

## 1. Method (as implemented, not fitted)

All in [rerank.py](../../src/cga/pipelines/evaluate/rerank.py).

**Anchors.** Gold **train** pairs only (1,195 rows: 1,192 NCIT sources, 1,195 DOID targets; a few sources have two partners, so `partners` is set-valued). Val and test pairs are never anchors, so a query's own pair never is. (The plan quotes 1,174 usable pairs, which is the Sprint 5 training count; the module takes every train alignment row whose endpoints are a graph_1 and a graph_2 node.) Hops are taken in each ontology's own graph over the ingest edges, stored child → parent, so parents (`dst`) and children (`src`) are distinguished.

**Features** for a query source `n` and a candidate `d` in the first stage's top-K (10 in total):

| Group | Feature | Definition |
|---|---|---|
| scores | `cos_sapbert`, `cos_a11`, `cos_a12` | cosines from A0 and from the A17 / A18 seed ensembles of A11 / A12 |
| scores | `score_first`, `gap_top1` | A19 cosine and its gap to the query's top-1 A19 cosine |
| anchor | `parent_agree` | number of parents `a` of `n` whose anchor partner is a parent of `d` |
| anchor | `child_agree` | the same for children |
| anchor | `hop2_agree` | share of anchored nodes within 2 hops of `n` (undirected, `n` excluded) with a partner within 2 hops of `d` (`d` excluded); 0 if none |
| granularity | `granularity_mismatch` | some parent or child `a` of `n` has `d` among its anchor partners |
| exclusivity | `exclusive` | `d` is the train partner of a source other than `n` |

The `a11` / `a12` features are the seed ensembles, not single seeds, because those are the members of the first stage; `score_first` is therefore the mean of those two cosines, and ridge regularisation resolves the collinearity.

**Model.** `s_i = w · f_i` on features standardised over the fitting queries, `L = −mean_q log softmax(s)[gold] + l2 · ‖w‖²` over queries whose gold target is in the top-K, minimised with L-BFGS in `torch` (float64). Beyond the top-K the first-stage order is kept, and a query whose gold target is outside the top-K is unchanged. Ties inside the top-K keep the first-stage order.

**Selection protocol (written, not run).** `scripts/srga_rerank_fit.py`: 5-fold CV grouped by source node on val, `K ∈ {10, 20, 50}`, `l2 ∈ {0.01, 0.1, 1}`; the reported CV MRR re-selects `(K, l2)` inside each outer training fold (nested), so it is not optimistic. Paired bootstrap CIs (1,000 resamples) for A21 vs A22 / A23 / A24 / the first stage, and the fit gate. Variants: A21 all groups; A22 without exclusivity; A23 scores only; A24 = A21 on anchors whose DOID partners are cyclically shifted among the anchors (same endpoints, correspondences destroyed).

## 2. What was implemented

- [rerank.py](../../src/cga/pipelines/evaluate/rerank.py) (new): `Anchors`, `build_anchors` (with the shuffle control), `OntologyGraph`, `anchor_features`, `build_query_features`, `featurize_split` (uses the evaluator's own `_build_queries`, so queries and candidate lists are identical), `fit_listwise`, `gold_rank_after_rerank`, `reranked_top_ids`, `grouped_folds`, `select_config`, `nested_cv_ranks`, JSON save and load.
- [evaluators/gold_alignment_rerank.py](../../src/cga/pipelines/evaluate/evaluators/gold_alignment_rerank.py) (new): evaluator kind `gold_alignment_rerank`. Same metrics, queries and modes (`global`, `candidates`) as `gold_alignment`; params name the first-stage method, the three score methods and the re-ranker file. Rows carry `method_kind: rerank` and `method_run_id: <first stage run id>+rerank:<name>`. `gold_alignment` itself is unchanged.
- [pipeline.py](../../src/cga/pipelines/evaluate/pipeline.py) and the `EvalResources` protocol: additive `edges` (the directed edge rows). The existing undirected `graph` is untouched.
- Scripts (analysis only): [srga_rerank_probe.py](../../scripts/srga_rerank_probe.py) (run), [srga_rerank_fit.py](../../scripts/srga_rerank_fit.py) (not run on data), [srga_rerank_common.py](../../scripts/srga_rerank_common.py).
- Config [ncit_doid_srga_rerank_test.yaml](../../configs/experiments/ncit_doid_srga_rerank_test.yaml): the gated test readout (A19 reference, A21, A22, A23). Not run; its header says so. Its YAML loads, and the schema it uses was exercised by the zero-weight smoke run below.
- Tests (473 pass; 13 new in [test_rerank.py](../../tests/unit/pipelines/evaluate/test_rerank.py)):
  - feature values on a hand-built 2-graph fixture (parent agreement, child agreement, 2-hop share, granularity mismatch, exclusivity), and exclusivity ignoring the source's own anchor;
  - anchors exclude val and test pairs, and the query's own pair is never an anchor;
  - the shuffled control keeps the endpoints, shares no pair with the true anchors, and is deterministic;
  - the re-ranked order beyond K is unchanged; a gold target outside the top-K, or beyond K, keeps its first-stage rank; ties keep the first-stage order;
  - the listwise fit recovers a planted weight vector (correlation above 0.99 on 3,000 synthetic queries) and the apply step orders by the model's score;
  - a fit with no usable query keeps the first-stage order; grouped folds put each source in one fold; the CV helpers run;
  - evaluator: a zero-weight re-ranker equals `gold_alignment` on the first stage; a weighted run with shuffled anchors yields valid metrics; bad params raise `ConfigError`.

## 3. Probe results (val and test coverage; val-only firing)

Command: `uv run python scripts/srga_rerank_probe.py` (K = 20). No model is fitted and no test ranking is read: the test row is anchor coverage only.

**Coverage** (share of queries whose source has a train anchor among its neighbours):

| split | pairs | anchor within 1 hop | anchor within 2 hops |
|---|---|---|---|
| val | 211 | 0.336 | 0.834 |
| test | 3,280 | 0.362 | 0.824 |

The plan quotes 85.9% and 63.7% from design §2 (test); this probe measures 82.4% within 2 hops and 36.2% within 1 hop, using all 1,195 train rows and an undirected hop count. The two measures may have different definitions, which was not investigated.

**Firing on val**, 188 of 211 queries have the gold target in the first stage's top-20. "Top wrong" is the highest-ranked non-gold candidate.

| feature | fires for gold | fires for top wrong | ratio | share of all val queries where it fires for gold |
|---|---|---|---|---|
| any agreement (parent, child or 2-hop) | 0.559 | 0.452 | **1.24** | 0.498 |
| parent or child agreement | 0.186 | 0.090 | 2.06 | 0.166 |
| granularity mismatch | 0.000 | 0.053 | 0.00 | 0.000 |
| exclusive | 0.000 | 0.255 | 0.00 | 0.000 |

**Gate:** needs a ratio of at least 1.5 **and** a share of at least 0.20. Any agreement: 1.24 / 0.498 → fails the ratio. Parent or child: 2.06 / 0.166 → fails the share. **STOP.**

Reading:
- The 2-hop feature fires for the top wrong candidate nearly as often as for the gold target (the first stage's confusers are close to the gold target in the graph, so they inherit the same anchors). That is the "wrong, near" stratum from Phase 5.4, and it is where an anchor signal would need to discriminate.
- The 1-hop agreement is a real signal (2.1 times as frequent for the gold target), but it applies to only about 1 in 6 val queries, because only 34% of val sources have an anchored neighbour at all.
- `granularity_mismatch` and `exclusive` are never 1 for the gold target and often 1 for wrong candidates. Because test and val targets are almost never train targets, a re-ranker would learn a large negative weight on them and gain from the split, which is what the plan wanted `A22` to separate.

## 4. Smoke check of the evaluator on real data (no fit, no test)

A zero-weight re-ranker JSON (K = 20, all groups) run through `cga run evaluate` on val gives, for evaluate run `20261016T195809Z-5f83e03e`: Hits@1 0.5213, Hits@5 0.7630, Hits@10 0.8389, MRR **0.6278**, identical to A19's val row in [Phase 6.1](sprint_6_1.md). The experiment file used was temporary and deleted; the run directory is untracked.

## 5. Decision

**Phase 6.2 stops at the probe gate.** Per the plan, A21–A24 are not fitted and have no val or test results; the fit gate and the test readout never come into play. The first stage for Phase 6.3 stays A19, and Phase 6.3's optional "apply 6.2's weights" row does not apply.

What would change the call (not done, developer's decision):
- **Override and fit anyway.** `uv run python scripts/srga_rerank_fit.py` (about a minute, val only) gives A21–A24 with nested-CV MRR and the A21 vs A23 / A24 comparison. The fit gate (A21 beats A23 with a CI excluding 0 and A24 below A21) would then decide whether to spend the single test read. The probe suggests the likely outcome is a small gain from 1-hop agreement, with A21 ahead of A23 mainly through the exclusivity feature, which is why A22 would be the row to read.
- **Redefine "fires".** Parent or child agreement passes the ratio comfortably and misses the share by 3.4 points on val (0.166 vs 0.20). I chose both definitions before looking, and I am reporting both rather than picking the one that passes.

## 6. Limitations and caveats

- The gate is applied to the K = 20 first-stage top-K on val (211 queries, 188 with the gold target in the top-K). With a different K or a different "fires" definition the numbers move; the plan did not fix them, so the two natural readings are reported.
- The probe reads val only for firing. Test is used for anchor coverage only (it involves no model).
- `ruff` is not installed, so lint was not run.
- The fit script (`srga_rerank_fit.py`) and `nested_cv_ranks` were unit-tested on synthetic data but were **not run on the real data**, by design.

## 7. Files

- Created: `src/cga/pipelines/evaluate/rerank.py`, `src/cga/pipelines/evaluate/evaluators/gold_alignment_rerank.py`, `scripts/srga_rerank_{common,probe,fit}.py`, `configs/experiments/ncit_doid_srga_rerank_test.yaml`, `tests/unit/pipelines/evaluate/test_rerank.py`, this log.
- Modified: `src/cga/pipelines/evaluate/pipeline.py`, `src/cga/pipelines/evaluate/evaluators/__init__.py`, `dev_docs/sprints/sprint_6.md` (implementation boxes ticked; variants and test boxes marked not run).
- Untracked artifact: evaluate run `20261016T195809Z-5f83e03e` (the zero-weight smoke run, val).

## 8. Commands

```
uv run pytest tests/unit -q

uv run python scripts/srga_rerank_probe.py                 # the probe and gate (run)
uv run python scripts/srga_rerank_fit.py                   # NOT RUN: only after a deliberate override of the probe gate
uv run cga run evaluate --dataset ncit_doid_srga --experiment ncit_doid_srga_rerank_test   # NOT RUN: test, once, only if the fit gate passes
```
