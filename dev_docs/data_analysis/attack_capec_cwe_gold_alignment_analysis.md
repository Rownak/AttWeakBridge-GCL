# Why ATT&CK → CWE gold-alignment scores are low

Scope: `attack_cwe_ent` (ATT&CK Enterprise 20.0 × CWE v4.20), the evaluators `gold_alignment` and `gold_alignment_prf`, and the Sprint 10.6 / 11.4 results. No code was changed for this analysis.

Each finding is labelled:
- **[Verified]**: read in code or measured on the artifacts.
- **[Inferred]**: follows from the data but was not traced to its source.
- **[Recommendation]**: a proposed change, not yet made.

Measurements come from two scratch scripts, kept outside the repo, run on the ingest runs listed in §1 and the SecRoBERTa store `20261027T010723Z-8bc08adc`. Those scripts recompute the metrics over all 969 CWEs. The official runs use the 944 CWEs that every method shares, so the scripts' values differ in the fourth decimal place. For example, union full-view MRR is 0.0233 in the script and 0.0234 in the official run.

---

## 0. Summary

Five causes act together. They are listed from largest to smallest.

1. **[Verified] Bug: the CWE "description" text is often not the description.** `cwe_parser._flatten_block` searches `.//cwe:Description/xhtml:p`, which matches *any* nested `Description` element: those under `Detection_Method` and `Mitigation` as well as the weakness's own. When the real `<Description>` is plain text (no `xhtml:p`) and a nested one has paragraphs, the nested mitigation or detection text replaces it.
   - **Extent:** 241 of 969 CWEs (25%), and 53 of the 151 gold CWEs.
   - **Gold pairs hit:** 43–46% of gold pairs point at a CWE with a corrupted text.
   - **Hub CWEs are affected:** CWE-284, CWE-200, CWE-285 and CWE-309 have wrong texts. So do CWE-120 and CWE-732, which are the two "hub" CWEs that frozen SecRoBERTa ranks near the top for most techniques (CWE-120 is in the top 3 for 77 of 186 techniques).
   - **Effect:** Every text embedding, every pseudo-pair and every trained model on this dataset inherits the corruption.
2. **[Verified] P/R/F1 has a hard ceiling far below 1, and the reported "R ceiling" overstates it.** Gold is many-to-many, but `greedy_one_to_one` allows each CWE to be predicted once.
   - **True ceiling:** The best possible one-to-one recall is the maximum bipartite matching of the gold graph, which is 0.159–0.199 depending on source.
   - **Reported ceiling:** The summary tables print `unique sources / #gold`, which is 0.30–0.35.
3. **[Verified] The ranking metrics are computed per gold pair rather than per technique.** That is valid as a "filtered" protocol but harsher than the usual multi-relevant MRR. Per-technique MRR, which takes the rank of the highest-ranked relevant CWE, is about 2.4–2.7× the current value for the same embeddings (§4).
4. **[Inferred] The gold relation is "exploits a weakness", reached through CAPEC, not equivalence.**
   - Techniques describe adversary behaviour; CWEs describe software defects.
   - Many gold targets are generic Pillar/Class CWEs: CWE-284 is linked to 29 techniques and CWE-200 to 27.
   - Text similarity is a weak signal for such pairs even with clean texts.
5. **[Verified] The candidate pool is large and the gold is sparse.**
   - In the full view, one query faces 944 CWEs, of which about 3.4 are relevant.
   - Random MRR is 0.007 (full) and 0.038 (151-CWE pool).
   - Absolute values in the 0.02–0.08 range should be read against these baselines.

Items 1 and 2 are defects. Item 3 is a protocol choice that should be reported alongside a per-technique metric. Items 4 and 5 are properties of the task.

---

## 1. How the gold data is built

### 1.1 Files and row meaning

**[Verified]** There are four CSVs in `data/raw/capec/`, and the dataset YAML selects one per ingest (`alignments.equivalence.source`). All four share one schema. **One row is one path `ATT&CK technique → CAPEC pattern → CWE`**:

| Column | Meaning |
|---|---|
| `attack_id_cited` | Technique id as cited by the mapping source (may be an old or renamed id) |
| `attack_id_current` | Current ATT&CK 20.0 id; **this is the gold source column** |
| `attack_domain`, `attack_technique`, `attack_status`, `is_subtechnique`, `tactics`, `platforms` | Technique metadata |
| `capec_id`, `capec_name`, `capec_abstraction`, `capec_status` | The bridging CAPEC pattern |
| `cwe_id`, `cwe_name`, `cwe_abstraction` | The CWE; **this is the gold target column** |
| `edge_source` | Which mapping produced the ATT&CK→CAPEC edge |

| Source key | File | CSV rows | `edge_source` values |
|---|---|---|---|
| `stix` | `1_…from_stix_repo.csv` | 376 | `v12.1` (302), `v12.1+v20.0` (74) |
| `capec` | `2_…from_capec_xml.csv` | 657 | `capec-3.9` |
| `union` | `3_…union.csv` | 800 | all of the above plus `capec-3.9+v12.1` |
| `intersection` | `4_…intersection.csv` | 233 | `capec-3.9+v12.1` only |

### 1.2 Provenance rules

**[Inferred]** These rules come from the column values. The CSVs' build script is not in the repo; `build_cwe_attack_map.py` builds a different CAPEC-703 map.
- **`stix`:** The ATT&CK→CAPEC edges are the CAPEC external references in ATT&CK STIX bundles. `v12.1` means the reference is present in v12.1; `v12.1+v20.0` means it is present in both v12.1 and v20.0. The CAPEC→CWE step uses CAPEC's `Related_Weaknesses`.
- **`capec`:** The ATT&CK→CAPEC edges come from the CAPEC 3.9 XML `Taxonomy_Mapping[@Taxonomy_Name='ATTACK']`, and CAPEC→CWE from `Related_Weaknesses`.
- **`union` / `intersection`:** The union or intersection of the two at the (technique, CAPEC, CWE) level. The `edge_source` value is the joined tag.
- **`attack_id_cited` ≠ `attack_id_current`:** This holds in 82 stix rows and 92 union rows. These are cited ids remapped to current technique ids (renames and sub-technique splits).

### 1.3 How ingest turns rows into gold pairs

**[Verified]** `cga/data/gold_csv.py`, `adapters/attack_cwe/adapter.py`:
- Only `attack_id_current` and `cwe_id` are used. **CAPEC is dropped from the pair key** and kept only in `alignment_provenance.parquet` (`capec_ids`, `n_csv_rows`).
- Source ids resolve only to `attack-pattern` nodes. Old mitigations reuse technique ids; for example, T1003 is also a `course-of-action` node.
- Rows are deduplicated to distinct `(technique, CWE)` pairs. All pairs go to split `test` (`fractions [0,0,1]`), and the relation is labelled `equivalence`.

| Source | CSV rows | Duplicates | Empty `cwe_id` | Unknown technique | Gold pairs | Techniques | CWEs |
|---|---|---|---|---|---|---|---|
| stix | 376 | 69 | 20 | 0 | 287 | 96 | 77 |
| capec | 657 | 49 | 41 | 1 | 566 | 174 | 149 |
| intersection | 233 | 3 | 9 | 0 | 221 | 78 | 68 |
| union | 800 | 119 | 52 | 1 | 628 | 186 | 151 |

- **[Verified] Most duplicates are one pair reached through more than one CAPEC.** In union, 54 distinct pairs come through two or more CAPECs. The rest are rows repeated with different `attack_id_cited` or `edge_source`.
- **[Verified] Every `target_not_in_graph` drop is a row with an empty `cwe_id`.** These are CAPEC patterns with no related weakness. No real CWE id is missing from CWE v4.20, so the reason name is misleading but no gold is lost.

---

## 2. Many-to-many structure

**[Verified]** One technique maps to several CWEs and one CWE to several techniques. The gold CSV represents this as one row per path, and the gold set as one pair per (technique, CWE).

| Source | CWEs per technique: mean / median / max / share > 1 | Techniques per CWE: mean / median / max / share > 1 | Most-linked CWEs |
|---|---|---|---|
| stix | 2.99 / 1 / 10 / 43% | 3.73 / 1 / 19 / 47% | CWE-308 (19), CWE-200 (18) |
| capec | 3.25 / 1 / 23 / 47% | 3.80 / 2 / 26 / 60% | CWE-200 (26), CWE-284 (25) |
| intersection | 2.83 / 1 / 10 / 36% | 3.25 / 1 / 17 / 47% | CWE-200 (17), CWE-308 (13) |
| union | 3.38 / 2 / 23 / 51% | 4.16 / 2 / 29 / 62% | CWE-284 (29), CWE-200 (27) |

**[Inferred]** The fan-out comes from the CAPEC bridge. One CAPEC lists several weaknesses, often at different abstraction levels. For example, CAPEC-150 maps to both CWE-552 and CWE-1239 (a hardware-register CWE) for T1003. Because the gold is transitive, every CWE reachable through any linked CAPEC is "correct", whether or not it is the most specific match. In union, the 151 gold CWEs include 4 Pillar and 32 Class CWEs.

---

## 3. How the metrics are computed now

### 3.1 `gold_alignment` (Hits@k, MRR), `view_id = global`

**[Verified]** `evaluators/gold_alignment.py`, `metrics.py`:
- **Queries:** **one query per gold pair** `(t, c)`. Pairs whose endpoints are missing from any method's embeddings are dropped. In stix, 3 techniques are outside the GCN training universe, so 287 pairs become 284 queries.
- **Candidates:** every CWE shared by all methods (944 in the full view, 151 with `target_pool`), **minus the query's other gold CWEs** (`sibling_filter_splits: [test]`). This is the "filtered" setting from knowledge-graph link prediction.
- **Ranking:** cosine of L2-normalised vectors, ties broken by node id.
- **Hits@k:** 1 if `c` is in the top k of that filtered list. **MRR:** 1 / rank of `c` in that list.
- **Aggregation:** the mean over **pairs**. A technique with 23 gold CWEs contributes 23 queries; one with 1 CWE contributes 1.

### 3.2 `gold_alignment_prf` (P, R, F1)

**[Verified]** `evaluators/gold_alignment_prf.py`:
1. For each gold technique, take the cosine against every CWE and keep the top 3 (`top_k: 3`).
2. Sort all candidate rows by score and accept greedily. **Each technique is used once and each CWE is used once** (`greedy_one_to_one`), and only rows with score ≥ `threshold` (0.0) are accepted.
3. Compute the scores:
   - Precision = correct / predicted.
   - **Recall = correct / |gold pairs|.**
   - F1 is their harmonic mean.

---

## 4. Ranking with several unordered correct CWEs

The gold gives a *set* of relevant CWEs per technique with no order. The natural query unit is therefore the **technique**, and the relevant set is all of its gold CWEs.

**Is MRR valid?** Yes, if it is defined per technique as **1 / rank of the highest-ranked relevant CWE**. Hits@k becomes "any relevant CWE in the top k". This is the standard definition for multi-relevant retrieval, and `metrics.mrr` already implements it (`relevant` is a set). MRR does not depend on any order among the relevant items, so the missing order is not a problem.

**Its limit is that it rewards only the first hit.** It says nothing about whether the other 2–22 relevant CWEs were found. **[Recommendation]** Report it together with set-aware metrics:
- **MAP** (average precision per technique), or **R-precision**.
- **Recall@k** (|relevant in top k| / |relevant|) and/or **nDCG@k** with binary gains.

**The current per-pair filtered MRR is not wrong**, but it answers a different question: how high is *this particular* CWE once the other correct ones are removed. It also weights techniques by their number of gold CWEs.

**[Verified]** Frozen SecRoBERTa on the same embeddings, union source, 186 techniques:

| Metric | Full: SecRoBERTa | Full: random | Pool: SecRoBERTa | Pool: random |
|---|---|---|---|---|
| MRR per pair, filtered (current) | 0.0233 | 0.0070 | 0.0732 | 0.0385 |
| MRR per pair, unfiltered | 0.0231 | 0.0070 | 0.0700 | 0.0374 |
| **MRR per technique, first relevant** | **0.0618** | 0.0166 | **0.1774** | 0.0834 |
| MAP per technique | 0.0322 | 0.0106 | 0.1108 | 0.0557 |
| Hits@1 per technique | 0.016 | 0.000 | 0.097 | 0.016 |
| Hits@10 per technique | 0.145 | 0.022 | 0.333 | 0.199 |
| Recall@10 | 0.062 | 0.009 | 0.181 | 0.072 |
| nDCG@10 | 0.034 | 0.006 | 0.119 | 0.045 |

The other three sources show the same pattern: per-technique MRR is 2.3–2.7× the current value. Lift over random is similar under both definitions. So the per-pair protocol explains the *low absolute value* but not a *poor ranking of methods*. The sibling filter changes almost nothing (0.0233 vs 0.0231), because relevant CWEs are rarely ranked near each other.

---

## 5. Other factors that lower the numbers

### 5.1 Candidate pool size

**[Verified]**
- **Full view:** 944 candidates and about 3.4 relevant per technique. Random MRR is 0.007 and random Hits@10 is about 0.01.
- **Pool view (151 gold CWEs):** random MRR is 0.038, and Hits@10 per technique is 0.20 for random.
- **The pool is built from the gold**, so it is optimistic, and its MRR scale is not comparable to the full view.
- **[Recommendation]** Always report the lift over random next to raw values, as the summary script's `--lift` does.

### 5.2 One-to-one matching

This is the largest effect on P/R/F1.

**[Verified]**

| Source | Gold pairs | Max one-to-one matching (true recall ceiling) | `unique sources / #gold` (printed as "R ceiling") |
|---|---|---|---|
| stix | 287 | 50 → **0.174** | 0.334 |
| capec | 566 | 97 → **0.171** | 0.307 |
| intersection | 221 | 44 → **0.199** | 0.353 |
| union | 628 | 100 → **0.159** | 0.296 |

- **The assignment discards most correct predictions by design.** A CWE such as CWE-284, which has 29 gold techniques, can be credited once at most.
- **Hubness collapses the predicted set.**
  - CWE-120 appears in the top 3 of 77 of 186 techniques (full view); in the pool, CWE-732 appears in 99 of 186.
  - The top-3 lists contain only 124 distinct CWEs (full) or 70 (pool) across 186 techniques.
  - So greedy matching runs out of unused candidates: `#Pred` is about 92 for 186 techniques, and F1 then rests on 1–16 correct pairs.
- **[Recommendation]** For many-to-many gold, also report **P/R/F1 without the one-to-one constraint** (top-k per technique). For union, frozen SecRoBERTa scores:

  | View | top-1 P / R | top-3 P / R | top-10 P / R |
  |---|---|---|---|
  | Full | 0.016 / 0.005 | 0.016 / 0.014 | 0.016 / 0.048 |
  | Pool | 0.097 / 0.029 | 0.063 / 0.056 | 0.043 / 0.126 |

  If one-to-one matching is kept for comparability with BioGITOM, print the matching-based ceiling.

### 5.3 Threshold

**[Verified]** Cosine values between gold techniques and CWEs lie between 0.12 and 0.79, with a median of 0.42. `threshold: 0.0` therefore never filters anything; `#Pred` is limited by the one-to-one assignment alone.

### 5.4 Duplicates and unmapped entities

**[Verified]**
- Deduplication is correct: each distinct pair is scored once, in both evaluators.
- Empty-`cwe_id` rows are dropped. One technique id (capec and union sources) does not resolve to a current `attack-pattern` and is dropped.
- At evaluation, pairs whose technique has no embedding in some method are dropped from ranking (3 stix techniques for GCN rows).
- **PRF still counts every gold pair in `n_gold`**, including the ones no method can score. This lowers recall slightly (≤ 1%).

### 5.5 Source choice

**[Verified]**
- **`intersection`** (221 pairs, both mappings agree) gives the highest scores and the highest one-to-one ceiling (0.199).
- **`union`** has the most pairs and the widest fan-out (median 2 CWEs per technique, CWE-284 with 29 techniques), so per-pair recall and the one-to-one ceiling are lowest.
- **`stix` and `capec`** give similar scores.
- **[Inferred]** `intersection` is the cleanest gold for headline numbers. Union is the most complete but the noisiest.

---

## 6. Implementation and protocol issues that lower the metrics

| # | Issue | Status | Effect |
|---|---|---|---|
| 1 | **CWE description replaced by nested mitigation/detection text.** `cwe_parser._flatten_block(weakness, "Description")` uses a descendant search (`.//cwe:Description/xhtml:p`), so for 241 CWEs the `description` field holds `Mitigation` / `Detection_Method` paragraphs instead of the weakness's direct-child `<Description>`. Examples: CWE-284's text starts "Compartmentalize the system…"; CWE-120's starts "This weakness can often be detected using automated static analysis tools…". | **Verified bug** | Corrupts 43–46% of gold targets' texts, including the hub CWEs. Every embedding store, pseudo-pair set and trained model on `attack_cwe_ent` is built on these texts. It is the most likely single cause of the low scores and of the CWE-120 / CWE-732 hubness. |
| 2 | Summary "R ceiling" = `n_unique_sources / n_gold`. | Verified (reporting) | Overstates the one-to-one recall ceiling by about 2× (0.30 vs 0.16 for union). |
| 3 | Per-pair aggregation of Hits@k / MRR. | Verified (protocol choice) | About 2.5× lower than per-technique MRR, and techniques with many CWEs dominate. |
| 4 | One-to-one assignment on many-to-many gold. | Verified (protocol choice) | Recall can never exceed 0.16–0.20. |
| 5 | ATT&CK `description` text keeps `(Citation: …)` markers and Markdown links; the technique name is not part of the text variant. CWE text includes the name. | Verified (text content); effect inferred | Adds noise to technique embeddings. Likely small compared with #1. |
| 6 | Unseeded `gcl_train` (Sprint 11, section 4). | Verified | Trained rows are single random draws. It does not lower the metrics, but small differences cannot be read. |

**[Recommendation]** In priority order:
1. **Fix the description parser.** Read only the direct child, `weakness.find("cwe:Description")`, flattened with `itertext()`. Do the same for `Extended_Description`. Add a unit test on a CWE whose `<Description>` is plain text and whose mitigation has `xhtml:p` children, such as CWE-284.
2. **Re-ingest, re-embed and retrain, then re-evaluate.** The Sprint 10/11 numbers on `attack_cwe_ent` are not valid as text-encoder results until this is done.
3. **Add per-technique metrics** next to the current per-pair ones: first-relevant MRR, Hits@k (any relevant), MAP, Recall@k and nDCG@k.
4. **Add many-to-many P/R/F1** (top-k per technique, no one-to-one constraint), and replace the printed "R ceiling" with the maximum-matching ceiling for the one-to-one variant.
5. **Lead with `intersection`** as the headline gold, and keep `union` as the high-recall reference. Report the pool view only together with its random baseline.

---

## 7. Evidence

**Code:**
- `src/cga/adapters/attack_cwe/cwe_parser.py` (lines 70–85, 177)
- `src/cga/data/gold_csv.py`
- `src/cga/adapters/attack_cwe/adapter.py` (lines 150–175)
- `src/cga/pipelines/evaluate/evaluators/gold_alignment.py`
- `src/cga/pipelines/evaluate/evaluators/gold_alignment_prf.py`
- `src/cga/pipelines/evaluate/metrics.py`
- `scripts/unsup_summary.py` (`build_source_report`)

**Data:**
- Gold CSVs: `data/raw/capec/{1,2,3,4}_attack_capec_cwe__*.csv`.
- CWE catalog: `data/raw/enterprise_cwe/cwec_v4.20.xml`.
- Ingest runs:

  | Source | Ingest run |
  |---|---|
  | stix | `20261026T224906Z-a0725a37` |
  | capec | `20261026T224923Z-1b4b5b3d` |
  | intersection | `20261026T224935Z-dee50c06` |
  | union | `20261026T224948Z-0f7e98dc` |

- Embedding store: `artifacts/embeddings/attack_cwe_ent/hf_mean_pool-pt_SecRoBERTa/20261027T010723Z-8bc08adc`.

**Results compared:** Sprint 11.4 evaluate runs `20261027T0333…`–`0334…`, as listed in `dev_docs/execution_logs/sprint_11.md` §5.
