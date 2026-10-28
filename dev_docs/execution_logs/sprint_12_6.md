# Sprint 12.6 — Execution log: evaluate, plot, table and package the gated models

Phase 12.6 of [sprint_12.md](../sprints/sprint_12.md). Same Sprint 3 proxy protocol as Phase 12.3 (views `structural_r1`,
`structural_r2`, `textual` with the pinned SecRoBERTa_desc reference; ks 1 / 5 / 10 / 20; `cluster_quality` n_clusters
2–32), so the new rows are directly comparable with the 12.3 table. Models come from [sprint_12_5.md](sprint_12_5.md);
nothing was trained here and no evaluator was changed.

## 1. What was done

**Files created**
- `configs/experiments/attack_cwe_ent_paper_gated.yaml` — 9 methods pinned by run id: `mpnet-chunk`, `deepwalk`, `a0`, `a1`, `a2` (as in 12.3) and `g2`, `g3`, `g4`, `g5` (seed 42); `visualize: [g3, g5]`.
- `configs/experiments/attack_cwe_ent_paper_gated_seeds.yaml` — `g2`, `g3`, `g5` at seeds 42 / 43 / 44 (ids `g?-s<seed>`) plus frozen mpnet-chunk.
- `configs/experiments/attack_cwe_ent_paper_gated_hard.yaml` — mpnet-chunk, G3 / G5 on bottom-5 pairs and G3-hard / G5-hard.
- `configs/plots/attack_cwe_ent_paper_gated.yaml` — `precision_at_k`, `silhouette`, `loss_curves`; `evaluate_run` pinned to `20261028T170633Z-c2aecb88`.
- `scripts/paper_proxy_seed_table.py` and `tests/unit/test_paper_proxy_seed_table.py` (4 tests): groups methods by their `-s<seed>` suffix, writes `seed_table.md` / `.csv` with mean ± sample std (ddof 1) of the same 13 columns; a method without a seed suffix is a single value. It imports `build_table` from `paper_proxy_table.py`, which needed no change.
- Package `artifacts/paper/attack_cwe_ent/sprint_12_gated/` (section 5). Modified: `dev_docs/sprints/sprint_12.md` (Phase 12.6 boxes ticked).

**Runs**

| Step | Run id |
|---|---|
| Evaluate, main (9 methods; pinned for plot and table) | `20261028T170633Z-c2aecb88` |
| Evaluate, main repeat | `20261028T171108Z-c2aecb88` |
| Evaluate, seeds | `20261028T170633Z-afc3f972` |
| Evaluate, hard negatives | `20261028T170633Z-f0c17a13` |
| Plot | `20261028T171655Z-b3279b29` |

Main evaluate: 954 rows (106 per method × 9), `plots/tsne_g3.png`, `tsne_g5.png`.

## 2. Checks

- **Repeat identical:** `metrics.parquet` of the repeat equals the first run apart from `run_id`.
- **The five 12.3 rows reproduce exactly:** all 530 metric rows of the 12.3 run `20261028T044615Z-b20dd0c5` are in the new run with maximum absolute difference 0.0 and identical `n_queries` (the neighbourhood query set is the same).
- **`n_queries` equal across the 9 methods** in every `neigh` (view, role): `structural_r1` and `structural_r2` 1,981 (ATT&CK) / 944 (CWE); `textual` 1,983 / 944.
- **Table regeneration:** `paper_proxy_table.py` (main and hard runs) and `paper_proxy_seed_table.py` (seeds run) were run twice; the `.md` and `.csv` SHA-256 are identical between the runs (main `paper_table.md` `4c744587…`, `paper_table.csv` `61676556…`; hard `51055300…`; seeds `seed_table.md` `f0640918…`, `.csv` `beb91de1…`).
- **Figures:** `precision_at_k.png` (6 panels × 9 lines) and `silhouette.png` exist and render; the G4 line lies exactly under frozen mpnet-chunk (G4 = `LayerNorm(x)`). `loss_curves` wrote one PNG per model id under the dataset (as in 12.3); the G4 curve is empty by design (no training), so it is not packaged.
- Tests: `uv run pytest tests/unit -q` passes (see section 7).

## 3. Tables

### Main (`paper_table.md`, evaluate run `20261028T170633Z-c2aecb88`, seed 42; best per column in bold)

`r1` / `r2` = `structural_r1` / `structural_r2`, `text` = `textual`; A = ATT&CK, C = CWE.

| Method | r1 A MRR | r1 A P@10 | r1 C MRR | r1 C P@10 | r2 A MRR | r2 A P@10 | r2 C MRR | r2 C P@10 | text A MRR | text A P@10 | text C MRR | text C P@10 | Silh. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 full dual GCN | 0.6758 | 0.4255 | 0.5868 | 0.8075 | 0.9977 | 0.9020 | 0.9760 | 0.7974 | 0.4100 | 0.1205 | 0.4531 | 0.1724 | 0.0691 |
| A1 text-only GCN | 0.4597 | 0.2587 | **0.6397** | 0.7841 | 0.9531 | 0.7747 | 0.9575 | 0.7194 | 0.4789 | 0.1484 | 0.4840 | 0.1875 | 0.0814 |
| A2 graph-only GCN | 0.7798 | 0.5485 | 0.5507 | 0.8250 | **1.0000** | 0.9410 | 0.9966 | 0.8848 | 0.3523 | 0.0930 | 0.3816 | 0.1500 | 0.0911 |
| DeepWalk (frozen) | **0.8655** | **0.6887** | 0.5605 | **0.8443** | **1.0000** | **0.9801** | **1.0000** | **0.9466** | 0.3460 | 0.0855 | 0.3818 | 0.1500 | **0.0945** |
| G2 residual, no gate | 0.4069 | 0.2138 | 0.5115 | 0.6484 | 0.9147 | 0.6757 | 0.7848 | 0.4805 | 0.5324 | 0.1664 | 0.5430 | 0.2090 | 0.0744 |
| G3 gated residual (text) | 0.3881 | 0.2167 | 0.4761 | 0.5956 | 0.9231 | 0.7169 | 0.7988 | 0.5319 | 0.4920 | 0.1584 | 0.5267 | 0.2056 | 0.0804 |
| G4 gate fixed 0 | 0.3439 | 0.1588 | 0.2760 | 0.3398 | 0.9250 | 0.7273 | 0.6108 | 0.3244 | **0.5649** | 0.1907 | **0.6103** | 0.2421 | 0.0542 |
| G5 gated dual | 0.4769 | 0.2685 | 0.6063 | 0.7581 | 0.9505 | 0.7641 | 0.9250 | 0.6785 | 0.4929 | 0.1524 | 0.4921 | 0.1941 | 0.0804 |
| mpnet-chunk (frozen) | 0.3439 | 0.1589 | 0.2760 | 0.3398 | 0.9250 | 0.7273 | 0.6108 | 0.3244 | **0.5649** | **0.1908** | **0.6103** | **0.2422** | 0.0558 |

G4 and frozen mpnet-chunk are both bold on textual MRR (equal to 4 digits); mpnet-chunk leads the textual P@10 columns by 0.0001. G4 differs from frozen mpnet-chunk by at most 0.0016 (silhouette; the frozen store is scored on more nodes).

### Seeds (`seed_table.md`, seeds 42–44, mean ± sample std)

| Method (seeds) | r1 A MRR | r1 A P@10 | r1 C MRR | r1 C P@10 | r2 A MRR | r2 A P@10 | r2 C MRR | r2 C P@10 | text A MRR | text A P@10 | text C MRR | text C P@10 | Silh. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G2 residual, no gate (3) | 0.4125 ± 0.0053 | 0.2163 ± 0.0021 | 0.5176 ± 0.0056 | 0.6593 ± 0.0100 | 0.9165 ± 0.0019 | 0.6789 ± 0.0036 | 0.7887 ± 0.0036 | 0.4865 ± 0.0053 | 0.5309 ± 0.0018 | 0.1673 ± 0.0008 | 0.5437 ± 0.0034 | 0.2064 ± 0.0026 | 0.0732 ± 0.0012 |
| G3 gated residual (text) (3) | 0.3885 ± 0.0007 | 0.2168 ± 0.0013 | 0.4807 ± 0.0040 | 0.6023 ± 0.0066 | 0.9244 ± 0.0012 | 0.7167 ± 0.0019 | 0.8066 ± 0.0073 | 0.5357 ± 0.0056 | 0.4931 ± 0.0040 | 0.1573 ± 0.0010 | 0.5248 ± 0.0043 | 0.2043 ± 0.0012 | 0.0790 ± 0.0013 |
| G5 gated dual (3) | **0.4818 ± 0.0047** | **0.2724 ± 0.0035** | **0.6101 ± 0.0041** | **0.7593 ± 0.0012** | **0.9538 ± 0.0033** | **0.7654 ± 0.0011** | **0.9254 ± 0.0011** | **0.6774 ± 0.0014** | 0.4906 ± 0.0021 | 0.1522 ± 0.0006 | 0.4937 ± 0.0017 | 0.1936 ± 0.0004 | **0.0813 ± 0.0014** |
| mpnet-chunk (frozen) (1) | 0.3439 | 0.1589 | 0.2760 | 0.3398 | 0.9250 | 0.7273 | 0.6108 | 0.3244 | **0.5649** | **0.1908** | **0.6103** | **0.2422** | 0.0558 |

(Bold = best mean among these four rows only.) The standard deviation over seeds is at most 0.010 in any cell and mostly
below 0.005.

### Bottom-5 against hard negatives (`hard_table.md`, evaluate run `20261028T170633Z-f0c17a13`, seed 42)

| Method | r1 A MRR | r1 A P@10 | r1 C MRR | r1 C P@10 | r2 A MRR | r2 A P@10 | r2 C MRR | r2 C P@10 | text A MRR | text A P@10 | text C MRR | text C P@10 | Silh. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G3, bottom-5 | 0.3881 | 0.2167 | 0.4761 | 0.5956 | 0.9231 | 0.7169 | 0.7988 | 0.5319 | 0.4920 | 0.1584 | 0.5267 | 0.2056 | 0.0804 |
| G3, hard | 0.4407 | 0.2484 | 0.5507 | 0.6952 | 0.9451 | 0.7587 | 0.8765 | 0.6277 | 0.4999 | 0.1544 | 0.5266 | 0.2091 | 0.0865 |
| G5, bottom-5 | 0.4769 | 0.2685 | 0.6063 | 0.7581 | 0.9505 | 0.7641 | 0.9250 | 0.6785 | 0.4929 | 0.1524 | 0.4921 | 0.1941 | 0.0804 |
| G5, hard | 0.5141 | 0.2927 | 0.6676 | 0.8485 | 0.9655 | 0.7806 | 0.9570 | 0.7782 | 0.5053 | 0.1558 | 0.5134 | 0.2016 | 0.0806 |
| mpnet-chunk (frozen) | 0.3439 | 0.1589 | 0.2760 | 0.3398 | 0.9250 | 0.7273 | 0.6108 | 0.3244 | 0.5649 | 0.1908 | 0.6103 | 0.2422 | 0.0558 |

## 4. Reading (no tuning; single seed unless stated)

Success criterion of the idea ("A0-gated at least A1 within noise; structure never below frozen text"):

- **G5 against A1** (the gated dual against the text-only GCN; seed std of G5 ≤ 0.005). G5 is above A1 on all four `textual` columns (for example text CWE MRR 0.492 vs 0.484, P@10 0.194 vs 0.188; text ATT&CK P@10 0.152 vs 0.148, differences of 0.004–0.014) and level on silhouette (0.0813 vs 0.0814), and above A1 on ATT&CK `structural_r1` (MRR 0.477 vs 0.460). It is below A1 on CWE structural columns (r1 CWE MRR 0.606 vs 0.640; r2 CWE MRR 0.925 vs 0.958; P@10 0.678 vs 0.719 on r2 CWE). So the criterion holds on the text view and not uniformly on the structural views.
- **Gated models against frozen mpnet-chunk on the `textual` view: not met.** Every gated model is below frozen text on all four textual columns (G5 text CWE MRR 0.492 vs 0.610; G3 0.527; G2 0.543; seed std ≤ 0.004). The gated rows give up some of the text relation to the SecRoBERTa neighbours (0.07–0.12 MRR) in exchange for structure: on the structural views they are far above frozen text (G5 r1 CWE MRR 0.606 vs 0.276, r2 CWE P@10 0.678 vs 0.324), which frozen text does not carry at all. The gated models therefore do not stay at the frozen-text level on the text view; they sit between frozen text and the structure-driven rows.
- **G4 (gate fixed 0) reproduces frozen text** (within 0.0016), so the gap above is caused by training, not by the LayerNorm.
- **Gate against no gate (G3 against G2; seed spread ≤ 0.010).** G3 is higher on `structural_r2` (P@10 ATT&CK 0.717 vs 0.679; CWE 0.536 vs 0.487) and silhouette (0.079 vs 0.073), and lower on `textual` (text CWE MRR 0.525 vs 0.544; text ATT&CK MRR 0.493 vs 0.531) and on `structural_r1` (r1 CWE MRR 0.481 vs 0.518). The gate does not protect the text view in these metrics: G2, which adds the graph signal ungated, is closer to frozen text. The gate's mean value (≈ 0.09) means G3 adds a smaller correction than G2 (the residual output of `f` is scaled by the gate), and the differences point in different directions, so no overall advantage of the gate is shown here.
- **Gate arbitration against concatenation (G5 against A0).** G5 is higher than A0 on all four `textual` columns (text ATT&CK MRR 0.493 vs 0.410, text CWE P@10 0.194 vs 0.172) and silhouette (0.0804 vs 0.0691), and on CWE `structural_r1` MRR (0.606 vs 0.587). A0 is higher on ATT&CK structural columns (r1 MRR 0.676 vs 0.477; r2 MRR 0.998 vs 0.951) and on CWE r1 P@10 and r2 columns (0.808 vs 0.758; 0.797 vs 0.679). G5 keeps more of the text neighbourhood and A0 more of the structural one, so the gate trades structure for text relative to concatenation.
- **Gate behaviour (Phase 12.5).** The gate neither collapsed nor saturated; mean 0.09–0.19, p95 ≤ 0.21, Spearman with degree between −0.48 (G3 ATT&CK) and +0.30 (G5 CWE text gate): no single degree trend. The gated rows therefore use a small, intermediate share of the structure signal; whether this helps is judged by the columns above (not by the gate value).
- **Bottom-5 against hard negatives (seed 42 only).** Hard negatives improve the structural columns of both models, for example G3 r1 CWE MRR 0.476 → 0.551, r2 CWE P@10 0.532 → 0.628, G5 r1 CWE P@10 0.758 → 0.849, and G3 silhouette 0.0804 → 0.0865. Textual columns change by at most 0.012 (G5 text ATT&CK MRR 0.493 → 0.505; G3 unchanged). Differences of this size are above the seed spread of the bottom-5 models (≤ 0.010) but this comparison has one seed per variant.
- **Against the 12.3 rows.** All trained rows lie between the two frozen baselines in most columns. The gated models do not reach frozen DeepWalk on structural views (DeepWalk r2 CWE P@10 0.947 vs G5 0.678) or frozen mpnet-chunk on the text view.
- **Silhouette** (node sets differ, see caveat 4): DeepWalk 0.0945, A2 0.0911, A1 0.0814, G5 0.0804–0.0813, G3 0.0804, G2 0.0744, A0 0.0691, mpnet-chunk 0.0558, G4 0.0542.

## 5. Package: `artifacts/paper/attack_cwe_ent/sprint_12_gated/`

`paper_table.md` / `.csv` (9 rows), `seed_table.md` / `.csv`, `hard_table.md` / `.csv`, `figures/precision_at_k.png`,
`figures/silhouette.png`, `loss_curves/loss_curve_{g2,g3,g5}.png` (seed 42; G4 has no training), `projections/tsne_{g3,g5}.png`,
`gate/{g3,g5}/` (diagnostics tables and the two gate figures from Phase 12.5), and `runs.yaml` (every pinned run id of
12.1, 12.2 and 12.5, the evaluate / plot run ids, and the exact commands).

## 6. Caveats (carry into the paper text)

1. **Proxy relevance sets, not gold labels** (graph neighbours within radius 1 / 2 and top-10 cosine neighbours of the SecRoBERTa store).
2. **Not an alignment measure.** Neighbourhoods are scored within each graph, so these metrics do not measure ATT&CK → CWE alignment; a gold evaluation of the gated models (Sprint 11 style) is not part of this sprint and is the next step.
3. **Textual reference favours text rows.** SecRoBERTa is itself a text encoder, so frozen mpnet-chunk and G4 (≈ frozen text) lead the `textual` view by construction; a drop there partly reflects the models moving away from text, which is what the gate was meant to prevent but did not under this metric.
4. **Silhouette node sets differ across stores.** Trained rows (including the gated ones) cover 1,983 ATT&CK / 944 CWE nodes, frozen DeepWalk 2,688 / 944 and frozen mpnet-chunk 4,162 / 969 (the evaluator does not intersect candidates for `cluster_quality`); the silhouette column is indicative only. Unchanged evaluator.
5. **Seeds.** Three seeds only for G2, G3 and G5 (seed spread ≤ 0.010); one seed for every other row (frozen, A0, A1, A2, G4, G3-hard, G5-hard) and one pseudo-pair draw.
6. Gated-model design choices (per-dimension gate, bias −4, gate weight decay 1e-4, bottleneck 128, 100 epochs) were fixed in advance and not tuned; A0–A2 were trained at 100 epochs with their own settings (Sprint 11).
7. The trained models optimise a cross-graph pseudo-pair loss, not the within-graph proxies scored here.

## 7. Commands

```
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_paper_gated \
  --set evaluate.inputs.ingest_run=20261027T194643Z-5d3f8baf            # run twice (main and repeat)
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_paper_gated_seeds \
  --set evaluate.inputs.ingest_run=20261027T194643Z-5d3f8baf
uv run cga run evaluate --dataset attack_cwe_ent --experiment attack_cwe_ent_paper_gated_hard \
  --set evaluate.inputs.ingest_run=20261027T194643Z-5d3f8baf
uv run cga run plot --dataset attack_cwe_ent --config configs/plots/attack_cwe_ent_paper_gated.yaml
uv run python scripts/paper_proxy_table.py --dataset attack_cwe_ent --run 20261028T170633Z-c2aecb88
uv run python scripts/paper_proxy_table.py --dataset attack_cwe_ent --run 20261028T170633Z-f0c17a13
uv run python scripts/paper_proxy_seed_table.py --dataset attack_cwe_ent --run 20261028T170633Z-afc3f972
uv run pytest tests/unit -q
```

## 8. Status

Phase 12.6 acceptance met: the 9-row table, the seed table and the hard-negative table regenerate byte-identically from their
evaluate runs with one command each; the package holds every figure and run id referenced here; the gated models are fully
specified and reproducible (Phase 12.5). Sprint 12 phases 12.1–12.6 are complete.
