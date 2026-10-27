# Sprint 10.1: Gold-source selection, provenance and split

## What was implemented

`ingest` reads one ATT&CK → CWE gold CSV chosen in the dataset YAML. It writes `alignments.parquet` (schema unchanged, `split` filled from the YAML fractions) and a sidecar `alignment_provenance.parquet`. With `alignments: null`, ingest behaves as before and writes no sidecar.

| Item | Where |
|---|---|
| Gold reader: `GoldAlignmentConfig`, `read_gold_csv`, `group_id`, `assign_grouped_split` | `src/cga/data/gold_csv.py` (new) |
| `ALIGNMENT_PROVENANCE_SCHEMA` | `src/cga/data/schemas.py` |
| Optional `alignment_provenance()` / `alignment_report()` hooks (default `None`) | `src/cga/adapters/base.py` |
| `attack_cwe` adapter accepts `alignments`, calls the reader, fingerprints the selected CSV | `src/cga/adapters/attack_cwe/adapter.py` |
| Sidecar write, `alignment_provenance` manifest output, gold report merged into manifest `summary` | `src/cga/pipelines/ingest/pipeline.py` |
| Dataset config (full Phase 10.2 block) | `configs/datasets/attack_cwe_ent.yaml` (new) |
| Tests | `tests/unit/data/test_gold_csv.py`, `tests/unit/pipelines/ingest/test_gold_alignments.py` (new) |

## How

- **Resolution:** the adapter builds `external_id → node_id` maps from its own nodes (`Txxxx[.yyy]`, `CWE-n`) and passes them to the reader. Unresolved ids are dropped, never guessed.
- **Drops** are counted per CSV row: `source_not_in_graph`, `target_not_in_graph`, `duplicate`. A duplicate row repeats an already-kept pair (another CAPEC path to the same pair); it adds to that pair's provenance (`capec_ids`, `n_csv_rows`) instead of creating a second alignment. The manifest `summary` carries `gold_source`, `source_file`, `source_sha256`, rows, pairs, techniques, drops and pairs per split.
- **Provenance:** one row per kept pair, same order as `alignments.parquet`. Several distinct `edge_source` or `attack_id_cited` values for one pair are joined with `|`.
- **Split:** `split.fractions` is `[train, val, test]`, default `[0.0, 0.0, 1.0]`, so the whole selected gold set is `test` (training is unsupervised and never reads gold). For non-default fractions, `assign_grouped_split` hashes `sha256("{seed}:{group}")` into [0, 1) and cuts at the cumulative fractions; `group_regex: ^(T\d+)` keeps `T1003` and `T1003.001` together. The four sources are separate gold sets: one is read per ingest and the split is computed inside it only.
- **Unknown `source` key** raises `ConfigError` at adapter construction.
- `alignment_id` = sha1 of `node_1|node_2|equivalence`, as in the NCIT–DOID reader. The pipeline's `val_split` carving is untouched and unused.

## Gate 10.1

- `uv run pytest tests/unit -q`: **560 passed**.
- Default fractions: every kept pair is `test` (unit test and real data below).
- Non-default fractions: families stay together, the same seed gives the same assignment, a different seed a different one (unit tests); YAML fractions `[1, 0, 0]` give all `train` (ingest test).
- Every kept pair has a provenance row; ids and order match; `source_sha256` equals the file's.

## Real-data check

Four ingests with the real data, written to a scratch directory (`paths.processed` overridden), so `data/processed` and `LATEST` are untouched. Phase 10.2 still has to do and record the real runs.

| Source | CSV rows | Pairs kept (all `test`) | Techniques | Duplicate rows | Dropped: source / target |
|---|---|---|---|---|---|
| stix | 376 | 287 | 96 | 69 | 0 / 20 |
| capec | 657 | 566 | 174 | 49 | 1 / 41 |
| intersection | 233 | 221 | 78 | 3 | 0 / 9 |
| union | 800 | 628 | 186 | 119 | 1 / 52 |

- `nodes`, `edges` and `texts` parquet hashes are identical across the four ingests (`9983df13`, `dec3d851`, `72645370`), a head start on gate 10.2.
- The one dropped source is the mobile technique; target drops are CWEs absent from v4.20. Kept pairs are lower than the sprint's CSV distinct-pair counts (300/589/656/229) by exactly those drops.

## Notes

- Only the `fractions` default and its meaning differ from a conventional split; the hash split is still there for later supervised work but is not exercised by real data in this sprint.
- `n_techniques` counts distinct ATT&CK node ids (sub-techniques separate).

## Commands

```
uv run pytest tests/unit -q
uv run cga run ingest --dataset attack_cwe_ent \
  --set ingest.split.fractions=[1.0,0.0,0.0] \
  --set dataset.alignments.equivalence.source=<stix|capec|union|intersection>
```

Optional override of the gold split: `--set dataset.alignments.equivalence.split.fractions=[0.6,0.2,0.2]`. In 10.2, run `union` last so `LATEST` is `union`.

## Correction found during Phase 10.2

The first version of the adapter mapped a CSV id to the *first* node with that `external_id`. 224 ATT&CK external ids are shared by several objects: old `course-of-action` mitigations reuse technique ids. So 147 of 628 `union` pairs pointed at a mitigation node. Fixed in `AttackCweAdapter.iter_alignments`: only `attack-pattern` nodes are gold sources, and an id still shared by several candidates is left out (dropped and counted as `source_not_in_graph`), never guessed. Technique and CWE external ids are otherwise unique (867 technique ids, 0 duplicates). Test added (`test_gold_ignores_non_technique_nodes_sharing_an_external_id`); suite now 561 passed. Pair counts per source did not change, and the scratch ingests listed above are superseded by the Phase 10.2 runs (see `sprint_10_2.md`).
