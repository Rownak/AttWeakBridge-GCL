Yes, it can be built, but I expect little gain from learning λ, based on the experiments so far. Learning it also tends to go wrong in a predictable way.

**First, check which protocol the MRR 0.62 / Hits@10 0.86 come from.** I can't verify that. Our candidate-mode numbers (the dataset's own 101-candidate lists) already sit at about that level:

| test, candidates mode | MRR | Hits@10 |
|---|---|---|
| A12 GCN, no SapBERT block | 0.621 | 0.903 |
| A15 GAT, no SapBERT block | 0.607 | 0.905 |
| A14 GAT + SapBERT | 0.591 | 0.873 |

In global mode (ranking all ~8.5k DOID concepts) we're at 0.536 MRR and 0.77 Hits@10. If the published numbers are global-ranking, the gap is real; if they are candidate-ranking, we're already there. Settle this before investing more.

**Why learned λ probably won't help much**
1. **λ isn't the bottleneck.** In the Phase 5.4 λ curve (A3, val), MRR is flat from λ = 0.25 to 1 (0.476, 0.485, 0.484). Tuning a global λ is worth at most about 0.01.
2. **A jointly trained λ drifts the wrong way.** The adapter overfits the training pairs (train MRR 0.6–0.99 against val 0.5). The training loss therefore rewards a larger λ. A scalar trained on that loss would slide toward the no-floor models (A12/A15), which have the higher break rate. A trainable λ would have to be fit on val, not on the training loss. That is cheap (one or two scalars on 211 pairs), but it is just the λ curve again.
3. **A per-node gate can't see what matters.** The trade-off is that the floor variants break few SapBERT-correct queries (about 100) and fix fewer (about 290), while no-floor fixes about 600 but breaks about 410. A gate would need to know whether SapBERT is right for this query. That depends on the candidate list and score margin. A per-node λ sees one node only. The design doc made the same point when it proposed a bounded gate.
4. **A gate over graph features has no support yet.** A4 added degree and parent/child features and gained nothing on val.

An upper bound shows the size of the prize. A perfect per-query choice between SapBERT and A12 would reach Hits@1 of about 0.53 (1,123 + 627 of 3,280) against A12's 0.41. No node-level gate gets near that.

**What I'd try first, in order of evidence**
1. **Score-level fusion of A11 and A12, or a seed ensemble.** The two are complementary, since A11 rarely breaks and A12 fixes the most. It needs no training; it just sums their cosines. This is untested but cheap, and I'd check it on val first.
2. **A small query-dependent gate: the design's anchor-based re-ranker (§8).** It would use features like the SapBERT score, the A12 score, and "does this candidate's neighbourhood match the known partners of the query's neighbours". It is the only cross-graph structural signal we haven't used (63.7% of test pairs have such an anchor), and it targets the biggest remaining failure. A12 still leaves about 620 of the 901 "SapBERT wrong, confuser near" queries unfixed.
3. **More supervision for the final model.** After selection, retrain on train plus val, which adds 18% more pairs. The gap between train MRR 0.99 and val 0.52 says data is the limit, not λ.

The published model fine-tunes a BERT on ontology relationships, while ours keeps SapBERT frozen. If you want to close the global-ranking gap, the largest lever is probably adapting the text encoder (for example LoRA or a light fine-tune). It is still generic to text, but it changes the design, so I'd treat it as a separate sprint.

If you still want a learned λ, I'd run the cheap version: a val-fit λ per graph on A11 and A12. It would take a few minutes and could confirm the ≤0.01 expectation. Which would you like first: that, the A11 + A12 fusion, or the re-ranker plan?