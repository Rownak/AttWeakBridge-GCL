# Why set-based gold scores stay low (ATT&CK Enterprise × CWE, after Sprint 11.6–11.7)

Follow-up to [attack_capec_cwe_gold_alignment_analysis.md](attack_capec_cwe_gold_alignment_analysis.md) and [sprint_11_7.md](../execution_logs/sprint_11_7.md). No code or config was changed. Numbers come from ad hoc read-only scripts over the stores listed below.

**Setup.** Union gold, full view: 183 techniques, 944 CWE candidates (the CWEs shared by all four stores, out of 969).

| Store | Run |
|---|---|
| frozen SecRoBERTa | `hf_mean_pool-pt_SecRoBERTa/20261027T134403Z-8bc08adc` |
| DeepWalk | `deepwalk-deepwalk/20261027T135210Z-3b18ba0c` |
| triplet dual GCN | `ent-dual-gcn-triplet-s42b/20261027T141524Z-5ac0fba7` |
| InfoNCE dual GCN | `ent-dual-gcn-infonce-s42b/20261027T141732Z-3f70b1a2` |

Cosine ranking; ties are broken by node id.

## 1. Summary

The scores are low for five reasons. All of them are measured below; how much each one contributes is not separated.

1. **The text signal is weak.** For frozen SecRoBERTa, a gold pair's cosine is only 0.37 standard deviations above an average technique–CWE pair. The median gold CWE ranks 321 of 944. Only 15 of the 151 gold CWEs ever reach any technique's top-10.
2. **Hubs fill the top-10 lists.** Hardware / SoC CWEs (CWE-1311, -1302, -1267, -1290) take top-10 slots for most techniques. CWE-1311 alone appears in 108 of 183 lists.
3. **The GCNs are trained on near-zero gold signal.** Their cross-graph positives are frozen SecRoBERTa's own top-5 pairs. Only 22 of 14,084 distinct pseudo-pairs are gold pairs (precision 0.16%), and 53% of the technique-side positives are CWE-12xx/13xx (mostly hardware CWEs), against 15% of the pool. Training can at best copy the frozen teacher, including its errors.
4. **The graph side carries no cross-graph information.** The ATT&CK and CWE graphs share no edges, so DeepWalk spaces are unrelated. DeepWalk scores at random (median gold rank 519 against a random 472).
5. **The gold labels are transitive and skewed** (ATT&CK → CAPEC → CWE).
   - Many gold pairs have little or no wording in common: Audio Capture → CWE-267 "Privilege Defined With Unsafe Actions"; OS Credential Dumping → hardware-register CWEs.
   - Two generic CWEs (CWE-284, CWE-200) are gold for 27 techniques each.

Part of the low score is a limit of what the labels allow, not only a model fault (section 6).

## 2. Per-method overview (verified)

| Method | Median rank of gold CWEs | Median best gold rank per technique | Recall@10 | Techniques with any gold in top-10 |
|---|---|---|---|---|
| frozen SecRoBERTa | 321 | 105 | 0.066 | 12.0% |
| InfoNCE dual GCN | 334 | 126 | 0.036 | 7.7% |
| triplet dual GCN | 252 | 131 | 0.016 | 6.6% |
| DeepWalk | 519 | 336 | 0.009 | 3.8% |
| random (expected) | 472 | n/a | ≈0.011 | n/a |

Triplet has the best *median* gold rank but the worst top-10 among the dual models. It moves gold CWEs from the bottom half into the middle (ranks 50–300), not into the top. The top-k metrics do not see that kind of gain.

## 3. Cause 1: weak, compressed text similarity (verified)

| Store | Mean cosine, gold pairs | Mean cosine, all pairs | Std, all pairs | (gold − all) / std |
|---|---|---|---|---|
| frozen | 0.432 | 0.402 | 0.081 | 0.37 |
| triplet | 0.338 | 0.166 | 0.351 | 0.49 |
| InfoNCE | 0.320 | 0.266 | 0.287 | 0.19 |
| DeepWalk | 0.158 | 0.167 | 0.067 | −0.12 |

- **Frozen is anisotropic.** Every vector sits in a narrow cone: the mean within-ATT&CK cosine is 0.56 and the mean within-CWE cosine is 0.44. Mean pooling over generic security prose puts every pair at about 0.40, so small topical overlaps decide the ranks.
- **Text length and form differ.** Technique texts have a median of 1,459 characters and about 4 `(Citation: …)` markers each. CWE texts have a median of 393 characters. Mean pooling of a long narrative against a short definition is a weak match.
- **Cheap probe (not part of the pipeline):** mean-centring each graph's vectors raises frozen Recall@10 from 0.066 to 0.083. Adding a CSLS hub penalty raises it to 0.092. That is real but small, so anisotropy and hubs explain part of the gap, not most of it.

## 4. Cause 2: hubs (verified)

Number of the 183 top-10 lists that contain each hub:

| Store | Most frequent CWEs | Distinct CWEs in all top-10s |
|---|---|---|
| frozen | CWE-1311 (108), CWE-1302 (98), CWE-1267 (93), CWE-1290 (71), CWE-1427 (59) | 257 |
| InfoNCE | CWE-1311 (60), CWE-1267 (59), CWE-1294 (57), CWE-1292 (56), CWE-1290 (54) | 318 |
| triplet | CWE-1292 (54), CWE-1317 (51), CWE-1290 (44), CWE-1302 (43), CWE-1316 (35) | 290 |

- The hub texts are SoC descriptions full of access-control words: "security identifier", "transaction", "access (read/write) assets", "trusted / untrusted".
- Those words overlap with many ATT&CK techniques about privileges and access.
- None of these hubs is a frequent gold CWE.
- The 11.6 fix removed the old CWE-120/CWE-732 hubs, and these hardware hubs took their place.

## 5. Cause 3: the dual GCNs learn from the frozen teacher (verified)

The `rank_topk` training data from `s11b-rank_topk-k5` has these properties:

- Every example has a cross-graph `text_pos`, taken from frozen SecRoBERTa's top-5.
- Anchors are not only techniques. They include malware (3,745), intrusion sets (910), tools, campaigns and mitigations.
- **The pseudo-pairs barely touch the gold pairs.** Overlap with the 625 gold pairs is 22 (3.5% of gold); precision is 0.16%.
- **The positives are concentrated.** The 9,915 technique-side rows use 397 distinct CWEs. The top 5 (CWE-1331, -57, -1248, -1270, -1267) take 31% of the rows. CWE-12xx/13xx take 53%, against 15% of the pool.

**Consequence (inferred):** the GCN is trained to reproduce frozen's top-5, which is mostly hubs. The GNN smoothing over the ATT&CK and CWE graphs then adds graph-neighbour noise. The case studies in section 7 show:

- **InfoNCE** keeps most of frozen's hits (T1574.007 and T1110.002 stay at rank 1). It also loses some, for example T1134.003 goes from 2 to 477.
- **Triplet** reorders more freely. It finds a few gold CWEs frozen misses (T1110.002: CWE-309 at rank 2; T1003: CWE-552 at rank 1) but loses frozen's top hits.

Neither can systematically beat the teacher, because the teacher's labels hold almost no gold information.

## 6. Cause 4: what the gold labels allow (verified data, inferred effect)

### Transitive provenance

Each gold pair is technique → CAPEC → CWE. When one CAPEC maps to many CWEs, all of them are copied onto the technique.

- T1003 OS Credential Dumping gets 7 CWEs through CAPEC-150 "Collect Data from Common Resource Locations". Six of them are hardware data-remanence CWEs (CWE-1239, -1258, -1266, -1272, -1323, -1330).
- T1123 Audio Capture (via CAPEC-634) and T1113 Screen Capture (via CAPEC-648) each have only CWE-267 "Privilege Defined With Unsafe Actions". No text model ranks it near the top: frozen puts it at 910 and 814.

### Skew toward generic CWEs

- CWE-284 Improper Access Control and CWE-200 Exposure of Sensitive Information are each gold for 27 techniques.
- 18 techniques have CWE-284 as their *only* gold, and 17 have only CWE-200. They include Rootkit, Bootkit, Kernel Modules and Install Root Certificate. Frozen ranks CWE-284 at a median of 153 for its techniques.
- The password group (CWE-262, -263, -294, -308, -309, -521, -522) is the gold set of 9 techniques. Frozen never puts CWE-308, -309 or -522 in the top-10 for any of them.
- 183 techniques share only 95 distinct gold sets.

### Popularity reference (leaks the test labels; for scale only)

Ranking CWEs by how often they are gold gives Recall@10 = 0.347. Frozen gets 0.066.

The gold set rewards predicting a few generic, abstract CWEs. Text similarity instead favours specific, lexically close CWEs (CWE-427 for PATH hijacking, CWE-916 for password hashes). That mismatch is a property of the gold set, not a bug.

## 7. Case studies (union, full view; gold ranks out of 944)

| Technique | Number of gold CWEs | Frozen best | InfoNCE best | Triplet best | What happens |
|---|---|---|---|---|---|
| T1574.007 Path Interception by PATH | 10 | 1 (CWE-427) | 1 | 65 | Lexical match: "search path" is in both texts. The other gold CWEs inspected (7 of the remaining 9: CWE-15, -20, -200, -285, -302, -353, -426) rank 52–644 under frozen; they come through CAPEC and the text does not support them. |
| T1110.002 Password Cracking | 8 | 1 (CWE-916) | 1 | 2 (CWE-309) | Good case. The password-policy gold CWEs (CWE-308, -521) rank around 600 for frozen; the technique text is about hashes, not policy. |
| T1134.003 Make and Impersonate Token | 2 | 2 (CWE-384) | 477 | 387 | InfoNCE loses a frozen hit. Its top-5 are user-management and hardware-debug CWEs, which matches graph smoothing toward neighbours. |
| T1040 Network Sniffing | 6 | 350 | 35 | 75 | Frozen top-5 are all hardware hubs (CWE-1267, -1278, -1302, -1290, -1311). The obvious gold CWE-319 Cleartext Transmission ranks 538 for frozen. Training helps a little here. |
| T1078 Valid Accounts | 8 | 12 (CWE-1273) | 31 | 8 | Frozen top-5 include CWE-1302 (hub) and CWE-1220/1268 (hardware access policy). Password-policy gold CWEs rank 47–552. |
| T1003 OS Credential Dumping | 7 | 118 | 29 | 1 (CWE-552) | Six of the 7 gold CWEs are hardware data-remanence CWEs that come through CAPEC-150. Frozen's top hit is CWE-1392 Use of Default Credentials, which is plausible but not gold. |
| T1123 Audio Capture | 1 | 910 | 851 | 611 | The only gold CWE is CWE-267 (via CAPEC-634). Frozen retrieves CWE-69, CWE-1319 (electromagnetic), …. No overlap is possible. |
| T1113 Screen Capture | 1 | 814 | 783 | 588 | Same pattern: CWE-267 via CAPEC-648. Top-5 are path-equivalence and hardware CWEs. |

Frozen's best gold rank is 1 for 6 techniques and above 740 for 8; the worst eight include T1123, T1113, T1115 Clipboard Data, T1125 Video Capture and T1012 Query Registry. All of these are collection or discovery techniques whose CWE gold comes from one CAPEC.

## 8. Minor text issue found (verified, small)

- `_flatten_block` joins only `xhtml:p` children. When a description mixes `<p>` and `<ul>`, the list items are dropped. This behaviour predates 11.6.
- Example: CWE-284's Extended_Description has 1,251 characters but only 357 are kept, so the text reads "such as:When…".
- Of the 284 Description / Extended_Description blocks that use `xhtml:p`, 60 lose more than 20 characters this way.
- This affects a few gold CWEs, including CWE-284, but it is unlikely to be a main cause.

## 9. What is verified and what is not

- **Verified (computed above):** the rank and cosine statistics, the hub counts, pseudo-pair overlap and concentration, the gold skew, the CAPEC provenance of the cases, the centring/CSLS probe values, and the dropped list text.
- **Inferred, not tested:**
  - that GNN smoothing causes the InfoNCE and triplet losses in specific cases;
  - that the length and citation mismatch hurts mean pooling;
  - how much each cause contributes. There is no ablation that separates them.
- **Not checked:** other gold sources (stix, capec, intersection); the pool view; other text encoders (SecureBERT2); the `full_metadata` text variant; per-seed variation.

## 10. Options suggested by this analysis (not done)

1. **Hub correction at retrieval time** (mean-centring, CSLS). This is cheap and gives +0.03 Recall@10 for frozen in the probe. It needs an evaluator or embedder option and should be reported as a separate row.
2. **Better pseudo-pairs:**
   - anchor only on `attack-pattern` nodes;
   - drop or down-weight CWEs that are top-k for many anchors;
   - add CAPEC as a bridge graph, so cross-graph edges exist and DeepWalk / GCN have a real path.
3. **Text:**
   - strip `(Citation: …)` markers and markdown links from ATT&CK texts;
   - keep `<ul>/<li>` content in CWE texts;
   - try the CWE name plus description only, or a sentence-level encoder instead of mean-pooled MLM states.
4. **Evaluation:** report results per CWE abstraction level (Pillar/Class vs Base/Variant) and add a "gold-popularity" reference row, so the low ceiling caused by generic labels is visible.
5. **Treat it as a labelling question:** decide whether CWE-267 for Audio/Screen Capture and hardware CWEs for Credential Dumping should count as positives for a text model. Option: restrict to the `intersection` source or to Base/Variant CWEs for a "text-recoverable" subset.
