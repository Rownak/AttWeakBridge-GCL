## 3. Idea: gated residual graph adapter over frozen text

### Core novelty
Replace the plain GCN encoder with `z_i = LN( x_i + g_i ⊙ f(Â X)_i )`, where `x` is the frozen mpnet-chunk vector, `f` is a small graph transform, and the gate `g_i = σ(W [x_i ‖ (ÂX)_i] + b)` is per-node with bias initialised so `g ≈ 0`. The model **starts exactly at frozen text** and structure is added only where it reduces the loss. For the dual model, the same gate arbitrates between the text and graph branches instead of concatenation, so the weak, unaligned DeepWalk branch cannot degrade text (the A0 < A1 problem). The learned gate value is also an interpretable per-node "how much structure is trusted" score, usable as an uncertainty signal at inference (e.g. fall back to text for low-gate nodes).

### Supporting work
- GCNII (Chen et al., ICML 2020): initial residual to counter over-smoothing.
- Highway networks (Srivastava et al., 2015) and gated GNNs; adapters (Houlsby et al., ICML 2019) for the zero-init residual idea.
- The in-repo SapBERT residual adapter design ([sapbert_residual_graph_adapter.md](sapbert_residual_graph_adapter.md)) argues the same for NCIT–DOID; this idea ports it to the unsupervised setting and adds the gate as the contribution.

### Why it fits
Frozen mpnet-chunk is already strong and the pseudo-labels are noisy, so a model that is free to rewrite the embedding can overfit the noise, while the structural branch alone is below frozen. A constrained residual is the natural fix and makes the "does structure help?" question answerable per node.

### Minimal change
Model code only (`gcl_train` model): a `residual_gated` encoder option; loss, data and evaluation unchanged. Gate initialisation and a small weight decay on `W_g`.

### Inputs / gold
Same as today. **No gold needed.**

### Ablation
Same pseudo-label data as 11.14 (`s11c-rank_topk-k5`, bottom-5 and hard), 3–5 seeds:
1. current GCN;
2. residual, no gate (`x + f(Âx)`);
3. gated residual;
4. gated residual, gate fixed to 0 (should equal frozen: sanity check);
5. dual model with gate arbitration vs. concatenation.

Also report the gate histogram and its correlation with node degree. Success criterion: A0-gated ≥ A1 within noise and A2-style structure never below frozen text.

### Risks
- The gate may collapse to 0 (no benefit) or to 1 (reduces to plain GCN); both are informative, but report them as such.
- Zero-init gates can train slowly with a small learning rate and 100 epochs; may need a longer schedule, which confounds comparison with 11.14 (match epochs in the baseline).
- Moderate novelty by itself; it is strongest when combined with Idea 1.