"""Descriptive readout of what the SRGA-GAT adapter (A14) attends to (Sprint 5, Phase 5.6).

Rebuilds each seed's A14 model from its final checkpoint, runs the first ``GATConv`` layer on the
whole graph with ``return_attention_weights=True``, and averages the attention over heads. Each target
node's attention sums to 1 over its incoming edges plus its self-loop, so the first table reports how
that mass splits between the node itself, its **parents** and its **children**, next to the share a
uniform attention would give each (``count / (degree + 1)``). The undirected GCN of A11 cannot make
the parent/child distinction; if GAT's attention differs by direction, it learned something the GCN
cannot express.

The second table looks at the queries where SapBERT's top confuser is a 1-hop neighbour of the gold
target in DOID: how much attention does the gold target give that confuser, compared with its other
neighbours and with uniform attention?

This describes the model; it is not an explanation of why its ranking is right. Reads artifacts only
and writes nothing.

Example::

    uv run python scripts/srga_gat_attention.py --dataset ncit_doid_srga --split test \
        --model-id srga-gat-l0.25-s{seed}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from cga.data.aligned import AlignedMatrix, NodeUniverse
from cga.data.graph_store import read_edges
from cga.pipelines.gcl_train.models import _unit, build_residual_adapter_encoder
from srga_paired_ranks import _latest, query_ranks


def _build_graph(ingest_dir: Path, row_of: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the symmetrized ``edge_index`` and ``edge_type`` exactly as the ``gcl_train`` load step builds them."""
    pairs, types = [], []
    edges = read_edges(ingest_dir / "edges.parquet")
    for src, dst in zip(edges.column("src_id").to_pylist(), edges.column("dst_id").to_pylist()):
        if src in row_of and dst in row_of:
            pairs += [(row_of[src], row_of[dst]), (row_of[dst], row_of[src])]
            types += [0, 1]  # 0 = stored child -> parent, 1 = reversed (parent -> child)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous(), torch.tensor(types, dtype=torch.long)


def attention_of_run(run_dir: Path) -> dict:
    """Load one A14 run and return its first-layer, head-averaged attention.

    Args:
        run_dir: A ``gcl_train`` run directory (manifest, resolved config, final checkpoint).

    Returns:
        Dict with ``mass`` (per node: ``self``/``parent``/``child`` attention), ``count`` (per node
        parent and child neighbour counts), ``edge`` (``(source_row, target_row) -> (alpha, kind)`` with
        ``kind`` ``"parent"`` when the source is a parent of the target, else ``"child"``), ``row_of``
        and ``roles``.
    """
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    path = {i["name"]: Path(i["path"]) for i in manifest["inputs"]}
    cfg = yaml.safe_load((run_dir / "config.resolved.yaml").read_text(encoding="utf-8"))["gcl_train"]["model"]
    universe = NodeUniverse.from_parquet(path["training_data"] / "universe.parquet")
    row_of = {n: i for i, n in enumerate(universe.node_ids)}
    text = AlignedMatrix.load(path["text_embeddings"]).select(universe.node_ids).vectors
    x_hat = _unit(torch.tensor(text, dtype=torch.float32))
    edge_index, edge_type = _build_graph(path["ingest"], row_of)

    model = build_residual_adapter_encoder(
        adapter="gat",
        lam=cfg.get("lambda", 0.25),
        adapter_dim=cfg.get("adapter_dim", 256),
        text_dim=x_hat.shape[1],
        gat_heads=tuple(cfg.get("gat_heads", [8, 1])),
        residual_floor=cfg.get("residual_floor", True),
    )
    checkpoint = sorted((run_dir / "checkpoints").glob("epoch_*.pt"))[-1]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        _, (ei, alpha) = model.gat.gat1(x_hat, edge_index, return_attention_weights=True)
    alpha = alpha.mean(dim=1).numpy()  # average over heads; ``ei`` also holds the appended self-loops

    type_of = {(int(s), int(d)): int(t) for s, d, t in zip(*edge_index.numpy(), edge_type.numpy())}
    n = len(universe.node_ids)
    mass = {k: np.zeros(n) for k in ("self", "parent", "child")}
    count = {k: np.zeros(n) for k in ("parent", "child")}
    edge: dict[tuple[int, int], tuple[float, str]] = {}
    for s, d, a in zip(ei[0].tolist(), ei[1].tolist(), alpha):
        if s == d:
            mass["self"][d] += a
            continue
        kind = "parent" if type_of[(s, d)] == 1 else "child"  # a type-1 (reversed) edge delivers a parent's vector
        mass[kind][d] += a
        count[kind][d] += 1
        edge[(s, d)] = (float(a), kind)
    return {"mass": mass, "count": count, "edge": edge, "row_of": row_of, "roles": universe.graph_roles}


def main() -> None:
    """Print the attention-mass table by graph role and the confuser-attention table (mean over seeds)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ncit_doid_srga")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-id", required=True, help="model_id with '{seed}'")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()

    root = Path.cwd()
    ingest_dir = _latest(root / "data" / "processed" / args.dataset)
    sapbert = {"A0": AlignedMatrix.load(_latest(root / "artifacts" / "embeddings" / args.dataset / "hf_mean_pool-sapbert"))}
    queries, _, confusers = query_ranks(sapbert, ingest_dir, args.split)

    mass_rows: dict[str, list[list[float]]] = {"graph_1": [], "graph_2": []}
    conf_rows: dict[str, list[list[float]]] = {"parent": [], "child": []}
    n_queries = {"parent": 0, "child": 0}
    for seed in args.seeds:
        att = attention_of_run(_latest(root / "models" / "gcl" / args.dataset / args.model_id.format(seed=seed)))
        m, cnt = att["mass"], att["count"]
        degree = cnt["parent"] + cnt["child"]
        for role in mass_rows:
            in_role = np.array([r == role for r in att["roles"]])
            has_p, has_c = in_role & (cnt["parent"] > 0), in_role & (cnt["child"] > 0)
            any_nbr = in_role & (degree > 0)
            mass_rows[role].append(
                [
                    float(m["self"][any_nbr].mean()),
                    float((1.0 / (degree[any_nbr] + 1)).mean()),
                    float(m["parent"][has_p].mean()),
                    float((cnt["parent"][has_p] / (degree[has_p] + 1)).mean()),
                    float(m["child"][has_c].mean()),
                    float((cnt["child"][has_c] / (degree[has_c] + 1)).mean()),
                ]
            )
        incoming: dict[int, list[tuple[int, float, str]]] = {}
        for (s, d), (a, kind) in att["edge"].items():
            incoming.setdefault(d, []).append((s, a, kind))
        per_kind: dict[str, list[tuple[float, float, float]]] = {"parent": [], "child": []}
        for (_, target), confuser in zip(queries, confusers["A0"]):
            t, c = att["row_of"][target], att["row_of"][confuser]
            nbrs = incoming.get(t, [])
            hit = [(a, kind) for s, a, kind in nbrs if s == c]
            if not hit:
                continue  # the confuser is not a 1-hop neighbour of the target
            a_conf, kind = hit[0]
            others = [a for s, a, _ in nbrs if s != c]
            per_kind[kind].append((a_conf, float(np.mean(others)) if others else float("nan"), 1.0 / (len(nbrs) + 1)))
        for kind, rows in per_kind.items():
            arr = np.array(rows, dtype=float)
            conf_rows[kind].append(list(np.nanmean(arr, axis=0)))
            n_queries[kind] = len(rows)

    def cell(rows: list[list[float]], i: int) -> str:
        v = np.array([r[i] for r in rows])
        return f"{v.mean():.3f}" + (f" +- {v.std(ddof=1):.3f}" if len(v) > 1 else "")

    print(f"model_id={args.model_id} seeds={args.seeds} (first GATConv layer, mean over {8} heads)")
    print("\nAttention mass of a node (self + parents + children = 1), mean over nodes with at least one neighbour")
    print("| graph | self | self if uniform | parents | parents if uniform | children | children if uniform |")
    print("|---|---|---|---|---|---|---|")
    for role, label in (("graph_1", "NCIT"), ("graph_2", "DOID")):
        print(f"| {label} | " + " | ".join(cell(mass_rows[role], i) for i in range(6)) + " |")
    print(f"\nQueries ({args.split}) whose SapBERT top confuser is a 1-hop neighbour of the gold target (DOID):")
    print("| confuser relation | n queries | attention to the confuser | mean attention to the target's other neighbours | uniform 1/(deg+1) |")
    print("|---|---|---|---|---|")
    for kind in ("parent", "child"):
        print(f"| {kind} | {n_queries[kind]} | " + " | ".join(cell(conf_rows[kind], i) for i in range(3)) + " |")


if __name__ == "__main__":
    main()
