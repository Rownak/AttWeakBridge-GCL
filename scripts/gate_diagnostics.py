"""Gate diagnostics of a gated-residual ``gcl_train`` run (Sprint 12 Phase 12.5).

Reads ``node_embeddings/gate.npy`` (per-node mean gate, ``(n,)``; or ``(n, 2)`` text / graph for the dual variant) and
``node_embeddings/index.parquet`` of a model run, the node roles and edges of the pinned ingest, and reports per graph
role: the mean / median / 5th / 95th percentile of the gate, the share of nodes with gate below 0.05 and above 0.95,
and the Spearman correlation of the gate with the node degree (undirected degree in ``edges.parquet``). It also draws
a histogram per role and a gate-vs-degree scatter. Read-only on the run and the ingest; it writes
``gate_diagnostics.md`` / ``.json`` and two PNGs into ``--out``.

Example::

    uv run python scripts/gate_diagnostics.py \\
        --run models/gcl/attack_cwe_ent/ent-g3-gated-mpnetc-s42/<run> \\
        --ingest data/processed/attack_cwe_ent/20261027T194643Z-5d3f8baf --out artifacts/paper/gate/g3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LOW, HIGH = 0.05, 0.95
ROLE_NAMES = {"graph_1": "ATT&CK", "graph_2": "CWE"}


def node_degree(edges: pd.DataFrame, node_ids: list[str]) -> np.ndarray:
    """Undirected degree of each node (number of distinct neighbours) in an edge table.

    Args:
        edges: Table with ``src_id`` and ``dst_id``.
        node_ids: Nodes to report, in output order.

    Returns:
        ``(len(node_ids),)`` integer degrees; nodes without edges get 0.
    """
    pairs = pd.concat(
        [edges[["src_id", "dst_id"]].rename(columns={"src_id": "a", "dst_id": "b"}),
         edges[["dst_id", "src_id"]].rename(columns={"dst_id": "a", "src_id": "b"})]
    ).drop_duplicates()
    pairs = pairs[pairs["a"] != pairs["b"]]
    counts = pairs.groupby("a").size()
    return counts.reindex(node_ids).fillna(0).to_numpy(dtype=int)


def gate_stats(gate: np.ndarray, degree: np.ndarray) -> dict[str, float]:
    """Summarise one gate column against node degree.

    Args:
        gate: ``(m,)`` per-node mean gate of the nodes of one graph role.
        degree: ``(m,)`` degrees of the same nodes.

    Returns:
        Dict with ``n``, ``mean``, ``median``, ``p05``, ``p95``, ``share_below_0.05``, ``share_above_0.95`` and
        ``spearman_degree`` (``nan`` if either series is constant).
    """
    rho = pd.Series(gate).rank().corr(pd.Series(degree).rank())
    return {
        "n": int(len(gate)),
        "mean": float(gate.mean()),
        "median": float(np.median(gate)),
        "p05": float(np.percentile(gate, 5)),
        "p95": float(np.percentile(gate, 95)),
        "share_below_0.05": float((gate < LOW).mean()),
        "share_above_0.95": float((gate > HIGH).mean()),
        "spearman_degree": float(rho),
    }


def diagnose(run_dir: Path, ingest_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Compute the per-role statistics of every gate column of a run.

    Args:
        run_dir: Model run directory holding ``node_embeddings/gate.npy`` and ``index.parquet``.
        ingest_dir: Ingest run directory with ``nodes.parquet`` (roles) and ``edges.parquet``.

    Returns:
        ``{gate_column: {graph_role: stats}}`` where ``gate_column`` is ``"text"`` / ``"graph"`` for the dual variant
        and ``"gate"`` otherwise.

    Raises:
        FileNotFoundError: If the run has no ``gate.npy`` (the model has no learned gate).
        ValueError: If the gate rows do not match the exported node ids.
    """
    gate = np.load(run_dir / "node_embeddings" / "gate.npy")
    node_ids = pd.read_parquet(run_dir / "node_embeddings" / "index.parquet").sort_values("row")["node_id"].tolist()
    if gate.shape[0] != len(node_ids):
        raise ValueError(f"gate has {gate.shape[0]} rows, index has {len(node_ids)}")
    nodes = pd.read_parquet(ingest_dir / "nodes.parquet").set_index("node_id")
    roles = nodes.loc[node_ids, "graph_role"].to_numpy()
    degree = node_degree(pd.read_parquet(ingest_dir / "edges.parquet"), node_ids)
    columns = {"gate": gate} if gate.ndim == 1 else {"text": gate[:, 0], "graph": gate[:, 1]}
    return {
        name: {
            role: gate_stats(values[roles == role], degree[roles == role]) for role in sorted(set(roles.tolist()))
        }
        for name, values in columns.items()
    }


def to_markdown(report: dict[str, dict[str, dict[str, float]]]) -> str:
    """Render the statistics as a Markdown table (one row per gate column and graph role)."""
    header = "| Gate | Graph | n | mean | median | p05 | p95 | share < 0.05 | share > 0.95 | Spearman(degree) |"
    lines = [header, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, by_role in report.items():
        for role, s in by_role.items():
            lines.append(
                f"| {name} | {ROLE_NAMES.get(role, role)} | {s['n']} | {s['mean']:.4f} | {s['median']:.4f} | "
                f"{s['p05']:.4f} | {s['p95']:.4f} | {s['share_below_0.05']:.3f} | {s['share_above_0.95']:.3f} | "
                f"{s['spearman_degree']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def plot(run_dir: Path, ingest_dir: Path, out: Path) -> None:
    """Write ``gate_hist.png`` (histogram per role and gate column) and ``gate_vs_degree.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate = np.load(run_dir / "node_embeddings" / "gate.npy")
    node_ids = pd.read_parquet(run_dir / "node_embeddings" / "index.parquet").sort_values("row")["node_id"].tolist()
    nodes = pd.read_parquet(ingest_dir / "nodes.parquet").set_index("node_id")
    roles = nodes.loc[node_ids, "graph_role"].to_numpy()
    degree = node_degree(pd.read_parquet(ingest_dir / "edges.parquet"), node_ids)
    columns = {"gate": gate} if gate.ndim == 1 else {"text gate": gate[:, 0], "graph gate": gate[:, 1]}
    role_list = sorted(set(roles.tolist()))
    for kind in ("hist", "vs_degree"):
        fig, axes = plt.subplots(len(columns), len(role_list), figsize=(4 * len(role_list), 3 * len(columns)), squeeze=False)
        for i, (name, values) in enumerate(columns.items()):
            for j, role in enumerate(role_list):
                ax, mask = axes[i][j], roles == role
                if kind == "hist":
                    ax.hist(values[mask], bins=40)
                    ax.set_xlabel(name)
                    ax.set_ylabel("nodes")
                else:
                    ax.scatter(np.log1p(degree[mask]), values[mask], s=4, alpha=0.5)
                    ax.set_xlabel("log(1 + degree)")
                    ax.set_ylabel(name)
                ax.set_title(ROLE_NAMES.get(role, role))
        fig.tight_layout()
        fig.savefig(out / f"gate_{kind}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    """Parse arguments, compute the diagnostics and write the report and figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="model run directory (models/gcl/<dataset>/<model_id>/<run_id>)")
    ap.add_argument("--ingest", required=True, help="ingest run directory (nodes.parquet, edges.parquet)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    run_dir, ingest_dir, out = Path(args.run), Path(args.ingest), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = diagnose(run_dir, ingest_dir)
    (out / "gate_diagnostics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "gate_diagnostics.md").write_text(to_markdown(report), encoding="utf-8", newline="\n")
    plot(run_dir, ingest_dir, out)
    print(to_markdown(report))


if __name__ == "__main__":
    main()
