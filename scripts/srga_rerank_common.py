"""Shared inputs for the Sprint 6.2 re-ranker scripts (probe and fit): first stage, score sources, ingest rows.

The first stage is A19 = 0.5 * (A18 seed ensemble of A12) + 0.5 * (A17 seed ensemble of A11) (Phase 6.1);
the ``a11`` / ``a12`` cosine features are those two ensembles. Reads artifacts only.
"""

from __future__ import annotations

from pathlib import Path

from cga.data.aligned import AlignedMatrix
from cga.data.graph_store import read_alignments, read_edges, read_nodes
from cga.pipelines.evaluate.methods import fuse_matrices

A11_PATTERN = "srga-gcn-l0.25-s{seed}"
A12_PATTERN = "gcn-full-e200-s{seed}"
FIRST_STAGE_IDS = {"sapbert": "hf_mean_pool-sapbert", "a11": A11_PATTERN, "a12": A12_PATTERN, "first": "0.5*a11 + 0.5*a12"}


def latest(lineage: Path) -> Path:
    """Return the run directory a lineage's ``LATEST`` file points to."""
    return lineage / (lineage / "LATEST").read_text(encoding="utf-8").strip()


def load_inputs(dataset: str, seeds: list[int], w_a12: float = 0.5):
    """Load the score sources and ingest tables.

    Returns:
        ``(matrices, role_of, alignment_rows, edge_rows)`` where ``matrices`` maps
        ``sapbert / a11 / a12 / first`` to an ``AlignedMatrix``.
    """
    root = Path.cwd()
    ingest = latest(root / "data" / "processed" / dataset)
    role_of = {r["node_id"]: r["graph_role"] for r in read_nodes(ingest / "nodes.parquet").to_pylist()}
    align = read_alignments(ingest / "alignments.parquet").to_pylist()
    edges = read_edges(ingest / "edges.parquet").to_pylist()

    def members(pattern: str) -> list[AlignedMatrix]:
        return [AlignedMatrix.load(latest(root / "models" / "gcl" / dataset / pattern.format(seed=s)) / "node_embeddings") for s in seeds]

    m11, m12 = members(A11_PATTERN), members(A12_PATTERN)
    matrices = {
        "sapbert": AlignedMatrix.load(latest(root / "artifacts" / "embeddings" / dataset / "hf_mean_pool-sapbert")),
        "a11": fuse_matrices([(m, 1.0 / len(m11)) for m in m11]),
        "a12": fuse_matrices([(m, 1.0 / len(m12)) for m in m12]),
        "first": fuse_matrices(
            [(m, (1 - w_a12) / len(m11)) for m in m11] + [(m, w_a12 / len(m12)) for m in m12]
        ),
    }
    return matrices, role_of, align, edges
