"""Optional t-SNE scatter of learned embeddings, colored by ``graph_role`` (§5.5a).

Ported from ``legacy/GCL/gcl_analysis/1_tsne_plot.py::tsne_plot1``. Visualization
only — this module never computes or returns a number that feeds ``metrics.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cga.data.aligned import AlignedMatrix


def plot_tsne_by_role(
    matrix: AlignedMatrix,
    graph_role_by_node: dict[str, str],
    output_path: str | Path,
    *,
    seed: int,
    perplexity: float = 25.0,
) -> None:
    """Render a 2D t-SNE scatter of ``matrix``'s vectors, colored by ``graph_role``.

    Args:
        matrix: Learned node embeddings to visualize.
        graph_role_by_node: ``node_id`` -> ``graph_role`` for every row in ``matrix``.
        output_path: PNG file to write.
        seed: Random seed, for a reproducible layout given the same inputs.
        perplexity: t-SNE perplexity; clamped below ``n_samples`` when the
            universe is small (t-SNE requires ``perplexity < n_samples``).

    Side Effects:
        Writes a PNG to ``output_path``; does not return or log any metric.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    output_path = Path(output_path)
    n_samples = matrix.vectors.shape[0]
    effective_perplexity = min(perplexity, max(1.0, (n_samples - 1) / 3))

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        max_iter=5000,
        random_state=seed,
    )
    coords = tsne.fit_transform(matrix.vectors)
    roles = np.array([graph_role_by_node.get(nid, "unknown") for nid in matrix.node_ids])

    fig, ax = plt.subplots(figsize=(8, 5))
    for role in sorted(set(roles.tolist())):
        mask = roles == role
        ax.scatter(coords[mask, 0], coords[mask, 1], label=role, alpha=0.6)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
