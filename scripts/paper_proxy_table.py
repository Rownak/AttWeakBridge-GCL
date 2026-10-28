"""Paper comparison table of a Sprint 3 proxy-protocol evaluate run (Sprint 12 Phase 12.4).

Reads ``metrics.parquet`` of an evaluate run that holds a ``neighborhood`` evaluation (views ``structural_r1``,
``structural_r2``, ``textual``; roles ``graph_1`` / ``graph_2``) and a ``cluster_quality`` evaluation, and writes
``paper_table.md`` and ``paper_table.csv`` next to it. One row per method (in the order they first appear in the
file, i.e. the experiment config order). Columns: MRR and normalized precision@10 for each view and graph
(ATT&CK = ``graph_1``, CWE = ``graph_2``), and the mean silhouette over all ``n_clusters`` values and both graphs.
The Markdown table bolds the best value of each column; the CSV holds plain numbers. Output is deterministic, so the
same run always regenerates byte-identical files.

Example::

    uv run python scripts/paper_proxy_table.py --dataset attack_cwe_ent --run 20261028T044615Z-b20dd0c5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

VIEWS = ("structural_r1", "structural_r2", "textual")
ROLES = (("graph_1", "ATT&CK"), ("graph_2", "CWE"))
PRECISION_K = 10
DIGITS = 4


def build_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the wide comparison table from a long ``metrics.parquet`` frame.

    Args:
        metrics: Long metrics frame with columns ``evaluation_kind``, ``method_id``, ``method_label``, ``view_id``,
            ``graph_role``, ``param_value``, ``metric_name`` and ``metric_value``.

    Returns:
        Frame indexed by method label (config order) with one float column per ``"<view> <graph> MRR"`` /
        ``"<view> <graph> P@10"`` and a final ``"Silhouette"`` column.

    Raises:
        ValueError: If a required (method, view, role) cell or a method's silhouette is missing.
    """
    methods = metrics.drop_duplicates("method_id")[["method_id", "method_label"]]
    neigh = metrics[metrics["evaluation_kind"] == "neighborhood"]
    clusters = metrics[(metrics["evaluation_kind"] == "cluster_quality") & (metrics["metric_name"] == "silhouette")]
    rows = {}
    for method_id, label in zip(methods["method_id"], methods["method_label"]):
        sub = neigh[neigh["method_id"] == method_id]
        row = {}
        for view in VIEWS:
            for role, role_name in ROLES:
                cell = sub[(sub["view_id"] == view) & (sub["graph_role"] == role)]
                # MRR does not depend on k (it is repeated per k); the mean is the single value.
                mrr = cell[cell["metric_name"] == "mrr"]["metric_value"]
                prec = cell[(cell["metric_name"] == "normalized_precision_at_k") & (cell["param_value"] == PRECISION_K)][
                    "metric_value"
                ]
                if mrr.empty or prec.empty:
                    raise ValueError(f"missing cell: method={method_id} view={view} role={role}")
                row[f"{view} {role_name} MRR"] = float(mrr.mean())
                row[f"{view} {role_name} P@{PRECISION_K}"] = float(prec.iloc[0])
        sil = clusters[clusters["method_id"] == method_id]["metric_value"]
        if sil.empty:
            raise ValueError(f"missing silhouette: method={method_id}")
        row["Silhouette"] = float(sil.mean())
        rows[label] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def to_markdown(table: pd.DataFrame) -> str:
    """Render the table as GitHub Markdown with the best value per column in bold.

    Args:
        table: Output of :func:`build_table`.

    Returns:
        Markdown text ending in a newline. Ties for best are all bolded.
    """
    best = table.round(DIGITS).max(axis=0)
    header = "| Method | " + " | ".join(table.columns) + " |"
    sep = "|---|" + "---:|" * len(table.columns)
    lines = [header, sep]
    for label, row in table.round(DIGITS).iterrows():
        cells = []
        for col, value in row.items():
            text = f"{value:.{DIGITS}f}"
            cells.append(f"**{text}**" if value == best[col] else text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_table(run_dir: Path) -> tuple[Path, Path]:
    """Write ``paper_table.md`` and ``paper_table.csv`` into an evaluate run directory.

    Args:
        run_dir: Evaluate run directory containing ``metrics.parquet``.

    Returns:
        The ``(markdown, csv)`` paths written.

    Side effects:
        Creates or overwrites the two files in ``run_dir``.
    """
    table = build_table(pd.read_parquet(run_dir / "metrics.parquet"))
    md_path, csv_path = run_dir / "paper_table.md", run_dir / "paper_table.csv"
    md_path.write_text(to_markdown(table), encoding="utf-8", newline="\n")
    table.round(DIGITS).to_csv(csv_path, index_label="method", lineterminator="\n")
    return md_path, csv_path


def main() -> None:
    """Parse arguments and write the table for the given evaluate run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True, help="evaluate run id")
    ap.add_argument("--root", default="artifacts/evaluations", help="evaluations root directory")
    args = ap.parse_args()
    for path in write_table(Path(args.root) / args.dataset / args.run):
        print(path)


if __name__ == "__main__":
    main()
