"""Copy an embedding store to another dataset lineage (without ``shards/``) and verify it against that lineage's ingest.

Used when a new dataset id re-ingests the same nodes (e.g. ``ncit_doid_unsup`` from ``ncit_doid_srga_full``) so the
frozen BERT / DeepWalk vectors need not be recomputed. The run id is kept, so ``manifest.json`` stays truthful about
the vectors; its recorded ``inputs`` still name the source ingest.

Checks (all must pass, otherwise the destination is removed and the script exits non-zero):

1. every copied file is byte-identical to its source;
2. ``AlignedMatrix.load`` accepts the copy (it re-checks the hashes recorded in ``manifest.json``);
3. the store's ``node_id`` order is the destination ingest's ``nodes.parquet`` order restricted to the stored nodes
   (a store may omit nodes with empty text or no edges, but may not reorder or add any);
4. with ``--text-variant``: the ``input_sha256`` of each ``ok`` row equals the ingest's ``text_sha256`` for that variant.

Usage:
    uv run python scripts/copy_embedding_store.py \\
        --src artifacts/embeddings/ncit_doid_srga_full/hf_mean_pool-bert_base/20261009T191700Z-1278c23a \\
        --dst-lineage artifacts/embeddings/ncit_doid_unsup/hf_mean_pool-bert_base \\
        --ingest data/processed/ncit_doid_unsup/20261026T024157Z-3bde93ef --text-variant label_synonyms_definition
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

from cga.data.aligned import AlignedMatrix

_FILES = ("vectors.npy", "index.parquet", "status.parquet", "manifest.json", "stats.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_and_verify(src: Path, dst_lineage: Path, ingest: Path, text_variant: str | None) -> list[str]:
    """Copy ``src`` into ``dst_lineage/{run_id}`` and verify it; return the human-readable check results.

    Args:
        src: Source run directory (``.../{embedder_id}/{run_id}``).
        dst_lineage: Destination ``.../{dataset}/{embedder_id}`` directory (created if needed).
        ingest: Destination dataset's ingest run directory.
        text_variant: Text variant to hash-check against (text embedders only).

    Returns:
        One line per check.

    Raises:
        AssertionError: If a check fails (the caller removes the copy).
    """
    dst = dst_lineage / src.name
    dst.mkdir(parents=True, exist_ok=False)
    for name in _FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    report = []

    for name in _FILES:
        if (src / name).is_file():
            assert _sha256(src / name) == _sha256(dst / name), f"{name} differs from the source"
    report.append("files byte-identical to the source")

    matrix = AlignedMatrix.load(dst)
    report.append(f"AlignedMatrix.load ok: {matrix.vectors.shape}")

    ingest_ids = pq.read_table(ingest / "nodes.parquet").column("node_id").to_pylist()
    stored = set(matrix.node_ids)
    assert stored <= set(ingest_ids), f"{len(stored - set(ingest_ids))} stored nodes are not in the ingest"
    assert matrix.node_ids == [n for n in ingest_ids if n in stored], "stored node order differs from the ingest order"
    report.append(f"node order matches the ingest: {len(stored)} of {len(ingest_ids)} ingest nodes stored")

    if text_variant:
        texts = pq.read_table(ingest / "texts.parquet").to_pylist()
        ingest_sha = {r["node_id"]: r["text_sha256"] for r in texts if r["text_variant"] == text_variant}
        status = pq.read_table(dst / "status.parquet").to_pylist()
        ok = [r for r in status if r["status"] == "ok"]
        n_match = sum(1 for r in ok if ingest_sha.get(r["node_id"]) == r["input_sha256"])
        assert n_match == len(ok), f"input_sha256 matches the '{text_variant}' text for only {n_match} of {len(ok)} nodes"
        report.append(f"input_sha256 matches '{text_variant}' text for {n_match} of {len(ok)} ok nodes")

    (dst_lineage / "LATEST").write_text(src.name, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst-lineage", type=Path, required=True)
    parser.add_argument("--ingest", type=Path, required=True)
    parser.add_argument("--text-variant", default=None)
    args = parser.parse_args()
    try:
        for line in copy_and_verify(args.src, args.dst_lineage, args.ingest, args.text_variant):
            print("ok:", line)
    except AssertionError as err:
        shutil.rmtree(args.dst_lineage / args.src.name, ignore_errors=True)
        print(f"FAILED: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
