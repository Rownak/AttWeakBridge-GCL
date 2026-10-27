"""Write a seeded random-vector embedding store (Sprint 10.6 baseline) in the usual store layout.

The store has the node ids (and order) of an existing reference store and standard-normal vectors of its dimension, so it
runs through the existing ``embedding_store`` method with no new method kind and its rows have the same ranking queries as
the reference encoder's. It is written to ``artifacts/embeddings/{dataset}/random-{label}/{run_id}`` with a ``LATEST``
pointer. The run id is derived from the seed and the reference store, so a repeat run reproduces it.

Usage:
    uv run python scripts/make_random_store.py --dataset attack_cwe_ent \\
        --reference artifacts/embeddings/attack_cwe_ent/hf_mean_pool-SecureBERT2/<run id> --seed 42 --label rand42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cga.core.io import write_json
from cga.data.aligned import AlignedMatrix


def make_random_store(reference: Path, out_lineage: Path, *, seed: int, label: str, dataset: str) -> Path:
    """Write a random store with ``reference``'s node ids and dimension; return the run directory.

    Args:
        reference: Reference store run directory (node ids and dimension are copied).
        out_lineage: Lineage directory ``.../{dataset}/random-{label}`` (created if needed).
        seed: RNG seed of the Gaussian draw.
        label: Store label (``rand42``); recorded in the manifest.
        dataset: Dataset id, recorded in the manifest.

    Returns:
        The run directory that holds ``vectors.npy``, ``index.parquet`` and ``manifest.json``.
    """
    ref = AlignedMatrix.load(reference)
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal(ref.vectors.shape).astype(np.float32)
    tag = hashlib.sha256(f"{seed}|{reference.name}|{ref.vectors.shape}".encode("utf-8")).hexdigest()[:8]
    run_id = f"rand-s{seed}-{tag}"
    run_dir = out_lineage / run_id
    AlignedMatrix(vectors=vectors, node_ids=list(ref.node_ids)).save(run_dir)
    # Extend the store manifest with the usual identity fields; the hash keys written by save() stay.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "run_id": run_id,
            "pipeline": "make_random_store",
            "dataset_id": dataset,
            "seed": seed,
            "label": label,
            "reference_store": str(reference),
            "note": "standard-normal vectors; a random-ranking baseline, not an encoder",
        }
    )
    write_json(manifest, manifest_path)
    (out_lineage / "LATEST").write_text(run_id, encoding="utf-8")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", required=True, help="e.g. rand42; the embedder id is random-<label>")
    ap.add_argument("--root", type=Path, default=Path("artifacts/embeddings"))
    args = ap.parse_args()
    run_dir = make_random_store(
        args.reference, args.root / args.dataset / f"random-{args.label}", seed=args.seed, label=args.label, dataset=args.dataset
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
