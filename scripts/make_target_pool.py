"""Write the shared target-side candidate pool (Sprint 10.6): the CWE nodes that appear in the ``union`` gold.

The pool is the set of distinct ``node_2_id`` of one ingest run's ``alignments.parquet`` (the ``union`` run). For every
other ingest run given with ``--check`` the script verifies that all of its gold targets lie inside the pool and exits
non-zero otherwise, so one pool can be shared by all four gold sources.

The output is a JSON object ``{"node_ids": [...sorted...], "source_run": ..., "sha256": ..., "n": ...}`` read by the
``target_pool`` option of the ``gold_alignment`` and ``gold_alignment_prf`` evaluators. The pool is built from gold, so a
pool view is optimistic by construction; read it next to a random baseline on the same pool.

Usage:
    uv run python scripts/make_target_pool.py \\
        --ingest data/processed/attack_cwe_ent/<union run> \\
        --check data/processed/attack_cwe_ent/<stix run> --check ... \\
        --out data/processed/attack_cwe_ent/target_pool_union.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from cga.pipelines.evaluate.evaluators.target_pool import pool_sha256


def gold_targets(ingest_dir: Path) -> set[str]:
    """Return the distinct gold ``node_2_id`` of one ingest run."""
    return set(pq.read_table(ingest_dir / "alignments.parquet").column("node_2_id").to_pylist())


def build_pool(ingest_dir: Path, check_dirs: list[Path]) -> dict:
    """Build the pool payload from ``ingest_dir`` and verify it covers every ``check_dirs`` gold.

    Args:
        ingest_dir: Ingest run directory whose gold targets define the pool (the ``union`` run).
        check_dirs: Other ingest run directories whose gold targets must be inside the pool.

    Returns:
        The JSON-ready payload.

    Raises:
        ValueError: If a checked run has a gold target outside the pool, or the pool is empty.
    """
    ids = sorted(gold_targets(ingest_dir))
    if not ids:
        raise ValueError(f"{ingest_dir} has no gold targets")
    pool = set(ids)
    for other in check_dirs:
        outside = sorted(gold_targets(other) - pool)
        if outside:
            raise ValueError(f"{other.name}: {len(outside)} gold target(s) outside the pool, e.g. {outside[:5]}")
    return {"node_ids": ids, "n": len(ids), "source_run": ingest_dir.name, "sha256": pool_sha256(ids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ingest", type=Path, required=True, help="union ingest run directory")
    ap.add_argument("--check", type=Path, action="append", default=[], help="other ingest run directory (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        payload = build_pool(args.ingest, args.check)
    except ValueError as err:
        print(f"FAILED: {err}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {args.out}: {payload['n']} CWE nodes (sha256 {payload['sha256'][:12]}), checked {len(args.check)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
