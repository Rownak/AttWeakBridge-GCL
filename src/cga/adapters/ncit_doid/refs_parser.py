"""``refs_equiv/*.tsv`` -> alignment records, and the ranking candidate pool.

Gold equivalence mappings (``SrcEntity``, ``TgtEntity``, ``Score``) are tab-
separated files with pre-defined train/test splits; ``test.cands.tsv`` carries
a per-row candidate pool for ranking evaluation, stored as a Python-literal
tuple-of-strings, parsed with ``ast.literal_eval`` — never ``eval`` — so a
malformed or hostile file can only raise ``ValueError``/``SyntaxError``, not
execute arbitrary code.
"""

from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path
from typing import Iterator

from cga.adapters.ncit_doid.owl_parser import iri_fragment
from cga.data.ids import make_node_id
from cga.data.schemas import AlignmentRecord


def _read_tsv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def parse_candidate_pools(path: str | Path) -> dict[tuple[str, str], list[str]]:
    """Parse ``test.cands.tsv`` into ``{(src_iri, tgt_iri): [candidate_iri, ...]}``.

    Raises:
        ValueError: If a ``TgtCandidates`` cell is not a valid Python literal
            tuple/list of strings.
    """
    pools: dict[tuple[str, str], list[str]] = {}
    for row in _read_tsv_rows(path):
        raw = row["TgtCandidates"]
        parsed = ast.literal_eval(raw)
        if not isinstance(parsed, (tuple, list)) or not all(isinstance(x, str) for x in parsed):
            raise ValueError(f"TgtCandidates cell is not a tuple/list of strings: {raw!r}")
        pools[(row["SrcEntity"], row["TgtEntity"])] = list(parsed)
    return pools


def iter_refs_alignments(
    *,
    graph_1_key: str,
    graph_2_key: str,
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
    candidates_path: str | Path | None = None,
) -> Iterator[AlignmentRecord]:
    """Yield one ``AlignmentRecord`` per row across the train/test equivalence files.

    ``SrcEntity`` is assumed to be a ``graph_1`` (source ontology) IRI and
    ``TgtEntity`` a ``graph_2`` (target ontology) IRI, matching ``refs_equiv``'s
    column order and the dataset config's ``graph_1``/``graph_2`` assignment.
    Candidate pools (when ``candidates_path`` is given) are attached only to
    the matching test-split row.
    """
    candidate_pools = parse_candidate_pools(candidates_path) if candidates_path else {}

    for split, path in (("train", train_path), ("test", test_path)):
        if path is None:
            continue
        for row in _read_tsv_rows(path):
            src_iri = row["SrcEntity"]
            tgt_iri = row["TgtEntity"]
            score = float(row.get("Score", 1.0) or 1.0)
            node_1_id = make_node_id(graph_1_key, iri_fragment(src_iri))
            node_2_id = make_node_id(graph_2_key, iri_fragment(tgt_iri))
            alignment_id = hashlib.sha1(
                f"{node_1_id}|{node_2_id}|equivalence".encode("utf-8")
            ).hexdigest()
            candidates = candidate_pools.get((src_iri, tgt_iri))
            candidate_ids = (
                [make_node_id(graph_2_key, iri_fragment(c)) for c in candidates]
                if candidates is not None
                else None
            )
            yield AlignmentRecord(
                alignment_id=alignment_id,
                node_1_id=node_1_id,
                node_2_id=node_2_id,
                relation="equivalence",
                score=score,
                split=split,
                candidates=candidate_ids,
            )
