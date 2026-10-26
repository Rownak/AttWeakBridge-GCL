"""Tests for the refs_equiv TSV parser against committed fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from cga.adapters.ncit_doid.refs_parser import iter_refs_alignments, parse_candidate_pools

FIXTURES = Path(__file__).parent / "fixtures"
TRAIN = FIXTURES / "refs_equiv_train.tsv"
TEST = FIXTURES / "refs_equiv_test.tsv"
CANDS = FIXTURES / "refs_equiv_test_cands.tsv"


def test_parse_candidate_pools_uses_literal_eval_not_eval():
    pools = parse_candidate_pools(CANDS)
    key = (
        "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C3224",
        "http://purl.obolibrary.org/obo/DOID_4362",
    )
    assert pools[key] == [
        "http://purl.obolibrary.org/obo/DOID_120",
        "http://purl.obolibrary.org/obo/DOID_9999",
    ]


def test_parse_candidate_pools_rejects_non_string_literal(tmp_path):
    bad = tmp_path / "bad_cands.tsv"
    bad.write_text("SrcEntity\tTgtEntity\tTgtCandidates\na\tb\t(1, 2, 3)\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_candidate_pools(bad)


def test_iter_refs_alignments_assigns_splits_and_ids():
    alignments = list(
        iter_refs_alignments(
            graph_1_key="ncit", graph_2_key="doid", train_path=TRAIN, test_path=TEST
        )
    )
    by_split = {a.split: a for a in alignments}
    assert by_split["train"].node_1_id == "ncit:C7058"
    assert by_split["train"].node_2_id == "doid:DOID_120"
    assert by_split["test"].node_1_id == "ncit:C3224"
    assert by_split["test"].node_2_id == "doid:DOID_4362"
    assert all(a.relation == "equivalence" for a in alignments)
    assert all(a.score == 1.0 for a in alignments)


def test_iter_refs_alignments_attaches_candidates_to_matching_row():
    alignments = list(
        iter_refs_alignments(
            graph_1_key="ncit",
            graph_2_key="doid",
            train_path=TRAIN,
            test_path=TEST,
            candidates_path=CANDS,
        )
    )
    train_row = next(a for a in alignments if a.split == "train")
    test_row = next(a for a in alignments if a.split == "test")
    assert train_row.candidates is None
    assert test_row.candidates == ["doid:DOID_120", "doid:DOID_9999"]


def test_iter_refs_alignments_empty_when_no_paths_given():
    alignments = list(iter_refs_alignments(graph_1_key="ncit", graph_2_key="doid"))
    assert alignments == []
