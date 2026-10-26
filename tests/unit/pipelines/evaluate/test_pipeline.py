"""End-to-end tests for the ``evaluate`` pipeline against fabricated methods/references."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import cga.pipelines  # noqa: F401  (registers pipelines)
from cga.core.artifacts import write_latest
from cga.core.errors import ConfigError
from cga.core.io import read_json
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix
from cga.data.schemas import EDGES_SCHEMA, METRICS_SCHEMA, NODES_SCHEMA

G1_IDS = ["attack:1", "attack:2", "attack:3", "attack:4"]
G2_IDS = ["cwe:1", "cwe:2", "cwe:3", "cwe:4"]
ALL_IDS = G1_IDS + G2_IDS

# A path within graph_1 (1-2-3-4) and a path within graph_2 (1-2-3-4), so
# hop-radius relevance has real structure to score.
EDGES = [
    ("attack:1", "attack:2"),
    ("attack:2", "attack:3"),
    ("attack:3", "attack:4"),
    ("cwe:1", "cwe:2"),
    ("cwe:2", "cwe:3"),
    ("cwe:3", "cwe:4"),
]


def _write_embedding_store(store_dir: Path, node_ids: list[str], *, dim: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(node_ids), dim)).astype(np.float32)
    AlignedMatrix(vectors=vectors, node_ids=node_ids).save(store_dir)
    write_latest(store_dir.parent, store_dir.name)
    return store_dir.name


def _write_ingest_run(ingest_dir: Path, node_ids: list[str]) -> None:
    rows = [
        {
            "node_id": nid,
            "graph_key": nid.split(":")[0],
            "graph_role": "graph_1" if nid.startswith("attack") else "graph_2",
            "source_id": nid.split(":")[1],
            "external_id": nid,
            "name": nid,
            "node_type": None,
            "metadata": "{}",
            "has_text": True,
            "source_version": "test",
            "raw_sha256": "0" * 64,
        }
        for nid in node_ids
    ]
    pq.write_table(pa.Table.from_pylist(rows, schema=NODES_SCHEMA), ingest_dir / "nodes.parquet")

    edge_rows = [
        {
            "edge_id": f"e{i}",
            "src_id": src,
            "dst_id": dst,
            "graph_role": "graph_1" if src.startswith("attack") else "graph_2",
            "relation": "related",
            "directed": False,
            "weight": 1.0,
        }
        for i, (src, dst) in enumerate(EDGES)
        if src in node_ids and dst in node_ids
    ]
    pq.write_table(pa.Table.from_pylist(edge_rows, schema=EDGES_SCHEMA), ingest_dir / "edges.parquet")


def _build_config(
    tmp_path: Path,
    *,
    methods: list[dict] | None = None,
    evaluations: list[dict] | None = None,
    visualize: bool = False,
) -> dict:
    return {
        "project": "cga",
        "seed": 42,
        "device": "cpu",
        "paths": {
            "raw": str(tmp_path / "raw"),
            "processed": str(tmp_path / "processed"),
            "artifacts": str(tmp_path / "artifacts"),
            "models": str(tmp_path / "models"),
        },
        "logging": {"level": "INFO", "file": False},
        "run": {"name": None},
        "dataset": {"id": "attack_cwe"},
        "evaluate": {
            "inputs": {"ingest_run": "latest"},
            "methods": methods if methods is not None else [],
            "evaluations": evaluations if evaluations is not None else [],
            "visualize": {"enabled": visualize, "perplexity": 5.0, "methods": []},
        },
    }


def _setup_ingest_run(config: dict, *, node_ids: list[str] | None = None) -> Path:
    node_ids = node_ids or ALL_IDS
    dataset_id = config["dataset"]["id"]
    ingest_dir = Path(config["paths"]["processed"]) / dataset_id / "20260101T000000Z-aaaaaaaa"
    ingest_dir.mkdir(parents=True, exist_ok=True)
    _write_ingest_run(ingest_dir, node_ids)
    write_latest(ingest_dir.parent, ingest_dir.name)
    return ingest_dir


def _setup_method_embedding_store(
    config: dict, method_id: str, *, node_ids: list[str] | None = None, dim: int = 6, seed: int = 1
) -> dict:
    """Write a fake embedding store and return its ``methods:`` config entry."""
    node_ids = node_ids or ALL_IDS
    dataset_id = config["dataset"]["id"]
    embedder_id = f"fake-{method_id}"
    store_dir = Path(config["paths"]["artifacts"]) / "embeddings" / dataset_id / embedder_id / f"run-{method_id}"
    _write_embedding_store(store_dir, node_ids, dim=dim, seed=seed)
    return {
        "method_id": method_id,
        "label": method_id,
        "kind": "embedding_store",
        "embedder_id": embedder_id,
        "ref": "latest",
    }


def _setup_method_gcl_run(
    config: dict, method_id: str, model_id: str, *, node_ids: list[str] | None = None, dim: int = 8, seed: int = 2
) -> dict:
    node_ids = node_ids or ALL_IDS
    dataset_id = config["dataset"]["id"]
    model_lineage_dir = Path(config["paths"]["models"]) / "gcl" / dataset_id / model_id
    gcl_dir = model_lineage_dir / f"run-{method_id}"
    rng_vectors_dir = gcl_dir / "node_embeddings"
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(len(node_ids), dim)).astype(np.float32)
    AlignedMatrix(vectors=vectors, node_ids=node_ids).save(rng_vectors_dir)
    write_latest(model_lineage_dir, gcl_dir.name)  # LATEST lives beside the run dir, not inside it
    return {
        "method_id": method_id,
        "label": method_id,
        "kind": "gcl_run",
        "model_id": model_id,
        "ref": "latest",
    }


def _run_evaluate(config: dict) -> RunContext:
    pipeline_cls = pipelines.get("evaluate")
    pipeline = pipeline_cls()
    run_root = Path(config["paths"]["artifacts"]) / "evaluations" / config["dataset"]["id"]
    ctx = RunContext.create(
        pipeline_name="evaluate",
        pipeline_version="0.0.0-test",
        dataset_id=config["dataset"]["id"],
        config=config,
        run_root=run_root,
    )
    pipeline.run(ctx)
    return ctx


NEIGHBORHOOD_EVAL = {
    "evaluation_id": "neigh",
    "kind": "neighborhood",
    "params": {
        "ks": [1, 2],
        "roles": ["graph_1", "graph_2"],
        "views": [
            {"view_id": "structural_r1", "provider": "hop_radius", "params": {"radius": 1}},
        ],
    },
}


def test_evaluate_end_to_end_writes_metrics_for_two_methods(tmp_path):
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    method_b = _setup_method_gcl_run(config, "gcn", "gcn", seed=2)
    config["evaluate"]["methods"] = [method_a, method_b]
    config["evaluate"]["evaluations"] = [NEIGHBORHOOD_EVAL]
    _setup_ingest_run(config)

    ctx = _run_evaluate(config)

    table = read_parquet_helper(ctx.run_dir / "metrics.parquet")
    assert table.schema.equals(METRICS_SCHEMA, check_metadata=False)
    method_ids = set(table.column("method_id").to_pylist())
    assert method_ids == {"deepwalk", "gcn"}
    roles = set(table.column("graph_role").to_pylist())
    assert roles == {"graph_1", "graph_2"}
    metric_names = set(table.column("metric_name").to_pylist())
    assert metric_names == {"normalized_precision_at_k", "recall_at_k", "hit_at_k", "mrr"}
    for value in table.column("metric_value").to_pylist():
        assert 0.0 <= value <= 1.0

    digest = read_json(ctx.run_dir / "metrics.json")
    assert digest["schema_version"] == 2
    assert "neigh" in digest["evaluations"]

    manifest = read_json(ctx.run_dir / "manifest.json")
    assert manifest["status"] == "success"
    assert set(manifest["summary"]["methods"]) == {"deepwalk", "gcn"}
    assert any(inp["name"] == "method:deepwalk" for inp in manifest["inputs"])
    assert any(inp["name"] == "method:gcn" for inp in manifest["inputs"])
    assert not (ctx.run_dir / "plots").exists()


def test_evaluate_visualize_enabled_writes_one_plot_per_method(tmp_path):
    config = _build_config(tmp_path, visualize=True)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    config["evaluate"]["methods"] = [method_a]
    config["evaluate"]["evaluations"] = [NEIGHBORHOOD_EVAL]
    _setup_ingest_run(config)

    ctx = _run_evaluate(config)

    plot_path = ctx.run_dir / "plots" / "tsne_deepwalk.png"
    assert plot_path.is_file()
    manifest = read_json(ctx.run_dir / "manifest.json")
    assert any(o["name"] == "plot_tsne_deepwalk" for o in manifest["outputs"])


def test_evaluate_unknown_evaluation_kind_raises(tmp_path):
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    config["evaluate"]["methods"] = [method_a]
    config["evaluate"]["evaluations"] = [{"evaluation_id": "bogus_eval", "kind": "bogus", "params": {}}]
    _setup_ingest_run(config)

    with pytest.raises(ConfigError):
        _run_evaluate(config)


def test_evaluate_unknown_relevance_provider_raises(tmp_path):
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    config["evaluate"]["methods"] = [method_a]
    config["evaluate"]["evaluations"] = [
        {
            "evaluation_id": "neigh",
            "kind": "neighborhood",
            "params": {"ks": [1], "views": [{"view_id": "v", "provider": "bogus_provider", "params": {}}]},
        }
    ]
    _setup_ingest_run(config)

    with pytest.raises(ConfigError):
        _run_evaluate(config)


def test_evaluate_duplicate_method_id_raises(tmp_path):
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    method_b = dict(method_a)  # same method_id
    config["evaluate"]["methods"] = [method_a, method_b]
    config["evaluate"]["evaluations"] = [NEIGHBORHOOD_EVAL]
    _setup_ingest_run(config)

    with pytest.raises(ConfigError):
        _run_evaluate(config)


def test_evaluate_duplicate_evaluation_id_raises(tmp_path):
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    config["evaluate"]["methods"] = [method_a]
    config["evaluate"]["evaluations"] = [NEIGHBORHOOD_EVAL, dict(NEIGHBORHOOD_EVAL)]
    _setup_ingest_run(config)

    with pytest.raises(ConfigError):
        _run_evaluate(config)


def test_evaluate_cluster_quality_evaluator_runs(tmp_path):
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    config["evaluate"]["methods"] = [method_a]
    config["evaluate"]["evaluations"] = [
        {
            "evaluation_id": "clusters",
            "kind": "cluster_quality",
            "params": {"roles": ["graph_1", "graph_2"], "n_clusters": [2]},
        }
    ]
    _setup_ingest_run(config)

    ctx = _run_evaluate(config)

    table = read_parquet_helper(ctx.run_dir / "metrics.parquet")
    assert set(table.column("metric_name").to_pylist()) == {"silhouette"}
    assert set(table.column("param_value").to_pylist()) == {2}
    assert all(v is None for v in table.column("view_id").to_pylist())


def test_evaluate_experiment_overlay_replaces_default_methods_list(tmp_path):
    """Mirrors how --experiment overlays a thin default: a fresh methods list wins outright."""
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
    method_b = _setup_method_gcl_run(config, "gcn", "gcn", seed=2)
    # Simulate deep_merge's "lists replace wholesale" semantics directly.
    config["evaluate"]["methods"] = []  # thin default
    config["evaluate"]["methods"] = [method_a, method_b]  # overlay replaces it
    config["evaluate"]["evaluations"] = [NEIGHBORHOOD_EVAL]
    _setup_ingest_run(config)

    ctx = _run_evaluate(config)
    table = read_parquet_helper(ctx.run_dir / "metrics.parquet")
    assert set(table.column("method_id").to_pylist()) == {"deepwalk", "gcn"}


def test_evaluate_is_deterministic_across_runs(tmp_path):
    def _one_run(root: Path) -> pa.Table:
        config = _build_config(root)
        method_a = _setup_method_embedding_store(config, "deepwalk", seed=1)
        method_b = _setup_method_gcl_run(config, "gcn", "gcn", seed=2)
        config["evaluate"]["methods"] = [method_a, method_b]
        config["evaluate"]["evaluations"] = [NEIGHBORHOOD_EVAL]
        _setup_ingest_run(config)
        ctx = _run_evaluate(config)
        return read_parquet_helper(ctx.run_dir / "metrics.parquet")

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    table_a = _one_run(tmp_path / "a")
    table_b = _one_run(tmp_path / "b")

    # run_id differs run to run; compare every other column.
    cols = [c for c in table_a.column_names if c != "run_id"]
    assert table_a.select(cols).equals(table_b.select(cols))


def test_evaluate_query_set_sha256_identical_across_methods(tmp_path):
    """All methods scored in one (view, role) must share the same query_set_sha256."""
    config = _build_config(tmp_path)
    method_a = _setup_method_embedding_store(config, "deepwalk", node_ids=ALL_IDS, seed=1)
    # method_b is missing one graph_1 node -> that node must leave every
    # method's candidate pool, not just method_b's.
    method_b = _setup_method_embedding_store(
        config, "secroberta", node_ids=[nid for nid in ALL_IDS if nid != "attack:4"], seed=2
    )
    config["evaluate"]["methods"] = [method_a, method_b]
    config["evaluate"]["evaluations"] = [NEIGHBORHOOD_EVAL]
    _setup_ingest_run(config)

    ctx = _run_evaluate(config)
    digest = read_json(ctx.run_dir / "metrics.json")
    scopes = digest["evaluations"]["neigh"]["scopes"]
    g1_scope = scopes["structural_r1::graph_1"]
    assert "query_set_sha256" in g1_scope
    # Every method row (checked via the table) should be scored against the
    # method-intersected pool: attack:4 must not appear as a candidate in
    # either method's similarity computation for graph_1's n_queries.
    table = read_parquet_helper(ctx.run_dir / "metrics.parquet")
    mask = pc.and_(
        pc.equal(table.column("graph_role"), "graph_1"),
        pc.equal(table.column("view_id"), "structural_r1"),
    )
    g1_rows = table.filter(mask)
    n_queries_values = set(g1_rows.column("n_queries").to_pylist())
    assert len(n_queries_values) == 1  # identical across both methods


def read_parquet_helper(path: Path) -> pa.Table:
    return pq.read_table(path)
