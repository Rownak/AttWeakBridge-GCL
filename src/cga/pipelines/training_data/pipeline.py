"""``training_data`` pipeline: contrastive example creation (target_architecture.md §5.3).

Five steps: ``build_universe`` (intersection of ``status == ok`` nodes across
both embedding stores; ``split`` carried from the ingest run's
``splits.parquet``; ``held_out`` derived as ``split in {val, test}`` OR the
node appears in a ``test``-split ``alignments.parquet`` row) -> ``similarity``
(cosine matrices restricted to non-held-out rows) -> ``sample`` (a registered
sampler) -> ``assign_example_split`` (every example inherits its anchor's
``split``) -> ``validate_and_manifest`` (schema/FK checks plus the leakage
assertion that no held-out node id appears in ``examples.parquet``).

``held_out`` nodes are never dropped from ``universe.parquet`` — ``gcl_train``
and ``evaluate`` still need their embeddings — they are simply never eligible
to be drawn as an anchor/positive/negative by any sampler, which is what keeps
a node reserved for evaluating the trained model out of the examples that
trained it. The ``supervised`` sampler additionally receives every gold
alignment row so it can exclude gold targets of all splits from its negatives. The node partition itself (``split``) is drawn once
by ``ingest`` (§5.1.1), identically for every dataset; this pipeline never
computes its own split.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa

from cga.core.artifacts import ArtifactRef, write_latest
from cga.core.errors import ArtifactValidationError, ConfigError
from cga.core.io import write_json
from cga.core.pipeline import Pipeline, Step
from cga.core.registry import pipelines
from cga.core.registry import samplers as sampler_registry
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix, NodeUniverse
from cga.data.example_store import write_examples
from cga.data.graph_store import read_alignments, read_nodes, read_splits
from cga.data.schemas import EXAMPLES_SCHEMA
from cga.pipelines.training_data.samplers import ExampleTuple, supervised_positives
from cga.pipelines.training_data.similarity import (
    SimilarityMatrix,
    compute_graph_similarity,
    compute_text_similarity,
)

_HELD_OUT_NODE_SPLITS = frozenset({"val", "test"})
_HELD_OUT_ALIGNMENT_SPLITS = frozenset({"test"})

_ID_COLUMNS = ["anchor_id", "text_pos_id", "text_neg_id", "graph_pos_id", "graph_neg_id"]

# Key params included in the default sampler_id per sampler name, in the order
# they appear in the id -- keeps rank_topk.k and percentile.{top_pct,bottom_pct}
# sweep points from clobbering each other's LATEST (mirrors build_model_id).
_SAMPLER_ID_PARAMS = {
    "rank_topk": ("k",),
    "percentile": ("top_pct", "bottom_pct"),
    "supervised": ("negatives_per_positive",),
}


def build_sampler_id(sampler_cfg: dict[str, Any]) -> str:
    """Return the ``sampler_id`` partitioning ``training_data/{dataset_id}/{sampler_id}/``.

    Defaults to ``{sampler.name}-{key params}`` (e.g. ``rank_topk-k30`` or
    ``percentile-top0.04-bottom0.04``) so distinct sweep points over the same
    sampler keep separate ``LATEST`` pointers -- without this, a second
    ``rank_topk`` run with a different ``k`` would silently clobber the
    first's ``LATEST``, making it impossible for a sweep to compare multiple
    points of the same sampler as ``latest``. An explicit
    ``training_data.sampler.sampler_id`` overrides the default, mirroring
    ``gcl_train.model.model_id``.
    """
    explicit = sampler_cfg.get("sampler_id")
    if explicit:
        return explicit

    name = sampler_cfg.get("name", "rank_topk")
    parts = [name]
    for param in _SAMPLER_ID_PARAMS.get(name, ()):
        if param in sampler_cfg:
            parts.append(f"{param}{sampler_cfg[param]}")
    return "-".join(parts)


class _TrainingDataState:
    """Mutable state threaded across this run's steps (one instance per run)."""

    def __init__(self) -> None:
        self.text_matrix: AlignedMatrix | None = None
        self.graph_matrix: AlignedMatrix | None = None
        self.universe: NodeUniverse | None = None
        self.alignments_table: pa.Table | None = None
        self.text_g1_g2: SimilarityMatrix | None = None
        self.graph_g1: SimilarityMatrix | None = None
        self.graph_g2: SimilarityMatrix | None = None
        self.sampled_examples: list[ExampleTuple] = []
        self.sampler_name: str = ""
        self.sampler_stats: dict[str, int] = {}
        self.examples_table: pa.Table | None = None
        self.output_dir: Path | None = None


def _resolve_embedding_store(ctx: RunContext, name: str, cfg: dict[str, Any]) -> Path:
    lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "embeddings" / ctx.dataset_id / cfg["embedder_id"]
    ref = ArtifactRef(lineage_dir=lineage_dir, ref=cfg.get("run", "latest"))
    return ctx.resolve(name, ref)


class BuildUniverseStep:
    """Step 1: intersection of ``status == ok`` nodes across both stores; carry ``split``, flag ``held_out``."""

    name = "build_universe"

    def __init__(self, state: _TrainingDataState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        cfg = ctx.config["training_data"]
        inputs_cfg = cfg["inputs"]

        ingest_ref = ArtifactRef(
            lineage_dir=Path(ctx.config["paths"]["processed"]) / ctx.dataset_id,
            ref=inputs_cfg["ingest_run"],
        )
        ingest_dir = ctx.resolve("ingest", ingest_ref)
        nodes_table = read_nodes(ingest_dir / "nodes.parquet")
        graph_role_by_node = dict(
            zip(nodes_table.column("node_id").to_pylist(), nodes_table.column("graph_role").to_pylist())
        )

        splits_table = read_splits(ingest_dir / "splits.parquet")
        node_split_by_node = dict(
            zip(splits_table.column("node_id").to_pylist(), splits_table.column("split").to_pylist())
        )

        text_dir = _resolve_embedding_store(ctx, "text_embeddings", inputs_cfg["text_embeddings"])
        graph_dir = _resolve_embedding_store(ctx, "graph_embeddings", inputs_cfg["graph_embeddings"])
        text_matrix = AlignedMatrix.load(text_dir)
        graph_matrix = AlignedMatrix.load(graph_dir)
        self._state.text_matrix = text_matrix
        self._state.graph_matrix = graph_matrix

        policy = cfg.get("universe_policy", "intersection")
        if policy == "intersection":
            eligible_ids = set(text_matrix.node_ids) & set(graph_matrix.node_ids)
        elif policy == "text_only":
            eligible_ids = set(text_matrix.node_ids)
        else:
            raise ConfigError(f"Unknown training_data.universe_policy '{policy}'")

        # Preserve nodes.parquet order so universe rows are deterministic given
        # a fixed ingest run, independent of either store's internal ordering.
        ordered_ids = [nid for nid in nodes_table.column("node_id").to_pylist() if nid in eligible_ids]

        alignments_table = read_alignments(ingest_dir / "alignments.parquet")
        self._state.alignments_table = alignments_table

        # held_out is the union of two independent conditions (§5.3 step 1):
        # (a) this dataset's own uniform node partition put the node in
        #     val/test, or (b) a dataset-provided alignment split (e.g.
        #     ncit_doid's OAEI refs_equiv/test.tsv) marks it as a test pair
        #     endpoint. (b) is vacuous for datasets without alignments.
        alignment_held_out_ids: set[str] = set()
        if alignments_table.num_rows > 0:
            for n1, n2, split in zip(
                alignments_table.column("node_1_id").to_pylist(),
                alignments_table.column("node_2_id").to_pylist(),
                alignments_table.column("split").to_pylist(),
            ):
                if split in _HELD_OUT_ALIGNMENT_SPLITS:
                    alignment_held_out_ids.add(n1)
                    alignment_held_out_ids.add(n2)

        split_by_id = [node_split_by_node[nid] for nid in ordered_ids]
        held_out_by_id = [
            split in _HELD_OUT_NODE_SPLITS or nid in alignment_held_out_ids
            for nid, split in zip(ordered_ids, split_by_id)
        ]

        universe = NodeUniverse(
            node_ids=ordered_ids,
            graph_roles=[graph_role_by_node[nid] for nid in ordered_ids],
            split=split_by_id,
            held_out=held_out_by_id,
        )
        self._state.universe = universe

        self._state.output_dir = ctx.run_dir
        universe.save(ctx.run_dir / "universe.parquet")
        ctx.logger.info(
            "build_universe: %d nodes (%d held_out)", len(universe.node_ids), sum(universe.held_out)
        )


class SimilarityStep:
    """Step 2: cosine matrices, cross-graph (text) and within-graph (structural), over non-held-out rows."""

    name = "similarity"

    def __init__(self, state: _TrainingDataState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert state.universe is not None and state.text_matrix is not None and state.graph_matrix is not None

        graph_role_by_node = dict(zip(state.universe.node_ids, state.universe.graph_roles))
        held_out_by_node = dict(zip(state.universe.node_ids, state.universe.held_out))

        state.text_g1_g2 = compute_text_similarity(
            state.text_matrix, graph_role_by_node=graph_role_by_node, held_out_by_node=held_out_by_node
        )
        state.graph_g1 = compute_graph_similarity(
            state.graph_matrix, role="graph_1", graph_role_by_node=graph_role_by_node, held_out_by_node=held_out_by_node
        )
        state.graph_g2 = compute_graph_similarity(
            state.graph_matrix, role="graph_2", graph_role_by_node=graph_role_by_node, held_out_by_node=held_out_by_node
        )
        ctx.logger.info(
            "similarity: text %dx%d, graph_1 %dx%d, graph_2 %dx%d",
            len(state.text_g1_g2.row_ids),
            len(state.text_g1_g2.col_ids),
            len(state.graph_g1.row_ids),
            len(state.graph_g1.row_ids),
            len(state.graph_g2.row_ids),
            len(state.graph_g2.row_ids),
        )


class SampleStep:
    """Step 3: run the configured sampler, restricted to the non-held-out similarity views."""

    name = "sample"

    def __init__(self, state: _TrainingDataState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert (
            state.text_g1_g2 is not None
            and state.graph_g1 is not None
            and state.graph_g2 is not None
            and state.universe is not None
        )
        cfg = ctx.config["training_data"]["sampler"]
        sampler_name = cfg["name"]
        sampler_fn = sampler_registry.get(sampler_name)
        state.sampler_name = sampler_name

        alignments = None
        if sampler_name == "supervised":
            assert state.alignments_table is not None
            # Pass the full gold set (every split), unfiltered: the sampler
            # needs test pairs to keep their targets out of the negatives.
            # Positives are still train-only and restricted to the
            # non-held-out matrices, where held_out already covers val/test
            # node splits and test-pair endpoints (§5.3 step 1), so a test
            # endpoint can never enter an example.
            alignments = [
                {"node_1_id": n1, "node_2_id": n2, "split": split}
                for n1, n2, split in zip(
                    state.alignments_table.column("node_1_id").to_pylist(),
                    state.alignments_table.column("node_2_id").to_pylist(),
                    state.alignments_table.column("split").to_pylist(),
                )
            ]
            used, n_skipped = supervised_positives(alignments, state.text_g1_g2, state.graph_g1)
            state.sampler_stats = {
                "n_gold_train_pairs": sum(1 for row in alignments if row["split"] == "train"),
                "n_gold_train_pairs_used": len(used),
                "n_gold_train_pairs_skipped": n_skipped,
            }

        examples = sampler_fn(
            text_g1_g2=state.text_g1_g2,
            graph_g1=state.graph_g1,
            graph_g2=state.graph_g2,
            config=cfg,
            seed=ctx.seed,
            alignments=alignments,
        )
        state.sampled_examples = examples
        ctx.logger.info("sample: %s produced %d examples", sampler_name, len(examples))


def _example_id(sampler: str, tup: ExampleTuple) -> str:
    payload = "|".join((sampler, *tup))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class AssignExampleSplitStep:
    """Step 4: every example inherits the ``split`` of its anchor node (§5.3 step 4).

    The node partition is the single source of truth (§5.1.1); this step never
    draws its own split. In practice every unsupervised example's anchor is
    ``train`` (``build_universe``/``similarity``/``sample`` already excluded
    ``val``/``test`` anchors from the sampling pool), so this step mostly
    asserts that invariant rather than deciding anything — but it exists so a
    future sampler that deliberately emits held-out-anchor examples (e.g. for
    a val-loss curve) is labeled correctly without a schema change.
    """

    name = "assign_example_split"

    def __init__(self, state: _TrainingDataState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert state.universe is not None
        split_by_node = dict(zip(state.universe.node_ids, state.universe.split))

        rows = [self._row(tup, split_by_node[tup[0]]) for tup in state.sampled_examples]
        table = pa.Table.from_pylist(rows, schema=EXAMPLES_SCHEMA)
        state.examples_table = table

    def _row(self, tup: ExampleTuple, split: str) -> dict[str, Any]:
        anchor_id, text_pos_id, text_neg_id, graph_pos_id, graph_neg_id = tup
        return {
            "example_id": _example_id(self._state.sampler_name, tup),
            "anchor_id": anchor_id,
            "text_pos_id": text_pos_id,
            "text_neg_id": text_neg_id,
            "graph_pos_id": graph_pos_id,
            "graph_neg_id": graph_neg_id,
            "sampler": self._state.sampler_name,
            "split": split,
        }


class ValidateAndManifestStep:
    """Step 5: schema/FK checks, the leakage assertion, ``stats.json``, ``LATEST``."""

    name = "validate_and_manifest"

    def __init__(self, state: _TrainingDataState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert state.examples_table is not None and state.universe is not None and state.output_dir is not None

        held_out_ids = {nid for nid, ho in zip(state.universe.node_ids, state.universe.held_out) if ho}
        example_ids = state.examples_table.column("example_id").to_pylist()
        for column in _ID_COLUMNS:
            ref_ids = state.examples_table.column(column).to_pylist()
            for example_id, ref_id in zip(example_ids, ref_ids):
                if ref_id in held_out_ids:
                    raise ArtifactValidationError(
                        f"Example '{example_id}' references held-out node '{ref_id}' via '{column}' "
                        "— this must never happen"
                    )

        examples_path = state.output_dir / "examples.parquet"
        write_examples(state.examples_table, examples_path, universe=state.universe)

        split_counts: dict[str, int] = defaultdict(int)
        for split in state.examples_table.column("split").to_pylist():
            split_counts[split] += 1

        stats = {
            "n_universe_nodes": len(state.universe.node_ids),
            "n_held_out_nodes": len(held_out_ids),
            "sampler": state.sampler_name,
            "n_examples": state.examples_table.num_rows,
            "n_examples_by_split": dict(split_counts),
            **state.sampler_stats,
        }
        stats_path = state.output_dir / "stats.json"
        write_json(stats, stats_path)

        for name, path in (
            ("universe", state.output_dir / "universe.parquet"),
            ("examples", examples_path),
            ("stats", stats_path),
        ):
            ctx.manifest.record_output(name, path)

        ctx.manifest.summary = stats


@pipelines.register("training_data")
class TrainingDataPipeline(Pipeline):
    """Runs universe construction, similarity, sampling, and split assignment."""

    name = "training_data"

    def build_steps(self, ctx: RunContext) -> list[Step]:
        state = _TrainingDataState()
        self._state = state
        return [
            BuildUniverseStep(state),
            SimilarityStep(state),
            SampleStep(state),
            AssignExampleSplitStep(state),
            ValidateAndManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        sampler_id = build_sampler_id(ctx.config["training_data"]["sampler"])
        lineage_dir = Path(ctx.config["paths"]["artifacts"]) / "training_data" / ctx.dataset_id / sampler_id
        write_latest(lineage_dir, ctx.run_id)
