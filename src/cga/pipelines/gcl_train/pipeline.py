"""``gcl_train`` pipeline: graph contrastive learning (target_architecture.md §5.4).

Four steps: ``load`` (resolve the ``training_data`` run's manifest to find its
embedding stores and ``edges.parquet`` without letting the caller pass them by
hand; build a ``NodeUniverse``; ``select()`` every ``AlignedMatrix`` onto it;
remap ``edges.parquet``/``examples.parquet`` node ids to universe rows,
raising on any id outside the universe) -> ``train`` (full-batch loop,
``alpha * L_text + (1 - alpha) * L_graph``) -> ``export`` (final embeddings as
an ``AlignedMatrix``) -> ``manifest``/``LATEST``.

Only train-split rows of ``examples.parquet`` are used as the loss's anchor/
positive/negative triples; ``val``/``test`` nodes still get embeddings (the
forward pass is full-batch over the whole universe) but are never referenced
by a training example — enforced upstream by ``training_data``'s leakage
check and re-verified here by the universe-membership check on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

from cga.core.artifacts import ArtifactRef, write_latest
from cga.core.errors import ArtifactValidationError, ConfigError
from cga.core.io import read_json, write_json
from cga.core.pipeline import Pipeline, Step
from cga.core.registry import pipelines
from cga.core.run import RunContext
from cga.data.aligned import AlignedMatrix, NodeUniverse
from cga.data.example_store import read_examples
from cga.data.graph_store import read_edges
from cga.pipelines.gcl_train import losses as _losses  # noqa: F401  (registers triplet/infonce)
from cga.pipelines.gcl_train.models import build_dual_encoder
from cga.pipelines.gcl_train.trainer import ExampleTensors, sha256_of_universe, train_model

_ID_COLUMNS = ["anchor_id", "text_pos_id", "text_neg_id", "graph_pos_id", "graph_neg_id"]


def build_model_id(model_cfg: dict[str, Any]) -> str:
    """Return the ``model_id`` partitioning ``models/gcl/{dataset_id}/{model_id}/``.

    Defaults to ``model.encoder`` (``gcn``/``gat``) so each architecture keeps
    its own ``LATEST`` pointer -- without this, a `gat` run's ``LATEST`` write
    silently clobbers a `gcn` run's, making it impossible for `evaluate` to
    reference both as `latest` in one comparison (target_architecture.md §5.4).
    An explicit ``model.model_id`` overrides the default, e.g. for two `gcn`
    runs with different hyperparameters that should not share a lineage.
    """
    return model_cfg.get("model_id") or model_cfg.get("encoder", "gcn")


class _GclTrainState:
    """Mutable state threaded across this run's steps (one instance per run)."""

    def __init__(self) -> None:
        self.universe: NodeUniverse | None = None
        self.text_features: torch.Tensor | None = None
        self.struct_features: torch.Tensor | None = None
        self.edge_index: torch.Tensor | None = None
        self.train_examples: ExampleTensors | None = None
        self.n_examples_total: int = 0
        self.n_examples_train: int = 0
        self.model: torch.nn.Module | None = None
        self.final_embeddings: torch.Tensor | None = None
        self.training_data_run_id: str = ""
        self.text_embedding_run_id: str = ""
        self.graph_embedding_run_id: str = ""


def _find_input(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in manifest.get("inputs", []):
        if entry["name"] == name:
            return entry
    raise ArtifactValidationError(f"training_data run manifest has no recorded input named '{name}'")


class LoadStep:
    """Step 1: resolve the training_data run's manifest, build tensors on the universe."""

    name = "load"

    def __init__(self, state: _GclTrainState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        cfg = ctx.config["gcl_train"]

        training_data_lineage = Path(ctx.config["paths"]["artifacts"]) / "training_data" / ctx.dataset_id
        training_data_ref = ArtifactRef(
            lineage_dir=training_data_lineage, ref=cfg["inputs"]["training_data_run"]
        )
        training_data_dir = ctx.resolve("training_data", training_data_ref)

        td_manifest = read_json(training_data_dir / "manifest.json")
        state.training_data_run_id = training_data_dir.name

        text_input = _find_input(td_manifest, "text_embeddings")
        graph_input = _find_input(td_manifest, "graph_embeddings")
        ingest_input = _find_input(td_manifest, "ingest")
        state.text_embedding_run_id = text_input["run_id"]
        state.graph_embedding_run_id = graph_input["run_id"]

        # Re-resolving (rather than trusting the manifest path verbatim)
        # keeps this run's own manifest.inputs entries accurate and lets
        # `ctx.resolve` record them, per §8.2's "one call" rule.
        text_dir = ctx.resolve(
            "text_embeddings", ArtifactRef(lineage_dir=Path(text_input["path"]).parent, ref=text_input["run_id"])
        )
        graph_dir = ctx.resolve(
            "graph_embeddings",
            ArtifactRef(lineage_dir=Path(graph_input["path"]).parent, ref=graph_input["run_id"]),
        )
        ingest_dir = ctx.resolve(
            "ingest", ArtifactRef(lineage_dir=Path(ingest_input["path"]).parent, ref=ingest_input["run_id"])
        )

        universe = NodeUniverse.from_parquet(training_data_dir / "universe.parquet")
        state.universe = universe

        text_matrix = AlignedMatrix.load(text_dir).select(universe.node_ids)
        graph_matrix = AlignedMatrix.load(graph_dir).select(universe.node_ids)

        device = ctx.device
        state.text_features = torch.tensor(text_matrix.vectors, dtype=torch.float32, device=device)
        state.struct_features = torch.tensor(graph_matrix.vectors, dtype=torch.float32, device=device)

        row_of = {nid: i for i, nid in enumerate(universe.node_ids)}

        edges_table = read_edges(ingest_dir / "edges.parquet")
        src_ids = edges_table.column("src_id").to_pylist()
        dst_ids = edges_table.column("dst_id").to_pylist()
        edge_pairs: list[tuple[int, int]] = []
        n_edges_dropped = 0
        # edges.parquet covers every ingested node, but the universe (under
        # universe_policy: intersection) only keeps nodes present in *both*
        # embedding stores -- a node with missing/empty text has no text
        # embedding and so is legitimately absent from the universe. An edge
        # touching such a node carries no usable text-tower input and is
        # dropped from message passing rather than treated as an error, which
        # is a routine, expected consequence of the intersection policy
        # (unlike an out-of-universe example id, which would indicate the
        # upstream leakage check failed).
        for src, dst in zip(src_ids, dst_ids):
            if src not in row_of or dst not in row_of:
                n_edges_dropped += 1
                continue
            edge_pairs.append((row_of[src], row_of[dst]))
            edge_pairs.append((row_of[dst], row_of[src]))  # undirected message passing, matches legacy usage

        if n_edges_dropped:
            ctx.logger.info(
                "load: dropped %d/%d edges with an endpoint outside the universe",
                n_edges_dropped,
                edges_table.num_rows,
            )

        state.edge_index = (
            torch.tensor(edge_pairs, dtype=torch.long, device=device).t().contiguous()
            if edge_pairs
            else torch.empty((2, 0), dtype=torch.long, device=device)
        )

        examples_table = read_examples(training_data_dir / "examples.parquet")
        universe_ids = set(universe.node_ids)
        for column in _ID_COLUMNS:
            for example_id, ref_id in zip(
                examples_table.column("example_id").to_pylist(), examples_table.column(column).to_pylist()
            ):
                if ref_id not in universe_ids:
                    raise ArtifactValidationError(
                        f"Example '{example_id}' references '{column}'='{ref_id}' outside the universe"
                    )

        state.n_examples_total = examples_table.num_rows
        train_mask = pc.equal(examples_table.column("split"), "train")
        train_table = examples_table.filter(train_mask)
        state.n_examples_train = train_table.num_rows

        state.train_examples = ExampleTensors(
            anchor=_id_column_to_rows(train_table, "anchor_id", row_of, device),
            text_pos=_id_column_to_rows(train_table, "text_pos_id", row_of, device),
            text_neg=_id_column_to_rows(train_table, "text_neg_id", row_of, device),
            graph_pos=_id_column_to_rows(train_table, "graph_pos_id", row_of, device),
            graph_neg=_id_column_to_rows(train_table, "graph_neg_id", row_of, device),
        )

        ctx.logger.info(
            "load: universe=%d nodes, %d edges (directed pairs), %d/%d examples are train-split",
            len(universe.node_ids),
            state.edge_index.shape[1],
            state.n_examples_train,
            state.n_examples_total,
        )


def _id_column_to_rows(table: pa.Table, column: str, row_of: dict[str, int], device: str) -> torch.Tensor:
    ids = table.column(column).to_pylist()
    return torch.tensor([row_of[nid] for nid in ids], dtype=torch.long, device=device)


class TrainStep:
    """Step 2: full-batch training loop."""

    name = "train"

    def __init__(self, state: _GclTrainState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert (
            state.universe is not None
            and state.text_features is not None
            and state.struct_features is not None
            and state.edge_index is not None
            and state.train_examples is not None
        )

        if len(state.train_examples) == 0:
            raise ArtifactValidationError(
                "No train-split examples found in the training_data run's examples.parquet; "
                "cannot train with zero anchor/positive/negative triples"
            )

        cfg = ctx.config["gcl_train"]
        model_cfg = cfg["model"]
        loss_cfg = cfg["loss"]
        optim_cfg = cfg["optim"]
        train_cfg = cfg["train"]

        loss_name = loss_cfg["name"]
        if loss_name == "triplet":
            loss_kwargs = {"margin": loss_cfg.get("margin", 1.0)}
        elif loss_name == "infonce":
            loss_kwargs = {"temperature": loss_cfg.get("temperature", 0.1)}
        else:
            raise ConfigError(f"Unknown gcl_train.loss.name '{loss_name}'; expected 'triplet' or 'infonce'")

        model = build_dual_encoder(
            text_dim=state.text_features.shape[1],
            struct_dim=state.struct_features.shape[1],
            encoder=model_cfg.get("encoder", "gcn"),
            hidden_dim=model_cfg.get("hidden_dim", 128),
            out_dim=model_cfg.get("out_dim", 128),
            gat_heads=tuple(model_cfg.get("gat_heads", [8, 1])),
        ).to(ctx.device)
        state.model = model

        optimizer_name = optim_cfg.get("name", "adam")
        if optimizer_name != "adam":
            raise ConfigError(f"Unknown gcl_train.optim.name '{optimizer_name}'; only 'adam' is supported")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=optim_cfg.get("lr", 1.0e-3), weight_decay=optim_cfg.get("weight_decay", 0.0)
        )

        universe_sha256 = sha256_of_universe(state.universe.node_ids)

        final_embeddings = train_model(
            model=model,
            text_features=state.text_features,
            struct_features=state.struct_features,
            edge_index=state.edge_index,
            train_examples=state.train_examples,
            loss_name=loss_name,
            loss_kwargs=loss_kwargs,
            alpha=loss_cfg.get("alpha", 0.5),
            optimizer=optimizer,
            epochs=train_cfg.get("epochs", 400),
            checkpoint_every=train_cfg.get("checkpoint_every", 50),
            checkpoint_dir=ctx.run_dir / "checkpoints",
            universe_sha256=universe_sha256,
            metrics_path=ctx.run_dir / "metrics.jsonl",
            logger=ctx.logger,
            log_every=train_cfg.get("log_every", 10),
        )
        state.final_embeddings = final_embeddings


class ExportStep:
    """Step 3: final per-node embeddings as an AlignedMatrix keyed by node_id."""

    name = "export"

    def __init__(self, state: _GclTrainState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert state.final_embeddings is not None and state.universe is not None

        vectors = state.final_embeddings.cpu().numpy().astype(np.float32, copy=False)
        matrix = AlignedMatrix(vectors=vectors, node_ids=list(state.universe.node_ids))
        export_dir = ctx.run_dir / "node_embeddings"
        matrix.save(export_dir)
        ctx.logger.info("export: wrote %d x %d node_embeddings to %s", *vectors.shape, export_dir)


class ManifestStep:
    """Step 4: write config.resolved.yaml, stats/summary, record outputs, LATEST."""

    name = "manifest"

    def __init__(self, state: _GclTrainState) -> None:
        self._state = state

    def run(self, ctx: RunContext) -> None:
        state = self._state
        assert state.universe is not None

        import yaml

        config_path = ctx.run_dir / "config.resolved.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(ctx.config, f, sort_keys=True)

        for name, path in (
            ("config_resolved", config_path),
            ("metrics", ctx.run_dir / "metrics.jsonl"),
            ("node_embeddings_vectors", ctx.run_dir / "node_embeddings" / "vectors.npy"),
            ("node_embeddings_index", ctx.run_dir / "node_embeddings" / "index.parquet"),
        ):
            if path.is_file():
                ctx.manifest.record_output(name, path)

        ctx.manifest.summary = {
            "model_id": build_model_id(ctx.config["gcl_train"]["model"]),
            "training_data_run_id": state.training_data_run_id,
            "text_embedding_run_id": state.text_embedding_run_id,
            "graph_embedding_run_id": state.graph_embedding_run_id,
            "universe_sha256": sha256_of_universe(state.universe.node_ids),
            "n_nodes": len(state.universe.node_ids),
            "n_examples": state.n_examples_total,
            "n_examples_train": state.n_examples_train,
        }


@pipelines.register("gcl_train")
class GclTrainPipeline(Pipeline):
    """Runs universe/tensor loading, full-batch training, and embedding export."""

    name = "gcl_train"

    def build_steps(self, ctx: RunContext) -> list[Step]:
        state = _GclTrainState()
        self._state = state
        return [
            LoadStep(state),
            TrainStep(state),
            ExportStep(state),
            ManifestStep(state),
        ]

    def run(self, ctx: RunContext) -> None:
        super().run(ctx)
        model_id = build_model_id(ctx.config["gcl_train"]["model"])
        lineage_dir = Path(ctx.config["paths"]["models"]) / "gcl" / ctx.dataset_id / model_id
        write_latest(lineage_dir, ctx.run_id)
