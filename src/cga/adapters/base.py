"""Dataset adapter interface: the only place dataset-specific knowledge lives.

Every adapter reads one dataset's native files and yields canonical records
(``NodeRecord``, ``EdgeRecord``, ``TextRecord``, ``AlignmentRecord``, per
``cga.data.schemas``). The ``ingest`` pipeline drains these iterators, assigns
``graph_role`` from the adapter's ``GraphSpec``s, and writes the four canonical
tables — it does not know what a "STIX bundle" or an "OWL class" is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

from pydantic import BaseModel

from cga.core.registry import adapters as _adapters
from cga.data.schemas import AlignmentRecord, EdgeRecord, NodeRecord, TextRecord


@dataclass(frozen=True)
class GraphSpec:
    """Identifies one side of a dataset's graph pair.

    Attributes:
        key: The dataset's short name for this graph (e.g. ``"attack"``,
            ``"cwe"``), used as the ``graph_key`` half of every node_id.
        role: ``"graph_1"`` (source/left) or ``"graph_2"`` (target/right).
            Carries no semantics beyond which side of the pair this is.
        display_name: Human-readable name for logs and reports.
    """

    key: str
    role: str
    display_name: str


class DatasetAdapter(ABC):
    """Reads one dataset's native files and yields canonical records.

    A concrete subclass is registered under a ``dataset_id`` via
    ``@register_adapter(name)`` and constructed from that dataset's
    ``adapter_config`` block (validated against ``ConfigModel``). The
    ``ingest`` pipeline is the only caller.
    """

    name: str
    ConfigModel: type[BaseModel]
    graphs: tuple[GraphSpec, GraphSpec]

    def validate_sources(self) -> None:
        """Check that every source file this adapter needs is present and readable.

        Raises:
            MissingInputError: If a required source file is missing, with the
                actionable path. Default implementation is a no-op; adapters
                with source files should override.
        """
        return None

    @abstractmethod
    def iter_nodes(self) -> Iterator[NodeRecord]:
        """Yield one ``NodeRecord`` per entity, in source order."""
        raise NotImplementedError

    @abstractmethod
    def iter_edges(self) -> Iterator[EdgeRecord]:
        """Yield one ``EdgeRecord`` per intra-graph relation.

        Must never yield an edge whose endpoints are in different graphs —
        the ``ingest`` pipeline treats that as a hard error, not a filter.
        """
        raise NotImplementedError

    @abstractmethod
    def iter_texts(self) -> Iterator[TextRecord]:
        """Yield one ``TextRecord`` per ``(node_id, text_variant)`` pair."""
        raise NotImplementedError

    def iter_alignments(self) -> Iterator[AlignmentRecord]:
        """Yield ground-truth alignment pairs, if this dataset has any.

        Default: no ground truth.
        """
        return iter(())

    def source_fingerprint(self) -> dict[str, str]:
        """Return ``{relative_path: sha256}`` for every raw file this adapter read.

        Recorded as manifest inputs by the ``ingest`` pipeline. Default: none.
        """
        return {}


T = TypeVar("T", bound=type[DatasetAdapter])


def register_adapter(name: str) -> Callable[[T], T]:
    """Register a ``DatasetAdapter`` subclass under ``name`` (the ``dataset_id``).

    Raises:
        ConfigError: If ``name`` is already registered.
    """

    def decorator(cls: T) -> T:
        cls.name = name
        _adapters.register(name)(cls)
        return cls

    return decorator
