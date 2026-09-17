"""ATT&CK STIX JSON bundle -> canonical records.

Walks the bundle's flat ``objects`` array (legacy/data_preprocessing/2_Mitre_attack_to_json.py
and 4_Attack_graph_create.py, ported and merged): any object with an ``id`` is a
node; objects with ``type == "relationship"`` become edges via ``source_ref`` /
``target_ref``. Dangling relationship endpoints (a ref not present in the bundle)
are counted and reported rather than silently dropped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from cga.data.ids import make_node_id
from cga.data.schemas import EdgeRecord, NodeRecord, TextRecord


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_sha256(obj: dict[str, Any]) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class StixParseResult:
    """Parsed contents of one ATT&CK STIX bundle.

    Attributes:
        objects_by_id: Every object in the bundle keyed by its STIX ``id``.
        n_dangling_relationships: Relationship objects whose ``source_ref`` or
            ``target_ref`` does not resolve to an object in the bundle.
    """

    objects_by_id: dict[str, dict[str, Any]]
    n_dangling_relationships: int


def parse_stix_bundle(path: str | Path) -> StixParseResult:
    """Load an ATT&CK STIX bundle and index its objects by id.

    Raises:
        MissingInputError: Never here — callers check existence via
            ``validate_sources``; a missing/unreadable file surfaces as
            ``OSError``/``json.JSONDecodeError``.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    objects_by_id: dict[str, dict[str, Any]] = {}
    for obj in bundle.get("objects", []):
        obj_id = obj.get("id")
        if obj_id:
            objects_by_id[obj_id] = obj

    n_dangling = 0
    for obj in objects_by_id.values():
        if obj.get("type") != "relationship":
            continue
        if obj.get("source_ref") not in objects_by_id or obj.get("target_ref") not in objects_by_id:
            n_dangling += 1

    return StixParseResult(objects_by_id=objects_by_id, n_dangling_relationships=n_dangling)


def iter_stix_nodes(
    result: StixParseResult, *, graph_key: str, graph_role: str, source_version: str
) -> Iterator[NodeRecord]:
    """Yield one ``NodeRecord`` per non-relationship STIX object with an ``id``."""
    for obj_id, obj in result.objects_by_id.items():
        if obj.get("type") == "relationship":
            continue
        yield NodeRecord(
            node_id=make_node_id(graph_key, obj_id),
            graph_key=graph_key,
            graph_role=graph_role,
            source_id=obj_id,
            external_id=_external_id(obj),
            name=obj.get("name") or "",
            node_type=obj.get("type"),
            metadata=json.dumps({}, sort_keys=True),
            has_text=bool(obj.get("description")),
            source_version=source_version,
            raw_sha256=_record_sha256(obj),
        )


def _external_id(obj: dict[str, Any]) -> str:
    refs = obj.get("external_references") or []
    if refs and "external_id" in refs[0]:
        return refs[0]["external_id"] or ""
    return ""


def iter_stix_edges(
    result: StixParseResult, *, graph_key: str, graph_role: str
) -> Iterator[EdgeRecord]:
    """Yield one ``EdgeRecord`` per relationship object with both endpoints present."""
    for obj in result.objects_by_id.values():
        if obj.get("type") != "relationship":
            continue
        source_ref = obj.get("source_ref")
        target_ref = obj.get("target_ref")
        if source_ref not in result.objects_by_id or target_ref not in result.objects_by_id:
            continue
        relation = obj.get("relationship_type") or "related-to"
        src_id = make_node_id(graph_key, source_ref)
        dst_id = make_node_id(graph_key, target_ref)
        edge_id = hashlib.sha1(f"{src_id}|{dst_id}|{relation}".encode("utf-8")).hexdigest()
        yield EdgeRecord(
            edge_id=edge_id,
            src_id=src_id,
            dst_id=dst_id,
            graph_role=graph_role,
            relation=relation,
            directed=True,
        )


def iter_stix_texts(
    result: StixParseResult, *, graph_key: str, text_variants: dict[str, list[str]]
) -> Iterator[TextRecord]:
    """Yield ``TextRecord`` rows per ``text_variants`` (variant -> ordered field names).

    Supported field names: ``description``, ``detection`` (``x_mitre_detection``).
    """
    field_map = {
        "description": "description",
        "detection": "x_mitre_detection",
    }
    for obj_id, obj in result.objects_by_id.items():
        if obj.get("type") == "relationship":
            continue
        node_id = make_node_id(graph_key, obj_id)
        for variant, fields in text_variants.items():
            parts = []
            for field in fields:
                key = field_map.get(field, field)
                value = obj.get(key) or ""
                if value:
                    parts.append(value.strip())
            text = ". ".join(parts)
            yield TextRecord(
                node_id=node_id,
                text_variant=variant,
                text=text,
                text_sha256=_sha256_text(text),
                n_chars=len(text),
            )
