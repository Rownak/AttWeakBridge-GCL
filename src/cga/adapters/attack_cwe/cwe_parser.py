"""CWE XML catalog -> canonical records.

``xml.etree.ElementTree`` under the CWE-7/XHTML namespaces (ported from
legacy/data_preprocessing/1_CWE_xml_to_json.py and 3_CWE_grpah_create.py).
Nodes are ``Weakness`` elements; edges are ``Related_Weaknesses`` entries
filtered by ``nature`` (default ``[ChildOf]``, configurable).
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from cga.data.ids import make_node_id
from cga.data.schemas import EdgeRecord, NodeRecord, TextRecord

_NS = {
    "cwe": "http://cwe.mitre.org/cwe-7",
    "xhtml": "http://www.w3.org/1999/xhtml",
}


@dataclass
class RelatedWeakness:
    """One ``Related_Weakness`` entry."""

    nature: str | None
    cwe_id: str | None


@dataclass
class WeaknessInfo:
    """Flattened text and metadata for one ``Weakness`` element."""

    weakness_id: str
    name: str
    description: str
    extended_description: str
    modes_of_introduction: str
    common_consequences: str
    detection_methods: str
    potential_mitigations: str
    demonstrative_examples: str
    related_weaknesses: list[RelatedWeakness] = field(default_factory=list)


@dataclass
class CweParseResult:
    """Parsed contents of one CWE XML catalog.

    Attributes:
        weaknesses: Every ``Weakness`` element keyed by its ``ID``.
        version: The catalog's ``Version`` attribute.
    """

    weaknesses: dict[str, WeaknessInfo]
    version: str


def _text_or_default(elem: ET.Element | None, default: str = "N/A") -> str:
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def _joined_paragraphs(parent: ET.Element | None, tag: str) -> str | None:
    """Join ``xhtml:p`` children of ``tag`` under ``parent``, or ``None`` if there are none."""
    if parent is None:
        return None
    p_elems = parent.findall(f".//cwe:{tag}/xhtml:p", _NS)
    if not p_elems:
        return None
    return "".join(ET.tostring(e, encoding="unicode", method="text").strip() for e in p_elems)


def _flatten_block(weakness: ET.Element, tag: str) -> str:
    """Flatten the weakness's own direct-child ``tag`` element.

    Only direct children are read: a descendant search (``.//``) also matches the ``Description``
    elements nested in ``Detection_Method`` and ``Mitigation``, which then replace a plain-text
    weakness description (241 of 969 CWEs in v4.20).

    Args:
        weakness: A ``Weakness`` element.
        tag: ``Description`` or ``Extended_Description``.

    Returns:
        ``xhtml:p`` children joined, else the element's whitespace-normalised text, else ``"N/A"``.
    """
    block = weakness.find(f"cwe:{tag}", _NS)
    if block is None:
        return "N/A"
    paragraphs = block.findall("xhtml:p", _NS)
    if paragraphs:
        return "".join(ET.tostring(e, encoding="unicode", method="text").strip() for e in paragraphs)
    text = " ".join("".join(block.itertext()).split())
    return text or "N/A"


def _parse_modes_of_introduction(weakness: ET.Element) -> str:
    parts = []
    for intro in weakness.findall(".//cwe:Introduction", _NS):
        phase = _text_or_default(intro.find(".//cwe:Phase", _NS))
        note = _joined_paragraphs(intro, "Note") or _text_or_default(intro.find(".//cwe:Note", _NS))
        if phase != "N/A" and note != "N/A":
            parts.append(f"{phase}: {note}")
        elif note != "N/A":
            parts.append(f". {note}")
    return "".join(parts)


def _parse_consequences(weakness: ET.Element) -> str:
    parts = []
    for consequence in weakness.findall(".//cwe:Consequence", _NS):
        scopes = [s.text.strip() for s in consequence.findall(".//cwe:Scope", _NS) if s.text]
        impacts = [i.text.strip() for i in consequence.findall(".//cwe:Impact", _NS) if i.text]
        note = _joined_paragraphs(consequence, "Note") or _text_or_default(
            consequence.find(".//cwe:Note", _NS)
        )
        scope_str = "Scopes: " + "".join(s for s in scopes if s != "N/A")
        impact_str = "Impacts: " + "".join(i for i in impacts if i != "N/A")
        if note != "N/A":
            parts.append(f"{scope_str}. {impact_str}. Note: {note}")
    return "".join(parts)


def _parse_detection_methods(weakness: ET.Element) -> str:
    parts = []
    for method in weakness.findall(".//cwe:Detection_Method", _NS):
        method_name = _text_or_default(method.find(".//cwe:Method", _NS))
        description = _joined_paragraphs(method, "Description") or _text_or_default(
            method.find(".//cwe:Description", _NS)
        )
        text = ""
        if method_name != "N/A":
            text += f" Method Name: {method_name}"
        if description != "N/A":
            text += f". Description: {description}"
        parts.append(text)
    return "".join(parts)


def _parse_mitigations(weakness: ET.Element) -> str:
    parts = []
    for mitigation in weakness.findall(".//cwe:Mitigation", _NS):
        phase = _text_or_default(mitigation.find(".//cwe:Phase", _NS))
        description = "".join(
            ET.tostring(e, encoding="unicode", method="text").strip()
            for e in mitigation.findall(".//cwe:Description", _NS)
        )
        parts.append(f"{phase} : {description}")
    return "".join(parts)


def _parse_examples(weakness: ET.Element) -> str:
    parts = []
    for example in weakness.findall(".//cwe:Demonstrative_Example", _NS):
        intro_text = _joined_paragraphs(example, "Intro_Text") or _text_or_default(
            example.find(".//cwe:Intro_Text", _NS)
        )
        body_text = _joined_paragraphs(example, "Body_Text") or _text_or_default(
            example.find(".//cwe:Body_Text", _NS)
        )
        if intro_text != "N/A" and body_text != "N/A":
            parts.append(f"{intro_text}. {body_text}")
        elif body_text != "N/A":
            parts.append(f". {body_text}")
        elif intro_text != "N/A":
            parts.append(f". {intro_text}")
    return "".join(parts)


def parse_cwe_catalog(path: str | Path) -> CweParseResult:
    """Parse a CWE XML catalog into ``WeaknessInfo`` records keyed by CWE ID."""
    tree = ET.parse(Path(path))
    root = tree.getroot()
    version = root.attrib.get("Version", "")

    weaknesses: dict[str, WeaknessInfo] = {}
    for weakness in root.findall(".//cwe:Weakness", _NS):
        weakness_id = weakness.get("ID") or ""
        related = [
            RelatedWeakness(nature=r.get("Nature"), cwe_id=r.get("CWE_ID"))
            for r in weakness.findall(".//cwe:Related_Weaknesses/cwe:Related_Weakness", _NS)
        ]
        weaknesses[weakness_id] = WeaknessInfo(
            weakness_id=weakness_id,
            name=weakness.get("Name") or "",
            description=_flatten_block(weakness, "Description"),
            extended_description=_flatten_block(weakness, "Extended_Description"),
            modes_of_introduction=_parse_modes_of_introduction(weakness),
            common_consequences=_parse_consequences(weakness),
            detection_methods=_parse_detection_methods(weakness),
            potential_mitigations=_parse_mitigations(weakness),
            demonstrative_examples=_parse_examples(weakness),
            related_weaknesses=related,
        )

    return CweParseResult(weaknesses=weaknesses, version=version)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_sha256(info: WeaknessInfo) -> str:
    payload = {
        "ID": info.weakness_id,
        "Name": info.name,
        "Description": info.description,
        "Extended_Description": info.extended_description,
        "Modes_Of_Introduction": info.modes_of_introduction,
        "Common_Consequences": info.common_consequences,
        "Detection_Methods": info.detection_methods,
        "Potential_Mitigations": info.potential_mitigations,
        "Demonstrative_Examples": info.demonstrative_examples,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def iter_cwe_nodes(result: CweParseResult, *, graph_key: str, graph_role: str) -> Iterator[NodeRecord]:
    """Yield one ``NodeRecord`` per ``Weakness`` element."""
    for weakness_id, info in result.weaknesses.items():
        has_text = info.description != "N/A" or info.extended_description != "N/A"
        yield NodeRecord(
            node_id=make_node_id(graph_key, weakness_id),
            graph_key=graph_key,
            graph_role=graph_role,
            source_id=weakness_id,
            external_id=f"CWE-{weakness_id}",
            name=info.name,
            node_type="Weakness",
            metadata=json.dumps({}, sort_keys=True),
            has_text=has_text,
            source_version=result.version,
            raw_sha256=_record_sha256(info),
        )


def iter_cwe_edges(
    result: CweParseResult, *, graph_key: str, graph_role: str, edge_relations: list[str]
) -> Iterator[EdgeRecord]:
    """Yield one ``EdgeRecord`` per ``Related_Weaknesses`` entry whose nature is allowed.

    Edge direction is parent -> child (``source`` = the related CWE, ``target`` =
    the weakness that declares the relation), matching the legacy graph builder.
    Endpoints not present in ``result.weaknesses`` are skipped.
    """
    for weakness_id, info in result.weaknesses.items():
        for related in info.related_weaknesses:
            if related.nature not in edge_relations:
                continue
            if not related.cwe_id or related.cwe_id not in result.weaknesses:
                continue
            src_id = make_node_id(graph_key, related.cwe_id)
            dst_id = make_node_id(graph_key, weakness_id)
            edge_id = hashlib.sha1(
                f"{src_id}|{dst_id}|{related.nature}".encode("utf-8")
            ).hexdigest()
            yield EdgeRecord(
                edge_id=edge_id,
                src_id=src_id,
                dst_id=dst_id,
                graph_role=graph_role,
                relation=related.nature,
                directed=True,
            )


def iter_cwe_texts(
    result: CweParseResult, *, graph_key: str, text_variants: dict[str, list[str]]
) -> Iterator[TextRecord]:
    """Yield ``TextRecord`` rows per ``text_variants`` (variant -> ordered field names).

    Supported field names: ``name``, ``description``, ``extended_description``,
    ``modes_of_introduction``, ``common_consequences``, ``detection_methods``,
    ``potential_mitigations``, ``demonstrative_examples``.
    """
    for weakness_id, info in result.weaknesses.items():
        node_id = make_node_id(graph_key, weakness_id)
        for variant, fields in text_variants.items():
            parts = []
            for attr in fields:
                value = getattr(info, attr, "") or ""
                if value and value != "N/A":
                    parts.append(value.strip())
            text = " ".join(parts)
            yield TextRecord(
                node_id=node_id,
                text_variant=variant,
                text=text,
                text_sha256=_sha256_text(text),
                n_chars=len(text),
            )
