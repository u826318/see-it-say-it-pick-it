"""Strict structured-output contract for visual object inventories."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator


QWEN_BOX_SCALE = 1000.0
QWEN_CONFIDENCE_SCALE = 100.0


@dataclass(frozen=True)
class InventoryRelation:
    predicate: str
    object_id: str


@dataclass(frozen=True)
class InventoryItem:
    object_id: str
    label: str
    attributes: tuple[str, ...]
    confidence: float
    box: tuple[float, float, float, float]
    estimated_width_mm: float | None
    relations: tuple[InventoryRelation, ...]
    role: str = "graspable_object"
    graspable: bool | None = None


class InventoryValidationError(ValueError):
    def __init__(self, violations: tuple[str, ...]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def inventory_json_schema() -> dict[str, object]:
    relation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["predicate", "object_id"],
        "properties": {
            "predicate": {"type": "string", "minLength": 1, "maxLength": 80},
            "object_id": {"type": "string", "minLength": 1, "maxLength": 120},
        },
    }
    item = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "object_id",
            "label",
            "attributes",
            "confidence",
            "box",
            "estimated_width_mm",
            "relations",
        ],
        "properties": {
            "object_id": {"type": "string", "minLength": 1, "maxLength": 120},
            "label": {"type": "string", "minLength": 1, "maxLength": 120},
            "attributes": {
                "type": "array",
                "maxItems": 12,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 120},
            },
            "confidence": {"type": "number"},
            "box": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
            },
            "estimated_width_mm": {
                "anyOf": [
                    {"type": "number", "exclusiveMinimum": 0},
                    {"type": "null"},
                ]
            },
            "relations": {
                "type": "array",
                "maxItems": 20,
                "items": relation,
            },
            "role": {"enum": ["graspable_object", "container", "hand", "human", "background", "support_surface"]},
            "graspable": {"type": "boolean"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["objects"],
        "properties": {
            "objects": {
                "type": "array",
                "minItems": 1,
                "maxItems": 40,
                "items": item,
            }
        },
    }


def _json_path(parts: Any) -> str:
    return "$" + "".join(
        f"[{part}]" if isinstance(part, int) else f".{part}" for part in parts
    )


def _normalize_box(values: list[object]) -> tuple[float, float, float, float]:
    return tuple(  # type: ignore[return-value]
        _normalize_qwen_number(value, QWEN_BOX_SCALE) for value in values
    )


def _normalize_confidence(value: object) -> float:
    return _normalize_qwen_number(value, QWEN_CONFIDENCE_SCALE)


def _normalize_qwen_number(value: object, scale: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return number
    return min(1.0, max(0.0, number / scale))


def _canonicalize_object_ids(objects: list[dict[str, Any]]) -> None:
    bases = [raw["object_id"].strip() for raw in objects]
    counts = Counter(base for base in bases if base)
    reserved = set(bases)
    used: set[str] = set()
    unambiguous_ids: dict[str, str] = {}

    for raw, base in zip(objects, bases, strict=True):
        if not base:
            continue
        if counts[base] == 1:
            canonical = base
        else:
            suffix = 1
            canonical = f"{base}_{suffix}"
            while canonical in reserved or canonical in used:
                suffix += 1
                canonical = f"{base}_{suffix}"
        raw["object_id"] = canonical
        used.add(canonical)
        if counts[base] == 1:
            unambiguous_ids[base] = canonical

    for raw in objects:
        relations = []
        for relation in raw["relations"]:
            target = relation["object_id"].strip()
            if counts[target] > 1 or target == raw["object_id"]:
                continue
            if target in unambiguous_ids:
                relation["object_id"] = unambiguous_ids[target]
            relations.append(relation)
        raw["relations"] = relations


def validate_inventory_json(text: str) -> tuple[InventoryItem, ...]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise InventoryValidationError((f"response is not valid JSON: {exc}",)) from exc

    validator = Draft202012Validator(inventory_json_schema())
    schema_errors = sorted(
        validator.iter_errors(payload), key=lambda error: tuple(error.absolute_path)
    )
    if schema_errors:
        raise InventoryValidationError(
            tuple(
                f"{_json_path(error.absolute_path)}: {error.message}"
                for error in schema_errors
            )
        )

    _canonicalize_object_ids(payload["objects"])

    violations: list[str] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(payload["objects"]):
        prefix = f"$.objects[{index}]"
        object_id = raw["object_id"].strip()
        label = raw["label"].strip()
        if not object_id:
            violations.append(f"{prefix}.object_id must not be blank")
        elif object_id in seen_ids:
            violations.append(f"{prefix}.object_id duplicate object_id {object_id!r}")
        seen_ids.add(object_id)
        if not label:
            violations.append(f"{prefix}.label must not be blank")

        attributes = tuple(value.strip() for value in raw["attributes"])
        if any(not value for value in attributes):
            violations.append(f"{prefix}.attributes must not contain blank values")

        confidence = _normalize_confidence(raw["confidence"])
        box = _normalize_box(raw["box"])
        width = raw["estimated_width_mm"]
        if not math.isfinite(confidence):
            violations.append(f"{prefix}.confidence must be finite")
        if any(not math.isfinite(value) for value in box):
            violations.append(f"{prefix}.box values must be finite")
        elif not (box[0] < box[2] and box[1] < box[3]):
            violations.append(f"{prefix}.box coordinates must be ordered")
        if width is not None and (
            not math.isfinite(float(width)) or float(width) <= 0
        ):
            violations.append(f"{prefix}.estimated_width_mm must be positive and finite")

        for relation_index, relation in enumerate(raw["relations"]):
            relation_prefix = f"{prefix}.relations[{relation_index}]"
            if not relation["predicate"].strip():
                violations.append(f"{relation_prefix}.predicate must not be blank")
            target = relation["object_id"].strip()
            if not target:
                violations.append(f"{relation_prefix}.object_id must not be blank")

    if violations:
        raise InventoryValidationError(tuple(violations))

    from .domain import infer_scene_role
    return tuple(
        InventoryItem(
            object_id=raw["object_id"].strip(),
            label=raw["label"].strip(),
            attributes=tuple(value.strip() for value in raw["attributes"]),
            confidence=_normalize_confidence(raw["confidence"]),
            box=_normalize_box(raw["box"]),
            estimated_width_mm=(
                float(raw["estimated_width_mm"])
                if raw["estimated_width_mm"] is not None
                else None
            ),
            relations=tuple(
                InventoryRelation(
                    predicate=relation["predicate"].strip(),
                    object_id=relation["object_id"].strip(),
                )
                for relation in raw["relations"]
            ),
            role=raw.get("role") or infer_scene_role(raw["label"]),
            graspable=raw.get("graspable"),
        )
        for raw in payload["objects"]
    )
