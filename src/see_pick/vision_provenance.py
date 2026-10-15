"""Stable identity for every input that changes cached vision observations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .inventory_contract import (
    QWEN_BOX_SCALE,
    QWEN_CONFIDENCE_SCALE,
    inventory_json_schema,
)


INVENTORY_PROMPT_VERSION = "inventory-v8"


def inventory_prompt() -> str:
    return (
        "Inventory every distinct visible manipulable tabletop object, with one record "
        "per visible instance. Pay particular attention to gears, cubes, generic parts, "
        "trays, plates, saucers, cups, fruit, and visible human hands. Use the most "
        "specific visually supported noun, with color and discriminative attributes "
        "stored separately. Draw a tight box around each individual object rather than "
        "its surrounding surface or a group. Include partially occluded objects when "
        "they remain individually identifiable. Do not invent hidden objects or infer "
        "objects from an instruction. Return exactly one JSON document matching the "
        "enforced schema. Use stable unique object_id values. "
        "Boxes use Qwen's native [x0,y0,x1,y1] coordinate space from 0 to 1000; "
        "they are normalized internally. Confidence uses a 0 to 100 percentage and "
        "is normalized internally. Set estimated_width_mm to null when uncertain. Use "
        "relations for supports and in_hand_of evidence. Classify every item with role "
        "graspable_object, container, hand, human, background, or support_surface. "
        "Use container only for a visible insertion opening, not a support surface. "
        "For containers, provide separate graspable evidence; leave graspable false when uncertain."
    )


def vision_pipeline_manifest(config: Mapping[str, Any]) -> dict[str, object]:
    from .vision import RECOVERY_PROMPT
    from .vision_quality import QUALITY_CONTRACT, RECOVERY_CONTRACT
    from .metric_geometry import METRIC_GEOMETRY_CONTRACT
    from .visual_evidence import VISUAL_EVIDENCE_CONTRACT

    try:
        inventory = config["vision"]["inventory"]
        segmentation = config["vision"]["segmentation"]
        return {
            "inventory": {
                "model_id": str(inventory["model_id"]),
                "revision": str(inventory["revision"]),
                "max_pixels": (
                    int(inventory["max_pixels"])
                    if inventory.get("max_pixels") is not None
                    else None
                ),
                "load_in_4bit": bool(inventory.get("load_in_4bit", False)),
                "prompt_version": INVENTORY_PROMPT_VERSION,
                "prompt": inventory_prompt(),
                "schema": inventory_json_schema(),
                "normalization": {
                    "box_scale": QWEN_BOX_SCALE,
                    "confidence_scale": QWEN_CONFIDENCE_SCALE,
                    "range_policy": "clamp",
                },
            },
            "segmentation": {
                "model_id": str(segmentation["model_id"]),
                "revision": str(segmentation["revision"]),
            },
            "quality": dict(QUALITY_CONTRACT),
            "metric_geometry": dict(METRIC_GEOMETRY_CONTRACT),
            "visual_evidence": dict(VISUAL_EVIDENCE_CONTRACT),
            "recovery": {**RECOVERY_CONTRACT, "prompt": RECOVERY_PROMPT},
        }
    except KeyError as exc:
        raise ValueError(f"missing vision configuration: {exc.args[0]}") from exc


def vision_pipeline_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        vision_pipeline_manifest(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def recovery_pipeline_fingerprint(config, *, base_fingerprint: str, queries,
                                  parser_fingerprint: str, contract=None) -> str:
    from dataclasses import asdict
    import inspect
    from . import perception_recovery
    from .perception_recovery import RECOVERY_CONTRACT
    from .visual_evidence import VISUAL_EVIDENCE_CONTRACT
    settings = config["perception_recovery"]
    manifest = {"base_vision": base_fingerprint, "parser": parser_fingerprint,
        "queries": sorted((asdict(query) for query in queries), key=lambda item: json.dumps(item, sort_keys=True)),
        "detector": {"model_id": settings["model_id"], "revision": settings["revision"],
                     "min_confidence": settings["min_confidence"]},
        "segmentation": dict(config["vision"]["segmentation"]),
        "implementation": inspect.getsource(perception_recovery),
        "recovery": RECOVERY_CONTRACT if contract is None else contract,
        "visual_evidence": VISUAL_EVIDENCE_CONTRACT}
    return "sha256:" + hashlib.sha256(json.dumps(manifest, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()
