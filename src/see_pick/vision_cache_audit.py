"""Deterministic, secret-free summaries of cached vision observations."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from .visual_canonicalization import mark_duplicate_mask_conflicts
from .domain import VisionObservation
from .vision_cache import CACHE_FORMAT_VERSION


def audit_scene(
    scene_id: str,
    image_key: str,
    observation: VisionObservation,
    *, reference_recovery=None,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for obj in mark_duplicate_mask_conflicts(observation.objects):
        box = obj.bbox
        quality = obj.quality
        assert quality is not None
        records.append(
            {
                "object_id": obj.object_id,
                "label": obj.label,
                "attributes": list(obj.attributes),
                "confidence": obj.confidence,
                "bbox": [box.u_min, box.v_min, box.u_max, box.v_max],
                "mask_area": quality.mask_area,
                "mask_in_box_area": quality.mask_in_box_area,
                "mask_in_box_ratio": quality.mask_in_box_ratio,
                "is_groundable": quality.groundable,
                "quality": asdict(quality),
            }
        )
    return {
        **({"reference_recovery": [asdict(item) for item in reference_recovery.records],
            "absence_evidence": [asdict(item) for item in reference_recovery.absence_evidence],
            "recovered_objects": [obj.object_id for obj in reference_recovery.objects]}
           if reference_recovery is not None else {}),
        "scene_id": scene_id,
        "image_key": image_key,
        "inventory_count": observation.inventory_count,
        "segmented_count": len(observation.objects),
        "recovery": [asdict(item) for item in observation.recovery],
        "dropped": [
            {
                "object_id": item.object_id,
                "label": item.label,
                "reason": item.reason,
            }
            for item in observation.dropped
        ],
        "objects": records,
    }


def write_cache_audit(
    path: Path,
    scenes: Sequence[dict[str, object]],
    pipeline_fingerprint: str,
) -> Path:
    payload = {
        "version": CACHE_FORMAT_VERSION,
        "pipeline_fingerprint": pipeline_fingerprint,
        "scenes": list(scenes),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
