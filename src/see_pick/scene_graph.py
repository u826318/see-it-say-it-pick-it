"""Deterministic scene normalization and image-space relations."""

from __future__ import annotations

import numpy as np

from .visual_canonicalization import canonicalize_detections
from .domain import CanonicalObject, DetectedObject, RelationStatus, SceneGraph, SpatialRelation, effective_mask
from .relation_evidence import derive_verified_relations, verify_containment


def _centroid(obj: CanonicalObject) -> tuple[float, float]:
    rows, cols = np.nonzero(effective_mask(obj.mask, obj.bbox))
    return float(cols.mean()), float(rows.mean())


def build_relations(
    objects: tuple[CanonicalObject, ...],
) -> tuple[SpatialRelation, ...]:
    objects = tuple(obj for obj in objects if obj.quality is None or obj.quality.groundable)
    relations = list(derive_verified_relations(objects))
    for container in objects:
        if container.role != "container":
            continue
        for subject in objects:
            evidence = verify_containment(subject, container)
            key = (subject.object_id, "inside", container.object_id)
            if evidence.status == RelationStatus.VERIFIED and not any(
                    (r.subject_id, r.predicate, r.object_id) == key for r in relations):
                relations.append(SpatialRelation(*key, confidence=evidence.confidence))
    for index, first in enumerate(objects):
        first_u, first_v = _centroid(first)
        for second in objects[index + 1 :]:
            second_u, second_v = _centroid(second)
            if first_u != second_u:
                left, right = (first, second) if first_u < second_u else (second, first)
                relations.extend(
                    (
                        SpatialRelation(left.object_id, "left_of", right.object_id),
                        SpatialRelation(right.object_id, "right_of", left.object_id),
                    )
                )
            if first_v != second_v:
                above, below = (first, second) if first_v < second_v else (second, first)
                relations.extend(
                    (
                        SpatialRelation(above.object_id, "above", below.object_id),
                        SpatialRelation(below.object_id, "below", above.object_id),
                    )
                )
    return tuple(relations)


def build_scene_graph(
    scene_id: str,
    image_shape: tuple[int, int],
    detections: tuple[DetectedObject, ...],
    *, gripper_max_opening_mm: float = 85.0,
) -> SceneGraph:
    objects = canonicalize_detections(detections)
    return SceneGraph(scene_id, image_shape, objects, build_relations(objects),
                      gripper_max_opening_mm=gripper_max_opening_mm)
