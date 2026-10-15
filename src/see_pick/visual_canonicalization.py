"""Visual conflict detection and deduplication over lexically normalized identity."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .canonicalization import canonicalize_detection, decompose_noun, normalize_colors
from .domain import CanonicalObject, DetectedObject, effective_mask
from .vision_quality import QUALITY_CONTRACT, assess_detection_quality

def _iou(first: DetectedObject, second: DetectedObject) -> float:
    left, right = effective_mask(first.mask, first.bbox), effective_mask(second.mask, second.bbox)
    if left.shape != right.shape: return 0.0
    intersection = np.logical_and(left, right).sum(); union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0

def _compatible_identity(left: DetectedObject, right: DetectedObject) -> bool:
    a, b = decompose_noun(left.label), decompose_noun(right.label)
    ac = set(normalize_colors((*a.colors, *left.attributes))); bc = set(normalize_colors((*b.colors, *right.attributes)))
    return a.head == b.head and not (ac and bc and ac.isdisjoint(bc))

def mark_duplicate_mask_conflicts(detections: tuple[DetectedObject, ...]) -> tuple[DetectedObject, ...]:
    conflicts = [[] for _ in detections]
    for index, first in enumerate(detections):
        for other_index in range(index + 1, len(detections)):
            second = detections[other_index]
            if not _compatible_identity(first, second) and _iou(first, second) >= QUALITY_CONTRACT["duplicate_mask_iou"]:
                conflicts[index].append(second.object_id); conflicts[other_index].append(first.object_id)
    result = []
    for item, ids in zip(detections, conflicts, strict=True):
        quality = item.quality or assess_detection_quality(item.mask, item.bbox, item.confidence)
        if quality.status == "conflict": quality = assess_detection_quality(item.mask, item.bbox, item.confidence)
        if ids: quality = replace(quality, status="conflict", reasons=(*quality.reasons, "duplicate_mask_conflict"), conflicting_object_ids=tuple(sorted(ids)))
        result.append(replace(item, quality=quality))
    return tuple(result)

def canonicalize_detections(detections: tuple[DetectedObject, ...]) -> tuple[CanonicalObject, ...]:
    detections = mark_duplicate_mask_conflicts(detections); deduplicated = []; groups = []
    for candidate in detections:
        duplicate_index = next((i for i, existing in enumerate(deduplicated)
            if _compatible_identity(existing, candidate) and existing.quality.status != "conflict"
            and candidate.quality.status != "conflict" and _iou(existing, candidate) >= QUALITY_CONTRACT["same_object_iou"]), None)
        if duplicate_index is None: deduplicated.append(candidate); groups.append({candidate.object_id}); continue
        groups[duplicate_index].add(candidate.object_id)
        if candidate.confidence > deduplicated[duplicate_index].confidence: deduplicated[duplicate_index] = candidate
    source_to_runtime = {source_id: f"obj_{index:03d}" for index, group in enumerate(groups, 1) for source_id in group}
    objects = []
    for index, detection in enumerate(deduplicated, 1):
        canonical = canonicalize_detection(detection, f"obj_{index:03d}")
        relations = tuple((predicate, source_to_runtime[target]) for predicate, target in detection.relation_hints
                          if target in source_to_runtime and source_to_runtime[target] != canonical.object_id)
        quality = canonical.quality
        if quality is not None:
            quality = replace(quality, conflicting_object_ids=tuple(sorted({source_to_runtime[source]
                for source in quality.conflicting_object_ids if source in source_to_runtime})))
        objects.append(replace(canonical, relation_hints=relations, quality=quality))
    return tuple(objects)
