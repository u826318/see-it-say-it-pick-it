"""Mask quality contracts independent of inventory and lexical normalization."""
from __future__ import annotations

from dataclasses import replace
import cv2
import numpy as np

from .domain import BBox, MIN_LEGAL_TARGET_AREA, VisionQuality, effective_mask

QUALITY_CONTRACT = {"version": "mask-quality-v4", "effective_region": "boolean-mask-intersect-half-open-bbox-v1",
    "component_connectivity": 8,
    "min_target_area": MIN_LEGAL_TARGET_AREA, "max_hole_ratio": 0.5,
    "min_dominant_component_ratio": 0.9, "min_extent_ratio": 0.5,
    "duplicate_mask_iou": 0.95, "same_object_iou": 0.8}
RECOVERY_CONTRACT = {"version": "targeted-perception-v2",
    "identity_policy": "preserve-original-fields-update-box-confidence-only", "max_extra_passes": 1,
    "min_inventory_confidence": 0.5, "scope": "failed-or-uncertain-inventory-only"}

def assess_mask_quality(mask: np.ndarray, box: BBox) -> VisionQuality:
    effective = effective_mask(mask, box)
    region = effective[box.v_min:box.v_max, box.u_min:box.u_max].astype(np.uint8)
    area = int(region.sum()); box_width = box.u_max - box.u_min; box_height = box.v_max - box.v_min
    reasons = []; components = largest_area = 0; largest_ratio = hole_ratio = width_ratio = height_ratio = 0.0
    if not area:
        reasons.append("empty")
    else:
        if area < MIN_LEGAL_TARGET_AREA: reasons.append("tiny")
        count, _, stats, _ = cv2.connectedComponentsWithStats(region,
            connectivity=QUALITY_CONTRACT["component_connectivity"])
        sizes = stats[1:, cv2.CC_STAT_AREA]; components = count - 1
        largest_area = int(sizes.max()); largest_ratio = largest_area / area
        if largest_area < MIN_LEGAL_TARGET_AREA: reasons.append("no_stable_component")
        if largest_ratio < QUALITY_CONTRACT["min_dominant_component_ratio"]: reasons.append("disconnected")
        background = np.pad(1 - region, 1, constant_values=1)
        _, labels = cv2.connectedComponents(background, connectivity=4)
        hole_area = int(((background == 1) & (labels != labels[0, 0])).sum()); hole_ratio = hole_area / (area + hole_area)
        if hole_ratio > QUALITY_CONTRACT["max_hole_ratio"]: reasons.append("hollow")
        rows, cols = np.nonzero(region); width_ratio = float((cols.max()-cols.min()+1)/box_width); height_ratio = float((rows.max()-rows.min()+1)/box_height)
        if max(width_ratio, height_ratio) < QUALITY_CONTRACT["min_extent_ratio"]: reasons.append("loose_box")
    status = "invalid" if largest_area < MIN_LEGAL_TARGET_AREA else "uncertain" if reasons else "usable"
    return VisionQuality(status, tuple(reasons), int(np.count_nonzero(mask)), area, area/(box_width*box_height), components, largest_ratio, hole_ratio, width_ratio, height_ratio)

def assess_detection_quality(mask: np.ndarray, box: BBox, confidence: float) -> VisionQuality:
    quality = assess_mask_quality(mask, box)
    if confidence < RECOVERY_CONTRACT["min_inventory_confidence"]:
        quality = replace(quality, status="uncertain" if quality.groundable else quality.status,
                          reasons=(*quality.reasons, "low_inventory_confidence"))
    return quality

def select_mask_candidate(masks: np.ndarray, scores: np.ndarray, box: BBox) -> np.ndarray:
    qualities = [assess_mask_quality(mask, box) for mask in masks]
    index = max(range(len(qualities)), key=lambda i: (qualities[i].groundable,
        qualities[i].status != "invalid", float(scores[i]), -i))
    return masks[index].astype(bool)
