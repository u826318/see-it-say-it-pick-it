"""Convert a planned first pick into a safe image-space target pixel."""

from __future__ import annotations

import cv2
import numpy as np
from .visual_evidence import VISUAL_EVIDENCE_CONTRACT
from .vision_quality import QUALITY_CONTRACT

from see_pick.domain import (
    MIN_LEGAL_TARGET_AREA,
    CanonicalObject,
    PickAction,
    PlanDecision,
    SceneGraph,
    effective_mask,
)


class GroundingError(ValueError):
    """Raised when a plan cannot be grounded to a valid foreground pixel."""


def _validated_mask(mask: np.ndarray) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise GroundingError("mask must be a numpy array")
    if mask.ndim != 2:
        raise GroundingError("mask must be two-dimensional")
    if mask.dtype != np.bool_:
        raise GroundingError("mask must have boolean dtype")
    if not np.any(mask):
        raise GroundingError("mask is empty")
    return mask


def _nearest_mask_pixel_to_centroid(mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.nonzero(mask)
    centroid_v = float(rows.mean())
    centroid_u = float(cols.mean())
    squared_distance = (rows - centroid_v) ** 2 + (cols - centroid_u) ** 2
    index = int(np.argmin(squared_distance))
    return int(cols[index]), int(rows[index])


def select_target_pixel(mask: np.ndarray) -> tuple[int, int]:
    """Select a deterministic interior foreground pixel, returned as ``(u, v)``."""

    valid_mask = _validated_mask(mask)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(valid_mask.astype(np.uint8),
        connectivity=QUALITY_CONTRACT["component_connectivity"])
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < MIN_LEGAL_TARGET_AREA:
        raise GroundingError("mask has no stable target component")
    valid_mask = labels == largest
    padded = np.pad(valid_mask.astype(np.uint8), pad_width=1, constant_values=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    maximum = float(distance.max())

    # One-pixel-wide components have no meaningful interior clearance. Their
    # centroid-nearest foreground pixel is more stable than an endpoint chosen
    # by distance-transform tie-breaking.
    if maximum <= VISUAL_EVIDENCE_CONTRACT["grounding_min_clearance"]:
        return _nearest_mask_pixel_to_centroid(valid_mask)

    flat_index = int(np.argmax(distance))
    v, u = np.unravel_index(flat_index, distance.shape)
    if not valid_mask[v, u]:
        raise GroundingError("selected target pixel lies outside the mask")
    return int(u), int(v)


def validate_target_mask(
    target: CanonicalObject,
    graph: SceneGraph,
) -> None:
    legal_target_region(target, graph)


def legal_target_region(
    target: CanonicalObject,
    graph: SceneGraph,
    *,
    min_area: int = MIN_LEGAL_TARGET_AREA,
) -> np.ndarray:
    """Return foreground pixels that also lie inside the detected box."""

    if target.quality is not None and not target.quality.groundable:
        raise GroundingError("unusable vision evidence: " + ", ".join(target.quality.reasons))
    mask = _validated_mask(target.mask)
    if mask.shape != graph.image_shape:
        raise GroundingError(
            f"mask shape {mask.shape!r} does not match image shape {graph.image_shape!r}"
        )

    height, width = graph.image_shape
    box = target.bbox
    if box.u_max > width or box.v_max > height:
        raise GroundingError("bounding box extends outside the image")

    legal = effective_mask(mask, box)
    if int(legal.sum()) < min_area:
        raise GroundingError(
            "mask and bounding box intersection is too small for a stable target"
        )
    return legal


def resolve_first_pick(
    decision: PlanDecision, graph: SceneGraph
) -> tuple[int, int]:
    """Ground the object named by the plan's first pick action."""

    first_action = decision.actions[0]
    if not isinstance(first_action, PickAction):
        raise GroundingError("first action must be a pick")

    target = graph.object_by_id(first_action.object_id)
    if target is None:
        raise GroundingError(f"unknown first-pick object: {first_action.object_id}")
    legal_mask = legal_target_region(target, graph)
    for other in graph.objects:
        if (
            other.object_id != target.object_id
            and other.confidence > target.confidence
            and other.mask.shape == graph.image_shape
        ):
            legal_mask &= ~effective_mask(other.mask, other.bbox)
    if np.count_nonzero(legal_mask) < MIN_LEGAL_TARGET_AREA:
        raise GroundingError("higher-confidence objects leave no stable target region")
    return select_target_pixel(legal_mask)
