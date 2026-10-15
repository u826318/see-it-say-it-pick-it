"""Verify advisory semantic relations using observed image-space geometry."""

from __future__ import annotations

import cv2
import numpy as np

from .domain import CanonicalObject, RelationEvidence, RelationStatus, SpatialRelation, effective_mask
from .visual_evidence import VISUAL_EVIDENCE_CONTRACT


# Distinct visible surfaces may touch or partly overlap. Majority overlap of
# the smaller mask cannot distinguish contact from duplicate/occluded surfaces.
_MAX_OVERLAP_FRACTION = 0.5
# One raster pixel of dilation recognizes adjacent boundaries, including corners.
_CONTACT_KERNEL = np.ones((3, 3), dtype=np.uint8)
# A support boundary must cover at least a quarter of the smaller horizontal
# span; isolated edge/corner contact is insufficient evidence of support.
_MIN_SUPPORT_CONTACT_FRACTION = 0.25
_SEMANTIC_PREDICATES = frozenset({"supports", "on_top_of", "in_hand_of", "inside"})


def _usable_masks(subject: CanonicalObject, obj: CanonicalObject) -> bool:
    return (
        (subject.quality is None or subject.quality.groundable)
        and (obj.quality is None or obj.quality.groundable)
        and subject.mask.ndim == obj.mask.ndim == 2
        and subject.mask.shape == obj.mask.shape
    )


def _support_geometry(
    lower: np.ndarray, upper: np.ndarray
) -> tuple[RelationStatus, str]:
    lower_y, lower_x = np.nonzero(lower)
    upper_y, upper_x = np.nonzero(upper)
    if lower_y.mean() <= upper_y.mean():
        return RelationStatus.CONTRADICTED, "geometry:reversed_support_direction"
    columns = np.intersect1d(lower_x, upper_x)
    if not columns.size:
        return RelationStatus.CONTRADICTED, "geometry:horizontal_separation"
    # Compare actual contours in common columns, not just touching boxes. Zero
    # gap permits a shared raster boundary; one means immediately adjacent rows.
    lower_top = np.argmax(lower[:, columns], axis=0)
    upper_bottom = upper.shape[0] - 1 - np.argmax(upper[::-1, columns], axis=0)
    gaps = lower_top - upper_bottom
    contact_columns = np.count_nonzero((gaps >= 0) & (gaps <= 1))
    if not contact_columns:
        if np.any(gaps < 0):
            return RelationStatus.UNCERTAIN, "geometry:overlapping_support_contours"
        return RelationStatus.CONTRADICTED, "geometry:vertical_separation"
    smaller_span = min(int(np.ptp(lower_x)) + 1, int(np.ptp(upper_x)) + 1)
    if contact_columns / smaller_span < _MIN_SUPPORT_CONTACT_FRACTION:
        return RelationStatus.UNCERTAIN, "geometry:partial_support_contact"
    return RelationStatus.VERIFIED, "geometry:directed_support_contact"


def verify_relation(
    subject: CanonicalObject, predicate: str, obj: CanonicalObject
) -> RelationEvidence:
    """Return evidence without promoting unverified model hints to relations."""

    def evidence(status: RelationStatus, reason: str) -> RelationEvidence:
        confidence = (
            0.0 if status == RelationStatus.UNCERTAIN
            else max(0.0, min(1.0, subject.confidence, obj.confidence))
        )
        return RelationEvidence(
            status, predicate, subject.object_id, obj.object_id, confidence, (reason,)
        )

    if subject.object_id == obj.object_id:
        return evidence(RelationStatus.CONTRADICTED, "geometry:self_relation")
    if predicate not in _SEMANTIC_PREDICATES:
        return evidence(RelationStatus.UNCERTAIN, "geometry:unsupported_predicate")
    if predicate == "in_hand_of" and obj.canonical_noun != "hand":
        return evidence(RelationStatus.CONTRADICTED, "geometry:object_is_not_hand")
    if not _usable_masks(subject, obj):
        return evidence(RelationStatus.UNCERTAIN, "geometry:incomplete_masks")

    subject_mask = effective_mask(subject.mask, subject.bbox)
    object_mask = effective_mask(obj.mask, obj.bbox)
    if not subject_mask.any() or not object_mask.any():
        return evidence(RelationStatus.UNCERTAIN, "geometry:incomplete_masks")
    overlap = np.count_nonzero(subject_mask & object_mask)
    if predicate == "inside":
        contract = VISUAL_EVIDENCE_CONTRACT
        subject_area = np.count_nonzero(subject_mask)
        union = np.count_nonzero(subject_mask | object_mask)
        if overlap / union >= contract["containment_max_duplicate_iou"]:
            return evidence(RelationStatus.UNCERTAIN, "geometry:duplicate_containment_surfaces")
        rows, cols = np.nonzero(subject_mask)
        centroid_inside = object_mask[int(round(rows.mean())), int(round(cols.mean()))]
        fraction = overlap / subject_area
        if centroid_inside and fraction >= contract["containment_min_overlap"]:
            return evidence(RelationStatus.VERIFIED, "geometry:contained_centroid_and_mask")
        if not centroid_inside and fraction <= contract["containment_max_outside_overlap"]:
            return evidence(RelationStatus.CONTRADICTED, "geometry:outside_container_mask")
        return evidence(RelationStatus.UNCERTAIN, "geometry:partial_containment")
    smaller_area = min(np.count_nonzero(subject_mask), np.count_nonzero(object_mask))
    if overlap / smaller_area > _MAX_OVERLAP_FRACTION:
        return evidence(RelationStatus.UNCERTAIN, "geometry:excessive_mask_overlap")
    if predicate in {"supports", "on_top_of"}:
        lower, upper = ((subject_mask, object_mask) if predicate == "supports"
                        else (object_mask, subject_mask))
        status, reason = _support_geometry(lower, upper)
        return evidence(status, reason)
    if overlap:
        return evidence(RelationStatus.VERIFIED, "geometry:bounded_mask_overlap")
    contact = cv2.dilate(subject_mask.astype(np.uint8), _CONTACT_KERNEL)
    if np.any(contact.astype(bool) & object_mask):
        return evidence(RelationStatus.VERIFIED, "geometry:mask_boundary_contact")
    return evidence(RelationStatus.CONTRADICTED, "geometry:separated_masks")


def verify_containment(subject: CanonicalObject, container: CanonicalObject) -> RelationEvidence:
    """Image-space containment; no claim about depth or physical insertion."""
    return verify_relation(subject, "inside", container)


def derive_verified_relations(
    objects: tuple[CanonicalObject, ...],
) -> tuple[SpatialRelation, ...]:
    """Promote only geometrically verified hints between known runtime objects."""
    by_id = {obj.object_id: obj for obj in objects}
    relations: dict[tuple[str, str, str], SpatialRelation] = {}
    for subject in objects:
        for predicate, object_id in subject.relation_hints:
            obj = by_id.get(object_id)
            if obj is None:
                continue
            evidence = verify_relation(subject, predicate, obj)
            if evidence.status == RelationStatus.VERIFIED:
                key = (subject.object_id, predicate, object_id)
                relations[key] = SpatialRelation(
                    *key, confidence=evidence.confidence, source="geometry"
                )
    return tuple(relations.values())
