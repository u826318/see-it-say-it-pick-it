"""Deterministic metric footprints and conservative physical impossibility evidence."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .domain import (CanonicalObject, DetectedObject, MetricFootprint, PhysicalEvidence, PixelFitEvidence,
                     Scene, effective_mask)
from .vision_quality import assess_mask_quality


PIXEL_FIT_MARGIN = 0.15


METRIC_GEOMETRY_CONTRACT = {
    "version": "metric-footprint-v2", "min_points": 32, "min_coverage": 0.5,
    "verified_confidence": 0.9, "relative_tolerance": 0.05, "absolute_tolerance_mm": 2.0,
    "xyz_reliability": 0.98, "depth_reliability": 0.94, "provisional_confidence_cap": 0.8,
    "mad_multiplier": 8, "mad_floor_mm": 1.0,
    "requires_complete_mask_boundary": True,
    "local_depth_radius_px": 4, "local_depth_max_range_ratio": 0.1,
    "local_pixel_scale_reliability": 0.6,
}


def measure_mask_footprint(mask: np.ndarray, *, xyz: np.ndarray | None,
                           depth_mm: np.ndarray | None, K: np.ndarray,
                           base_T_camera: np.ndarray | None) -> MetricFootprint | None:
    """Prefer actual finite in-mask points, never table-ray intersections.

    A rigid camera-to-base transform makes base XY the tabletop projection.
    Camera XY alone is only provisional and cannot authorize a physical veto.
    MAD trimming removes isolated coordinate/depth outliers without percentile
    shrinking the edges of an elongated object.
    The last local-depth pixel-scale fallback is an explicit view-plane
    approximation, never verified physical evidence.
    """
    region = np.asarray(mask, dtype=bool)
    if region.ndim != 2 or np.count_nonzero(region) < METRIC_GEOMETRY_CONTRACT["min_points"]:
        return None
    candidates = []
    intrinsics = None
    if xyz is not None and xyz.shape == (*region.shape, 3):
        candidates.append((np.asarray(xyz[region], dtype=float), "xyz", METRIC_GEOMETRY_CONTRACT["xyz_reliability"]))
    if depth_mm is not None and depth_mm.shape == region.shape:
        intrinsics = np.asarray(K, dtype=float)
        if (intrinsics.shape == (3, 3) and np.isfinite(intrinsics).all()
                and intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0):
            rows, cols = np.nonzero(region)
            z = np.asarray(depth_mm[region], dtype=float)
            candidates.append((np.column_stack(((cols - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * z / intrinsics[1, 1], z)), "depth", METRIC_GEOMETRY_CONTRACT["depth_reliability"]))
    transform = np.asarray(base_T_camera, dtype=float) if base_T_camera is not None else None
    if transform is not None and (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-4)):
        return None
    boundary = region & ~cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for raw_points, source, reliability in candidates:
        valid = np.isfinite(raw_points).all(axis=1) & (raw_points[:, 2] > 0)
        points = raw_points[valid]
        if len(points) < METRIC_GEOMETRY_CONTRACT["min_points"]:
            continue
        median = np.median(points, axis=0)
        mad = np.median(np.abs(points - median), axis=0)
        trimmed = (np.abs(points - median) <= np.maximum(
            METRIC_GEOMETRY_CONTRACT["mad_multiplier"] * mad,
            METRIC_GEOMETRY_CONTRACT["mad_floor_mm"])).all(axis=1)
        points = points[trimmed]
        retained = np.zeros(len(raw_points), dtype=bool)
        retained[np.flatnonzero(valid)[trimmed]] = True
        if (METRIC_GEOMETRY_CONTRACT["requires_complete_mask_boundary"]
                and np.any(boundary[region] & ~retained)):
            reliability = min(reliability, METRIC_GEOMETRY_CONTRACT["provisional_confidence_cap"])
        coverage = len(points) / np.count_nonzero(region)
        if (len(points) < METRIC_GEOMETRY_CONTRACT["min_points"]
                or coverage < METRIC_GEOMETRY_CONTRACT["min_coverage"]):
            continue
        if transform is not None:
            points = points @ transform[:3, :3].T + transform[:3, 3]
            projection = "base_xy"
        else:
            projection = "camera_xy"
            reliability = min(reliability, METRIC_GEOMETRY_CONTRACT["provisional_confidence_cap"])
        if region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any():
            reliability = min(reliability, METRIC_GEOMETRY_CONTRACT["provisional_confidence_cap"])
        footprint = _planar_footprint(points[:, :2],
            f"{source}:{projection}:mad8_min_area_rect_caliper_v1", float(reliability * coverage))
        if footprint is not None:
            return footprint

    # Last resort: a consistent depth patch near the in-mask centroid supplies
    # only a view-plane pixel scale. The remaining mask pixels have no measured
    # depth; this distinct approximate method is never accepted by _reliable.
    if (depth_mm is None or depth_mm.shape != region.shape or intrinsics is None
            or intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0):
        return None
    rows, cols = np.nonzero(region)
    center = np.argmin((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2)
    row, col = rows[center], cols[center]
    radius = METRIC_GEOMETRY_CONTRACT["local_depth_radius_px"]
    patch = np.zeros(region.shape, dtype=bool)
    patch[max(0, row - radius):row + radius + 1,
          max(0, col - radius):col + radius + 1] = True
    local = np.asarray(depth_mm[patch & region], dtype=float)
    measured = local[np.isfinite(local) & (local > 0)]
    if len(measured) < METRIC_GEOMETRY_CONTRACT["min_points"]:
        return None
    median = float(np.median(measured))
    mad = float(np.median(np.abs(measured - median)))
    measured = measured[np.abs(measured - median) <= max(
        METRIC_GEOMETRY_CONTRACT["mad_multiplier"] * mad,
        METRIC_GEOMETRY_CONTRACT["mad_floor_mm"])]
    coverage = len(measured) / len(local)
    if (len(measured) < METRIC_GEOMETRY_CONTRACT["min_points"]
            or coverage < METRIC_GEOMETRY_CONTRACT["min_coverage"]
            or np.ptp(measured) > median * METRIC_GEOMETRY_CONTRACT["local_depth_max_range_ratio"]):
        return None
    median = float(np.median(measured))
    planar = np.column_stack((cols * median / intrinsics[0, 0],
                              rows * median / intrinsics[1, 1]))
    return _planar_footprint(planar, "local_depth_pixel_scale:camera_xy:min_area_rect_caliper_v1",
        METRIC_GEOMETRY_CONTRACT["local_pixel_scale_reliability"] * coverage)


def _planar_extents(points: np.ndarray) -> tuple[float, float] | None:
    """Unit-preserving caliper and enclosing-rectangle extents for 2D points."""
    planar = np.ascontiguousarray(points, dtype=np.float32)
    if len(planar) < 3 or not np.isfinite(planar).all():
        return None
    _, extents, _ = cv2.minAreaRect(planar)
    short, long = sorted(extents)
    if short <= 0 or not np.isfinite(extents).all():
        return None
    # Minimum caliper width is a sound insertion-width lower bound even for
    # irregular hulls, whose minimum-area rectangle can have a wider short side.
    hull = cv2.convexHull(planar).reshape(-1, 2).astype(float)
    edges = np.roll(hull, -1, axis=0) - hull
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals[lengths > 0] / lengths[lengths > 0, None]
    projections = hull @ normals.T
    short = float(np.min(np.ptp(projections, axis=0)))
    return short, max(short, float(long))


def _planar_footprint(points: np.ndarray, method: str, confidence: float) -> MetricFootprint | None:
    """Wrap measured/approximated metric points without conflating pixel units."""
    extents = _planar_extents(points)
    return MetricFootprint(*extents, method, float(confidence)) if extents is not None else None


def usable_effective_mask(obj: CanonicalObject,
                          image_shape: tuple[int, int] | None = None) -> np.ndarray | None:
    """Require both reported quality and current in-box mask geometry to be usable."""
    if (obj.mask.ndim != 2 or (image_shape is not None and obj.mask.shape != image_shape)
            or (obj.quality is not None and not obj.quality.groundable)):
        return None
    if not assess_mask_quality(obj.mask, obj.bbox).groundable:
        return None
    return effective_mask(obj.mask, obj.bbox)


def pixel_fit_evidence(target: CanonicalObject, destination: CanonicalObject, *,
                       image_shape: tuple[int, int] | None = None) -> PixelFitEvidence:
    """Compare non-overlapping pixel calipers, independent of semantic role labels."""
    proxy_note = "Destination body mask is only an opening proxy."
    first = usable_effective_mask(target, image_shape)
    second = usable_effective_mask(destination, image_shape)
    quality = {"target_quality": "usable" if first is not None else "unusable",
               "destination_quality": "usable" if second is not None else "unusable"}

    def unknown(reason: str) -> PixelFitEvidence:
        return PixelFitEvidence("unknown", "pixel_body_mask_opening_proxy_unknown",
                                f"{reason} {proxy_note}", PIXEL_FIT_MARGIN, **quality)

    if first is None or second is None or first.shape != second.shape:
        return unknown("Both masks must have usable quality and matching image geometry.")
    if np.any(first & second):
        return unknown("Overlapping masks cannot establish a fit proxy.")
    extents = []
    for mask in (first, second):
        rows, cols = np.nonzero(mask)
        measured = _planar_extents(np.column_stack((cols, rows)))
        if measured is None:
            return unknown("Nonzero caliper geometry is unavailable.")
        extents.append(measured[0])
    target_width_px, opening_proxy_px = extents
    ratio = target_width_px / opening_proxy_px
    # Ratio comparison preserves the exact 1.15 boundary despite product rounding.
    status = ("closed" if ratio <= 1 - PIXEL_FIT_MARGIN else
              "violated" if ratio > 1 + PIXEL_FIT_MARGIN else "unknown")
    return PixelFitEvidence(status, f"pixel_body_mask_opening_proxy_{status}",
        f"Pixel caliper ratio with engineering margin; {proxy_note}", PIXEL_FIT_MARGIN,
        target_width_px, opening_proxy_px, ratio, **quality)


def enrich_metric_geometry(scene: Scene, detections: tuple[DetectedObject, ...]) -> tuple[DetectedObject, ...]:
    """The single staging-time enrichment path for base and recovered masks."""
    K = np.asarray(scene.calibration.get("K", []), dtype=float)
    transform = scene.calibration.get("base_T_camera")
    return tuple(replace(obj, metric_footprint=measure_mask_footprint(
        effective_mask(obj.mask, obj.bbox), xyz=scene.xyz, depth_mm=scene.depth_mm,
        K=K, base_T_camera=transform)) for obj in detections)


def _reliable(footprint: MetricFootprint | None, obj: CanonicalObject) -> bool:
    return (footprint is not None and footprint.confidence >= METRIC_GEOMETRY_CONTRACT["verified_confidence"]
            and footprint.method in {
                "xyz:base_xy:mad8_min_area_rect_caliper_v1",
                "depth:base_xy:mad8_min_area_rect_caliper_v1"}
            and (obj.quality is None or obj.quality.groundable))


def _tolerance(footprint: MetricFootprint) -> float:
    """Explicit engineering bounds, not a statistically calibrated probability.

    Allow 5% systematic scale error, 2 mm absolute error, and lost-coverage
    uncertainty proportional to the long extent. Borderline cases never veto.
    """
    return ((METRIC_GEOMETRY_CONTRACT["relative_tolerance"] + 1 - footprint.confidence)
            * footprint.max_extent_mm + METRIC_GEOMETRY_CONTRACT["absolute_tolerance_mm"])


def gripper_evidence(obj: CanonicalObject, max_opening_mm: float) -> PhysicalEvidence:
    footprint = obj.metric_footprint
    if (not _reliable(footprint, obj) or not np.isfinite(max_opening_mm)
            or max_opening_mm <= 0):
        return PhysicalEvidence("unknown", "gripper_geometry_unknown", "No reliable metric grasp width.")
    lower = max(0.0, footprint.min_extent_mm - _tolerance(footprint))
    impossible = lower > max_opening_mm
    return PhysicalEvidence("impossible" if impossible else "not_proven_impossible",
        "gripper_limit_exceeded" if impossible else "gripper_limit_not_exceeded",
        f"Metric insertion width lower bound {lower:.2f} mm; gripper opening {max_opening_mm:.2f} mm.",
        lower, float(max_opening_mm))


def fit_evidence(target: CanonicalObject, destination: CanonicalObject) -> PhysicalEvidence:
    if destination.role != "container":
        return PhysicalEvidence("unknown", "container_opening_unknown",
                                "Destination has no explicit container-opening role.")
    first, second = target.metric_footprint, destination.metric_footprint
    if not _reliable(first, target) or not _reliable(second, destination):
        return PhysicalEvidence("unknown", "container_geometry_unknown", "No reliable metric fit bounds.")
    lower = max(0.0, first.min_extent_mm - _tolerance(first))
    # The enclosing rectangle's long side bounds both axes. Its sqrt(2)
    # diagonal is an upper bound for any planar opening direction, allowing
    # rotation and avoiding a false denial from independent side comparisons.
    upper = np.sqrt(2.0) * (second.max_extent_mm + _tolerance(second))
    impossible = lower > upper
    return PhysicalEvidence("impossible" if impossible else "not_proven_impossible",
        "container_nonfit" if impossible else "container_nonfit_not_proven",
        f"Metric insertion lower bound {lower:.2f} mm; opening upper bound {upper:.2f} mm.",
        lower, float(upper))
