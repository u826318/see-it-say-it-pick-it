"""Closed-set color measurements over RGB foreground, computed during staging.

HSV separates neutral surfaces; chromatic pixels use nearest Lab prototypes.
Confidence combines pixel dominance, prototype margin and sample support. It
requires usable mask geometry; source identities and raw descriptions survive.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from .domain import BBox, DetectedObject, VisualColorEvidence, effective_mask
from .vision_quality import assess_mask_quality


VISUAL_EVIDENCE_CONTRACT = {
    "version": "masked-hsv-lab-v2", "method_prefix": "masked_pixels:",
    "algorithm": "HSV-neutral-Lab-prototype-dominance-margin-sample-support",
    "region": "original-RGB-effective-mask-only",
    "quality_policy": "groundable-mask-quality-required",
    "high_confidence": .9, "min_samples": 16, "mixed_fraction": .15,
    "neutral_saturation": .18, "black_value": .18, "white_value": .86,
    "full_margin": .35, "max_lab_distance": 65.,
    "prototypes_rgb": {
        "red": (255, 0, 0), "orange": (255, 128, 0), "amber": (255, 191, 0),
        "yellow": (255, 230, 0), "green": (30, 170, 50), "blue": (30, 50, 230),
        "purple": (130, 40, 200), "pink": (245, 140, 185), "brown": (130, 70, 25),
    },
    "containment_min_overlap": .9, "containment_max_outside_overlap": .1,
    "containment_max_duplicate_iou": .95,
    "grounding_min_clearance": 1.,
}


def measure_mask_colors(image: np.ndarray, mask: np.ndarray,
                        bbox: BBox) -> VisualColorEvidence | None:
    """Measure only mask/bbox foreground in an original uint8 RGB image."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color evidence requires uint8 RGB HxWx3")
    if mask.shape != image.shape[:2]:
        raise ValueError("color evidence mask must match the image")
    if not assess_mask_quality(mask, bbox).groundable:
        return None
    pixels = image[effective_mask(mask, bbox)]
    contract = VISUAL_EVIDENCE_CONTRACT
    rgb = pixels.astype(np.float32).reshape(-1, 1, 3) / 255
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, 0]
    neutral = hsv[:, 1] < contract["neutral_saturation"]
    labels = np.full(len(pixels), "gray", dtype=object)
    labels[neutral & (hsv[:, 2] <= contract["black_value"])] = "black"
    labels[neutral & (hsv[:, 2] >= contract["white_value"])] = "white"
    margins = np.ones(len(pixels))
    chromatic = ~neutral
    if chromatic.any():
        names = tuple(contract["prototypes_rgb"])
        prototypes = np.array(list(contract["prototypes_rgb"].values()), np.float32)[:, None] / 255
        lab = cv2.cvtColor(rgb[chromatic], cv2.COLOR_RGB2Lab)[:, 0]
        prototype_lab = cv2.cvtColor(prototypes, cv2.COLOR_RGB2Lab)[:, 0]
        distances = np.linalg.norm(lab[:, None] - prototype_lab[None], axis=2)
        order = np.argsort(distances, axis=1)
        best = np.take_along_axis(distances, order[:, :2], axis=1)
        labels[chromatic] = np.array(names, dtype=object)[order[:, 0]]
        margin = (best[:, 1] - best[:, 0]) / np.maximum(best[:, 1], 1e-6)
        margins[chromatic] = np.minimum(1, margin / contract["full_margin"])
        margins[chromatic] *= np.minimum(1, contract["max_lab_distance"] / np.maximum(best[:, 0], 1e-6))
    names, counts = np.unique(labels, return_counts=True)
    order = sorted(range(len(names)), key=lambda index: (-int(counts[index]), str(names[index])))
    colors = tuple(str(names[index]) for index in order
                   if counts[index] / len(pixels) >= contract["mixed_fraction"])
    dominant = str(names[order[0]])
    dominance = float(counts[order[0]] / len(pixels))
    confidence = dominance * float(margins[labels == dominant].mean())
    confidence *= min(1, len(pixels) / contract["min_samples"])
    return VisualColorEvidence(colors or (dominant,), confidence,
        contract["method_prefix"] + contract["version"])


def trusted_measured_color(evidence: VisualColorEvidence | None) -> bool:
    """A single confidently measured color can arbitrate source contradiction."""
    return (evidence is not None and len(evidence.colors) == 1
            and evidence.confidence >= VISUAL_EVIDENCE_CONTRACT["high_confidence"]
            and evidence.method.startswith(VISUAL_EVIDENCE_CONTRACT["method_prefix"]))


def enrich_visual_colors(image: np.ndarray, objects: tuple[DetectedObject, ...]
                         ) -> tuple[DetectedObject, ...]:
    return tuple(replace(obj, color_evidence=measure_mask_colors(image, obj.mask, obj.bbox))
                 if obj.color_evidence is None else obj for obj in objects)
