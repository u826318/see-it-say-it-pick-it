"""Portable, deterministic cache for expensive local vision observations."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .domain import (AbsenceEvidence, BBox, DetectedObject, DroppedDetection,
                     MetricFootprint, PerceptionRecovery, Scene, VisionObservation, VisionQuality, VisualColorEvidence)


CACHE_FORMAT_VERSION = 4


class CacheCompatibilityError(ValueError):
    """Raised when an otherwise readable entry belongs to another contract."""


@dataclass(frozen=True)
class CacheProbe:
    status: Literal["missing", "valid", "incompatible", "corrupt"]
    observation: VisionObservation | None = None
    error: str | None = None

    @property
    def objects(self) -> tuple[DetectedObject, ...] | None:
        return self.observation.objects if self.observation is not None else None


def image_cache_key(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def scene_geometry_fingerprint(scene: Scene | None) -> str:
    """Bind metric measurements to every input consumed by geometry enrichment."""
    calibration = scene.calibration if scene is not None else {}
    inputs = {
        "depth_mm": scene.depth_mm if scene is not None else None,
        "xyz": scene.xyz if scene is not None else None,
        "K": np.asarray(calibration.get("K", []), dtype=float),
        "base_T_camera": (np.asarray(calibration["base_T_camera"], dtype=float)
                          if calibration.get("base_T_camera") is not None else None),
    }
    digests = {name: image_cache_key(value) if value is not None else None
               for name, value in inputs.items()}
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode("ascii")).hexdigest()


class VisionCacheBackend:
    """Store and replay ``VisionBackend`` observations by image digest."""

    def __init__(self, root: Path, *, pipeline_fingerprint: str) -> None:
        self.root = Path(root)
        if not pipeline_fingerprint:
            raise ValueError("pipeline_fingerprint must not be empty")
        self.pipeline_fingerprint = pipeline_fingerprint

    def _path(self, image: np.ndarray) -> Path:
        return self.root / f"{image_cache_key(image)}.npz"

    def probe(self, image: np.ndarray, *, scene: Scene | None = None) -> CacheProbe:
        if not self._path(image).is_file():
            return CacheProbe("missing")
        try:
            return CacheProbe("valid", observation=self.read_observation(image, scene=scene))
        except CacheCompatibilityError as exc:
            return CacheProbe("incompatible", error=str(exc))
        except Exception as exc:
            return CacheProbe("corrupt", error=f"{type(exc).__name__}: {exc}")

    def quarantine(self, image: np.ndarray, root: Path) -> Path:
        source = self._path(image)
        if not source.is_file():
            raise FileNotFoundError(f"vision cache entry does not exist: {source}")
        destination = Path(root) / "corrupt-cache" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        return destination

    def store(
        self, image: np.ndarray, observation: VisionObservation, *, scene: Scene | None = None
    ) -> Path:
        return self._store(image, observation, scene=scene)

    def _store(self, image: np.ndarray, observation: VisionObservation, *,
               allow_empty: bool = False, extra_metadata: dict | None = None,
               scene: Scene | None = None) -> Path:
        from .visual_canonicalization import mark_duplicate_mask_conflicts

        objects = observation.objects
        if not objects and not allow_empty:
            raise ValueError("refusing to cache an empty observation")
        height, width = image.shape[:2]
        for obj in objects:
            if obj.mask.shape != (height, width):
                raise ValueError(
                    f"mask for {obj.object_id} has shape {obj.mask.shape}, "
                    f"expected {(height, width)}"
                )
            if obj.bbox.u_max > width or obj.bbox.v_max > height:
                raise ValueError(
                    f"bounding box for {obj.object_id} extends outside the image"
                )

        objects = mark_duplicate_mask_conflicts(objects)
        metadata = {
            **(extra_metadata or {}),
            "version": CACHE_FORMAT_VERSION,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "geometry_fingerprint": scene_geometry_fingerprint(scene),
            "image_shape": [height, width],
            "inventory_count": observation.inventory_count,
            "recovery": [asdict(item) for item in observation.recovery],
            "dropped": [
                {
                    "object_id": item.object_id,
                    "label": item.label,
                    "reason": item.reason,
                }
                for item in observation.dropped
            ],
            "objects": [
                {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "attributes": list(obj.attributes),
                    "confidence": obj.confidence,
                    "bbox": [
                        obj.bbox.u_min,
                        obj.bbox.v_min,
                        obj.bbox.u_max,
                        obj.bbox.v_max,
                    ],
                    "graspable": obj.graspable,
                    "role": obj.role,
                    "estimated_width_mm": obj.estimated_width_mm,
                    "relation_hints": [list(item) for item in obj.relation_hints],
                    "quality": asdict(obj.quality),
                    "color_evidence": asdict(obj.color_evidence) if obj.color_evidence else None,
                    "metric_footprint": asdict(obj.metric_footprint) if obj.metric_footprint else None,
                }
                for obj in objects
            ],
        }
        flattened = (
            np.stack([obj.mask for obj in objects]).reshape(len(objects), -1)
            if objects
            else np.empty((0, height * width), dtype=bool)
        )
        packed_masks = np.packbits(flattened, axis=1)
        metadata_bytes = np.frombuffer(
            json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
            dtype=np.uint8,
        )

        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path(image)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                metadata=metadata_bytes,
                packed_masks=packed_masks,
            )
        os.replace(temporary, destination)
        return destination

    def observe(self, image: np.ndarray) -> tuple[DetectedObject, ...]:
        return self.read_observation(image).objects

    def observe_scene(self, scene: Scene) -> tuple[DetectedObject, ...]:
        return self.read_observation(scene.rgb, scene=scene).objects

    def read_observation(self, image: np.ndarray, *, scene: Scene | None = None) -> VisionObservation:
        path = self._path(image)
        if not path.is_file():
            raise FileNotFoundError(f"vision cache miss for {image_cache_key(image)}: {path}")

        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(payload["metadata"].tobytes().decode("utf-8"))
            if metadata.get("version") != CACHE_FORMAT_VERSION:
                raise CacheCompatibilityError(
                    f"incompatible vision cache version in {path}"
                )
            if metadata.get("pipeline_fingerprint") != self.pipeline_fingerprint:
                raise CacheCompatibilityError(
                    f"incompatible vision pipeline fingerprint in {path}"
                )
            if metadata.get("geometry_fingerprint") != scene_geometry_fingerprint(scene):
                raise CacheCompatibilityError(f"incompatible scene geometry in {path}; run cache-vision")
            height, width = (int(value) for value in metadata["image_shape"])
            if image.shape[:2] != (height, width):
                raise ValueError(f"cached image shape mismatch in {path}")
            records = metadata["objects"]
            packed_masks = payload["packed_masks"]
            masks = np.unpackbits(
                packed_masks,
                axis=1,
                count=height * width,
            ).reshape(len(records), height, width).astype(bool)

        objects = []
        for record, mask in zip(records, masks, strict=True):
            bbox = BBox(*(int(value) for value in record["bbox"]))
            if bbox.u_max > width or bbox.v_max > height:
                raise ValueError(f"cached bounding box extends outside the image in {path}")
            objects.append(
                DetectedObject(
                    object_id=str(record["object_id"]),
                    label=str(record["label"]),
                    attributes=tuple(str(item) for item in record["attributes"]),
                    confidence=float(record["confidence"]),
                    bbox=bbox,
                    mask=mask,
                    graspable=bool(record["graspable"]),
                    role=record.get("role"),
                    estimated_width_mm=(
                        float(record["estimated_width_mm"])
                        if record["estimated_width_mm"] is not None
                        else None
                    ),
                    relation_hints=tuple(
                        (str(predicate), str(object_id))
                        for predicate, object_id in record["relation_hints"]
                    ),
                    quality=VisionQuality(**{
                        **record["quality"],
                        "reasons": tuple(record["quality"]["reasons"]),
                        "conflicting_object_ids": tuple(record["quality"]["conflicting_object_ids"]),
                    }),
                    color_evidence=(VisualColorEvidence(**{**record["color_evidence"],
                        "colors": tuple(record["color_evidence"]["colors"])})
                        if record.get("color_evidence") else None),
                    metric_footprint=(MetricFootprint(**record["metric_footprint"])
                                      if record.get("metric_footprint") else None),
                )
            )
        dropped = tuple(
            DroppedDetection(
                object_id=str(item["object_id"]),
                label=str(item["label"]),
                reason=str(item["reason"]),
            )
            for item in metadata["dropped"]
        )
        return VisionObservation(
            inventory_count=int(metadata["inventory_count"]),
            objects=tuple(objects),
            dropped=dropped,
            recovery=tuple(PerceptionRecovery(**{**item, "trigger": tuple(item["trigger"])})
                           for item in metadata["recovery"]),
        )


class RecoveryCacheBackend:
    """A separately fingerprinted layer; its writes never touch base observations."""

    def __init__(self, root: Path, *, fingerprint: str):
        self._objects = VisionCacheBackend(Path(root) / "recovery", pipeline_fingerprint=fingerprint)
        self.fingerprint = fingerprint

    def _path(self, image: np.ndarray) -> Path:
        return self._objects._path(image)

    def store(self, image: np.ndarray, observation) -> Path:
        return self._objects._store(image, VisionObservation.complete(observation.objects),
            allow_empty=True, extra_metadata={
                "reference_recovery": [asdict(record) for record in observation.records],
                "absence_evidence": [asdict(item) for item in observation.absence_evidence],
            })

    def read(self, image: np.ndarray):
        from .perception_recovery import (RecoveryObservation, RecoveryRecord, ReferenceQuery,
                                          valid_absence_evidence)
        observation = self._objects.read_observation(image)
        with np.load(self._path(image), allow_pickle=False) as payload:
            metadata = json.loads(payload["metadata"].tobytes().decode("utf-8"))
        records = []
        for item in metadata["reference_recovery"]:
            query = item["query"]
            records.append(RecoveryRecord(**{**item, "query": ReferenceQuery(**{
                **query, "colors": tuple(query["colors"]), "roles": tuple(query["roles"])})}))
        evidence = tuple(AbsenceEvidence(**{**item, "colors": tuple(item["colors"]),
            "measured_colors": tuple(item["measured_colors"])}) for item in metadata["absence_evidence"])
        if any(not valid_absence_evidence(item) for item in evidence):
            raise ValueError("invalid recovery absence evidence")
        return RecoveryObservation(observation.objects, tuple(records), evidence)

    def probe(self, image: np.ndarray) -> CacheProbe:
        try:
            self.read(image)
            return CacheProbe("valid")
        except FileNotFoundError:
            return CacheProbe("missing")
        except CacheCompatibilityError as exc:
            return CacheProbe("incompatible", error=str(exc))
        except Exception as exc:
            return CacheProbe("corrupt", error=f"{type(exc).__name__}: {exc}")


class ComposedVisionCacheBackend:
    def __init__(self, base: VisionCacheBackend, recovery_by_image: dict[str, RecoveryCacheBackend]):
        self.base = base
        self.recovery_by_image = recovery_by_image

    def _recovery(self, image):
        try:
            return self.recovery_by_image[image_cache_key(image)].read(image)
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError("missing, incompatible, or corrupt recovery cache; run cache-vision") from exc

    def observe(self, image: np.ndarray) -> tuple[DetectedObject, ...]:
        from .perception_recovery import compose_recovered_detections
        return compose_recovered_detections(self.base.observe(image), self._recovery(image).objects)

    def observe_scene(self, scene: Scene) -> tuple[DetectedObject, ...]:
        from .perception_recovery import compose_recovered_detections
        return compose_recovered_detections(self.base.observe_scene(scene), self._recovery(scene.rgb).objects)

    def read_absence_evidence(self, image: np.ndarray) -> tuple[AbsenceEvidence, ...]:
        return self._recovery(image).absence_evidence
