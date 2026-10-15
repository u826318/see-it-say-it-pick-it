"""Atomic, explicit diagnostics for local vision generation attempts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GenerationRecord:
    prompt_version: str
    model_revision: str
    raw_response: str
    valid: bool
    violations: tuple[str, ...]
    image_shape: tuple[int, int]
    max_pixels: int | None
    elapsed_seconds: float
    gpu: Mapping[str, object] | None


class VisionDiagnosticStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def _validate_image_key(image_key: str) -> None:
        if re.fullmatch(r"[0-9a-f]{64}", image_key) is None:
            raise ValueError("image key must be a lowercase SHA-256 digest")

    @staticmethod
    def _write_atomic(path: Path, payload: object) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        return path

    def record_attempt(
        self, image_key: str, attempt: int, record: GenerationRecord
    ) -> Path:
        self._validate_image_key(image_key)
        if attempt < 1:
            raise ValueError("attempt must be at least 1")
        payload = {
            "prompt_version": record.prompt_version,
            "model_revision": record.model_revision,
            "raw_response": record.raw_response,
            "valid": record.valid,
            "violations": list(record.violations),
            "image_shape": list(record.image_shape),
            "max_pixels": record.max_pixels,
            "elapsed_seconds": record.elapsed_seconds,
            "gpu": dict(record.gpu) if record.gpu is not None else None,
        }
        return self._write_atomic(
            self.root / image_key / f"attempt-{attempt}.json", payload
        )

    def write_scene_manifest(self, scene_id: str, image_key: str) -> Path:
        self._validate_image_key(image_key)
        if not scene_id.strip():
            raise ValueError("scene_id must not be blank")
        destination = self.root / "scene-manifest.json"
        if destination.is_file():
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"invalid scene manifest in {destination}")
        else:
            payload = {}
        payload[scene_id] = image_key
        return self._write_atomic(destination, payload)
