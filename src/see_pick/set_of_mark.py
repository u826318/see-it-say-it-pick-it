"""Numbered visual references used only to narrow valid object candidates."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol, TypeAlias

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError

from .domain import Ambiguous, CanonicalObject, ResolutionOutcome, Resolved


MarkMap: TypeAlias = dict[int, str]
GenerateMarkers: TypeAlias = Callable[[np.ndarray, str], str]


class MarkerSelector(Protocol):
    def select(
        self,
        image: np.ndarray,
        instruction: str,
        mark_map: MarkMap,
    ) -> tuple[int, ...]: ...


class _MarkerSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    marker_ids: tuple[StrictInt, ...]


def marker_json_schema() -> dict[str, object]:
    return _MarkerSelection.model_json_schema()


def parse_marker_ids(raw: str) -> tuple[int, ...]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("response did not contain JSON")
    try:
        return _MarkerSelection.model_validate_json(match.group(0)).marker_ids
    except ValidationError as exc:
        raise ValueError(f"invalid marker selection: {exc}") from exc


def render_set_of_mark(
    image: np.ndarray,
    objects: tuple[CanonicalObject, ...],
) -> tuple[np.ndarray, MarkMap]:
    overlay = image.copy()
    mark_map: MarkMap = {}
    height, width = image.shape[:2]
    for marker_id, obj in enumerate(
        sorted(objects, key=lambda item: item.object_id), start=1
    ):
        mark_map[marker_id] = obj.object_id
        u = min(width - 1, max(0, (obj.bbox.u_min + obj.bbox.u_max) // 2))
        v = min(height - 1, max(0, (obj.bbox.v_min + obj.bbox.v_max) // 2))
        cv2.circle(overlay, (u, v), 10, (255, 255, 255), -1)
        cv2.circle(overlay, (u, v), 10, (0, 0, 0), 2)
        text_origin = (max(0, u - 5), min(height - 1, max(10, v + 5)))
        cv2.putText(
            overlay,
            str(marker_id),
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay, mark_map


def constrain_with_markers(
    outcome: ResolutionOutcome,
    marker_ids: tuple[int, ...],
    mark_map: MarkMap,
) -> ResolutionOutcome:
    if not isinstance(outcome, Ambiguous):
        return outcome
    allowed = set(outcome.candidate_ids)
    selected = sorted(
        {
            object_id
            for marker_id in marker_ids
            if (object_id := mark_map.get(marker_id)) in allowed
        }
    )
    if len(selected) == 1:
        return Resolved(selected[0], ("set_of_mark",))
    if len(selected) > 1:
        distinctions = tuple(
            distinction
            for candidate_id, distinction in zip(
                outcome.candidate_ids, outcome.distinctions, strict=True
            )
            if candidate_id in selected
        )
        return Ambiguous(tuple(selected), distinctions)
    return outcome


class JsonMarkerSelector:
    def __init__(self, generate: GenerateMarkers) -> None:
        self._generate = generate

    def select(
        self,
        image: np.ndarray,
        instruction: str,
        mark_map: MarkMap,
    ) -> tuple[int, ...]:
        prompt = (
            f"Instruction: {instruction}\n"
            f"Visible marker numbers: {json.dumps(sorted(mark_map))}\n"
            'Return JSON only as {"marker_ids":[int,...]}.'
        )
        feedback = ""
        for _ in range(2):
            raw = self._generate(image, prompt + feedback)
            try:
                return parse_marker_ids(raw)
            except ValueError as exc:
                feedback = f"\nPrevious response was invalid: {exc}"
        raise ValueError("marker selection remained invalid")
