"""Bounded open-vocabulary recovery for nouns missing from the base inventory."""

from __future__ import annotations

import unicodedata
from typing import Protocol

import numpy as np

from .canonicalization import normalize_noun
from .domain import DetectedObject


GENERIC_NOUNS = {
    "everything",
    "item",
    "object",
    "thing",
    "this",
    "that",
    "these",
    "those",
    "one",
    "it",
}


class LocalizationFallback(Protocol):
    def detect(
        self, image: np.ndarray, query: str
    ) -> tuple[DetectedObject, ...]: ...


def validate_visual_query(query: str) -> str:
    if any(unicodedata.category(char).startswith("C") for char in query):
        raise ValueError("visual query contains control characters")
    cleaned = " ".join(query.split())
    if not cleaned:
        raise ValueError("visual query is empty")
    if len(cleaned) > 80:
        raise ValueError("visual query exceeds 80 characters")
    noun = normalize_noun(cleaned)
    if not noun or noun in GENERIC_NOUNS:
        raise ValueError("visual query must name a concrete object")
    return noun
