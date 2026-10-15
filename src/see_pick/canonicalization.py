"""Decompose literal labels into heads, closed colors and open modifiers.

The head ontology contains only cog/gear, a lexical synonym for a toothed
wheel. Category relatives (mug/cup, block/cube, dish/plate, citrus varieties)
retain separate identities. English pre-nominal phrases assign the rightmost
token its head role before classifying preceding colors/modifiers; this is
intentionally not a general language parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .domain import CanonicalObject, DetectedObject, infer_scene_role
from .visual_evidence import trusted_measured_color


COLORS = frozenset(
    {
        "amber",
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "grey",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "violet",
        "yellow",
    }
)

# Head-only equivalence: never add descriptive phrases or category relatives.
HEAD_SYNONYMS = {"cog": "gear"}
_WORD_TOKENS = re.compile(r"[a-z0-9]+", re.IGNORECASE | re.ASCII)


@dataclass(frozen=True, slots=True)
class NounParts:
    head: str
    colors: tuple[str, ...]
    modifiers: tuple[str, ...]
    head_span: tuple[int, int] | None = None


def _words(value: str) -> tuple[str, ...]:
    return tuple(match.group().lower() for match in _WORD_TOKENS.finditer(value))


def _singular_head(word: str) -> str:
    if word.endswith(("ches", "shes", "xes", "zes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def decompose_noun(value: str) -> NounParts:
    tokens = tuple(_WORD_TOKENS.finditer(value))
    words = tuple(token.group().lower() for token in tokens)
    # A lexeme's role precedes color lookup: a color homograph can be the noun.
    head = _singular_head(words[-1]) if words else ""
    return NounParts(
        head=HEAD_SYNONYMS.get(head, head),
        colors=normalize_colors(words[:-1]),
        modifiers=tuple(dict.fromkeys(
            word for word in words[:-1] if word not in COLORS
        )),
        head_span=tokens[-1].span() if tokens else None,
    )


def normalize_noun(value: str) -> str:
    return decompose_noun(value).head


def normalize_colors(values: tuple[str, ...]) -> tuple[str, ...]:
    colors: list[str] = []
    for value in values:
        for word in _words(value):
            normalized = "gray" if word == "grey" else word
            if word in COLORS and normalized not in colors:
                colors.append(normalized)
    return tuple(colors)


def normalize_modifiers(values: tuple[str, ...]) -> tuple[str, ...]:
    """Open descriptive evidence is tokenized, never mapped to phrase aliases."""
    return tuple(dict.fromkeys(word for value in values for word in _words(value)
                               if word not in COLORS))


def _normalized_attributes(detection: DetectedObject) -> tuple[str, ...]:
    parts = decompose_noun(detection.label)
    attributes = list(parts.modifiers)
    for value in detection.attributes:
        for word in _words(value):
            if (word not in COLORS and normalize_noun(word) != parts.head
                    and word not in attributes):
                attributes.append(word)
    return tuple(attributes)


def _aliases(detection: DetectedObject, noun: str) -> tuple[str, ...]:
    words = _words(detection.label)
    raw_without_color = " ".join(
        word for index, word in enumerate(words)
        if index == len(words) - 1 or word not in COLORS
    )
    aliases: list[str] = []
    for value in (noun, raw_without_color):
        if value and value not in aliases:
            aliases.append(value)
    return tuple(aliases)


def canonicalize_detection(
    detection: DetectedObject, runtime_id: str
) -> CanonicalObject:
    parts = decompose_noun(detection.label)
    noun = parts.head
    colors = normalize_colors((*parts.colors, *detection.attributes))
    if not colors and trusted_measured_color(detection.color_evidence):
        measured = normalize_colors(detection.color_evidence.colors)
        if measured:
            colors = measured
    evidence = ("vision:graspable",) if detection.graspable else ("vision:not_graspable",)
    quality = detection.quality
    role = detection.role or infer_scene_role(detection.label)
    if quality is not None and not quality.groundable:
        evidence += tuple(f"vision:{reason}" for reason in quality.reasons)
    return CanonicalObject(
        object_id=runtime_id,
        canonical_noun=noun,
        aliases=_aliases(detection, noun),
        colors=colors,
        attributes=_normalized_attributes(detection),
        confidence=detection.confidence,
        bbox=detection.bbox,
        mask=detection.mask,
        role=role,
        graspable=(detection.graspable and role in {"graspable_object", "container"}
                   and (quality is None or quality.groundable)),
        graspability_evidence=evidence,
        estimated_width_mm=detection.estimated_width_mm,
        raw_label=detection.label,
        raw_attributes=detection.attributes,
        relation_hints=detection.relation_hints,
        quality=quality,
        color_evidence=detection.color_evidence,
        metric_footprint=detection.metric_footprint,
    )
