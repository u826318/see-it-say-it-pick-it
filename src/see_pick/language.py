"""Scene-blind flat language boundary and deterministic source interpretation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from .canonicalization import COLORS, decompose_noun, normalize_noun
from .domain import (
    LiteralEvidence, LiteralSpan, ObjectReference, SourceConjunction, SourceTask,
    TaskIntent, TaskProgram,
)


Operation = Literal["pick", "place", "unsupported"]
Relation = Literal[
    "left_of", "right_of", "above", "below", "nearest_to", "farthest_from",
    "inside", "on_top_of", "under",
]
Direction = Literal["left", "right", "top", "bottom"]
Extremum = Literal["leftmost", "rightmost", "topmost", "bottommost", "nearest", "farthest"]
Quantity = Annotated[StrictInt, Field(ge=1)] | Literal["all"] | None


class FlatCommand(BaseModel):
    """One model proposal; only literal phrases and shallow structural slots."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    operation: Operation
    target_phrase: str | None = Field(default=None, min_length=1)
    destination_phrase: str | None = Field(default=None, min_length=1)
    relation: Relation | None = None
    relation_phrase: str | None = Field(default=None, min_length=1)
    quantity: Quantity = None
    ordinal: StrictInt | None = Field(default=None, ge=1)
    ordinal_from: Direction | None = None
    extremum: Extremum | None = None
    exclusion_phrases: tuple[str, ...] = ()
    placement_anchor_phrase: str | None = Field(default=None, min_length=1)
    context_phrases: tuple[str, ...] = ()
    second_clause: str | None = Field(default=None, min_length=1)


FLAT_COMMAND_SCHEMA = FlatCommand.model_json_schema()

# Closed vocabulary is shared with coverage validation, never with scene identity.
OPERATIONS = {"pick up": "pick", "pick": "pick", "lift": "pick", "grab": "pick",
              "put": "place", "place": "place", "rotate": "unsupported",
              "throw": "unsupported", "push": "unsupported", "drop": "unsupported",
              "move": "place", "pull": "unsupported", "take": "unsupported",
              "give": "unsupported"}
QUANTITIES = {**{word: n for n, word in enumerate(
    "one two three four five six seven eight nine ten".split(), 1)},
    "all": "all", "both": 2, "everything": "all"}
ORDINALS = {word: n for n, word in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth".split(), 1)}
ORDER_AXES = {"from the left": "left_to_right", "from the right": "right_to_left",
              "from the top": "top_to_bottom", "from the bottom": "bottom_to_top"}
EXTREMA = {
    **{word: word for word in ("leftmost", "rightmost", "topmost", "bottommost", "nearest", "farthest")},
    "on the left": "leftmost", "on the right": "rightmost",
    "at the top": "topmost", "at the bottom": "bottommost",
}
RELATIONS = {
    "left of": "left_of", "to the left of": "left_of",
    "right of": "right_of", "to the right of": "right_of",
    "above": "above", "below": "below",
    "nearest to": "nearest_to", "closest to": "nearest_to", "next to": "nearest_to",
    "farthest from": "farthest_from",
    "on top of": "on_top_of", "lying on": "on_top_of", "under": "under",
    "from": "inside",
}
DESTINATIONS = {word: True for word in ("in", "inside", "into", "on", "onto", "to")}
NEGATIONS = {word: True for word in ("do not", "don't", "never", "not", "no")}
CONJUNCTIONS = {word: word for word in ("and", "then", "or", "but")}
EXCLUSIONS = {word: True for word in ("except", "excluding", "without")}
NO_MOVEMENT = re.compile(r"\bwithout\s+moving\b", re.IGNORECASE)
STATE_PRESERVATION = re.compile(
    r"\band\s+leave\s+(?:it|them|this|that|these|those)\s+where\s+(?:it|they|this|that|these|those)\s+(?:is|are)\b",
    re.IGNORECASE,
)
CLEAR_TOP = "with nothing on it"
DEICTICS = frozenset({"it", "them", "this", "that", "these", "those", "one", "ones", "here", "there"})
GENERIC_NOUNS = {"item", "object", "thing", "part", "everything"}
FUNCTIONAL_TOKENS = DEICTICS | {"with", "to", "from"}
IGNORED_REFERENCE_TOKENS = frozenset({"the", "a", "an", "please"})
LEXICONS = {"operation": OPERATIONS, "color": {color: color for color in sorted(COLORS)},
            "quantity": QUANTITIES, "ordinal": ORDINALS, "order_axis": ORDER_AXES,
            "extremum": EXTREMA, "relation": RELATIONS, "destination": DESTINATIONS,
            "negation": NEGATIONS, "conjunction": CONJUNCTIONS, "exclusion": EXCLUSIONS,
            "clear_top": {CLEAR_TOP: True}}
# Shared head-role contract for consumers of literal references. Color words
# can also name objects (orange); only non-color marker tokens are functional.
NON_NOUN_HEAD_TOKENS = frozenset(
    word for kind, lexicon in LEXICONS.items() if kind != "color"
    for phrase in lexicon for word in phrase.split()
) | FUNCTIONAL_TOKENS | IGNORED_REFERENCE_TOKENS
_PHRASES = {phrase: (kind, value) for kind, lexicon in LEXICONS.items()
            for phrase, value in lexicon.items()}
_MARKERS = re.compile(r"\b(?:" + "|".join(
    r"\s+".join(map(re.escape, p.split())) for p in sorted(_PHRASES, key=len, reverse=True)
) + r"|\d+(?:st|nd|rd|th)?)\b", re.IGNORECASE)


def source_reference_mentions(program: TaskProgram) -> tuple[tuple[str, int, str], ...]:
    """Validated noun paths, global source positions and normalized heads."""
    return tuple(sorted(
        ((f"clauses.{index}." + evidence.path.removesuffix(".noun"), evidence.span.start,
          normalize_noun(evidence.span.text))
         for index, clause in enumerate(program.clauses) for evidence in clause.evidence
         if evidence.kind == "noun"), key=lambda mention: mention[1]))


def source_reference_head(mentions: tuple[tuple[str, int, str], ...], path: str) -> tuple[int, str]:
    """Resolve lexical noun inheritance only; never choose a scene object."""
    position, head = next((position, head) for name, position, head in mentions if name == path)
    if head == "one":
        if ".exclusions." in path and path.rsplit(".", 1)[1].isdigit():
            return position, source_reference_head(mentions, path.split(".exclusions.")[0] + ".target")[1]
        head = next((noun for _, offset, noun in reversed(mentions)
                     if offset < position and noun not in GENERIC_NOUNS | DEICTICS
                     and noun not in NON_NOUN_HEAD_TOKENS), "one")
    return position, head


def source_reference_qualified(reference: ObjectReference, *, exclusions: bool = False) -> bool:
    """Source-local qualifiers belong to the reference, not its path position."""
    parts = decompose_noun(reference.noun)
    other = "other" in (*parts.modifiers, *reference.attributes)
    descriptive = tuple(term for term in (*reference.attributes, *parts.modifiers)
                        if term not in FUNCTIONAL_TOKENS | IGNORED_REFERENCE_TOKENS)
    return any((reference.colors, parts.colors, descriptive,
                reference.ordinal, reference.extremum, reference.relation, reference.clear_top, exclusions, other))


def can_inherit_reference_head(reference: ObjectReference, head: str, *, exclusions: bool = False) -> bool:
    """Only qualified lexical one/ones can inherit a concrete noun head."""
    return (decompose_noun(reference.noun).head == "one" and head != "one"
            and source_reference_qualified(reference, exclusions=exclusions))


def source_references_bound(program: TaskProgram) -> bool:
    """Reject pointing references that noun inheritance cannot bind without a scene."""
    mentions = source_reference_mentions(program)
    for path, _, _ in mentions:
        _, index, *fields = path.split(".")
        intent = program.clauses[int(index)].intent
        reference = intent
        for field in fields:
            reference = reference[int(field)] if field.isdigit() else getattr(reference, field)
        position, head = source_reference_head(mentions, path)
        parts = decompose_noun(reference.noun)
        if "other" in (*parts.modifiers, *reference.attributes) and not any(
                offset < position and noun == head for _, offset, noun in mentions):
            return False
        if extract_closed_class(reference.noun).deictic_only or decompose_noun(reference.noun).head == "one":
            if not can_inherit_reference_head(reference, head,
                                              exclusions=fields == ["target"] and bool(intent.exclusions)):
                return False
    return True


@dataclass(frozen=True, slots=True)
class ClosedClassFact:
    kind: str
    value: str | int | bool
    span: LiteralSpan


@dataclass(frozen=True, slots=True)
class ClosedClassFacts:
    markers: tuple[ClosedClassFact, ...]
    operation: str | None
    colors: tuple[str, ...]
    quantity: int | str | None
    ordinal: int | None
    ordinal_from: str | None
    extremum: str | None
    relation: str | None
    negated: bool
    exclusions: tuple[str, ...]
    deictic_only: bool


def _placement_markers(text: str, markers: list[ClosedClassFact],
                       inherited_operation: str | None = None) -> list[ClosedClassFact]:
    # Prepositions describe the picked object for non-placement clauses.
    relation_by_preposition = {
        "in": "inside", "inside": "inside", "into": "inside",
        "on": "on_top_of", "onto": "on_top_of",
    }
    for index, marker in enumerate(markers):
        if marker.kind != "destination":
            continue
        previous = [m for m in markers[:index] if m.kind == "operation"]
        operation = previous[-1].value if previous else inherited_operation
        if operation != "place":
            literal = " ".join(marker.span.text.lower().split())
            if literal in relation_by_preposition:
                markers[index] = ClosedClassFact("relation", relation_by_preposition[literal], marker.span)

    # A placement with two locative prepositions qualifies the source set
    # before naming the final destination: "put the parts in the bin on the mat".
    place_destinations = [
        index for index, marker in enumerate(markers)
        if marker.kind == "destination"
    ]
    if len(place_destinations) == 2:
        first = place_destinations[0]
        operation = next(
            (marker.value for marker in reversed(markers[:first]) if marker.kind == "operation"),
            inherited_operation,
        )
        literal = " ".join(markers[first].span.text.lower().split())
        if operation == "place" and literal in relation_by_preposition:
            span = markers[first].span
            relative_link = re.search(r"\bthat\s+is\s+$", text[:span.start], re.IGNORECASE)
            if relative_link is not None:
                span = LiteralSpan(
                    start=relative_link.start(),
                    end=span.end,
                    text=text[relative_link.start():span.end],
                )
            markers[first] = ClosedClassFact(
                "relation", relation_by_preposition[literal], span
            )

    # In a place clause, a sole "on top of" introduces the destination.
    # A separate into/on marker leaves it as a reference qualifier instead.
    for index, marker in enumerate(markers):
        if marker.kind != "relation" or marker.value != "on_top_of":
            continue
        previous = [m for m in markers[:index] if m.kind == "operation"]
        operation = previous[-1].value if previous else inherited_operation
        if operation != "place":
            continue
        lower = max((m.span.end for m in markers[:index]
                     if m.kind == "operation"), default=0)
        upper = min((m.span.start for m in markers[index + 1:]
                     if m.kind == "operation"), default=len(text))
        constraint = NO_MOVEMENT.search(text, lower, upper)
        if constraint is not None:
            upper = constraint.start()
        if not any(m.kind == "destination" and lower <= m.span.start < upper for m in markers):
            markers[index] = ClosedClassFact("destination", True, marker.span)
    return markers


def extract_closed_class(text: str, *, inherited_operation: str | None = None) -> ClosedClassFacts:
    markers = []
    for match in _MARKERS.finditer(text):
        literal = match.group()
        normalized = " ".join(literal.lower().split())
        if normalized in _PHRASES:
            kind, value = _PHRASES[normalized]
        elif normalized.isdigit():
            kind, value = "quantity", int(normalized)
        else:
            kind, value = "ordinal", int(normalized[:-2])
        if normalized == "one":
            following = re.match(r"\s+([\w'-]+)", text[match.end():])
            if following is None or following[1].lower() in (
                    set(CONJUNCTIONS) | set(DESTINATIONS) | set(EXCLUSIONS)
                    | {phrase.split()[0] for phrase in RELATIONS}):
                continue  # Pronominal noun head, rather than a count before a noun.
        markers.append(ClosedClassFact(kind, value, LiteralSpan(
            start=match.start(), end=match.end(), text=literal)))
    markers = _placement_markers(text, markers, inherited_operation)

    def first(kind):
        return next((marker.value for marker in markers if marker.kind == kind), None)

    content = [word.group().lower() for word in re.finditer(r"\b[\w'-]+\b", text)
               if word.group().lower() not in IGNORED_REFERENCE_TOKENS
               and not any(m.span.start <= word.start() < m.span.end for m in markers)]
    axis = first("order_axis")
    return ClosedClassFacts(
        markers=tuple(markers), operation=first("operation"),
        colors=tuple(m.value for m in markers if m.kind == "color"), quantity=first("quantity"),
        ordinal=first("ordinal"), ordinal_from=axis.split("_to_")[0] if axis else None,
        extremum=first("extremum"), relation=first("relation"),
        negated=any(m.kind == "negation" for m in markers),
        exclusions=tuple(m.span.text for m in markers if m.kind == "exclusion"),
        deictic_only=bool(content) and all(word in DEICTICS for word in content),
    )


def parse_literal_command(text: str) -> FlatCommand | None:
    """Accept only one explicit simple pick/place with concrete literal phrases."""
    if len(text) > 256:
        return None
    match = re.fullmatch(
        r"\s*(?:please\s+)?(?P<op>pick\s+up|pick|lift|grab|place|put)\s+"
        r"(?P<body>[^.!?]+?)[.!?]?\s*", text, re.IGNORECASE)
    if match is None:
        return None
    facts = extract_closed_class(text)
    if any(m.kind not in {"operation", "color", "destination"} for m in facts.markers):
        return None
    if sum(m.kind == "operation" for m in facts.markers) != 1:
        return None
    destinations = [m.span for m in facts.markers if m.kind == "destination"]
    if len(destinations) != (1 if facts.operation == "place" else 0):
        return None

    def phrase(start, end):
        literal = re.sub(r"^(?:the|a|an)\s+", "", text[start:end].strip(), flags=re.IGNORECASE)
        if re.fullmatch(r"[\w'\s-]+", literal) is None:
            return None
        words = re.findall(r"[\w'-]+", literal)
        if not 1 <= len(words) <= 5 or any(w.lower() in FUNCTIONAL_TOKENS for w in words):
            return None
        return literal

    target = phrase(match.start("body"), destinations[0].start if destinations else match.end("body"))
    destination = phrase(destinations[0].end, match.end("body")) if destinations else None
    if target is None or (facts.operation == "place" and destination is None):
        return None
    return FlatCommand(operation=facts.operation, target_phrase=target, destination_phrase=destination)


def _reference_phrase(text: str) -> str | None:
    phrase = text.strip().rstrip(".!?").strip()
    phrase = re.sub(r"^(?:the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"^(?:all|both)\s+(?:the\s+)?", "", phrase, flags=re.IGNORECASE)
    return phrase if phrase and re.fullmatch(r"[\w'\s-]+", phrase) else None


def _exclusion_phrases(text: str) -> tuple[str, ...]:
    return tuple(
        phrase for part in re.split(r"\s+and\s+", text.strip().rstrip(".!?"), flags=re.IGNORECASE)
        if (phrase := _reference_phrase(part)) is not None
    )


def _unsupported_command(text: str) -> FlatCommand | None:
    unsafe_patterns = (
        r"(?P<op>throw|drop)\s+(?P<target>.+?)\s+(?P<context>into|in|onto|on)\s+(?P<landmark>.+)",
        r"(?P<op>throw|push|take|drop)\s+(?P<target>.+?)\s+(?P<context>off)\s+(?P<landmark>.+)",
        r"(?P<op>pull)\s+(?P<target>.+?)\s+(?P<context>out\s+from\s+under)\s+(?P<landmark>.+)",
        r"(?P<op>take)\s+(?P<target>.+?)\s+(?P<context>out\s+of)\s+(?P<landmark>.+)",
    )
    for pattern in unsafe_patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match is not None:
            target = _reference_phrase(match["target"])
            if target is None:
                return None
            context = text[match.start("context"):]
            return FlatCommand(operation="unsupported", target_phrase=target,
                               context_phrases=(context,))
    give = re.fullmatch(r"give\s+(?P<context>me|us|him|her|them)\s+(?P<target>.+)", text,
                        re.IGNORECASE)
    if give is not None:
        target = _reference_phrase(give["target"])
        return (FlatCommand(operation="unsupported", target_phrase=target,
                            context_phrases=(give["context"],)) if target else None)
    return None


def _transfer_command(main, operation, destinations, relations, exclusions, context):
    if not (operation.value == "place" and len(relations) == len(destinations) == 1
            and relations[0].value in {"inside", "on_top_of"}
            and relations[0].span.end <= destinations[0].span.start):
        return None
    target_text = main[operation.span.end:relations[0].span.start]
    target_text = re.sub(r"\s+that\s+is\s*$", "", target_text, flags=re.IGNORECASE)
    target = _reference_phrase(target_text)
    source_text = main[relations[0].span.end:destinations[0].span.start]
    if re.search(r"\b(?:and|then|or|but)\b", source_text, re.IGNORECASE):
        return None
    source = _reference_phrase(source_text)
    exclusion_start = exclusions[0].span.start if exclusions else len(main)
    destination = _reference_phrase(main[destinations[0].span.end:exclusion_start])
    excluded = _exclusion_phrases(main[exclusions[0].span.end:]) if exclusions else ()
    if not all((target, source, destination)):
        return None
    return FlatCommand(operation="place", target_phrase=target,
                       relation=relations[0].value, relation_phrase=source,
                       destination_phrase=destination, quantity="all",
                       exclusion_phrases=excluded, context_phrases=context)


def _under_placement_command(main, operation, destinations, relations, exclusions, context):
    if not (operation.value == "place" and len(relations) == len(destinations) == 1
            and relations[0].value == "under"
            and relations[0].span.end <= destinations[0].span.start and not exclusions):
        return None
    target = _reference_phrase(main[operation.span.end:relations[0].span.start])
    anchor = _reference_phrase(main[relations[0].span.end:destinations[0].span.start])
    destination = _reference_phrase(main[destinations[0].span.end:])
    if not all((target, anchor, destination)):
        return None
    return FlatCommand(operation="place", target_phrase=target,
                       placement_anchor_phrase=anchor,
                       destination_phrase=destination, context_phrases=context)


def _placement_command(main, operation, destinations, relations, exclusions, context):
    if operation.value != "place" or len(destinations) != 1 or relations:
        return None
    destination_marker = destinations[0]
    exclusion_before = bool(exclusions and exclusions[0].span.start < destination_marker.span.start)
    target_end = exclusions[0].span.start if exclusion_before else destination_marker.span.start
    target = _reference_phrase(main[operation.span.end:target_end])
    if exclusion_before:
        destination = _reference_phrase(main[destination_marker.span.end:])
        excluded = _exclusion_phrases(main[exclusions[0].span.end:destination_marker.span.start])
    else:
        exclusion_start = exclusions[0].span.start if exclusions else len(main)
        destination = _reference_phrase(main[destination_marker.span.end:exclusion_start])
        excluded = _exclusion_phrases(main[exclusions[0].span.end:]) if exclusions else ()
    if not target or not destination:
        return None
    return FlatCommand(operation="place", target_phrase=target,
                       destination_phrase=destination, exclusion_phrases=excluded,
                       context_phrases=context)


def _pick_command(main, operation, destinations, relations, exclusions, clear_top, selectors):
    if operation.value != "pick":
        return None
    if len(clear_top) == 1 and not relations + destinations + exclusions:
        target = _reference_phrase(main[operation.span.end:clear_top[0].span.start])
        return FlatCommand(operation="pick", target_phrase=target) if target else None
    if len(relations) == 1 and not destinations + exclusions + clear_top + selectors:
        target = _reference_phrase(main[operation.span.end:relations[0].span.start])
        landmark = _reference_phrase(main[relations[0].span.end:])
        if target and landmark:
            return FlatCommand(operation="pick", target_phrase=target,
                               relation=relations[0].value, relation_phrase=landmark)
    return None


def parse_structured_command(text: str) -> FlatCommand | None:
    """Recognize compositional grammar while leaving open phrases source-bound."""
    if len(text) > 256:
        return None
    stripped = text.strip().rstrip(".!?").strip()
    if command := _unsupported_command(stripped):
        return command

    state = STATE_PRESERVATION.search(stripped)
    main = stripped[:state.start()].rstrip() if state else stripped
    context = (state.group(),) if state else ()
    facts = extract_closed_class(main)
    markers = list(facts.markers)
    operations = [marker for marker in markers if marker.kind == "operation"]
    destinations = [marker for marker in markers if marker.kind == "destination"]
    relations = [marker for marker in markers if marker.kind == "relation"]
    exclusions = [marker for marker in markers if marker.kind == "exclusion"]
    clear_top = [marker for marker in markers if marker.kind == "clear_top"]
    if len(operations) != 1:
        return None
    operation = operations[0]
    selectors = [marker for marker in markers
                 if marker.kind in {"ordinal", "order_axis", "extremum", "quantity"}]
    return (_transfer_command(main, operation, destinations, relations, exclusions, context)
            or _under_placement_command(main, operation, destinations, relations, exclusions, context)
            or _placement_command(main, operation, destinations, relations, exclusions, context)
            or _pick_command(main, operation, destinations, relations, exclusions, clear_top, selectors))


def _source_span(text: str, start: int, end: int) -> LiteralSpan:
    return LiteralSpan(start=start, end=end, text=text[start:end])


def _bind_clause(text: str, start: int, end: int, flat: FlatCommand,
                 inherited: SourceTask | None, index: int) -> SourceTask:
    """Bind open phrases inside source-owned reference scopes and modifiers."""
    from .intent_codec import IntentDecodeError

    inherited_operation = inherited.intent.operation if inherited else None
    local_markers = list(extract_closed_class(
        text[start:end], inherited_operation=inherited_operation).markers)
    markers = tuple(ClosedClassFact(m.kind, m.value, _source_span(
        text, start + m.span.start, start + m.span.end))
        for m in local_markers)
    evidence = []
    constraint_markers = tuple(NO_MOVEMENT.finditer(text, start, end))
    if not constraint_markers and (state := STATE_PRESERVATION.search(text, start, end)) is not None:
        constraint_markers = (state,)
    main_end = constraint_markers[0].start() if constraint_markers else end
    constraints = []
    for position, marker in enumerate(constraint_markers):
        boundary = constraint_markers[position + 1].start() if position + 1 < len(constraint_markers) else end
        constraint_end = len(text[:boundary].rstrip().rstrip(".!?").rstrip())
        constraints.append(_source_span(text, marker.start(), constraint_end))
    for phrase in flat.context_phrases:
        matches = list(re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text[start:end]))
        if len(matches) != 1:
            raise IntentDecodeError("context phrase must bind exactly once in its source scope")
        span = _source_span(text, start + matches[0].start(), start + matches[0].end())
        if span not in constraints:
            constraints.append(span)

    def bind(kind, path, span):
        evidence.append(LiteralEvidence(kind=kind, path=path, span=span))

    def of_kind(kind, lo=start, hi=main_end):
        return [m for m in markers if m.kind == kind and lo <= m.span.start < m.span.end <= hi]

    operations = of_kind("operation")
    if len(operations) > 1:
        raise IntentDecodeError("unseparated operation literals")
    if operations:
        operation = operations[0].value
        bind("operation", "operation", operations[0].span)
        body_start = operations[0].span.end
    elif inherited is not None:
        operation, body_start = inherited.intent.operation, start
    else:
        raise IntentDecodeError("operation requires a source literal")

    destinations, relations, exclusions = of_kind("destination"), of_kind("relation"), of_kind("exclusion")
    clear_top_markers = of_kind("clear_top")

    def reference(phrase, lo, hi, path):
        if phrase is None:
            raise IntentDecodeError("reference requires a literal phrase")
        matches = list(re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text[lo:hi]))
        if len(matches) != 1:
            raise IntentDecodeError("reference phrase must bind exactly once in its source scope")
        noun_start, noun_end = lo + matches[0].start(), lo + matches[0].end()
        # Strip only closed leading selectors; open noun wording stays verbatim.
        leading = re.match(r"(?:the|a|an)\s+", text[noun_start:noun_end], re.IGNORECASE)
        if leading:
            noun_start += leading.end()
        while True:
            modifier = next((m for m in markers if m.span.start == noun_start
                             and m.kind in {"color", "quantity", "ordinal", "extremum"}
                             and m.span.end < noun_end), None)
            if modifier is None:
                break
            noun_start = modifier.span.end
            while noun_start < noun_end and text[noun_start].isspace():
                noun_start += 1
        noun = _source_span(text, noun_start, noun_end)
        bind("noun", path + ".noun", noun)
        colors = []
        # A color at the actual noun head (orange, amber) is an identity literal.
        words = list(re.finditer(r"\b[\w'-]+\b", noun.text))
        if not words:
            raise IntentDecodeError("reference phrase requires a noun literal")
        head = words[-1]
        head_start = noun.start + head.start()
        for marker in of_kind("color", lo, hi):
            if marker.span.start >= head_start:
                continue
            if marker.value not in colors:
                bind("color", f"{path}.colors.{len(colors)}", marker.span)
                colors.append(marker.value)
        qualifiers = {}
        for kind, field in (("ordinal", "ordinal"), ("order_axis", "order_axis"), ("extremum", "extremum")):
            values = of_kind(kind, lo, hi)
            if len(values) > 1:
                raise IntentDecodeError("multiple selectors in one reference")
            if values:
                qualifiers[field] = values[0].value
                bind(kind, path + "." + field, values[0].span)
        for marker in of_kind("conjunction", noun.start, noun.end):
            bind("conjunction", path + ".noun", marker.span)
        return ObjectReference(noun=noun.text, colors=tuple(colors), **qualifiers)

    if operation == "unsupported":
        target = reference(flat.target_phrase, body_start, main_end, "target")
        for position, span in enumerate(sorted(constraints, key=lambda item: item.start)):
            bind("attribute", f"user_constraints.{position}", span)
        intent = TaskIntent(operation=operation, target=target,
                            user_constraints=tuple(span.text for span in sorted(
                                constraints, key=lambda item: item.start)),
                            rationale="source-bound command")
        return SourceTask(intent=intent, source=_source_span(text, start, end), evidence=tuple(evidence),
                          inherited_operation_from=index - 1 if not operations and inherited is not None else None)

    if len(destinations) > 1 or len(relations) > 1 or len(clear_top_markers) > 1:
        raise IntentDecodeError("flat command allows one destination, relation and unary state")
    target_end = min([m.span.start for m in destinations + relations + exclusions + clear_top_markers]
                     + [main_end])
    target = reference(flat.target_phrase, body_start, target_end, "target")
    if clear_top_markers:
        marker = clear_top_markers[0]
        bind("clear_top", "target.clear_top", marker.span)
        target = target.model_copy(update={"clear_top": True})
    destination, placement_landmark = None, None
    if destinations:
        marker = destinations[0]
        bind("destination", "destination", marker.span)
        destination_end = min([m.span.start for m in relations + exclusions if m.span.start > marker.span.end] + [main_end])
        phrase = flat.destination_phrase
        if phrase is None and " ".join(marker.span.text.lower().split()) == "on top of":
            phrase = flat.relation_phrase or text[marker.span.end:destination_end].strip().rstrip(".!?")
            placement_landmark = flat.relation_phrase
        destination = reference(phrase, marker.span.end, destination_end, "destination")
    elif flat.destination_phrase is not None:
        raise IntentDecodeError("destination phrase has no source marker")
    placement_anchor = None
    if flat.placement_anchor_phrase is not None:
        if len(relations) != 1 or relations[0].value != "under" or not destinations:
            raise IntentDecodeError("placement anchor requires under before a destination")
        marker = relations[0]
        placement_anchor = reference(flat.placement_anchor_phrase, marker.span.end,
                                     destinations[0].span.start, "placement_anchor")
        bind("relation", "placement_relation", marker.span)
    elif relations:
        marker = relations[0]
        landmark_end = min([m.span.start for m in destinations + exclusions if m.span.start > marker.span.end] + [main_end])
        prefix = "destination" if destinations and destinations[0].span.start < marker.span.start else "target"
        landmark = reference(flat.relation_phrase, marker.span.end, landmark_end, prefix + ".relation_target")
        bind("relation", prefix + ".relation", marker.span)
        ref = target if prefix == "target" else destination
        ref = ref.model_copy(update={"relation": marker.value, "relation_target": landmark})
        if prefix == "target":
            target = ref
        else:
            destination = ref
    elif flat.relation_phrase is not None and flat.relation_phrase != placement_landmark:
        raise IntentDecodeError("relation phrase has no source marker")

    excluded = []
    phrases = tuple(phrase for phrase in flat.exclusion_phrases
                    if not any(phrase in constraint.text for constraint in constraints))
    if exclusions:
        if not phrases:
            # The marker owns the whole remaining reference when grouping is omitted.
            phrases = (text[exclusions[0].span.end:main_end].strip().rstrip(".!?"),)
        cursor = exclusions[0].span.end
        for position, phrase in enumerate(phrases):
            phrase_start = text.find(phrase, cursor, main_end)
            if phrase_start < 0:
                raise IntentDecodeError("exclusion phrase is not in its source scope")
            phrase_end = phrase_start + len(phrase)
            bind("exclusion", f"exclusions.{position}", exclusions[min(position, len(exclusions) - 1)].span)
            for marker in of_kind("conjunction", cursor, phrase_start):
                bind("conjunction", f"exclusions.{position}.noun", marker.span)
            excluded.append(reference(phrase, cursor, phrase_end, f"exclusions.{position}"))
            cursor = phrase_end
    elif phrases:
        raise IntentDecodeError("exclusion phrase has no source marker")

    for position, constraint in enumerate(sorted(constraints, key=lambda item: item.start)):
        bind("attribute", f"user_constraints.{position}", constraint)

    quantities = of_kind("quantity", body_start, target_end)
    if len(quantities) > 1:
        raise IntentDecodeError("multiple target quantities")
    if quantities:
        bind("quantity", "quantity", quantities[0].span)
    negation_end = body_start if operations else next(
        item.span.start for item in evidence if item.path == "target.noun")
    negations = of_kind("negation", start, negation_end)
    if negations:
        bind("negation", "negated", negations[0].span)
    ordered_constraints = tuple(span.text for span in sorted(constraints, key=lambda item: item.start))
    intent = TaskIntent(operation=operation, target=target, destination=destination,
                        placement_relation="under" if placement_anchor is not None else None,
                        placement_anchor=placement_anchor,
                        quantity=quantities[0].value if quantities else None,
                        negated=bool(negations), exclusions=tuple(excluded),
                        user_constraints=ordered_constraints, rationale="source-bound command")
    return SourceTask(intent=intent, source=_source_span(text, start, end), evidence=tuple(evidence),
                      inherited_operation_from=index - 1 if not operations and inherited is not None else None)


def build_task_program(text: str, parse_clause: Callable[[str], FlatCommand], *,
                       max_clauses: int = 8) -> TaskProgram:
    """Build a bounded program; code owns every source span and closed fact."""
    from .intent_codec import IntentDecodeError, validate_source_coverage

    if not 1 <= max_clauses <= 16:
        raise ValueError("max_clauses must be between 1 and 16")
    clauses, conjunctions = [], []
    start, end = 0, len(text)
    while True:
        local = text[start:end]
        flat = parse_literal_command(local) or parse_structured_command(local) or parse_clause(local)
        if not isinstance(flat, FlatCommand):
            raise IntentDecodeError("clause parser must return FlatCommand")
        first_end, next_start, link = end, None, None
        if flat.second_clause is not None:
            if len(clauses) + 1 >= max_clauses:
                raise IntentDecodeError("clause limit exceeded")
            suffix = local.rstrip().rstrip(".!?")
            second = flat.second_clause
            if not (local.rstrip().endswith(second) or suffix.endswith(second)):
                raise IntentDecodeError("second_clause must be an exact source suffix")
            next_start = start + local.rfind(second)
            separator = re.search(r"\b(and|then|or|but)\s+$", text[start:next_start], re.IGNORECASE)
            if separator is None:
                raise IntentDecodeError("second_clause must follow a source conjunction")
            link_start = start + separator.start(1)
            first_end = link_start
            while first_end > start and text[first_end - 1].isspace():
                first_end -= 1
            link = SourceConjunction(kind=separator[1].lower(), after_clause=len(clauses),
                                     span=_source_span(text, link_start, start + separator.end(1)))
        try:
            clause = _bind_clause(text, start, first_end, flat,
                                  clauses[-1] if clauses else None, len(clauses))
        except ValidationError as exc:
            raise IntentDecodeError("invalid source-bound clause structure") from exc
        clauses.append(clause)
        if next_start is None:
            break
        conjunctions.append(link)
        start = next_start
    program = TaskProgram(clauses=tuple(clauses), conjunctions=tuple(conjunctions))
    findings = validate_source_coverage(text, program)
    if findings:
        raise IntentDecodeError("task program has source coverage findings", findings)
    return program
