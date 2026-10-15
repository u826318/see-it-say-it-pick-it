"""Head identity and hard constraints precede soft descriptive evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Protocol

import cv2
import numpy as np

from .canonicalization import COLORS, decompose_noun, normalize_colors, normalize_modifiers, normalize_noun
from .domain import (
    Ambiguous, CandidateSetOutcome, CanonicalObject, EmptyCandidateSet, IntentResolution,
    Missing, ObjectReference, ResolutionFilterStage, ResolutionOutcome, Resolved,
    ResolvedSelection, SelectionOutcome,
    SceneGraph, TaskIntent, Unsupported, effective_mask, MIN_LEGAL_TARGET_AREA,
    TaskProgram, ResolvedProgram, ResolvedStep, ResolvedConstraint, ProgramDependency, VerifiedAbsence,
)
from .intent_codec import decode_program
from .resolution_evidence import hard_resolution_closed
from .language import (GENERIC_NOUNS, NON_NOUN_HEAD_TOKENS, can_inherit_reference_head, extract_closed_class,
                       source_reference_head, source_reference_mentions)


# Raster-level uncertainty shared by ordinal, unary extremum and distance ties.
GEOMETRY_TOLERANCE_PX = 2.0
_EXTREMUM_AXES = {
    "leftmost": "left_to_right", "rightmost": "right_to_left",
    "topmost": "top_to_bottom", "bottommost": "bottom_to_top",
}
# Constraint parsing has a deliberately narrow grammar; open phrases must not
# be reduced to their final noun and silently change the prohibited object.
_NO_MOVEMENT_MODIFIERS = frozenset({"small", "large", "cracked", "chipped", "matte", "glossy"})
# Parser-owned lexical roles plus this constraint grammar's movement verb.
_NO_MOVEMENT_NON_NOUNS = NON_NOUN_HEAD_TOKENS | {"moving"}
_NO_MOVEMENT_REFERENCE = re.compile(
    r"without\s+moving\s+(?:(?:the|a|an)\s+)?"
    r"(?P<phrase>[a-z]+(?:\s+[a-z]+)*)[.!?]?", re.I | re.ASCII)


class SemanticHeadFallback(Protocol):
    """Reserved provider seam. Injection does not enable unbenchmarked recovery."""

    def suggest_heads(self, head: str, observed_heads: tuple[str, ...]) -> tuple[str, ...]: ...


def _centroid(obj: CanonicalObject) -> tuple[float, float]:
    rows, cols = np.nonzero(effective_mask(obj.mask, obj.bbox))
    return float(cols.mean()), float(rows.mean())


def _mask_gap(obj: CanonicalObject, anchor: CanonicalObject) -> float:
    """Shortest image-space distance between two effective object masks."""
    candidate = effective_mask(obj.mask, obj.bbox)
    landmark = effective_mask(anchor.mask, anchor.bbox)
    distances = cv2.distanceTransform(
        (~landmark).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(distances[candidate].min())


def _usable_geometry(obj: CanonicalObject, graph: SceneGraph) -> bool:
    return ((obj.quality is None or obj.quality.groundable)
            and obj.mask.shape == graph.image_shape
            and np.count_nonzero(effective_mask(obj.mask, obj.bbox)) >= MIN_LEGAL_TARGET_AREA)


def _uses_geometry(reference: ObjectReference) -> bool:
    return (reference.relation is not None or reference.ordinal is not None
            or reference.extremum is not None or reference.clear_top)


def _ids(objects: list[CanonicalObject]) -> tuple[str, ...]:
    return tuple(sorted(obj.object_id for obj in objects))


def _stage(name: str, criterion: str, before: list[CanonicalObject],
           after: list[CanonicalObject], *, hard: bool = True,
           scores: tuple[tuple[str, float], ...] = ()) -> ResolutionFilterStage:
    return ResolutionFilterStage(name, criterion, hard, _ids(before), _ids(after),
                                 tuple(sorted(scores)))


def _scoped(stages: tuple[ResolutionFilterStage, ...], scope: str) -> tuple[ResolutionFilterStage, ...]:
    return tuple(replace(stage, scope=scope if stage.scope == "reference"
                         else f"{scope}.{stage.scope}") for stage in stages)


def _ambiguous(objects: list[CanonicalObject], stages: list[ResolutionFilterStage]) -> Ambiguous:
    ordered = sorted(objects, key=lambda obj: obj.object_id)
    return Ambiguous(
        _ids(ordered),
        tuple(f"{obj.object_id}: {' '.join((*obj.colors, *obj.attributes, obj.canonical_noun))}"
              for obj in ordered),
        tuple(stages),
    )


def _no_candidates(reason: str, stages: list[ResolutionFilterStage], *,
                   candidate_set: bool) -> Missing | EmptyCandidateSet:
    if candidate_set:
        return EmptyCandidateSet(tuple(stages))
    return Missing(reason, tuple(stages))


def _ordered_candidates(candidates: list[CanonicalObject], axis: str,
                        ordinal: int) -> tuple[list[CanonicalObject], tuple[tuple[str, float], ...]]:
    coordinate_index = 0 if axis in {"left_to_right", "right_to_left"} else 1
    sign = -1 if axis in {"right_to_left", "bottom_to_top"} else 1
    coordinates = {obj.object_id: _centroid(obj)[coordinate_index] for obj in candidates}
    scores = tuple((obj.object_id, coordinates[obj.object_id]) for obj in candidates)
    ordered = sorted(candidates, key=lambda obj: (sign * coordinates[obj.object_id], obj.object_id))
    if ordinal > len(ordered):
        return [], scores
    selected = coordinates[ordered[ordinal - 1].object_id]
    return [obj for obj in ordered
            if abs(coordinates[obj.object_id] - selected) <= GEOMETRY_TOLERANCE_PX], scores


def _inherit_head(reference: ObjectReference, head: str) -> ObjectReference:
    parts = decompose_noun(reference.noun)
    return reference.model_copy(update={
        "noun": head,
        "colors": tuple(dict.fromkeys((*reference.colors, *parts.colors))),
        "attributes": tuple(dict.fromkeys((*reference.attributes, *parts.modifiers))),
    })


@dataclass(frozen=True)
class _DiscourseContext:
    # Validated noun literals retain their global source positions and paths.
    mentions: tuple[tuple[str, int, str], ...]
    resolved: tuple[tuple[int, str, tuple[str, ...]], ...] = ()

    def head(self, path: str) -> tuple[int, str]:
        return source_reference_head(self.mentions, path)

    def bind(self, reference: ObjectReference, path: str, *, exclusions: bool = False
             ) -> tuple[ObjectReference, tuple[str, ...]]:
        parts = decompose_noun(reference.noun)
        position, head = self.head(path)
        other = "other" in (*parts.modifiers, *reference.attributes)
        if can_inherit_reference_head(reference, head, exclusions=exclusions):
            reference = _inherit_head(reference, head)
        if not other:
            return reference, ()
        reference = _inherit_head(reference, normalize_noun(reference.noun))
        reference = reference.model_copy(update={
            "attributes": tuple(term for term in reference.attributes if term != "other")})
        ids = tuple(sorted({object_id for offset, noun, bound in self.resolved
                            if offset < position and noun == normalize_noun(reference.noun)
                            for object_id in bound}))
        return reference, ids

    def remember(self, path: str, outcome: CandidateSetOutcome | SelectionOutcome) -> _DiscourseContext:
        bindings = []
        if hard_resolution_closed(outcome):
            ids = (outcome.object_id,) if isinstance(outcome, Resolved) else outcome.object_ids
            position, head = self.head(path)
            bindings.append((position, head, ids))
        # Nested landmarks and exclusions are resolved mentions too. Their
        # scoped terminal attribute stages identify the bound ID without reruns.
        for stage in outcome.filter_stages:
            if stage.stage != "attributes" or stage.scope == "reference" or len(stage.surviving_ids) != 1:
                continue
            scoped = tuple(item for item in outcome.filter_stages
                           if item.scope == stage.scope or item.scope.startswith(stage.scope + "."))
            if not hard_resolution_closed(stages=scoped):
                continue
            if (stage.scope.startswith("exclusions.") and stage.scope.removeprefix("exclusions.").isdigit()
                    and not any(item.stage == "exclusion" and item.criterion == stage.surviving_ids[0]
                                for item in outcome.filter_stages)):
                continue
            nested = (path.rsplit(".", 1)[0] + "." + stage.scope
                      if stage.scope.startswith("exclusions.") else path + "." + stage.scope)
            position, head = self.head(nested)
            bindings.append((position, head, stage.surviving_ids))
        return replace(self, resolved=(*self.resolved, *bindings))


def resolve_reference(
    reference: ObjectReference, graph: SceneGraph, *,
    exclusions: tuple[ObjectReference, ...] = (),
    semantic_fallback: SemanticHeadFallback | None = None,
    require_complete: bool = False,
    grasp_candidates_only: bool = True,
    candidate_set: bool = False,
) -> CandidateSetOutcome:
    """Resolve one reference with hard filters before soft descriptive evidence."""
    return _resolve_reference(reference, graph, exclusions=exclusions, semantic_fallback=semantic_fallback,
                              require_complete=require_complete, grasp_candidates_only=grasp_candidates_only,
                              candidate_set=candidate_set)


def _resolve_reference(
    reference: ObjectReference, graph: SceneGraph, *,
    exclusions: tuple[ObjectReference, ...] = (),
    semantic_fallback: SemanticHeadFallback | None = None,
    require_complete: bool = False,
    grasp_candidates_only: bool = True,
    candidate_set: bool = False,
    context: _DiscourseContext | None = None,
    path: str = "",
) -> CandidateSetOutcome:
    """Filter hard evidence first; rank lexical modifiers only within survivors.

    Semantic fallback is deliberately disabled pending separate benchmark evidence.
    Source literals remain unchanged; only explicit lexical references inherit a head.
    Complete-set mode distinguishes unavailable potential members from hard
    nonmatches before any geometry or soft ranking can discard them.
    """
    literal_noun = reference.noun
    other = "other" in (*decompose_noun(reference.noun).modifiers, *reference.attributes)
    excluded_ids: tuple[str, ...] = ()
    if context is not None:
        reference, excluded_ids = context.bind(reference, path, exclusions=bool(exclusions))
    if other and not excluded_ids:
        return Ambiguous((), ("other requires a preceding resolved object of the same head",))
    if extract_closed_class(reference.noun).deictic_only:
        stage = ResolutionFilterStage("deictic_reference", "unbound pointing reference", True, (), ())
        return Ambiguous((), ("pointing reference requires a visible object description",), (stage,))
    if reference.extremum in {"nearest", "farthest"}:
        stage = ResolutionFilterStage("extremum", reference.extremum + ": distance reference unspecified",
                                      True, (), ())
        return Ambiguous((), ("Which distance reference should be used?",), (stage,))
    candidates = sorted(graph.objects, key=lambda obj: obj.object_id)
    parts = decompose_noun(reference.noun)
    matched = [obj for obj in candidates if parts.head in GENERIC_NOUNS
               or normalize_noun(obj.canonical_noun) == parts.head]
    stages = [_stage("head", f"{literal_noun} -> {parts.head}", candidates, matched)]
    candidates = matched
    if not candidates:
        stages.append(_stage("semantic_fallback", "disabled: independent benchmark required",
                             candidates, candidates, hard=False))
        return _no_candidates(f"no visible head matches {reference.noun}", stages,
                              candidate_set=candidate_set)
    if grasp_candidates_only:
        matched = [obj for obj in candidates if obj.role == "graspable_object"
                   or (obj.role == "container" and obj.graspable)]
        stages.append(_stage("scene_role", "graspable_object or independently graspable container", candidates, matched))
        candidates = matched
        if not candidates:
            return _no_candidates(f"no graspable object matches {reference.noun}", stages,
                                  candidate_set=candidate_set)
    stages.append(_stage("semantic_fallback", "disabled: independent benchmark required",
                         candidates, candidates, hard=False))
    if not candidates:
        return _no_candidates(f"no visible head matches {reference.noun}", stages,
                              candidate_set=candidate_set)

    colors = set(normalize_colors((*reference.colors, *parts.colors, *reference.attributes)))
    invalid_colors = tuple(color for color in reference.colors if color not in COLORS)
    matched = ([] if invalid_colors else
               [obj for obj in candidates if colors <= set(normalize_colors(obj.colors))])
    stages.append(_stage("color", ",".join(sorted(colors)) if not invalid_colors
                         else f"unsupported color: {','.join(invalid_colors)}", candidates, matched))
    candidates = matched
    if not candidates:
        from .perception_recovery import valid_absence_evidence
        for evidence in graph.absence_evidence:
            if (evidence.noun == parts.head and set(evidence.colors) == colors
                    and valid_absence_evidence(evidence)):
                return VerifiedAbsence("base and targeted observations plus measured color contrast",
                    tuple(stages), evidence=(evidence.source, evidence.query_phrase,
                        evidence.contrast_object_id, evidence.method, ",".join(evidence.measured_colors)))
        return _no_candidates(f"no visible object satisfies colors for {reference.noun}", stages,
                              candidate_set=candidate_set)

    if excluded_ids:
        matched = [obj for obj in candidates if obj.object_id not in excluded_ids]
        stages.append(_stage("exclusion", ",".join(excluded_ids), candidates, matched))
        candidates = matched

    for index, exclusion in enumerate(exclusions):
        if can_inherit_reference_head(exclusion, parts.head):
            exclusion = _inherit_head(exclusion, parts.head)
        excluded = _resolve_reference(exclusion, graph,
            require_complete=require_complete and _uses_geometry(exclusion), context=context,
            path=path.rsplit(".", 1)[0] + f".exclusions.{index}" if context is not None else "")
        stages.extend(_scoped(excluded.filter_stages, f"exclusions.{index}"))
        if not isinstance(excluded, Resolved):
            stages.append(_stage("exclusion", "unresolved exclusion", candidates, []))
            return replace(excluded, filter_stages=tuple(stages))
        requested_modifiers = (*decompose_noun(exclusion.noun).modifiers,
                               *normalize_modifiers(exclusion.attributes))
        attribute_stage = next(stage for stage in excluded.filter_stages
                               if stage.stage == "attributes" and stage.scope == "reference")
        if requested_modifiers and not any(score > 0 for _, score in attribute_stage.scores):
            stages.append(_stage("exclusion", "no positive descriptive evidence", candidates, []))
            return Missing("cannot identify descriptive exclusion", tuple(stages))
        matched = [obj for obj in candidates if obj.object_id != excluded.object_id]
        stages.append(_stage("exclusion", excluded.object_id, candidates, matched))
        candidates = matched
    if not candidates:
        return _no_candidates("all candidates excluded", stages, candidate_set=candidate_set)

    if require_complete:
        available, indeterminate = [], []
        for obj in candidates:
            (available if _usable_geometry(obj, graph) else indeterminate).append(obj)
        stages.append(_stage("membership_evidence", "potential members after hard constraints and ID exclusions",
                             candidates, available))
        if indeterminate:
            reasons = "; ".join(f"{obj.object_id}:" + (
                ",".join(obj.quality.reasons) if obj.quality is not None and not obj.quality.groundable
                else "unusable_effective_mask") for obj in indeterminate)
            stages.append(_stage("membership_indeterminate", reasons, candidates, indeterminate))
            return Missing("cannot establish complete selection: potential members have unavailable evidence",
                           tuple(stages))

    if _uses_geometry(reference):
        matched = [obj for obj in candidates if obj.quality is None or obj.quality.groundable]
        rejected = [f"{obj.object_id}:{','.join(obj.quality.reasons)}"
                    for obj in candidates if obj.quality is not None and not obj.quality.groundable]
        stages.append(_stage("vision_quality", "geometry requires usable evidence; " + "; ".join(rejected),
                             candidates, matched))
        candidates = matched
        if not candidates:
            return Missing("no usable vision evidence for geometry selection", tuple(stages))

    if reference.clear_top:
        covered = {
            relation.object_id if relation.predicate == "on_top_of" else relation.subject_id
            for relation in graph.relations
            if relation.source == "geometry" and relation.predicate in {"on_top_of", "supports"}
        }
        matched = [obj for obj in candidates if obj.object_id not in covered]
        stages.append(_stage("clear_top", "no verified object on candidate", candidates, matched))
        candidates = matched
        if not candidates:
            return _no_candidates("no visible object has a clear top", stages,
                                  candidate_set=candidate_set)

    if reference.relation is not None:
        assert reference.relation_target is not None
        target = _resolve_reference(reference.relation_target, graph, require_complete=require_complete,
                                    grasp_candidates_only=False, context=context, path=path + ".relation_target")
        stages.extend(_scoped(target.filter_stages, "relation_target"))
        if not isinstance(target, Resolved):
            stages.append(_stage("relation", "unresolved landmark", candidates, []))
            return replace(target, filter_stages=tuple(stages))
        anchor = graph.object_by_id(target.object_id)
        assert anchor is not None
        if ((anchor.quality is not None and not anchor.quality.groundable)
                or (require_complete and not _usable_geometry(anchor, graph))):
            criterion = f"{anchor.object_id}:" + (
                ",".join(anchor.quality.reasons) if anchor.quality is not None else "unusable_effective_mask")
            stages.append(_stage("landmark_quality", criterion, candidates, []))
            return Missing("landmark has unusable vision evidence", tuple(stages))
        scores: tuple[tuple[str, float], ...] = ()
        if reference.relation in {"nearest_to", "farthest_from"}:
            scores = tuple((obj.object_id, _mask_gap(obj, anchor))
                           for obj in candidates if obj.object_id != target.object_id)
            select = min if reference.relation == "nearest_to" else max
            selected = select((score for _, score in scores), default=0.0)
            related_ids = {object_id for object_id, score in scores
                           if abs(score - selected) <= GEOMETRY_TOLERANCE_PX}
        elif reference.relation in {"on_top_of", "under"}:
            supports = {(r.subject_id, r.object_id) if r.predicate == "supports"
                        else (r.object_id, r.subject_id)
                        for r in graph.relations if r.source == "geometry"
                        and r.predicate in {"supports", "on_top_of"}}
            related_ids = ({upper for lower, upper in supports if lower == target.object_id}
                           if reference.relation == "on_top_of" else
                           {lower for lower, upper in supports if upper == target.object_id})
        else:
            related_ids = {relation.subject_id for relation in graph.relations
                           if relation.predicate == reference.relation
                           and relation.object_id == target.object_id}
        matched = [obj for obj in candidates if obj.object_id in related_ids]
        metric = ":mask_gap" if reference.relation in {"nearest_to", "farthest_from"} else ""
        stages.append(_stage("relation", f"{reference.relation}:{target.object_id}{metric}",
                             candidates, matched, scores=scores))
        candidates = matched
        if not candidates:
            return _no_candidates(f"no visible object satisfies {reference.relation}", stages,
                                  candidate_set=candidate_set)

    if reference.ordinal is not None or reference.extremum is not None:
        axis = reference.order_axis if reference.ordinal is not None else _EXTREMUM_AXES[reference.extremum]
        assert axis is not None
        ordinal = reference.ordinal or 1
        matched, scores = _ordered_candidates(candidates, axis, ordinal)
        selector = f"{reference.ordinal}:{axis}" if reference.ordinal is not None else reference.extremum
        stages.append(_stage("ordinal" if reference.ordinal is not None else "extremum",
                             f"{selector};mask_centroid;tolerance_px={GEOMETRY_TOLERANCE_PX}",
                             candidates, matched, scores=scores))
        if not matched:
            return _no_candidates(f"only {len(candidates)} candidates for ordinal {ordinal}", stages,
                                  candidate_set=candidate_set)
        candidates = matched

    modifiers = set((*parts.modifiers, *normalize_modifiers(reference.attributes)))
    scores = tuple((obj.object_id, float(len(modifiers & set(normalize_modifiers(
        (*obj.attributes, *decompose_noun(obj.raw_label).modifiers)))))) for obj in candidates)
    best = max(score for _, score in scores)
    preferred_ids = {object_id for object_id, score in scores if score == best}
    matched = [obj for obj in candidates if obj.object_id in preferred_ids]
    stages.append(_stage("attributes", ",".join(sorted(modifiers)), candidates, matched,
                         hard=False, scores=scores))
    if len(matched) == 1:
        return Resolved(matched[0].object_id, ("head_and_hard_constraints", "soft_modifier_evidence"),
                        tuple(stages))
    return _ambiguous(matched, stages)


def resolve_target(intent: TaskIntent, graph: SceneGraph) -> ResolutionOutcome:
    if intent.operation == "unsupported":
        return Unsupported(intent.rationale)
    if intent.negated:
        stage = _stage("negation", "negated operation", list(graph.objects), [])
        return Unsupported("negated operation is not executable", (stage,))
    assert intent.target is not None
    if intent.exclusions:
        return resolve_reference(intent.target, graph, exclusions=intent.exclusions)
    return resolve_reference(intent.target, graph)


def resolve_intent(intent: TaskIntent, graph: SceneGraph) -> IntentResolution:
    target = resolve_target(intent, graph)
    destination = resolve_reference(intent.destination, graph, grasp_candidates_only=False) if intent.destination is not None else None
    return IntentResolution(intent.operation, target, destination)


def resolve_selection(intent: TaskIntent, graph: SceneGraph) -> SelectionOutcome:
    """Resolve the complete requested set without guessing a subset."""
    return _resolve_selection(intent, graph)


def _resolve_selection(intent: TaskIntent, graph: SceneGraph, *,
                       context: _DiscourseContext | None = None, path: str = "") -> SelectionOutcome:
    """Resolve a complete set; never interpret unresolved exclusion/landmark as a set."""
    if intent.negated:
        if intent.operation == "unsupported" or intent.target is None:
            return Unsupported("negative clause has no resolvable target")
        outcome = _resolve_reference(intent.target, graph, exclusions=intent.exclusions,
                                    require_complete=intent.quantity is not None,
                                    candidate_set=True, context=context, path=path)
        if isinstance(outcome, EmptyCandidateSet):
            ids = ()
        elif isinstance(outcome, Resolved):
            ids = (outcome.object_id,)
        elif (isinstance(outcome, Ambiguous) and outcome.filter_stages
              and outcome.filter_stages[-1].stage == "attributes"
              and outcome.filter_stages[-1].scope == "reference"):
            ids = outcome.candidate_ids
        else:
            return outcome
        stage = ResolutionFilterStage("negative_set", "complete prohibited candidate set", True,
                                      ids, ids)
        return ResolvedSelection(ids, (*outcome.filter_stages, stage))
    elif intent.quantity is not None and intent.operation != "unsupported":
        outcome = _resolve_reference(intent.target, graph, exclusions=intent.exclusions, require_complete=True,
                                     context=context, path=path)
    elif intent.operation != "unsupported":
        outcome = _resolve_reference(intent.target, graph, exclusions=intent.exclusions, context=context, path=path)
    else:
        outcome = resolve_target(intent, graph)
    if isinstance(outcome, Resolved):
        ids = (outcome.object_id,)
    elif (isinstance(outcome, Ambiguous) and intent.quantity not in (None, 1)
          and outcome.filter_stages and outcome.filter_stages[-1].stage == "attributes"
          and outcome.filter_stages[-1].scope == "reference"
          and intent.target.ordinal is None and intent.target.extremum is None
          and intent.target.relation != "nearest_to"):
        ids = outcome.candidate_ids
    else:
        return outcome
    objects = [graph.object_by_id(object_id) for object_id in ids]
    usable = [obj for obj in objects if obj is not None and _usable_geometry(obj, graph)]
    stages = [*outcome.filter_stages, _stage("groundable", "usable effective mask for every selected member",
                                           objects, usable)]
    if len(usable) != len(objects):
        return Missing("selected set contains unusable vision evidence", tuple(stages))
    quantity = intent.quantity or 1
    stages.append(_stage("quantity", str(quantity), objects,
                         objects if quantity == "all" or quantity == len(ids) else []))
    if quantity != "all" and len(ids) != quantity:
        if len(ids) > quantity:
            return _ambiguous(objects, stages)
        return Missing(f"requested {quantity} objects but only {len(ids)} match", tuple(stages))
    return ResolvedSelection(ids, tuple(stages))


def _no_movement_reference(constraint: str) -> ObjectReference | Unsupported:
    """Consume a complete simple noun phrase with a concrete noun head."""
    match = _NO_MOVEMENT_REFERENCE.fullmatch(constraint.strip())
    if match is not None:
        phrase = match["phrase"]
        words = phrase.lower().split()
        head = words[-1]
        # Control words and unfinished qualifiers cannot fill the noun slot.
        # Color homographs retain the existing noun-head semantics (orange).
        noun_head = head not in _NO_MOVEMENT_NON_NOUNS | _NO_MOVEMENT_MODIFIERS
        if noun_head and all(word in COLORS or word in _NO_MOVEMENT_MODIFIERS for word in words[:-1]):
            return ObjectReference(noun="object" if phrase.lower() in {"anything", "everything"} else phrase)
    return Unsupported("The movement constraint needs a supported simple reference.", code="unsupported_constraint")


def resolve_program(program: TaskProgram, graph: SceneGraph, *, instruction: str) -> ResolvedProgram:
    """Bind all source-covered selections before deriving runtime dependencies."""
    program = decode_program(program.model_dump_json(), instruction)
    context = _DiscourseContext(source_reference_mentions(program))
    selections, destinations, anchors = [], [], []
    for index, clause in enumerate(program.clauses):
        prefix = f"clauses.{index}."
        selection = _resolve_selection(clause.intent, graph, context=context, path=prefix + "target")
        selections.append(selection)
        context = context.remember(prefix + "target", selection)
        scoped = [(name, outcomes) for name, outcomes in
                  (("destination", destinations), ("placement_anchor", anchors))
                  if getattr(clause.intent, name) is not None]
        for name, outcomes in sorted(scoped, key=lambda item: context.head(prefix + item[0])[0]):
            reference = getattr(clause.intent, name)
            outcome = _resolve_reference(reference, graph, grasp_candidates_only=name != "destination",
                                         context=context, path=prefix + name)
            outcomes.append(outcome)
            context = context.remember(prefix + name, outcome)
        if clause.intent.destination is None:
            destinations.append(None)
        if clause.intent.placement_anchor is None:
            anchors.append(None)
    selections, destinations = tuple(selections), tuple(destinations)
    primary_steps = tuple(ResolvedStep(index, clause.intent.operation, object_id,
                              destinations[index].object_id if isinstance(destinations[index], Resolved) else None)
                  for index, clause in enumerate(program.clauses)
                  if isinstance(selections[index], ResolvedSelection)
                  and not clause.intent.negated
                  and (clause.intent.operation == "pick" or isinstance(destinations[index], Resolved))
                  for object_id in selections[index].object_ids)
    preparation = []
    anchor_failure = None
    for index, clause in enumerate(program.clauses):
        if clause.intent.placement_anchor is None:
            continue
        anchor = anchors[index]
        destination = destinations[index]
        if not isinstance(anchor, Resolved):
            anchor_failure = anchor
        elif not isinstance(destination, Resolved):
            anchor_failure = destination
        else:
            preparation.append(ResolvedStep(index, "place", anchor.object_id, destination.object_id))
    preparation_steps = tuple(preparation)
    steps = (*preparation_steps, *primary_steps)
    dependencies = set()
    for before in range(len(preparation_steps)):
        for after in range(len(preparation_steps), len(steps)):
            if steps[before].clause_index == steps[after].clause_index:
                dependencies.add((before, after, "source:placement_anchor"))
    for link in program.conjunctions:
        if link.kind == "then":
            for before, first in enumerate(steps):
                for after, second in enumerate(steps):
                    if first.clause_index <= link.after_clause < second.clause_index:
                        dependencies.add((before, after, f"source:then:{link.span.start}"))
    for relation in graph.relations:
        if relation.source != "geometry" or relation.predicate not in {"supports", "on_top_of"}:
            continue
        upper, lower = ((relation.object_id, relation.subject_id) if relation.predicate == "supports"
                        else (relation.subject_id, relation.object_id))
        for before, first in enumerate(steps):
            for after, second in enumerate(steps):
                if first.object_id == upper and second.object_id == lower:
                    dependencies.add((before, after, f"geometry:{upper}:on_top_of:{lower}"))
    failure = anchor_failure
    if any(link.kind == "or" for link in program.conjunctions):
        failure = Ambiguous((), ("alternative operations require clarification",))
    negative_ids = {object_id for index, clause in enumerate(program.clauses) if clause.intent.negated
                    and isinstance(selections[index], ResolvedSelection)
                    and hard_resolution_closed(selections[index])
                    for object_id in selections[index].object_ids}
    positive_ids = {step.object_id for step in steps}
    constraints = []
    for index, clause in enumerate(program.clauses):
        for instance, constraint in enumerate(clause.intent.user_constraints):
            reference = _no_movement_reference(constraint)
            if isinstance(reference, Unsupported):
                constraints.append(ResolvedConstraint(index, instance, constraint, reference))
                failure = reference
                continue
            prohibited = resolve_selection(TaskIntent(operation="pick", target=reference, negated=True,
                                                     quantity="all", rationale=constraint), graph)
            constraints.append(ResolvedConstraint(index, instance, constraint, prohibited))
            if not isinstance(prohibited, ResolvedSelection):
                failure = prohibited
    if negative_ids & positive_ids:
        failure = Unsupported("The same object is both prohibited and requested.", code="contradictory_request")
    elif not steps and negative_ids:
        failure = Unsupported("The request does not authorize moving an object.")
    return ResolvedProgram(
        selections, destinations, steps,
        tuple(ProgramDependency(*edge) for edge in sorted(dependencies)), failure,
        preparation_steps=preparation_steps,
        constraints=tuple(constraints),
        anchors=tuple(anchors),
    )
