"""Evidence-based safety policy for resolved tabletop intents."""

from __future__ import annotations

import numpy as np

from .domain import (
    IntentResolution,
    Resolved,
    SafetyApproval,
    SafetyDenial,
    SafetyOutcome,
    SceneGraph,
    TaskIntent,
    TaskProgram,
    ResolvedProgram,
    effective_mask,
    MIN_LEGAL_TARGET_AREA,
    CanonicalObject,
    PhysicalEvidence,
    ObligationLedger,
)
from .request_policy import RequestPolicy
from .program_order import remaining_scene, validated_step_order
from .metric_geometry import fit_evidence, gripper_evidence
from .language import extract_closed_class
from .obligations import evaluate_physical_obligations, normalize_ledger, physical_step_closed


UNSAFE_DESTINATION_NOUNS = {"person", "hand"}


def _denial(code: str, reason: str) -> SafetyDenial:
    return SafetyDenial(code, reason)


def placement_fit_evidence(target: CanonicalObject, destination: CanonicalObject,
                           markers: tuple[str, ...]) -> PhysicalEvidence:
    """Insertion must be explicit in this clause's source-bound destination marker."""
    if len(markers) != 1 or " ".join(markers[0].lower().split()) not in {"in", "into"}:
        return PhysicalEvidence("unknown", "container_insertion_unknown",
                                "Placement does not establish insertion through an opening.")
    return fit_evidence(target, destination)


def assess_intent(
    intent: TaskIntent,
    resolution: IntentResolution,
    graph: SceneGraph,
    *,
    policy: RequestPolicy,
    placement_markers: tuple[str, ...] | None = None,
) -> SafetyOutcome:
    if policy.disposition != "allow":
        return _denial(policy.code, f"{policy.reason} [{'; '.join(policy.evidence)}]")
    if intent.operation == "unsupported":
        return _denial("unsupported_operation", "No supported manipulation intent.")

    if not isinstance(resolution.target, Resolved):
        return SafetyApproval((*policy.evidence, "no physical action until reference resolves"))

    target = graph.object_by_id(resolution.target.object_id)
    if target is None:
        return _denial("missing_evidence", "resolved target is not visible")
    if target.role not in {"graspable_object", "container"} or not target.graspable:
        return _denial("not_graspable", "target is not graspable")
    grasp = gripper_evidence(target, graph.gripper_max_opening_mm)
    if grasp.status == "impossible":
        return _denial(grasp.code, grasp.reason)
    def hazard(subject_id: str, predicate: str, object_id: str) -> str | None:
        if subject_id == target.object_id:
            if predicate == "in_hand_of":
                return "take_from_hand"
            if predicate == "supports":
                return "supporting_object"
        if predicate == "on_top_of" and object_id == target.object_id:
            return "supporting_object"
        return None

    for relation in graph.relations:
        code = hazard(relation.subject_id, relation.predicate, relation.object_id)
        if code and relation.source == "geometry":
            reference = f"{relation.subject_id}:{relation.predicate}:{relation.object_id}"
            return _denial(code, f"Verified manipulation hazard [geometry:{reference}]")

    if intent.operation != "place" or not isinstance(resolution.destination, Resolved):
        return SafetyApproval((*policy.evidence, "target is visible and graspable"))

    if target.object_id == resolution.destination.object_id:
        return _denial("invalid_destination", "object cannot be placed onto itself")
    destination = graph.object_by_id(resolution.destination.object_id)
    if destination is None:
        return _denial("missing_evidence", "resolved destination is not visible")
    if destination.role in {"hand", "human", "background"} or destination.canonical_noun in UNSAFE_DESTINATION_NOUNS:
        return _denial(
            "unsafe_destination",
            "cannot place an object on a person or hand",
        )
    if placement_markers is None:
        placement_markers = tuple(marker.span.text for marker in extract_closed_class(policy.instruction).markers
                                  if marker.kind == "destination")
    fit = placement_fit_evidence(target, destination, placement_markers)
    if fit.status == "impossible":
        return _denial(fit.code, fit.reason)
    return SafetyApproval((*policy.evidence, "target and destination are safe"))


def assess_program(program: TaskProgram, resolution: ResolvedProgram, graph: SceneGraph,
                   *, policy: RequestPolicy, step_order: tuple[int, ...] | None = None,
                   ledger: ObligationLedger | None = None) -> SafetyOutcome:
    """Check structure; standalone callers also receive completed physical duties."""
    try:
        order = validated_step_order(program, resolution, step_order)
    except ValueError as exc:
        return _denial("dependency_conflict", str(exc))
    ledger, completed = normalize_ledger(program, resolution, ledger, graph=graph)
    steps = tuple(resolution.steps[index] for index in order)
    target_ids = [step.object_id for step in steps]
    if len(target_ids) != len(set(target_ids)):
        return _denial("repeated_target", "Repeated manipulation requires updated scene evidence.")
    if any(step.destination_id in target_ids for step in steps if step.destination_id is not None):
        return _denial("moving_destination", "A moving destination requires updated scene evidence.")
    moved: set[str] = set()
    evidence = list(policy.evidence)
    for index in order:
        step = resolution.steps[index]
        remaining = remaining_scene(graph, moved)
        if step.operation not in {"pick", "place"}:
            return _denial("unsupported_operation", "No supported manipulation intent.")
        for role, object_id in (("target", step.object_id), ("destination", step.destination_id)):
            if object_id is None:
                continue
            obj = remaining.object_by_id(object_id)
            if (obj is None or (obj.quality is not None and not obj.quality.groundable)
                    or obj.mask.shape != graph.image_shape
                    or np.count_nonzero(effective_mask(obj.mask, obj.bbox)) < MIN_LEGAL_TARGET_AREA):
                return _denial(f"{role}_quality", f"{role.capitalize()} has unusable vision evidence.")
        if step.operation == "place":
            if step.object_id == step.destination_id:
                return _denial("invalid_destination", "object cannot be placed onto itself")
            destination = remaining.object_by_id(step.destination_id)
            if destination is None:
                return _denial("missing_evidence", "resolved destination is not visible")
            if (destination.role in {"hand", "human", "background"}
                    or destination.canonical_noun in UNSAFE_DESTINATION_NOUNS):
                return _denial("unsafe_destination", "cannot place an object on a person or hand")
        evidence.append(f"clause:{step.clause_index}:safe:{step.object_id}")
        if ledger is not None and physical_step_closed(ledger, index):
            moved.add(step.object_id)
    if not completed:
        ledger = evaluate_physical_obligations(ledger, program, resolution, graph,
                                               step_order=order)
    return SafetyApproval(tuple(evidence), ledger)
