"""Evidence-typed duties built from source bindings and deterministic scene evidence."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
import json
import math

import numpy as np

from .domain import (
    Ambiguous, EvaluatedObligationLedger, EvaluationContext, Missing, Obligation, ObligationEvidence, ObligationKey, ObligationLedger, ObligationStatus,
    ResolutionOutcome, Resolved, ResolvedConstraint, ResolvedProgram, ResolvedSelection, SelectionOutcome,
    RelationStatus, SceneGraph, TaskProgram, Unsupported, VerifiedAbsence,
)
from .metric_geometry import fit_evidence, gripper_evidence, pixel_fit_evidence, usable_effective_mask
from .program_order import remaining_scene, validated_step_order
from .relation_evidence import verify_relation
from .resolution_evidence import hard_resolution_closed, soft_narrowing_stage


def _explicit_insertion(program: TaskProgram, clause_index: int) -> bool:
    return any(item.kind == "destination" and item.path == "destination"
               and item.span.text.strip().lower() in {"in", "into"}
               for item in program.clauses[clause_index].evidence)


def resolution_obligation(key: ObligationKey, outcome: SelectionOutcome | ResolutionOutcome | None) -> Obligation:
    """Shared identity semantics for targets, destinations, anchors and constraints."""
    if isinstance(outcome, VerifiedAbsence):
        evidence = ObligationEvidence("perception", "verified_absence", "; ".join(outcome.evidence))
        return Obligation(key, status="violated", evidence=(evidence,))
    if isinstance(outcome, (ResolvedSelection, Resolved)):
        ids = outcome.object_ids if isinstance(outcome, ResolvedSelection) else (outcome.object_id,)
        if not ids:
            evidence = ObligationEvidence("resolver", "missing_evidence", "No object identity was resolved.")
            return Obligation(key, status="unknown", evidence=(evidence,))
        soft = soft_narrowing_stage(outcome.filter_stages)
        if soft is not None:
            evidence = ObligationEvidence("resolver", "soft_selection_evidence", soft.stage)
            # Dependency IDs describe a landmark/exclusion, not this selection.
            candidates = soft.input_ids if soft.scope == "reference" else ids
            return Obligation(key, status="open", candidates=candidates, evidence=(evidence,),
                              question=f"Which object do you mean: {'; '.join(candidates)}?")
        evidence = ObligationEvidence("resolver", "selection_resolved", ", ".join(ids))
        return Obligation(key, status="closed", evidence=(evidence,), candidates=ids)
    if isinstance(outcome, Ambiguous):
        code = ("deictic_reference" if any(stage.stage == "deictic_reference" for stage in outcome.filter_stages)
                else "selection_ambiguous")
        evidence = ObligationEvidence("resolver", code,
                                       "; ".join(outcome.distinctions) or "selection requires clarification")
        return Obligation(key, status="open", evidence=(evidence,), candidates=outcome.candidate_ids,
                          question=f"Which object do you mean: {'; '.join(outcome.distinctions)}?")
    if isinstance(outcome, (Missing, Unsupported)):
        evidence = ObligationEvidence("resolver", outcome.code if isinstance(outcome, Unsupported)
                                       else "missing_evidence", outcome.reason)
        return Obligation(key, status="unknown", evidence=(evidence,))
    evidence = ObligationEvidence("resolver", "missing_evidence", "Reference resolution is unavailable.")
    return Obligation(key, status="unknown", evidence=(evidence,))


def build_ledger(program: TaskProgram, resolution: ResolvedProgram) -> ObligationLedger:
    """Record duties without parsing language, inspecting pixels, or deciding actions."""
    obligations = []
    for index, clause in enumerate(program.clauses):
        outcome = resolution.selections[index] if index < len(resolution.selections) else None
        if not clause.intent.negated or not hard_resolution_closed(outcome):
            obligations.append(resolution_obligation(ObligationKey("selection_unique", index), outcome))
        if clause.intent.operation == "place":
            outcome = resolution.destinations[index] if index < len(resolution.destinations) else None
            obligations.append(resolution_obligation(ObligationKey("destination_unique", index), outcome))
        if clause.intent.placement_anchor is not None:
            outcome = resolution.anchors[index] if index < len(resolution.anchors) else None
            obligations.append(resolution_obligation(ObligationKey("anchor_unique", index), outcome))
        for instance, constraint in enumerate(clause.intent.user_constraints):
            obligations.append(Obligation(ObligationKey("user_constraint_discharged", index, None, instance),
                                          subject=constraint, status="unknown"))

    for index, step in enumerate(resolution.steps):
        obligations.append(Obligation(ObligationKey("grasp_feasible", step.clause_index, index),
                                      subject=step.object_id, status="unknown"))
        if step.operation == "place" and _explicit_insertion(program, step.clause_index):
            obligations.append(Obligation(ObligationKey("container_fits", step.clause_index, index),
                                          subject=step.object_id, status="unknown"))
        for instance, _ in enumerate(program.clauses[step.clause_index].intent.user_constraints):
            obligations.append(Obligation(ObligationKey("no_collateral_motion", step.clause_index, index, instance),
                                          subject=step.object_id, status="unknown"))
    return ObligationLedger(tuple(obligations))


def _canonical_evidence(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": value.shape,
                "sha256": sha256(value.tobytes(order="C")).hexdigest()}
    if isinstance(value, np.generic):
        return _canonical_evidence(value.item())
    if is_dataclass(value):
        return {"type": type(value).__name__, "fields": {
            item.name: _canonical_evidence(getattr(value, item.name)) for item in fields(value)}}
    if isinstance(value, dict):
        return {key: _canonical_evidence(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_canonical_evidence(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_float": repr(value)}
    return value


def _fingerprint(value) -> str:
    encoded = json.dumps(_canonical_evidence(value), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


def _binding_signature(program: TaskProgram, resolution: ResolvedProgram) -> str:
    return _fingerprint((program.model_dump(mode="json"), resolution))


def evaluation_context(program: TaskProgram, resolution: ResolvedProgram, graph: SceneGraph) -> EvaluationContext:
    """Pure content snapshot, including mask bytes rather than ndarray identity."""
    return EvaluationContext(_binding_signature(program, resolution), _fingerprint(graph))


def normalize_ledger(program: TaskProgram, resolution: ResolvedProgram,
                     ledger: ObligationLedger | None, *, graph: SceneGraph | None = None) -> tuple[ObligationLedger, bool]:
    """Validate evaluated capability and required shape; raw evidence is untrusted."""
    required = build_ledger(program, resolution)
    if (not isinstance(ledger, EvaluatedObligationLedger)
            or ledger.context.bindings_sha256 != _binding_signature(program, resolution)
            or (graph is not None and ledger.context.scene_sha256 != _fingerprint(graph))):
        return required, False
    supplied = {item.key: item for item in ledger.obligations}
    normalized = tuple(supplied[item.key] if item.key in supplied
                       and supplied[item.key].subject == item.subject
                       and supplied[item.key].unobtainable_default == item.unobtainable_default
                       else item for item in required.obligations)
    completed = (set(supplied) == {item.key for item in required.obligations}
                 and all(item.key in supplied and supplied[item.key].subject == item.subject
                         and supplied[item.key].unobtainable_default == item.unobtainable_default
                         for item in required.obligations))
    return (ledger, True) if completed else (ObligationLedger(normalized), False)


def decisive_evidence(obligation: Obligation) -> ObligationEvidence | None:
    """Use an aggregate's selected proof; single-source duties use their first proof."""
    return obligation.decisive or next(iter(obligation.evidence), None)


def physical_step_closed(ledger: ObligationLedger, step_index: int) -> bool:
    duties = tuple(item for item in ledger.obligations if item.key.step_index == step_index
                   and item.key.kind in {"grasp_feasible", "container_fits", "no_collateral_motion"})
    return bool(duties) and all(item.status == "closed" for item in duties)


def _evaluate_grasp(obligation: Obligation, graph: SceneGraph) -> Obligation:
    target = graph.object_by_id(obligation.subject)
    if target is None:
        return replace(obligation, status="unknown", evidence=(ObligationEvidence(
            "perception", "grasp_target_missing", "Resolved grasp target is unavailable."),), decisive=None)
    metric = gripper_evidence(target, graph.gripper_max_opening_mm)
    evidence = (ObligationEvidence("metric", metric.code, metric.reason),)
    if metric.status == "impossible":
        return replace(obligation, status="violated", evidence=evidence, decisive=evidence[0])
    if target.role not in {"graspable_object", "container"} or not target.graspable:
        proof = ObligationEvidence("perception", "not_graspable", f"role={target.role}; graspable={target.graspable}")
        return replace(obligation, status="violated", evidence=(*evidence, proof), decisive=proof)
    for relation in graph.relations:
        if relation.source != "geometry":
            continue
        incident_hazard = (
            relation.subject_id == target.object_id and relation.predicate in {"supports", "in_hand_of"}
            or relation.object_id == target.object_id and relation.predicate == "on_top_of")
        if incident_hazard:
            proof = ObligationEvidence("geometry", "verified_manipulation_hazard",
                f"{relation.subject_id}:{relation.predicate}:{relation.object_id}")
            return replace(obligation, status="violated", evidence=(*evidence, proof), decisive=proof)
    if usable_effective_mask(target, graph.image_shape) is None:
        proof = ObligationEvidence("perception", "grasp_mask_unusable", "Current mask/quality cannot support grasp closure.")
        return replace(obligation, status="unknown", evidence=(*evidence, proof), decisive=proof)
    if metric.status == "not_proven_impossible":
        return replace(obligation, status="closed", evidence=evidence, decisive=evidence[0])
    proxy = ObligationEvidence("perception", "grasp_proxy_usable",
        f"role={target.role}; graspable=True; effective mask/quality usable; no verified manipulation hazard.")
    return replace(obligation, status="closed", evidence=(*evidence, proxy), decisive=proxy)


def _constraint_binding(obligation: Obligation, resolution: ResolvedProgram) -> ResolvedConstraint | None:
    return next((item for item in resolution.constraints
                 if (item.clause_index, item.instance_index) ==
                 (obligation.key.clause_index, obligation.key.instance_index)), None)


def _constraint_evidence(binding: ResolvedConstraint | None, *, target: str | None = None,
                         prohibited: str | None = None, status: str, reason: str) -> ObligationEvidence:
    return ObligationEvidence("geometry" if prohibited is not None else "resolver",
        "collateral_identity_conflict" if target is not None and target == prohibited else
        "collateral_support_evidence" if prohibited is not None else "constraint_resolution_evidence",
        json.dumps({"constraint": binding.source if binding else None,
                    "clause_index": binding.clause_index if binding else None,
                    "instance_index": binding.instance_index if binding else None,
                    "target": target, "prohibited": prohibited, "predicate": "supports",
                    "status": status, "reason": reason}, sort_keys=True))


def _constraint_resolution(obligation: Obligation, binding: ResolvedConstraint | None) -> Obligation:
    identity = resolution_obligation(obligation.key, binding.outcome if binding is not None else None)
    return replace(identity, subject=obligation.subject)


def _combined_status(statuses: tuple[ObligationStatus, ...]) -> ObligationStatus:
    if "violated" in statuses:
        return "violated"
    if not statuses or "unknown" in statuses:
        return "unknown"
    return "open" if "open" in statuses else "closed"


@dataclass(frozen=True, slots=True)
class _SupportEvidenceGraph:
    verified: dict[str, set[str]]
    possible: dict[str, set[str]]
    incomplete_object_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.incomplete_object_ids


def _pairwise_support_graph(graph: SceneGraph) -> _SupportEvidenceGraph:
    """Evaluate every usable pair; absence of stored hints is never absence of support."""
    objects = sorted(graph.objects, key=lambda item: item.object_id)
    masks = {item.object_id: usable_effective_mask(item, graph.image_shape) for item in objects}
    incomplete = tuple(item.object_id for item in objects if masks[item.object_id] is None)
    usable = tuple(item for item in objects if masks[item.object_id] is not None)
    verified: dict[str, set[str]] = {}
    possible: dict[str, set[str]] = {}
    for lower in usable:
        for upper in usable:
            if lower.object_id == upper.object_id:
                continue
            overlap = (masks[lower.object_id] & masks[upper.object_id]).any()
            status = (RelationStatus.UNCERTAIN if overlap else
                      verify_relation(lower, "supports", upper).status)
            if status != RelationStatus.CONTRADICTED:
                possible.setdefault(lower.object_id, set()).add(upper.object_id)
            if status == RelationStatus.VERIFIED:
                verified.setdefault(lower.object_id, set()).add(upper.object_id)
    return _SupportEvidenceGraph(verified, possible, incomplete)


def _support_path(edges: dict[str, set[str]], lower: str, upper: str) -> tuple[str, ...]:
    """Deterministic directed reachability; cycles terminate."""
    pending, seen = deque([(lower,)]), {lower}
    while pending:
        path = pending.popleft()
        for object_id in sorted(edges.get(path[-1], ())):
            if object_id == upper:
                return (*path, object_id)
            if object_id not in seen:
                seen.add(object_id)
                pending.append((*path, object_id))
    return ()


def _evaluate_collateral(obligation: Obligation, binding: ResolvedConstraint | None,
                         graph: SceneGraph, support: _SupportEvidenceGraph, moved: set[str]) -> Obligation:
    identity = _constraint_resolution(obligation, binding)
    if identity.status != "closed":
        return identity
    statuses, evidence = [], []
    target = graph.object_by_id(obligation.subject)
    for object_id in binding.outcome.object_ids:
        prohibited = graph.object_by_id(object_id)
        relation_status, reason, status = "uncertain", "Target or prohibited object is unavailable.", "unknown"
        if obligation.subject == object_id:
            relation_status, reason, status = "verified", "The target is the prohibited object.", "violated"
        elif target is not None and object_id in moved:
            proof = ObligationEvidence("geometry", "already_moved", json.dumps({
                "constraint": binding.source, "target": obligation.subject, "prohibited": object_id,
                "status": "closed", "reason": "A completed earlier step removed this prohibited object."}, sort_keys=True))
            statuses.append("closed")
            evidence.append(proof)
            continue
        elif target is not None and prohibited is not None:
            first = usable_effective_mask(target, graph.image_shape)
            second = usable_effective_mask(prohibited, graph.image_shape)
            if first is None or second is None:
                reason = "Target or prohibited effective mask/quality is unusable."
            elif (first & second).any():
                reason = "Effective-mask overlap does not prove physical contact."
            elif path := _support_path(support.verified, target.object_id, object_id):
                relation_status, status = "verified", "violated"
                reason = "Verified support path: " + " -> ".join(path)
            elif path := _support_path(support.possible, target.object_id, object_id):
                reason = "Possible support path includes uncertain coupling: " + " -> ".join(path)
            elif not support.complete:
                reason = "Support graph incomplete: unusable effective masks or quality for " + ", ".join(
                    support.incomplete_object_ids)
            else:
                relation = verify_relation(target, "supports", prohibited)
                relation_status, reason = relation.status.value, "; ".join(relation.reasons)
                status = {RelationStatus.VERIFIED: "violated", RelationStatus.CONTRADICTED: "closed",
                          RelationStatus.UNCERTAIN: "unknown"}[relation.status]
        statuses.append(status)
        evidence.append(_constraint_evidence(binding, target=obligation.subject, prohibited=object_id,
                                             status=relation_status, reason=reason))
    status = _combined_status(tuple(statuses))
    proof = evidence[statuses.index(status)] if status in statuses else None
    return replace(obligation, status=status, evidence=tuple(evidence), decisive=proof)


def _evaluate_constraint(obligation: Obligation, resolution: ResolvedProgram,
                         ledger: ObligationLedger) -> Obligation:
    binding = _constraint_binding(obligation, resolution)
    identity = _constraint_resolution(obligation, binding)
    keys = tuple(ObligationKey("no_collateral_motion", step.clause_index, index, obligation.key.instance_index)
                 for index, step in enumerate(resolution.steps) if step.clause_index == obligation.key.clause_index)
    by_key = {item.key: item for item in ledger.obligations}
    missing = _constraint_evidence(binding, status="unknown", reason="A required collateral duty has no evaluated evidence.")
    duties = tuple(by_key[key] if key in by_key else Obligation(key, status="unknown", evidence=(missing,))
                   for key in keys) or (Obligation(obligation.key, status="unknown", evidence=(missing,)),)
    rows = (*duties, identity)
    status = _combined_status(tuple(item.status for item in rows))
    selected = next(item for item in rows if item.status == status)
    evidence = tuple(proof for item in rows for proof in item.evidence)
    return replace(obligation, status=status, evidence=evidence, decisive=decisive_evidence(selected),
                   candidates=selected.candidates, question=selected.question if status == "open" else None)


def evaluate_physical_obligations(ledger: ObligationLedger, program: TaskProgram,
                                  resolution: ResolvedProgram, graph: SceneGraph, *,
                                  step_order: tuple[int, ...] | None = None) -> EvaluatedObligationLedger:
    """Evaluate existing physical duties from resolved evidence; never decide actions."""
    step_order = validated_step_order(program, resolution, step_order)
    ledger, _ = normalize_ledger(program, resolution, ledger, graph=graph)
    updated = ledger
    moved: set[str] = set()
    for index in step_order:
        step = resolution.steps[index]
        remaining = remaining_scene(graph, moved)
        duties = tuple(item for item in ledger.obligations
                       if item.key.step_index == index and item.key.clause_index == step.clause_index
                       and item.subject == step.object_id)
        support = (_pairwise_support_graph(remaining) if any(item.key.kind == "no_collateral_motion"
                   for item in duties) else None)
        for obligation in duties:
            if obligation.key.kind == "grasp_feasible":
                evaluated = _evaluate_grasp(obligation, remaining)
            elif obligation.key.kind == "no_collateral_motion":
                evaluated = _evaluate_collateral(obligation, _constraint_binding(obligation, resolution), remaining, support, moved)
            elif obligation.key.kind == "container_fits":
                if step.operation != "place" or not _explicit_insertion(program, step.clause_index):
                    continue
                target = remaining.object_by_id(step.object_id)
                destination = remaining.object_by_id(step.destination_id)
                if target is None or destination is None:
                    evaluated = replace(obligation, status="unknown", evidence=(ObligationEvidence(
                        "perception", "container_fit_object_missing", "Target or destination is unavailable."),), decisive=None)
                else:
                    metric = fit_evidence(target, destination)
                    evidence = (ObligationEvidence("metric", metric.code, metric.reason),)
                    if metric.status == "impossible":
                        evaluated = replace(obligation, status="violated", evidence=evidence, decisive=evidence[0])
                    else:
                        pixel = pixel_fit_evidence(target, destination, image_shape=remaining.image_shape)
                        proof = ObligationEvidence("geometry", pixel.code, json.dumps(asdict(pixel), sort_keys=True))
                        evaluated = replace(obligation, status=pixel.status, evidence=(*evidence, proof), decisive=proof)
            else:
                continue
            updated = updated.replace(evaluated)
        if physical_step_closed(updated, index):
            moved.add(step.object_id)
    for obligation in ledger.obligations:
        if obligation.key.kind == "user_constraint_discharged":
            updated = updated.replace(_evaluate_constraint(obligation, resolution, updated))
    return EvaluatedObligationLedger(updated.obligations, context=evaluation_context(program, resolution, graph))
