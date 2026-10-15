"""Source programs flow through deterministic resolution, safety and compilation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from time import perf_counter

from pydantic import JsonValue

from . import resolver
from .compiler import compile_denial, compile_program_result, decision_reason_code
from .domain import (
    Instruction, PlanDecision, RelationEvidence, RelationStatus,
    ObligationKey, ResultEntry, Scene, SceneGraph, SafetyApproval, SafetyDenial, Violation,
)
from .grounding import GroundingError, resolve_first_pick
from .intent_codec import decode_program
from .planner import ProgramBackend
from .request_policy import derive_request_policy
from .relation_evidence import verify_relation
from .safety import assess_program, placement_fit_evidence
from .scene_graph import build_scene_graph
from .vision import VisionBackend
from .metric_geometry import gripper_evidence
from .obligations import build_ledger, evaluate_physical_obligations, physical_step_closed
from .program_order import remaining_scene, validated_step_order
from .request_policy import RequestPolicy


@dataclass(frozen=True, slots=True)
class PipelineTiming:
    instruction_id: str
    vision_ms: float
    planning_ms: float
    grounding_ms: float
    total_ms: float
    events: tuple[Violation, ...] = ()
    reason_code: str | None = None


def run_instruction(scene: Scene, instruction: Instruction, vision: VisionBackend,
                    intent_backend: ProgramBackend) -> ResultEntry:
    entries, _ = run_batch_with_timings(((scene, instruction),), vision, intent_backend)
    return entries[0]


def _run_with_graph_timed(instruction, graph, intent_backend):
    events = []
    started = perf_counter()
    stage = "parse"
    step_order = ()
    try:
        proposed = intent_backend.interpret_program(instruction.text)
        program = decode_program(proposed.model_dump_json(), instruction.text)
        events.append(Violation("program_parsed", f"clauses={len(program.clauses)}", stage="parse",
                                details=program.model_dump(mode="json")))
        stage = "target_resolution"
        resolution = resolver.resolve_program(program, graph, instruction=instruction.text)
        order_denial = None
        try:
            step_order = validated_step_order(program, resolution)
        except ValueError as exc:
            step_order = ()
            order_denial = SafetyDenial("dependency_conflict", str(exc))
        for index, selection in enumerate(resolution.selections):
            events.append(Violation("program_selection", type(selection).__name__, stage="target_resolution",
                details={"clause_index": index, "status": type(selection).__name__.lower(), **asdict(selection)}))
            destination = resolution.destinations[index]
            events.append(Violation("program_destination", type(destination).__name__, stage="destination_resolution",
                details={"clause_index": index, "status": type(destination).__name__.lower(),
                         **(asdict(destination) if destination is not None else {})}))
        stage = "relation_evidence"
        events.append(Violation("relation_evidence", "observed geometry evidence", stage=stage,
                                details={"relations": _relation_details(graph)}))
        events.append(Violation("program_dependencies", "source and verified geometry", stage=stage,
            details={"steps": [asdict(step) for step in resolution.steps],
                     "dependencies": [asdict(edge) for edge in resolution.dependencies]}))
        stage = "safety"
        ledger = build_ledger(program, resolution)
        if order_denial is None:
            ledger = evaluate_physical_obligations(ledger, program, resolution, graph, step_order=step_order)
        events.append(Violation("obligation_ledger", "physical duties unavailable after dependency conflict"
                                if order_denial is not None else "resolution and evaluated physical duties", stage=stage,
                                details=asdict(ledger)))
        moved: set[str] = set()
        for index in step_order:
            step = resolution.steps[index]
            remaining = remaining_scene(graph, moved)
            target = remaining.object_by_id(step.object_id)
            if target is None:
                continue
            physical = [gripper_evidence(target, graph.gripper_max_opening_mm)]
            destination = remaining.object_by_id(step.destination_id) if step.destination_id else None
            if destination is not None:
                markers = tuple(item.span.text for item in program.clauses[step.clause_index].evidence
                                if item.kind == "destination" and item.path == "destination")
                physical.append(placement_fit_evidence(target, destination, markers))
            for evidence in physical:
                events.append(Violation(evidence.code, evidence.reason, stage=stage,
                    object_id=target.object_id, details={"clause_index": step.clause_index,
                        "destination_id": step.destination_id, **asdict(evidence)}))
            if physical_step_closed(ledger, index):
                moved.add(step.object_id)
        policy = derive_request_policy(instruction.text, program, resolution)
        events.append(Violation(f"request_policy_{policy.code}", policy.reason, stage=stage,
            details={"disposition": policy.disposition, "code": policy.code, "reason": policy.reason,
                     "evidence": list(policy.evidence)}))
        safety = order_denial or assess_program(program, resolution, graph, policy=policy, step_order=step_order,
                                                ledger=ledger)
        approved = isinstance(safety, SafetyApproval)
        events.append(Violation("safety_approved" if approved else "safety_denied",
            "; ".join(safety.evidence) if approved else safety.reason, stage=stage,
            details={"status": "approved" if approved else "denied", **asdict(safety)}))
        stage = "compilation"
        compiled = compile_program_result(program, resolution, safety, policy=policy, ledger=ledger,
                                           step_order=step_order)
        decision = compiled.decision
        reason_code = decision_reason_code(compiled)
    except Exception as exc:
        # This per-instruction boundary preserves one output on infrastructure
        # failure as well as malformed/cached language input. Interrupts propagate.
        invalid_input = isinstance(exc, (ValueError, TypeError, KeyError, AttributeError, FileNotFoundError))
        reason_code = "interpretation_failed" if stage == "parse" and invalid_input else "system_failure"
        events.append(Violation(reason_code, str(exc), stage=stage,
            details={"status": "error", "reason": str(exc), "reason_code": reason_code}))
        compiled = compile_denial(SafetyDenial(reason_code,
            f"Unable to {'interpret the instruction' if stage == 'parse' else 'complete the request'}: {exc}"))
        decision = compiled.decision
    planning_ms = (perf_counter() - started) * 1000
    grounding_started = perf_counter()
    pixel = None
    if isinstance(decision, PlanDecision):
        try:
            pixel = resolve_first_pick(decision, graph)
            events.append(Violation("grounding_verified", "target pixel is inside mask", stage="grounding",
                object_id=decision.actions[0].object_id,
                details={"status": "verified", "target_pixel": list(pixel)}))
        except GroundingError as exc:
            reason_code = "missing_perception"
            events.append(Violation("grounding_error", str(exc), stage="grounding",
                details={"status": "error", "reason": str(exc), "reason_code": reason_code}))
            compiled = compile_denial(SafetyDenial("missing_evidence", f"Unable to ground the first pick safely: {exc}"))
            decision = compiled.decision
    elif stage != "parse":
        events.append(Violation("grounding_skipped", decision.kind, stage="grounding",
            details={"status": "skipped", "decision_kind": decision.kind}))
    details = {"decision": decision.model_dump(mode="json"), "reason_code": decision_reason_code(compiled)}
    cause = compiled.terminal_cause
    if isinstance(cause, ObligationKey):
        details["terminal_cause"] = {"source": "obligation", "key": asdict(cause)}
    elif isinstance(cause, RequestPolicy):
        details["terminal_cause"] = {"source": "source_policy", "code": cause.code}
    elif isinstance(cause, SafetyDenial):
        details["terminal_cause"] = {"source": "structural_safety", "code": cause.code}
    if isinstance(decision, PlanDecision):
        details["ordered_steps"] = [asdict(resolution.steps[index]) for index in step_order]
    events.append(Violation("decision_compiled", decision.kind, stage="compilation", details=details))
    entry = ResultEntry(instruction_id=instruction.instruction_id, target_pixel=pixel, decision=decision)
    return entry, planning_ms, (perf_counter() - grounding_started) * 1000, tuple(events)


def _relation_details(graph: SceneGraph) -> list[dict[str, JsonValue]]:
    evidence_by_relation: dict[tuple[str, str, str], RelationEvidence] = {}
    for relation in graph.relations:
        if relation.source == "geometry":
            key = (relation.subject_id, relation.predicate, relation.object_id)
            evidence_by_relation[key] = RelationEvidence(
                RelationStatus.VERIFIED, relation.predicate, relation.subject_id,
                relation.object_id, relation.confidence, ("geometry:scene_graph_relation",))
    for subject in graph.objects:
        for predicate, object_id in subject.relation_hints:
            obj = graph.object_by_id(object_id)
            if obj is not None:
                key = (subject.object_id, predicate, object_id)
                if key not in evidence_by_relation:
                    evidence_by_relation[key] = verify_relation(subject, predicate, obj)
    return [asdict(evidence_by_relation[key]) for key in sorted(evidence_by_relation)]


def run_batch(scene_instructions: Iterable[tuple[Scene, Instruction]], vision: VisionBackend,
              intent_backend: ProgramBackend) -> tuple[ResultEntry, ...]:
    entries, _ = run_batch_with_timings(scene_instructions, vision, intent_backend)
    return entries


def run_batch_with_timings(scene_instructions: Iterable[tuple[Scene, Instruction]], vision: VisionBackend,
                          intent_backend: ProgramBackend,
                          ) -> tuple[tuple[ResultEntry, ...], tuple[PipelineTiming, ...]]:
    entries, timings, graphs = [], [], {}
    for scene, instruction in scene_instructions:
        started = perf_counter()
        if instruction.scene_id != scene.scene_id:
            raise ValueError(f"instruction scene {instruction.scene_id!r} does not match {scene.scene_id!r}")
        graph = graphs.get(scene.scene_id)
        vision_ms = 0.0
        if graph is None:
            vision_started = perf_counter()
            objects = (vision.observe_scene(scene) if hasattr(vision, "observe_scene")
                       else vision.observe(scene.rgb))
            graph = build_scene_graph(scene.scene_id, scene.rgb.shape[:2], objects,
                                      gripper_max_opening_mm=scene.gripper_max_opening_mm)
            if hasattr(vision, "read_absence_evidence"):
                graph = replace(graph, absence_evidence=vision.read_absence_evidence(scene.rgb))
            vision_ms = (perf_counter() - vision_started) * 1000
            graphs[scene.scene_id] = graph
        entry, planning_ms, grounding_ms, events = _run_with_graph_timed(instruction, graph, intent_backend)
        reason_code = next((e.details["reason_code"] for e in reversed(events) if "reason_code" in e.details), None)
        entries.append(entry)
        timings.append(PipelineTiming(instruction.instruction_id, vision_ms, planning_ms, grounding_ms,
                                     (perf_counter() - started) * 1000, events, reason_code))
    return tuple(entries), tuple(timings)
