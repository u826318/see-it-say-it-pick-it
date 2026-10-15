"""The sole decision boundary: source authority, structural safety, then duties."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import (
    AskUserDecision, Decision, Obligation, ObligationKey, ObligationLedger,
    PickAction, PlaceAction, PlanDecision, RefuseDecision, Resolved, ResolvedProgram,
    ResolvedSelection, ResolvedStep, SafetyApproval, SafetyDenial, SafetyOutcome, TaskProgram, Unsupported,
)
from .obligations import build_ledger, decisive_evidence, normalize_ledger
from .program_order import ordered_program_steps as ordered_program_steps, validated_step_order
from .request_policy import RequestPolicy, alternative_request_policy


@dataclass(frozen=True, slots=True)
class CompilationResult:
    decision: Decision
    terminal_cause: ObligationKey | RequestPolicy | SafetyDenial | None
    reason_code: str


def _structural_denial(program: TaskProgram, resolution: ResolvedProgram) -> SafetyDenial | None:
    if (len(resolution.selections) != len(program.clauses)
            or len(resolution.destinations) != len(program.clauses)):
        return SafetyDenial("incomplete_program", "Incomplete program resolution.")
    if isinstance(resolution.failure, Unsupported) and not any(
            binding.outcome == resolution.failure for binding in resolution.constraints):
        return SafetyDenial(resolution.failure.code, resolution.failure.reason)
    expected, complete = list(resolution.preparation_steps), True
    for index, clause in enumerate(program.clauses):
        if clause.intent.negated:
            continue
        if clause.intent.operation == "unsupported":
            return SafetyDenial("unsupported_operation", "Program contains a non-executable clause.")
        selection, destination = resolution.selections[index], resolution.destinations[index]
        if not isinstance(selection, ResolvedSelection):
            complete = False
            continue
        quantity = clause.intent.quantity or 1
        if (not selection.object_ids or len(set(selection.object_ids)) != len(selection.object_ids)
                or (quantity != "all" and len(selection.object_ids) != quantity)):
            return SafetyDenial("invalid_selection", "Program selection does not preserve the requested quantity.")
        if clause.intent.operation == "place" and not isinstance(destination, Resolved):
            complete = False
            continue
        expected.extend(ResolvedStep(index, clause.intent.operation, object_id,
                        destination.object_id if isinstance(destination, Resolved) else None)
                        for object_id in selection.object_ids)
    if complete and tuple(expected) != resolution.steps:
        return SafetyDenial("incomplete_program", "Program steps do not preserve every selected object.")
    if complete and not expected:
        return SafetyDenial("no_movement", "The request does not authorize moving an object.")
    return None


def _safety_reason_code(cause: SafetyDenial) -> str:
    if cause.code in {"interpretation_failed", "system_failure"}:
        return cause.code
    if cause.code in {"missing_evidence", "target_quality", "destination_quality"}:
        return "missing_perception"
    if cause.code in {"invalid_destination", "dependency_conflict", "moving_destination", "repeated_target",
                      "not_graspable", "gripper_limit_exceeded", "container_nonfit", "invalid_selection",
                      "incomplete_program"}:
        return "physical_impossibility"
    if cause.code in {"contradictory_request", "unsupported_constraint", "no_movement", "unsupported_operation"}:
        return cause.code
    return "unsafe_operation"


def compile_denial(cause: SafetyDenial) -> CompilationResult:
    """Attribute a structural/infrastructure veto at the same decision boundary."""
    return CompilationResult(RefuseDecision(reason=cause.reason), cause, _safety_reason_code(cause))


def _obligation_reason_code(obligation: Obligation) -> str:
    proof = decisive_evidence(obligation)
    codes = {proof.code} if proof is not None else set()
    if "verified_absence" in codes:
        return "verified_absence"
    if "collateral_identity_conflict" in codes:
        return "contradictory_request"
    if "unsupported_constraint" in codes:
        return "unsupported_constraint"
    if obligation.status == "open":
        return "deictic_reference" if "deictic_reference" in codes else "ambiguous_reference"
    if obligation.status == "unknown":
        return "missing_perception"
    if "verified_manipulation_hazard" in codes:
        return "unsafe_operation"
    return "physical_impossibility"


def compile_program_result(program: TaskProgram, resolution: ResolvedProgram, safety: SafetyOutcome,
                           *, policy: RequestPolicy | None = None, ledger: ObligationLedger | None = None,
                           step_order: tuple[int, ...] | None = None) -> CompilationResult:
    """Choose exactly one terminal cause, before building any action."""
    if policy is None:
        policy = alternative_request_policy(program)
    if policy is not None and policy.disposition == "deny":
        return CompilationResult(RefuseDecision(reason=policy.reason), policy, policy.code)
    denial = safety if isinstance(safety, SafetyDenial) else _structural_denial(program, resolution)
    if denial is None:
        try:
            step_order = validated_step_order(program, resolution, step_order)
        except ValueError as exc:
            denial = SafetyDenial("dependency_conflict", str(exc))
    if denial is not None:
        return compile_denial(denial)
    # The explicit argument is retained for API compatibility, not authorization.
    # Only structural safety can hand the compiler an evaluated scene capability.
    normalized, completed = normalize_ledger(program, resolution,
        safety.ledger if isinstance(safety, SafetyApproval) else None)
    ledger = normalized if completed else build_ledger(program, resolution)

    def key_order(item: Obligation):
        key = item.key
        return (key.clause_index, -1 if key.step_index is None else key.step_index, key.instance_index, key.kind)

    for candidates in (ledger.violated(), ledger.blocking(), ledger.open()):
        if not candidates:
            continue
        obligation = min(candidates, key=key_order)
        if obligation.status == "open":
            question = obligation.question or (
                f"Please clarify {obligation.key.kind} in clause {obligation.key.clause_index + 1}"
                + (f": {', '.join(obligation.candidates)}" if obligation.candidates else "") + "?")
            decision = AskUserDecision(question=question)
        else:
            proof = decisive_evidence(obligation)
            reason = (proof.detail if proof is not None else
                      f"Required evidence for {obligation.key.kind} is unavailable.")
            decision = RefuseDecision(reason=reason)
        return CompilationResult(decision, obligation.key, _obligation_reason_code(obligation))
    if policy is not None and policy.disposition == "clarify":
        return CompilationResult(AskUserDecision(question=policy.reason), policy, policy.code)
    actions = []
    for index in step_order:
        step = resolution.steps[index]
        actions.append(PickAction(object_id=step.object_id))
        if step.operation == "place":
            actions.append(PlaceAction(object_id=step.destination_id))
    return CompilationResult(PlanDecision(actions=tuple(actions)), None, "planned")


def compile_program(program: TaskProgram, resolution: ResolvedProgram, safety: SafetyOutcome,
                    *, policy: RequestPolicy | None = None, ledger: ObligationLedger | None = None) -> Decision:
    """Public Decision wrapper; only safety's context-bound ledger authorizes motion."""
    return compile_program_result(program, resolution, safety, policy=policy, ledger=ledger).decision


def decision_reason_code(result: CompilationResult) -> str:
    """Use the selected compilation cause; never independently rerun precedence."""
    return result.reason_code
