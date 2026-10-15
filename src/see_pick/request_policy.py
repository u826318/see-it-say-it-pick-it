"""Source-text authority and supported program shape, independent of perception."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from .domain import IntentResolution, ResolvedProgram, TaskIntent, TaskProgram
from .intent_codec import validate_source_coverage, requires_program


_PICK = r"pick(?:\s+up)?|grab|grasp|lift|take"
_PLACE = r"(?<!in )place|put|move|relocate"
_PROHIBITED = r"throw|toss|drop|push\b.*\b(?:off|over the edge)"
_HANDOFF = r"give|hand|pass"
_DEICTIC = r"it|them|this|that|these|those|one|ones"
_INQUIRY = r"where|which|what|how|tell|describe|explain"
_INSPECTION = rf"inspect|look|find|locate|identify|show|point|{_INQUIRY}"
_NEGATION = re.compile(r"\b(?:do not|don't|never|must not|cannot|can't)\b", re.I)
_CLAUSES = re.compile(r"[.;!?]|\b(?:and|but|then)\b", re.I)


@dataclass(frozen=True, slots=True)
class RequestPolicy:
    instruction: str
    disposition: Literal["allow", "clarify", "deny"]
    code: str
    reason: str
    evidence: tuple[str, ...]


def alternative_request_policy(program: TaskProgram, *, instruction: str = "") -> RequestPolicy | None:
    """Source alternatives require an explicit choice, including standalone compilation."""
    if any(link.kind == "or" for link in program.conjunctions):
        return RequestPolicy(instruction, "clarify", "alternative_request",
                             "Please choose which alternative should be performed.", ("source:or",))
    return None


def derive_request_policy(
    instruction: str,
    intent: TaskIntent | TaskProgram,
    resolution: IntentResolution | ResolvedProgram | None = None,
) -> RequestPolicy:
    """Check source authority only; resolution is retained for caller compatibility."""
    return _derive_request_policy(instruction, intent, program_member=False)


def _derive_request_policy(instruction, intent, *, program_member=False,
                           inherited_operation=None):
    def inspection_clause(clause: str) -> bool:
        query = re.search(rf"\b(?:{_INSPECTION})\b", clause)
        action = re.search(rf"\b(?:{_PICK}|{_PLACE}|{_PROHIBITED})\b", clause)
        return query is not None and (action is None or query.start() < action.start())

    text = " ".join(instruction.lower().split())
    clause_parts: list[str] = []
    for sentence in re.split(r"[.;!?]", text):
        # A question/explanation governs coordinated verbs until its sentence ends.
        if inspection_clause(sentence) and re.search(rf"\b(?:{_INQUIRY})\b", sentence):
            clause_parts.append(sentence.strip())
        else:
            clause_parts.extend(clause.strip() for clause in _CLAUSES.split(sentence) if clause.strip())
    clauses = tuple(clause_parts)
    positive = tuple(clause for clause in clauses if not _NEGATION.search(clause))
    negative = tuple(clause for clause in clauses if _NEGATION.search(clause))

    def has(pattern: str, parts: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b(?:{pattern})\b", part) for part in parts)

    def policy(disposition: Literal["allow", "clarify", "deny"], code: str, reason: str) -> RequestPolicy:
        return RequestPolicy(instruction, disposition, code, reason, (f"request:{instruction}",))

    action_clauses = tuple(clause for clause in positive if not inspection_clause(clause))
    pick = has(_PICK, action_clauses) or inherited_operation == "pick"
    place = has(_PLACE, action_clauses) or inherited_operation == "place"
    no_move = has(rf"{_PICK}|{_PLACE}", negative) or has(
        r"leave\b.*\b(?:in place|alone|where (?:it|they|this|that|these|those) (?:is|are))|"
        r"keep\b.*\b(?:still|in place)|no movement", clauses
    )
    if has(_PROHIBITED, action_clauses):
        return policy("deny", "unsafe_operation", "The requested operation is unsafe.")
    if not isinstance(intent, TaskProgram) and no_move and (pick or place):
        return policy("deny", "contradictory_request", "The request both requires and prohibits movement.")
    if not isinstance(intent, TaskProgram) and no_move:
        return policy("deny", "no_movement", "The request does not authorize moving an object.")
    if not (pick or place):
        if has(_INSPECTION, positive):
            return policy("deny", "inspection_only", "Inspection alone does not authorize a physical action.")
        if has(_HANDOFF, action_clauses) and has(_DEICTIC, action_clauses):
            return policy("clarify", "deictic_reference", "Which visible object do you mean?")
        return policy("deny", "unsupported_operation", "The request contains no supported manipulation operation.")
    if isinstance(intent, TaskProgram):
        if validate_source_coverage(instruction, intent):
            return policy("deny", "uncovered_request", "The program does not preserve the original request.")
        alternative = alternative_request_policy(intent, instruction=instruction)
        if alternative is not None:
            return alternative
        if not any(not clause.intent.negated for clause in intent.clauses):
            return policy("deny", "no_movement", "The request does not authorize moving an object.")
        for clause in intent.clauses:
            if clause.intent.negated:
                continue
            inherited = clause.intent.operation if clause.inherited_operation_from is not None else None
            clause_policy = _derive_request_policy(clause.source.text, clause.intent, program_member=True,
                                                    inherited_operation=inherited)
            if clause_policy.disposition != "allow":
                return policy(clause_policy.disposition, clause_policy.code, clause_policy.reason)
            if any(not re.fullmatch(r"without\s+moving\s+.+", item, re.I)
                   for item in clause.intent.user_constraints):
                return policy("deny", "unsupported_constraint", "The program contains an unsupported execution constraint.")
        return policy("allow", "movement_requested", "The request authorizes every program clause.")
    if not program_member and (len(action_clauses) > 1 or any(requires_program(part) for part in action_clauses)
                               or intent.quantity not in (None, 1) or intent.exclusions):
        return policy("clarify", "program_required", "This request requires a complete source-covered task program.")
    destination_named = has(rf"(?:{_PLACE})\b.+\b(?:to|onto|on|into|in|inside)\s+\S+", action_clauses)
    if inherited_operation == "place":
        destination_named = has(r"(?:to|onto|on|into|in|inside)\s+\S+", action_clauses)
    if place and (not destination_named or intent.operation != "place" or intent.destination is None):
        return policy("clarify", "missing_destination", "Where should the object be placed?")
    if intent.operation != ("place" if place else "pick"):
        return policy("clarify", "operation_mismatch", "Please clarify the requested manipulation operation.")
    return policy("allow", "movement_requested", "The request authorizes this manipulation operation.")
