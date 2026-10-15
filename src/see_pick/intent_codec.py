"""Flat model JSON transport and internal program source-coverage validation."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .canonicalization import decompose_noun
from .domain import (
    CoverageFinding, LiteralEvidence, LiteralSpan, ObjectReference,
    TaskIntent, TaskProgram,
)
from .language import FLAT_COMMAND_SCHEMA, NO_MOVEMENT, FlatCommand, extract_closed_class


class IntentDecodeError(ValueError):
    """A model response cannot be converted to a valid task intent."""

    def __init__(self, message: str, findings: tuple[CoverageFinding, ...] = ()):
        super().__init__(message)
        self.findings = findings or (CoverageFinding(code="invalid_structure", message="invalid structure"),)


INTENT_SCHEMA = FLAT_COMMAND_SCHEMA
_PROGRAM_SCHEMA = TaskProgram.model_json_schema()
_FLAT_SCHEMA = FlatCommand.model_json_schema()

_FINDING_CODES = frozenset({
    "invalid_structure", "invalid_span", "overlapping_clauses", "out_of_clause",
    "invalid_binding", "value_mismatch", "duplicate_binding", "missing_evidence",
    "reference_scope", "invalid_conjunction", "missing_conjunction",
    "uncovered_literal", "uncovered_text",
})


def _schema_path(path: str | None) -> str | None:
    """Allow schema-owned property names only; canonicalize array indices."""
    if not path:
        return None

    def variants(schema):
        if "$ref" in schema:
            yield from variants(_PROGRAM_SCHEMA["$defs"][schema["$ref"].rsplit("/", 1)[-1]])
        elif "anyOf" in schema:
            for option in schema["anyOf"]:
                yield from variants(option)
        else:
            yield schema

    candidates = [_PROGRAM_SCHEMA, _PROGRAM_SCHEMA["$defs"]["TaskIntent"], _FLAT_SCHEMA]
    parts = []
    for part in path.split("."):
        is_index = part == "*" or (part.isascii() and part.isdigit())
        candidates = [child for candidate in candidates for schema in variants(candidate)
                      if (child := schema.get("items") if is_index
                          else schema.get("properties", {}).get(part)) is not None]
        if not candidates:
            return None
        parts.append("*" if is_index else part)
    return ".".join(parts)


def _safe_finding(instruction, code, kind=None, span=None, path=None) -> CoverageFinding:
    """Only source slices and schema-owned vocabulary may cross the repair boundary."""
    code = code if code in _FINDING_CODES else "invalid_structure"
    source_span = None
    if span is not None and 0 <= span.start < span.end <= len(instruction):
        source_span = LiteralSpan(start=span.start, end=span.end,
                                  text=instruction[span.start:span.end])
    return CoverageFinding(code=code, kind=kind, span=source_span,
                           path=_schema_path(path), message=code.replace("_", " "))


def build_intent_messages(
    instruction: str,
    validation_feedback: tuple[CoverageFinding, ...] = (),
) -> list[dict[str, str]]:
    examples = (
        (
            "Lift the third brass token from the right nearest to the ceramic vase",
            {"operation": "pick", "target_phrase": "brass token", "relation": "nearest_to",
             "relation_phrase": "ceramic vase", "ordinal": 3, "ordinal_from": "right"},
        ),
        (
            "Pick all wooden counters except the chipped counter",
            {"operation": "pick", "target_phrase": "wooden counters", "quantity": "all",
             "exclusion_phrases": ["chipped counter"]},
        ),
        (
            "Place the linen pouch into the violet basket and the metal disc into the shallow tray",
            {"operation": "place", "target_phrase": "linen pouch",
             "destination_phrase": "violet basket",
             "second_clause": "the metal disc into the shallow tray"},
        ),
        (
            "Rotate the glass prism",
            {"operation": "unsupported", "target_phrase": "glass prism"},
        ),
    )
    messages = [
        {"role": "system", "content": (
            "Interpret one instruction clause as FlatCommand JSON. Copy target_phrase, "
            "destination_phrase, relation_phrase and exclusion_phrases verbatim from the source. "
            "Use second_clause only for a remaining executable clause, copied exactly as a source "
            "suffix after a conjunction. A noun phrase such as salt and pepper shaker stays one "
            "reference. Source literals own operations and closed selectors. Never infer objects "
            "from a scene. JSON schema: " + json.dumps(INTENT_SCHEMA, separators=(",", ":"))
        )},
    ]
    for example_instruction, answer in examples:
        messages.extend((
            {"role": "user", "content": json.dumps({
                "instruction": example_instruction, "coverage_findings": []}, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ))
    messages.append({"role": "user", "content": json.dumps({
            "instruction": instruction,
            "coverage_findings": [
                _safe_finding(instruction, f.code, f.kind, f.span, f.path).model_dump(mode="json")
                for f in validation_feedback
            ],
        }, ensure_ascii=False)})
    return messages


def decode_flat_command(raw: str) -> FlatCommand:
    try:
        return FlatCommand.model_validate_json(raw)
    except ValidationError as exc:
        findings = tuple(_safe_finding(
            "", "invalid_structure", path=".".join(map(str, error["loc"])))
            for error in exc.errors(include_input=False)
        )
        raise IntentDecodeError("invalid flat command structure", findings) from exc


def decode_intent(raw: str, instruction: str) -> TaskIntent:
    from .language import build_task_program
    program = build_task_program(instruction, lambda _: decode_flat_command(raw))
    if len(program.clauses) != 1:
        raise IntentDecodeError("intent requires a single clause")
    return program.clauses[0].intent


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def _markers(instruction: str, *, inherited_operation: str | None = None):
    for marker in extract_closed_class(
            instruction, inherited_operation=inherited_operation).markers:
        yield marker.kind, marker.value, marker.span


def _semantic_fields(intent: TaskIntent) -> dict[str, tuple[str, object]]:
    fields: dict[str, tuple[str, object]] = {"operation": ("operation", intent.operation)}

    def reference(ref: ObjectReference, path: str):
        fields[path + ".noun"] = ("noun", ref.noun)
        for name, kind in (("colors", "color"), ("attributes", "attribute")):
            for index, value in enumerate(getattr(ref, name)):
                fields[f"{path}.{name}.{index}"] = (kind, value)
        for name in ("ordinal", "order_axis", "extremum", "relation"):
            value = getattr(ref, name)
            if value is not None:
                fields[path + "." + name] = (name, value)
        if ref.relation_target is not None:
            reference(ref.relation_target, path + ".relation_target")
        if ref.clear_top:
            fields[path + ".clear_top"] = ("clear_top", True)

    if intent.target:
        reference(intent.target, "target")
    if intent.destination:
        fields["destination"] = ("destination", True)
        reference(intent.destination, "destination")
    if intent.placement_relation is not None:
        fields["placement_relation"] = ("relation", intent.placement_relation)
        reference(intent.placement_anchor, "placement_anchor")
    if intent.quantity is not None:
        fields["quantity"] = ("quantity", intent.quantity)
    if intent.negated:
        fields["negated"] = ("negation", True)
    for index, ref in enumerate(intent.exclusions):
        fields[f"exclusions.{index}"] = ("exclusion", True)
        reference(ref, f"exclusions.{index}")
    for index, value in enumerate(intent.user_constraints):
        fields[f"user_constraints.{index}"] = ("attribute", _normal(value))
    return fields


def validate_source_coverage(instruction: str, program: TaskProgram) -> tuple[CoverageFinding, ...]:
    """Own both directions: semantic fields need literals; source constraints need fields.

    Closed markers must have matching typed evidence, even inside a noun or quoted
    constraint, except color homographs occupying a bound noun's head role.
    All other content words must be preserved literally. This is a
    conservative structural check, not a general natural-language proof.
    """
    findings: list[CoverageFinding] = []
    covered: list[tuple[str, LiteralSpan]] = []
    noun_head_spans: set[tuple[int, int]] = set()
    execution_constraints: list[LiteralSpan] = []

    def issue(code, kind=None, span=None, path=None):
        findings.append(_safe_finding(instruction, code, kind, span, path))

    def valid_span(span: LiteralSpan) -> bool:
        if instruction[span.start:span.end] != span.text or span.end > len(instruction):
            issue("invalid_span", span=span)
            return False
        return True

    markers = list(_markers(instruction))
    # Marker roles use source-bound clauses, never a guessed split at "and".
    # Keep markers outside those clauses so dropped source text stays detectable.
    for clause in program.clauses:
        source = clause.source
        if instruction[source.start:source.end] != source.text or source.end > len(instruction):
            continue
        markers = [(kind, value, span) for kind, value, span in markers
                   if not source.start <= span.start < span.end <= source.end]
        inherited = clause.intent.operation if clause.inherited_operation_from is not None else None
        for kind, value, span in _markers(source.text, inherited_operation=inherited):
            markers.append((kind, value, LiteralSpan(start=source.start + span.start,
                end=source.start + span.end, text=span.text)))
    for index, clause in enumerate(program.clauses):
        valid_span(clause.source)
        if index and program.clauses[index - 1].source.end > clause.source.start:
            issue("overlapping_clauses", span=clause.source)
        fields = _semantic_fields(clause.intent)
        if clause.inherited_operation_from is not None:
            parent = clause.inherited_operation_from
            if (index == 0 or parent != index - 1
                    or clause.intent.operation != program.clauses[parent].intent.operation
                    or any(kind == "operation" and clause.source.start <= span.start < clause.source.end
                           for kind, _, span in markers)):
                issue("invalid_binding", "operation", clause.source, "operation")
            else:
                fields.pop("operation")
        bound: set[str] = set()
        bindings: dict[str, LiteralEvidence] = {}
        for evidence in clause.evidence:
            kind, path, span = evidence.kind, evidence.path, evidence.span
            if not valid_span(span):
                continue
            if not clause.source.start <= span.start < span.end <= clause.source.end:
                issue("out_of_clause", kind, span, path)
                continue
            if kind == "conjunction" and path in fields and fields[path][0] == "noun":
                noun_evidence = next((e for e in clause.evidence if e.path == path and e.kind == "noun"), None)
                valid_noun_link = (noun_evidence is not None
                    and noun_evidence.span.start <= span.start < span.end <= noun_evidence.span.end)
                if path.startswith("exclusions.") and noun_evidence is not None:
                    position = int(path.split(".")[1])
                    previous = next((e for e in clause.evidence
                        if e.kind == "noun" and e.path == f"exclusions.{position - 1}.noun"), None)
                    valid_noun_link |= (previous is not None
                        and previous.span.end <= span.start < span.end <= noun_evidence.span.start)
                if valid_noun_link and any(k == kind and s == span for k, _, s in markers):
                    covered.append((kind, span))
                else:
                    issue("invalid_binding", kind, span, path)
                continue
            if path not in fields or fields[path][0] != kind:
                issue("invalid_binding", kind, span, path)
                continue
            value = fields[path][1]
            if kind == "noun":
                matches = span.text == value
            elif kind == "attribute":
                matches = _normal(span.text) == _normal(value)
            else:
                matches = any(k == kind and v == value and s == span for k, v, s in markers)
            if not matches:
                issue("value_mismatch", kind, span, path)
                continue
            if path in bound:
                issue("duplicate_binding", kind, span, path)
            bound.add(path)
            bindings[path] = evidence
            covered.append((kind, span))
            if kind == "noun" and (head_span := decompose_noun(span.text).head_span) is not None:
                noun_head_spans.add((span.start + head_span[0], span.start + head_span[1]))
        for path, (kind, _) in fields.items():
            if path not in bound:
                issue("missing_evidence", kind, path=f"clauses.{index}.intent.{path}")
        movement_markers = tuple(NO_MOVEMENT.finditer(instruction, clause.source.start, clause.source.end))
        for position, marker in enumerate(movement_markers):
            boundary = movement_markers[position + 1].start() if position + 1 < len(movement_markers) else clause.source.end
            constraint_end = len(instruction[:boundary].rstrip().rstrip(".!?").rstrip())
            constraint = next((e.span for path, e in bindings.items()
                if path.startswith("user_constraints.") and e.span.start == marker.start()
                and e.span.end == constraint_end), None)
            if constraint is None:
                issue("missing_evidence", "attribute", LiteralSpan(
                    start=marker.start(), end=marker.end(), text=marker.group()), "user_constraints")
            elif not any(kind == "operation" and constraint.start <= span.start < constraint.end
                         for kind, _, span in markers):
                execution_constraints.append(constraint)
        execution_constraints.extend(
            evidence.span for path, evidence in bindings.items()
            if path.startswith("user_constraints.") and evidence.span not in execution_constraints
            and not any(kind == "operation"
                        and evidence.span.start <= span.start < span.end <= evidence.span.end
                        for kind, _, span in markers)
        )
        # Bind modifiers to their literal noun phrase, not merely a matching
        # keyword elsewhere in the clause (especially target vs destination).
        nouns = sorted((e for e in bindings.values() if e.kind == "noun"), key=lambda e: e.span.start)
        for position, noun in enumerate(nouns):
            prefix = noun.path.removesuffix(".noun")
            lower = nouns[position - 1].span.end if position else clause.source.start
            upper = nouns[position + 1].span.start if position + 1 < len(nouns) else clause.source.end
            for path, evidence in bindings.items():
                if (path.startswith(prefix + ".colors.") or path.startswith(prefix + ".attributes.")
                        or path in {prefix + ".ordinal", prefix + ".extremum"}):
                    head = decompose_noun(noun.span.text).head_span
                    modifier_end = noun.span.start + head[0] if head is not None else noun.span.start
                    prefix_modifier = lower <= evidence.span.start < evidence.span.end <= modifier_end
                    postfix_extremum = (path == prefix + ".extremum"
                        and noun.span.end <= evidence.span.start < evidence.span.end <= upper)
                    if not (prefix_modifier or postfix_extremum):
                        issue("reference_scope", evidence.kind, evidence.span, path)
                if path == prefix + ".order_axis":
                    if not noun.span.end <= evidence.span.start < evidence.span.end <= upper:
                        issue("reference_scope", evidence.kind, evidence.span, path)
        for path, evidence in bindings.items():
            before, after = None, None
            if evidence.kind == "destination":
                before, after = "target.noun", "destination.noun"
            elif evidence.kind == "exclusion":
                before, after = "target.noun", path + ".noun"
            elif evidence.kind == "relation" and path == "placement_relation":
                before, after = "target.noun", "placement_anchor.noun"
            elif evidence.kind == "relation":
                prefix = path.removesuffix(".relation")
                before, after = prefix + ".noun", prefix + ".relation_target.noun"
            elif evidence.kind == "operation":
                after = "target.noun"
            elif evidence.kind == "quantity":
                target_noun = bindings.get("target.noun")
                if target_noun is None or evidence.span != target_noun.span:
                    after = "target.noun"
            elif evidence.kind == "negation":
                after = "operation"
            if ((before in bindings and bindings[before].span.end > evidence.span.start)
                    or (after in bindings and evidence.span.end > bindings[after].span.start)):
                issue("reference_scope", evidence.kind, evidence.span, path)

    links: set[int] = set()
    for conjunction in program.conjunctions:
        span = conjunction.span
        if not valid_span(span):
            continue
        index = conjunction.after_clause
        if (index >= len(program.clauses) - 1 or index in links
                or not program.clauses[index].source.end <= span.start < span.end
                <= program.clauses[index + 1].source.start
                or _normal(span.text) != conjunction.kind):
            issue("invalid_conjunction", "conjunction", span)
            continue
        links.add(index)
        covered.append(("conjunction", span))
    if len(links) != len(program.clauses) - 1:
        issue("missing_conjunction", "conjunction")
    for kind, _, span in markers:
        if kind == "color" and (span.start, span.end) in noun_head_spans:
            continue
        if any(constraint.start <= span.start < span.end <= constraint.end for constraint in execution_constraints):
            continue
        if (kind, span) not in covered:
            issue("uncovered_literal", kind, span)
    for word in re.finditer(r"\b[\w'-]+\b", instruction):
        if word.group().lower() in {"the", "a", "an", "please"}:
            continue
        if not any(span.start <= word.start() and word.end() <= span.end for _, span in covered):
            issue("uncovered_text", span=LiteralSpan(start=word.start(), end=word.end(), text=word.group()))
    return tuple(findings)


def decode_program(raw: str, instruction: str) -> TaskProgram:
    try:
        program = TaskProgram.model_validate_json(raw)
    except ValidationError as exc:
        # No model response or scene data is echoed into a repair prompt.
        findings = tuple(_safe_finding(instruction, "invalid_structure", path=".".join(map(str, e["loc"])))
                         for e in exc.errors(include_input=False))
        raise IntentDecodeError("invalid task program structure", findings) from exc
    findings = validate_source_coverage(instruction, program)
    if findings:
        raise IntentDecodeError("task program has source coverage findings", findings)
    return program


def program_to_intent(program: TaskProgram) -> TaskIntent:
    """Fail closed when a complete program cannot fit the legacy one-intent boundary."""
    if len(program.clauses) != 1 or program.conjunctions:
        raise IntentDecodeError("legacy adapter requires a single clause")
    intent = program.clauses[0].intent
    if intent.quantity not in (None, 1) or intent.negated or intent.exclusions or intent.user_constraints:
        raise IntentDecodeError("legacy adapter cannot execute preserved program constraints")
    return intent


def requires_program(instruction: str) -> bool:
    """Prevent a legacy backend from silently reducing source collection semantics."""
    return any(kind in {"conjunction", "exclusion"} or (kind == "quantity" and value != 1)
               for kind, value, _ in _markers(instruction))
