"""Source-bound reference queries and offline observation recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .canonicalization import COLORS, canonicalize_detection, decompose_noun, normalize_colors, normalize_modifiers
from .domain import AbsenceEvidence, DetectedObject, TaskProgram, VisionObservation
from .localization_fallback import LocalizationFallback, validate_visual_query
from .visual_canonicalization import _compatible_identity, _iou, mark_duplicate_mask_conflicts
from .vision_quality import QUALITY_CONTRACT
from .visual_evidence import VISUAL_EVIDENCE_CONTRACT, enrich_visual_colors


RECOVERY_CONTRACT = {
    "version": "source-reference-recovery-v2",
    "query_version": "literal-bound-terms-v1",
    "detector_identity": "single-query-source-phrase-v1",
    "calls_per_scene_query": 1,
    "quality": dict(QUALITY_CONTRACT),
}


@dataclass(frozen=True, slots=True)
class ReferenceQuery:
    phrase: str
    noun: str
    colors: tuple[str, ...]
    roles: tuple[str, ...]


def collect_reference_queries(programs: tuple[TaskProgram, ...]) -> tuple[ReferenceQuery, ...]:
    """Ignore source positions and commentary; only bound visual terms leave parsing."""
    queries = {}
    for program in programs:
        for clause in program.clauses:
            def collect(reference, path, role):
                if reference is None:
                    return
                spans = [item.span for item in clause.evidence
                         if item.path == path + ".noun"
                         or item.path.startswith(path + ".colors.")
                         or item.path.startswith(path + ".attributes.")]
                # Literal spans can overlap when a noun phrase already includes color.
                terms = []
                end = -1
                for span in sorted(spans, key=lambda span: (span.start, -span.end)):
                    if span.start >= end:
                        terms.append(" ".join(span.text.lower().split()))
                        end = span.end
                phrase = " ".join(terms)
                parts = decompose_noun(reference.noun)
                try:
                    validate_visual_query(phrase)
                    validate_visual_query(reference.noun)
                except ValueError:
                    pass
                else:
                    colors = normalize_colors((*parts.colors, *reference.colors))
                    key = (parts.head, tuple(sorted(colors)), tuple(sorted(set(
                        normalize_modifiers((*parts.modifiers, *reference.attributes))))))
                    if key in queries:
                        previous = queries[key]
                        queries[key] = ReferenceQuery(previous.phrase, previous.noun, previous.colors,
                            tuple(dict.fromkeys((*previous.roles, role))))
                    else:
                        queries[key] = ReferenceQuery(phrase, parts.head, colors, (role,))
                collect(reference.relation_target, path + ".relation_target", "relation_anchor")

            collect(clause.intent.target, "target", "target")
            collect(clause.intent.destination, "destination", "destination")
            for index, reference in enumerate(clause.intent.exclusions):
                collect(reference, f"exclusions.{index}", "exclusion")
    return tuple(queries.values())


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    query: ReferenceQuery
    status: str
    object_id: str | None = None
    score: float | None = None
    label: str | None = None


@dataclass(frozen=True, slots=True)
class RecoveryObservation:
    objects: tuple[DetectedObject, ...] = ()
    records: tuple[RecoveryRecord, ...] = ()
    absence_evidence: tuple[AbsenceEvidence, ...] = ()


def _matches(obj, query):
    modifiers = normalize_modifiers((query.phrase,))
    modifiers = tuple(term for term in modifiers if term != query.noun
                      and decompose_noun(term).head != query.noun)
    return (obj.canonical_noun == query.noun and set(query.colors) <= set(obj.colors)
            and set(modifiers) <= set(obj.attributes))


def unmatched_reference_queries(observation: VisionObservation,
                                queries: tuple[ReferenceQuery, ...]) -> tuple[ReferenceQuery, ...]:
    objects = [canonicalize_detection(obj, obj.object_id)
               for obj in mark_duplicate_mask_conflicts(observation.objects)]
    return tuple(query for query in queries
                 if not any(_matches(obj, query) and obj.quality.groundable for obj in objects))


def valid_absence_evidence(item: AbsenceEvidence) -> bool:
    return (bool(item.noun and item.colors and item.query_phrase and item.contrast_object_id)
            and item.source == "base_nonmatch+targeted_detector_miss+measured_contrast"
            and set(item.colors) <= COLORS and bool(item.measured_colors)
            and set(item.measured_colors) <= COLORS
            and set(item.colors).isdisjoint(item.measured_colors)
            and item.confidence >= VISUAL_EVIDENCE_CONTRACT["high_confidence"]
            and item.confidence <= 1
            and item.method.startswith(VISUAL_EVIDENCE_CONTRACT["method_prefix"]))


def recover_references(image: np.ndarray, base_observation: VisionObservation,
                       queries: tuple[ReferenceQuery, ...], detector: LocalizationFallback | None
                       ) -> RecoveryObservation:
    base = mark_duplicate_mask_conflicts(base_observation.objects)
    canonical_base = [canonicalize_detection(obj, obj.object_id) for obj in base]
    batches, proposals = [], []
    existing_ids = {obj.object_id for obj in base}
    next_index = 1
    for query in dict.fromkeys(queries):
        validate_visual_query(query.phrase)
        if any(_matches(obj, query) and obj.quality.groundable for obj in canonical_base):
            batches.append((query, None))
            continue
        if detector is None:
            raise ValueError("unmatched recovery reference requires a detector during cache-vision")
        batch = []
        for proposal in enrich_visual_colors(image, detector.detect(image, query.phrase)):
            while f"recovered_{next_index:03d}" in existing_ids:
                next_index += 1
            object_id = f"recovered_{next_index:03d}"
            next_index += 1
            existing_ids.add(object_id)
            batch.append(replace(proposal, object_id=object_id, estimated_width_mm=None,
                                 relation_hints=()))
        batches.append((query, tuple(batch)))
        proposals.extend(batch)

    # Every query contributes observations before acceptance. A conflicting
    # identity invalidates both sides, regardless of proposal/query order.
    checked = mark_duplicate_mask_conflicts((*base, *proposals))
    base = checked[:len(base)]
    checked_by_id = {obj.object_id: obj for obj in checked}
    observed_proposals = [canonicalize_detection(obj, obj.object_id) for obj in proposals]
    objects, records, absence = [], [], []
    for query, batch in batches:
        canonical = [canonicalize_detection(obj, obj.object_id) for obj in base]
        if batch is None:
            status = "base_match" if any(_matches(obj, query) and obj.quality.groundable
                                         for obj in canonical) else "unresolved"
            records.append(RecoveryRecord(query, status))
            continue
        accepted = False
        for raw in batch:
            proposal = checked_by_id[raw.object_id]
            observed = canonicalize_detection(proposal, proposal.object_id)
            status = "accepted"
            if proposal.quality.status == "conflict":
                status = "rejected_conflict"
            elif not _matches(observed, query):
                status = "rejected_identity"
            elif not proposal.quality.groundable:
                status = "rejected_quality"
            elif any(_iou(proposal, other) >= QUALITY_CONTRACT["same_object_iou"]
                     and _compatible_identity(proposal, other) and other.quality.groundable
                     and _matches(canonicalize_detection(other, other.object_id), query)
                     for other in (*base, *objects)):
                status = "duplicate"
            if status in {"accepted", "rejected_conflict"}:
                # Keep conflicting observations in the layer so composition
                # also invalidates any base object on the other side.
                objects.append(proposal)
            if status == "accepted":
                accepted = True
            records.append(RecoveryRecord(query, status, proposal.object_id, proposal.confidence, proposal.label))
        if accepted:
            continue
        # Rejected proposals still contain possible evidence; only an actual
        # detector miss plus a usable independent contrast can establish absence.
        contrast = None
        if not batch and not any(_matches(obj, query) for obj in canonical):
            for obj in base:
                evidence = obj.color_evidence
                if decompose_noun(obj.label).head != query.noun or not obj.quality.groundable or evidence is None:
                    continue
                candidate = AbsenceEvidence(query.noun, query.colors, query.phrase, obj.object_id,
                    evidence.colors, evidence.confidence, evidence.method)
                if valid_absence_evidence(candidate):
                    contrast = candidate
                    break
        if contrast is not None:
            absence.append(contrast)
        records.append(RecoveryRecord(query, "verified_absence" if contrast else "unresolved"))
    absence = [item for item in absence if not any(_matches(obj,
        ReferenceQuery(item.query_phrase, item.noun, item.colors, ())) for obj in observed_proposals)]
    records = [replace(record, status="unresolved")
        if record.status == "verified_absence" and not any(
            item.query_phrase == record.query.phrase for item in absence) else record for record in records]
    return RecoveryObservation(tuple(objects), tuple(records), tuple(absence))


def compose_recovered_detections(base: tuple[DetectedObject, ...],
                                 recovered: tuple[DetectedObject, ...]) -> tuple[DetectedObject, ...]:
    """Compose compatible observed descriptions/geometry, preserving base IDs/links."""
    checked = mark_duplicate_mask_conflicts((*base, *recovered))
    objects = list(checked[:len(base)])
    for proposal in checked[len(base):]:
        duplicate = next((index for index, obj in enumerate(objects)
            if _compatible_identity(obj, proposal)
            and obj.quality.status != "conflict" and proposal.quality.status != "conflict"
            and _iou(obj, proposal) >= QUALITY_CONTRACT["same_object_iou"]), None)
        if duplicate is None:
            objects.append(proposal)
        elif proposal.quality.groundable:
            original = objects[duplicate]
            parts = decompose_noun(proposal.label)
            attributes = tuple(dict.fromkeys((*original.attributes, *proposal.attributes,
                                              *parts.colors, *parts.modifiers)))
            geometry = original if original.quality.groundable else proposal
            objects[duplicate] = replace(geometry, object_id=original.object_id,
                label=original.label, attributes=attributes, relation_hints=original.relation_hints,
                role=original.role, graspable=original.graspable,
                color_evidence=original.color_evidence or proposal.color_evidence)
    return mark_duplicate_mask_conflicts(tuple(objects))
