"""Stable domain types shared by perception, planning, and result writing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictInt,
    field_validator,
    model_validator,
)


MIN_LEGAL_TARGET_AREA = 4
SceneRole: TypeAlias = Literal["graspable_object", "container", "hand", "human", "background", "support_surface"]


def infer_scene_role(label: str) -> SceneRole:
    """Backward-compatible role inference for inventories predating typed roles."""
    words = set(re.findall(r"[a-z0-9]+", label.lower()))
    if words & {"hand", "hands"}:
        return "hand"
    if words & {"person", "human", "arm", "people"}:
        return "human"
    if words & {"background", "wall", "floor"}:
        return "background"
    if words & {"table", "surface", "tray", "plate", "saucer", "countertop", "shelf"}:
        return "support_surface"
    return "graspable_object"


@dataclass(frozen=True, slots=True)
class BBox:
    """Half-open bounding box in original-image pixel coordinates."""

    u_min: int
    v_min: int
    u_max: int
    v_max: int

    def __post_init__(self) -> None:
        if self.u_min < 0 or self.v_min < 0:
            raise ValueError("bounding-box coordinates must be non-negative")
        if self.u_min >= self.u_max or self.v_min >= self.v_max:
            raise ValueError("bounding-box coordinates must be ordered")


@dataclass(frozen=True, slots=True)
class VisionQuality:
    """Mask geometry evidence; no pixels or model payloads in telemetry."""

    status: Literal["usable", "uncertain", "invalid", "conflict"]
    reasons: tuple[str, ...]
    mask_area: int
    mask_in_box_area: int
    mask_in_box_ratio: float
    component_count: int
    largest_component_ratio: float
    hole_ratio: float
    extent_width_ratio: float
    extent_height_ratio: float
    conflicting_object_ids: tuple[str, ...] = ()

    @property
    def groundable(self) -> bool:
        return self.status == "usable"


def effective_mask(mask: np.ndarray, box: BBox) -> np.ndarray:
    """Shared image-space foreground region; preserve raw masks for audit."""
    if mask.ndim != 2:
        raise ValueError("segmentation mask must be two-dimensional")
    region = np.zeros(mask.shape, dtype=bool)
    region[box.v_min:box.v_max, box.u_min:box.u_max] = mask[
        box.v_min:box.v_max, box.u_min:box.u_max
    ]
    return region


@dataclass(frozen=True, slots=True)
class VisualColorEvidence:
    """Measured color support, retained separately from source descriptions."""
    colors: tuple[str, ...]
    confidence: float
    method: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("visual color confidence must be finite and normalized")


@dataclass(frozen=True, slots=True)
class MetricFootprint:
    """Measured planar short/long extents in mm; confidence is evidence quality."""

    min_extent_mm: float
    max_extent_mm: float
    method: str
    confidence: float

    def __post_init__(self) -> None:
        if (not np.isfinite([self.min_extent_mm, self.max_extent_mm, self.confidence]).all()
                or not 0 < self.min_extent_mm <= self.max_extent_mm
                or not 0 <= self.confidence <= 1 or not self.method):
            raise ValueError("metric footprint requires finite ordered extents and confidence")


@dataclass(frozen=True, slots=True)
class PhysicalEvidence:
    status: Literal["impossible", "not_proven_impossible", "unknown"]
    code: str
    reason: str
    lower_bound_mm: float | None = None
    upper_bound_mm: float | None = None


@dataclass(frozen=True, slots=True)
class PixelFitEvidence:
    """Scale-free image evidence; destination body is only an opening proxy."""

    status: Literal["closed", "violated", "unknown"]
    code: str
    reason: str
    margin: float
    target_width_px: float | None = None
    opening_proxy_px: float | None = None
    ratio: float | None = None
    target_quality: str | None = None
    destination_quality: str | None = None


@dataclass(frozen=True, slots=True)
class AbsenceEvidence:
    noun: str
    colors: tuple[str, ...]
    query_phrase: str
    contrast_object_id: str
    measured_colors: tuple[str, ...]
    confidence: float
    method: str
    source: str = "base_nonmatch+targeted_detector_miss+measured_contrast"


@dataclass(frozen=True, slots=True)
class DetectedObject:
    object_id: str
    label: str
    attributes: tuple[str, ...]
    confidence: float
    bbox: BBox
    mask: np.ndarray = field(repr=False, compare=False)
    role: SceneRole | None = None
    graspable: bool = True
    estimated_width_mm: float | None = None
    relation_hints: tuple[tuple[str, str], ...] = ()
    quality: VisionQuality | None = None
    color_evidence: VisualColorEvidence | None = None
    metric_footprint: MetricFootprint | None = None


@dataclass(frozen=True, slots=True)
class DroppedDetection:
    object_id: str
    label: str
    reason: str


@dataclass(frozen=True, slots=True)
class PerceptionRecovery:
    object_id: str
    label: str
    trigger: tuple[str, ...]
    outcome: Literal["recovered", "unresolved", "rejected"]
    method: Literal["inventory_reprompt", "segment_retry"]
    attempts: int = 1


@dataclass(frozen=True, slots=True)
class VisionObservation:
    inventory_count: int
    objects: tuple[DetectedObject, ...]
    dropped: tuple[DroppedDetection, ...] = ()
    recovery: tuple[PerceptionRecovery, ...] = ()

    def __post_init__(self) -> None:
        if self.inventory_count != len(self.objects) + len(self.dropped):
            raise ValueError("inventory_count must equal kept and dropped detections")

    @classmethod
    def complete(cls, objects: tuple[DetectedObject, ...]) -> VisionObservation:
        return cls(inventory_count=len(objects), objects=objects)


@dataclass(frozen=True, slots=True)
class CanonicalObject:
    """A perception result normalized for deterministic reasoning."""

    object_id: str
    canonical_noun: str
    aliases: tuple[str, ...]
    colors: tuple[str, ...]
    attributes: tuple[str, ...]
    confidence: float
    bbox: BBox
    mask: np.ndarray = field(repr=False, compare=False)
    role: SceneRole = "graspable_object"
    graspable: bool = True
    graspability_evidence: tuple[str, ...] = ()
    estimated_width_mm: float | None = None
    raw_label: str = ""
    relation_hints: tuple[tuple[str, str], ...] = ()
    raw_attributes: tuple[str, ...] = ()
    quality: VisionQuality | None = None
    color_evidence: VisualColorEvidence | None = None
    metric_footprint: MetricFootprint | None = None


@dataclass(frozen=True, slots=True)
class SpatialRelation:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source: Literal["geometry", "vision"] = "geometry"


class RelationStatus(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class RelationEvidence:
    status: RelationStatus
    predicate: str
    subject_id: str
    object_id: str
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SceneGraph:
    scene_id: str
    image_shape: tuple[int, int]
    objects: tuple[CanonicalObject, ...]
    relations: tuple[SpatialRelation, ...] = ()
    absence_evidence: tuple[AbsenceEvidence, ...] = ()
    gripper_max_opening_mm: float = 85.0

    def object_by_id(self, object_id: str) -> CanonicalObject | None:
        return next((obj for obj in self.objects if obj.object_id == object_id), None)


@dataclass(frozen=True, slots=True)
class Instruction:
    instruction_id: str
    scene_id: str
    text: str
    category: str | None = None


@dataclass(frozen=True, slots=True)
class Scene:
    scene_id: str
    rgb: np.ndarray = field(repr=False, compare=False)
    depth_mm: np.ndarray | None = field(default=None, repr=False, compare=False)
    xyz: np.ndarray | None = field(default=None, repr=False, compare=False)
    calibration: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    gripper_max_opening_mm: float = 85.0


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ObjectReference(_FrozenModel):
    noun: str = Field(min_length=1)
    colors: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    ordinal: int | None = Field(default=None, ge=1)
    order_axis: Literal[
        "left_to_right",
        "right_to_left",
        "top_to_bottom",
        "bottom_to_top",
    ] | None = None
    extremum: Literal["leftmost", "rightmost", "topmost", "bottommost", "nearest", "farthest"] | None = None
    relation: Literal[
        "left_of", "right_of", "above", "below", "nearest_to", "farthest_from",
        "inside", "on_top_of", "under",
    ] | None = None
    relation_target: ObjectReference | None = None
    clear_top: bool = Field(default=False, strict=True)

    @field_validator("noun")
    @classmethod
    def validate_noun_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("noun must not be blank")
        return value

    @field_validator("colors", "attributes")
    @classmethod
    def normalize_terms(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                normalized
                for value in values
                if (normalized := " ".join(value.lower().split()))
            )
        )

    @model_validator(mode="after")
    def validate_spatial_qualifiers(self) -> ObjectReference:
        if (self.relation is None) != (self.relation_target is None):
            raise ValueError("relation and relation_target must be provided together")
        if (self.ordinal is None) != (self.order_axis is None):
            raise ValueError("ordinal and order_axis must be provided together")
        if self.extremum is not None and self.ordinal is not None:
            raise ValueError("extremum and ordinal are alternative unary selectors")
        return self


class TaskIntent(_FrozenModel):
    operation: Literal["pick", "place", "unsupported"]
    target: ObjectReference | None = None
    destination: ObjectReference | None = None
    placement_relation: Literal["under"] | None = None
    placement_anchor: ObjectReference | None = None
    user_constraints: tuple[str, ...] = ()
    quantity: Annotated[StrictInt, Field(ge=1)] | Literal["all"] | None = None
    negated: bool = Field(default=False, strict=True)
    exclusions: tuple[ObjectReference, ...] = ()
    rationale: str = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def rationale_must_have_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rationale must not be blank")
        return value

    @model_validator(mode="after")
    def validate_operation_arguments(self) -> TaskIntent:
        if self.operation in {"pick", "place"} and self.target is None:
            raise ValueError(f"{self.operation} requires a target")
        if self.operation == "place" and self.destination is None:
            raise ValueError("place requires a destination")
        if self.operation != "place" and self.destination is not None:
            raise ValueError("only place accepts a destination")
        if (self.placement_relation is None) != (self.placement_anchor is None):
            raise ValueError("placement relation and anchor must be provided together")
        if self.placement_relation is not None and self.operation != "place":
            raise ValueError("only place accepts a placement anchor")
        return self


EvidenceKind: TypeAlias = Literal[
    "operation", "noun", "color", "attribute", "quantity", "ordinal",
    "order_axis", "extremum", "relation", "destination", "negation", "conjunction", "exclusion",
    "clear_top",
]


class LiteralSpan(_FrozenModel):
    """Half-open Unicode character offsets into the unmodified instruction."""

    start: StrictInt = Field(ge=0)
    end: StrictInt = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_offsets(self) -> LiteralSpan:
        if self.start >= self.end:
            raise ValueError("literal span offsets must be ordered")
        return self


class LiteralEvidence(_FrozenModel):
    kind: EvidenceKind
    path: str = Field(min_length=1)
    span: LiteralSpan


class SourceTask(_FrozenModel):
    intent: TaskIntent
    source: LiteralSpan
    evidence: tuple[LiteralEvidence, ...] = Field(min_length=1)
    inherited_operation_from: StrictInt | None = Field(default=None, ge=0)


class SourceConjunction(_FrozenModel):
    kind: Literal["and", "then", "or", "but"]
    after_clause: StrictInt = Field(ge=0)
    span: LiteralSpan


class TaskProgram(_FrozenModel):
    """Source interpretation only; contains no grounding or executable actions."""

    clauses: tuple[SourceTask, ...] = Field(min_length=1, max_length=16)
    conjunctions: tuple[SourceConjunction, ...] = ()

    @model_validator(mode="after")
    def validate_clause_links(self) -> TaskProgram:
        links = [link.after_clause for link in self.conjunctions]
        if sorted(links) != list(range(len(self.clauses) - 1)):
            raise ValueError("every adjacent clause requires exactly one conjunction")
        for index, clause in enumerate(self.clauses):
            inherited = clause.inherited_operation_from
            if inherited is not None and (index == 0 or inherited != index - 1
                    or clause.intent.operation != self.clauses[inherited].intent.operation):
                raise ValueError("inherited operation must match the preceding clause")
        return self


class CoverageFinding(_FrozenModel):
    code: str
    kind: EvidenceKind | None = None
    span: LiteralSpan | None = None
    path: str | None = None
    message: str


@dataclass(frozen=True, slots=True)
class ResolutionFilterStage:
    """JSON-safe deterministic candidate evidence, including unsuccessful stages."""

    stage: str
    criterion: str
    hard: bool
    input_ids: tuple[str, ...]
    surviving_ids: tuple[str, ...]
    scores: tuple[tuple[str, float], ...] = ()
    scope: str = "reference"


@dataclass(frozen=True, slots=True)
class Resolved:
    object_id: str
    evidence: tuple[str, ...] = ()
    filter_stages: tuple[ResolutionFilterStage, ...] = ()


@dataclass(frozen=True, slots=True)
class Ambiguous:
    candidate_ids: tuple[str, ...]
    distinctions: tuple[str, ...]
    filter_stages: tuple[ResolutionFilterStage, ...] = ()


@dataclass(frozen=True, slots=True)
class Missing:
    reason: str
    filter_stages: tuple[ResolutionFilterStage, ...] = ()


@dataclass(frozen=True, slots=True)
class VerifiedAbsence(Missing):
    """Explicit absence evidence; a detector nonmatch is only Missing."""

    evidence: tuple[str, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        if not self.evidence or any(not item.strip() for item in self.evidence):
            raise ValueError("verified absence requires explicit evidence")


@dataclass(frozen=True, slots=True)
class Unsupported:
    reason: str
    filter_stages: tuple[ResolutionFilterStage, ...] = ()
    code: str = "unsupported_operation"


ResolutionOutcome: TypeAlias = Resolved | Ambiguous | Missing | Unsupported


@dataclass(frozen=True, slots=True)
class EmptyCandidateSet:
    """A complete candidate-set query that deterministically matched nothing."""

    filter_stages: tuple[ResolutionFilterStage, ...] = ()


CandidateSetOutcome: TypeAlias = ResolutionOutcome | EmptyCandidateSet


@dataclass(frozen=True, slots=True)
class ResolvedSelection:
    object_ids: tuple[str, ...]
    filter_stages: tuple[ResolutionFilterStage, ...] = ()


SelectionOutcome: TypeAlias = ResolvedSelection | Ambiguous | Missing | Unsupported


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    clause_index: int
    operation: Literal["pick", "place"]
    object_id: str
    destination_id: str | None = None


@dataclass(frozen=True, slots=True)
class ProgramDependency:
    before: int
    after: int
    evidence: str


@dataclass(frozen=True, slots=True)
class ResolvedConstraint:
    """Source instance and its actual resolver outcome, retained for physical duties."""

    clause_index: int
    instance_index: int
    source: str
    outcome: SelectionOutcome


@dataclass(frozen=True, slots=True)
class ResolvedProgram:
    """Runtime grounding of the source TaskProgram, never a model-authored IR."""

    selections: tuple[SelectionOutcome, ...]
    destinations: tuple[ResolutionOutcome | None, ...]
    steps: tuple[ResolvedStep, ...] = ()
    dependencies: tuple[ProgramDependency, ...] = ()
    failure: Ambiguous | Missing | Unsupported | None = None
    preparation_steps: tuple[ResolvedStep, ...] = ()
    constraints: tuple[ResolvedConstraint, ...] = ()
    anchors: tuple[ResolutionOutcome | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(self.anchors))


ObligationKind: TypeAlias = Literal[
    "selection_unique", "destination_unique", "anchor_unique", "grasp_feasible", "container_fits",
    "no_collateral_motion", "user_constraint_discharged",
]
ObligationStatus: TypeAlias = Literal["closed", "open", "unknown", "violated"]


@dataclass(frozen=True, slots=True)
class ObligationKey:
    """Stable identity, including repeated constraints within the same clause."""

    kind: ObligationKind
    clause_index: int
    step_index: int | None = None
    instance_index: int = 0


@dataclass(frozen=True, slots=True)
class ObligationEvidence:
    """Deterministic evidence, never a model-authored rationale."""

    source: Literal["resolver", "geometry", "metric", "perception", "source_text", "policy"]
    code: str
    detail: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if (not self.code.strip() or not self.detail.strip()
                or not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("obligation evidence requires code, detail and finite normalized confidence")


@dataclass(frozen=True, slots=True)
class Obligation:
    key: ObligationKey
    subject: str | None = None
    status: ObligationStatus = "open"
    evidence: tuple[ObligationEvidence, ...] = ()
    candidates: tuple[str, ...] = ()
    question: str | None = None
    unobtainable_default: Literal["block", "permit"] = "block"
    decisive: ObligationEvidence | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if any(not isinstance(item, ObligationEvidence) for item in self.evidence):
            raise ValueError("obligation evidence must contain evidence records")
        if self.status in {"closed", "violated"} and not self.evidence:
            raise ValueError("closed and violated obligations require positive evidence")
        if self.decisive is not None and self.decisive not in self.evidence:
            raise ValueError("decisive evidence must belong to the obligation evidence tuple")


@dataclass(frozen=True, slots=True)
class ObligationLedger:
    """Immutable obligation facts and pure status queries."""

    obligations: tuple[Obligation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "obligations", tuple(self.obligations))
        if len({item.key for item in self.obligations}) != len(self.obligations):
            raise ValueError("duplicate obligation keys")

    def open(self) -> tuple[Obligation, ...]:
        return tuple(item for item in self.obligations if item.status == "open")

    def unknown(self) -> tuple[Obligation, ...]:
        return tuple(item for item in self.obligations if item.status == "unknown")

    def violated(self) -> tuple[Obligation, ...]:
        return tuple(item for item in self.obligations if item.status == "violated")

    def blocking(self) -> tuple[Obligation, ...]:
        return tuple(item for item in self.obligations
                     if item.status == "unknown" and item.unobtainable_default == "block")

    def replace(self, obligation: Obligation) -> ObligationLedger:
        if not any(item.key == obligation.key for item in self.obligations):
            raise KeyError(obligation.key)
        return ObligationLedger(tuple(obligation if item.key == obligation.key else item
                                      for item in self.obligations))


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Content identity of source bindings and the root physical scene."""

    bindings_sha256: str
    scene_sha256: str


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluatedObligationLedger(ObligationLedger):
    """Capability returned by physical evaluation, rather than caller-authored facts."""

    context: EvaluationContext


@dataclass(frozen=True, slots=True)
class IntentResolution:
    operation: Literal["pick", "place", "unsupported"]
    target: ResolutionOutcome
    destination: ResolutionOutcome | None = None


@dataclass(frozen=True, slots=True)
class SafetyApproval:
    evidence: tuple[str, ...] = ()
    ledger: EvaluatedObligationLedger | None = None

    def __post_init__(self) -> None:
        if self.ledger is not None and not isinstance(self.ledger, EvaluatedObligationLedger):
            raise ValueError("Safety approval requires an evaluated obligation ledger")


@dataclass(frozen=True, slots=True)
class SafetyDenial:
    code: str
    reason: str


SafetyOutcome: TypeAlias = SafetyApproval | SafetyDenial


class PickAction(_FrozenModel):
    action: Literal["pick"] = "pick"
    object_id: str = Field(min_length=1)


class PlaceAction(_FrozenModel):
    action: Literal["place"] = "place"
    object_id: str = Field(min_length=1)


RobotAction: TypeAlias = Annotated[PickAction | PlaceAction, Field(discriminator="action")]


class PlanDecision(_FrozenModel):
    kind: Literal["plan"] = "plan"
    actions: tuple[RobotAction, ...] = Field(min_length=1)

    @field_validator("actions")
    @classmethod
    def first_action_must_be_pick(
        cls, actions: tuple[RobotAction, ...]
    ) -> tuple[RobotAction, ...]:
        if not isinstance(actions[0], PickAction):
            raise ValueError("first action must be pick")
        return actions


class AskUserDecision(_FrozenModel):
    kind: Literal["ask_user"] = "ask_user"
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_have_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class RefuseDecision(_FrozenModel):
    kind: Literal["refuse"] = "refuse"
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def reason_must_have_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reason must not be blank")
        return value.strip()


Decision: TypeAlias = Annotated[
    PlanDecision | AskUserDecision | RefuseDecision,
    Field(discriminator="kind"),
]


class ResultEntry(_FrozenModel):
    instruction_id: str = Field(min_length=1)
    target_pixel: tuple[StrictInt, StrictInt] | None
    decision: Decision

    @field_validator("target_pixel")
    @classmethod
    def pixel_must_be_non_negative(
        cls, value: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        if value is not None and (value[0] < 0 or value[1] < 0):
            raise ValueError("target pixel coordinates must be non-negative")
        return value


TelemetryStage: TypeAlias = Literal[
    "parse", "target_resolution", "destination_resolution", "relation_evidence",
    "safety", "compilation", "grounding",
]


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    message: str
    action_index: int | None = None
    object_id: str | None = None
    stage: TelemetryStage | None = None
    details: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            if not isinstance(self.details, dict):
                raise TypeError("details must be an object")
            details = json.loads(json.dumps(self.details, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("telemetry details must be finite JSON values") from exc
        object.__setattr__(self, "details", redact_telemetry(details))
        object.__setattr__(self, "message", redact_telemetry(self.message))
        object.__setattr__(self, "code", redact_telemetry(self.code))
        if self.object_id is not None:
            object.__setattr__(self, "object_id", redact_telemetry(self.object_id))


_TELEMETRY_SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
)
_TELEMETRY_SECRET_FIELDS = frozenset({
    "api_key", "api_token", "access_token", "authorization", "password",
    "model_tokens", "tokens",
})


def redact_telemetry(value: JsonValue) -> JsonValue:
    """Redact known credential formats throughout JSON telemetry values."""
    if isinstance(value, str):
        for pattern in _TELEMETRY_SECRET_PATTERNS:
            value = pattern.sub("[REDACTED]", value)
        return value
    if isinstance(value, list):
        return [redact_telemetry(item) for item in value]
    if isinstance(value, dict):
        return {
            redact_telemetry(key): (
                "[REDACTED]" if key.lower() in _TELEMETRY_SECRET_FIELDS
                else redact_telemetry(item)
            )
            for key, item in value.items()
        }
    return value

