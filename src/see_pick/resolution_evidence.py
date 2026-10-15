"""Shared hard-closure queries for scene identity evidence."""

from .domain import ResolutionFilterStage, Resolved, ResolvedSelection


def soft_narrowing_stage(stages: tuple[ResolutionFilterStage, ...]) -> ResolutionFilterStage | None:
    return next((stage for stage in stages
                 if not stage.hard and len(stage.surviving_ids) < len(stage.input_ids)), None)


def hard_resolution_closed(outcome: object = None, *, stages: tuple[ResolutionFilterStage, ...] | None = None) -> bool:
    """Query an identity outcome or the scoped stages of an already bound nested ID."""
    if stages is not None:
        return soft_narrowing_stage(stages) is None
    if not isinstance(outcome, (Resolved, ResolvedSelection)):
        return False
    if not hard_resolution_closed(stages=outcome.filter_stages):
        return False
    # A complete negative query may prove an empty prohibited set, but cannot
    # bind any object IDs into discourse. Other empty selections lack closure.
    return (isinstance(outcome, Resolved) or bool(outcome.object_ids)
            or any(stage.stage == "negative_set" and stage.hard for stage in outcome.filter_stages))
