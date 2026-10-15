"""Runtime-only telemetry serialization, independent of evaluation tooling."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .domain import redact_telemetry
if TYPE_CHECKING:
    from .pipeline import PipelineTiming


def write_run_telemetry(
    timings: tuple[PipelineTiming, ...],
    model_revisions: dict[str, str],
    destination: Path,
) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = redact_telemetry({
        "model_revisions": dict(sorted(model_revisions.items())),
        "instructions": [asdict(timing) for timing in timings],
    })
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return destination
