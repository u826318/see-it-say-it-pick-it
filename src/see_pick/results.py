"""Conversion, validation, and atomic writing of competition results."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .domain import AskUserDecision, PickAction, PlanDecision, RefuseDecision, ResultEntry


class ResultValidationError(ValueError):
    """Raised before any invalid submission document can be written."""


def result_entry_to_official(entry: ResultEntry) -> dict[str, Any]:
    decision = entry.decision
    if isinstance(decision, PlanDecision):
        actions = [
            {"action": action.action, "object": action.object_id}
            for action in decision.actions
        ]
    elif isinstance(decision, AskUserDecision):
        actions = [{"action": "ask_user", "question": decision.question}]
    elif isinstance(decision, RefuseDecision):
        actions = [{"action": "refuse", "reason": decision.reason}]
    else:  # pragma: no cover - closed domain union, defensive at serialization boundary
        raise ResultValidationError(f"unsupported decision type: {type(decision).__name__}")

    return {
        "instruction_id": entry.instruction_id,
        "target_pixel": list(entry.target_pixel) if entry.target_pixel is not None else None,
        "actions": actions,
    }


def _schema_text() -> str:
    path = Path(__file__).resolve().parents[2] / "schema" / "result.schema.json"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    packaged = resources.files("see_pick").joinpath("result.schema.json")
    try:
        return packaged.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ResultValidationError("packaged official result schema not found") from exc


def validate_results_document(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ResultValidationError("results document must be an object")
    if set(payload) != {"team", "results"}:
        raise ResultValidationError("results document must contain only team and results")
    if not isinstance(payload["team"], str) or not payload["team"].strip():
        raise ResultValidationError("team must be a non-empty string")
    if not isinstance(payload["results"], list):
        raise ResultValidationError("results must be a list")

    schema = json.loads(_schema_text())
    validator = Draft202012Validator(schema)
    for index, entry in enumerate(payload["results"]):
        errors = sorted(validator.iter_errors(entry), key=lambda error: list(error.path))
        if errors:
            raise ResultValidationError(f"result {index}: {errors[0].message}")
        action_kinds = [action["action"] for action in entry["actions"]]
        terminal = [kind for kind in action_kinds if kind in {"ask_user", "refuse"}]
        if terminal:
            if len(action_kinds) != 1:
                raise ResultValidationError(
                    f"result {index}: ask_user or refuse must be the only action"
                )
            if entry["target_pixel"] is not None:
                raise ResultValidationError(
                    f"result {index}: ask_user/refuse requires null target_pixel"
                )
            continue

        is_plan = all(kind in {"pick", "place"} for kind in action_kinds)
        if not is_plan:
            raise ResultValidationError(f"result {index}: invalid action sequence")
        if action_kinds[0] != "pick":
            raise ResultValidationError(f"result {index}: plan must start with pick")
        if is_plan and entry["target_pixel"] is None:
            raise ResultValidationError(f"result {index}: plan requires target_pixel")


def _ordered_entries(
    expected_ids: Sequence[str], entries: Sequence[ResultEntry]
) -> list[ResultEntry]:
    expected = list(expected_ids)
    if len(set(expected)) != len(expected):
        raise ResultValidationError("duplicate expected instruction ID")
    counts = Counter(entry.instruction_id for entry in entries)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise ResultValidationError(f"duplicate result ID: {duplicates[0]}")

    expected_set = set(expected)
    actual_set = set(counts)
    unknown = sorted(actual_set - expected_set)
    missing = sorted(expected_set - actual_set)
    if unknown:
        raise ResultValidationError(f"unknown result ID: {unknown[0]}")
    if missing:
        raise ResultValidationError(f"missing result ID: {missing[0]}")

    by_id = {entry.instruction_id: entry for entry in entries}
    return [by_id[instruction_id] for instruction_id in expected]


def write_results_atomic(
    team: str,
    expected_ids: Sequence[str],
    entries: Sequence[ResultEntry],
    destination: Path,
) -> None:
    """Validate the complete batch, then atomically replace ``destination``."""

    ordered = _ordered_entries(expected_ids, entries)
    payload = {
        "team": team,
        "results": [result_entry_to_official(entry) for entry in ordered],
    }
    validate_results_document(payload)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
