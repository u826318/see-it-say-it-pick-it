"""Scene-blind Apertus transport for flat language proposals."""

from __future__ import annotations

from typing import Protocol

import httpx

from .domain import TaskProgram
from .intent_codec import IntentDecodeError, build_intent_messages, decode_flat_command
from .language import FLAT_COMMAND_SCHEMA, build_task_program


class ProgramBackend(Protocol):
    def interpret_program(self, instruction: str) -> TaskProgram: ...


class ApertusPlanner:
    def __init__(self, model: str, *, base_url: str = "http://127.0.0.1:8000/v1",
                 client: httpx.Client | None = None, timeout: float = 120.0) -> None:
        self.model = model
        self.last_outcome = "failed"
        self.client = client or httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout)

    def interpret_program(self, instruction: str) -> TaskProgram:
        feedback = ()
        self.last_outcome = "failed"
        for attempt in range(2):
            model_used = False

            def parse_clause(text):
                nonlocal model_used
                model_used = True
                offset = len(instruction) - len(text)
                local_feedback = tuple(finding.model_copy(update={"span": (
                    finding.span.model_copy(update={"start": finding.span.start - offset,
                                                    "end": finding.span.end - offset})
                    if finding.span is not None and finding.span.start >= offset else None)
                }) for finding in feedback)
                response = self.client.post("chat/completions", json={
                    "model": self.model, "temperature": 0, "max_tokens": 4096,
                    "response_format": {"type": "json_object", "schema": FLAT_COMMAND_SCHEMA},
                    "messages": build_intent_messages(text, local_feedback),
                })
                response.raise_for_status()
                try:
                    content = response.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise IntentDecodeError("invalid completion envelope") from exc
                if not isinstance(content, str):
                    raise IntentDecodeError("completion content must be text")
                return decode_flat_command(content)

            try:
                program = build_task_program(instruction, parse_clause)
            except IntentDecodeError as exc:
                feedback = exc.findings
            else:
                self.last_outcome = "repaired" if attempt else "model" if model_used else "literal"
                return program
        raise IntentDecodeError("Apertus task program remained invalid after one repair", feedback)
