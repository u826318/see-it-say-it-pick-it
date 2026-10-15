"""Validated source programs, cached independently from visual observations."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from . import canonicalization, intent_codec, language
from .domain import CoverageFinding, TaskProgram
from .intent_codec import IntentDecodeError, build_intent_messages, decode_program


SuccessOutcome = Literal["literal", "model", "repaired"]


def intent_extractor_contract() -> dict:
    """Track closed vocabulary and code that owns source binding and marker roles."""
    return {
        "language": inspect.getsource(language),
        "coverage": inspect.getsource(intent_codec),
        "lexicons": language.LEXICONS,
        "deictics": sorted(language.DEICTICS),
        "no_movement": [language.NO_MOVEMENT.pattern, language.NO_MOVEMENT.flags],
        "placement_markers": inspect.getsource(language._placement_markers),
        "noun_decomposition": {
            "decompose_noun": inspect.getsource(intent_codec.decompose_noun),
            "helpers": {name: inspect.getsource(getattr(canonicalization, name))
                        for name in ("NounParts", "_singular_head", "normalize_colors", "_words")},
            "word_tokens": [canonicalization._WORD_TOKENS.pattern, canonicalization._WORD_TOKENS.flags],
            "colors": sorted(canonicalization.COLORS),
            "head_synonyms": dict(canonicalization.HEAD_SYNONYMS),
        },
    }


class RuntimeArtifact(BaseModel):
    """Exact served weights, independent of their upstream model identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_min_length=1)
    format: str
    repo_id: str
    revision: str
    filename: str
    quantization: str


def intent_manifest(model_id: str, revision: str, *, prompt: object,
                    schema: dict, extractor_contract: object,
                    runtime_artifact: RuntimeArtifact | None = None) -> dict:
    return {"model_id": model_id, "revision": revision, "prompt": prompt,
                "runtime_artifact": runtime_artifact.model_dump(mode="json") if runtime_artifact else None,
                "schema": schema, "extractor_contract": extractor_contract,
                "generation": {"temperature": 0, "max_tokens": 4096, "repairs": 1}}


def intent_fingerprint(model_id: str, revision: str, *, prompt: object,
                       schema: dict, extractor_contract: object,
                       runtime_artifact: RuntimeArtifact | None = None) -> str:
    contract = intent_manifest(model_id, revision, prompt=prompt, schema=schema,
                               extractor_contract=extractor_contract, runtime_artifact=runtime_artifact)
    return hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def executable_program(instruction: str, program: TaskProgram | None) -> bool:
    """Stage authorized programs whose deictics have source-local noun bindings."""
    from .request_policy import derive_request_policy
    if program is None or derive_request_policy(instruction, program).disposition != "allow":
        return False
    return language.source_references_bound(program)


class IntentCacheEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    fingerprint: str
    instruction_digest: str
    outcome: Literal["literal", "model", "repaired", "failed"]
    program: TaskProgram | None = None
    findings: tuple[CoverageFinding, ...] = ()

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.outcome == "failed":
            if self.program is not None or not self.findings:
                raise ValueError("failed intent needs diagnostics and no program")
        elif self.program is None or self.findings:
            raise ValueError("successful intent needs a program and no diagnostics")
        return self


class IntentCacheBackend:
    def __init__(self, root: Path, *, fingerprint: str):
        if not fingerprint:
            raise ValueError("intent fingerprint must not be empty")
        self.root = Path(root) / "intents"
        self.fingerprint = fingerprint

    @staticmethod
    def _key(instruction: str) -> str:
        return hashlib.sha256(instruction.encode("utf-8")).hexdigest()

    def _store(self, instruction: str, **payload) -> Path:
        entry = IntentCacheEntry(fingerprint=self.fingerprint,
                                 instruction_digest=self._key(instruction), **payload)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / f"{entry.instruction_digest}.json"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                             dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(entry.model_dump_json() + "\n")
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    def store_success(self, instruction: str, program: TaskProgram, *,
                      outcome: SuccessOutcome = "model") -> Path:
        validated = decode_program(program.model_dump_json(), instruction)
        return self._store(instruction, program=validated, outcome=outcome)

    def store_failure(self, instruction: str, failure: IntentDecodeError) -> Path:
        safe = json.loads(build_intent_messages(instruction, failure.findings)[-1]["content"])
        return self._store(instruction, outcome="failed", findings=safe["coverage_findings"])

    def read(self, instruction: str) -> IntentCacheEntry:
        key = self._key(instruction)
        entry = IntentCacheEntry.model_validate_json(
            (self.root / f"{key}.json").read_text(encoding="utf-8"))
        if entry.fingerprint != self.fingerprint or entry.instruction_digest != key:
            raise ValueError("incompatible intent cache contract or instruction digest")
        if entry.program is not None:
            decode_program(entry.program.model_dump_json(), instruction)
        else:
            safe = json.loads(build_intent_messages(instruction, entry.findings)[-1]["content"])
            if safe["coverage_findings"] != [f.model_dump(mode="json") for f in entry.findings]:
                raise ValueError("invalid cached intent diagnostics")
        return entry

    def contains(self, instruction: str) -> bool:
        try:
            self.read(instruction)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def require_executable(self, instructions: Iterable[str]) -> int:
        """Validate every source-bound entry, retaining individual typed failures."""
        count = sum(executable_program(text, self.read(text).program) for text in dict.fromkeys(instructions))
        if count == 0:
            raise ValueError("intent cache contains no executable instructions; diagnostics retained")
        return count

    def require_complete(self, instructions: Iterable[str]) -> int:
        """Reject parser failures before expensive downstream model work."""
        unique = tuple(dict.fromkeys(instructions))
        executable = self.require_executable(unique)
        failed = sum(self.read(text).program is None for text in unique)
        if failed:
            raise ValueError(f"intent cache contains {failed} failed intent(s); diagnostics retained")
        return executable


class CachedProgramBackend:
    def __init__(self, cache: IntentCacheBackend):
        self.cache = cache

    def interpret_program(self, instruction: str) -> TaskProgram:
        entry = self.cache.read(instruction)
        if entry.program is None:
            raise IntentDecodeError("cached intent failure", entry.findings)
        return entry.program
