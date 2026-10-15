"""Command-line boundaries for offline competition inference and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .data import iter_scene_instructions, load_challenge
from .telemetry import write_run_telemetry
from .pipeline import run_batch_with_timings
from .planner import ApertusPlanner, ProgramBackend
from .intent_codec import IntentDecodeError, build_intent_messages
from .intent_cache import (CachedProgramBackend, IntentCacheBackend, RuntimeArtifact,
                          executable_program, intent_extractor_contract, intent_fingerprint)
from .language import FLAT_COMMAND_SCHEMA
from .results import ResultValidationError, validate_results_document, write_results_atomic
from .vision import (
    LocalVisionBackend,
    OpenVocabularyVisionAdapter,
    VisionBackend,
    VisionObservationError,
)
from .vision_cache import (ComposedVisionCacheBackend, RecoveryCacheBackend, VisionCacheBackend, image_cache_key)
from .vision_cache_audit import audit_scene, write_cache_audit
from .vision_diagnostics import VisionDiagnosticStore
from .vision_provenance import recovery_pipeline_fingerprint, vision_pipeline_fingerprint
from .visual_evidence import enrich_visual_colors
from .perception_recovery import collect_reference_queries, recover_references, unmatched_reference_queries
from .metric_geometry import enrich_metric_geometry


def _read_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("offline") is not True:
        raise ValueError("config must set offline: true")
    return config


def _model_revisions(config: dict) -> dict[str, str]:
    try:
        return {
            "apertus": str(config["apertus"]["revision"]),
            "qwen": str(config["vision"]["inventory"]["revision"]),
            "sam": str(config["vision"]["segmentation"]["revision"]),
        }
    except KeyError as exc:
        raise ValueError(f"missing model configuration: {exc.args[0]}") from exc


def _load_live_vision_backend(
    config: dict, *, diagnostics_root: Path | None = None
) -> LocalVisionBackend:
    try:
        inventory = config["vision"]["inventory"]
        segmentation = config["vision"]["segmentation"]
        return LocalVisionBackend.from_pretrained(
            str(inventory["model_id"]),
            str(inventory["revision"]),
            str(segmentation["model_id"]),
            str(segmentation["revision"]),
            qwen_load_in_4bit=bool(inventory.get("load_in_4bit", False)),
            qwen_max_pixels=(
                int(inventory["max_pixels"])
                if inventory.get("max_pixels") is not None
                else None
            ),
            diagnostics_root=diagnostics_root,
        )
    except KeyError as exc:
        raise ValueError(f"missing model configuration: {exc.args[0]}") from exc


def _intent_cache(config: dict, root: Path) -> IntentCacheBackend:
    aperture = config["apertus"]
    return IntentCacheBackend(root, fingerprint=intent_fingerprint(
        str(aperture["model_id"]), str(aperture["revision"]),
        prompt=build_intent_messages(""), schema=FLAT_COMMAND_SCHEMA,
        extractor_contract=intent_extractor_contract(),
        runtime_artifact=(RuntimeArtifact.model_validate(aperture["runtime_artifact"])
                          if aperture.get("runtime_artifact") is not None else None),
    ))


def _load_program_backend(config: dict, *, intent_cache: Path | None = None) -> ProgramBackend:
    try:
        if intent_cache is not None:
            return CachedProgramBackend(_intent_cache(config, intent_cache))
        aperture = config["apertus"]
        base_url = str(aperture["base_url"])
        if urlparse(base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Apertus base_url must use a loopback host")
        return ApertusPlanner(str(aperture["model_id"]), base_url=base_url)
    except KeyError as exc:
        raise ValueError(f"missing model configuration: {exc.args[0]}") from exc


def _load_backends(
    config_path: Path, *, vision_cache: Path | None = None, intent_cache: Path | None = None,
    scene_programs=None,
) -> tuple[VisionBackend, ProgramBackend]:
    config = _read_config(config_path)
    recovery_enabled = config.get("perception_recovery", {}).get("enabled", False)
    if recovery_enabled and (vision_cache is None or intent_cache is None or scene_programs is None):
        raise ValueError("recovery inference requires --vision-cache and --intent-cache")
    planner = _load_program_backend(config, intent_cache=intent_cache)
    vision = (VisionCacheBackend(vision_cache, pipeline_fingerprint=vision_pipeline_fingerprint(config))
              if vision_cache is not None else _load_live_vision_backend(config))
    if recovery_enabled:
        vision = ComposedVisionCacheBackend(vision, _recovery_layers(
            config, vision_cache, intent_cache, scene_programs))
    return vision, planner


def _load_recovery_detector(
    config: dict,
    vision: object,
) -> OpenVocabularyVisionAdapter:
    settings = config["perception_recovery"]
    try:
        return OpenVocabularyVisionAdapter.from_pretrained(
            str(settings["model_id"]),
            str(settings["revision"]),
            vision.segment,
            min_confidence=float(settings["min_confidence"]),
        )
    except KeyError as exc:
        raise ValueError(f"missing perception recovery setting: {exc.args[0]}") from exc


def _scene_programs(config, instructions, intent_cache):
    cache = _intent_cache(config, intent_cache)
    grouped = {}
    for instruction in instructions:
        programs = grouped.setdefault(instruction.scene_id, [])
        entry = cache.read(instruction.text)
        if entry.program is not None and entry.program not in programs:
            programs.append(entry.program)
    return {scene_id: tuple(programs) for scene_id, programs in grouped.items()}


def _recovery_layers(config, root, intent_cache, scene_programs):
    # The mapping is keyed by image digest by the caller, never by instruction ID.
    parser = _intent_cache(config, intent_cache).fingerprint
    base = vision_pipeline_fingerprint(config)
    return {key: RecoveryCacheBackend(root, fingerprint=recovery_pipeline_fingerprint(config,
        base_fingerprint=base + ":" + hashlib.sha256((Path(root) / f"{key}.npz").read_bytes()).hexdigest(),
        queries=collect_reference_queries(programs), parser_fingerprint=parser))
        for key, programs in scene_programs.items()}


def _infer(args: argparse.Namespace) -> int:
    config = _read_config(args.config)
    scenes, instructions = load_challenge(args.dataset)
    scene_programs = None
    if config.get("perception_recovery", {}).get("enabled", False):
        if args.intent_cache is None:
            raise ValueError("perception recovery requires --intent-cache")
        grouped = _scene_programs(config, instructions, args.intent_cache)
        scene_programs = {image_cache_key(scene.rgb): grouped.get(scene.scene_id, ()) for scene in scenes}
    vision, planner = _load_backends(
        args.config,
        vision_cache=args.vision_cache,
        intent_cache=args.intent_cache,
        scene_programs=scene_programs,
    )
    entries, timings = run_batch_with_timings(
        iter_scene_instructions(args.dataset),
        vision,
        planner,
    )
    write_results_atomic(
        args.team,
        tuple(item.instruction_id for item in instructions),
        entries,
        args.output,
    )
    telemetry = args.telemetry or args.output.with_suffix(".telemetry.json")
    write_run_telemetry(timings, _model_revisions(config), telemetry)
    return 0


def _cache_vision(args: argparse.Namespace) -> int:
    config = _read_config(args.config)
    recovery_enabled = config.get("perception_recovery", {}).get("enabled", False)
    intent_root = getattr(args, "intent_cache", None)
    if recovery_enabled and intent_root is None:
        raise ValueError("perception recovery requires --intent-cache")
    scenes, instructions = load_challenge(args.dataset)
    if intent_root is not None:
        _intent_cache(config, intent_root).require_executable(item.text for item in instructions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        scenes = scenes[: args.limit]
    pipeline_fingerprint = vision_pipeline_fingerprint(config)
    cache = VisionCacheBackend(
        args.output,
        pipeline_fingerprint=pipeline_fingerprint,
    )
    grouped = _scene_programs(config, instructions, intent_root) if recovery_enabled else {}
    programs_by_image = {image_cache_key(scene.rgb): grouped.get(scene.scene_id, ()) for scene in scenes}
    diagnostics = VisionDiagnosticStore(args.diagnostics)
    probes = []
    for scene in scenes:
        key = image_cache_key(scene.rgb)
        diagnostics.write_scene_manifest(scene.scene_id, key)
        probe = cache.probe(scene.rgb, scene=scene)
        if probe.status == "corrupt":
            cache.quarantine(scene.rgb, args.diagnostics)
        probes.append(probe)

    vision = detector = None
    if any(probe.status != "valid" for probe in probes):
        vision = _load_live_vision_backend(
            config, diagnostics_root=args.diagnostics
        )

    audit_by_scene_id = {
        scene.scene_id: audit_scene(
            scene.scene_id,
            image_cache_key(scene.rgb),
            probe.observation,
        )
        for scene, probe in zip(scenes, probes, strict=True)
        if probe.status == "valid" and probe.observation is not None
    }

    def current_audit() -> list[dict[str, object]]:
        return [
            audit_by_scene_id[scene.scene_id]
            for scene in scenes
            if scene.scene_id in audit_by_scene_id
        ]

    audit_path = args.diagnostics / "vision-cache-audit.json"
    write_cache_audit(audit_path, current_audit(), pipeline_fingerprint)
    for index, (scene, probe) in enumerate(zip(scenes, probes, strict=True), start=1):
        if probe.status == "valid":
            assert probe.observation is not None
            observation = probe.observation
            status = "cached"
            destination = args.output / f"{image_cache_key(scene.rgb)}.npz"
        else:
            assert vision is not None
            observation = vision.observe_with_diagnostics(scene.rgb)
            if not observation.objects:
                raise VisionObservationError(observation)
            observation = replace(observation, objects=enrich_visual_colors(scene.rgb, observation.objects))
            observation = replace(observation, objects=enrich_metric_geometry(scene, observation.objects))
            destination = cache.store(scene.rgb, observation, scene=scene)
            status = (
                "recomputed"
                if probe.status in {"corrupt", "incompatible"}
                else "computed"
            )
        reference_recovery = None
        if recovery_enabled:
            key = image_cache_key(scene.rgb)
            layer = _recovery_layers(config, args.output, intent_root, {key: programs_by_image[key]})[key]
            if layer.probe(scene.rgb).status != "valid":
                queries = collect_reference_queries(programs_by_image[key])
                if unmatched_reference_queries(observation, queries):
                    if vision is None:
                        vision = _load_live_vision_backend(config, diagnostics_root=args.diagnostics)
                    if detector is None:
                        detector = _load_recovery_detector(config, vision)
                reference_recovery = recover_references(scene.rgb, observation, queries, detector)
                reference_recovery = replace(reference_recovery,
                    objects=enrich_metric_geometry(scene, reference_recovery.objects))
                layer.store(scene.rgb, reference_recovery)
            else:
                reference_recovery = layer.read(scene.rgb)
        audit_by_scene_id[scene.scene_id] = audit_scene(
            scene.scene_id,
            image_cache_key(scene.rgb),
            observation,
            reference_recovery=reference_recovery,
        )
        write_cache_audit(audit_path, current_audit(), pipeline_fingerprint)
        print(
            f"[{index}/{len(scenes)}] {scene.scene_id}: "
            f"{status} {len(observation.objects)} objects, "
            f"{len(observation.dropped)} dropped -> {destination}",
            flush=True,
        )
    return 0


def _cache_intents(args: argparse.Namespace) -> int:
    config = _read_config(args.config)
    _, instructions = load_challenge(args.dataset)
    cache = _intent_cache(config, args.output)
    counts = dict.fromkeys(("literal", "model", "repaired", "failed"), 0)
    executable = 0
    planner = None
    for text in dict.fromkeys(item.text for item in instructions):
        if cache.contains(text):
            entry = cache.read(text)
        else:
            if planner is None:
                planner = _load_program_backend(config)
            try:
                program = planner.interpret_program(text)
            except IntentDecodeError as exc:
                cache.store_failure(text, exc)
            else:
                cache.store_success(text, program, outcome=planner.last_outcome)
            entry = cache.read(text)
        counts[entry.outcome] += 1
        executable += executable_program(text, entry.program)
    print("Intent cache: " + " ".join(f"{key}={value}" for key, value in counts.items())
          + f" executable={executable}", flush=True)
    cache.require_executable(item.text for item in instructions)
    return 0


def _preflight_intents(args: argparse.Namespace) -> int:
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    planner = _load_program_backend(_read_config(args.config), intent_cache=args.intent_cache)
    checked, executable = 0, 0
    for _, instruction in iter_scene_instructions(args.dataset):
        try:
            program = planner.interpret_program(instruction.text)
        except IntentDecodeError as exc:
            summary = "failed=" + ",".join(finding.code for finding in exc.findings)
        else:
            operations = ",".join(clause.intent.operation for clause in program.clauses)
            summary = f"clauses={len(program.clauses)} operations={operations}"
            executable += executable_program(instruction.text, program)
        checked += 1
        print(f"[{checked}/{args.limit}] {instruction.instruction_id}: {summary}", flush=True)
        if checked == args.limit:
            break
    if checked < args.limit:
        raise ValueError(f"dataset contains only {checked} instructions")
    if executable == 0:
        raise ValueError("planner preflight found no executable instructions")
    print(f"Planner preflight passed: {executable} intents executable, {checked - executable} failed or unsupported",
          flush=True)
    return 0


def _validate(args: argparse.Namespace) -> int:
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    validate_results_document(payload)
    if args.dataset is not None:
        _, instructions = load_challenge(args.dataset)
        expected = [item.instruction_id for item in instructions]
        actual = [item["instruction_id"] for item in payload["results"]]
        if actual != expected:
            raise ResultValidationError(
                "result instruction IDs do not exactly match dataset source order"
            )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    try:
        from .evaluate import (
            FailureDiagnostic, read_run_telemetry, run_official_evaluation,
            write_failure_diagnostics,
        )
    except ModuleNotFoundError as exc:
        raise ValueError("Evaluation tooling is excluded from the Colab release; use the full repository outside inference") from exc
    summary = run_official_evaluation(args.results, args.dataset, args.report)
    print(summary.stdout, end="" if summary.stdout.endswith("\n") else "\n")
    if summary.invalid_entries:
        raise ResultValidationError(
            f"official evaluator reported {summary.invalid_entries} invalid entries"
        )
    report = json.loads(args.report.read_text(encoding="utf-8"))
    timings = {}
    model_revisions = {}
    if args.telemetry is not None:
        timings, model_revisions = read_run_telemetry(args.telemetry)

    def details_for(row: dict) -> str:
        parts = [str(row.get("error") or row.get("robot_check") or "").strip()]
        timing = timings.get(str(row["instruction_id"]))
        if timing is not None:
            for event in timing.events:
                context = []
                if event.object_id is not None:
                    context.append(f"object={event.object_id}")
                if event.action_index is not None:
                    context.append(f"action={event.action_index}")
                suffix = f" [{', '.join(context)}]" if context else ""
                parts.append(f"{event.code}: {event.message}{suffix}")
        return "; ".join(part for part in parts if part)

    records = tuple(
        FailureDiagnostic(
            instruction_id=str(row["instruction_id"]),
            category=str(row.get("category") or "unknown"),
            decision_kind=str(row.get("answer") or "invalid"),
            grounding_failed=row.get("expected") == "plan" and row.get("score") != 1.0,
            refusal_failed=row.get("expected") != "plan" and row.get("score") != 1.0,
            latency_ms=(
                timings[str(row["instruction_id"])].total_ms
                if str(row["instruction_id"]) in timings
                else 0.0
            ),
            details=details_for(row),
        )
        for row in report.get("instructions", [])
    )
    write_failure_diagnostics(records, model_revisions, args.diagnostics)
    return 0


def _benchmark_localizer(args: argparse.Namespace) -> int:
    try:
        from .evaluate import write_localization_benchmark
    except ModuleNotFoundError as exc:
        raise ValueError("Evaluation tooling is excluded from the Colab release; use the full repository outside inference") from exc
    write_localization_benchmark(
        args.annotations, args.base, args.candidate, args.output
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="see-pick")
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer = subparsers.add_parser("infer", help="run the local competition pipeline")
    infer.add_argument("--dataset", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--team", required=True)
    infer.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    infer.add_argument("--telemetry", type=Path)
    infer.add_argument("--vision-cache", type=Path)
    infer.add_argument("--intent-cache", type=Path)
    infer.set_defaults(handler=_infer)

    cache_vision = subparsers.add_parser(
        "cache-vision", help="run local vision once per scene and persist observations"
    )
    cache_vision.add_argument("--dataset", type=Path, required=True)
    cache_vision.add_argument("--output", type=Path, required=True)
    cache_vision.add_argument(
        "--config", type=Path, default=Path("configs/models.yaml")
    )
    cache_vision.add_argument(
        "--diagnostics",
        type=Path,
        default=Path("artifacts/vision-diagnostics"),
    )
    cache_vision.add_argument("--limit", type=int)
    cache_vision.add_argument("--intent-cache", type=Path)
    cache_vision.set_defaults(handler=_cache_vision)

    cache_intents = subparsers.add_parser(
        "cache-intents", help="interpret source instructions and persist validated programs"
    )
    cache_intents.add_argument("--dataset", type=Path, required=True)
    cache_intents.add_argument("--output", type=Path, required=True)
    cache_intents.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    cache_intents.set_defaults(handler=_cache_intents)

    preflight = subparsers.add_parser(
        "preflight-intents", help="validate Apertus structured intents before inference"
    )
    preflight.add_argument("--dataset", type=Path, required=True)
    preflight.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    preflight.add_argument("--intent-cache", type=Path)
    preflight.add_argument("--limit", type=int, default=5)
    preflight.set_defaults(handler=_preflight_intents)

    validate = subparsers.add_parser("validate", help="validate a result document")
    validate.add_argument("--results", type=Path, required=True)
    validate.add_argument("--dataset", type=Path)
    validate.set_defaults(handler=_validate)

    evaluate = subparsers.add_parser("evaluate", help="run official scoring")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--results", type=Path, required=True)
    evaluate.add_argument("--report", type=Path, default=Path("artifacts/report.json"))
    evaluate.add_argument(
        "--diagnostics", type=Path, default=Path("artifacts/diagnostics.json")
    )
    evaluate.add_argument("--telemetry", type=Path)
    evaluate.set_defaults(handler=_evaluate)

    benchmark = subparsers.add_parser(
        "benchmark-localizer",
        help="compare input-only localization predictions with separate annotations",
    )
    benchmark.add_argument("--annotations", type=Path, required=True)
    benchmark.add_argument("--base", type=Path, required=True)
    benchmark.add_argument("--candidate", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.set_defaults(handler=_benchmark_localizer)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
