"""Production challenge loading with no access to evaluation answers."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from starter.scene_io import load_scene as load_official_scene

from .domain import Instruction, Scene


def _read_instruction_rows(root: Path) -> list[dict[str, str]]:
    path = root / "instructions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"instruction_id", "scene", "instruction"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path}: expected columns {sorted(required)}")
    return rows


def load_challenge(root: Path) -> tuple[tuple[Scene, ...], tuple[Instruction, ...]]:
    """Load scenes and instructions in source CSV order."""

    root = Path(root)
    rows = _read_instruction_rows(root)
    scenes_by_id: dict[str, Scene] = {}
    instructions: list[Instruction] = []

    for row in rows:
        scene_id = row["scene"].strip()
        instruction_id = row["instruction_id"].strip()
        text = row["instruction"].strip()
        category = (row.get("category") or "").strip() or None
        if not scene_id or not instruction_id or not text:
            raise ValueError(f"{root / 'instructions.csv'}: blank required field")

        if scene_id not in scenes_by_id:
            folder = root / scene_id
            rgb_path = folder / "rgb.png"
            if not rgb_path.is_file():
                raise FileNotFoundError(rgb_path)
            loaded = load_official_scene(folder)
            scenes_by_id[scene_id] = Scene(
                scene_id=scene_id,
                rgb=loaded.rgb,
                depth_mm=loaded.depth_mm,
                xyz=loaded.xyz,
                calibration={
                    "K": loaded.K.tolist(),
                    "base_T_camera": (
                        loaded.base_T_camera.tolist()
                        if loaded.base_T_camera is not None
                        else None
                    ),
                },
                gripper_max_opening_mm=float(
                    loaded.gripper_max_opening_mm
                    if loaded.gripper_max_opening_mm is not None
                    else 85.0
                ),
            )

        instructions.append(
            Instruction(
                instruction_id=instruction_id,
                scene_id=scene_id,
                text=text,
                category=category,
            )
        )

    return tuple(scenes_by_id.values()), tuple(instructions)


def iter_scene_instructions(root: Path) -> Iterator[tuple[Scene, Instruction]]:
    scenes, instructions = load_challenge(root)
    by_id = {scene.scene_id: scene for scene in scenes}
    for instruction in instructions:
        yield by_id[instruction.scene_id], instruction
