"""Package only source instructions and scene inputs, never evaluator assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_colab_archive import ROOT, _write, verify_manifest


def build_dataset_archive(dataset: Path, output: Path) -> Path:
    dataset = dataset.resolve()
    with (dataset / "instructions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Dataset has no instructions")
    scene_names = []
    ids = set()
    for row in rows:
        scene = row["scene"]
        path = PurePosixPath(scene)
        if (not scene or path.is_absolute() or ".." in path.parts or "\\" in scene
            or path.as_posix() != scene or not (dataset / scene).resolve().is_relative_to(dataset)
            or {"ground_truth", "masks", "labels", "answers", "expected"}.intersection(path.parts)):
            raise ValueError(f"Unsafe scene path: {scene}")
        if not row["instruction_id"] or row["instruction_id"] in ids or not row["instruction"].strip():
            raise ValueError("Blank or duplicate instruction input")
        ids.add(row["instruction_id"])
        if scene not in scene_names:
            scene_names.append(scene)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=("instruction_id", "scene", "instruction"), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    payloads = {"instructions.csv": csv_buffer.getvalue().encode("utf-8")}
    selected = [dataset / "calibration.json"]
    for name in scene_names:
        folder = dataset / name
        selected.append(folder / "rgb.png")
        selected.extend(folder / asset for asset in ("depth.png", "xyz.npy", "calibration.json")
                        if (folder / asset).is_file())
    for source in selected:
        if not source.resolve().is_relative_to(dataset):
            raise ValueError(f"Scene input escapes dataset: {source}")
        if source.name == "calibration.json":
            calibration = json.loads(source.read_text(encoding="utf-8"))
            value = {"camera": {"intrinsics": {"K": calibration["camera"]["intrinsics"]["K"]}}}
            value.update({key: calibration[key] for key in ("base_T_camera", "gripper_max_opening_mm")
                          if key in calibration})
            content = json.dumps(value, sort_keys=True).encode("utf-8") + b"\n"
        else:
            content = source.read_bytes()
        payloads[source.relative_to(dataset).as_posix()] = content
    output = output.resolve()
    if output.is_relative_to(dataset):
        raise ValueError("Output archive must be outside the dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w") as archive:
        for name, content in sorted(payloads.items()):
            _write(archive, name, content)
        _write(archive, "DATASET_MANIFEST.json", json.dumps({
            "version": 1, "sha256": {name: hashlib.sha256(content).hexdigest()
                                     for name, content in sorted(payloads.items())}
        }, indent=2).encode("utf-8") + b"\n")
    verify_manifest(output, "DATASET_MANIFEST.json")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/see-it-say-it-pick-it-dataset-inputs.zip")
    args = parser.parse_args()
    print(build_dataset_archive(args.dataset, args.output))


if __name__ == "__main__":
    main()
