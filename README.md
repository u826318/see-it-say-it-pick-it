# See It, Say It, Pick It

Hack Apertus 2026, Track 2A.

This repository contains the local inference pipeline and the required submission artifacts. Apertus v1.5 8B performs language reasoning. Qwen2.5-VL, SAM 2.1 and Grounding DINO provide local visual perception. No proprietary model or external inference API is used.

## Submission files

- `results.json` - public development-set output
- `results_heldout.json` - held-out output
- `report/technical-report.pdf` - two-page technical report
- `slides/presentation.pdf` - five-slide presentation
- `colab/Hack_Apertus_2A_Heldout_Colab.ipynb` - reproducible A100 workflow
- `colab/runtime-source.zip` - exact source archive used for the held-out run

`SOURCE_COMMIT.txt` records the source revision used to produce the held-out results.

## Run in Colab

1. Open `colab/Hack_Apertus_2A_Heldout_Colab.ipynb` in Google Colab.
2. Select an A100 GPU runtime.
3. Upload `colab/runtime-source.zip` when prompted.
4. Upload an input-only dataset ZIP containing `instructions.csv`, `calibration.json` and the referenced scene files.
5. Add a Hugging Face read token when prompted and run all cells.

The notebook validates the dataset, runs all model stages locally, validates the official result schema and exports the result ZIP.

To create an input-only dataset ZIP from an arbitrary challenge-format scene folder:

```bash
python scripts/build_dataset_archive.py --dataset /path/to/scenes
```

## Local CLI

Install the package:

```bash
python -m pip install -e .
```

The staged pipeline uses the following commands:

```bash
python -m see_pick.cli cache-intents --dataset DATASET --output INTENT_CACHE --config configs/models.yaml
python -m see_pick.cli cache-vision --dataset DATASET --output VISION_CACHE --intent-cache INTENT_CACHE --config configs/models.yaml
python -m see_pick.cli infer --dataset DATASET --output results.json --team "See It, Say It, Pick It" --vision-cache VISION_CACHE --intent-cache INTENT_CACHE --config configs/models.yaml
python -m see_pick.cli validate --results results.json --dataset DATASET
```

The configured Apertus endpoint and pinned model artifacts must be available before running the local CLI. The Colab notebook performs that setup automatically.

## License

Code is licensed under Apache-2.0. Model weights are not included and retain their respective licenses.
