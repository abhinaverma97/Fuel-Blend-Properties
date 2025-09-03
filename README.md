# Fuel Blend Properties — Shell Hackathon

Project scripts and notebooks for predicting fuel blend properties (BlendProperty1..BlendProperty10). This repository contains the training and inference scripts used during a Shell AI Hackathon entry (placed top 18 out of ~5500). The README below explains the project layout, how to run the scripts locally, and helpful notes about environment, GPU flags, and reproducibility.

## Repository structure

- `dataset/` — CSVs used for training and testing (`train.csv`, `test.csv`, `sample_solution.csv`)
- `src/` — main training/inference scripts
  - `single-output.py` — per-target OOF stacking pipeline (stacking base learners and a meta-learner), feature engineering, feature selection, saves submissions and OOFs
  - `multi-output.py` — multi-output modeling utilities and pipelines (feature selection, multi-output regressors)
- `reports/` — EDA and notes (`full.txt`)
- `workflow.png` — process diagram
- `requirements.txt` — Python dependencies
- `outputs/` — directory where scripts write submission and OOF CSVs (create if missing)

## Quick notes
- All training scripts expect to be run from the repository root.
- Scripts write to `outputs/` (relative to `src/` usage). Create the `outputs/` directory first if it doesn't exist.
- GPU-accelerated learners are used in the scripts (XGBoost, LightGBM, CatBoost). If you do not have a GPU, either run on CPU-only by switching device flags in the scripts, or install CPU-only versions of the libraries.

## Requirements

Install dependencies from `requirements.txt` into a virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Prepare outputs dir

Run from the repository root (PowerShell):

```powershell
# create outputs directory if missing
New-Item -ItemType Directory -Path .\outputs -Force
```

## Run training / generate predictions

- Single-target stacking pipeline (per-target feature selection + stacking):

```powershell
# from repo root
python .\src\single-output.py
```

- Multi-output pipeline (if you want to run the multi-target script):

```powershell
# from repo root
python .\src\multi-output.py
```

Notes:
- Scripts expect `dataset/` files present (`train.csv`, `test.csv`). They will save CSVs into `outputs/` such as `single-output.csv` and `oof-single-output.csv`.
- If your environment has no GPU, edit the model definitions in `src/*.py` to remove GPU-specific args (for example, remove `device='gpu'`, `task_type='GPU'`, or change `tree_method='gpu_hist'`).

## Reproducibility & tips
- The scripts set random seeds for reproducibility (where supported). For full determinism you may need to set additional library-specific flags and ensure consistent versions of libraries.
- Consider replacing model cloning via `type(model)(**model.get_params())` with `sklearn.base.clone(model)` for safer cloning when adapting code.
- The single-output script applies `log1p` transforms when a target is skewed and shifts negative values before applying the log. Keep that behavior if reusing the script.

## Outputs
- Final test predictions: `outputs/single-output.csv` (or equivalent from `multi-output.py`)
- OOF predictions: `outputs/oof-single-output.csv`

## Suggestions / next steps
- Bundle model artifacts after training (`joblib.dump`) so the trained models can be re-used for fast inference.
- Add a small `examples/` folder containing a minimal input JSON or CSV and a short script to load models and predict.
- Add unit tests for data processing and a lightweight smoke test which runs the pipeline on a tiny subset of data.

## Licence & acknowledgements
Use this repository as you wish for experiments and reproducibility. If you publish results derived from this code, consider keeping the original authorship/acknowledgement line.

If you want, I can add a short `Makefile` or `tasks.json` to automate common commands, convert GPU flags to CPU fallbacks automatically, or add a small `examples/` harness and a CI smoke test.
