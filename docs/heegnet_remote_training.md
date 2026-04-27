# HEEGNet Remote Training (GitHub Push Checklist)

This guide covers what to push and how to run HEEGNet training on another device.

## 1) What to push

Push code and configs only:

- `scripts/train_heegnet.py`
- `scripts/_bootstrap.py`
- `src/eeg_thesis/`
- `heegnet/HEEGNet/` (vendor model code used by the training script)
- `requirements.txt`
- `requirements-heegnet.txt`
- `configs/`
- `docs/` and `README.md`

Do **not** push local data/checkpoints/caches. `.gitignore` handles most of this.

## 2) Create GitHub repo and push

From the project root:

```bash
git init
git add .
git commit -m "Prepare HEEGNet training pipeline for remote runs"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

This repo uses git submodules for vendor model code:

- `eegpt/EEGPT`
- `heegnet/HEEGNet`

## 3) Setup on remote machine

```bash
git clone --recurse-submodules <YOUR_GITHUB_REPO_URL>
cd thesis
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-heegnet.txt
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## 4) Provide data on remote machine

Create a local `data/` tree expected by your loaders (not tracked in git).  
Use the same folder naming scheme as in your current environment.

## 5) Smoke test run

```bash
python3 scripts/train_heegnet.py \
  --task age4 \
  --eyes closed \
  --max-per-group 8 \
  --limit-total 64 \
  --epochs 3 \
  --min-epochs 1 \
  --batch-size 16 \
  --device cpu
```

Or using config:

```bash
python3 scripts/train_heegnet.py --config configs/heegnet_age4.yaml --limit-total 64 --epochs 3 --device cpu
```

## 6) Full run template

```bash
python3 scripts/train_heegnet.py \
  --config configs/heegnet_age4.yaml
```

## Notes

- Validation metrics include `val_loss`, `val_score` (balanced accuracy), `val_f1_score`, `val_macro_f1`.
- If domain batches are too small in debug mode, the script automatically falls back to a regular shuffled dataloader.
