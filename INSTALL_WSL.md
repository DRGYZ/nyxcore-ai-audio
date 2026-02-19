# NyxCore Phase 3 on Windows via WSL2 (Ubuntu)

This guide is for Windows users who want to run NyxCore Phase 3 with Essentia in WSL2.

## 1) Install WSL2 + Ubuntu (PowerShell as Administrator)

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then confirm:

```powershell
wsl -l -v
```

You should see Ubuntu with version `2`.

## 2) Open Ubuntu and install system packages

In Start Menu, open `Ubuntu`, then run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git ffmpeg
```

## 3) Go to your Windows project folder from WSL

If your project is at `C:\Users\YAZAN\Desktop\YMusic`, in Ubuntu:

```bash
cd /mnt/c/Users/YAZAN/Desktop/YMusic
pwd
```

## 4) Create and activate venv, install NyxCore

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## 5) Install Essentia and verify import

Try:

```bash
pip install essentia
python -c "import essentia; print('essentia ok')"
```

Optional quick check:

```bash
python -c "import essentia.standard as es; print('essentia.standard ok')"
```

## 6) Run NyxCore commands

From project root in Ubuntu (venv active):

```bash
python -m nyxcore.cli scan music --out data/reports
python -m nyxcore.cli analyze music --out data/reports --backend dummy
python -m nyxcore.cli analyze music --out data/reports --backend essentia --limit 20
python -m nyxcore.cli apply-ai music --in data/reports/analysis_preview.jsonl --fields energy,bpm,tags,genre --dry-run
```

## 7) Troubleshooting

### If `pip install essentia` fails

- Upgrade build tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install essentia
```

- If it still fails, your Python version/wheel combo may be unsupported. Try a different Python version in WSL (commonly 3.10/3.11) and recreate the venv.

### If Essentia imports but model inference is unavailable

- NyxCore currently uses a minimal Essentia path for core analysis.
- Some advanced ML model pipelines require TensorFlow-enabled Essentia builds and model files.
- If needed, keep using:

```bash
python -m nyxcore.cli analyze music --backend dummy
```

### Check Python path and venv activation

```bash
which python
python -V
pip -V
```

Expected: paths should point inside `.venv` (for example `.venv/bin/python`).

