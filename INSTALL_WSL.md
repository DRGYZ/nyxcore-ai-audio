# NyxCore on Windows via WSL2 (Ubuntu)

This guide covers the current NyxCore CLI and local web stack on Windows through WSL2.

## 1. Install WSL2 and Ubuntu

Run PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then confirm:

```powershell
wsl -l -v
```

Ubuntu should show version `2`.

## 2. Install system packages in Ubuntu

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git ffmpeg curl
```

## 3. Open the project folder from WSL

If the repo is at `C:\Users\YAZAN\Desktop\YMusic`:

```bash
cd /mnt/c/Users/YAZAN/Desktop/YMusic
pwd
```

## 4. Create a virtual environment and install NyxCore

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[web,dev]"
```

Optional extras:

```bash
pip install -e ".[audio-analysis]"
pip install -e ".[clap]"
pip install -e ".[audio-analysis,clap,web,dev]"
```

NyxCore uses the packaged default config automatically unless you pass `--config`.

## 5. Run the CLI workflow

```bash
python -m nyxcore.cli scan music --out data/reports
python -m nyxcore.cli duplicates music --out data/reports
python -m nyxcore.cli health music --out data/reports
python -m nyxcore.cli review music --out data/reports
```

Action plan and history workflow:

```bash
python -m nyxcore.cli review-plan music --out data/reports --item-id <item-id>
python -m nyxcore.cli apply-review-plan data/reports/review_plan.json --out data/reports
python -m nyxcore.cli history --out data/reports
```

Playlist workflow:

```bash
python -m nyxcore.cli playlist music --query "focus music" --out data/reports
python -m nyxcore.cli save-playlist music --out data/reports --name "Focus Set" --query "focus music"
python -m nyxcore.cli list-playlists --out data/reports
```

## 6. Run the FastAPI backend

```bash
source .venv/bin/activate
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000
```

API URL:

- `http://127.0.0.1:8000`

## 7. Run the frontend

If Node.js is already available in WSL:

```bash
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

If Node.js is not installed yet, install Node 18+ first. `nvm` is the least painful path in WSL for keeping Node current.

Frontend URL:

- `http://127.0.0.1:5173`

## 8. Essentia and analysis extras

Try:

```bash
pip install essentia
python -c "import essentia; print('essentia ok')"
```

Optional deeper import check:

```bash
python -c "import essentia.standard as es; print('essentia.standard ok')"
```

If you only need the review, history, playlist, API, and frontend workflows, Essentia is optional.

## 9. Troubleshooting

### If `pip install essentia` fails

```bash
python -m pip install --upgrade pip setuptools wheel
pip install essentia
```

If it still fails, the current Python version may not match an available wheel. Python 3.10 or 3.11 is usually the next thing to try in WSL.

### If the API starts but the frontend falls back to mock data

- confirm the backend is running on `127.0.0.1:8000`
- confirm the frontend is running on `127.0.0.1:5173`
- check the browser console and API terminal output for request failures

### If the wrong Python is active

```bash
which python
python -V
pip -V
```

Those paths should point inside `.venv`.
