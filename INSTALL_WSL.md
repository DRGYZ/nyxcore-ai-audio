# NyxCore on Windows via WSL2 (Ubuntu)

This guide covers the current NyxCore CLI, FastAPI backend, demo fixture, and local web UI on Windows through WSL2.

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

If you plan to run the frontend from WSL, also install Node 18+.

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

## 5. Generate the demo library

Use the built-in demo fixture for the safest first run:

```bash
python demo/create_demo_library.py --force
```

Default output:

- `demo/generated/sample-library`

## 6. Run the CLI demo flow

```bash
python -m nyxcore.cli duplicates demo/generated/sample-library --out data/reports
python -m nyxcore.cli health demo/generated/sample-library --out data/reports
python -m nyxcore.cli review demo/generated/sample-library --out data/reports
python -m nyxcore.cli save-playlist demo/generated/sample-library --out data/reports --name "Ambient Focus" --query "ambient focus instrumental"
python -m nyxcore.cli list-playlists --out data/reports
```

If you want history data too, generate and apply one plan:

```bash
python -m nyxcore.cli review-plan demo/generated/sample-library --out data/reports --item-id <item-id-from-review.json>
python -m nyxcore.cli apply-review-plan data/reports/review_plan.json --out data/reports
python -m nyxcore.cli history --out data/reports
```

## 7. Run the FastAPI backend

Point the API at the generated demo library:

```bash
source .venv/bin/activate
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/reports"
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000
```

API URL:

- `http://127.0.0.1:8000`

## 8. Run the frontend

If Node.js is already available in WSL:

```bash
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend URL:

- `http://127.0.0.1:5173`

If the frontend shows mock fallback instead of live data:

- confirm the backend is running on `127.0.0.1:8000`
- confirm `NYXCORE_WEB_MUSIC_DIR` points to the generated demo library
- confirm `NYXCORE_WEB_OUT_DIR` points to `data/reports`

## 9. Optional analysis extras

Essentia and CLAP are optional for the demo, review, history, playlist, API, and frontend workflows.

Try:

```bash
pip install essentia
python -c "import essentia; print('essentia ok')"
```

Optional deeper import check:

```bash
python -c "import essentia.standard as es; print('essentia.standard ok')"
```

## 10. Troubleshooting

### If `ffmpeg` is missing

```bash
which ffmpeg
ffmpeg -version
```

The demo fixture generator depends on `ffmpeg`.

### If the API starts but the frontend falls back to mock data

- confirm the backend is running on `127.0.0.1:8000`
- confirm the frontend is running on `127.0.0.1:5173`
- confirm `NYXCORE_WEB_MUSIC_DIR` and `NYXCORE_WEB_OUT_DIR` are exported in the API terminal
- check the browser console and API terminal output for request failures

### If the wrong Python is active

```bash
which python
python -V
pip -V
```

Those paths should point inside `.venv`.
