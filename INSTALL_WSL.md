# NyxCore on Windows via WSL2 (Ubuntu)

This guide walks through setting up the NyxCore CLI, FastAPI backend, demo library, and local web UI on Windows using WSL2 (Ubuntu).

---

## 1. Prerequisites

### Windows Subsystem for Linux (WSL2)
Open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu
```

Reboot if prompted, then verify WSL2 is active:

```powershell
wsl -l -v
```

Ensure the Ubuntu distribution shows version `2`.

---

## 2. System Packages in Ubuntu

Open your Ubuntu terminal and install the core build and runtime packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git curl
```

### Optional Media & Audio Tools
- **`ffmpeg`**: Optional for core library review. `mutagen` reads and writes audio metadata in pure Python. FFmpeg is only required if you plan to synthesize test audio using `demo/create_demo_library.py`, or if using experimental CLAP audio transcoding:
  ```bash
  sudo apt install -y ffmpeg
  ```
- **Node.js (18+)**: Required if running the frontend inside WSL:
  ```bash
  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
  sudo apt install -y nodejs
  ```

---

## 3. Clone or Navigate to the Repository

Navigate to your workspace directory:

```bash
# If cloned inside WSL:
cd ~/nyxcore-ai-audio

# Or if accessing from a Windows drive mount:
cd /mnt/c/path/to/nyxcore-ai-audio
```

---

## 4. Virtual Environment & Python Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[web,dev]"
```

Verify the environment:

```bash
which python
python -V
```

The output should point to `.venv/bin/python` with Python 3.11 or newer.

---

## 5. Demo Fixture Setup

For a clean, risk-free first run, generate the synthetic demo fixture:

```bash
python demo/create_demo_library.py demo/generated/sample-library --force
```

> **Note**: Generating synthetic audio requires `ffmpeg` on `PATH`. If you already have an audio folder, you can skip this step and point directly to your music directory.

Run the core detection and review pipeline against the sample library:

```bash
python -m nyxcore.cli duplicates demo/generated/sample-library --out data/reports
python -m nyxcore.cli health demo/generated/sample-library --out data/reports
python -m nyxcore.cli review demo/generated/sample-library --out data/reports
```

To test the safe mutation and history pipeline:

```bash
# Generate a review plan for an exact duplicate item (check data/reports/review.json for item_id)
python -m nyxcore.cli review-plan demo/generated/sample-library --out data/reports --item-id <exact-duplicate-item-id>

# Apply the action plan (moves non-preferred copy to .nyxcore_quarantine)
python -m nyxcore.cli apply-review-plan data/reports/review_plan.json --music demo/generated/sample-library --out data/reports

# Inspect the recorded history ledger
python -m nyxcore.cli history --out data/reports

# Reverse the batch back to original file locations when supported
python -m nyxcore.cli undo-review-action <batch-id> --music demo/generated/sample-library --out data/reports
```

---

## 6. Running the Local FastAPI Backend

Launch the local API server pointed at your target music folder and reports directory:

```bash
source .venv/bin/activate
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/reports"
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000
```

Verify the API is running:
- **API Status**: [http://127.0.0.1:8000/api/status](http://127.0.0.1:8000/api/status)
- **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 7. Running the React Web UI

In a separate terminal window:

```bash
cd web
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Open your browser:
- **Web UI**: [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## 8. Troubleshooting

### Frontend shows "Local API Disconnected"
1. Verify the backend process is running on `127.0.0.1:8000`.
2. Confirm `NYXCORE_WEB_MUSIC_DIR` and `NYXCORE_WEB_OUT_DIR` are exported in the terminal running `uvicorn`.
3. Check that WSL port forwarding is working or connect directly via `127.0.0.1:5173`.

### "ffmpeg is required to generate the NyxCore demo library"
The `demo/create_demo_library.py` script requires `ffmpeg` to synthesize sine-wave audio tones. Install FFmpeg in Ubuntu (`sudo apt install -y ffmpeg`) or test with an existing audio directory.

### Interrupted or Lock-Stalled Mutations
NyxCore employs a library-scoped lock during plan execution. If a process was terminated mid-operation:
```bash
python -m nyxcore.cli recover-review-action --action inspect --music <music-dir> --out data/reports
```
Incomplete operations will be safely classified; only provably safe states offer `finalize` or `abort`.
