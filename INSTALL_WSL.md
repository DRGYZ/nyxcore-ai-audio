# NyxCore on Windows via WSL2 (Ubuntu)

The synthetic review demo was verified on Windows 11 with Ubuntu 24.04 LTS under WSL2, using Python 3.12, Node.js 22.22.2, and FFmpeg 6.1.1. Python 3.11 also passed the backend suite. For a native Windows setup, see the [PowerShell quickstart](README.md#native-windows-quickstart).

## 1. Prerequisites

If Ubuntu is not installed yet, open PowerShell as Administrator:

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot if prompted. Run `wsl -l -v` and confirm the Ubuntu distribution shows version `2`.

## 2. System Packages in Ubuntu

In Ubuntu, install the packages for the demo:

```bash
sudo apt update
sudo apt install -y python3 python3-venv git curl ffmpeg
```

FFmpeg generates the synthetic audio. It is optional for core library review.

Install **Node.js 22** inside Ubuntu (22.13 or later within 22.x). The verified setup used the [NodeSource Node.js 22 package](https://github.com/nodesource/distributions/blob/master/DEV_README.md#using-ubuntu-nodejs-22). If you already have compatible Linux Node and npm installations, keep them.

```bash
node --version
npm --version
```

## 3. Clone the Repository

Keep the checkout, virtual environment, and demo files on the WSL Linux filesystem. A checkout under `/mnt/c` and reuse of Windows dependencies were not verified.

```bash
cd ~
git clone https://github.com/DRGYZ/nyxcore-ai-audio.git
cd nyxcore-ai-audio
```

If you already have a checkout inside WSL, use that directory instead.

## 4. Virtual Environment & Python Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[web,dev]"
python -V
```

Ubuntu 24.04 uses Python 3.12. Run `which python` to check that it points to `.venv/bin/python`.

## 5. Demo Fixture and Reports

For normal use, build the frontend once and launch from the repository root with the virtual environment active.

```bash
(cd web && npm ci --engine-strict && npm run build)
python demo/create_demo_library.py
python -m nyxcore.launch --demo --out data/demo-reports
```

Open the printed `http://127.0.0.1:8000/` URL in your Windows browser if needed. Use `--port` to choose a free port and `--no-browser` to skip opening a browser. Ctrl+C stops the launcher. To change libraries, restart with `--music "/path/to/library"` and a separate `--out` directory. The launcher blocks apply, restore, and undo at the API. For the optional CLI reports, see the [demo and report commands](README.md#2-generate-the-demo-and-reports).

## 6. Run the FastAPI Backend for Development

From the repository root with the virtual environment active:

```bash
export NYXCORE_WEB_MUSIC_DIR="$(pwd)/demo/generated/sample-library"
export NYXCORE_WEB_OUT_DIR="$(pwd)/data/demo-reports"
uvicorn nyxcore.webapi.app:app --reload --host 127.0.0.1 --port 8000
```

Open [API status](http://127.0.0.1:8000/api/status) and check that `music_path` points at the synthetic fixture.

## 7. Run the Vite Frontend for Development

In a second Ubuntu terminal, from the repository root:

```bash
cd web
npm ci --engine-strict
npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Open the [web UI](http://127.0.0.1:5173/) in a Windows browser and follow the [plan preview walkthrough](README.md#4-launch-the-react-web-ui-and-inspect-a-plan). Browsing, searching, and generating a plan do not change audio files. Applying plans, restoring or reversing batches, stale-lock takeover, and recovery writes are outside this demo.

## 8. Troubleshooting

### Frontend shows "Local API is not connected"

1. Open [API status](http://127.0.0.1:8000/api/status) in the Windows browser. If it fails, check the backend terminal and try the same URL with `curl` inside Ubuntu. If it works only inside Ubuntu, check Windows-to-WSL localhost access.
2. Confirm the backend points at the demo paths, then click **Check Connection** in the UI.

### Vite cannot bind port 5173

If Windows or WSL already uses `5173`, you can leave that service running. Start FastAPI as shown above. Then, from `web/` after the locked install, use Vite's built-in proxy on the alternate port:

```bash
VITE_API_BASE_URL=/api node --input-type=module <<'JS'
import { createServer } from "vite";

const server = await createServer({
  server: {
    host: "127.0.0.1",
    port: 55174,
    strictPort: true,
    proxy: { "/api": { target: "http://127.0.0.1:8000", timeout: 3000 } },
  },
});
await server.listen();
server.printUrls();
JS
```

Open [Overview on port 55174](http://127.0.0.1:55174/) and follow the same plan preview walkthrough. The Windows browser was verified with this setup on WSL2. Both services bind to loopback, and the browser sends API requests through Vite on the same origin. The stock API allows browser requests from port `5173`, so changing only the Vite port would fail. Press Ctrl+C in the frontend terminal when finished.

### FFmpeg is missing

Install FFmpeg inside Ubuntu (`sudo apt install -y ffmpeg`) and rerun the generator.
