# NyxCore Demo Fixture

Use `demo/create_demo_library.py` to generate a tiny local music library for demos and screenshots without committing binary media to Git.

## Generate the demo library

```bash
python demo/create_demo_library.py
```

Default output:

- `demo/generated/sample-library/`

Replace it safely:

```bash
python demo/create_demo_library.py demo/generated/sample-library --force
```

Requirements:

- `ffmpeg` on `PATH`
- Python environment with NyxCore installed

## What the fixture is designed to show

- exact duplicate review items
- likely duplicate review items
- missing metadata
- weak or placeholder metadata
- missing artwork
- low-bitrate audio
- saved-playlist candidates for text queries like `ambient focus instrumental`

## Suggested commands

```bash
python -m nyxcore.cli scan demo/generated/sample-library --out data/reports
python -m nyxcore.cli duplicates demo/generated/sample-library --out data/reports
python -m nyxcore.cli health demo/generated/sample-library --out data/reports
python -m nyxcore.cli review demo/generated/sample-library --out data/reports
python -m nyxcore.cli save-playlist demo/generated/sample-library --out data/reports --name "Ambient Focus" --query "ambient focus instrumental"
python -m nyxcore.cli list-playlists --out data/reports
```

The generator also writes a manifest and a README into the generated library so the expected findings stay visible next to the fixture itself.
