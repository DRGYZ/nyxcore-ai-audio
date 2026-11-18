# NyxCore Desktop 01: Pressed & Cataloged

Pressed & Cataloged is the approved identity for Desktop 01. The desktop launcher and browser workflow use the same React application and production build. Argon's original reference captures are preserved in [screenshots/redesign](../screenshots/redesign/).

## Visual rules

- Use warm obsidian surfaces, amber actions, alabaster primary text, pewter supporting text, and restrained jade, warning, and danger signals.
- Space Grotesk carries headings and collection identity. Inter carries controls and reading text. Newsreader provides selective editorial emphasis. JetBrains Mono is reserved for paths, hashes, timing, formats, and technical evidence. Fonts are served locally.
- Preserve the sleeve and record references. They are placeholders, not invented album covers. Existing provider thumbnails remain real images.
- Reuse `Panel`, `Button`, `Chip`, `PathBlock`, `Modal`, and `SleeveArt`. Palette and typography live in `web/tailwind.config.ts` and `web/src/index.css`.
- Use borders and spacing for hierarchy. Avoid neon, glass effects, decorative loading motion, and marketing-style dashboard embellishments.
- Keep paths, comparison evidence, provenance, changed values, collision reasons, and recovery instructions accessible. A narrow table may scroll within its panel; the page must not overflow horizontally.
- Preserve amber keyboard focus indicators, semantic controls, modal focus trapping and restoration, native disclosure behavior, and reduced-motion support.

## Native integration

`nyxcore/desktop.py` retains the native Windows title bar, WebView2 renderer, single-instance guard, navigation policy, loopback backend, and shutdown implementation. Only its background, caption, and foreground colors change for this identity. The Windows high-contrast exception remains in place.

Build `web/` before opening `.venv/Scripts/nyxcore-desktop.exe` in this checkout. The editable package must resolve `nyxcore.launch.SOURCE_ROOT` to this checkout, rather than a preserved baseline worktree. See [the desktop source setup](../INSTALL_DESKTOP.md).

## Product boundaries

Desktop 01 remains review-only. Staging proposals and triage may update the separate workspace; they must not modify audio. Backend restrictions remain authoritative. File mutations, reversal, and mutating recovery are unavailable in this launcher.

Search and track inspection expose artwork-presence flags, not embedded image URLs. Those screens keep sleeve placeholders. Existing MusicBrainz/Cover Art Archive choices and metadata plan thumbnails use their existing API contracts. A general embedded-artwork endpoint is outside this integration.

The native launcher blocks downloads. Health export is therefore disabled with an explanation in desktop mode; it remains available in the browser. Do not relax the download policy to make that button appear functional.

Acceptance evidence and outstanding limitations are recorded in [DESKTOP_DESIGN_ACCEPTANCE.md](DESKTOP_DESIGN_ACCEPTANCE.md).
