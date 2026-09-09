# PRD build

This folder lives at `adapter-service-prd-build/` in the AdapterOps repo; the
deliverables it produces sit in the repo root beside it. Three deliverables, one source. They must never drift.

| File | What it is |
|---|---|
| `source.html` | **Single source of truth.** Lean: Google Fonts `<link>`, mermaid loaded from cdnjs behind a `file://` guard. This is what gets published as the Artifact. |
| `../adapter-service-prd.html` | Generated, **gitignored**. Fonts and mermaid inlined, zero external refs, opens offline. Run the build to produce it after a fresh clone. |
| `../adapter-service-prd.md` | Hand-maintained Markdown mirror of the same content. |

## Changing the document

1. Edit `source.html`.
2. `python3 build-offline.py` — regenerates `../adapter-service-prd.html`.
3. Publish `source.html` to the existing Artifact URL (same URL, keeps version history).
4. Apply the same content change to `../adapter-service-prd.md`.

Steps 2–4 are not optional. Skipping any one of them is how the copies diverge.

## Why the two HTML files differ

The published Artifact renders ```mermaid``` blocks natively and loads Google Fonts
over the network, so inlining 3.2 MB of mermaid plus 375 KB of base64 fonts there
would be dead weight on every page load. The offline copy has no network, so it
needs both. Same content, different packaging.

## vendor/

Pinned so the build works with no network:

- `mermaid-10.9.1.min.js` — from cdnjs.
- `fonts/` — latin-subset woff2 for IBM Plex Mono / Sans / Sans Condensed and
  Source Serif 4, plus `manifest.json`. Nine files, not twelve: IBM Plex Sans and
  Source Serif 4 ship as variable fonts covering a weight range, so one file backs
  several weights and is declared once with e.g. `font-weight: 400 600`.

Latin only — non-goal 11 is English-only. Other scripts fall back to a system face.

## build-offline.py

Fails loudly rather than producing a half-inlined page: it exits non-zero if the
Google Fonts link or the mermaid loader can't be found in `source.html` (i.e. the
head or script block changed shape), or if any `gstatic`/`googleapis`/`cdnjs`
reference survives into the output.

The output path is relative to this script (`../adapter-service-prd.html`), so the build
works wherever the repo is cloned. Pass a path as the first argument to override it.

`vendor/` is committed on purpose: the build must be reproducible offline, and §9 of the
PRD makes the same argument for mirroring datasets against upstream takedown.

The generated HTML is *not* committed — it is 3.6 MB, and tracking it would add a fresh
multi-megabyte blob to history on every PRD edit. `source.html` plus `vendor/` reproduce
it exactly, so nothing is lost. A fresh clone has no `../adapter-service-prd.html` until
the build is run.
