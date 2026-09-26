# Context-Aware Video Segmentation & Intelligent Ad Placement

An end-to-end pipeline that turns a long-form video into an IAB-compliant, ad-supported deliverable. It **segments** the video into semantic scenes, **scores every potential ad break** for cut-safety and pacing, **matches approved breaks to brands** from a data-driven catalogue, and exports:

- an **IAB VMAP 1.0.1 ad manifest** (`manifest.xml`),
- a **fully transparent debug trace** (`debug.json`) showing *why* every break was approved/rejected and *which* brand was chosen,
- a **playable in-browser demo** that consumes only static, precomputed artifacts.

Nothing is hardcoded: brands are data, thresholds live in YAML config, and a **Phase 7 automated holdout gate** must pass all five checks before the output is considered "safe to submit".

---

## Table of Contents

1. [Key Features](#key-features)
2. [How It Works — Three Questions, Three Subsystems](#how-it-works--three-questions-three-subsystems)
3. [Architecture — Phase by Phase](#architecture--phase-by-phase)
4. [Hard Constraints & How They Are Enforced](#hard-constraints--how-they-are-enforced)
5. [Repository Structure](#repository-structure)
6. [Requirements](#requirements)
7. [Installation & Quick Start](#installation--quick-start)
8. [Running the Full Pipeline](#running-the-full-pipeline)
9. [Per-Phase Scripts & Utilities](#per-phase-scripts--utilities)
10. [The Playable Demo](#the-playable-demo)
11. [Configuration](#configuration)
12. [Brand Catalogue & Adding a 9th Brand](#brand-catalogue--adding-a-9th-brand)
13. [Data Contracts (Schemas)](#data-contracts-schemas)
14. [Tests](#tests)
15. [Evaluation & Submission Gates (Phase 7)](#evaluation--submission-gates-phase-7)
16. [Generated Artifacts](#generated-artifacts)
17. [Development Status & Git History](#development-status--git-history)
18. [Troubleshooting](#troubleshooting)
19. [Known Limitations](#known-limitations)

---

## Key Features

- **Scene-aware cut safety** — breaks are scored on hard sub-signals (sentence-boundary alignment, silence gap, shot alignment, action continuity), never a single opaque number.
- **Constraint-driven pacing** — a dynamic-programming (weighted interval scheduling) selector enforces `max_breaks_per_hour`, `min_gap_seconds`, `target_ad_load_pct`, and a cold-open protection window.
- **Zero negative-context violations** — brands whose `negative_contexts` match a scene are **hard-filtered before ranking**, using synonym-aware semantic similarity (e.g. `funeral` also blocks on `wake`, `cremation`, `mourning`).
- **Generalization by data, not code** — adding a 9th brand to the catalogue is a single JSON entry; the matcher picks it up with zero code changes (covered by an automated CI-able test).
- **Transparent output** — every decision (scene tags, scores, reasons, ranked brands, blocked brands) is emitted to `debug.json`.
- **Standards-compliant export** — generated manifests validate against the public IAB **VMAP 1.0.1** XSD (a build gate, not optional polish).
- **Live-runnable demo** — all ML inference is precomputed; the demo runs from static files with zero network/API calls at presentation time.
- **Phase-organized codebase** — the git history itself is structured phase by phase, mirroring the phased engineering spec this project was built against.

## How It Works — Three Questions, Three Subsystems

The pipeline is organized around the three decisions that matter for ad placement:

| Question | Subsystem | Failure mode it prevents |
|----------|-----------|--------------------------|
| **Where** can we cut? | Shot-boundary detection + cut-safety scorer | Mid-sentence or mid-action cuts |
| **Whether** to cut here? | Ad-pacing rules engine | Too many breaks / breaks too close together / over ad-load |
| **What** ad goes here? | Brand-matching engine (embedding similarity) | Wrong brand for the scene, or a hard-blocked brand slipping through |

### Pipeline Flow

```
raw video + audio
      │  Phase 1  PySceneDetect shots → CLIP frame similarity + ASR topic continuity
      ▼
scenes.json ──► Phase 2  cut-safety scoring (where) ──► break_candidates.json
                                                          │
                                              Phase 3  pacing selection (whether)  ◄── config/pacing_rules.yaml
                                                          │
                                              Phase 4  brand matching (what)       ◄── data/brand_catalogue.json
                                                          │
                                              Phase 5  IAB VMAP 1.0.1 manifest + debug trace
                                                          │
                                              Phase 6  demo player (static consumer)
                                                          ▲
                                              Phase 7  automated holdout gate (safe-to-submit PASS/FAIL)
```

---

## Architecture — Phase by Phase

Each phase consumes **versioned, inspectable JSON artifacts** and writes the next one — this is what makes the debug trace easy, and what lets any single component be swapped without touching its neighbours.

| Phase | Scope | Input → Output | Key detail |
|-------|-------|----------------|------------|
| **0** | Repo & data scaffolding | `brand_catalogue.json` + configs + JSON Schemas → schema-validated data | `scripts/validate_schemas.py` + `tests/test_schemas.py` form the Phase 0 acceptance gate |
| **1** | Shot detection & scene segmentation ("chunking") | video → `scenes.json`, `whisper_words.json` | PySceneDetect shots merged by CLIP frame-embedding similarity + ASR topic continuity; zero-shot activity tagging from `config/activity_taxonomy.yaml` |
| **2** | Cut-safety scoring ("Where") | `scenes.json` → `break_candidates.json` | `cut_safety_score ∈ [0,1]` from sentence-boundary alignment (Whisper word timestamps), silence-gap RMS, shot-boundary alignment, action-continuity penalty; per-signal reasons logged |
| **3** | Ad-pacing rules ("Whether") | `break_candidates.json` → approved/rejected decisions | DP weighted-interval scheduling honoring `max_breaks_per_hour`, `min_gap_seconds`, `target_ad_load_pct`, `first_break_not_before_sec`, `cut_safety_floor` |
| **4** | Brand matching ("What") | break candidates + catalogue → `brand_matches.json` | **Hard-block** (not soft penalty) on negative contexts *before* ranking; rank survivors by weighted embedding similarity to `target_contexts`, dominant activity weighted highest |
| **5** | VMAP manifest + debug JSON | approved breaks + matches → `manifest.xml`, `debug.json` | Jinja2 XML template implementing IAB VMAP 1.0.1; validated against the public XSD with `lxml`; debug trace covers scenes → candidates → decisions → brand rankings |
| **6** | Playable demo | `manifest.xml` + `debug.json` + ad assets → `demo/index.html` | video.js player pauses at each approved `timeOffset`, plays the matched brand creative, resumes at the exact prior position; served by a range-enabled HTTP server |
| **7** | Held-out evaluation harness | any new video + catalogue → PASS/FAIL gate | 5 gates: mid-sentence cuts, negative-context violations, pacing constraints, VMAP XSD validity, hardcoding audit |

`run_pipeline.py` is the master runner that chains Phases 1–6 on any input video and can trigger the Phase 7 gate with a single flag.

## Hard Constraints & How They Are Enforced

1. **No hardcoding** — brands live in `data/brand_catalogue.json`; every threshold lives in `config/*.yaml`. Enforced by `scripts/check_hardcoding.py` / `scripts/check_no_hardcoding.sh` (grep/scan fail on literal brand names in source) and re-checked by the Phase 7 gate.
2. **Zero negative-context violations** — implemented as a **hard filter that runs before ranking**, not a negative weight a strong positive score could outweigh. Covered by adversarial tests using synonym scenes (`wake`, `cremation`, `mourning`).
3. **Generalize to unseen brands** — brand #9 is one JSON entry, zero code edits. Enforced by `tests/test_new_brand_generalization.py` which simulates adding a 9th brand and reruns the matcher.
4. **Live-runnable demo** — all ML inference is precomputed ahead of time; the demo runs from a static manifest + debug JSON with no external API calls at presentation time.

---

## Repository Structure

```
adtech-video-pipeline/
├── config/                         # Single source of truth for all thresholds
│   ├── pacing_rules.yaml           #   Ad-break scheduling rules (breaks/hour, gap, ad-load%)
│   ├── model_versions.yaml         #   Pinned ML model versions & segmentation settings
│   └── activity_taxonomy.yaml      #   Zero-shot activity label list (Phase 1/4)
├── data/
│   ├── brand_catalogue.json        # 8 synthetic brands (data, not code)
│   ├── sample_videos/              # Drop source videos here (.gitkeep commits the dir)
│   └── processed/<video_id>/       # GENERATED per-video pipeline artifacts (gitignored)
├── schemas/
│   ├── brand.schema.json           # JSON Schema for catalogue entries
│   ├── scene.schema.json           # JSON Schema for scene objects
│   ├── break_candidate.schema.json # JSON Schema for break candidates
│   └── vmap-1.0.1.xsd              # IAB VMAP 1.0.1 XSD (Phase 5 validation gate)
├── pipeline/                       # Core library (importable, testable modules)
│   ├── shot_detect.py              #   Phase 1: PySceneDetect shots
│   ├── scene_segment.py            #   Phase 1: merge shots → scenes
│   ├── scene_understand.py         #   Phase 1: zero-shot activity tagging
│   ├── cut_safety.py               #   Phase 2: cut-safety scoring ("Where")
│   ├── pacing_engine.py            #   Phase 3: rules + DP selection ("Whether")
│   ├── brand_matcher.py            #   Phase 4: hard-block + rank ("What")
│   ├── vmap_builder.py             #   Phase 5: IAB VMAP 1.0.1 XML + XSD validate
│   └── debug_emitter.py            #   Phase 5: full debug.json trace
├── run_pipeline.py                 # Master CLI: video → manifest + debug (+demo/eval)
├── demo/
│   └── index.html                  # video.js demo player (static consumer)
├── scripts/                        # Runners, validators, tooling (see §Per-Phase Scripts)
├── tests/                          # pytest suites + labeled ground-truth fixtures
└── README.md
```

---

## Requirements

- **Python 3.10+** (developed on 3.14)
- **ffmpeg** on `PATH` (used for audio extraction in Phase 1 and demo asset generation). The pipeline first tries a local override path, then falls back to the `ffmpeg` executable on `PATH`.
- Optional but recommended for full inference: a working **PyTorch** install (CPU is fine for the sample workflows) and network access for one-time model downloads (faster-whisper, CLIP, SentenceTransformer).

### Python packages

```text
Core:      numpy  pyyaml  jsonschema  pytest
Vision:    opencv-python  scenedetect (PySceneDetect)
ML:        torch  faster-whisper  sentence-transformers  open-clip-torch
Export:    jinja2  lxml
Audio:     librosa or pydub        (optional, deeper silence/energy analysis)
```

## Installation & Quick Start

```bash
# 1) Install core dependencies
pip install numpy pyyaml jsonschema pytest opencv-python

# 2) Install pipeline / ML dependencies (for full end-to-end runs)
pip install scenedetect faster-whisper sentence-transformers open-clip-torch jinja2 lxml

# 3) Verify the environment (optional dev utility)
python scripts/check_env.py

# 4) Validate every JSON file in data/ against schemas/  (Phase 0 acceptance)
python scripts/validate_schemas.py

# 5) Run the test suite
pytest tests/ -v
```

The quickest smoke test that exercises the whole Phase 0 → Phase 5 chain **without** heavy ML models is to generate the synthetic sample data and manifests:

```bash
# 6) Generate synthetic ad assets + precomputed demo manifests (Phase 6 prep)
python scripts/create_sample_ad_assets.py
python scripts/prepare_demo_dataset.py
```

`prepare_demo_dataset.py` reuses the pipeline libraries (`vmap_builder`, `debug_emitter`) to emit valid `manifest.xml` + `debug.json` for all four sample videos — the same artifacts the demo player consumes.

---

## Running the Full Pipeline

The master runner executes Phases 1 → 5 end-to-end on **any** input video, then optionally links the output into the demo and runs the Phase 7 gate:

```bash
python run_pipeline.py <video_path> [--catalogue PATH] [--demo] [--eval]

# Examples
python run_pipeline.py data/sample_videos/bhojon_bilashi.mp4
python run_pipeline.py data/sample_videos/mohanagar.mp4 --demo --eval
```

| Flag | Purpose |
|------|---------|
| `video_path` | Input video file (`.mp4`, `.mkv`, …); its stem becomes the `video_id` |
| `--catalogue PATH` | Brand catalogue override (default `data/brand_catalogue.json`) |
| `--demo` | Copy `manifest.xml`/`debug.json` into `demo/` and hard-link the video for the browser player |
| `--eval` | Run the Phase 7 submission gate on this run's outputs (exit non-zero on failure) |

**What it produces** — everything lands in `data/processed/<video_id>/`:

| Output | Phase | What's inside |
|--------|-------|---------------|
| `whisper_words.json` | 1 | Word-level ASR timestamps used by cut safety |
| `scenes.json` | 1 | Merged scenes with `activity_tags`, `dominant_activity`, `asr_text` |
| `break_candidates.json` | 2 | Every scene boundary scored for cut safety |
| `brand_matches.json` | 4 | Hard-blocked brands + ranked survivors per break |
| `manifest.xml` | 5 | IAB VMAP 1.0.1 ad manifest (XSD-validatable) |
| `debug.json` | 5 | Full transparent trace of the entire decision chain |

> If Whisper/audio tooling is unavailable, the pipeline degrades gracefully: it warns and continues with empty ASR data (cut safety then relies on the remaining sub-signals).

---

## Per-Phase Scripts & Utilities

| Script | Phase | Purpose / Usage |
|--------|-------|-----------------|
| `scripts/validate_schemas.py` | 0 | Validate every JSON in `data/` against `schemas/` → `python scripts/validate_schemas.py` |
| `scripts/check_env.py` | dev | Prints Python/pip environment and detects local venvs |
| `scripts/check_download.py` | dev | Reports size of the local HuggingFace faster-whisper cache |
| `scripts/check_hardcoding.py` | 4 | AST-based audit: fail if any brand name/ID appears as a literal in source |
| `scripts/check_no_hardcoding.sh` | 4 | Bash CI gate equivalent (grep-fail on literal brand strings) |
| `scripts/run_phase1_pipeline.py` | 1 | Standalone Phase 1 runner: shots → scenes → ASR for sample videos; writes the manual spot-check report |
| `scripts/create_phase2_files.py` | 2 | Bootstrap codegen that scaffolded `cut_safety.py`, the eval script, and labeled fixtures |
| `scripts/eval_cut_safety.py` | 2 | Precision/Recall/F1 vs. human labels → `python scripts/eval_cut_safety.py [threshold]` |
| `scripts/run_phase4_pipeline.py` | 4 | Phase 4 runner: scene understanding (if needed) + brand matching |
| `scripts/run_phase5_pipeline.py` | 5 | Phase 5 runner: VMAP manifest + debug trace for a processed `video_id` |
| `scripts/prepare_demo_dataset.py` | 6 | Generate synthetic ad MP4s + demo manifests/debug traces for all samples |
| `scripts/create_sample_ad_assets.py` | 6 | Generate the 8 brand creative MP4s into `assets/ads/` + `demo/assets/ads/` |
| `scripts/server.py` | 6 | Range-enabled (HTTP 206) static server for the demo → `python scripts/server.py 8000 demo` |
| `scripts/build_holdout_labels.py` | 7 | Interactive CLI to label boundaries on a new video as `safe`/`jarring` |
| `scripts/eval_holdout.py` | 7 | The 5-gate submission gate → `python scripts/eval_holdout.py <video>` |

## The Playable Demo

The demo is a **static consumer** of precomputed artifacts — no ML, no network calls at presentation time. It loads `demo/manifest.xml` + `demo/debug.json` and the source video, and at every approved ad break: **pauses** the main video → **plays** the matched brand's creative → **resumes** at the exact prior position.

```bash
# 1. Generate ad creatives + demo manifests (one-time)
python scripts/create_sample_ad_assets.py
python scripts/prepare_demo_dataset.py

# 2. Serve the demo from the demo/ directory
python scripts/server.py 8000 demo

# 3. Open in a browser
#    http://localhost:8000/index.html
```

**Why not `python -m http.server`?** The stdlib server lacks RFC 7233 **HTTP 206 Partial Content** support. Without byte-range responses, browsers cannot seek in HTML5 videos and restart playback from 0:00. `scripts/server.py` adds 206 support with zero dependencies, so scrubbing is instant.

The on-screen debug overlay shows, per break: `cut_safety_score`, scene `dominant_activity`, the matched brand, and why it was chosen — answering "how do we know it's not hardcoded" live.

> Because demo outputs are generated, they are **gitignored** (`demo/data/`, `demo/*.mp4`, `demo/manifest.xml`, `demo/debug.json`, `assets/ads/*.mp4`). A fresh clone regenerates them with the two commands above.

---

## Configuration

Everything tunable lives in `config/` — **no pipeline code references numeric thresholds as literals**.

### `config/pacing_rules.yaml` — the pacing engine's single source of truth

| Key | Default | Meaning |
|-----|---------|---------|
| `max_breaks_per_hour` | `6` | Max ad breaks in any rolling 60-min window (IAB/BARB 4–6 range) |
| `min_gap_seconds` | `480` | Min seconds between end of one pod and start of the next (8 min) |
| `target_ad_load_pct` | `12.0` | Max ad seconds as % of runtime (AVOD ceiling benchmark) |
| `first_break_not_before_sec` | `300` | Protects the cold-open; no break before 5 minutes |
| `cut_safety_floor` | `0.55` | Minimum cut-safety score to reach the pacing stage |
| `max_pod_duration_sec` | `60` | Cap on a single ad pod (anti "pod stuffing") |

### `config/model_versions.yaml` — pinned ML model versions

Whisper (size, backend `faster-whisper`), CLIP (`ViT-B/32`), SentenceTransformer (`all-mpnet-base-v2`), PySceneDetect detector settings (AdaptiveDetector threshold / min scene length), and scene-segmentation tuning (shot-similarity threshold, min/max scene duration, ASR pause threshold, topic-change cue phrases like *"meanwhile"*, *"the next day"*).

### `config/activity_taxonomy.yaml` — zero-shot activity labels

The label list scene understanding scores against (e.g. `family meal`, `celebration party`, `road trip driving`). **Add new labels freely — no code changes required.**

---

## Brand Catalogue & Adding a 9th Brand

The catalogue ships with **8 synthetic brands** across varied categories. **No real company names are used.**

| Brand | Category | Tone |
|-------|----------|------|
| ZestCola | Beverage | Energetic |
| CrunchBite | Fast Food | Fun |
| ShelterWise Insurance | Insurance | Reassuring |
| Novatel Mobile | Telecom | Modern |
| StreamFlix | Streaming Service | Aspirational |
| VoltDrive Auto | Automobile | Aspirational |
| MunchPop Snacks | Snack | Playful |
| QuickKart | E-Commerce | Upbeat |

Each entry carries the contextual data the matcher needs — target contexts to *match*, negative contexts to *hard-block*, and the creative asset:

```json
{
  "brand_id": "brand_novatel",
  "display_name": "Novatel Mobile",
  "category": "telecom",
  "target_contexts": ["commute", "tech_meeting", "outdoor_adventure"],
  "negative_contexts": ["funeral", "grief", "hospital", "tragedy"],
  "creative_asset": "assets/ads/novatel_20s.mp4",
  "duration_sec": 20,
  "tone": "modern"
}
```

**Adding a 9th brand = one JSON entry** in `data/brand_catalogue.json` (plus one `assets/ads/<brand>.mp4` creative generated by `scripts/create_sample_ad_assets.py`). No code is touched — `tests/test_new_brand_generalization.py` proves the matcher selects/blocks the new brand purely from its data.

## Data Contracts (Schemas)

Contracts are locked **before** pipeline code — everything is built against them:

| Schema | Validates |
|--------|-----------|
| `schemas/brand.schema.json` | Every entry in `data/brand_catalogue.json` |
| `schemas/scene.schema.json` | Scene objects (`scene_id`, `start_ts`, `end_ts`, `activity_tags`, `dominant_activity`, `asr_text`, …) |
| `schemas/break_candidate.schema.json` | Break candidates (`candidate_id`, timestamps, `cut_safety_score`, `cut_safety_reasons`, pacing fields, final decision) |
| `schemas/vmap-1.0.1.xsd` | The exported `manifest.xml` (IAB VMAP 1.0.1, via `lxml`) |

`python scripts/validate_schemas.py` validates every JSON file in `data/` against its matching schema and exits non-zero on any failure (used as a Phase 0 acceptance gate).

---

## Tests

```bash
pytest tests/ -v          # full suite
pytest tests/test_schemas.py -v   # Phase 0 acceptance gate only
```

| Test file | Phase | Verifies |
|-----------|-------|----------|
| `test_schemas.py` | 0 | All data validates against schemas; malformed catalogue rejected |
| `test_scene_segment.py` | 1 | Shot merging, scene boundaries, schema conformance |
| `test_cut_safety.py` | 2 | Sentence-boundary, silence, shot, and action sub-signals |
| `test_pacing_engine.py` | 3 | DP selection + every pacing constraint never violated |
| `test_brand_matcher.py` | 4 | **Hard-block** of negative contexts (incl. synonym scenes) |
| `test_new_brand_generalization.py` | 4 | 9th brand requires zero code changes |
| `test_vmap_builder.py` | 5 | XSD-valid manifests + `debug.json`/VMAP consistency |

Fixtures: `tests/labeled_boundaries.json` (26 human-judged `safe`/`jarring` boundaries), `tests/labeled_ground_truth_context.json`, and the Phase 1 manual spot-check report (`tests/manual_review_phase1.md`).

---

## Evaluation & Submission Gates (Phase 7)

`scripts/eval_holdout.py <video_path> [--catalogue PATH] [--skip-pipeline]` runs the pipeline end-to-end on any new/held-out video and enforces **five gates**. Exit code `0` only if ALL pass:

| Gate | Check | Fail condition |
|------|-------|----------------|
| 1. Mid-sentence cuts | `check_mid_sentence_cuts` | Any cut inside an unfinished sentence (target: **0**) |
| 2. Negative-context safety | `check_negative_context_violations` | Hard-blocked brand appears in any slot (target: **0** — scored-zero criterion) |
| 3. Pacing constraints | `check_pacing_constraints` | min gap / breaks-per-hour / ad-load / cold-open / cut floor violations |
| 4. VMAP validity | `check_vmap_validity` | `manifest.xml` fails IAB VMAP 1.0.1 XSD validation |
| 5. No hardcoding | `check_no_hardcoding` | Any literal brand name/ID found in `pipeline/*.py` |

Run it standalone or as part of the master runner: `python run_pipeline.py <video> --eval` → *"FINAL RESULT: [ PASS ] - All submission gates satisfied. Safe to submit."*

---

## Generated Artifacts

The following are **build outputs, regenerated on demand, and intentionally gitignored**:

| Artifact | Generated by |
|----------|--------------|
| `data/processed/<video_id>/*` | `run_pipeline.py` (or per-phase runners) |
| `assets/ads/*.mp4`, `demo/assets/ads/*.mp4` | `scripts/create_sample_ad_assets.py` |
| `demo/data/*`, `demo/manifest.xml`, `demo/debug.json`, `demo/*.mp4` | `scripts/prepare_demo_dataset.py` / `run_pipeline.py --demo` |
| `__pycache__/`, `.pytest_cache/` | Python/pytest |

---

## Development Status & Git History

All phases (0–7) are **implemented** and committed, and the git history is deliberately **phase-ordered** so reviewers can follow the build progression:

| Commit | Contents |
|--------|----------|
| Phase 0 | Repo scaffold, data contracts, schemas, configs, catalogue, tooling |
| Phase 1 | Shot detection + scene segmentation + activity tagging |
| Phase 2 | Cut-safety scoring |
| Phase 3 | Pacing rules engine |
| Phase 4 | Brand matching with hard-block |
| Phase 5 | VMAP manifest + debug trace |
| Phase 6 | Demo player + asset generation |
| Phase 7 | Held-out evaluation harness |
| Final | `run_pipeline.py` master CLI |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ffmpeg` not found during audio extraction | Put `ffmpeg` on `PATH`, or edit the fallback path in `run_pipeline.py` / `create_sample_ad_assets.py` |
| Heavy ML models failing to load offline | The pipeline emits warnings and continues with reduced signals (e.g. empty ASR); for full fidelity run `prepare_demo_dataset.py` and the master runner once with network access |
| Demo video restarts from 0:00 on seek | You're using `python -m http.server` — use `python scripts/server.py 8000 demo` (HTTP 206) |
| XSD validation fails on generated manifest | Verify with `python scripts/run_phase5_pipeline.py <video_id>` and check `debug.json` for break data issues |
| `pytest` cannot find tests | Run from the repo root; `tests/__init__.py` + root-level config make discovery deterministic |

---

## Known Limitations

- **Synthetic ad assets** — the 8 brand creatives are procedurally generated MP4s (brand name + tagline + countdown), meant for end-to-end demonstration rather than broadcast-grade creative.
- **Precomputed sample outputs** — sample-video manifests in `demo/` are regenerated by `prepare_demo_dataset.py`; the demo makes no live API calls by design.
- **ASR graceful degradation** — without Whisper/torch, cut-safety runs on shot/silence/action signals only (sentence-boundary sub-signal is skipped).
- **Single-machine paths** — a few dev utilities still carry local, machine-specific paths (e.g. `scripts/check_env.py`, `scripts/check_download.py`, ffmpeg override). These are tooling aids, not pipeline logic, and are documented as such.

---

*See the [Architecture — Phase by Phase](#architecture--phase-by-phase) section for the phase-by-phase spec and the acceptance criteria each stage was built against.*
