# Context-Aware Video Segmentation & Intelligent Ad Placement

An end-to-end pipeline that segments long-form video into scenes, scores potential ad-break points for safety and pacing, matches surviving breaks to brands from a JSON catalogue, and exports IAB VMAP 1.0.1 ad manifests + debug JSON + a playable demo.

## Three Questions, Three Subsystems

| Question | Subsystem | Key concern |
|----------|-----------|-------------|
| **Where** can we cut? | Shot boundary detector + cut-safety scorer | No mid-sentence or mid-action cuts |
| **Whether** to cut here? | Ad-pacing rules engine | Frequency limits, ad-load budget |
| **What** ad goes here? | Brand-matching engine (embedding similarity) | Zero tolerance for negative-context violations |

## Hard Constraints

1. **No hardcoding** — brands are data (`data/brand_catalogue.json`), all thresholds in YAML config
2. **Zero negative-context violations** — hard filter, not soft penalty
3. **Generalize to unseen brands** — adding brand #9 = one JSON entry, zero code changes
4. **Live-runnable demo** — all ML inference precomputed; demo runs from static manifest

## Project Structure

```
adtech-video-pipeline/
├── config/
│   ├── pacing_rules.yaml          # Ad-break scheduling thresholds
│   └── model_versions.yaml        # Pinned ML model versions
├── data/
│   ├── brand_catalogue.json       # 8 synthetic brands (data, not code)
│   └── sample_videos/             # Drop video files here
├── schemas/
│   ├── scene.schema.json          # JSON Schema for scene objects
│   ├── break_candidate.schema.json # JSON Schema for break candidates
│   └── brand.schema.json          # JSON Schema for brand entries
├── pipeline/
│   ├── __init__.py
│   ├── shot_detect.py             # Phase 1: PySceneDetect
│   ├── scene_segment.py           # Phase 1: merge shots → scenes
│   ├── scene_understand.py        # Phase 1: zero-shot activity tagging
│   ├── cut_safety.py              # Phase 2: "Where" — cut-safety scoring
│   ├── pacing_engine.py           # Phase 3: "Whether" — rules engine
│   ├── brand_matcher.py           # Phase 4: "What" — hard-block + rank
│   ├── vmap_builder.py            # Phase 5: IAB VMAP 1.0.1 XML
│   └── run_pipeline.py            # CLI entrypoint: video → manifest + debug
├── demo/
│   └── index.html                 # video.js player consuming manifest
├── scripts/
│   ├── validate_schemas.py        # Validate data/ against schemas/
│   ├── eval_holdout.py            # Held-out evaluation harness
│   └── check_no_hardcoding.sh     # Grep-fail on literal brand names
├── tests/
│   ├── test_schemas.py            # Phase 0 acceptance tests
│   ├── test_cut_safety.py
│   ├── test_pacing_engine.py
│   ├── test_brand_matcher.py
│   └── test_new_brand_generalization.py
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install jsonschema pyyaml pytest

# Validate schemas (Phase 0 acceptance)
python scripts/validate_schemas.py

# Run tests
pytest tests/test_schemas.py -v
```

## Brand Catalogue

The catalogue ships with 8 synthetic brands across varied categories. **No real company names are used.** Adding a new brand is a single JSON entry — no code changes required.

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
