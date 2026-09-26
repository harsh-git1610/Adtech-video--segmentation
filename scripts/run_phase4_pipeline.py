#!/usr/bin/env python3
"""
Phase 4 Integration Runner
Reads Phase 3 output (approved break candidates) + Phase 1 scenes.json,
runs scene_understand.py (if activity_tags missing), then brand_matcher.py,
and writes the full ranked match result to data/processed/<video_id>/brand_matches.json.

Usage:
    python scripts/run_phase4_pipeline.py <video_id> [--video-path PATH] [--out-path PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("phase4")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.scene_understand import tag_scenes
from pipeline.brand_matcher import BrandMatcher


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Brand Matching")
    parser.add_argument("video_id", help="Video ID (subfolder under data/processed/)")
    parser.add_argument("--video-path", default=None, help="Path to source video file (for keyframe extraction)")
    parser.add_argument("--out-path", default=None, help="Override output JSON path")
    parser.add_argument("--catalogue", default=None, help="Override brand catalogue path")
    parser.add_argument("--block-threshold", type=float, default=0.62)
    args = parser.parse_args()

    processed_dir = REPO_ROOT / "data" / "processed" / args.video_id

    # ── Load scenes.json ────────────────────────────────────────────────
    scenes_path = processed_dir / "scenes.json"
    if not scenes_path.exists():
        logger.error("scenes.json not found at %s — run Phase 1/2 first", scenes_path)
        sys.exit(1)

    with open(scenes_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    logger.info("Loaded %d scenes from %s", len(scenes), scenes_path)

    # ── Load break candidates (Phase 3 output) ──────────────────────────
    candidates_path = processed_dir / "break_candidates.json"
    if not candidates_path.exists():
        logger.error("break_candidates.json not found — run Phase 2/3 first")
        sys.exit(1)

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    approved = [c for c in candidates if c.get("final_decision") == "BREAK_APPROVED"]
    logger.info("%d/%d candidates approved for brand matching", len(approved), len(candidates))

    # ── Scene Understanding: populate activity_tags if missing ───────────
    if any("activity_tags" not in s or not s["activity_tags"] for s in scenes):
        logger.info("activity_tags missing on some scenes — running scene_understand...")
        tag_scenes(scenes, video_path=args.video_path)
        # Write updated scenes back
        with open(scenes_path, "w", encoding="utf-8") as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        logger.info("Updated scenes.json with activity_tags")
    else:
        logger.info("activity_tags already present on all scenes, skipping scene_understand")

    # ── Brand Matching ───────────────────────────────────────────────────
    matcher = BrandMatcher(
        catalogue_path=args.catalogue,
        block_threshold=args.block_threshold
    )
    results = matcher.match_all(approved, scenes)

    # ── Write Output ─────────────────────────────────────────────────────
    out_path = Path(args.out_path) if args.out_path else processed_dir / "brand_matches.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Brand matches written to %s", out_path)
    logger.info(
        "Summary: %d breaks matched | top brands: %s",
        len(results),
        [r.get("top_brand") for r in results]
    )


if __name__ == "__main__":
    main()
