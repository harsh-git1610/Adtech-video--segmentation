#!/usr/bin/env python3
"""
Phase 5 Integration Runner: VMAP & Debug Emitter

Reads:
  - data/processed/<video_id>/scenes.json (Phase 1 / 4)
  - data/processed/<video_id>/break_candidates.json (Phase 2 / 3)
  - data/processed/<video_id>/brand_matches.json (Phase 4)
  - data/brand_catalogue.json

Emits:
  - data/processed/<video_id>/manifest.xml (IAB VMAP 1.0.1 compliant)
  - data/processed/<video_id>/debug.json (Complete pipeline trace)

Usage:
  python scripts/run_phase5_pipeline.py <video_id>
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
logger = logging.getLogger("phase5")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.vmap_builder import build_vmap_manifest
from pipeline.debug_emitter import assemble_debug_trace


def main():
    parser = argparse.ArgumentParser(description="Phase 5: VMAP Manifest & Debug Trace Generation")
    parser.add_argument("video_id", help="Video ID (subfolder under data/processed/)")
    parser.add_argument("--catalogue", default=None, help="Override brand catalogue path")
    args = parser.parse_args()

    processed_dir = REPO_ROOT / "data" / "processed" / args.video_id
    if not processed_dir.exists():
        logger.error("Directory not found: %s", processed_dir)
        sys.exit(1)

    # 1. Load scenes.json
    scenes_path = processed_dir / "scenes.json"
    if not scenes_path.exists():
        logger.error("scenes.json missing at %s", scenes_path)
        sys.exit(1)
    with open(scenes_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # 2. Load break_candidates.json
    candidates_path = processed_dir / "break_candidates.json"
    if not candidates_path.exists():
        logger.error("break_candidates.json missing at %s", candidates_path)
        sys.exit(1)
    with open(candidates_path, "r", encoding="utf-8") as f:
        break_candidates = json.load(f)

    # 3. Load brand_matches.json
    brand_matches_path = processed_dir / "brand_matches.json"
    brand_matches = []
    if brand_matches_path.exists():
        with open(brand_matches_path, "r", encoding="utf-8") as f:
            brand_matches = json.load(f)

    # 4. Load brand catalogue to map creative assets and metadata
    catalogue_path = Path(args.catalogue) if args.catalogue else REPO_ROOT / "data" / "brand_catalogue.json"
    catalogue_by_id = {}
    if catalogue_path.exists():
        with open(catalogue_path, "r", encoding="utf-8") as f:
            for b in json.load(f):
                catalogue_by_id[b["brand_id"]] = b

    # Map matched brands into approved candidates
    match_map = {m["candidate_id"]: m for m in brand_matches if "candidate_id" in m}
    approved_breaks = []

    for cand in break_candidates:
        if cand.get("final_decision") == "BREAK_APPROVED":
            cid = cand.get("candidate_id")
            m_info = match_map.get(cid, {})
            top_brand_id = m_info.get("top_brand")
            brand_meta = catalogue_by_id.get(top_brand_id, {})

            cand["matched_brand"] = {
                "brand_id": top_brand_id or "brand_generic",
                "display_name": brand_meta.get("display_name", top_brand_id or "Sponsored Ad"),
                "creative_asset": brand_meta.get("creative_asset", "assets/ads/default.mp4"),
                "duration_sec": brand_meta.get("duration_sec", 15.0)
            }
            approved_breaks.append(cand)

    logger.info("Found %d approved breaks out of %d total candidates", len(approved_breaks), len(break_candidates))

    # 5. Build and validate manifest.xml
    manifest_path = processed_dir / "manifest.xml"
    logger.info("Building IAB VMAP 1.0.1 manifest at %s...", manifest_path)
    build_vmap_manifest(
        approved_breaks=approved_breaks,
        output_path=str(manifest_path),
        validate=True
    )
    logger.info("VMAP manifest successfully generated and validated against XSD!")

    # 6. Build debug.json
    debug_path = processed_dir / "debug.json"
    logger.info("Assembling full debug trace at %s...", debug_path)
    assemble_debug_trace(
        video_id=args.video_id,
        scenes=scenes,
        break_candidates=break_candidates,
        brand_matches=brand_matches,
        output_path=str(debug_path)
    )
    logger.info("Debug trace successfully emitted: %s", debug_path)


if __name__ == "__main__":
    main()
