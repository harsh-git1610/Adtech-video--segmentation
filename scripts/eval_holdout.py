#!/usr/bin/env python3
"""
Submission Gate & Holdout Evaluation Script (Phase 7 / Final Gate)

Performs comprehensive end-to-end evaluation and verification of the pipeline:
1. Runs full pipeline (Phases 1-5) on the specified video and brand catalogue.
2. Checks for mid-sentence cut violations (target: 0; any violation is fatal).
3. Checks for negative-context brand safety violations (target: 0; scored-zero criterion).
4. Verifies all pacing rules (min gap, max breaks/hour, max ad seconds, buffers, cut floor).
5. Validates generated manifest.xml against IAB VMAP 1.0.1 XSD.
6. Enforces zero hardcoding of brand names/identifiers in pipeline code.
7. Emits a clean PASS/FAIL summary with exit code 0 only if ALL checks pass.

Usage:
    python scripts/eval_holdout.py <video_path> [--catalogue PATH] [--skip-pipeline]
    Example: python scripts/eval_holdout.py data/sample_videos/bhojon_bilashi.mp4
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.vmap_builder import validate_vmap, build_vmap_manifest
from pipeline.debug_emitter import assemble_debug_trace
from pipeline.brand_matcher import BrandMatcher, load_brand_catalogue
from pipeline.pacing_engine import load_pacing_rules, apply_pacing_rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_holdout")


def get_video_duration(video_path: Path) -> float:
    """Extract video duration in seconds via OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if frames > 0 and fps > 0:
            return float(frames / fps)
    except Exception:
        pass
    return 1200.0  # safe default fallback (~20 mins)


def run_pipeline_end_to_end(video_path: Path, catalogue_path: Path, processed_dir: Path):
    """
    Executes Phases 1-5 end-to-end. If processed data already exists,
    ensures all artifacts (scenes, candidates, brand_matches, manifest.xml, debug.json)
    are refreshed and strictly synchronized.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    video_id = video_path.stem

    scenes_file = processed_dir / "scenes.json"
    candidates_file = processed_dir / "break_candidates.json"

    # If scenes or candidates do not exist, run prepare script or generate base candidates
    if not scenes_file.exists() or not candidates_file.exists():
        logger.info("Initializing baseline scenes and candidates for %s...", video_id)
        import scripts.prepare_demo_dataset as prep
        prep.main()

    with open(scenes_file, "r", encoding="utf-8") as f:
        scenes = json.load(f)
    with open(candidates_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    # Phase 3: Pacing Engine
    video_duration = get_video_duration(video_path)
    logger.info("Running Phase 3 Pacing Engine (duration=%.1fs)...", video_duration)
    candidates = apply_pacing_rules(candidates, total_video_duration_sec=video_duration)

    # Phase 4: Brand Matching
    logger.info("Running Phase 4 Brand Matching against catalogue...")
    matcher = BrandMatcher(catalogue_path=str(catalogue_path))
    brand_matches = matcher.match_all(candidates, scenes)

    with open(processed_dir / "brand_matches.json", "w", encoding="utf-8") as f:
        json.dump(brand_matches, f, indent=2)

    # Phase 5: VMAP Manifest & Debug Trace
    catalogue_by_id = {b["brand_id"]: b for b in load_brand_catalogue(str(catalogue_path))}
    match_map = {m["candidate_id"]: m for m in brand_matches}
    approved_breaks = []

    for cand in candidates:
        if cand.get("final_decision") == "BREAK_APPROVED":
            cid = cand.get("candidate_id")
            m_info = match_map.get(cid, {})
            top_brand_id = m_info.get("top_brand")
            brand_meta = catalogue_by_id.get(top_brand_id, {})
            cand["matched_brand"] = {
                "brand_id": top_brand_id,
                "display_name": brand_meta.get("display_name", top_brand_id),
                "creative_asset": brand_meta.get("creative_asset", "assets/ads/default.mp4"),
                "duration_sec": brand_meta.get("duration_sec", 15.0)
            }
            approved_breaks.append(cand)

    manifest_path = processed_dir / "manifest.xml"
    debug_path = processed_dir / "debug.json"

    build_vmap_manifest(approved_breaks, output_path=str(manifest_path), validate=False)
    assemble_debug_trace(video_id, scenes, candidates, brand_matches, output_path=str(debug_path))
    logger.info("End-to-end pipeline run complete. Artifacts written to %s", processed_dir)


def check_mid_sentence_cuts(debug_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check that NO approved break is placed midway through spoken dialogue."""
    violations = []
    for cand in debug_data.get("break_candidates", []):
        pacing = cand.get("pacing", {})
        if pacing.get("final_decision") == "BREAK_APPROVED":
            cut_reasons = cand.get("cut_safety", {}).get("reasons", {})
            if cut_reasons.get("sentence_boundary_aligned") is False:
                violations.append(
                    f"Candidate {cand.get('candidate_id')} at {cand.get('timestamp')}s: "
                    f"approved cut occurs mid-sentence (sentence_boundary_aligned=False)"
                )
    return len(violations) == 0, violations


def check_negative_context_violations(
    debug_data: Dict[str, Any], catalogue_path: Path
) -> Tuple[bool, List[str]]:
    """
    Check that NO approved brand placement violates negative context constraints.
    Treat any failure as a hard blocking disqualifier.
    """
    violations = []
    with open(catalogue_path, "r", encoding="utf-8") as f:
        catalogue = json.load(f)
    catalogue_by_id = {b["brand_id"]: b for b in catalogue}

    for cand in debug_data.get("break_candidates", []):
        pacing = cand.get("pacing", {})
        if pacing.get("final_decision") == "BREAK_APPROVED":
            bs = cand.get("brand_selection", {})
            matched = bs.get("matched_brand", {})
            brand_id = matched.get("brand_id") if matched else None
            cid = cand.get("candidate_id", "unknown")
            ts = cand.get("timestamp", 0.0)

            if not brand_id or brand_id not in catalogue_by_id:
                violations.append(f"Candidate {cid}: approved break has missing or invalid matched_brand")
                continue

            # 1. Verify candidate brand was never in blocked_brands list
            blocked_brands = bs.get("blocked_brands", [])
            blocked_ids = {b.get("brand_id") for b in blocked_brands}
            if brand_id in blocked_ids:
                reason = next((b.get("block_reason") for b in blocked_brands if b.get("brand_id") == brand_id), "hard-blocked")
                violations.append(
                    f"Candidate {cid} at {ts}s: Brand '{brand_id}' was marked as BLOCKED in trace but still selected! "
                    f"Reason: {reason}"
                )

            # 2. Direct semantic check against brand's negative_contexts
            brand_obj = catalogue_by_id[brand_id]
            negative_contexts = [nc.lower().strip() for nc in brand_obj.get("negative_contexts", [])]
            activity_tags = bs.get("activity_tags_used", [])

            for tag in activity_tags:
                tag_label = tag.get("label", "").lower().strip()
                conf = float(tag.get("confidence", 0.0))
                if conf < 0.30 or not tag_label:
                    continue

                for neg in negative_contexts:
                    # Check exact phrase, substring, or keyword intersection
                    if neg in tag_label or tag_label in neg:
                        violations.append(
                            f"Candidate {cid} at {ts}s: Brand '{brand_id}' placed in unsafe scene! "
                            f"Tag '{tag_label}' (conf={conf:.2f}) matches negative_context '{neg}'"
                        )
                        break

    return len(violations) == 0, violations


def check_pacing_constraints(debug_data: Dict[str, Any], video_duration: float) -> Tuple[bool, List[str]]:
    """Validate all constraints from pacing_rules.yaml against approved breaks."""
    violations = []
    rules = load_pacing_rules()

    min_gap = float(rules.get("min_gap_between_breaks_sec", 120.0))
    max_breaks_hr = int(rules.get("max_breaks_per_hour", 4))
    max_ad_sec_hr = float(rules.get("max_ad_seconds_per_hour", 120.0))
    max_break_dur = float(rules.get("max_ad_duration_per_break_sec", 30.0))
    floor = float(rules.get("cut_safety_floor", 0.55))
    first_buf = float(rules.get("first_break_buffer_sec", 180.0))
    last_buf = float(rules.get("last_break_buffer_sec", 180.0))

    approved = [
        c for c in debug_data.get("break_candidates", [])
        if c.get("pacing", {}).get("final_decision") == "BREAK_APPROVED"
    ]
    approved.sort(key=lambda x: float(x.get("timestamp", 0.0)))

    # 1. Cut safety floor
    for c in approved:
        score = c.get("cut_safety", {}).get("score", 0.0)
        if score < floor:
            violations.append(f"Candidate {c.get('candidate_id')} score {score:.3f} < cut_safety_floor {floor:.2f}")

    # 2. Min gap constraint
    for i in range(1, len(approved)):
        gap = float(approved[i]["timestamp"]) - float(approved[i-1]["timestamp"])
        if gap < min_gap:
            violations.append(
                f"Min gap violation between {approved[i-1]['candidate_id']} ({approved[i-1]['timestamp']}s) "
                f"and {approved[i]['candidate_id']} ({approved[i]['timestamp']}s): gap={gap:.1f}s < {min_gap}s"
            )

    # 3. Rolling 1-hour window breaks count and ad seconds
    timestamps = [float(c["timestamp"]) for c in approved]
    durations = [
        float(c.get("brand_selection", {}).get("matched_brand", {}).get("duration_sec", 15.0) or 15.0)
        for c in approved
    ]

    for i in range(len(timestamps)):
        t_start = timestamps[i]
        t_end = t_start + 3600.0
        in_window_indices = [k for k in range(i, len(timestamps)) if timestamps[k] < t_end]

        if len(in_window_indices) > max_breaks_hr:
            violations.append(f"Rolling 1hr window from {t_start:.1f}s has {len(in_window_indices)} breaks > max {max_breaks_hr}")

        total_ad_sec = sum(durations[k] for k in in_window_indices)
        if total_ad_sec > max_ad_sec_hr:
            violations.append(f"Rolling 1hr window from {t_start:.1f}s has {total_ad_sec} ad seconds > max {max_ad_sec_hr}s")

    # 4. Individual break max duration
    for c, dur in zip(approved, durations):
        if dur > max_break_dur:
            violations.append(f"Candidate {c.get('candidate_id')} ad duration {dur}s > max {max_break_dur}s")

    # 5. Buffer checks (applied only when video is long enough)
    if video_duration > (first_buf + last_buf + min_gap):
        for c in approved:
            ts = float(c["timestamp"])
            if ts < first_buf:
                violations.append(f"Candidate {c.get('candidate_id')} timestamp {ts}s < first_break_buffer {first_buf}s")
            if ts > (video_duration - last_buf):
                violations.append(f"Candidate {c.get('candidate_id')} timestamp {ts}s > end buffer {video_duration - last_buf}s")

    return len(violations) == 0, violations


def check_vmap_validity(manifest_path: Path) -> Tuple[bool, List[str]]:
    """Validate manifest.xml against IAB VMAP 1.0.1 XSD."""
    if not manifest_path.exists():
        return False, [f"Manifest file not found: {manifest_path}"]
    try:
        validate_vmap(str(manifest_path))
        return True, []
    except Exception as e:
        return False, [f"VMAP validation error: {e}"]


def check_no_hardcoding(catalogue_path: Path) -> Tuple[bool, List[str]]:
    """Ensure zero brand names/IDs are hardcoded in pipeline/*.py."""
    violations = []
    try:
        with open(catalogue_path, "r", encoding="utf-8") as f:
            brands = json.load(f)

        terms = set()
        for b in brands:
            terms.add(b["brand_id"].lower())
            display = b.get("display_name", "").replace(" ", "").lower()
            if len(display) > 3:
                terms.add(display)

        pipeline_dir = REPO_ROOT / "pipeline"
        for py_file in pipeline_dir.glob("**/*.py"):
            with open(py_file, "r", encoding="utf-8") as pf:
                code_lower = pf.read().lower()
            for t in terms:
                if t in code_lower:
                    violations.append(f"Hardcoded brand term '{t}' found in {py_file.name}")
    except Exception as e:
        violations.append(f"Error checking hardcoding: {e}")

    return len(violations) == 0, violations


def main():
    parser = argparse.ArgumentParser(description="AdTech Pipeline Holdout Evaluation Gate")
    parser.add_argument("video_path", help="Path to video file (.mp4)")
    parser.add_argument("--catalogue", default="data/brand_catalogue.json", help="Brand catalogue path")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip running pipeline if artifacts exist")
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    catalogue_path = Path(args.catalogue).resolve()
    if not catalogue_path.exists():
        logger.error("Catalogue file not found: %s", catalogue_path)
        sys.exit(1)

    video_id = video_path.stem
    processed_dir = REPO_ROOT / "data" / "processed" / video_id

    print("=" * 72)
    print("ADTECH PIPELINE SUBMISSION GATE - HOLDOUT EVALUATION")
    print(f"Target Video    : {video_path.name}")
    print(f"Brand Catalogue : {catalogue_path.name}")
    print(f"Output Directory: {processed_dir}")
    print("=" * 72)

    # 1. Run pipeline end-to-end
    if not args.skip_pipeline or not (processed_dir / "debug.json").exists():
        run_pipeline_end_to_end(video_path, catalogue_path, processed_dir)

    # Load debug.json
    debug_path = processed_dir / "debug.json"
    manifest_path = processed_dir / "manifest.xml"

    with open(debug_path, "r", encoding="utf-8") as f:
        debug_data = json.load(f)

    video_duration = get_video_duration(video_path)

    # 2. Gate Checks
    results = {}

    print("\n[Gate 1/5] Checking Mid-Sentence Cut Safety...")
    p1, v1 = check_mid_sentence_cuts(debug_data)
    results["mid_sentence_cuts"] = (p1, v1)
    if p1:
        print("  [OK] PASS: Zero mid-sentence cuts detected across all approved breaks.")
    else:
        print(f"  [X] FAIL: {len(v1)} mid-sentence cuts detected:")
        for msg in v1:
            print(f"    - {msg}")

    print("\n[Gate 2/5] Checking Negative-Context Brand Safety...")
    p2, v2 = check_negative_context_violations(debug_data, catalogue_path)
    results["brand_safety"] = (p2, v2)
    if p2:
        print("  [OK] PASS: Zero negative context violations. All approved brands are context-safe.")
    else:
        print(f"  [X] FAIL: {len(v2)} negative context violations found:")
        for msg in v2:
            print(f"    - {msg}")

    print("\n[Gate 3/5] Verifying Pacing Rules & Scheduling Constraints...")
    p3, v3 = check_pacing_constraints(debug_data, video_duration)
    results["pacing_rules"] = (p3, v3)
    if p3:
        print("  [OK] PASS: All pacing constraints satisfied (min gap, rolling hour limits, floor).")
    else:
        print(f"  [X] FAIL: {len(v3)} pacing violations found:")
        for msg in v3:
            print(f"    - {msg}")

    print("\n[Gate 4/5] Validating IAB VMAP 1.0.1 Manifest XML...")
    p4, v4 = check_vmap_validity(manifest_path)
    results["vmap_xsd"] = (p4, v4)
    if p4:
        print("  [OK] PASS: Manifest conforms strictly to IAB VMAP 1.0.1 specification.")
    else:
        print(f"  [X] FAIL: Manifest validation failed:")
        for msg in v4:
            print(f"    - {msg}")

    print("\n[Gate 5/5] Auditing for Hardcoded Brand Identifiers...")
    p5, v5 = check_no_hardcoding(catalogue_path)
    results["no_hardcoding"] = (p5, v5)
    if p5:
        print("  [OK] PASS: Pipeline logic contains zero literal brand names or IDs.")
    else:
        print(f"  [X] FAIL: {len(v5)} hardcoded references detected:")
        for msg in v5:
            print(f"    - {msg}")

    # Summary
    all_passed = all(r[0] for r in results.values())
    print("\n" + "=" * 72)
    if all_passed:
        print("FINAL RESULT: [ PASS ] - All submission gates satisfied. Safe to submit.")
        print("=" * 72)
        sys.exit(0)
    else:
        print("FINAL RESULT: [ FAIL ] - One or more submission gates failed. DO NOT SUBMIT.")
        print("=" * 72)
        sys.exit(1)


if __name__ == "__main__":
    main()
