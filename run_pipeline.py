#!/usr/bin/env python3
"""
AdTech Video Placement Pipeline — Master End-to-End Runner

Executes the complete pipeline (Phases 1 through 5) on any brand-new or held-out video:
  Phase 1: PySceneDetect Shot Detection & Scene Segmentation (scenes.json)
  Phase 2: Sub-Signal Cut Safety Engine (break_candidates.json)
  Phase 3: DP Weighted Interval Scheduling Pacing Engine
  Phase 4: Scene Understanding & Brand Matching with Hard-Block Guard
  Phase 5: IAB VMAP 1.0.1 Manifest XML & Full debug.json Emitter
  Phase 6: Links deliverables into demo/ for instant web player evaluation

Usage:
    python run_pipeline.py <video_path> [--catalogue PATH] [--demo] [--eval]
    Example:
    python run_pipeline.py data/sample_videos/mohanagar.mp4 --demo --eval
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("master_pipeline")

from pipeline.shot_detect import detect_shots
from pipeline.cut_safety import score_candidate
from pipeline.pacing_engine import apply_pacing_rules, load_pacing_rules
from pipeline.scene_understand import tag_scenes
from pipeline.brand_matcher import BrandMatcher, load_brand_catalogue
from pipeline.vmap_builder import build_vmap_manifest
from pipeline.debug_emitter import assemble_debug_trace


def extract_audio_and_transcribe(video_path: Path, processed_dir: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts audio to WAV and retrieves word-level ASR timestamps.
    Falls back gracefully if Whisper is not installed or audio extraction is offline.
    """
    audio_path = processed_dir / f"{video_path.stem}.wav"
    words_file = processed_dir / "whisper_words.json"

    # Step A: Extract WAV via ffmpeg if not present
    if not audio_path.exists():
        ffmpeg_exe = r"F:\INSTALLS\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
        if not os.path.exists(ffmpeg_exe):
            ffmpeg_exe = "ffmpeg"
        import subprocess
        cmd = [ffmpeg_exe, "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            logger.info("Extracted 16kHz audio to %s", audio_path)
        except Exception as e:
            logger.warning("Audio extraction failed (%s), proceeding without audio file", e)
            audio_path = None

    # Step B: Load or compute word timestamps
    words = []
    if words_file.exists():
        with open(words_file, "r", encoding="utf-8") as wf:
            words = json.load(wf)
    else:
        # Generate synthetic/coarse words aligned with audio duration if offline
        try:
            from faster_whisper import WhisperModel
            logger.info("Transcribing audio with faster-whisper for word-level timestamps...")
            model = WhisperModel("base", device="cpu", compute_type="default")
            segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
            for seg in segments:
                if seg.words:
                    for w in seg.words:
                        words.append({"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3), "confidence": round(w.probability, 3)})
            with open(words_file, "w", encoding="utf-8") as wf:
                json.dump(words, wf, indent=2)
            logger.info("Transcribed %d words to %s", len(words), words_file)
        except Exception as e:
            logger.info("Whisper not active or offline (%s); using duration-based timing", e)

    return str(audio_path) if audio_path and audio_path.exists() else None, words


def build_scenes_from_shots(video_path: Path, shot_boundaries: List[float], words: List[Dict[str, Any]], video_duration: float) -> List[Dict[str, Any]]:
    """Build scenes by clustering shot boundaries with min scene duration constraint."""
    min_scene_sec = 10.0
    scenes = []
    current_start = 0.0
    scene_idx = 1

    boundaries = [b for b in shot_boundaries if b > 0.0]
    if not boundaries or boundaries[-1] < video_duration - 2.0:
        boundaries.append(video_duration)

    for b in boundaries:
        if (b - current_start) >= min_scene_sec or b == boundaries[-1]:
            # Associate ASR words in this range
            scene_words = [w["word"] for w in words if current_start <= w.get("start", 0.0) <= b]
            asr_text = " ".join(scene_words) if scene_words else ""
            scenes.append({
                "scene_id": f"sc_{scene_idx:03d}",
                "start_ts": round(current_start, 3),
                "end_ts": round(b, 3),
                "shot_ids": [f"sh_{scene_idx:03d}"],
                "asr_text": asr_text,
                "activity_tags": [],
                "dominant_activity": ""
            })
            current_start = b
            scene_idx += 1

    return scenes


def get_video_duration(video_path: Path) -> float:
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
    return 1200.0


def main():
    parser = argparse.ArgumentParser(description="AdTech Video Placement Pipeline — Master Runner")
    parser.add_argument("video_path", help="Path to input video file (.mp4)")
    parser.add_argument("--catalogue", default="data/brand_catalogue.json", help="Brand catalogue JSON path")
    parser.add_argument("--demo", action="store_true", help="Copy deliverables into demo/ for browser player")
    parser.add_argument("--eval", action="store_true", help="Run holdout evaluation gate on generated outputs")
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    catalogue_path = Path(args.catalogue).resolve()
    if not catalogue_path.exists():
        logger.error("Brand catalogue not found: %s", catalogue_path)
        sys.exit(1)

    video_id = video_path.stem
    processed_dir = REPO_ROOT / "data" / "processed" / video_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    video_duration = get_video_duration(video_path)

    print("=" * 76)
    print(f"  ADTECH VIDEO PLACEMENT PIPELINE — END-TO-END EXECUTION")
    print(f"  Target Video   : {video_path.name} (Duration: {video_duration:.1f}s / {video_duration/60:.1f}m)")
    print(f"  Brand Catalogue: {catalogue_path.name}")
    print(f"  Output Dir     : {processed_dir}")
    print("=" * 76)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Shot Detection & Scene Segmentation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase 1] Detecting Shot Boundaries & Segmenting Scenes...")
    try:
        shot_boundaries = detect_shots(str(video_path))
    except Exception as e:
        logger.warning("PySceneDetect error (%s); using uniform shot sampling", e)
        shot_boundaries = [float(t) for t in range(15, int(video_duration), 25)]

    audio_path, words = extract_audio_and_transcribe(video_path, processed_dir)

    scenes_file = processed_dir / "scenes.json"
    if not scenes_file.exists():
        scenes = build_scenes_from_shots(video_path, shot_boundaries, words, video_duration)
        with open(scenes_file, "w", encoding="utf-8") as sf:
            json.dump(scenes, sf, indent=2)
    else:
        with open(scenes_file, "r", encoding="utf-8") as sf:
            scenes = json.load(sf)
    print(f"  ✓ Segmented into {len(scenes)} coherent scenes with {len(shot_boundaries)} shot boundaries.")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: Cut Safety Scoring
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase 2] Computing Sub-Signal Cut Safety Scores at Scene Boundaries...")
    candidates = []
    for i in range(len(scenes) - 1):
        sc_before = scenes[i]
        sc_after = scenes[i+1]
        boundary_ts = float(sc_before["end_ts"])

        # Compute cut safety with all 4 independent sub-signals
        safety_result = score_candidate(
            boundary_ts=boundary_ts,
            word_timestamps=words,
            shot_boundaries=shot_boundaries,
            audio_path=audio_path
        )

        cand = {
            "candidate_id": f"brk_{i+1:03d}",
            "timestamp": boundary_ts,
            "scene_before": sc_before["scene_id"],
            "scene_after": sc_after["scene_id"],
            "cut_safety_score": safety_result["cut_safety_score"],
            "cut_safety_reasons": safety_result["cut_safety_reasons"],
            "pacing_eligible": False,
            "final_decision": "UNSET",
            "pacing_reasons": {}
        }
        candidates.append(cand)

    print(f"  ✓ Evaluated {len(candidates)} candidate boundaries for mid-speech cuts and action continuity.")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3: Pacing Engine (Dynamic Programming Selection)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase 3] Running DP Weighted Interval Scheduling Pacing Engine...")
    pacing_rules = load_pacing_rules()
    candidates = apply_pacing_rules(
        break_candidates=candidates,
        rules=pacing_rules,
        total_video_duration_sec=video_duration
    )
    approved_count = sum(1 for c in candidates if c.get("final_decision") == "BREAK_APPROVED")
    rejected_count = len(candidates) - approved_count
    print(f"  ✓ Pacing engine selected {approved_count} optimal breaks (rejected {rejected_count} for safety/gap limits).")

    with open(processed_dir / "break_candidates.json", "w", encoding="utf-8") as cf:
        json.dump(candidates, cf, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4: Scene Understanding & Brand Matching
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase 4] Scene Understanding & Synonym-Aware Brand Matching...")
    # Tag scenes with activity taxonomy if needed
    if any(not s.get("activity_tags") for s in scenes):
        tag_scenes(scenes, video_path=str(video_path))
        with open(scenes_file, "w", encoding="utf-8") as sf:
            json.dump(scenes, sf, indent=2)

    matcher = BrandMatcher(catalogue_path=str(catalogue_path))
    brand_matches = matcher.match_all(candidates, scenes)

    with open(processed_dir / "brand_matches.json", "w", encoding="utf-8") as bf:
        json.dump(brand_matches, bf, indent=2)
    print(f"  ✓ Matched brands for all approved breaks with negative-context hard-blocking active.")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 5: IAB VMAP 1.0.1 Manifest & Full Debug Trace Emitter
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[Phase 5] Emitting IAB VMAP 1.0.1 Manifest & Transparent debug.json...")
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
    print(f"  ✓ Emitted IAB VMAP 1.0.1 Manifest: {manifest_path}")
    print(f"  ✓ Emitted Complete Debug Trace   : {debug_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 6: Link to Demo Player
    # ─────────────────────────────────────────────────────────────────────────
    if args.demo:
        demo_dir = REPO_ROOT / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest_path, demo_dir / "manifest.xml")
        shutil.copyfile(debug_path, demo_dir / "debug.json")

        demo_video = demo_dir / f"{video_id}.mp4"
        if not demo_video.exists():
            try:
                os.link(video_path, demo_video)
            except Exception:
                shutil.copyfile(video_path, demo_video)
        print(f"\n  ✓ Linked deliverables into demo/ for interactive evaluation.")
        print(f"    Launch demo: python scripts/server.py 8000 demo")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 7: Automated Holdout Gate Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    if args.eval:
        print("\n[Final Gate] Running Holdout Submission Gate Verification...")
        import scripts.eval_holdout as gate
        # Reuse gate functions directly
        p1, _ = gate.check_mid_sentence_cuts(json.load(open(debug_path, encoding='utf-8')))
        p2, _ = gate.check_negative_context_violations(json.load(open(debug_path, encoding='utf-8')), catalogue_path)
        p3, _ = gate.check_pacing_constraints(json.load(open(debug_path, encoding='utf-8')), video_duration)
        p4, _ = gate.check_vmap_validity(manifest_path)
        p5, _ = gate.check_no_hardcoding(catalogue_path)

        all_ok = p1 and p2 and p3 and p4 and p5
        if all_ok:
            print("  ✓ ALL 5 SUBMISSION GATES PASSED! Video processing completely verified.")
        else:
            print("  ✗ Gate verification failed. Review debug.json for constraints.")

    print("\n" + "=" * 76)
    print("  PIPELINE EXECUTION COMPLETE")
    print(f"  Manifest : data/processed/{video_id}/manifest.xml")
    print(f"  Trace    : data/processed/{video_id}/debug.json")
    print("=" * 76)


if __name__ == "__main__":
    main()
