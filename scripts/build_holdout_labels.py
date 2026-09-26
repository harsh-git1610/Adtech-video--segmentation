#!/usr/bin/env python3
"""
Holdout Ground-Truth Labeling Utility (Phase 2 & Phase 7)

Assists in quickly labeling a set of real scene/shot boundaries from any new or held-out video
as 'safe' or 'jarring' for cut-safety evaluation.

Reuses the exact format from tests/labeled_boundaries.json:
[
  {
    "video_id": "bhojon_bilashi",
    "timestamp": 16.20,
    "label": "safe",
    "rationale": "Speech concluded before shot change, natural scene transition"
  }
]

Usage:
    # 1. Interactive terminal labeling of detected boundaries:
    python scripts/build_holdout_labels.py data/sample_videos/mohanagar.mp4

    # 2. Add a single label directly via CLI:
    python scripts/build_holdout_labels.py data/sample_videos/mohanagar.mp4 --add 14.50 safe "Clean pause between dialogue exchanges"

    # 3. Automatically evaluate precision/recall after labeling:
    python scripts/build_holdout_labels.py data/sample_videos/mohanagar.mp4 --evaluate
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LABELED_FILE = REPO_ROOT / "tests" / "labeled_boundaries.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_existing_labels(file_path: Path) -> List[Dict[str, Any]]:
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_labels(labels: List[Dict[str, Any]], file_path: Path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved {len(labels)} boundary labels to {file_path}")


def get_candidate_timestamps(video_id: str) -> List[float]:
    """Retrieve candidate boundary timestamps from processed data or scenes."""
    candidates_file = PROCESSED_DIR / video_id / "break_candidates.json"
    if candidates_file.exists():
        with open(candidates_file, "r", encoding="utf-8") as f:
            return [round(float(c.get("timestamp", 0.0)), 2) for c in json.load(f)]

    scenes_file = PROCESSED_DIR / video_id / "scenes.json"
    if scenes_file.exists():
        with open(scenes_file, "r", encoding="utf-8") as f:
            scenes = json.load(f)
            return [round(float(s.get("end_ts", 0.0)), 2) for s in scenes[:-1]]

    # Fallback default boundary candidates
    return [15.5, 32.0, 58.5, 105.0, 142.0, 195.0, 255.0]


def main():
    parser = argparse.ArgumentParser(description="Quick Ground-Truth Boundary Labeling Tool")
    parser.add_argument("video_path", help="Path to video file (.mp4)")
    parser.add_argument("--out", default=str(LABELED_FILE), help="Output JSON path to save labels")
    parser.add_argument("--add", nargs=3, metavar=("TIMESTAMP", "LABEL", "RATIONALE"),
                        help="Add a single label directly without interactive prompts")
    parser.add_argument("--evaluate", action="store_true", help="Run cut-safety precision/recall eval after labeling")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    video_id = video_path.stem
    out_path = Path(args.out)

    labels = load_existing_labels(out_path)

    # Mode 1: Direct single label addition via CLI
    if args.add:
        ts = round(float(args.add[0]), 2)
        lbl = args.add[1].lower().strip()
        if lbl not in ("safe", "jarring"):
            print(f"Error: Label must be 'safe' or 'jarring', got '{lbl}'")
            sys.exit(1)
        rat = args.add[2].strip()

        # Check if already labeled
        existing_idx = next((i for i, item in enumerate(labels) if item.get("video_id") == video_id and abs(item.get("timestamp", 0.0) - ts) < 0.1), None)
        entry = {"video_id": video_id, "timestamp": ts, "label": lbl, "rationale": rat}

        if existing_idx is not None:
            labels[existing_idx] = entry
            print(f"Updated existing label at {ts}s for {video_id}")
        else:
            labels.append(entry)
            print(f"Added new label at {ts}s for {video_id}: [{lbl.upper()}] {rat}")

        save_labels(labels, out_path)

    # Mode 2: Interactive session
    elif not args.evaluate:
        print("=" * 64)
        print(f"Interactive Ground-Truth Labeling for: {video_id}")
        print("Commands: [s]afe, [j]arring, [skip], [q]uit")
        print("=" * 64)

        candidates = get_candidate_timestamps(video_id)
        existing_ts = {round(item["timestamp"], 2) for item in labels if item.get("video_id") == video_id}

        unlabeled = [ts for ts in candidates if ts not in existing_ts]
        print(f"Found {len(candidates)} total candidates ({len(unlabeled)} unlabeled).\n")

        for ts in unlabeled:
            print(f"\n--- Candidate Boundary at {ts:.2f}s ---")
            ans = input("Label (s=safe / j=jarring / skip / q=quit) [s]: ").strip().lower()
            if ans == 'q':
                break
            if ans in ('skip', 'k'):
                continue

            lbl = "jarring" if ans in ('j', 'jarring') else "safe"
            rat = input(f"Rationale for {lbl.upper()}: ").strip()
            if not rat:
                rat = "Full sentence conclusion with stable shot transition" if lbl == "safe" else "Mid-sentence dialogue cut"

            labels.append({
                "video_id": video_id,
                "timestamp": ts,
                "label": lbl,
                "rationale": rat
            })
            save_labels(labels, out_path)

    # Mode 3: Optional evaluation trigger
    if args.evaluate:
        print("\nRunning Cut-Safety Precision / Recall Evaluation...")
        from scripts.eval_cut_safety import evaluate_cut_safety
        results = evaluate_cut_safety(threshold=0.55, verbose=True)
        print(f"\nEval Results: Precision={results['precision']:.3f} | Recall={results['recall']:.3f} | F1={results['f1']:.3f}")


if __name__ == "__main__":
    main()
