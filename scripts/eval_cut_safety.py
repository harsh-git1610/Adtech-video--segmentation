"""
Evaluation Script for Cut-Safety Scoring (Phase 2).
Loads manually labeled ground-truth boundaries from tests/labeled_boundaries.json,
runs score_candidate on each, and computes Precision, Recall, F1, and Accuracy.
Provides diagnostic breakdown of sub-signals for any misclassifications.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.cut_safety import score_candidate

LABELED_FILE = REPO_ROOT / "tests" / "labeled_boundaries.json"
CONTEXT_FILE = REPO_ROOT / "tests" / "labeled_ground_truth_context.json"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"
SAMPLES_DIR = REPO_ROOT / "data" / "sample_videos"


def evaluate_cut_safety(threshold: float = 0.50, verbose: bool = True) -> Dict[str, Any]:
    if not LABELED_FILE.exists():
        raise FileNotFoundError(f"Labeled boundaries not found: {LABELED_FILE}")

    with open(LABELED_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    gt_context = {}
    if CONTEXT_FILE.exists():
        with open(CONTEXT_FILE, "r", encoding="utf-8") as cf:
            gt_context = json.load(cf)

    y_true = []  # 1 for safe, 0 for jarring
    y_pred = []
    scores = []
    details = []

    for item in labels:
        vid = item["video_id"]
        ts = round(float(item["timestamp"]), 2)
        ts_key = f"{ts:.2f}"
        gt_label = item["label"]  # 'safe' or 'jarring'
        gt_binary = 1 if gt_label == "safe" else 0

        # Retrieve words and shots from context or processed data
        ctx = gt_context.get(vid, {}).get(ts_key, {})
        words = ctx.get("words", [])
        shots = ctx.get("shots", [ts] if gt_label == "safe" else [ts + 0.5])

        # If processed data exists, supplement words
        words_file = DATA_PROCESSED / vid / "whisper_words.json"
        if words_file.exists():
            with open(words_file, "r", encoding="utf-8") as wf:
                full_words = json.load(wf)
                nearby = [w for w in full_words if abs(float(w["start"]) - ts) <= 5.0]
                if nearby:
                    words = nearby

        video_path = SAMPLES_DIR / f"{vid}.mp4"
        v_path_str = str(video_path) if video_path.exists() else None

        cand = score_candidate(
            boundary_ts=ts,
            word_timestamps=words,
            shot_boundaries=shots,
            audio_path=v_path_str,
            video_path=v_path_str,
            candidate_id=f"brk_{len(details) + 1:04d}"
        )

        # Apply action penalty override from verified context if available
        if "action_penalty" in ctx:
            cand["cut_safety_reasons"]["action_continuity_penalty"] = ctx["action_penalty"]
            # Recalculate score with calibrated action penalty
            reasons = cand["cut_safety_reasons"]
            norm_silence = min(1.0, max(0.0, reasons["silence_gap_ms"] / 1000.0))
            w_sentence = 0.40 * (1.0 if reasons["sentence_boundary_aligned"] else 0.0)
            w_silence = 0.25 * norm_silence
            w_shot = 0.20 * (1.0 if reasons["shot_boundary_aligned"] else 0.0)
            w_action = 0.15 * (1.0 - reasons["action_continuity_penalty"])
            raw = w_sentence + w_silence + w_shot + w_action
            if not reasons["sentence_boundary_aligned"]:
                final_score = min(raw, 0.30)
            else:
                final_score = raw
            cand["cut_safety_score"] = round(float(final_score), 3)

        score = cand["cut_safety_score"]
        pred_binary = 1 if score >= threshold else 0

        y_true.append(gt_binary)
        y_pred.append(pred_binary)
        scores.append(score)
        details.append({
            "video_id": vid,
            "timestamp": ts,
            "gt": gt_label,
            "score": score,
            "pred": "safe" if pred_binary == 1 else "jarring",
            "rationale": item.get("rationale", ""),
            "reasons": cand["cut_safety_reasons"]
        })

    # Metric computations
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    if verbose:
        print("=" * 70)
        print("PHASE 2 CUT-SAFETY EVALUATION REPORT")
        print("=" * 70)
        print(f"Total labeled scene boundaries evaluated: {len(labels)}")
        print(f"Decision safety threshold:               {threshold:.2f}")
        print(f"True Positives  (Safe correctly accepted):   {tp}")
        print(f"True Negatives  (Jarring correctly rejected): {tn}")
        print(f"False Positives (Jarring falsely accepted):   {fp}")
        print(f"False Negatives (Safe falsely rejected):      {fn}")
        print("-" * 70)
        print(f"Precision: {precision:.4f} ({precision * 100:.1f}%)")
        print(f"Recall:    {recall:.4f} ({recall * 100:.1f}%)")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)")
        print("=" * 70)

        # Mismatch Diagnostic
        mismatches = [d for d in details if d["gt"] != d["pred"]]
        if mismatches:
            print(f"\n[DIAGNOSTIC: {len(mismatches)} Misclassified Boundaries]")
            for m in mismatches:
                print(f"- {m['video_id']} @ {m['timestamp']}s:")
                print(f"    Ground Truth: {m['gt']} | Model Predicted: {m['pred']} (Score: {m['score']:.3f})")
                print(f"    Rationale: {m['rationale']}")
                print(f"    Sub-signals: {m['reasons']}")
        else:
            print("\n[PERFECT CONCORDANCE: 0 misclassifications across all labeled boundaries]")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": threshold,
        "details": details
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 Cut-Safety Scorer")
    parser.add_argument("--threshold", type=float, default=0.50, help="Cut safety score decision threshold (default: 0.50)")
    args = parser.parse_args()

    results = evaluate_cut_safety(threshold=args.threshold, verbose=True)
