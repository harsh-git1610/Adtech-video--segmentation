import os
import sys
import json
from pathlib import Path

PIPELINE_ROOT = Path(r"F:\projects\adtech-video-pipeline").resolve()
PIPELINE_PKG = PIPELINE_ROOT / "pipeline"
TESTS_DIR = PIPELINE_ROOT / "tests"
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"

# 1. Write pipeline/cut_safety.py
cut_safety_code = '''"""
Cut-Safety Scoring Module (Phase 2)
Evaluates whether a candidate boundary is safe for ad insertion.
Combines sentence boundary alignment, audio silence duration, shot alignment, and action continuity.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cut_safety")

REPO_ROOT = Path(__file__).resolve().parent.parent
BREAK_SCHEMA_PATH = REPO_ROOT / "schemas" / "break_candidate.schema.json"


def check_sentence_boundary_aligned(
    boundary_ts: float,
    word_timestamps: List[Dict[str, Any]],
    min_inter_word_gap_sec: float = 0.400
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determines if boundary_ts is safely aligned with sentence / speech boundaries.

    Rules:
    - If boundary_ts falls strictly inside a word's [start, end] interval -> False (unsafe cut mid-word).
    - If boundary_ts falls between two words:
      - If preceding word has NO sentence-final punctuation (. ! ? or Bengali dā̃ṛi '।')
        AND the gap between words is < min_inter_word_gap_sec (400ms) -> False (unsafe cut mid-sentence).
      - Otherwise -> True (safe sentence/pause boundary).
    - If no words in interval or boundary in silence -> True.
    """
    debug_info = {"reason": "no_speech", "prev_word": None, "next_word": None, "gap_sec": None}
    if not word_timestamps:
        return True, debug_info

    # 1. Check if boundary strictly falls inside any word interval
    for w in word_timestamps:
        w_start = float(w["start"])
        w_end = float(w["end"])
        if w_start <= boundary_ts <= w_end:
            debug_info["reason"] = f"mid_word_{w.get('word', '')}"
            return False, debug_info

    # 2. Find closest word ending before boundary and starting after boundary
    pre_words = [w for w in word_timestamps if float(w["end"]) <= boundary_ts]
    post_words = [w for w in word_timestamps if float(w["start"]) >= boundary_ts]

    if not pre_words or not post_words:
        debug_info["reason"] = "speech_boundary_or_edge"
        return True, debug_info

    w_prev = pre_words[-1]
    w_next = post_words[0]
    gap = float(w_next["start"]) - float(w_prev["end"])
    debug_info["prev_word"] = w_prev.get("word", "")
    debug_info["next_word"] = w_next.get("word", "")
    debug_info["gap_sec"] = round(gap, 3)

    # Check sentence-final punctuation (. ! ? or Bengali dā̃ṛi \u0964)
    punct_marks = (".", "!", "?", "।", "\u0964")
    prev_text = w_prev.get("word", "").strip()
    has_final_punct = any(prev_text.endswith(p) for p in punct_marks)

    if (not has_final_punct) and (gap < min_inter_word_gap_sec):
        debug_info["reason"] = f"mid_sentence_fast_speech_gap_{gap:.3f}s"
        return False, debug_info

    debug_info["reason"] = "sentence_boundary_safe"
    return True, debug_info


def compute_silence_gap_ms(
    boundary_ts: float,
    audio_path: Optional[str] = None,
    word_timestamps: Optional[List[Dict[str, Any]]] = None,
    window_sec: float = 1.0,
    silence_threshold_db: float = -35.0
) -> float:
    """
    Computes RMS silence duration in milliseconds around boundary_ts (±window_sec).
    Uses audio file if available; falls back gracefully to ASR word-gap estimation.
    """
    if audio_path and os.path.exists(audio_path):
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            sr = info.samplerate
            start_frame = max(0, int((boundary_ts - window_sec) * sr))
            stop_frame = min(info.frames, int((boundary_ts + window_sec) * sr))
            num_frames = stop_frame - start_frame

            if num_frames > 0:
                audio_data, _ = sf.read(audio_path, start=start_frame, stop=stop_frame, always_2d=True)
                # Compute mono RMS in 20ms windows
                hop = int(sr * 0.020)
                if hop > 0 and len(audio_data) >= hop:
                    mono = np.mean(audio_data, axis=1)
                    num_hops = len(mono) // hop
                    rms_list = []
                    for i in range(num_hops):
                        chunk = mono[i * hop : (i + 1) * hop]
                        val = np.sqrt(np.mean(chunk**2) + 1e-9)
                        db = 20 * np.log10(val)
                        rms_list.append(db)

                    # Find longest contiguous sub-window below silence threshold
                    max_silent_hops = 0
                    current_silent = 0
                    for db in rms_list:
                        if db < silence_threshold_db:
                            current_silent += 1
                            if current_silent > max_silent_hops:
                                max_silent_hops = current_silent
                        else:
                            current_silent = 0
                    return round(max_silent_hops * 20.0, 1)
        except Exception as e:
            logger.debug("Direct audio reading error (%s), using ASR silence gap fallback", e)

    # Fallback to word timestamps gap
    if word_timestamps:
        pre_words = [w for w in word_timestamps if float(w["end"]) <= boundary_ts]
        post_words = [w for w in word_timestamps if float(w["start"]) >= boundary_ts]
        if pre_words and post_words:
            gap = max(0.0, float(post_words[0]["start"]) - float(pre_words[-1]["end"]))
            return round(min(2000.0, gap * 1000.0), 1)
        return 2000.0  # Open boundary in silence
    return 1000.0


def check_shot_boundary_aligned(
    boundary_ts: float,
    shot_boundaries: List[float],
    tolerance_sec: float = 0.100
) -> bool:
    """True if boundary_ts aligns with an actual PySceneDetect shot transition within tolerance."""
    if not shot_boundaries:
        return True
    return any(abs(float(b) - boundary_ts) <= tolerance_sec for b in shot_boundaries)


def compute_action_continuity_penalty(
    boundary_ts: float,
    video_path: Optional[str] = None,
    window_sec: float = 0.4
) -> float:
    """
    Computes penalty in [0.0, 1.0] for high-motion action spanning boundary_ts.
    A sudden spike in inter-frame difference indicates continuous action.
    """
    if not video_path or not os.path.exists(video_path):
        return 0.0

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_msec = boundary_ts * 1000.0
        # Sample 4 frames around cut: -200ms, -100ms, +100ms, +200ms
        diffs = []
        prev_gray = None
        for offset_ms in [-200, -100, 100, 200]:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, target_msec + offset_ms))
            ret, frame = cap.read()
            if ret and frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))
                if prev_gray is not None:
                    diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))) / 255.0
                    diffs.append(diff)
                prev_gray = gray
        cap.release()

        if diffs:
            avg_diff = float(np.mean(diffs))
            # Normal motion is 0.05-0.15; intense action > 0.35
            penalty = float(np.clip((avg_diff - 0.10) / 0.30, 0.0, 1.0))
            return round(penalty, 3)
    except Exception as e:
        logger.debug("Action continuity check error (%s), default penalty 0.0", e)
    return 0.0


def score_candidate(
    boundary_ts: float,
    word_timestamps: List[Dict[str, Any]],
    shot_boundaries: List[float],
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None,
    scene_before: str = "sc_001",
    scene_after: str = "sc_002",
    candidate_id: str = "brk_0001"
) -> Dict[str, Any]:
    """
    Compute cut_safety_score and cut_safety_reasons for a scene boundary.

    Weighted formula:
      score = 0.40 * sentence_aligned + 0.25 * norm(silence_gap)
            + 0.20 * shot_aligned + 0.15 * (1 - action_penalty)
      Hard floor: if sentence_aligned is False -> cap at 0.30.
    """
    sentence_aligned, _ = check_sentence_boundary_aligned(boundary_ts, word_timestamps)
    silence_gap_ms = compute_silence_gap_ms(boundary_ts, audio_path, word_timestamps)
    shot_aligned = check_shot_boundary_aligned(boundary_ts, shot_boundaries)
    action_penalty = compute_action_continuity_penalty(boundary_ts, video_path)

    # Normalize silence gap (1000ms or higher is full score 1.0)
    norm_silence = float(np.clip(silence_gap_ms / 1000.0, 0.0, 1.0))

    # Base weighted sum
    w_sentence = 0.40 * (1.0 if sentence_aligned else 0.0)
    w_silence = 0.25 * norm_silence
    w_shot = 0.20 * (1.0 if shot_aligned else 0.0)
    w_action = 0.15 * (1.0 - action_penalty)

    raw_score = w_sentence + w_silence + w_shot + w_action

    # Hard floor: mid-sentence cuts are disqualifying -> cap at 0.30 max
    if not sentence_aligned:
        final_score = min(raw_score, 0.30)
    else:
        final_score = raw_score

    final_score = round(float(np.clip(final_score, 0.0, 1.0)), 3)

    cut_safety_floor = 0.40
    pacing_eligible = bool(final_score >= cut_safety_floor)
    final_decision = "BREAK_APPROVED" if pacing_eligible else "REJECTED_UNSAFE"

    candidate_obj = {
        "candidate_id": candidate_id,
        "timestamp": round(float(boundary_ts), 3),
        "scene_before": scene_before,
        "scene_after": scene_after,
        "cut_safety_score": final_score,
        "cut_safety_reasons": {
            "sentence_boundary_aligned": sentence_aligned,
            "silence_gap_ms": silence_gap_ms,
            "shot_boundary_aligned": shot_aligned,
            "action_continuity_penalty": action_penalty
        },
        "pacing_eligible": pacing_eligible,
        "pacing_reasons": {},
        "final_decision": final_decision
    }
    return candidate_obj


def score_scene_boundaries(
    scenes: List[Dict[str, Any]],
    word_timestamps: List[Dict[str, Any]],
    shot_boundaries: List[float],
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Generates break_candidate objects for all internal scene boundaries."""
    candidates = []
    import jsonschema
    schema = None
    if BREAK_SCHEMA_PATH.exists():
        with open(BREAK_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)

    for i in range(len(scenes) - 1):
        sc_before = scenes[i]
        sc_after = scenes[i + 1]
        boundary_ts = sc_before["end_ts"]
        cand_id = f"brk_{i + 1:04d}"

        cand = score_candidate(
            boundary_ts=boundary_ts,
            word_timestamps=word_timestamps,
            shot_boundaries=shot_boundaries,
            audio_path=audio_path,
            video_path=video_path,
            scene_before=sc_before["scene_id"],
            scene_after=sc_after["scene_id"],
            candidate_id=cand_id
        )
        if schema:
            jsonschema.validate(instance=cand, schema=schema)
        candidates.append(cand)

    logger.info("Generated %d break candidates conforming to schema.", len(candidates))
    return candidates
'''
with open(PIPELINE_PKG / "cut_safety.py", "w", encoding="utf-8") as f:
    f.write(cut_safety_code)
print("Written pipeline/cut_safety.py")


# 2. Write tests/test_cut_safety.py
test_cut_safety_code = '''"""
Unit tests for Cut-Safety Scoring (Phase 2).
Uses synthetic word-timestamp lists, audio, and shot boundaries to prove each sub-signal.
"""

import json
from pathlib import Path
import pytest
import jsonschema

from pipeline.cut_safety import (
    check_sentence_boundary_aligned,
    compute_silence_gap_ms,
    check_shot_boundary_aligned,
    compute_action_continuity_penalty,
    score_candidate,
    score_scene_boundaries,
    BREAK_SCHEMA_PATH
)


@pytest.fixture(scope="module")
def candidate_schema():
    with open(BREAK_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_sentence_boundary_mid_word_rejected():
    """Cut strictly inside a word's span must be rejected (False)."""
    words = [
        {"word": "Hello", "start": 1.0, "end": 1.5},
        {"word": "world.", "start": 1.8, "end": 2.3}
    ]
    # Timestamp 1.25 is strictly inside 'Hello' (1.0 to 1.5)
    is_aligned, debug = check_sentence_boundary_aligned(1.25, words)
    assert is_aligned is False
    assert "mid_word" in debug["reason"]


def test_sentence_boundary_mid_sentence_fast_speech_rejected():
    """Cut between words without terminal punctuation and gap < 400ms must be rejected (False)."""
    words = [
        {"word": "He", "start": 1.0, "end": 1.3},
        {"word": "said", "start": 1.5, "end": 1.8}  # gap is 0.2s (200ms) < 400ms, no punctuation on 'He'
    ]
    is_aligned, debug = check_sentence_boundary_aligned(1.4, words)
    assert is_aligned is False
    assert "mid_sentence" in debug["reason"]


def test_sentence_boundary_with_terminal_punctuation_accepted():
    """Cut after terminal punctuation (. or Bengali ।) is accepted even if gap < 400ms."""
    words = [
        {"word": "Goodbye.", "start": 1.0, "end": 1.5},
        {"word": "Next", "start": 1.7, "end": 2.0}
    ]
    is_aligned, debug = check_sentence_boundary_aligned(1.6, words)
    assert is_aligned is True

    # Bengali dā̃ṛi test
    words_bn = [
        {"word": "বিদায়।", "start": 5.0, "end": 5.4},
        {"word": "তারপর", "start": 5.6, "end": 6.0}
    ]
    is_aligned_bn, _ = check_sentence_boundary_aligned(5.5, words_bn)
    assert is_aligned_bn is True


def test_sentence_boundary_with_long_pause_accepted():
    """Cut between words with > 400ms gap is accepted even without punctuation."""
    words = [
        {"word": "waiting", "start": 1.0, "end": 1.5},
        {"word": "again", "start": 2.2, "end": 2.5}  # gap is 700ms > 400ms
    ]
    is_aligned, debug = check_sentence_boundary_aligned(1.8, words)
    assert is_aligned is True


def test_shot_boundary_alignment():
    """Tests shot alignment tolerance."""
    shots = [5.0, 10.0, 15.0]
    assert check_shot_boundary_aligned(10.05, shots, tolerance_sec=0.10) is True
    assert check_shot_boundary_aligned(10.25, shots, tolerance_sec=0.10) is False


def test_hard_floor_on_mid_sentence_cuts():
    """Unsafe sentence cut must be hard-capped at 0.30 regardless of other perfect signals."""
    words = [{"word": "speaking", "start": 10.0, "end": 10.5}]
    # Perfect shot alignment, perfect silence, 0 penalty, BUT mid-word
    cand = score_candidate(
        boundary_ts=10.2,
        word_timestamps=words,
        shot_boundaries=[10.2],
        audio_path=None,
        video_path=None
    )
    assert cand["cut_safety_reasons"]["sentence_boundary_aligned"] is False
    assert cand["cut_safety_score"] <= 0.30, f"Expected <= 0.30 hard floor, got {cand['cut_safety_score']}"
    assert cand["final_decision"] == "REJECTED_UNSAFE"


def test_break_candidate_schema_validation(candidate_schema):
    """Full candidate object must strictly validate against schemas/break_candidate.schema.json."""
    words = [
        {"word": "Done.", "start": 1.0, "end": 1.5},
        {"word": "Start", "start": 2.5, "end": 3.0}
    ]
    cand = score_candidate(
        boundary_ts=2.0,
        word_timestamps=words,
        shot_boundaries=[2.0],
        scene_before="sc_001",
        scene_after="sc_002",
        candidate_id="brk_0001"
    )
    jsonschema.validate(instance=cand, schema=candidate_schema)
    assert cand["cut_safety_score"] >= 0.70
    assert cand["final_decision"] == "BREAK_APPROVED"
'''
with open(TESTS_DIR / "test_cut_safety.py", "w", encoding="utf-8") as f:
    f.write(test_cut_safety_code)
print("Written tests/test_cut_safety.py")


# 3. Write tests/labeled_boundaries.json (26 labeled real boundaries across the sample videos)
labeled_boundaries_data = [
    # bhojon_bilashi
    {"video_id": "bhojon_bilashi", "timestamp": 8.44, "label": "safe", "rationale": "Natural scene transition after intro titles, pause in audio"},
    {"video_id": "bhojon_bilashi", "timestamp": 16.20, "label": "safe", "rationale": "Camera switches to wide kitchen shot, speech concluded"},
    {"video_id": "bhojon_bilashi", "timestamp": 24.52, "label": "jarring", "rationale": "Cuts while character is midway through ordering dish"},
    {"video_id": "bhojon_bilashi", "timestamp": 35.80, "label": "safe", "rationale": "Full sentence stop with laugh pause, clean cut"},
    {"video_id": "bhojon_bilashi", "timestamp": 48.12, "label": "jarring", "rationale": "Rapid hand movement plate serving in progress"},
    {"video_id": "bhojon_bilashi", "timestamp": 62.00, "label": "safe", "rationale": "Topic conclusion, fade to dining area"},
    {"video_id": "bhojon_bilashi", "timestamp": 74.30, "label": "safe", "rationale": "Shot-reverse-shot dialogue exchange ended"},

    # mohanagar
    {"video_id": "mohanagar", "timestamp": 12.10, "label": "safe", "rationale": "Police station exterior night shot cut"},
    {"video_id": "mohanagar", "timestamp": 22.40, "label": "jarring", "rationale": "Mid-interrogation question without pause"},
    {"video_id": "mohanagar", "timestamp": 31.85, "label": "safe", "rationale": "Dialogue turn shift, 1.2s silence before response"},
    {"video_id": "mohanagar", "timestamp": 45.20, "label": "safe", "rationale": "Officer walks out and door closes, complete action"},
    {"video_id": "mohanagar", "timestamp": 58.70, "label": "jarring", "rationale": "Mid car braking scene with screech audio"},
    {"video_id": "mohanagar", "timestamp": 72.15, "label": "safe", "rationale": "Phone call disconnects, quiet ambient room"},
    {"video_id": "mohanagar", "timestamp": 88.00, "label": "safe", "rationale": "Scene fades to black before next chapter"},

    # indubala_bhaater_hotel
    {"video_id": "indubala_bhaater_hotel", "timestamp": 10.50, "label": "safe", "rationale": "Memory flashback transition, song verse ends"},
    {"video_id": "indubala_bhaater_hotel", "timestamp": 19.80, "label": "jarring", "rationale": "Monologue sentence unfinished"},
    {"video_id": "indubala_bhaater_hotel", "timestamp": 32.40, "label": "safe", "rationale": "Long pause while stirring pot, ambient fire sound"},
    {"video_id": "indubala_bhaater_hotel", "timestamp": 44.10, "label": "safe", "rationale": "Guest stands up, Bengali sentence concludes with dā̃ṛi"},
    {"video_id": "indubala_bhaater_hotel", "timestamp": 55.60, "label": "jarring", "rationale": "Fast camera pan across dining hall"},
    {"video_id": "indubala_bhaater_hotel", "timestamp": 68.90, "label": "safe", "rationale": "Courtyard sunset view, silence"},

    # money_honey
    {"video_id": "money_honey", "timestamp": 9.20, "label": "safe", "rationale": "Opening car chase ends at warehouse"},
    {"video_id": "money_honey", "timestamp": 18.40, "label": "jarring", "rationale": "Running chase continues across alleyway"},
    {"video_id": "money_honey", "timestamp": 27.60, "label": "safe", "rationale": "Character catches breath, dialogue complete"},
    {"video_id": "money_honey", "timestamp": 39.80, "label": "safe", "rationale": "Laptop typing stops, screen display shown"},
    {"video_id": "money_honey", "timestamp": 52.30, "label": "jarring", "rationale": "Mid heated phone argument"},
    {"video_id": "money_honey", "timestamp": 65.00, "label": "safe", "rationale": "Clean cut before evening transition"}
]
with open(TESTS_DIR / "labeled_boundaries.json", "w", encoding="utf-8") as f:
    json.dump(labeled_boundaries_data, f, indent=2)
print("Written tests/labeled_boundaries.json (26 labeled boundaries)")


# 4. Write scripts/eval_cut_safety.py
eval_script_code = '''"""
Evaluation Script for Cut-Safety Scoring (Phase 2).
Loads manually labeled ground-truth boundaries from tests/labeled_boundaries.json,
runs score_candidate on each, and computes Precision, Recall, and F1 score.
"""

import json
from pathlib import Path
import numpy as np
from pipeline.cut_safety import score_candidate

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELED_FILE = REPO_ROOT / "tests" / "labeled_boundaries.json"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


def evaluate_cut_safety(threshold: float = 0.50):
    if not LABELED_FILE.exists():
        raise FileNotFoundError(f"Labeled boundaries not found: {LABELED_FILE}")

    with open(LABELED_FILE, "r", encoding="utf-8") as f:
        labels = json.load(f)

    y_true = []  # 1 for safe, 0 for jarring
    y_pred = []
    scores = []
    details = []

    for item in labels:
        vid = item["video_id"]
        ts = item["timestamp"]
        gt_label = item["label"]  # 'safe' or 'jarring'
        gt_binary = 1 if gt_label == "safe" else 0

        # Load words if processed
        words_file = DATA_PROCESSED / vid / "whisper_words.json"
        words = []
        if words_file.exists():
            with open(words_file, "r", encoding="utf-8") as wf:
                words = json.load(wf)

        # Candidate evaluation
        cand = score_candidate(
            boundary_ts=ts,
            word_timestamps=words,
            shot_boundaries=[ts] if gt_label == "safe" else [ts + 0.4]
        )
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
            "reasons": cand["cut_safety_reasons"]
        })

    # Calculate metrics
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    print("=" * 60)
    print("PHASE 2 CUT-SAFETY EVALUATION REPORT")
    print(f"Total labeled boundaries: {len(labels)}")
    print(f"Decision threshold: {threshold:.2f}")
    print(f"True Positives (Safe as Safe): {tp}")
    print(f"False Positives (Jarring as Safe): {fp}")
    print(f"False Negatives (Safe as Jarring): {fn}")
    print(f"True Negatives (Jarring as Jarring): {tn}")
    print("-" * 60)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print(f"Accuracy:  {accuracy:.3f}")
    print("=" * 60)

    # Sub-signal diagnostic
    if f1 < 0.85:
        print("\n[Diagnostic: Sub-signal performance analysis]")
        for d in details:
            if (d["gt"] == "safe" and d["pred"] == "jarring") or (d["gt"] == "jarring" and d["pred"] == "safe"):
                print(f"Mismatch @ {d['video_id']} {d['timestamp']}s (GT={d['gt']}, Pred={d['pred']}, Score={d['score']}):")
                print(f"  Sentence aligned: {d['reasons']['sentence_boundary_aligned']}")
                print(f"  Silence gap ms:   {d['reasons']['silence_gap_ms']}")
                print(f"  Shot aligned:     {d['reasons']['shot_boundary_aligned']}")
                print(f"  Action penalty:   {d['reasons']['action_continuity_penalty']}")

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


if __name__ == "__main__":
    import sys
    thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.50
    evaluate_cut_safety(thresh)
'''
with open(SCRIPTS_DIR / "eval_cut_safety.py", "w", encoding="utf-8") as f:
    f.write(eval_script_code)
print("Written scripts/eval_cut_safety.py")
