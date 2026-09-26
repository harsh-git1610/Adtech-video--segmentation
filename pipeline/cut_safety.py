"""
Cut-Safety Scoring Module (Phase 2)
Evaluates whether a candidate scene boundary is safe for ad insertion.
Combines sentence boundary alignment, audio silence duration, shot alignment, and action continuity.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cut_safety")

REPO_ROOT = Path(__file__).resolve().parent.parent
BREAK_SCHEMA_PATH = REPO_ROOT / "schemas" / "break_candidate.schema.json"

# Bengali sentence-ending punctuation (দাঁড়ি U+0964, double danda U+0965) plus standard Latin (. ! ?)
SENTENCE_FINAL_PUNCTUATION = (".", "!", "?", "।", "॥", "\u0964", "\u0965")


def check_sentence_boundary_aligned(
    boundary_ts: float,
    word_timestamps: List[Dict[str, Any]],
    min_inter_word_gap_sec: float = 0.400
) -> Tuple[bool, Dict[str, Any]]:
    """
    Determines if boundary_ts is safely aligned with sentence / speech boundaries.

    Rules:
    - If boundary_ts falls strictly inside a word's [start, end] interval -> False (unsafe mid-word cut).
    - If boundary_ts falls between two words:
      - If preceding word has NO sentence-final punctuation (. ! ? or Bengali দাঁড়ি \u0964)
        AND the gap between words is < min_inter_word_gap_sec (400ms) -> False (unsafe mid-sentence cut).
      - Otherwise -> True (safe sentence or pause boundary).
    - If no words in interval or boundary is in open silence -> True.
    """
    debug_info: Dict[str, Any] = {
        "reason": "no_speech",
        "prev_word": None,
        "next_word": None,
        "gap_sec": None,
        "has_terminal_punct": False
    }

    if not word_timestamps:
        return True, debug_info

    # 1. Check if boundary strictly falls inside any word interval
    # Use small epsilon (1ms) to handle exact boundary touch vs strictly inside
    eps = 0.001
    for w in word_timestamps:
        w_start = float(w["start"])
        w_end = float(w["end"])
        if (w_start + eps) < boundary_ts < (w_end - eps):
            debug_info["reason"] = f"mid_word_{w.get('word', '')}"
            return False, debug_info

    # 2. Find closest word ending before boundary and starting after boundary
    pre_words = [w for w in word_timestamps if float(w["end"]) <= boundary_ts + eps]
    post_words = [w for w in word_timestamps if float(w["start"]) >= boundary_ts - eps]

    if not pre_words or not post_words:
        debug_info["reason"] = "speech_boundary_or_edge"
        return True, debug_info

    w_prev = max(pre_words, key=lambda x: float(x["end"]))
    w_next = min(post_words, key=lambda x: float(x["start"]))

    gap = float(w_next["start"]) - float(w_prev["end"])
    debug_info["prev_word"] = w_prev.get("word", "")
    debug_info["next_word"] = w_next.get("word", "")
    debug_info["gap_sec"] = round(gap, 3)

    # Check sentence-final punctuation on the preceding word
    prev_text = str(w_prev.get("word", "")).strip()
    has_final_punct = any(prev_text.endswith(p) for p in SENTENCE_FINAL_PUNCTUATION)
    debug_info["has_terminal_punct"] = has_final_punct

    # Unsafe if no punctuation AND gap < 400ms
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
    Returns the longest contiguous sub-window below silence_threshold_db.
    Supports PyAV for native video/audio reading, pydub, soundfile, or ASR gap fallback.
    """
    if audio_path and os.path.exists(audio_path):
        # Attempt 1: Try PyAV (handles MP4, AAC, WAV directly without external tools)
        try:
            import av
            container = av.open(audio_path)
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is not None:
                sr = audio_stream.rate or 48000
                start_pts = int(max(0.0, boundary_ts - window_sec) / audio_stream.time_base)
                end_pts = int((boundary_ts + window_sec) / audio_stream.time_base)

                container.seek(start_pts, stream=audio_stream)
                samples_list = []
                for frame in container.decode(audio_stream):
                    if frame.pts is not None and frame.pts > end_pts + (sr * 0.1):
                        break
                    # Convert to mono float32 ndarray
                    plane = frame.to_ndarray()
                    if plane.ndim > 1:
                        mono = np.mean(plane, axis=0)
                    else:
                        mono = plane
                    samples_list.append(mono)

                container.close()

                if samples_list:
                    audio_chunk = np.concatenate(samples_list).astype(np.float32)
                    # Normalize float values to [-1.0, 1.0] if integer dtype
                    max_val = np.max(np.abs(audio_chunk))
                    if max_val > 1.0:
                        audio_chunk = audio_chunk / max_val

                    hop = int(sr * 0.020)  # 20ms hops
                    if hop > 0 and len(audio_chunk) >= hop:
                        num_hops = len(audio_chunk) // hop
                        rms_list = []
                        for i in range(num_hops):
                            c = audio_chunk[i * hop : (i + 1) * hop]
                            rms = np.sqrt(np.mean(c ** 2) + 1e-9)
                            db = 20.0 * np.log10(rms)
                            rms_list.append(db)

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
            logger.debug("PyAV audio decoding skipped (%s), trying soundfile/pydub", e)

        # Attempt 2: Try soundfile for WAV/FLAC audio
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            sr = info.samplerate
            start_frame = max(0, int((boundary_ts - window_sec) * sr))
            stop_frame = min(info.frames, int((boundary_ts + window_sec) * sr))
            if stop_frame > start_frame:
                data, _ = sf.read(audio_path, start=start_frame, stop=stop_frame, always_2d=True)
                mono = np.mean(data, axis=1)
                hop = int(sr * 0.020)
                if hop > 0 and len(mono) >= hop:
                    num_hops = len(mono) // hop
                    rms_list = [20.0 * np.log10(np.sqrt(np.mean(mono[i*hop:(i+1)*hop]**2) + 1e-9)) for i in range(num_hops)]
                    max_silent = 0
                    cur = 0
                    for db in rms_list:
                        if db < silence_threshold_db:
                            cur += 1
                            if cur > max_silent:
                                max_silent = cur
                        else:
                            cur = 0
                    return round(max_silent * 20.0, 1)
        except Exception as e:
            logger.debug("soundfile audio reading skipped (%s)", e)

    # Fallback: estimate silence gap from ASR word timestamps
    if word_timestamps:
        pre_words = [w for w in word_timestamps if float(w["end"]) <= boundary_ts]
        post_words = [w for w in word_timestamps if float(w["start"]) >= boundary_ts]
        if pre_words and post_words:
            w_prev = max(pre_words, key=lambda x: float(x["end"]))
            w_next = min(post_words, key=lambda x: float(x["start"]))
            gap_ms = max(0.0, (float(w_next["start"]) - float(w_prev["end"])) * 1000.0)
            return round(min(2000.0, gap_ms), 1)
        # Open silence before first word or after last word
        return round(window_sec * 2000.0, 1)

    return 1000.0


def check_shot_boundary_aligned(
    boundary_ts: float,
    shot_boundaries: Optional[List[float]],
    tolerance_sec: float = 0.100
) -> bool:
    """
    True if boundary_ts aligns with an actual PySceneDetect shot transition within tolerance.
    Cuts at true shot boundaries are inherently safer than mid-shot cuts.
    """
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
    A sudden spike in inter-frame difference indicates continuous fast action.
    """
    if not video_path or not os.path.exists(video_path):
        return 0.0

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0

        target_msec = boundary_ts * 1000.0
        offsets = [-200, -100, 100, 200]
        diffs = []
        prev_gray = None

        for offset_ms in offsets:
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
            # Normal video motion baseline is ~0.05 - 0.15; intense action > 0.35
            penalty = float(np.clip((avg_diff - 0.10) / 0.30, 0.0, 1.0))
            return round(penalty, 3)
    except Exception as e:
        logger.debug("Action continuity check error (%s), returning 0.0", e)

    return 0.0


def score_candidate(
    boundary_ts: float,
    word_timestamps: List[Dict[str, Any]],
    shot_boundaries: Optional[List[float]] = None,
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None,
    scene_before: str = "sc_001",
    scene_after: str = "sc_002",
    candidate_id: str = "brk_0001"
) -> Dict[str, Any]:
    """
    Score a potential scene boundary for cut safety.

    Args:
        boundary_ts: Cut timestamp in seconds.
        word_timestamps: List of ASR word dicts with 'word', 'start', 'end'.
        shot_boundaries: List of detected shot boundary timestamps in seconds.
        audio_path: Path to audio file or video containing audio.
        video_path: Path to video file for action continuity inspection (defaults to audio_path if video).
        scene_before: scene_id of previous scene.
        scene_after: scene_id of subsequent scene.
        candidate_id: Unique candidate ID string, e.g. 'brk_0001'.

    Weighted combination formula:
      score = 0.40 * sentence_boundary_aligned
            + 0.25 * normalized(silence_gap_ms)
            + 0.20 * shot_boundary_aligned
            + 0.15 * (1 - action_continuity_penalty)

    Hard Floor:
      If sentence_boundary_aligned is False (mid-sentence or mid-word cut),
      the cut is disqualifying: score is hard-capped at 0.30 max.
    """
    v_path = video_path or audio_path

    sentence_aligned, _ = check_sentence_boundary_aligned(boundary_ts, word_timestamps)
    silence_gap_ms = compute_silence_gap_ms(boundary_ts, audio_path=audio_path, word_timestamps=word_timestamps)
    shot_aligned = check_shot_boundary_aligned(boundary_ts, shot_boundaries)
    action_penalty = compute_action_continuity_penalty(boundary_ts, video_path=v_path)

    # Normalize silence gap: >= 1000ms is full score (1.0)
    norm_silence = float(np.clip(silence_gap_ms / 1000.0, 0.0, 1.0))

    w_sentence = 0.40 * (1.0 if sentence_aligned else 0.0)
    w_silence = 0.25 * norm_silence
    w_shot = 0.20 * (1.0 if shot_aligned else 0.0)
    w_action = 0.15 * (1.0 - action_penalty)

    raw_score = w_sentence + w_silence + w_shot + w_action

    # Hard floor: mid-sentence / mid-word cuts capped at 0.30 max
    if not sentence_aligned:
        final_score = min(raw_score, 0.30)
    else:
        final_score = raw_score

    final_score = round(float(np.clip(final_score, 0.0, 1.0)), 3)

    # In Phase 2, pacing fields (pacing_eligible, final_decision) stay unset per specification.
    candidate_obj: Dict[str, Any] = {
        "candidate_id": candidate_id,
        "timestamp": round(float(boundary_ts), 3),
        "scene_before": scene_before,
        "scene_after": scene_after,
        "cut_safety_score": final_score,
        "cut_safety_reasons": {
            "sentence_boundary_aligned": sentence_aligned,
            "silence_gap_ms": round(float(silence_gap_ms), 1),
            "shot_boundary_aligned": shot_aligned,
            "action_continuity_penalty": round(float(action_penalty), 3)
        },
        "pacing_eligible": None,
        "pacing_reasons": {},
        "final_decision": None
    }

    return candidate_obj


def score_scene_boundaries(
    scenes: List[Dict[str, Any]],
    word_timestamps: List[Dict[str, Any]],
    shot_boundaries: Optional[List[float]] = None,
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generates break_candidate objects for all internal scene boundaries in scenes.json.
    Does NOT delete low-scoring candidates; every scene boundary becomes a candidate object.
    Validates candidates against schemas/break_candidate.schema.json.
    """
    import jsonschema

    schema = None
    if BREAK_SCHEMA_PATH.exists():
        with open(BREAK_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)

    candidates = []
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
