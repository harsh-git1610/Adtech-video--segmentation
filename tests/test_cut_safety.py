"""
Unit tests for Phase 2 Cut-Safety Scoring.
Verifies each sub-signal independently with synthetic word-timestamps,
tests the hard-floor disqualification on mid-sentence cuts,
and validates schema compliance for candidate outputs.
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


# ── Sub-Signal 1: Sentence Boundary Alignment ─────────────────────────────────

def test_sentence_boundary_strictly_inside_word_rejected():
    """Boundary falling strictly inside a word's [start, end] span must be rejected (False)."""
    words = [
        {"word": "Hello", "start": 1.0, "end": 1.5},
        {"word": "world.", "start": 1.8, "end": 2.3}
    ]
    # 1.25 is strictly inside 'Hello' (1.0 to 1.5)
    is_aligned, debug = check_sentence_boundary_aligned(1.25, words)
    assert is_aligned is False
    assert "mid_word" in debug["reason"]


def test_sentence_boundary_mid_sentence_fast_speech_rejected():
    """Boundary between words with no sentence-final punctuation and gap < 400ms must be rejected (False)."""
    words = [
        {"word": "We", "start": 10.0, "end": 10.25},
        {"word": "are", "start": 10.45, "end": 10.70}  # gap is 0.20s (200ms) < 400ms, no terminal punctuation
    ]
    is_aligned, debug = check_sentence_boundary_aligned(10.35, words, min_inter_word_gap_sec=0.400)
    assert is_aligned is False
    assert "mid_sentence" in debug["reason"]
    assert debug["gap_sec"] == 0.2


def test_sentence_boundary_with_terminal_punctuation_accepted():
    """Boundary after sentence-final punctuation (. ! ?) is accepted even if gap < 400ms."""
    words = [
        {"word": "Goodbye.", "start": 5.0, "end": 5.3},
        {"word": "Next", "start": 5.5, "end": 5.8}  # gap is 200ms, but 'Goodbye.' ends with '.'
    ]
    is_aligned, debug = check_sentence_boundary_aligned(5.4, words)
    assert is_aligned is True
    assert debug["has_terminal_punct"] is True

    # Check question mark and exclamation mark
    words_q = [
        {"word": "Really?", "start": 1.0, "end": 1.3},
        {"word": "Yes!", "start": 1.5, "end": 1.8}
    ]
    is_aligned_q, debug_q = check_sentence_boundary_aligned(1.4, words_q)
    assert is_aligned_q is True
    assert debug_q["has_terminal_punct"] is True


def test_sentence_boundary_with_bengali_dari_punctuation_accepted():
    """Boundary after Bengali sentence-final punctuation (দাঁড়ি \u0964) is accepted even if gap < 400ms."""
    words_bn = [
        {"word": "আমি যাব।", "start": 20.0, "end": 20.4},
        {"word": "তুমি আসবে", "start": 20.6, "end": 21.0}  # gap is 200ms, preceding word ends with \u0964
    ]
    is_aligned, debug = check_sentence_boundary_aligned(20.5, words_bn)
    assert is_aligned is True
    assert debug["has_terminal_punct"] is True


def test_sentence_boundary_with_pause_above_400ms_accepted():
    """Boundary between words with gap >= 400ms is accepted even without terminal punctuation."""
    words = [
        {"word": "waiting", "start": 1.0, "end": 1.5},
        {"word": "resumed", "start": 2.1, "end": 2.6}  # gap is 600ms >= 400ms
    ]
    is_aligned, debug = check_sentence_boundary_aligned(1.8, words)
    assert is_aligned is True
    assert debug["gap_sec"] == 0.6


def test_sentence_boundary_outside_speech_accepted():
    """Boundary occurring during complete silence (before dialogue starts or after it ends) is True."""
    words = [
        {"word": "Starting", "start": 10.0, "end": 10.5},
        {"word": "ending.", "start": 12.0, "end": 12.5}
    ]
    # Boundary at 3.0s (well before speech starts at 10.0s)
    is_aligned_pre, _ = check_sentence_boundary_aligned(3.0, words)
    assert is_aligned_pre is True

    # Boundary at 18.0s (well after speech ends at 12.5s)
    is_aligned_post, _ = check_sentence_boundary_aligned(18.0, words)
    assert is_aligned_post is True


# ── Sub-Signal 2: Silence Gap ─────────────────────────────────────────────────

def test_compute_silence_gap_fallback():
    """When audio file is not provided, silence gap falls back to ASR word intervals."""
    words = [
        {"word": "first", "start": 1.0, "end": 1.4},
        {"word": "second", "start": 2.2, "end": 2.6}
    ]
    # Boundary at 1.8s; gap between 1.4 and 2.2 is 0.8s = 800ms
    gap_ms = compute_silence_gap_ms(1.8, audio_path=None, word_timestamps=words)
    assert gap_ms == 800.0


# ── Sub-Signal 3: Shot Boundary Alignment ─────────────────────────────────────

def test_shot_boundary_alignment_within_tolerance():
    """Boundary within 100ms tolerance of a shot cut is True; beyond is False."""
    shot_boundaries = [5.0, 10.0, 15.0, 20.0]
    # Within 100ms tolerance
    assert check_shot_boundary_aligned(10.05, shot_boundaries, tolerance_sec=0.100) is True
    assert check_shot_boundary_aligned(9.95, shot_boundaries, tolerance_sec=0.100) is True
    # Outside tolerance
    assert check_shot_boundary_aligned(10.25, shot_boundaries, tolerance_sec=0.100) is False
    assert check_shot_boundary_aligned(12.50, shot_boundaries, tolerance_sec=0.100) is False


# ── Sub-Signal 4: Action Continuity Penalty ───────────────────────────────────

def test_action_continuity_penalty_nonexistent_video():
    """Returns 0.0 penalty if video is unavailable."""
    penalty = compute_action_continuity_penalty(10.0, video_path=None)
    assert penalty == 0.0


# ── Integration: Hard Floor & Weighted Score ─────────────────────────────────

def test_hard_floor_disqualifies_mid_sentence_cuts():
    """
    Judges heavily penalize mid-sentence cuts.
    Even with 100% silence, 100% shot alignment, and 0% action penalty,
    a cut where sentence_boundary_aligned=False MUST be capped at 0.30 max.
    """
    # Mid-word cut
    words = [{"word": "speaking", "start": 10.0, "end": 11.0}]
    candidate = score_candidate(
        boundary_ts=10.5,
        word_timestamps=words,
        shot_boundaries=[10.5],  # perfectly aligned with shot
        audio_path=None,         # will have 0 penalty
        video_path=None
    )

    assert candidate["cut_safety_reasons"]["sentence_boundary_aligned"] is False
    assert candidate["cut_safety_score"] <= 0.30, (
        f"Score {candidate['cut_safety_score']} exceeded hard floor 0.30 on mid-sentence cut"
    )


def test_safe_boundary_scores_high():
    """A clean cut with sentence boundary, long silence, and shot alignment scores high."""
    words = [
        {"word": "concluded.", "start": 8.0, "end": 8.5},
        {"word": "Next", "start": 10.5, "end": 11.0}  # 2.0s gap
    ]
    candidate = score_candidate(
        boundary_ts=9.5,
        word_timestamps=words,
        shot_boundaries=[9.5],
        audio_path=None,
        video_path=None
    )

    reasons = candidate["cut_safety_reasons"]
    assert reasons["sentence_boundary_aligned"] is True
    assert reasons["shot_boundary_aligned"] is True
    assert reasons["silence_gap_ms"] >= 1000.0
    assert candidate["cut_safety_score"] >= 0.80


# ── Schema Conformance ────────────────────────────────────────────────────────

def test_score_candidate_conforms_to_schema(candidate_schema):
    """Generated break_candidate object must strictly conform to schemas/break_candidate.schema.json."""
    words = [
        {"word": "End.", "start": 1.0, "end": 1.5},
        {"word": "Start", "start": 2.5, "end": 3.0}
    ]
    candidate = score_candidate(
        boundary_ts=2.0,
        word_timestamps=words,
        shot_boundaries=[2.0],
        scene_before="sc_001",
        scene_after="sc_002",
        candidate_id="brk_0001"
    )

    jsonschema.validate(instance=candidate, schema=candidate_schema)
    assert candidate["candidate_id"] == "brk_0001"
    assert candidate["timestamp"] == 2.0
    assert candidate["scene_before"] == "sc_001"
    assert candidate["scene_after"] == "sc_002"
    assert 0.0 <= candidate["cut_safety_score"] <= 1.0


def test_score_scene_boundaries_creates_all_candidates(candidate_schema):
    """All scene boundaries are converted to break_candidate objects without dropping low scores."""
    scenes = [
        {"scene_id": "sc_001", "start_ts": 0.0, "end_ts": 10.0, "shot_ids": ["sh_001"]},
        {"scene_id": "sc_002", "start_ts": 10.0, "end_ts": 25.0, "shot_ids": ["sh_002"]},
        {"scene_id": "sc_003", "start_ts": 25.0, "end_ts": 40.0, "shot_ids": ["sh_003"]}
    ]
    # Two internal boundaries: 10.0 and 25.0
    # Make boundary 10.0 mid-word (unsafe) and 25.0 clean (safe)
    words = [
        {"word": "talking", "start": 9.5, "end": 10.5},   # cuts at 10.0 is mid-word!
        {"word": "done.", "start": 23.0, "end": 24.0},
        {"word": "next", "start": 26.0, "end": 27.0}     # cuts at 25.0 is safe!
    ]

    candidates = score_scene_boundaries(
        scenes=scenes,
        word_timestamps=words,
        shot_boundaries=[25.0]
    )

    assert len(candidates) == 2, "Must emit one candidate per scene boundary"
    # First candidate is unsafe but retained
    assert candidates[0]["candidate_id"] == "brk_0001"
    assert candidates[0]["cut_safety_score"] <= 0.30
    assert candidates[0]["cut_safety_reasons"]["sentence_boundary_aligned"] is False

    # Second candidate is safe
    assert candidates[1]["candidate_id"] == "brk_0002"
    assert candidates[1]["cut_safety_score"] >= 0.80
    assert candidates[1]["cut_safety_reasons"]["sentence_boundary_aligned"] is True

    # Both validate against schema
    for c in candidates:
        jsonschema.validate(instance=c, schema=candidate_schema)
