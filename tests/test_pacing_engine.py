"""
Unit and Property-Based Tests for Ad-Pacing Engine (Phase 3).
Verifies:
1. Fixed regression test on a hand-constructed 2-hour timeline with mathematically known optimum.
2. Safety floor enforcement (scores < cut_safety_floor marked REJECTED_UNSAFE).
3. Early cutoff enforcement (breaks before first_break_not_before_sec marked REJECTED_PACING).
4. Invariants enforcement: min_gap, rolling hourly cap, and total ad-load budget.
5. Randomized property testing across varied synthesized timelines.
6. Schema compliance with schemas/break_candidate.schema.json.
"""

import json
import random
from pathlib import Path
import pytest
import jsonschema

from pipeline.pacing_engine import (
    apply_pacing_rules,
    solve_weighted_interval_schedule,
    load_pacing_rules
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BREAK_SCHEMA_PATH = REPO_ROOT / "schemas" / "break_candidate.schema.json"


@pytest.fixture(scope="module")
def candidate_schema():
    with open(BREAK_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 1. Hand-Constructed 2-Hour Timeline Regression Test ───────────────────────

def test_hand_constructed_2hour_timeline_matches_known_optimum():
    """
    Fixed regression test on a 2-hour (7200s) timeline with 14 candidates.
    Rules:
      min_gap = 480s (8 min)
      first_break = 300s (5 min)
      max_breaks_per_hour = 6
      cut_safety_floor = 0.55
      target_ad_load_pct = 12.0%
      ad_duration = 15s

    Setup:
    - brk_1 @ 150s, score 0.90 -> Ineligible (150 < 300)
    - brk_2 @ 400s, score 0.40 -> Unsafe (0.40 < 0.55)
    - brk_3 @ 600s, score 0.80 -> Compatible with brk_5
    - brk_4 @ 800s, score 0.95 -> High score, but conflicts with both brk_3 (diff 200s) and brk_5 (diff 300s)
    - brk_5 @ 1100s, score 0.85 -> Compatible with brk_3 (1100 - 600 = 500 >= 480)
      * Note: Greedy-by-score would pick brk_4 (0.95), blocking brk_3 and brk_5 (total 0.95).
      * DP optimal choice between {brk_3, brk_4, brk_5} is {brk_3, brk_5} with score 0.80 + 0.85 = 1.65!
    - brk_6 @ 1650s, score 0.88 -> Compatible (1650 - 1100 = 550 >= 480)
    - brk_7 @ 2200s, score 0.92 -> Compatible (2200 - 1650 = 550 >= 480)
    - brk_8 @ 2700s, score 0.75 -> Compatible (2700 - 2200 = 500 >= 480)
    - brk_9 @ 3250s, score 0.82 -> Compatible (3250 - 2700 = 550 >= 480)
      * In first hour: brk_3, brk_5, brk_6, brk_7, brk_8, brk_9 = 6 breaks (exactly max_breaks_per_hour!)
    - brk_10 @ 3500s, score 0.90 -> Conflicts with brk_9 (diff 250 < 480) AND would exceed 6 breaks in first hour
    - brk_11 @ 4250s, score 0.85 -> In second hour, compatible (4250 - 3250 = 1000 >= 480; 4250 - 3600 = 650 > 600)
    - brk_12 @ 4750s, score 0.89 -> Compatible (4750 - 4250 = 500 >= 480)
    - brk_13 @ 5300s, score 0.91 -> Compatible (5300 - 4750 = 550 >= 480)
    - brk_14 @ 5850s, score 0.86 -> Compatible (5850 - 5300 = 550 >= 480)

    Expected Optimal Approved Breaks:
    brk_3, brk_5, brk_6, brk_7, brk_8, brk_9, brk_11, brk_12, brk_13, brk_14 (10 breaks total)
    Total expected score = 0.80 + 0.85 + 0.88 + 0.92 + 0.75 + 0.82 + 0.85 + 0.89 + 0.91 + 0.86 = 8.53
    """
    candidates = [
        {"candidate_id": "brk_0001", "timestamp": 150.0, "cut_safety_score": 0.90, "scene_before": "s1", "scene_after": "s2", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1200, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0002", "timestamp": 400.0, "cut_safety_score": 0.40, "scene_before": "s2", "scene_after": "s3", "cut_safety_reasons": {"sentence_boundary_aligned": False, "silence_gap_ms": 200, "shot_boundary_aligned": False, "action_continuity_penalty": 0.5}},
        {"candidate_id": "brk_0003", "timestamp": 600.0, "cut_safety_score": 0.80, "scene_before": "s3", "scene_after": "s4", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1000, "shot_boundary_aligned": True, "action_continuity_penalty": 0.1}},
        {"candidate_id": "brk_0004", "timestamp": 800.0, "cut_safety_score": 0.95, "scene_before": "s4", "scene_after": "s5", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1500, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0005", "timestamp": 1100.0, "cut_safety_score": 0.85, "scene_before": "s5", "scene_after": "s6", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1200, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0006", "timestamp": 1650.0, "cut_safety_score": 0.88, "scene_before": "s6", "scene_after": "s7", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1100, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0007", "timestamp": 2200.0, "cut_safety_score": 0.92, "scene_before": "s7", "scene_after": "s8", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1300, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0008", "timestamp": 2700.0, "cut_safety_score": 0.75, "scene_before": "s8", "scene_after": "s9", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 900, "shot_boundary_aligned": True, "action_continuity_penalty": 0.1}},
        {"candidate_id": "brk_0009", "timestamp": 3250.0, "cut_safety_score": 0.82, "scene_before": "s9", "scene_after": "s10", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1000, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0010", "timestamp": 3500.0, "cut_safety_score": 0.90, "scene_before": "s10", "scene_after": "s11", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1400, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0011", "timestamp": 4250.0, "cut_safety_score": 0.85, "scene_before": "s11", "scene_after": "s12", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1100, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0012", "timestamp": 4750.0, "cut_safety_score": 0.89, "scene_before": "s12", "scene_after": "s13", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1200, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0013", "timestamp": 5300.0, "cut_safety_score": 0.91, "scene_before": "s13", "scene_after": "s14", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1300, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0014", "timestamp": 5850.0, "cut_safety_score": 0.86, "scene_before": "s14", "scene_after": "s15", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1000, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}}
    ]

    rules = {
        "min_gap_seconds": 480,
        "first_break_not_before_sec": 300,
        "max_breaks_per_hour": 6,
        "cut_safety_floor": 0.55,
        "target_ad_load_pct": 12.0,
        "default_ad_duration_sec": 15
    }

    results = apply_pacing_rules(
        break_candidates=candidates,
        rules=rules,
        total_video_duration_sec=7200.0
    )

    approved = [c for c in results if c["final_decision"] == "BREAK_APPROVED"]
    approved_ids = [c["candidate_id"] for c in approved]

    # Verify DP chose {brk_3, brk_5} instead of greedy single {brk_4}
    assert "brk_0003" in approved_ids
    assert "brk_0005" in approved_ids
    assert "brk_0004" not in approved_ids, "DP must prefer higher aggregate sum of (brk_3 + brk_5) over brk_4"

    # Verify early break (< 300s) rejected
    assert results[0]["candidate_id"] == "brk_0001"
    assert results[0]["final_decision"] == "REJECTED_PACING"

    # Verify below-floor break (< 0.55) rejected unsafe
    assert results[1]["candidate_id"] == "brk_0002"
    assert results[1]["final_decision"] == "REJECTED_UNSAFE"
    assert results[1]["pacing_eligible"] is False

    # Verify all expected approved IDs
    expected_approved = [
        "brk_0003", "brk_0005", "brk_0006", "brk_0007",
        "brk_0008", "brk_0009", "brk_0011", "brk_0012",
        "brk_0013", "brk_0014"
    ]
    assert approved_ids == expected_approved, f"Expected {expected_approved}, got {approved_ids}"

    total_score = sum(c["cut_safety_score"] for c in approved)
    assert abs(total_score - 8.53) < 1e-4


# ── 2. Structural & Invariant Assertions on Results ──────────────────────────

def verify_all_invariants(candidates_out, rules, runtime_sec):
    """Universal invariant checker for any pacing output."""
    min_gap = float(rules.get("min_gap_seconds", 480))
    max_per_hour = int(rules.get("max_breaks_per_hour", 6))
    first_break = float(rules.get("first_break_not_before_sec", 300))
    floor = float(rules.get("cut_safety_floor", 0.55))
    target_pct = float(rules.get("target_ad_load_pct", 12.0))
    ad_dur = float(rules.get("default_ad_duration_sec", 15.0))

    approved = [c for c in candidates_out if c["final_decision"] == "BREAK_APPROVED"]
    app_ts = [float(c["timestamp"]) for c in approved]

    # Invariant 1: Sorted strictly ascending
    assert app_ts == sorted(app_ts)

    # Invariant 2: No break before first_break
    for t in app_ts:
        assert t >= first_break - 1e-4, f"Approved break at {t}s < first_break {first_break}s"

    # Invariant 3: min_gap between consecutive breaks
    for i in range(len(app_ts) - 1):
        gap = app_ts[i + 1] - app_ts[i]
        assert gap >= min_gap - 1e-4, f"Gap {gap:.2f}s < min_gap {min_gap}s between {app_ts[i]} and {app_ts[i+1]}"

    # Invariant 4: No more than max_breaks_per_hour in any rolling 3600s window
    for i in range(len(app_ts)):
        t_start = app_ts[i]
        count_in_window = sum(1 for t in app_ts if t_start <= t <= t_start + 3600.0 + 1e-4)
        assert count_in_window <= max_per_hour, (
            f"Violated hourly limit: {count_in_window} breaks in [{t_start}, {t_start + 3600}] > max {max_per_hour}"
        )

    # Invariant 5: Ad-load budget ceiling
    max_budget_sec = runtime_sec * (target_pct / 100.0)
    total_ad_sec = len(approved) * ad_dur
    assert total_ad_sec <= max_budget_sec + 1e-4, (
        f"Exceeded ad-load budget: {total_ad_sec}s > {max_budget_sec}s"
    )

    # Invariant 6: No below-floor candidate approved
    for c in approved:
        assert c["cut_safety_score"] >= floor, (
            f"Approved candidate {c['candidate_id']} has score {c['cut_safety_score']} < floor {floor}"
        )

    # Invariant 7: All candidates have valid decision & boolean eligibility
    for c in candidates_out:
        assert c["final_decision"] in ("BREAK_APPROVED", "REJECTED_UNSAFE", "REJECTED_PACING")
        assert isinstance(c["pacing_eligible"], bool)


# ── 3. Randomized Property-Based Tests ───────────────────────────────────────

def test_randomized_timelines_property_invariants():
    """
    Property-based test: synthesizes 35 varied timelines with randomized
    timestamps, safety scores, counts, durations, and rules.
    Verifies that the invariants are NEVER violated in any scenario.
    """
    random.seed(42)

    for run_idx in range(35):
        runtime_sec = random.uniform(600.0, 7200.0)
        num_candidates = random.randint(5, 50)

        raw_ts = sorted(random.sample(range(10, int(runtime_sec) - 10), min(num_candidates, int(runtime_sec) - 20)))
        candidates = []
        for i, ts in enumerate(raw_ts):
            score = round(random.uniform(0.20, 0.99), 3)
            candidates.append({
                "candidate_id": f"brk_{i+1:04d}",
                "timestamp": float(ts),
                "cut_safety_score": score,
                "scene_before": f"sc_{i:03d}",
                "scene_after": f"sc_{i+1:03d}",
                "cut_safety_reasons": {
                    "sentence_boundary_aligned": score >= 0.55,
                    "silence_gap_ms": 1000.0 if score >= 0.55 else 200.0,
                    "shot_boundary_aligned": True,
                    "action_continuity_penalty": 0.0
                }
            })

        rules = {
            "min_gap_seconds": random.choice([240, 360, 480, 600]),
            "max_breaks_per_hour": random.randint(3, 8),
            "first_break_not_before_sec": random.choice([120, 300, 400]),
            "cut_safety_floor": random.choice([0.45, 0.50, 0.55, 0.60]),
            "target_ad_load_pct": random.choice([8.0, 10.0, 12.0, 15.0]),
            "default_ad_duration_sec": random.choice([15, 20, 30])
        }

        output = apply_pacing_rules(
            break_candidates=candidates,
            rules=rules,
            total_video_duration_sec=runtime_sec
        )

        verify_all_invariants(output, rules, runtime_sec)


# ── 4. Schema Conformance ─────────────────────────────────────────────────────

def test_pacing_output_conforms_to_schema(candidate_schema):
    """Every candidate object output by apply_pacing_rules must strictly validate against JSON Schema."""
    candidates = [
        {"candidate_id": "brk_0001", "timestamp": 100.0, "cut_safety_score": 0.30, "scene_before": "s1", "scene_after": "s2", "cut_safety_reasons": {"sentence_boundary_aligned": False, "silence_gap_ms": 100, "shot_boundary_aligned": False, "action_continuity_penalty": 0.5}},
        {"candidate_id": "brk_0002", "timestamp": 400.0, "cut_safety_score": 0.80, "scene_before": "s2", "scene_after": "s3", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1200, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}},
        {"candidate_id": "brk_0003", "timestamp": 500.0, "cut_safety_score": 0.85, "scene_before": "s3", "scene_after": "s4", "cut_safety_reasons": {"sentence_boundary_aligned": True, "silence_gap_ms": 1400, "shot_boundary_aligned": True, "action_continuity_penalty": 0.0}}
    ]

    output = apply_pacing_rules(
        break_candidates=candidates,
        total_video_duration_sec=3600.0
    )

    for c in output:
        jsonschema.validate(instance=c, schema=candidate_schema)
        assert c["final_decision"] in ("BREAK_APPROVED", "REJECTED_UNSAFE", "REJECTED_PACING")
        assert isinstance(c["pacing_eligible"], bool)
        assert isinstance(c["pacing_reasons"], dict)
