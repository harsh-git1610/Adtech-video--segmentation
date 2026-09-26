"""
Ad-Pacing Rules Engine (Phase 3)
Evaluates "whether a break is warranted here" using Dynamic Programming
(Weighted Interval Scheduling with rolling hourly limits and ad-load budget constraints).
"""

import os
import sys
import copy
import logging
import bisect
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("pacing_engine")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "pacing_rules.yaml"


def load_pacing_rules(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load pacing configuration from YAML."""
    c_path = Path(config_path) if config_path else CONFIG_PATH
    if not c_path.exists():
        logger.warning("Pacing config %s not found, using broadcast defaults.", c_path)
        return {
            "max_breaks_per_hour": 6,
            "min_gap_seconds": 480,
            "target_ad_load_pct": 12.0,
            "first_break_not_before_sec": 300,
            "cut_safety_floor": 0.55,
            "default_ad_duration_sec": 15
        }
    with open(c_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("max_breaks_per_hour", 6)
    cfg.setdefault("min_gap_seconds", 480)
    cfg.setdefault("target_ad_load_pct", 12.0)
    cfg.setdefault("first_break_not_before_sec", 300)
    cfg.setdefault("cut_safety_floor", 0.55)
    cfg.setdefault("default_ad_duration_sec", 15)
    return cfg


def solve_weighted_interval_schedule(
    candidates: List[Dict[str, Any]],
    min_gap_sec: float,
    max_per_hour: int,
    max_total_breaks: int,
    first_break_not_before_sec: float
) -> Tuple[float, List[int]]:
    """
    Optimal selection via Dynamic Programming (Weighted Interval Scheduling).
    Maximizes sum of cut_safety_score subject to:
      1. t_i >= first_break_not_before_sec
      2. t_next - t_curr >= min_gap_sec
      3. At most max_per_hour breaks in any rolling 3600-second window
      4. Total selected breaks <= max_total_breaks

    Args:
        candidates: Filtered candidates sorted strictly ascending by timestamp.
        min_gap_sec: Minimum spacing between consecutive breaks in seconds.
        max_per_hour: Maximum allowed breaks within any 3600-second interval.
        max_total_breaks: Upper limit on total approved breaks.
        first_break_not_before_sec: Earliest allowable break timestamp.

    Returns:
        (optimal_score, list_of_selected_indices_in_candidates)
    """
    n = len(candidates)
    if n == 0 or max_total_breaks <= 0:
        return 0.0, []

    timestamps = [float(c["timestamp"]) for c in candidates]
    scores = [float(c["cut_safety_score"]) for c in candidates]

    # Pre-filter out any candidate before first_break_not_before_sec
    valid_indices = [i for i in range(n) if timestamps[i] >= first_break_not_before_sec - 1e-4]
    if not valid_indices:
        return 0.0, []

    m = len(valid_indices)
    ts = [timestamps[idx] for idx in valid_indices]
    sc = [scores[idx] for idx in valid_indices]

    # Precompute next compatible index for each candidate (jump past min_gap_sec)
    next_compat = []
    for i in range(m):
        target_t = ts[i] + min_gap_sec - 1e-4
        nxt = bisect.bisect_left(ts, target_t)
        next_compat.append(nxt)

    memo: Dict[Tuple[int, int, Tuple[float, ...]], Tuple[float, Tuple[int, ...]]] = {}

    def clean_window(window: Tuple[float, ...], current_t: float) -> Tuple[float, ...]:
        cutoff = current_t - 3600.0 - 1e-4
        return tuple(t for t in window if t > cutoff)

    def dp(i: int, count: int, window: Tuple[float, ...]) -> Tuple[float, Tuple[int, ...]]:
        if i >= m or count >= max_total_breaks:
            return 0.0, ()

        curr_t = ts[i]
        active_w = clean_window(window, curr_t)
        state = (i, count, active_w)
        if state in memo:
            return memo[state]

        # Option 1: Skip candidate i
        best_score, best_chosen = dp(i + 1, count, active_w)

        # Option 2: Take candidate i (if compatible with rolling hourly limit)
        if len(active_w) < max_per_hour and count + 1 <= max_total_breaks:
            new_window = active_w + (curr_t,)
            nxt_i = next_compat[i]
            take_score, take_chosen = dp(nxt_i, count + 1, new_window)
            total_take = sc[i] + take_score
            if total_take > best_score + 1e-9:
                best_score = total_take
                best_chosen = (i,) + take_chosen

        memo[state] = (best_score, best_chosen)
        return memo[state]

    opt_score, chosen_sub_indices = dp(0, 0, ())
    original_indices = [valid_indices[sub_idx] for sub_idx in chosen_sub_indices]
    return round(opt_score, 4), original_indices


def apply_pacing_rules(
    break_candidates: List[Dict[str, Any]],
    rules: Optional[Dict[str, Any]] = None,
    config_path: Optional[str] = None,
    total_video_duration_sec: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates a list of break candidates against pacing rules.

    Args:
        break_candidates: Candidate objects from Phase 2 (with cut_safety_score).
        rules: Dict of pacing configuration (loaded from config_path if omitted).
        config_path: Path to pacing_rules.yaml.
        total_video_duration_sec: Duration of video in seconds (estimated from max timestamp if omitted).

    Returns:
        Updated list of break_candidate objects with:
          - pacing_eligible: bool
          - final_decision: "BREAK_APPROVED" | "REJECTED_UNSAFE" | "REJECTED_PACING"
          - pacing_reasons: dict with breaks_so_far_this_hour, min_gap_satisfied, ad_load_budget_remaining_sec
    """
    cfg = rules or load_pacing_rules(config_path)

    cut_safety_floor = float(cfg.get("cut_safety_floor", 0.55))
    min_gap_sec = float(cfg.get("min_gap_seconds", 480))
    max_per_hour = int(cfg.get("max_breaks_per_hour", 6))
    target_ad_load_pct = float(cfg.get("target_ad_load_pct", 12.0))
    first_break_sec = float(cfg.get("first_break_not_before_sec", 300))
    ad_duration_sec = float(cfg.get("default_ad_duration_sec", 15.0))

    out_candidates = [copy.deepcopy(c) for c in break_candidates]
    if not out_candidates:
        return []

    # Sort strictly by timestamp
    out_candidates.sort(key=lambda x: float(x.get("timestamp", 0.0)))

    # Determine video runtime
    if total_video_duration_sec is not None and total_video_duration_sec > 0:
        runtime_sec = float(total_video_duration_sec)
    else:
        max_ts = max(float(c.get("timestamp", 0.0)) for c in out_candidates)
        runtime_sec = max_ts + 60.0

    max_ad_load_budget_sec = runtime_sec * (target_ad_load_pct / 100.0)
    max_allowed_breaks = int(max_ad_load_budget_sec // ad_duration_sec)

    logger.info(
        "Pacing engine: runtime=%.1fs, budget=%.1fs, max_allowed_breaks=%d, floor=%.2f, min_gap=%.1fs",
        runtime_sec, max_ad_load_budget_sec, max_allowed_breaks, cut_safety_floor, min_gap_sec
    )

    # Step 1: Safety floor filtering
    eligible_indices = []
    for idx, cand in enumerate(out_candidates):
        score = float(cand.get("cut_safety_score", 0.0))
        if score < cut_safety_floor:
            cand["pacing_eligible"] = False
            cand["final_decision"] = "REJECTED_UNSAFE"
            cand["pacing_reasons"] = {
                "cut_safety_score": round(score, 3),
                "cut_safety_floor": cut_safety_floor,
                "reason": "below_cut_safety_floor"
            }
        else:
            cand["pacing_eligible"] = True
            cand["final_decision"] = "REJECTED_PACING"
            cand["pacing_reasons"] = {
                "min_gap_satisfied": False,
                "ad_load_budget_remaining_sec": round(max_ad_load_budget_sec, 1)
            }
            eligible_indices.append(idx)

    if not eligible_indices:
        logger.info("No candidates passed cut safety floor %.2f", cut_safety_floor)
        return out_candidates

    # Step 2: Solve optimal subset among safety-passing candidates using DP
    eligible_subset = [out_candidates[idx] for idx in eligible_indices]
    opt_score, chosen_sub_indices = solve_weighted_interval_schedule(
        candidates=eligible_subset,
        min_gap_sec=min_gap_sec,
        max_per_hour=max_per_hour,
        max_total_breaks=max_allowed_breaks,
        first_break_not_before_sec=first_break_sec
    )

    approved_global_indices = set(eligible_indices[sub_idx] for sub_idx in chosen_sub_indices)
    logger.info("DP selected %d approved breaks out of %d eligible candidates (total score: %.3f)",
                len(approved_global_indices), len(eligible_indices), opt_score)

    # Step 3: Populate bookkeeping fields and reasons
    approved_timestamps: List[float] = []
    approved_count_so_far = 0

    for idx, cand in enumerate(out_candidates):
        ts = float(cand["timestamp"])

        if idx in approved_global_indices:
            approved_count_so_far += 1
            approved_timestamps.append(ts)

            # Rolling 1-hour count
            breaks_in_hour = sum(1 for t in approved_timestamps if (ts - 3600.0) < t <= ts)
            remaining_budget = max(0.0, max_ad_load_budget_sec - (approved_count_so_far * ad_duration_sec))

            cand["pacing_eligible"] = True
            cand["final_decision"] = "BREAK_APPROVED"
            cand["pacing_reasons"] = {
                "breaks_so_far_this_hour": breaks_in_hour,
                "min_gap_satisfied": True,
                "ad_load_budget_remaining_sec": round(remaining_budget, 1)
            }
        elif cand["pacing_eligible"]:
            cand["final_decision"] = "REJECTED_PACING"
            if ts < first_break_sec:
                cand["pacing_reasons"] = {
                    "min_gap_satisfied": False,
                    "first_break_not_before_sec": first_break_sec,
                    "reason": "before_first_break_threshold"
                }
            else:
                too_close = any(abs(ts - app_t) < min_gap_sec for app_t in approved_timestamps)
                remaining_budget = max(0.0, max_ad_load_budget_sec - (approved_count_so_far * ad_duration_sec))
                cand["pacing_reasons"] = {
                    "min_gap_satisfied": not too_close,
                    "ad_load_budget_remaining_sec": round(remaining_budget, 1),
                    "reason": "min_gap_violation_with_selected_break" if too_close else "suboptimal_candidate_density"
                }

    return out_candidates
