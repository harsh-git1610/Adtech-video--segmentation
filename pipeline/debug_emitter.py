"""
Debug Emitter (Phase 5)

Assembles the complete 'show your work' debug trace (debug.json) for an entire video:
- Full scene objects with ASR transcripts and activity tags
- Per break candidate: cut_safety_score + sub-signal reasons, pacing decisions + constraints,
  and full brand matching rankings and block-reasons
- Summary statistics for quick auditing and evaluation
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional


def assemble_debug_trace(
    video_id: str,
    scenes: List[Dict[str, Any]],
    break_candidates: List[Dict[str, Any]],
    brand_matches: Optional[List[Dict[str, Any]]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Assemble the complete debug.json data structure and optionally write it to disk.

    Args:
        video_id: Unique identifier of the video.
        scenes: Full scene objects from Phase 1 & 4.
        break_candidates: Candidates with cut safety (Phase 2) and pacing decisions (Phase 3).
        brand_matches: Brand match results (Phase 4) with rankings and block reasons.
        output_path: If provided, path where debug.json should be saved.
    """
    brand_match_map: Dict[str, Dict[str, Any]] = {}
    if brand_matches:
        for bm in brand_matches:
            cid = bm.get("candidate_id")
            if cid:
                brand_match_map[cid] = bm

    enriched_candidates = []
    approved_count = 0
    rejected_count = 0
    total_inserted_ad_sec = 0.0

    for cand in break_candidates:
        cid = cand.get("candidate_id", "")
        decision = cand.get("final_decision", "UNSET")
        if decision == "BREAK_APPROVED":
            approved_count += 1
        else:
            rejected_count += 1

        bm_data = brand_match_map.get(cid, {})

        # Inherit matched brand details if present in candidate or brand match
        matched_brand = cand.get("matched_brand")
        if not matched_brand and bm_data:
            top_brand_id = bm_data.get("top_brand")
            matched_brand = {
                "brand_id": top_brand_id,
                "display_name": None,
                "creative_asset": None,
                "duration_sec": 15.0
            }
            for rb in bm_data.get("ranked_brands", []):
                if rb.get("brand_id") == top_brand_id:
                    matched_brand["display_name"] = rb.get("display_name")
                    break

        if decision == "BREAK_APPROVED" and matched_brand:
            total_inserted_ad_sec += float(matched_brand.get("duration_sec", 15.0))

        enriched_entry = {
            "candidate_id": cid,
            "timestamp": cand.get("timestamp", 0.0),
            "scene_before": cand.get("scene_before"),
            "scene_after": cand.get("scene_after"),
            # Phase 2: Cut Safety
            "cut_safety": {
                "score": cand.get("cut_safety_score"),
                "reasons": cand.get("cut_safety_reasons", {})
            },
            # Phase 3: Pacing
            "pacing": {
                "eligible": cand.get("pacing_eligible"),
                "final_decision": decision,
                "reasons": cand.get("pacing_reasons", {})
            },
            # Phase 4: Brand Selection
            "brand_selection": {
                "matched_brand": matched_brand,
                "ranked_brands": bm_data.get("ranked_brands", []),
                "blocked_brands": bm_data.get("blocked_brands", []),
                "dominant_activity": bm_data.get("dominant_activity"),
                "activity_tags_used": bm_data.get("activity_tags_used", [])
            }
        }
        enriched_candidates.append(enriched_entry)

    debug_trace = {
        "metadata": {
            "video_id": video_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "1.0.0",
            "summary": {
                "total_scenes": len(scenes),
                "total_candidates": len(break_candidates),
                "approved_breaks": approved_count,
                "rejected_breaks": rejected_count,
                "total_inserted_ad_seconds": round(total_inserted_ad_sec, 2)
            }
        },
        "scenes": scenes,
        "break_candidates": enriched_candidates
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(debug_trace, f, indent=2, ensure_ascii=False)

    return debug_trace
