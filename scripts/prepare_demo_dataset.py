"""
Prepare Demo Dataset for Phase 6
Generates sample ad video MP4 assets in assets/ads and demo/assets/ads,
and generates complete, realistic manifest.xml and debug.json for bhojon_bilashi.
Ensures zero-friction evaluation when launching python -m http.server.
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.vmap_builder import build_vmap_manifest
from pipeline.debug_emitter import assemble_debug_trace


def main():
    print("Preparing demo dataset for Phase 6 evaluation...")

    # 1. Run ad asset generator
    import scripts.create_sample_ad_assets as ad_gen
    ad_gen.main()

    # 2. Define realistic scenes for bhojon_bilashi
    scenes = [
        {
            "scene_id": "sc_001",
            "start_ts": 0.0,
            "end_ts": 185.0,
            "dominant_activity": "celebration party",
            "activity_tags": [
                {"label": "celebration party", "confidence": 0.88},
                {"label": "youth gathering", "confidence": 0.65},
                {"label": "family meal", "confidence": 0.42}
            ],
            "asr_text": "Welcome to our grand celebration! Let the festivities begin."
        },
        {
            "scene_id": "sc_002",
            "start_ts": 185.0,
            "end_ts": 320.0,
            "dominant_activity": "casual dining",
            "activity_tags": [
                {"label": "casual dining", "confidence": 0.92},
                {"label": "family meal", "confidence": 0.81},
                {"label": "street food", "confidence": 0.54}
            ],
            "asr_text": "The aromas here are simply irresistible. Let's see what is on the menu today."
        },
        {
            "scene_id": "sc_003",
            "start_ts": 320.0,
            "end_ts": 465.0,
            "dominant_activity": "home entertainment television",
            "activity_tags": [
                {"label": "home entertainment television", "confidence": 0.85},
                {"label": "couch relaxation", "confidence": 0.74},
                {"label": "movie night", "confidence": 0.63}
            ],
            "asr_text": "Everyone gather around the screen for tonight's premiere."
        },
        {
            "scene_id": "sc_004",
            "start_ts": 465.0,
            "end_ts": 1227.0,
            "dominant_activity": "road trip driving",
            "activity_tags": [
                {"label": "road trip driving", "confidence": 0.89},
                {"label": "scenic nature landscape", "confidence": 0.76}
            ],
            "asr_text": "Driving out into the scenic hills as the journey comes to a peaceful close."
        }
    ]

    # 3. Define candidate boundaries
    candidates = [
        # Candidate 1: Rejected unsafe at 58.0s (mid-speech cut)
        {
            "candidate_id": "brk_unsafe_01",
            "timestamp": 58.0,
            "scene_before": "sc_001",
            "scene_after": "sc_001",
            "cut_safety_score": 0.380,
            "cut_safety_reasons": {
                "sentence_boundary_aligned": False,
                "silence_gap_ms": 60,
                "shot_boundary_aligned": False,
                "action_continuity_penalty": 0.4
            },
            "pacing_eligible": False,
            "final_decision": "REJECTED_UNSAFE",
            "pacing_reasons": {"reason": "cut_safety_score 0.380 < threshold 0.550"}
        },
        # Break 1: Approved at 185.0s (> 180s buffer) (ZestCola)
        {
            "candidate_id": "brk_001",
            "timestamp": 185.0,
            "scene_before": "sc_001",
            "scene_after": "sc_002",
            "cut_safety_score": 0.942,
            "cut_safety_reasons": {
                "sentence_boundary_aligned": True,
                "silence_gap_ms": 920,
                "shot_boundary_aligned": True,
                "action_continuity_penalty": 0.0
            },
            "pacing_eligible": True,
            "final_decision": "BREAK_APPROVED",
            "pacing_reasons": {"safe_floor_passed": True, "min_gap_satisfied": True},
            "matched_brand": {
                "brand_id": "brand_zestcola",
                "display_name": "ZestCola",
                "creative_asset": "assets/ads/zestcola_15s.mp4",
                "duration_sec": 5.0
            }
        },
        # Candidate 3: Rejected pacing at 240.0s (within 55s of brk_001)
        {
            "candidate_id": "brk_rej_pacing_01",
            "timestamp": 240.0,
            "scene_before": "sc_002",
            "scene_after": "sc_002",
            "cut_safety_score": 0.850,
            "cut_safety_reasons": {
                "sentence_boundary_aligned": True,
                "silence_gap_ms": 750,
                "shot_boundary_aligned": True,
                "action_continuity_penalty": 0.0
            },
            "pacing_eligible": False,
            "final_decision": "REJECTED_PACING",
            "pacing_reasons": {"reason": "min_gap_seconds violation with previous break (55s < 120s)"}
        },
        # Break 2: Approved at 320.0s (gap = 135s >= 120s) (CrunchBite)
        {
            "candidate_id": "brk_002",
            "timestamp": 320.0,
            "scene_before": "sc_002",
            "scene_after": "sc_003",
            "cut_safety_score": 0.915,
            "cut_safety_reasons": {
                "sentence_boundary_aligned": True,
                "silence_gap_ms": 840,
                "shot_boundary_aligned": True,
                "action_continuity_penalty": 0.0
            },
            "pacing_eligible": True,
            "final_decision": "BREAK_APPROVED",
            "pacing_reasons": {"safe_floor_passed": True, "min_gap_satisfied": True},
            "matched_brand": {
                "brand_id": "brand_crunchbite",
                "display_name": "CrunchBite",
                "creative_asset": "assets/ads/crunchbite_20s.mp4",
                "duration_sec": 5.0
            }
        },
        # Break 3: Approved at 465.0s (gap = 145s >= 120s) (StreamFlix)
        {
            "candidate_id": "brk_003",
            "timestamp": 465.0,
            "scene_before": "sc_003",
            "scene_after": "sc_004",
            "cut_safety_score": 0.880,
            "cut_safety_reasons": {
                "sentence_boundary_aligned": True,
                "silence_gap_ms": 780,
                "shot_boundary_aligned": True,
                "action_continuity_penalty": 0.0
            },
            "pacing_eligible": True,
            "final_decision": "BREAK_APPROVED",
            "pacing_reasons": {"safe_floor_passed": True, "min_gap_satisfied": True},
            "matched_brand": {
                "brand_id": "brand_streamflix",
                "display_name": "StreamFlix",
                "creative_asset": "assets/ads/streamflix_15s.mp4",
                "duration_sec": 5.0
            }
        }
    ]

    # 4. Brand match data (with adversarial block proofs)
    brand_matches = [
        {
            "candidate_id": "brk_001",
            "top_brand": "brand_zestcola",
            "dominant_activity": "celebration party",
            "activity_tags_used": [
                {"label": "celebration party", "confidence": 0.88},
                {"label": "youth gathering", "confidence": 0.65}
            ],
            "ranked_brands": [
                {"brand_id": "brand_zestcola", "display_name": "ZestCola", "score": 0.8240},
                {"brand_id": "brand_crunchbite", "display_name": "CrunchBite", "score": 0.6120},
                {"brand_id": "brand_streamflix", "display_name": "StreamFlix", "score": 0.4510}
            ],
            "blocked_brands": [
                {
                    "brand_id": "brand_shelterwise",
                    "display_name": "ShelterWise Insurance",
                    "block_reason": "Tone mismatch and low target affinity with high-energy party"
                }
            ]
        },
        {
            "candidate_id": "brk_002",
            "top_brand": "brand_crunchbite",
            "dominant_activity": "casual dining",
            "activity_tags_used": [
                {"label": "casual dining", "confidence": 0.92},
                {"label": "family meal", "confidence": 0.81}
            ],
            "ranked_brands": [
                {"brand_id": "brand_crunchbite", "display_name": "CrunchBite", "score": 0.8910},
                {"brand_id": "brand_munchpop", "display_name": "MunchPop Snacks", "score": 0.7430},
                {"brand_id": "brand_zestcola", "display_name": "ZestCola", "score": 0.6820}
            ],
            "blocked_brands": []
        },
        {
            "candidate_id": "brk_003",
            "top_brand": "brand_streamflix",
            "dominant_activity": "home entertainment television",
            "activity_tags_used": [
                {"label": "home entertainment television", "confidence": 0.85},
                {"label": "couch relaxation", "confidence": 0.74}
            ],
            "ranked_brands": [
                {"brand_id": "brand_streamflix", "display_name": "StreamFlix", "score": 0.9150},
                {"brand_id": "brand_novatel", "display_name": "Novatel Mobile", "score": 0.7200},
                {"brand_id": "brand_munchpop", "display_name": "MunchPop Snacks", "score": 0.6340}
            ],
            "blocked_brands": []
        }
    ]

    # Video datasets configuration
    video_configs = [
        {
            "video_id": "bhojon_bilashi",
            "scenes": scenes,
            "candidates": candidates,
            "brand_matches": brand_matches
        },
        {
            "video_id": "indubala_bhaater_hotel",
            "scenes": [
                {
                    "scene_id": "sc_ind_01",
                    "start_ts": 0.0,
                    "end_ts": 190.0,
                    "dominant_activity": "traditional cooking",
                    "activity_tags": [
                        {"label": "traditional cooking", "confidence": 0.94},
                        {"label": "kitchen preparation", "confidence": 0.88},
                        {"label": "heritage culture", "confidence": 0.75}
                    ],
                    "asr_text": "In this kitchen, the spices tell stories of an era gone by."
                },
                {
                    "scene_id": "sc_ind_02",
                    "start_ts": 190.0,
                    "end_ts": 340.0,
                    "dominant_activity": "family meal dining",
                    "activity_tags": [
                        {"label": "family meal dining", "confidence": 0.91},
                        {"label": "casual dining", "confidence": 0.84}
                    ],
                    "asr_text": "The patrons arrive daily for the warm rice and signature fish curry."
                },
                {
                    "scene_id": "sc_ind_03",
                    "start_ts": 340.0,
                    "end_ts": 600.0,
                    "dominant_activity": "nostalgic conversation",
                    "activity_tags": [
                        {"label": "nostalgic conversation", "confidence": 0.86},
                        {"label": "quiet contemplation", "confidence": 0.72}
                    ],
                    "asr_text": "Memories linger like the fragrant scent of mustard oil."
                }
            ],
            "candidates": [
                {
                    "candidate_id": "brk_ind_001",
                    "timestamp": 190.0,
                    "scene_before": "sc_ind_01",
                    "scene_after": "sc_ind_02",
                    "cut_safety_score": 0.935,
                    "cut_safety_reasons": {
                        "sentence_boundary_aligned": True,
                        "silence_gap_ms": 860,
                        "shot_boundary_aligned": True,
                        "action_continuity_penalty": 0.0
                    },
                    "pacing_eligible": True,
                    "final_decision": "BREAK_APPROVED",
                    "pacing_reasons": {"safe_floor_passed": True, "min_gap_satisfied": True},
                    "matched_brand": {
                        "brand_id": "brand_crunchbite",
                        "display_name": "CrunchBite",
                        "creative_asset": "assets/ads/crunchbite_20s.mp4",
                        "duration_sec": 5.0
                    }
                },
                {
                    "candidate_id": "brk_ind_002",
                    "timestamp": 340.0,
                    "scene_before": "sc_ind_02",
                    "scene_after": "sc_ind_03",
                    "cut_safety_score": 0.910,
                    "cut_safety_reasons": {
                        "sentence_boundary_aligned": True,
                        "silence_gap_ms": 910,
                        "shot_boundary_aligned": True,
                        "action_continuity_penalty": 0.0
                    },
                    "pacing_eligible": True,
                    "final_decision": "BREAK_APPROVED",
                    "pacing_reasons": {"safe_floor_passed": True, "min_gap_satisfied": True},
                    "matched_brand": {
                        "brand_id": "brand_zestcola",
                        "display_name": "ZestCola",
                        "creative_asset": "assets/ads/zestcola_15s.mp4",
                        "duration_sec": 5.0
                    }
                }
            ],
            "brand_matches": [
                {
                    "candidate_id": "brk_ind_001",
                    "top_brand": "brand_crunchbite",
                    "dominant_activity": "traditional cooking",
                    "activity_tags_used": [
                        {"label": "traditional cooking", "confidence": 0.94},
                        {"label": "family meal dining", "confidence": 0.88}
                    ],
                    "ranked_brands": [
                        {"brand_id": "brand_crunchbite", "display_name": "CrunchBite", "score": 0.9240},
                        {"brand_id": "brand_munchpop", "display_name": "MunchPop Snacks", "score": 0.7810},
                        {"brand_id": "brand_zestcola", "display_name": "ZestCola", "score": 0.6540}
                    ],
                    "blocked_brands": []
                },
                {
                    "candidate_id": "brk_ind_002",
                    "top_brand": "brand_zestcola",
                    "dominant_activity": "family meal dining",
                    "activity_tags_used": [
                        {"label": "family meal dining", "confidence": 0.91}
                    ],
                    "ranked_brands": [
                        {"brand_id": "brand_zestcola", "display_name": "ZestCola", "score": 0.8120},
                        {"brand_id": "brand_streamflix", "display_name": "StreamFlix", "score": 0.5210}
                    ],
                    "blocked_brands": []
                }
            ]
        },
        {
            "video_id": "mohanagar",
            "scenes": [
                {
                    "scene_id": "sc_moh_01",
                    "start_ts": 0.0,
                    "end_ts": 195.0,
                    "dominant_activity": "police station interrogation",
                    "activity_tags": [
                        {"label": "police station interrogation", "confidence": 0.96},
                        {"label": "crime drama", "confidence": 0.92},
                        {"label": "tense confrontation", "confidence": 0.85}
                    ],
                    "asr_text": "Tell me what happened that night at the intersection. No lies."
                },
                {
                    "scene_id": "sc_moh_02",
                    "start_ts": 195.0,
                    "end_ts": 350.0,
                    "dominant_activity": "investigative mystery",
                    "activity_tags": [
                        {"label": "investigative mystery", "confidence": 0.89},
                        {"label": "legal conflict", "confidence": 0.77}
                    ],
                    "asr_text": "The evidence does not align with your statement, Officer."
                }
            ],
            "candidates": [
                {
                    "candidate_id": "brk_moh_001",
                    "timestamp": 195.0,
                    "scene_before": "sc_moh_01",
                    "scene_after": "sc_moh_02",
                    "cut_safety_score": 0.928,
                    "cut_safety_reasons": {
                        "sentence_boundary_aligned": True,
                        "silence_gap_ms": 780,
                        "shot_boundary_aligned": True,
                        "action_continuity_penalty": 0.0
                    },
                    "pacing_eligible": True,
                    "final_decision": "BREAK_APPROVED",
                    "pacing_reasons": {"safe_floor_passed": True, "min_gap_satisfied": True},
                    "matched_brand": {
                        "brand_id": "brand_streamflix",
                        "display_name": "StreamFlix",
                        "creative_asset": "assets/ads/streamflix_15s.mp4",
                        "duration_sec": 5.0
                    }
                }
            ],
            "brand_matches": [
                {
                    "candidate_id": "brk_moh_001",
                    "top_brand": "brand_streamflix",
                    "dominant_activity": "police station interrogation",
                    "activity_tags_used": [
                        {"label": "crime drama", "confidence": 0.92},
                        {"label": "police station interrogation", "confidence": 0.96}
                    ],
                    "ranked_brands": [
                        {"brand_id": "brand_streamflix", "display_name": "StreamFlix", "score": 0.8920},
                        {"brand_id": "brand_novatel", "display_name": "Novatel Mobile", "score": 0.7150}
                    ],
                    "blocked_brands": [
                        {
                            "brand_id": "brand_zestcola",
                            "display_name": "ZestCola",
                            "block_reason": "Hard-block: negative context match with crime / confrontation (score = 0.0)"
                        },
                        {
                            "brand_id": "brand_crunchbite",
                            "display_name": "CrunchBite",
                            "block_reason": "Hard-block: negative context match with violence / crime interrogation (score = 0.0)"
                        }
                    ]
                }
            ]
        },
        {
            "video_id": "money_honey",
            "scenes": [
                {
                    "scene_id": "sc_mh_01",
                    "start_ts": 0.0,
                    "end_ts": 185.0,
                    "dominant_activity": "high stakes heist",
                    "activity_tags": [
                        {"label": "high stakes heist", "confidence": 0.95},
                        {"label": "action thriller", "confidence": 0.91},
                        {"label": "urban pursuit", "confidence": 0.82}
                    ],
                    "asr_text": "We have only three minutes before the vault alarm locks down the perimeter."
                },
                {
                    "scene_id": "sc_mh_02",
                    "start_ts": 185.0,
                    "end_ts": 360.0,
                    "dominant_activity": "fast paced getaway",
                    "activity_tags": [
                        {"label": "fast paced getaway", "confidence": 0.93},
                        {"label": "road trip driving", "confidence": 0.86}
                    ],
                    "asr_text": "Step on the gas! The patrol cars are closing in from the north."
                }
            ],
            "candidates": [
                {
                    "candidate_id": "brk_mh_001",
                    "timestamp": 185.0,
                    "scene_before": "sc_mh_01",
                    "scene_after": "sc_mh_02",
                    "cut_safety_score": 0.895,
                    "cut_safety_reasons": {
                        "sentence_boundary_aligned": True,
                        "silence_gap_ms": 720,
                        "shot_boundary_aligned": True,
                        "action_continuity_penalty": 0.0
                    },
                    "pacing_eligible": True,
                    "final_decision": "BREAK_APPROVED",
                    "pacing_reasons": {"safe_floor_passed": True, "min_gap_satisfied": True},
                    "matched_brand": {
                        "brand_id": "brand_novatel",
                        "display_name": "Novatel Mobile",
                        "creative_asset": "assets/ads/novatel_15s.mp4",
                        "duration_sec": 5.0
                    }
                }
            ],
            "brand_matches": [
                {
                    "candidate_id": "brk_mh_001",
                    "top_brand": "brand_novatel",
                    "dominant_activity": "high stakes heist",
                    "activity_tags_used": [
                        {"label": "action thriller", "confidence": 0.91},
                        {"label": "fast paced getaway", "confidence": 0.93}
                    ],
                    "ranked_brands": [
                        {"brand_id": "brand_novatel", "display_name": "Novatel Mobile", "score": 0.8840},
                        {"brand_id": "brand_streamflix", "display_name": "StreamFlix", "score": 0.8410}
                    ],
                    "blocked_brands": [
                        {
                            "brand_id": "brand_zestcola",
                            "display_name": "ZestCola",
                            "block_reason": "Hard-block: negative context match with heist violence (score = 0.0)"
                        }
                    ]
                }
            ]
        }
    ]

    demo_dir = REPO_ROOT / "demo"
    demo_data_dir = demo_dir / "data"
    demo_data_dir.mkdir(parents=True, exist_ok=True)

    for cfg in video_configs:
        v_id = cfg["video_id"]
        v_scenes = cfg["scenes"]
        v_candidates = cfg["candidates"]
        v_matches = cfg["brand_matches"]

        processed_dir = REPO_ROOT / "data" / "processed" / v_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        v_demo_dir = demo_data_dir / v_id
        v_demo_dir.mkdir(parents=True, exist_ok=True)

        approved = [c for c in v_candidates if c["final_decision"] == "BREAK_APPROVED"]

        # Manifest XML
        manifest_p1 = processed_dir / "manifest.xml"
        manifest_p2 = v_demo_dir / "manifest.xml"
        xml_content = build_vmap_manifest(approved, output_path=str(manifest_p1), validate=True)
        with open(manifest_p2, "w", encoding="utf-8") as f:
            f.write(xml_content)

        # Debug JSON
        debug_p1 = processed_dir / "debug.json"
        debug_p2 = v_demo_dir / "debug.json"
        assemble_debug_trace(v_id, v_scenes, v_candidates, v_matches, output_path=str(debug_p1))
        assemble_debug_trace(v_id, v_scenes, v_candidates, v_matches, output_path=str(debug_p2))

        # Also copy default bhojon_bilashi directly into demo/ root
        if v_id == "bhojon_bilashi":
            with open(demo_dir / "manifest.xml", "w", encoding="utf-8") as f:
                f.write(xml_content)
            with open(demo_dir / "debug.json", "w", encoding="utf-8") as f:
                with open(debug_p1, "r", encoding="utf-8") as df:
                    f.write(df.read())

        print(f"  [OK] Processed & emitted manifests for {v_id}")

    print("Demo dataset preparation for ALL 4 videos COMPLETE!")


if __name__ == "__main__":
    main()

