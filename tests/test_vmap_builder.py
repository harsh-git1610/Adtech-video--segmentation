"""
Tests for Phase 5: VMAP 1.0.1 Builder & Debug Trace Emitter
Validates XSD compliance, debug.json schema integrity, and consistency between VMAP and debug.json.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest

from pipeline.vmap_builder import (
    build_vmap_manifest,
    validate_vmap,
    format_timestamp_vmap,
    format_duration_hms,
)
from pipeline.debug_emitter import assemble_debug_trace

REPO_ROOT = Path(__file__).resolve().parent.parent
XSD_PATH = REPO_ROOT / "schemas" / "vmap-1.0.1.xsd"


@pytest.fixture
def sample_approved_breaks():
    return [
        {
            "candidate_id": "brk_001",
            "timestamp": 45.25,
            "final_decision": "BREAK_APPROVED",
            "pacing_eligible": True,
            "cut_safety_score": 0.88,
            "matched_brand": {
                "brand_id": "brand_zestcola",
                "display_name": "ZestCola",
                "creative_asset": "assets/ads/zestcola_15s.mp4",
                "duration_sec": 15
            }
        },
        {
            "candidate_id": "brk_002",
            "timestamp": 310.50,
            "final_decision": "BREAK_APPROVED",
            "pacing_eligible": True,
            "cut_safety_score": 0.92,
            "matched_brand": {
                "brand_id": "brand_crunchbite",
                "display_name": "CrunchBite",
                "creative_asset": "assets/ads/crunchbite_20s.mp4",
                "duration_sec": 20
            }
        },
        {
            "candidate_id": "brk_003",
            "timestamp": 620.00,
            "final_decision": "BREAK_APPROVED",
            "pacing_eligible": True,
            "cut_safety_score": 0.79,
            "matched_brand": {
                "brand_id": "brand_streamflix",
                "display_name": "StreamFlix",
                "creative_asset": "assets/ads/streamflix_15s.mp4",
                "duration_sec": 15
            }
        }
    ]


@pytest.fixture
def sample_full_candidates(sample_approved_breaks):
    rejected = [
        {
            "candidate_id": "brk_rej_01",
            "timestamp": 120.0,
            "final_decision": "REJECTED_UNSAFE",
            "pacing_eligible": False,
            "cut_safety_score": 0.42,
            "cut_safety_reasons": {"silence_gap_ms": 100}
        },
        {
            "candidate_id": "brk_rej_02",
            "timestamp": 330.0,
            "final_decision": "REJECTED_PACING_MIN_GAP",
            "pacing_eligible": False,
            "cut_safety_score": 0.85,
            "pacing_reasons": {"min_gap_violation": True}
        }
    ]
    return sample_approved_breaks + rejected


@pytest.fixture
def sample_scenes():
    return [
        {
            "scene_id": "sc_001",
            "start_ts": 0.0,
            "end_ts": 45.25,
            "dominant_activity": "celebration party",
            "activity_tags": [{"label": "celebration party", "confidence": 0.85}],
            "asr_text": "Hello and welcome to the party!"
        },
        {
            "scene_id": "sc_002",
            "start_ts": 45.25,
            "end_ts": 310.50,
            "dominant_activity": "casual dining",
            "activity_tags": [{"label": "casual dining", "confidence": 0.72}],
            "asr_text": "The food looks delicious today."
        }
    ]


class TestVMAPBuilder:

    def test_format_timestamp_vmap(self):
        assert format_timestamp_vmap(0.0) == "00:00:00.000"
        assert format_timestamp_vmap(45.25) == "00:00:45.250"
        assert format_timestamp_vmap(3661.05) == "01:01:01.050"
        assert format_timestamp_vmap(7200.0) == "02:00:00.000"

    def test_format_duration_hms(self):
        assert format_duration_hms(15) == "00:00:15"
        assert format_duration_hms(60) == "00:01:00"
        assert format_duration_hms(90) == "00:01:30"

    def test_vmap_manifest_structure_and_validation(self, sample_approved_breaks, tmp_path):
        manifest_file = tmp_path / "manifest.xml"
        xml_content = build_vmap_manifest(
            approved_breaks=sample_approved_breaks,
            output_path=str(manifest_file),
            xsd_path=str(XSD_PATH),
            validate=True
        )

        assert manifest_file.exists()
        assert 'xmlns:vmap="http://www.iab.net/videosuite/vmap"' in xml_content
        assert 'version="1.0"' in xml_content

        # Parse XML tree
        root = ET.fromstring(xml_content)
        # Check namespace handling
        ns = {"vmap": "http://www.iab.net/videosuite/vmap"}
        ad_breaks = root.findall("vmap:AdBreak", ns)
        assert len(ad_breaks) == len(sample_approved_breaks)

        for brk_elem, expected in zip(ad_breaks, sample_approved_breaks):
            assert brk_elem.attrib["breakType"] == "linear"
            assert brk_elem.attrib["breakId"] == expected["candidate_id"]
            expected_time = format_timestamp_vmap(expected["timestamp"])
            assert brk_elem.attrib["timeOffset"] == expected_time

    def test_invalid_vmap_raises_error(self, tmp_path):
        bad_xml_file = tmp_path / "invalid_manifest.xml"
        # Missing required breakType attribute
        bad_xml = """<?xml version="1.0" encoding="utf-8"?>
<vmap:VMAP xmlns:vmap="http://www.iab.net/videosuite/vmap" version="1.0">
  <vmap:AdBreak timeOffset="00:00:30.000">
  </vmap:AdBreak>
</vmap:VMAP>
"""
        bad_xml_file.write_text(bad_xml, encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            validate_vmap(str(bad_xml_file), xsd_path=str(XSD_PATH))
        assert "validation" in str(excinfo.value).lower() or "breaktype" in str(excinfo.value).lower()


class TestDebugTraceEmitter:

    def test_debug_json_schema_completeness(self, sample_scenes, sample_full_candidates, tmp_path):
        debug_file = tmp_path / "debug.json"
        trace = assemble_debug_trace(
            video_id="video_sample_01",
            scenes=sample_scenes,
            break_candidates=sample_full_candidates,
            output_path=str(debug_file)
        )

        assert debug_file.exists()
        with open(debug_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["metadata"]["video_id"] == "video_sample_01"
        assert loaded["metadata"]["summary"]["total_scenes"] == len(sample_scenes)
        assert loaded["metadata"]["summary"]["total_candidates"] == len(sample_full_candidates)
        assert loaded["metadata"]["summary"]["approved_breaks"] == 3
        assert loaded["metadata"]["summary"]["rejected_breaks"] == 2

        # Verify candidate enrichment
        for cand in loaded["break_candidates"]:
            assert "candidate_id" in cand
            assert "timestamp" in cand
            assert "cut_safety" in cand
            assert "pacing" in cand
            assert "brand_selection" in cand

    def test_no_timestamp_mismatches_between_vmap_and_debug(
        self, sample_scenes, sample_full_candidates, sample_approved_breaks, tmp_path
    ):
        """
        Verify that manifest.xml contains EXACTLY the timestamps of approved candidates in debug.json,
        and never leaks rejected candidate timestamps.
        """
        manifest_file = tmp_path / "manifest.xml"
        debug_file = tmp_path / "debug.json"

        # Emit both artifacts
        build_vmap_manifest(
            approved_breaks=sample_approved_breaks,
            output_path=str(manifest_file),
            xsd_path=str(XSD_PATH),
            validate=True
        )
        debug_trace = assemble_debug_trace(
            video_id="test_consistency",
            scenes=sample_scenes,
            break_candidates=sample_full_candidates,
            output_path=str(debug_file)
        )

        # Parse timestamps from manifest.xml
        root = ET.parse(manifest_file).getroot()
        ns = {"vmap": "http://www.iab.net/videosuite/vmap"}
        manifest_time_offsets = [b.attrib["timeOffset"] for b in root.findall("vmap:AdBreak", ns)]

        # Extract timestamps from debug.json
        approved_in_debug = [
            c for c in debug_trace["break_candidates"]
            if c["pacing"]["final_decision"] == "BREAK_APPROVED"
        ]
        rejected_in_debug = [
            c for c in debug_trace["break_candidates"]
            if c["pacing"]["final_decision"] != "BREAK_APPROVED"
        ]

        expected_time_offsets = [format_timestamp_vmap(c["timestamp"]) for c in approved_in_debug]
        rejected_time_offsets = [format_timestamp_vmap(c["timestamp"]) for c in rejected_in_debug]

        # 1. Total counts match
        assert len(manifest_time_offsets) == len(approved_in_debug)

        # 2. Every approved timestamp is present in VMAP manifest
        assert set(manifest_time_offsets) == set(expected_time_offsets)

        # 3. No rejected break timestamp leaked into manifest
        for rej_ts in rejected_time_offsets:
            assert rej_ts not in manifest_time_offsets
