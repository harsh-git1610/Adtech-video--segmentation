"""
Acceptance Test 2 (Phase 4): New Brand Generalization.

Verifies that adding a 9th brand to brand_catalogue.json requires ZERO changes to
pipeline/brand_matcher.py. The test:
  1. Creates a temporary catalogue with an extra brand.
  2. Instantiates BrandMatcher with that catalogue.
  3. Confirms the 9th brand participates in ranking and its negative blocks work.
  4. Verifies the 9th brand wins on a scene that fits its target contexts perfectly.

The 9th brand object is defined ONLY here as data — no pipeline code is touched.
"""

import json
import tempfile
from pathlib import Path
import pytest

from pipeline.brand_matcher import BrandMatcher

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = REPO_ROOT / "data" / "brand_catalogue.json"

# 9th brand defined as pure data in the test file — generic enough to validate generalization
NINTH_BRAND = {
    "brand_id": "brand_fitlife",
    "display_name": "FitLife Wellness",
    "category": "health_fitness",
    "target_contexts": [
        "gym workout",
        "morning run",
        "yoga session",
        "outdoor adventure",
        "healthy cooking",
        "sports competition",
        "marathon training"
    ],
    "negative_contexts": [
        "funeral",
        "grief",
        "hospital",
        "medical_emergency",
        "violence",
        "accident",
        "mourning",
        "tragedy",
        "eating_disorder_context"
    ],
    "creative_asset": "assets/ads/fitlife_20s.mp4",
    "duration_sec": 20,
    "tone": "motivational"
}


@pytest.fixture(scope="module")
def base_catalogue():
    with open(CATALOGUE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def extended_catalogue_path(base_catalogue, tmp_path_factory):
    """Write a catalogue containing original 8 brands + 1 new brand to a temp file."""
    extended = base_catalogue + [NINTH_BRAND]
    tmp_path = tmp_path_factory.mktemp("data") / "extended_catalogue.json"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(extended, f, indent=2)
    return str(tmp_path)


@pytest.fixture(scope="module")
def matcher_9(extended_catalogue_path):
    return BrandMatcher(
        catalogue_path=extended_catalogue_path,
        block_threshold=0.55,
        block_conf_floor=0.30
    )


def _scene(dominant, tags):
    return {
        "scene_id": "sc_gen_test",
        "start_ts": 0.0,
        "end_ts": 30.0,
        "dominant_activity": dominant,
        "dominant_activity": dominant,
        "activity_tags": [{"label": lbl, "confidence": c} for lbl, c in tags],
        "asr_text": ""
    }


def _candidate():
    return {
        "candidate_id": "brk_gen",
        "timestamp": 30.0,
        "cut_safety_score": 0.85,
        "scene_before": "sc_gen_test",
        "scene_after": "sc_gen_test",
        "final_decision": "BREAK_APPROVED"
    }


class TestNewBrandGeneralization:

    def test_ninth_brand_loaded(self, matcher_9):
        """BrandMatcher must have 9 brands when extended catalogue is passed."""
        assert len(matcher_9.brands) == 9, f"Expected 9 brands, got {len(matcher_9.brands)}"
        ids = [b["brand_id"] for b in matcher_9.brands]
        assert NINTH_BRAND["brand_id"] in ids

    def test_ninth_brand_appears_in_ranked_output(self, matcher_9):
        """
        On a gym/sports scene the 9th brand should appear in ranked output
        (not necessarily #1 — just present, indicating it was considered).
        """
        scene = _scene(
            dominant="gym workout",
            tags=[("gym workout", 0.80), ("sports competition", 0.60)]
        )
        result = matcher_9.match(_candidate(), scene_before=scene, scene_after=None)

        ranked_ids = [r["brand_id"] for r in result["ranked_brands"]]
        assert NINTH_BRAND["brand_id"] in ranked_ids, (
            f"9th brand not present in ranked output: {ranked_ids}"
        )

    def test_ninth_brand_wins_on_perfect_fit_scene(self, matcher_9):
        """
        On a scene that is a perfect fit for the 9th brand, it should rank #1.
        """
        scene = _scene(
            dominant="marathon training",
            tags=[
                ("marathon training", 0.88),
                ("outdoor adventure", 0.65),
                ("morning run", 0.55)
            ]
        )
        result = matcher_9.match(_candidate(), scene_before=scene, scene_after=None)

        assert result["n_surviving"] >= 1
        top = result["ranked_brands"][0]["brand_id"]
        # The 9th brand should win — its target_contexts are highly fit for this scene
        assert top == NINTH_BRAND["brand_id"], (
            f"Expected 9th brand to win on marathon scene but got '{top}'. "
            f"Full ranking: {result['ranked_brands']}"
        )

    def test_ninth_brand_blocked_on_funeral_scene(self, matcher_9):
        """
        The 9th brand lists 'funeral' in negative_contexts. A funeral scene must block it.
        """
        scene = _scene(
            dominant="funeral ceremony",
            tags=[("funeral ceremony", 0.85), ("mourning and grief", 0.60)]
        )
        result = matcher_9.match(_candidate(), scene_before=scene, scene_after=None)

        blocked_ids = {b["brand_id"] for b in result["blocked_brands"]}
        assert NINTH_BRAND["brand_id"] in blocked_ids, (
            f"9th brand should be blocked on funeral scene but survived. "
            f"Blocked: {blocked_ids}"
        )

    def test_ninth_brand_blocked_by_synonym_hospital(self, matcher_9):
        """
        '医院 hospital scene' (or just 'hospital scene') should block the 9th brand
        via its 'hospital' negative context.
        """
        scene = _scene(
            dominant="hospital scene",
            tags=[("hospital scene", 0.75), ("medical emergency", 0.50)]
        )
        result = matcher_9.match(_candidate(), scene_before=scene, scene_after=None)

        blocked_ids = {b["brand_id"] for b in result["blocked_brands"]}
        assert NINTH_BRAND["brand_id"] in blocked_ids, (
            f"9th brand (which has 'hospital' in negatives) must be blocked on 'hospital scene'. "
            f"Blocked: {blocked_ids}"
        )

    def test_original_brands_unaffected_by_ninth_brand(self, matcher_9, base_catalogue):
        """
        For a safe scene, all 8 original brands plus the 9th participate in ranking normally.
        The addition of the 9th brand must not cause spurious blocks on any original brand.
        """
        scene = _scene(
            dominant="birthday party",
            tags=[("birthday party", 0.82), ("celebration party", 0.70)]
        )
        result = matcher_9.match(_candidate(), scene_before=scene, scene_after=None)

        original_ids = {b["brand_id"] for b in base_catalogue}
        blocked_ids = {b["brand_id"] for b in result["blocked_brands"]}
        spuriously_blocked = blocked_ids & original_ids

        assert len(spuriously_blocked) == 0, (
            f"Adding 9th brand caused spurious blocks on original brands: {spuriously_blocked}"
        )
        # Total brands considered = 9
        total_considered = result["n_surviving"] + result["n_blocked"]
        assert total_considered == 9, f"Expected 9 total brands, got {total_considered}"

    def test_zero_code_changes_required(self):
        """
        Meta-test: BrandMatcher can be instantiated multiple times with different catalogue
        files without any import changes, class changes, or special-case code.
        This is the 'generalization' guarantee: brand data is hot-swappable.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([NINTH_BRAND], f)
            single_brand_path = f.name

        matcher_single = BrandMatcher(
            catalogue_path=single_brand_path,
            block_threshold=0.55
        )
        assert len(matcher_single.brands) == 1
        scene = _scene("gym workout", [("gym workout", 0.8)])
        result = matcher_single.match(_candidate(), scene_before=scene, scene_after=None)
        assert result["top_brand"] == NINTH_BRAND["brand_id"] or result["n_surviving"] == 1
