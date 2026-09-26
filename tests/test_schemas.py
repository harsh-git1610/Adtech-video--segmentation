#!/usr/bin/env python3
"""
test_schemas.py
---------------
Pytest suite for Phase 0 schema validation:

(a) brand_catalogue.json validates against brand.schema.json
(b) A deliberately malformed brand (missing negative_contexts) is REJECTED
(c) pacing_rules.yaml has all four required keys with correct types
"""

import copy
import json
from pathlib import Path

import pytest
import yaml
from jsonschema.validators import validator_for

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"
DATA_DIR = REPO_ROOT / "data"
CONFIG_DIR = REPO_ROOT / "config"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_validator(schema: dict):
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def brand_schema():
    return load_json(SCHEMAS_DIR / "brand.schema.json")


@pytest.fixture(scope="module")
def brand_catalogue():
    return load_json(DATA_DIR / "brand_catalogue.json")


@pytest.fixture(scope="module")
def pacing_rules():
    return load_yaml(CONFIG_DIR / "pacing_rules.yaml")


# ── (a) brand_catalogue.json validates against brand.schema.json ──────────────
class TestBrandCatalogueValid:
    """Every brand in the shipped catalogue must pass schema validation."""

    def test_catalogue_is_nonempty_array(self, brand_catalogue):
        assert isinstance(brand_catalogue, list), "Catalogue must be a JSON array"
        assert len(brand_catalogue) >= 8, (
            f"Expected at least 8 brands, got {len(brand_catalogue)}"
        )

    def test_each_brand_validates(self, brand_schema, brand_catalogue):
        validator = make_validator(brand_schema)
        for idx, brand in enumerate(brand_catalogue):
            brand_id = brand.get("brand_id", f"index_{idx}")
            errors = list(validator.iter_errors(brand))
            assert errors == [], (
                f"Brand '{brand_id}' failed validation:\n"
                + "\n".join(f"  - {e.message}" for e in errors)
            )

    def test_all_brand_ids_unique(self, brand_catalogue):
        ids = [b["brand_id"] for b in brand_catalogue]
        assert len(ids) == len(set(ids)), "Duplicate brand_id values found"

    def test_every_brand_has_negative_contexts(self, brand_catalogue):
        """Explicit check: no brand may have an empty negative_contexts list."""
        for brand in brand_catalogue:
            assert len(brand["negative_contexts"]) >= 1, (
                f"Brand '{brand['brand_id']}' has empty negative_contexts — "
                "this would silently defeat the hard-block safety filter"
            )


# ── (b) Malformed brand (missing negative_contexts) is REJECTED ──────────────
class TestMalformedBrandRejected:
    """A brand missing negative_contexts must fail schema validation."""

    def test_missing_negative_contexts_fails(self, brand_schema, brand_catalogue):
        validator = make_validator(brand_schema)
        # Take a valid brand and remove its negative_contexts
        malformed = copy.deepcopy(brand_catalogue[0])
        del malformed["negative_contexts"]
        errors = list(validator.iter_errors(malformed))
        assert len(errors) > 0, (
            "Schema MUST reject a brand with missing negative_contexts"
        )
        # Confirm the error actually mentions 'negative_contexts'
        error_messages = " ".join(e.message for e in errors)
        assert "negative_contexts" in error_messages, (
            f"Expected error about 'negative_contexts', got: {error_messages}"
        )

    def test_empty_negative_contexts_fails(self, brand_schema, brand_catalogue):
        validator = make_validator(brand_schema)
        # Take a valid brand and set negative_contexts to empty array
        malformed = copy.deepcopy(brand_catalogue[0])
        malformed["negative_contexts"] = []
        errors = list(validator.iter_errors(malformed))
        assert len(errors) > 0, (
            "Schema MUST reject a brand with empty negative_contexts (minItems: 1)"
        )

    def test_missing_brand_id_fails(self, brand_schema, brand_catalogue):
        validator = make_validator(brand_schema)
        malformed = copy.deepcopy(brand_catalogue[0])
        del malformed["brand_id"]
        errors = list(validator.iter_errors(malformed))
        assert len(errors) > 0, "Schema MUST reject a brand missing brand_id"

    def test_empty_target_contexts_fails(self, brand_schema):
        validator = make_validator(brand_schema)
        malformed = {
            "brand_id": "brand_test_bad",
            "display_name": "TestBad",
            "category": "test",
            "target_contexts": [],
            "negative_contexts": ["violence"],
            "creative_asset": "assets/ads/test.mp4",
            "duration_sec": 15,
            "tone": "neutral",
        }
        errors = list(validator.iter_errors(malformed))
        assert len(errors) > 0, (
            "Schema MUST reject a brand with empty target_contexts (minItems: 1)"
        )


# ── (c) pacing_rules.yaml has all four required keys with correct types ───────
class TestPacingRules:
    """pacing_rules.yaml must contain all required keys with correct types."""

    REQUIRED_KEYS = {
        "max_breaks_per_hour": int,
        "min_gap_seconds": int,
        "target_ad_load_pct": float,
        "first_break_not_before_sec": int,
    }

    def test_pacing_rules_file_exists(self):
        path = CONFIG_DIR / "pacing_rules.yaml"
        assert path.exists(), f"pacing_rules.yaml not found at {path}"

    def test_all_required_keys_present(self, pacing_rules):
        for key in self.REQUIRED_KEYS:
            assert key in pacing_rules, (
                f"Missing required key '{key}' in pacing_rules.yaml"
            )

    def test_correct_types(self, pacing_rules):
        for key, expected_type in self.REQUIRED_KEYS.items():
            value = pacing_rules[key]
            if expected_type is float:
                # YAML may parse 12.0 as float, but also accept int (12)
                assert isinstance(value, (int, float)), (
                    f"'{key}' should be numeric, got {type(value).__name__}: {value}"
                )
            else:
                assert isinstance(value, expected_type), (
                    f"'{key}' should be {expected_type.__name__}, "
                    f"got {type(value).__name__}: {value}"
                )

    def test_sensible_defaults(self, pacing_rules):
        """Sanity-check that defaults are in broadcast-reasonable ranges."""
        assert 1 <= pacing_rules["max_breaks_per_hour"] <= 12
        assert 60 <= pacing_rules["min_gap_seconds"] <= 1200
        assert 1.0 <= pacing_rules["target_ad_load_pct"] <= 25.0
        assert 60 <= pacing_rules["first_break_not_before_sec"] <= 900
