"""
Tests for Phase 1 Scene Segmentation.
Verifies schema compliance, duration constraints, and contiguity.
"""

import json
from pathlib import Path
import pytest
import jsonschema
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "model_versions.yaml"
SCHEMA_PATH = REPO_ROOT / "schemas" / "scene.schema.json"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_processed_scenes():
    scene_files = list(PROCESSED_DIR.glob("*/scenes.json"))
    return scene_files


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_processed_scenes_exist():
    scene_files = get_all_processed_scenes()
    assert len(scene_files) >= 1, "At least one processed scenes.json must exist in data/processed"


def test_every_scene_validates_against_schema(schema):
    scene_files = get_all_processed_scenes()
    assert len(scene_files) > 0

    for s_file in scene_files:
        with open(s_file, "r", encoding="utf-8") as f:
            scenes = json.load(f)
        assert isinstance(scenes, list)
        assert len(scenes) > 0

        for sc in scenes:
            jsonschema.validate(instance=sc, schema=schema)


def test_scene_duration_constraints(config):
    seg_cfg = config.get("scene_segmentation", {})
    min_dur = float(seg_cfg.get("min_scene_duration_sec", 8.0))
    max_dur = float(seg_cfg.get("max_scene_duration_sec", 180.0))

    scene_files = get_all_processed_scenes()
    for s_file in scene_files:
        with open(s_file, "r", encoding="utf-8") as f:
            scenes = json.load(f)

        for sc in scenes:
            dur = sc["end_ts"] - sc["start_ts"]
            # Allow minor floating point tolerance of 0.05s
            if len(scenes) > 1:
                assert dur >= min_dur - 0.05, f"Scene {sc['scene_id']} duration {dur}s < min {min_dur}s in {s_file}"
            assert dur <= max_dur + 0.05, f"Scene {sc['scene_id']} duration {dur}s > max {max_dur}s in {s_file}"


def test_scenes_are_contiguous_and_non_overlapping():
    scene_files = get_all_processed_scenes()
    for s_file in scene_files:
        with open(s_file, "r", encoding="utf-8") as f:
            scenes = json.load(f)

        assert len(scenes) > 0
        # First scene starts at 0.0
        assert scenes[0]["start_ts"] == 0.0, f"First scene does not start at 0.0 in {s_file}"

        for i in range(len(scenes) - 1):
            curr_end = scenes[i]["end_ts"]
            next_start = scenes[i + 1]["start_ts"]
            # Adjacent scenes must share boundary with epsilon <= 0.05s
            assert abs(curr_end - next_start) <= 0.05, (
                f"Scenes not contiguous: {scenes[i]['scene_id']} ends at {curr_end}, "
                f"{scenes[i+1]['scene_id']} starts at {next_start} in {s_file}"
            )
