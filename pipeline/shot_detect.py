"""
Shot boundary detection module using PySceneDetect.
Extracts shot transition timestamps from video files.
"""

import os
import logging
from typing import List, Optional
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shot_detect")

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_versions.yaml")


def load_pyscenedetect_config(config_path: Optional[str] = None) -> dict:
    """Load PySceneDetect settings from model_versions.yaml."""
    cfg_file = config_path or DEFAULT_CONFIG_PATH
    if os.path.exists(cfg_file):
        with open(cfg_file, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
            return full_cfg.get("pyscenedetect", {})
    return {}


def detect_shots(
    video_path: str,
    detector_type: Optional[str] = None,
    threshold: Optional[float] = None,
    min_scene_len: Optional[int] = None,
    config_path: Optional[str] = None
) -> List[float]:
    """
    Detect raw shot boundary timestamps in seconds for a given video.

    Args:
        video_path: Path to the input video file (e.g., .mp4).
        detector_type: Detector name ('AdaptiveDetector' or 'ContentDetector').
        threshold: Sensitivity threshold.
        min_scene_len: Minimum shot length in frames.
        config_path: Path to YAML config file.

    Returns:
        List of boundary timestamps in seconds (strictly ascending, excluding 0.0, including final duration).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cfg = load_pyscenedetect_config(config_path)
    
    det_name = detector_type or cfg.get("detector", "AdaptiveDetector")
    thresh = threshold if threshold is not None else float(cfg.get("adaptive_threshold", 3.0))
    min_len = min_scene_len if min_scene_len is not None else int(cfg.get("min_scene_len_frames", 15))

    logger.info("Initializing PySceneDetect for %s", video_path)
    logger.info("Config used: detector=%s, threshold=%s, min_scene_len_frames=%s", det_name, thresh, min_len)

    from scenedetect import open_video, SceneManager

    video = open_video(video_path)
    scene_manager = SceneManager()

    detector_instance = None
    if det_name == "AdaptiveDetector":
        try:
            from scenedetect.detectors import AdaptiveDetector
            detector_instance = AdaptiveDetector(
                adaptive_threshold=thresh,
                min_scene_len=min_len
            )
            logger.info("Using AdaptiveDetector(adaptive_threshold=%f, min_scene_len=%d)", thresh, min_len)
        except (ImportError, AttributeError) as e:
            logger.warning("AdaptiveDetector unavailable (%s), falling back to ContentDetector", e)
            det_name = "ContentDetector"

    if det_name == "ContentDetector" or detector_instance is None:
        from scenedetect.detectors import ContentDetector
        content_thresh = thresh if thresh > 10.0 else 27.0
        detector_instance = ContentDetector(
            threshold=content_thresh,
            min_scene_len=min_len
        )
        logger.info("Using ContentDetector(threshold=%f, min_scene_len=%d)", content_thresh, min_len)

    scene_manager.add_detector(detector_instance)

    logger.info("Detecting scenes across video frames...")
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    boundaries = []
    for start_time, end_time in scene_list:
        end_sec = round(end_time.get_seconds(), 3)
        if end_sec not in boundaries:
            boundaries.append(end_sec)

    if not boundaries:
        try:
            total_duration = round(video.duration.get_seconds(), 3)
            boundaries.append(total_duration)
        except Exception:
            pass

    logger.info("Detected %d shot boundaries in %s: %s", len(boundaries), video_path, boundaries[:10])
    return boundaries


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        v_path = sys.argv[1]
        shots = detect_shots(v_path)
        print(f"Detected {len(shots)} boundaries: {shots}")
    else:
        print("Usage: python shot_detect.py <video_path>")
