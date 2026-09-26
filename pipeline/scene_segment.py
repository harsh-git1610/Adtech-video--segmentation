"""
Scene Segmentation Module (Phase 1)
Transforms raw video and shot boundaries into semantically coherent scenes.
Merges consecutive shots based on visual CLIP embedding similarity and ASR continuity.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import yaml
import cv2
import jsonschema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scene_segment")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "model_versions.yaml"
SCENE_SCHEMA_PATH = REPO_ROOT / "schemas" / "scene.schema.json"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        logger.warning("Config path %s not found, using default parameters.", cfg_path)
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ClipEmbedder:
    """Extracts CLIP visual embeddings from video keyframes."""

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if self._cuda_available() else "cpu")
        self.model = None
        self.preprocess = None
        self.backend = None
        self._init_model()

    def _cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _init_model(self):
        logger.info("Initializing CLIP model (%s) on device: %s", self.model_name, self.device)
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained="openai", device=self.device
            )
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.backend = "open_clip"
            logger.info("Loaded CLIP via open_clip")
            return
        except Exception as e:
            logger.info("open_clip not available (%s)", e)

        try:
            import clip
            model, preprocess = clip.load(self.model_name, device=self.device)
            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.backend = "clip"
            logger.info("Loaded CLIP via openai-clip")
            return
        except Exception as e:
            logger.info("clip not available (%s)", e)

        try:
            import torch
            from transformers import CLIPProcessor, CLIPVisionModelWithProjection
            hf_id = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(hf_id)
            self.model = CLIPVisionModelWithProjection.from_pretrained(hf_id).to(self.device)
            self.model.eval()
            self.backend = "transformers"
            logger.info("Loaded CLIP via transformers (%s)", hf_id)
            return
        except Exception as e:
            logger.warning("Transformers CLIP failed (%s). Using color-histogram visual embedding fallback.", e)
            self.backend = "fallback"

    def encode_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.backend == "fallback":
            hist_b = cv2.calcHist([frame_bgr], [0], None, [170], [0, 256]).flatten()
            hist_g = cv2.calcHist([frame_bgr], [1], None, [171], [0, 256]).flatten()
            hist_r = cv2.calcHist([frame_bgr], [2], None, [171], [0, 256]).flatten()
            feat = np.concatenate([hist_b, hist_g, hist_r])
            norm = np.linalg.norm(feat)
            return (feat / (norm + 1e-9)).astype(np.float32)

        import torch
        from PIL import Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        with torch.no_grad():
            if self.backend in ("open_clip", "clip"):
                img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model.encode_image(img_tensor)
                features /= features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().flatten().astype(np.float32)
            elif self.backend == "transformers":
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                features = outputs.image_embeds
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features.cpu().numpy().flatten().astype(np.float32)


def get_video_info(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {"fps": float(fps), "duration": float(duration), "frame_count": float(frame_count)}


def extract_frame_at_timestamp(video_path: str, timestamp_sec: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_sec * 1000.0))
    success, frame = cap.read()
    cap.release()
    return frame if success else None


def transcribe_audio_whisper(
    video_path: str,
    model_size: str = "medium",
    language: Optional[str] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dev = device
    if not dev or dev == "auto":
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            dev = "cpu"

    comp = compute_type
    if not comp or comp == "default":
        comp = "float16" if dev == "cuda" else "int8"

    lang = language if (language and language != "auto") else None

    logger.info("Running faster-whisper (model=%s, device=%s, compute_type=%s, language=%s)",
                model_size, dev, comp, lang)

    segments_out = []
    words_out = []

    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device=dev, compute_type=comp)
        segments, info = model.transcribe(
            video_path,
            language=lang,
            word_timestamps=True,
            beam_size=5,
            vad_filter=True
        )

        for seg in segments:
            seg_text = seg.text.strip()
            segments_out.append({
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg_text
            })
            if seg.words:
                for w in seg.words:
                    words_out.append({
                        "word": w.word.strip(),
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3)
                    })

        logger.info("Whisper transcribed %d segments and %d words.", len(segments_out), len(words_out))
        return segments_out, words_out

    except Exception as e:
        logger.warning("faster-whisper transcription error: %s", e)
        return segments_out, words_out


def check_asr_topic_continuity(
    boundary_sec: float,
    words: List[Dict[str, Any]],
    pause_threshold: float = 2.0,
    scene_setting_phrases: Optional[List[str]] = None
) -> Tuple[bool, str]:
    """
    Topic continuity heuristic:
    1. Silence pause > 2.0s spanning boundary indicates natural scene / dialogue transition.
    2. Scene-setting transition phrases in following speech signify a new scene.
    """
    if not words:
        return True, "no_speech_context"

    pre_words = [w for w in words if w["end"] <= boundary_sec + 0.3]
    post_words = [w for w in words if w["start"] >= boundary_sec - 0.3]

    if pre_words and post_words:
        last_pre = pre_words[-1]
        first_post = post_words[0]
        silence_gap = first_post["start"] - last_pre["end"]
        if silence_gap > pause_threshold:
            return False, f"long_pause_{silence_gap:.2f}s"

    phrases = scene_setting_phrases or [
        "meanwhile", "later that day", "the next day", "next morning",
        "one year later", "few days later", "suddenly", "chapter", "episode"
    ]
    if post_words:
        next_chunk = " ".join([w["word"].lower() for w in post_words[:6]])
        for phrase in phrases:
            if phrase in next_chunk:
                return False, f"transition_phrase_{phrase}"

    return True, "continuous"


def segment_scenes(
    video_path: str,
    shot_boundaries: Optional[List[float]] = None,
    config_path: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = load_config(config_path)

    seg_cfg = cfg.get("scene_segmentation", {})
    whisper_cfg = cfg.get("whisper", {})
    clip_cfg = cfg.get("clip", {})

    sim_thresh = float(seg_cfg.get("shot_similarity_threshold", 0.82))
    min_dur = float(seg_cfg.get("min_scene_duration_sec", 8.0))
    max_dur = float(seg_cfg.get("max_scene_duration_sec", 180.0))
    pause_thresh = float(seg_cfg.get("max_asr_pause_threshold_sec", 2.0))
    phrases = seg_cfg.get("scene_setting_phrases", [])

    video_info = get_video_info(video_path)
    total_duration = video_info["duration"]

    if shot_boundaries is None:
        from pipeline.shot_detect import detect_shots
        shot_boundaries = detect_shots(video_path, config_path=config_path)

    sorted_boundaries = sorted(list(set([round(b, 3) for b in shot_boundaries if b > 0.0])))
    if not sorted_boundaries or sorted_boundaries[-1] < total_duration - 0.5:
        sorted_boundaries.append(round(total_duration, 3))

    shots = []
    prev_b = 0.0
    for b in sorted_boundaries:
        if b > prev_b + 0.05:
            shots.append((prev_b, b))
            prev_b = b

    logger.info("Total raw shots: %d (video duration: %.2fs)", len(shots), total_duration)

    whisper_model_size = whisper_cfg.get("model_size", "medium")
    whisper_lang = whisper_cfg.get("language", "auto")
    segments, words = transcribe_audio_whisper(
        video_path=video_path,
        model_size=whisper_model_size,
        language=whisper_lang,
        device=whisper_cfg.get("device"),
        compute_type=whisper_cfg.get("compute_type")
    )

    clip_embedder = ClipEmbedder(
        model_name=clip_cfg.get("model_name", "ViT-B/32"),
        device=whisper_cfg.get("device")
    )

    shot_embeddings = []
    for idx, (s_start, s_end) in enumerate(shots):
        midpoint = (s_start + s_end) / 2.0
        frame = extract_frame_at_timestamp(video_path, midpoint)
        if frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        emb = clip_embedder.encode_frame(frame)
        shot_embeddings.append(emb)

    scene_shot_groups: List[List[int]] = []
    current_group: List[int] = [0]

    for i in range(len(shots) - 1):
        emb_curr = shot_embeddings[i]
        emb_next = shot_embeddings[i + 1]

        norm_curr = np.linalg.norm(emb_curr)
        norm_next = np.linalg.norm(emb_next)
        if norm_curr > 0 and norm_next > 0:
            sim = float(np.dot(emb_curr, emb_next) / (norm_curr * norm_next))
        else:
            sim = 0.0

        boundary_ts = shots[i][1]
        topic_cont, topic_reason = check_asr_topic_continuity(
            boundary_sec=boundary_ts,
            words=words,
            pause_threshold=pause_thresh,
            scene_setting_phrases=phrases
        )

        merged_start = shots[current_group[0]][0]
        merged_end = shots[i + 1][1]
        would_exceed_max = (merged_end - merged_start) > max_dur

        can_merge = (sim >= sim_thresh) and topic_cont and (not would_exceed_max)

        if can_merge:
            current_group.append(i + 1)
        else:
            scene_shot_groups.append(current_group)
            current_group = [i + 1]

    if current_group:
        scene_shot_groups.append(current_group)

    # Enforce minimum scene duration by merging
    refined_groups: List[List[int]] = []
    idx = 0
    while idx < len(scene_shot_groups):
        grp = scene_shot_groups[idx]
        grp_start = shots[grp[0]][0]
        grp_end = shots[grp[-1]][1]
        dur = grp_end - grp_start

        if dur < min_dur:
            if refined_groups:
                prev_start = shots[refined_groups[-1][0]][0]
                if (grp_end - prev_start) <= max_dur:
                    refined_groups[-1].extend(grp)
                    idx += 1
                    continue
            if idx + 1 < len(scene_shot_groups):
                next_end = shots[scene_shot_groups[idx + 1][-1]][1]
                if (next_end - grp_start) <= max_dur:
                    scene_shot_groups[idx + 1] = grp + scene_shot_groups[idx + 1]
                    idx += 1
                    continue
        refined_groups.append(grp)
        idx += 1

    if not refined_groups and scene_shot_groups:
        refined_groups = scene_shot_groups

    final_scenes: List[Dict[str, Any]] = []

    with open(SCENE_SCHEMA_PATH, "r", encoding="utf-8") as sf:
        scene_schema = json.load(sf)

    for sc_idx, shot_indices in enumerate(refined_groups):
        sc_id = f"sc_{sc_idx + 1:03d}"
        sc_start = round(shots[shot_indices[0]][0], 3)
        sc_end = round(shots[shot_indices[-1]][1], 3)

        shot_ids = [f"sh_{j + 1:03d}" for j in shot_indices]

        overlapping_texts = []
        for seg in segments:
            if max(sc_start, seg["start"]) < min(sc_end, seg["end"]):
                if seg["text"]:
                    overlapping_texts.append(seg["text"])

        asr_text = " ".join(overlapping_texts).strip()

        embs = [shot_embeddings[j] for j in shot_indices]
        mean_emb = np.mean(embs, axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        emb_list = [round(float(v), 6) for v in mean_emb.tolist()]

        scene_obj = {
            "scene_id": sc_id,
            "start_ts": sc_start,
            "end_ts": sc_end,
            "shot_ids": shot_ids,
            "activity_tags": [],
            "dominant_activity": "",
            "asr_text": asr_text,
            "embedding": emb_list
        }

        jsonschema.validate(instance=scene_obj, schema=scene_schema)
        final_scenes.append(scene_obj)

    logger.info("Segmented video into %d validated scenes conforming to schema.", len(final_scenes))
    return final_scenes, words


def process_video_pipeline(
    video_path: str,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None
) -> Tuple[str, str]:
    v_path = Path(video_path).resolve()
    video_id = v_path.stem

    base_out = Path(output_dir) if output_dir else (REPO_ROOT / "data" / "processed")
    target_dir = base_out / video_id
    target_dir.mkdir(parents=True, exist_ok=True)

    scenes_file = target_dir / "scenes.json"
    words_file = target_dir / "whisper_words.json"

    logger.info("Processing video %s -> %s", v_path.name, target_dir)
    scenes, words = segment_scenes(str(v_path), config_path=config_path)

    with open(scenes_file, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d scenes to %s", len(scenes), scenes_file)

    with open(words_file, "w", encoding="utf-8") as f:
        json.dump(words, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %d word timestamps to %s", len(words), words_file)

    return str(scenes_file), str(words_file)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        vid = sys.argv[1]
        process_video_pipeline(vid)
    else:
        print("Usage: python scene_segment.py <video_path>")
