"""
Scene Understanding Module (Phase 4 prerequisite)
Populates activity_tags and dominant_activity on each scene using zero-shot CLIP
similarity between scene keyframes/text and a CONFIGURABLE label list from
config/activity_taxonomy.yaml.

NO hardcoded label lists in Python — all labels are data, loaded from YAML at runtime.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import yaml

logger = logging.getLogger("scene_understand")

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = REPO_ROOT / "config" / "activity_taxonomy.yaml"
MODEL_VERSIONS_PATH = REPO_ROOT / "config" / "model_versions.yaml"


def _load_taxonomy(taxonomy_path: Optional[str] = None) -> Dict[str, Any]:
    """Load activity label taxonomy from YAML. All label data lives here, never in code."""
    path = Path(taxonomy_path) if taxonomy_path else TAXONOMY_PATH
    if not path.exists():
        logger.warning("Taxonomy not found at %s, using empty label set", path)
        return {"activity_labels": [], "confidence_threshold": 0.10, "max_tags_per_scene": 8}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_model_versions(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else MODEL_VERSIONS_PATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class CLIPTextEmbedder:
    """
    Wraps open_clip's text encoder to produce L2-normalised embeddings
    for both image features (from a pre-encoded frame array) and text labels.
    Used for zero-shot classification via cosine similarity.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._init(pretrained)

    def _init(self, pretrained: str):
        import torch
        try:
            import open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=pretrained, device=self.device
            )
            self._model.eval()
            self._tokenize = open_clip.get_tokenizer(self.model_name)
            self._backend = "open_clip"
            logger.info("CLIPTextEmbedder: loaded %s (%s) via open_clip on %s",
                        self.model_name, pretrained, self.device)
        except Exception as e:
            logger.warning("open_clip unavailable (%s), will use fallback cosine hashing", e)
            self._backend = "fallback"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return (N, D) array of L2-normalised text embeddings."""
        if self._backend == "fallback":
            return self._fallback_embed(texts)
        import torch
        tokens = self._tokenize(texts).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    def embed_image(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return (1, D) L2-normalised image embedding from a BGR frame (numpy array)."""
        if self._backend == "fallback":
            return self._fallback_embed(["image"])
        import torch
        from PIL import Image
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().float().numpy()

    @staticmethod
    def _fallback_embed(texts: List[str]) -> np.ndarray:
        """Deterministic character-hash fallback for offline/test environments."""
        dim = 128
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            vec = rng.standard_normal(dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            out[i] = vec / (norm + 1e-9)
        return out

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between each row of a and each row of b. Returns (|a|, |b|) matrix."""
        a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return a_n @ b_n.T


def classify_scene_keyframe(
    frame_bgr: Optional[np.ndarray],
    asr_text: str,
    activity_labels: List[str],
    embedder: CLIPTextEmbedder,
    confidence_threshold: float = 0.10,
    max_tags: int = 8
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Zero-shot CLIP classification: returns ranked activity_tags and dominant_activity.

    Strategy:
      - If a video frame is available: use CLIP image embedding vs text label embeddings.
      - Also embed ASR text snippet and average with image embedding (text signal is important
        for dialogue-heavy content like Bengali drama).
      - Softmax-normalise raw cosine similarities to produce confidence scores in (0,1).

    Returns:
        (activity_tags, dominant_activity)
    """
    if not activity_labels:
        return [], ""

    label_embs = embedder.embed_texts(activity_labels)  # (N_labels, D)

    # Build query embedding: image (if available) + ASR text
    query_parts = []

    if frame_bgr is not None:
        try:
            img_emb = embedder.embed_image(frame_bgr)   # (1, D)
            query_parts.append(img_emb[0])
        except Exception as e:
            logger.debug("Image embedding failed: %s", e)

    if asr_text and len(asr_text.strip()) > 5:
        try:
            txt_emb = embedder.embed_texts([asr_text.strip()[:200]])  # (1, D)
            query_parts.append(txt_emb[0])
        except Exception as e:
            logger.debug("ASR text embedding failed: %s", e)

    if not query_parts:
        logger.warning("No query signal available for scene classification, using label averages")
        mean_label = np.mean(label_embs, axis=0)
        query_parts.append(mean_label)

    # Average all available query signals
    query = np.mean(np.stack(query_parts, axis=0), axis=0)
    query = query / (np.linalg.norm(query) + 1e-9)

    # Cosine similarities: shape (N_labels,)
    sims = (label_embs @ query).flatten()

    # Softmax to turn raw cosines into a probability distribution
    exp_sims = np.exp(sims - sims.max())
    confidences = exp_sims / (exp_sims.sum() + 1e-9)

    # Build activity_tags
    sorted_indices = np.argsort(-confidences)
    tags = []
    for idx in sorted_indices[:max_tags]:
        conf = float(confidences[idx])
        if conf >= confidence_threshold:
            tags.append({"label": activity_labels[idx], "confidence": round(conf, 4)})

    dominant = tags[0]["label"] if tags else ""
    return tags, dominant


def tag_scenes(
    scenes: List[Dict[str, Any]],
    video_path: Optional[str] = None,
    taxonomy_path: Optional[str] = None,
    config_path: Optional[str] = None,
    embedder: Optional[CLIPTextEmbedder] = None
) -> List[Dict[str, Any]]:
    """
    Populate activity_tags and dominant_activity on each scene dict.
    Mutates scenes in-place and returns them.
    """
    taxonomy = _load_taxonomy(taxonomy_path)
    activity_labels: List[str] = taxonomy.get("activity_labels", [])
    conf_thresh: float = float(taxonomy.get("confidence_threshold", 0.10))
    max_tags: int = int(taxonomy.get("max_tags_per_scene", 8))

    if not activity_labels:
        logger.error("No activity labels found in taxonomy. Check %s", taxonomy_path or TAXONOMY_PATH)
        return scenes

    mv_cfg = _load_model_versions(config_path)
    clip_cfg = mv_cfg.get("clip", {})
    model_name = clip_cfg.get("model_name", "ViT-B-32").replace("/", "-")  # open_clip uses hyphens

    if embedder is None:
        embedder = CLIPTextEmbedder(model_name=model_name, pretrained="openai")

    # Pre-embed all labels once
    logger.info("Pre-embedding %d activity labels...", len(activity_labels))
    embedder.embed_texts(activity_labels)  # warms up cache if needed

    for scene in scenes:
        scene_id = scene.get("scene_id", "?")

        # Extract midpoint frame from video if path provided
        frame_bgr = None
        if video_path and os.path.exists(video_path):
            try:
                import cv2
                mid_ts = (scene.get("start_ts", 0.0) + scene.get("end_ts", 0.0)) / 2.0
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_MSEC, mid_ts * 1000.0)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    frame_bgr = frame
            except Exception as e:
                logger.debug("Frame extraction failed for scene %s: %s", scene_id, e)

        asr_text = scene.get("asr_text", "")
        tags, dominant = classify_scene_keyframe(
            frame_bgr=frame_bgr,
            asr_text=asr_text,
            activity_labels=activity_labels,
            embedder=embedder,
            confidence_threshold=conf_thresh,
            max_tags=max_tags
        )

        scene["activity_tags"] = tags
        scene["dominant_activity"] = dominant
        logger.debug("Scene %s → dominant='%s', %d tags", scene_id, dominant, len(tags))

    logger.info("Tagged %d scenes with activity labels", len(scenes))
    return scenes
