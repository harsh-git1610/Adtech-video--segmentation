"""
Brand Matching Engine (Phase 4)
Matches each approved break candidate to the highest-scoring, non-blocked brand
from data/brand_catalogue.json.

DESIGN GUARANTEE: This file contains ZERO literal brand names, brand IDs, or
brand categories as string literals in control flow. All brand-specific behavior
flows through data loaded from brand_catalogue.json at runtime.

Architecture:
  1. Embed all brand contexts (target + negative) once at startup.
  2. For each break candidate, retrieve adjacent scene activity_tags.
  3. Run HARD BLOCK check: is_blocked(brand, scene_tags) — pure data comparison,
     no brand-specific code paths.
  4. Among surviving brands, compute ranking score and pick top brand.
  5. Store full ranked list + block reasons in the returned match_result dict.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import yaml

logger = logging.getLogger("brand_matcher")

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOGUE_PATH = REPO_ROOT / "data" / "brand_catalogue.json"
MODEL_VERSIONS_PATH = REPO_ROOT / "config" / "model_versions.yaml"

# Hard-block cosine-similarity threshold.
# Tune: lower = more aggressive blocking (fewer false negatives on synonyms).
DEFAULT_BLOCK_THRESHOLD = 0.62

# Minimum activity_tag confidence to include in hard-block check.
DEFAULT_BLOCK_CONF_FLOOR = 0.35

# Scoring weights
DOMINANT_WEIGHT = 0.70
SECONDARY_WEIGHT = 0.30


def load_brand_catalogue(catalogue_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load brand catalogue from JSON. All brand data is pure data — no code specialisation."""
    path = Path(catalogue_path) if catalogue_path else CATALOGUE_PATH
    if not path.exists():
        raise FileNotFoundError(f"Brand catalogue not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        brands = json.load(f)
    logger.info("Loaded %d brands from %s", len(brands), path)
    return brands


def _load_model_versions() -> Dict[str, Any]:
    if not MODEL_VERSIONS_PATH.exists():
        return {}
    with open(MODEL_VERSIONS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class TextEmbedder:
    """
    Semantic text embedder for brand context matching.
    Primary backend: sentence-transformers (all-mpnet-base-v2 or all-MiniLM-L6-v2).
    Fallback backend: open_clip text encoder with anisotropy de-biasing.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache: Dict[str, np.ndarray] = {}
        self._backend = "fallback"

        # 1. Try sentence-transformers first (preferred for text-to-text semantic similarity)
        try:
            from sentence_transformers import SentenceTransformer
            st_name = model_name or "all-MiniLM-L6-v2"
            self._st_model = SentenceTransformer(st_name, device=self.device)
            self._backend = "sentence_transformers"
            logger.info("TextEmbedder: loaded %s via sentence-transformers on %s", st_name, self.device)
            return
        except Exception as e:
            logger.debug("sentence-transformers unavailable (%s), trying open_clip fallback", e)

        # 2. Try open_clip text encoder with mean-centering
        try:
            import open_clip
            clip_model = model_name or "ViT-B-32"
            self._model, _, _ = open_clip.create_model_and_transforms(
                clip_model, pretrained="openai", device=self.device
            )
            self._model.eval()
            self._tokenize = open_clip.get_tokenizer(clip_model)
            self._backend = "open_clip_centered"
            # Compute a reference baseline center vector to subtract anisotropy bias
            baseline_prompts = [
                "a photo of something", "an object", "a scene", "a person", "an activity",
                "a concept", "outdoor view", "indoor room", "text description", "general context"
            ]
            tokens = self._tokenize(baseline_prompts).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self._clip_center = feats.mean(dim=0, keepdim=True)
                self._clip_center = (self._clip_center / self._clip_center.norm(dim=-1, keepdim=True)).cpu().numpy()
            logger.info("TextEmbedder: loaded %s via open_clip (anisotropy-centered) on %s", clip_model, self.device)
            return
        except Exception as e:
            logger.warning("open_clip unavailable (%s), using deterministic semantic hash", e)
            self._backend = "fallback"

    def embed(self, phrases: List[str]) -> np.ndarray:
        """
        Return (N, D) matrix of L2-normalised embeddings.
        Results are cached by exact phrase string.
        """
        uncached = [p for p in phrases if p not in self._cache]
        if uncached:
            new_embs = self._embed_batch(uncached)
            for p, emb in zip(uncached, new_embs):
                self._cache[p] = emb

        result = np.stack([self._cache[p] for p in phrases], axis=0)
        return result

    def _embed_batch(self, phrases: List[str]) -> List[np.ndarray]:
        if self._backend == "sentence_transformers":
            embs = self._st_model.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
            return [embs[i] for i in range(len(phrases))]

        if self._backend == "open_clip_centered":
            import torch
            tokens = self._tokenize(phrases).to(self.device)
            with torch.no_grad():
                feats = self._model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                feats_np = feats.cpu().float().numpy()
                # Subtract anisotropy center vector to expand dynamic range across topics
                centered = feats_np - 0.82 * self._clip_center
                norms = np.linalg.norm(centered, axis=-1, keepdims=True) + 1e-9
                normalized = centered / norms
            return [normalized[i] for i in range(len(phrases))]

        return [self._hash_embed(p) for p in phrases]

    @staticmethod
    def _hash_embed(phrase: str, dim: int = 128) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(phrase)) % (2**31))
        vec = rng.standard_normal(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Scalar cosine similarity between two unit vectors."""
        a_n = a / (np.linalg.norm(a) + 1e-9)
        b_n = b / (np.linalg.norm(b) + 1e-9)
        return float(np.clip(a_n @ b_n, -1.0, 1.0))

    def max_cosine(self, query_emb: np.ndarray, phrase_list: List[str]) -> float:
        """Max cosine similarity between query and any phrase in phrase_list."""
        if not phrase_list:
            return 0.0
        embs = self.embed(phrase_list)  # (N, D)
        q_n = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        sims = embs @ q_n  # (N,)
        return float(np.max(sims))

    def mean_max_cosine(self, query_phrases: List[str], target_phrases: List[str]) -> float:
        """Mean over query_phrases of max cosine similarity against target_phrases."""
        if not query_phrases or not target_phrases:
            return 0.0
        scores = []
        for qp in query_phrases:
            q_emb = self.embed([qp])[0]
            scores.append(self.max_cosine(q_emb, target_phrases))
        return float(np.mean(scores))


class BrandMatcher:
    """
    Stateful brand matcher. Pre-embeds all brand context phrases once at init,
    then applies hard-block + ranking on each break candidate.
    """

    def __init__(
        self,
        catalogue_path: Optional[str] = None,
        block_threshold: float = DEFAULT_BLOCK_THRESHOLD,
        block_conf_floor: float = DEFAULT_BLOCK_CONF_FLOOR,
        embedder: Optional[TextEmbedder] = None
    ):
        self.brands = load_brand_catalogue(catalogue_path)
        self.block_threshold = block_threshold
        self.block_conf_floor = block_conf_floor

        mv = _load_model_versions()
        st_cfg = mv.get("sentence_transformer", {})
        model_name = st_cfg.get("model_name", "all-MiniLM-L6-v2")

        self.embedder = embedder or TextEmbedder(model_name=model_name)

        # Pre-embed all brand context phrases
        self._pre_embed_brands()

    def _pre_embed_brands(self):
        """Warm up embedding cache for all brand context strings."""
        all_phrases = []
        for brand in self.brands:
            all_phrases.extend(brand.get("target_contexts", []))
            all_phrases.extend(brand.get("negative_contexts", []))
        if all_phrases:
            self.embedder.embed(list(set(all_phrases)))
            logger.info("Pre-embedded %d unique brand context phrases", len(set(all_phrases)))

    def is_blocked(self, brand: Dict[str, Any], activity_tags: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        HARD BLOCK: returns (True, reason) if ANY activity_tag label (above conf floor)
        is semantically close to ANY of the brand's negative_contexts.

        This is a boolean filter — NOT a penalty in a scoring function.
        It catches synonyms: "cremation" triggers against "funeral" negative context.
        """
        negative_contexts: List[str] = brand.get("negative_contexts", [])
        if not negative_contexts:
            return False, ""

        candidate_tags = [
            t for t in activity_tags
            if float(t.get("confidence", 0.0)) >= self.block_conf_floor
        ]
        if not candidate_tags:
            return False, ""

        neg_embs = self.embedder.embed(negative_contexts)  # (N_neg, D)

        for tag in candidate_tags:
            tag_label = str(tag.get("label", ""))
            if not tag_label:
                continue
            tag_emb = self.embedder.embed([tag_label])[0]  # (D,)

            # Compare tag embedding against ALL negative context embeddings
            sims = neg_embs @ tag_emb  # (N_neg,)
            max_sim = float(np.max(sims))

            if max_sim >= self.block_threshold:
                best_neg_idx = int(np.argmax(sims))
                best_neg = negative_contexts[best_neg_idx]
                reason = (
                    f"tag '{tag_label}' (conf={tag['confidence']:.2f}) "
                    f"semantically matches negative_context '{best_neg}' "
                    f"(cosine={max_sim:.3f} >= threshold={self.block_threshold})"
                )
                return True, reason

        return False, ""

    def score_brand(
        self,
        brand: Dict[str, Any],
        dominant_activity: str,
        other_tag_labels: List[str]
    ) -> float:
        """
        score = 0.70 * max_cosine(dominant_activity, target_contexts)
              + 0.30 * mean_max_cosine(other_tag_labels, target_contexts)
        """
        target_contexts: List[str] = brand.get("target_contexts", [])
        if not target_contexts:
            return 0.0

        dom_emb = self.embedder.embed([dominant_activity])[0] if dominant_activity else None
        dom_score = 0.0
        if dom_emb is not None:
            dom_score = self.max_cosine_raw(dom_emb, target_contexts)

        sec_score = 0.0
        if other_tag_labels:
            sec_score = self.embedder.mean_max_cosine(other_tag_labels, target_contexts)

        return round(DOMINANT_WEIGHT * dom_score + SECONDARY_WEIGHT * sec_score, 4)

    def max_cosine_raw(self, query_emb: np.ndarray, phrase_list: List[str]) -> float:
        if not phrase_list:
            return 0.0
        embs = self.embedder.embed(phrase_list)
        q_n = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        return float(np.max(embs @ q_n))

    def match(
        self,
        candidate: Dict[str, Any],
        scene_before: Optional[Dict[str, Any]],
        scene_after: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run full brand matching pipeline for one approved break candidate.

        Returns a match_result dict with:
          - top_brand: brand_id of the winning brand (or None)
          - ranked_brands: full list with scores, sorted descending
          - blocked_brands: brands excluded by hard block with reasons
          - debug: detailed sub-signal breakdown
        """
        # Merge activity signals from both adjacent scenes
        combined_tags: List[Dict[str, Any]] = []
        dominant_activity = ""

        for scene in [scene_before, scene_after]:
            if scene is None:
                continue
            tags = scene.get("activity_tags", [])
            combined_tags.extend(tags)
            if not dominant_activity and scene.get("dominant_activity"):
                dominant_activity = scene["dominant_activity"]

        # Deduplicate tags (use highest confidence if duplicate label)
        seen: Dict[str, float] = {}
        for t in combined_tags:
            lbl = t.get("label", "")
            conf = float(t.get("confidence", 0.0))
            if lbl and conf > seen.get(lbl, -1.0):
                seen[lbl] = conf
        deduped_tags: List[Dict[str, Any]] = [{"label": lbl, "confidence": c} for lbl, c in seen.items()]
        deduped_tags.sort(key=lambda x: -x["confidence"])

        other_tag_labels = [t["label"] for t in deduped_tags if t["label"] != dominant_activity][:6]

        ranked_brands = []
        blocked_brands = []

        for brand in self.brands:
            blocked, block_reason = self.is_blocked(brand, deduped_tags)
            if blocked:
                blocked_brands.append({
                    "brand_id": brand["brand_id"],
                    "display_name": brand.get("display_name", ""),
                    "block_reason": block_reason
                })
                continue

            score = self.score_brand(brand, dominant_activity, other_tag_labels)
            ranked_brands.append({
                "brand_id": brand["brand_id"],
                "display_name": brand.get("display_name", ""),
                "score": score,
                "dominant_signal": round(
                    self.max_cosine_raw(
                        self.embedder.embed([dominant_activity])[0] if dominant_activity else np.zeros(128),
                        brand.get("target_contexts", [])
                    ), 4
                )
            })

        ranked_brands.sort(key=lambda x: -x["score"])
        top_brand = ranked_brands[0]["brand_id"] if ranked_brands else None

        return {
            "candidate_id": candidate.get("candidate_id", ""),
            "timestamp": candidate.get("timestamp", 0.0),
            "dominant_activity": dominant_activity,
            "activity_tags_used": deduped_tags,
            "top_brand": top_brand,
            "ranked_brands": ranked_brands,
            "blocked_brands": blocked_brands,
            "n_surviving": len(ranked_brands),
            "n_blocked": len(blocked_brands)
        }

    def match_all(
        self,
        approved_candidates: List[Dict[str, Any]],
        scenes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Run match() for every approved break candidate.
        Looks up adjacent scene objects by scene_before / scene_after IDs.
        """
        scene_map: Dict[str, Dict[str, Any]] = {s["scene_id"]: s for s in scenes}
        results = []
        for candidate in approved_candidates:
            if candidate.get("final_decision") != "BREAK_APPROVED":
                continue
            sb = scene_map.get(candidate.get("scene_before", ""))
            sa = scene_map.get(candidate.get("scene_after", ""))
            result = self.match(candidate, scene_before=sb, scene_after=sa)
            results.append(result)
        logger.info(
            "Brand matching complete: %d candidates → %d matched, %d total blocks",
            len(results),
            sum(1 for r in results if r["top_brand"]),
            sum(r["n_blocked"] for r in results)
        )
        return results
