"""
our_method.py

Query-adaptive ASYMMETRIC paneling.

Difference from video_panels_baseline.py (Video Panels):
  - Video Panels: uniform sampling, query-blind, equal-size 2x2 cells
  - This:         1fps dense sampling -> CLIP scoring -> threshold filter
                  -> windowed selection of exactly 4K frames
                  -> 2x2 cross-split where cell AREA tracks CLIP score

GEOMETRY NOTE (important, and a real limitation):
A clean 2x2 grid has one vertical split (alpha) and one horizontal split
(beta) - two degrees of freedom controlling four areas. So areas are
forced to be separable:
    TL = alpha*beta      TR = (1-alpha)*beta
    BL = alpha*(1-beta)  BR = (1-alpha)*(1-beta)
Setting alpha = w_TL + w_BL (left-column mass) and beta = w_TL + w_TR
(top-row mass) matches target areas exactly when the weights are
separable, e.g. [0.45,0.25,0.20,0.10] -> max error 0.005. It degrades on
anti-diagonal patterns, e.g. [0.4,0.1,0.1,0.4] collapses to a uniform
grid (error 0.15). This is inherent to a 2x2 cross-split, not a bug -
escaping it needs a treemap layout, which breaks the fixed TL/TR/BL/BR
reading order.

Interface matches evaluate_panels_mlx.py's --method ours hook:
    from our_method import OurMethodConfig, our_method_pipeline, ClipScorer
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


# ---- Config --------------------------------------------------------

@dataclass
class OurMethodConfig:
    # Final budget: number of composite canvases sent to the model.
    # Each canvas holds 4 frames, so 4*C frames are selected in total.
    C: int = 16

    # Stage 1: dense candidate sampling
    sample_rate_fps: float = 1.0
    decode_width: int = 640      # downscale during decode to bound RAM
    decode_height: int = 360
    max_candidates: int = 1024   # hard cap on frames sent to CLIP

    # Stage 2: relevance scoring + filtering
    # Threshold applies to RAW cosine similarity, NOT min-max normalized
    # scores. Raw CLIP text-image similarity typically lands in
    # 0.15-0.35, so 0.15 is a meaningful absolute floor. Normalizing
    # first would make this threshold relative to each video's own score
    # spread, which is not what "discard irrelevant frames" means.
    raw_score_threshold: float = 0.15
    temporal_sigma: float = 0.0  # >0 smooths scores over time. Default OFF:
                                  # smoothing makes neighbouring frames more
                                  # similar, which flattens the within-canvas
                                  # score spread the asymmetry depends on.

    # Stage 3: selection strategy for picking exactly 4*C frames
    #   "windowed" - split timeline into 4*C windows, take best per window.
    #                Guarantees temporal coverage; one high-scoring scene
    #                cannot consume the whole budget.
    #   "topk"     - plain greedy top-4C by score. For ablation.
    selection_strategy: str = "windowed"

    # Stage 4: canvas layout
    canvas_w: int = 1024
    canvas_h: int = 576
    border_px: int = 3
    border_color: Tuple[int, int, int] = (0, 0, 0)
    # Clamp the splits so no cell degenerates into an unreadable sliver.
    # 0.25/0.75 gives at most a 9x area ratio between largest and smallest.
    min_split: float = 0.25
    max_split: float = 0.75

    # Raw CLIP cosine similarities occupy a narrow band (~0.14-0.34), so
    # normalizing them directly yields weights that all sit near 0.25 and
    # the asymmetry is barely visible. Softmax with a low temperature
    # spreads them out. Lower temperature = more extreme size differences.
    #   "linear"  - normalize raw scores directly (weak asymmetry)
    #   "softmax" - softmax(score / temperature), much stronger contrast
    score_sharpening: str = "softmax"
    softmax_temperature: float = 0.02

    # Debug overlays - MUST stay False for real evaluation runs.
    # Burning "score=0.42" into a frame feeds the model information about
    # your own selection process that has nothing to do with the video.
    draw_score_banner: bool = False
    draw_timestamp: bool = False   # timestamps are defensible (Video Panels
                                    # tested frame-time overlays); scores are not

    # CLIP
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_device: Optional[str] = None   # None -> auto (mps > cuda > cpu)


# ---- CLIP scorer ---------------------------------------------------

def _pick_device() -> str:
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClipScorer:
    """Loaded once, reused across every question - same singleton pattern
    as the MLX VLM model in evaluate_panels_mlx.py.

    Uses transformers' CLIP rather than open_clip so identical code runs
    on Kaggle (CUDA) and on Apple Silicon (MPS).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or _pick_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        print(f"[CLIP] {model_name} on {self.device}")

    @staticmethod
    def _as_tensor(out):
        """HF output types vary by version - pull the embedding tensor out."""
        import torch
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None:
                return v
        v = getattr(out, "last_hidden_state", None)
        if v is not None:
            return v[:, 0]
        return out[0]

    def score(self, query: str, frames: List[np.ndarray],
              batch_size: int = 32) -> np.ndarray:
        """RAW cosine similarity between the query and each frame.

        Returns raw values (typically ~0.15-0.35), NOT normalized - the
        threshold in OurMethodConfig is absolute and needs the true scale.
        """
        torch = self.torch
        with torch.no_grad():
            embeds = []
            for i in range(0, len(frames), batch_size):
                batch = [Image.fromarray(f) for f in frames[i:i + batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                f = self._as_tensor(self.model.get_image_features(**inputs))
                f = f / f.norm(dim=-1, keepdim=True)
                embeds.append(f.cpu())
            F = torch.cat(embeds, dim=0)

            t_in = self.processor(text=[query], return_tensors="pt",
                                   padding=True, truncation=True).to(self.device)
            Q = self._as_tensor(self.model.get_text_features(**t_in))
            Q = Q / Q.norm(dim=-1, keepdim=True)
            Q = Q.cpu()

            return (F @ Q.T).squeeze(-1).numpy()


# ---- Stage 1: frame extraction -------------------------------------

def extract_frames(video_path: str, config: OurMethodConfig):
    """1fps sampling via cv2 (decord is unreliable on Apple Silicon).
    Returns (frames as RGB uint8 arrays, timestamps in seconds)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    step = max(1, int(round(fps / config.sample_rate_fps)))
    indices = list(range(0, total, step))
    if len(indices) > config.max_candidates:
        keep = np.linspace(0, len(indices) - 1, config.max_candidates, dtype=int)
        indices = [indices[k] for k in keep]

    frames, timestamps = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if config.decode_width and config.decode_height:
            rgb = cv2.resize(rgb, (config.decode_width, config.decode_height),
                              interpolation=cv2.INTER_AREA)
        frames.append(rgb)
        timestamps.append(idx / fps)
    cap.release()
    return frames, timestamps


# ---- Stage 3: selection --------------------------------------------

def select_windowed(items, k: int):
    """Split the timeline into k equal windows, keep the best-scoring
    frame in each. Guarantees temporal coverage - a single high-scoring
    scene cannot consume the entire budget the way greedy top-k allows.

    items: list of (frame, timestamp, score)
    """
    if not items:
        return []
    k = min(k, len(items))
    ts = [it[1] for it in items]
    lo, hi = min(ts), max(ts)
    span = max(hi - lo, 1e-9)
    width = span / k

    buckets = [[] for _ in range(k)]
    for it in items:
        b = min(int((it[1] - lo) / width), k - 1)
        buckets[b].append(it)

    picked = [max(b, key=lambda x: x[2]) for b in buckets if b]

    # Some windows may be empty (frames aren't evenly distributed after
    # threshold filtering). Backfill from the best unpicked frames so we
    # still return exactly k, otherwise the last canvas comes out short.
    if len(picked) < k:
        chosen_ids = {id(p) for p in picked}
        rest = sorted((it for it in items if id(it) not in chosen_ids),
                       key=lambda x: -x[2])
        picked.extend(rest[:k - len(picked)])

    picked.sort(key=lambda x: x[1])
    return picked


def select_topk(items, k: int):
    """Plain greedy top-k by score, chronologically re-sorted. Ablation."""
    out = sorted(items, key=lambda x: -x[2])[:k]
    out.sort(key=lambda x: x[1])
    return out


# ---- Stage 4: asymmetric canvas ------------------------------------

def render_adaptive_canvas(chunk4, config: OurMethodConfig):
    """Render 4 frames into a 2x2 cross-split canvas.

    Reading order is fixed and chronological:
        TL = earliest, TR, BL, BR = latest
    Cell area tracks CLIP score via the marginal split described in the
    module docstring.

    chunk4: list of exactly 4 (frame, timestamp, score), any order -
            this function sorts them chronologically itself.
    """
    assert len(chunk4) == 4, f"expected 4 frames, got {len(chunk4)}"
    ordered = sorted(chunk4, key=lambda x: float(x[1]))
    scores = np.array([float(it[2]) for it in ordered], dtype=np.float64)

    if config.score_sharpening == "softmax":
        # Subtract the max before exponentiating (standard numerical
        # stability trick - shifts the exponent range without changing
        # the result). Low temperature widens the gaps between weights.
        z = (scores - scores.max()) / max(config.softmax_temperature, 1e-6)
        e = np.exp(z)
        w = e / e.sum()
    else:
        # Shift to non-negative before normalizing: raw cosine similarity
        # can be negative, and a negative weight would invert the split.
        shifted = scores - min(scores.min(), 0.0)
        total = shifted.sum()
        w = shifted / total if total > 1e-9 else np.full(4, 0.25)

    # Marginal split - see module docstring for why this is the correct
    # (and only) way to map 4 target areas onto a 2x2 cross-split.
    alpha = float(np.clip(w[0] + w[2], config.min_split, config.max_split))  # left col
    beta = float(np.clip(w[0] + w[1], config.min_split, config.max_split))   # top row

    b = config.border_px
    avail_w = config.canvas_w - 3 * b
    avail_h = config.canvas_h - 3 * b
    w_left = int(round(alpha * avail_w))
    w_right = avail_w - w_left
    h_top = int(round(beta * avail_h))
    h_bot = avail_h - h_top

    slots = [
        (b, b, w_left, h_top),                           # TL - earliest
        (2 * b + w_left, b, w_right, h_top),             # TR
        (b, 2 * b + h_top, w_left, h_bot),               # BL
        (2 * b + w_left, 2 * b + h_top, w_right, h_bot)  # BR - latest
    ]

    canvas = np.full((config.canvas_h, config.canvas_w, 3),
                      config.border_color, dtype=np.uint8)

    for (frame, ts, score), (x, y, cw, ch) in zip(ordered, slots):
        if cw <= 0 or ch <= 0:
            continue
        cell = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)

        if config.draw_score_banner or config.draw_timestamp:
            bh = max(20, int(ch * 0.08))
            cv2.rectangle(cell, (0, 0), (cw, bh), (0, 0, 0), -1)
            label = (f"t={ts:.1f}s score={score:.3f}"
                     if config.draw_score_banner else f"t={ts:.1f}s")
            cv2.putText(cell, label, (6, int(bh * 0.72)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.38, bh / 42.0),
                        (255, 255, 0), 1, cv2.LINE_AA)

        canvas[y:y + ch, x:x + cw] = cell

    achieved = np.array([w_left * h_top, w_right * h_top,
                          w_left * h_bot, w_right * h_bot], dtype=np.float64)
    achieved /= achieved.sum()

    meta = {
        "alpha": round(alpha, 3),
        "beta": round(beta, 3),
        "timestamps": [round(float(it[1]), 1) for it in ordered],
        "scores": [round(float(it[2]), 4) for it in ordered],
        "target_area": [round(float(x), 3) for x in w],
        "achieved_area": [round(float(x), 3) for x in achieved],
        # How far the separable cross-split had to compromise. Large values
        # mean this chunk's scores were anti-diagonal and the layout could
        # not represent them - see module docstring.
        "layout_error": round(float(np.abs(achieved - w).max()), 3),
        # Score spread within this canvas. Near zero means all four frames
        # scored alike, so the asymmetry does nothing here.
        "score_spread": round(float(scores.max() - scores.min()), 4),
    }
    return Image.fromarray(canvas), meta


# ---- Main entry point ----------------------------------------------

def our_method_pipeline(video_path: str, query: str,
                         scorer: ClipScorer,
                         config: OurMethodConfig,
                         return_meta: bool = False):
    """video_path + query -> list of C composite PIL Images.

    Signature matches evaluate_panels_mlx.py's --method ours hook.

    Pipeline:
      1. 1fps sampling
      2. CLIP scoring (raw cosine similarity)
      3. drop frames below raw_score_threshold
      4. select exactly 4*C frames (windowed, for temporal coverage)
      5. chronological chunks of 4 -> asymmetric 2x2 canvases
    """
    need = 4 * config.C

    frames, timestamps = extract_frames(video_path, config)
    if not frames:
        print("    [ours] no frames extracted")
        return ([], []) if return_meta else []

    scores = scorer.score(query, frames)

    if config.temporal_sigma > 0 and len(scores) > 3:
        from scipy.ndimage import gaussian_filter1d
        scores = gaussian_filter1d(scores, sigma=config.temporal_sigma)

    items = list(zip(frames, timestamps, scores.astype(float)))

    kept = [it for it in items if it[2] >= config.raw_score_threshold]
    n_dropped = len(items) - len(kept)

    # If thresholding left too few frames to fill C canvases, fall back to
    # the full pool. Better to include weak frames than to return fewer
    # canvases than the budget the model expects.
    if len(kept) < need:
        print(f"    [ours] threshold kept only {len(kept)}/{len(items)}, "
              f"need {need} - falling back to full pool")
        kept = items

    chosen = (select_topk(kept, need) if config.selection_strategy == "topk"
              else select_windowed(kept, need))

    print(f"    [ours] {len(frames)} candidates @ {config.sample_rate_fps}fps  "
          f"raw score min={scores.min():.3f} max={scores.max():.3f}  "
          f"dropped={n_dropped}  selected={len(chosen)}")

    canvases, metas = [], []
    for i in range(0, len(chosen) - 3, 4):
        canvas, meta = render_adaptive_canvas(chosen[i:i + 4], config)
        canvases.append(canvas)
        metas.append(meta)

    if metas:
        spreads = [m["score_spread"] for m in metas]
        errs = [m["layout_error"] for m in metas]
        print(f"    [ours] {len(canvases)} canvases  "
              f"mean within-canvas score spread={np.mean(spreads):.4f}  "
              f"mean layout error={np.mean(errs):.3f}")
        if np.mean(spreads) < 0.01:
            print("    [ours] WARNING: score spread is tiny - canvases are "
                  "effectively uniform, asymmetry is doing nothing")

    return (canvases, metas) if return_meta else canvases
