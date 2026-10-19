"""
video_panels_baseline.py

Reference reimplementation of Video Panels' method: uniform sampling,
query-blind, equal-size 2x2 cells. This is the BASELINE.

Our method (CLIP-scored selection, asymmetric cells) lives in
our_method.py.

gamma (fps_limit) is exposed so you can sweep the paper's Table 3
ablation values: 0, 0.5x, 1x, 2x, 3x fps.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image


@dataclass
class PanelConfig:
    """Config for the Video Panels baseline reimplementation."""
    C: int                    # frame budget - panels sent to the model
    panel_width: int = 2      # alpha
    panel_height: int = 2     # beta
    fps_limit: float = 1.0    # gamma - spacing constraint, in units of "x fps"
                              # (0 = no constraint, matches paper's gamma=0 row)
    border_px: int = 0        # separator between cells. Paper's own code has
                              # this ("Suggested value: 0") but a visible
                              # border may help the model read the composite
                              # as 4 distinct frames. Worth ablating.
    border_color: tuple = (0, 0, 0)
    # Output canvas size. None = keep the source frame's own resolution,
    # which is what the paper does. Set these to match OurMethodConfig's
    # canvas_w/canvas_h when comparing methods - otherwise the two
    # conditions send different numbers of visual tokens per panel and
    # the comparison measures resolution, not just frame selection.
    canvas_w: Optional[int] = None
    canvas_h: Optional[int] = None


def get_video_info(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def stack_frames_grid(frames, panel_width: int, panel_height: int,
                      border_px: int = 0, border_color: tuple = (0, 0, 0),
                      canvas_w=None, canvas_h=None):
    """Resize each frame into an equal-size cell and paste into a uniform
    grid. No relevance scores anywhere - every cell is the same size.

    border_px shrinks the cells to make room; the canvas stays the same
    overall size as the source frame either way.
    """
    w, h = panel_width, panel_height
    frames_per_grid = w * h
    # Default to the source frame's own size (paper behaviour); override
    # to match the other method's canvas when running a comparison.
    H = canvas_h if canvas_h else frames[0].shape[0]
    W = canvas_w if canvas_w else frames[0].shape[1]

    # borders sit between cells AND around the outer edge:
    # w+1 vertical lines for w columns, h+1 horizontal for h rows
    total_border_w = (w + 1) * border_px
    total_border_h = (h + 1) * border_px
    panel_W = (W - total_border_w) // w
    panel_H = (H - total_border_h) // h

    if panel_W <= 0 or panel_H <= 0:
        raise ValueError(
            f"border_px={border_px} too large for a {W}x{H} frame with a "
            f"{w}x{h} grid - no room left for the cells themselves."
        )

    n_panels = len(frames) // frames_per_grid
    panels = []
    for i in range(n_panels):
        chunk = frames[i * frames_per_grid: (i + 1) * frames_per_grid]
        # fill with border colour first, then paste cells on top -
        # whatever gaps remain become the borders
        canvas = np.full((H, W, 3), border_color, dtype=np.uint8)
        for idx, frame in enumerate(chunk):
            row, col = idx // w, idx % w
            resized = np.asarray(
                Image.fromarray(frame).resize((panel_W, panel_H), Image.LANCZOS)
            )
            y = (row + 1) * border_px + row * panel_H
            x = (col + 1) * border_px + col * panel_W
            canvas[y:y + panel_H, x:x + panel_W] = resized
        panels.append(Image.fromarray(canvas))
    return panels


def _normal_sampling(video_path: str, C_original: int, total_frames: int):
    """Plain uniform sampling, no merging. total_frames is passed in
    rather than re-read, to avoid opening a second VideoCapture on the
    same file (wasteful, and unreliable on some codecs/drives)."""
    cap = cv2.VideoCapture(video_path)
    indices = np.linspace(0, total_frames - 1, C_original, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def video_panels_pipeline(video_path: str, config: PanelConfig):
    """Video Panels baseline: video_path in, list of composite images out.

    No `query` argument - Video Panels is query-blind by design, and that
    absence is the structural difference from our_method_pipeline below.
    """
    fps, total_frames = get_video_info(video_path)

    C_original = config.C
    C_inflated = C_original * config.panel_width * config.panel_height
    offset = config.fps_limit * fps if config.fps_limit > 0 else 0

    cap = cv2.VideoCapture(video_path)

    # NOTE: threshold uses C_INFLATED, matching the paper's actual code:
    #     C = C * panel_width * panel_height      <- C reassigned first
    #     if (D > offset * C) and (...)           <- so this is inflated C
    # Their written formula (T = C if gamma*C >= D) appears to use the
    # un-inflated C, but the code is authoritative.
    threshold = offset * C_inflated
    will_panel = (total_frames > threshold) and (config.panel_width * config.panel_height > 1)
    print(f"    [paneling decision] fps={fps:.2f}  total_frames={total_frames}  "
          f"C_original={C_original}  C_inflated={C_inflated}  offset={offset:.1f}  "
          f"threshold={threshold:.0f} (=gamma*fps*C_inflated)  "
          f"-> {'PANELING' if will_panel else 'NORMAL SAMPLING (no merge)'}")

    if will_panel:
        indices = np.linspace(0, total_frames - 1, C_inflated, dtype=int)
        raw_frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

        if len(raw_frames) < C_inflated:
            n_readable = len(raw_frames)
            n_complete = n_readable // (config.panel_width * config.panel_height)
            if n_complete == 0:
                print(f"    [fallback] only {n_readable} frames readable, "
                      f"not enough for one panel - using normal sampling")
                return _normal_sampling(video_path, C_original, total_frames)
            print(f"    [partial] only {n_readable} frames readable, "
                  f"building {n_complete} panels instead of {C_original}")
            usable = raw_frames[:n_complete * config.panel_width * config.panel_height]
            return stack_frames_grid(usable, config.panel_width, config.panel_height,
                                     config.border_px, config.border_color,
                                     config.canvas_w, config.canvas_h)

        return stack_frames_grid(raw_frames, config.panel_width, config.panel_height,
                                 config.border_px, config.border_color,
                                 config.canvas_w, config.canvas_h)
    else:
        cap.release()
        return _normal_sampling(video_path, C_original, total_frames)


