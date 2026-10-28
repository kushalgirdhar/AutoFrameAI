"""
Vehicle Overlay Image Compositor — Pure Command-Line Interface (CLI)
Automated computer-vision compositing engine that blends vehicle cutouts onto background
scenes with realistic multi-layer contact shadows, surface reflections, perspective alignment,
and ambient floor bounce lighting.
"""

from __future__ import annotations

import os
import io
import sys
import time
import uuid
import math
import json
import random
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Union, Dict

import cv2
import numpy as np
from PIL import Image

try:
    from resource_tracker import ResourceTracker as _ResourceTracker
except ImportError:
    _ResourceTracker = None  # type: ignore[assignment]

try:
    from removeBg import trim_memory, get_process_memory_mb, VehicleBackgroundRemovalPipeline
except ImportError:
    def trim_memory():
        import gc
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def get_process_memory_mb() -> float:
        return 0.0

    VehicleBackgroundRemovalPipeline = None


def _notify(
    operation: str,
    error: Exception,
    ref_id: Optional[str] = None,
    extra: Optional[str] = None,
) -> None:
    """Error notifications are handled by PHP orchestrator."""
    pass


class _NullContext:
    """No-op context manager used when ResourceTracker is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *_): pass


@dataclass
class Placement:
    scale: float
    x: int
    y: int
    width: int
    height: int
    top_y: int
    bottom_y: int
    source_bbox: Tuple[int, int, int, int]
    background_floor_y: int
    target_contact_y: int


def transform_background(
    bg_img: np.ndarray,
    out_w: int,
    out_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
    scale: float = 1.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
    blur_px: float = 0.0,
) -> np.ndarray:
    """Adjusts background framing, scale, tone, and depth of field without border streaking."""
    h, w = bg_img.shape[:2]
    # Ensure base scale is sufficient to cover translation offsets without border smearing
    needed_w = out_w + 2 * abs(offset_x)
    needed_h = out_h + 2 * abs(offset_y)
    base_scale = max(needed_w / w, needed_h / h) * max(0.2, float(scale))
    nw = max(1, int(round(w * base_scale)))
    nh = max(1, int(round(h * base_scale)))
    interp = cv2.INTER_CUBIC if base_scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(bg_img, (nw, nh), interpolation=interp)

    x0 = (nw - out_w) // 2 - offset_x
    y0 = (nh - out_h) // 2 - offset_y

    x0 = max(0, min(nw - out_w, x0))
    y0 = max(0, min(nh - out_h, y0))

    out = resized[y0:y0 + out_h, x0:x0 + out_w].copy()

    if brightness != 0 or contrast != 1.0:
        out = np.clip(out.astype(np.float32) * float(contrast) + float(brightness), 0, 255).astype(np.uint8)

    if blur_px > 0.5:
        ksize = round(blur_px) * 2 + 1
        out = cv2.GaussianBlur(out, (ksize, ksize), float(blur_px))

    return out


def estimate_floor_line(bg: np.ndarray) -> Tuple[float, float]:
    """Estimate wall/floor seam as y = slope*x + intercept."""
    h, w = bg.shape[:2]
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 7.0)
    grad_y = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))

    y_min = int(h * 0.20)
    y_max = int(h * 0.85)
    if y_max <= y_min + 10:
        return 0.0, float(h * 0.60)

    strip_w = max(24, w // 24)
    points = []
    for x0 in range(0, w, strip_w):
        x1 = min(w, x0 + strip_w)
        profile = grad_y[y_min:y_max, x0:x1].mean(axis=1)
        profile = np.convolve(profile, np.ones(11, np.float32) / 11.0, mode='same')
        iy = int(np.argmax(profile))
        points.append((0.5 * (x0 + x1 - 1), y_min + iy))

    pts = np.asarray(points, dtype=np.float64)
    med_y = float(np.median(pts[:, 1]))
    keep = np.abs(pts[:, 1] - med_y) < max(18.0, h * 0.04)
    if keep.sum() < 5:
        return 0.0, med_y

    coef = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
    slope = float(np.clip(coef[0], -0.15, 0.15))
    intercept = float(np.median(pts[keep, 1] - slope * pts[keep, 0]))
    return slope, intercept


def warp_background_perspective(
    background: np.ndarray,
    tilt_degrees: float = 0.0,
    floor_seam_y: Optional[float] = None,
    floor_seam_info: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, float, float]:
    """Returns undistorted background with clean floor seam position."""
    h, w = background.shape[:2]
    if floor_seam_info is not None:
        seam_slope, seam_b = floor_seam_info
    else:
        seam_slope, seam_b = estimate_floor_line(background)
    center = (w - 1) * 0.5
    center_seam = float(floor_seam_y if floor_seam_y is not None else seam_slope * center + seam_b)
    if abs(tilt_degrees) < 0.01:
        return background, seam_slope, center_seam
    return background.copy(), seam_slope, center_seam


def warp_floor_plane(
    background: np.ndarray,
    floor_tilt_degrees: float = 0.0,
    floor_seam_y: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Compatibility alias for warp_background_perspective."""
    return warp_background_perspective(background, floor_tilt_degrees, floor_seam_y)


@dataclass
class SceneAnalysis:
    floor_slope: float
    floor_seam_y: float
    vanishing_point_x: float
    ceiling_color: np.ndarray
    floor_color: np.ndarray
    ambient_mean_lab: np.ndarray
    ambient_std_lab: np.ndarray
    light_direction_x: float
    key_light_x: float = 0.0


@dataclass
class VehicleAnalysis:
    bbox: Tuple[int, int, int, int]
    contact_points: Tuple[Tuple[int, int], Tuple[int, int]]
    base_y: int
    mean_lab: np.ndarray
    std_lab: np.ndarray
    has_original_shadow: bool
    original_shadow_mask: Optional[np.ndarray]
    auto_rotation_deg: float = 0.0
    front_is_left: bool = True



def estimate_key_light_x(bg_bgr: np.ndarray) -> float:
    """Estimates the normalized horizontal location (-1.0 to 1.0) of the brightest overhead ceiling key light."""
    h, w = bg_bgr.shape[:2]
    ceil_h = max(10, int(h * 0.30))
    ceil = bg_bgr[:ceil_h, :]
    gray = cv2.cvtColor(ceil, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Column-wise brightness
    col_lum = np.mean(gray, axis=0)
    p85 = np.percentile(col_lum, 85)
    weights = np.maximum(0.0, col_lum - p85)
    if np.sum(weights) > 1e-4:
        center_x = float(np.sum(np.arange(w) * weights) / np.sum(weights))
    else:
        center_x = float(w * 0.5)
    # Normalize to [-1.0, 1.0]
    norm_x = (center_x - (w * 0.5)) / max(1.0, w * 0.5)
    return float(np.clip(norm_x, -1.0, 1.0))


def analyze_scene(
    bg_bgr: np.ndarray,
    floor_seam_info: Optional[Tuple[float, float]] = None
) -> SceneAnalysis:
    """Performs automated scene analysis for floor seam, perspective, lighting & ambient tone."""
    h, w = bg_bgr.shape[:2]
    if floor_seam_info is not None:
        slope, seam_y = floor_seam_info
    else:
        slope, seam_y = estimate_floor_line(bg_bgr)
    center_x = (w - 1) * 0.5
    vanishing_x = center_x
    if abs(slope) > 0.005:
        vanishing_x = np.clip(center_x - (seam_y - (h * 0.45)) / slope, 0.0, float(w - 1))

    # Color profiling of floor (bottom 25%) and ceiling (top 20%)
    ceil_strip = bg_bgr[:int(h * 0.22), :]
    floor_strip = bg_bgr[int(h * 0.72):, :]
    ceil_color = np.mean(ceil_strip, axis=(0, 1)).astype(np.float32)
    floor_color = np.mean(floor_strip, axis=(0, 1)).astype(np.float32)

    # Fast ambient tone: convert mean BGR to LAB directly in <1ms without full float32 conversion
    bg_mean_bgr = np.mean(bg_bgr[::4, ::4], axis=(0, 1)).astype(np.uint8)
    amb_mean = cv2.cvtColor(bg_mean_bgr[None, None, :], cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    amb_std = np.std(bg_bgr[::4, ::4].astype(np.float32), axis=(0, 1)) + 1e-4

    # Light direction estimation based on horizontal brightness gradient in ceiling
    ceil_gray = cv2.cvtColor(ceil_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
    left_lum = float(np.mean(ceil_gray[:, :w // 2]))
    right_lum = float(np.mean(ceil_gray[:, w // 2:]))
    denom = max(1.0, left_lum + right_lum)
    light_dir_x = np.clip((right_lum - left_lum) / denom * 2.0, -1.0, 1.0)
    key_light_x = estimate_key_light_x(bg_bgr)

    return SceneAnalysis(
        floor_slope=float(slope),
        floor_seam_y=float(seam_y),
        vanishing_point_x=float(vanishing_x),
        ceiling_color=ceil_color,
        floor_color=floor_color,
        ambient_mean_lab=amb_mean,
        ambient_std_lab=amb_std,
        light_direction_x=float(light_dir_x),
        key_light_x=float(key_light_x),
    )


def segment_vehicle(img: np.ndarray, existing_alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Extracts a smooth, anti-aliased vehicle alpha mask with automatic ground shadow stripping.
    If the uploaded image has an attached cast shadow on the ground, that shadow is detected and removed first.
    """
    h, w = img.shape[:2]

    # 1. Handle pre-existing transparent alpha channel (RGBA PNG) and strip any attached ground shadow
    if existing_alpha is not None and np.any(existing_alpha < 250):
        clean_alpha = existing_alpha.copy()
        solid_mask = clean_alpha > 210
        if np.any(solid_mask):
            y_coords = np.arange(h)[:, None]
            max_solid_y_per_col = np.max(np.where(solid_mask, y_coords, -1), axis=0)
            has_solid = max_solid_y_per_col >= 0
            shadow_pixels = (y_coords > (max_solid_y_per_col[None, :] + 2)) & (clean_alpha < 210) & has_solid[None, :]
            clean_alpha[shadow_pixels] = 0
        return clean_alpha

    # 2. Reuse removeBg AI Pipeline if available for clean foreground extraction
    if VehicleBackgroundRemovalPipeline is not None:
        try:
            pipeline = VehicleBackgroundRemovalPipeline(model_name="u2netp")
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res = pipeline.process(pil_img, edge_refine="crisp")
            cutout_np = np.array(res["cutout_rgba"])
            return cutout_np[:, :, 3]
        except Exception:
            pass

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Sample outer border strips to estimate background color profile
    b = max(4, min(h // 25, w // 25))
    border_strips = np.concatenate([
        img[:b, :].reshape(-1, 3),
        img[:, :b].reshape(-1, 3),
        img[:, -b:].reshape(-1, 3),
        img[-b:, :b*2].reshape(-1, 3),
        img[-b:, -b*2:].reshape(-1, 3),
    ], axis=0).astype(np.float32)

    border_median = np.median(border_strips, axis=0)
    is_dark_bg = float(np.mean(border_median)) < 35.0

    # 2. Dark / Black Background Image Handling
    if is_dark_bg:
        max_bgr = np.max(img, axis=2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grad = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        grad_mag = np.abs(grad)
        
        is_fg = ((max_bgr > 12) | (grad_mag > 15)).astype(np.uint8) * 255
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        is_fg = cv2.morphologyEx(is_fg, cv2.MORPH_CLOSE, k_close, iterations=2)
    else:
        # 3. Studio, Showroom & Outdoor Segmentation with Ground Shadow Stripping
        gc_mask = np.full((h, w), cv2.GC_PR_FGD, dtype=np.uint8)
        m_x = max(4, int(w * 0.04))
        m_y = max(4, int(h * 0.04))
        
        # Border priors (Top, Left, Right and bottom ground margin)
        gc_mask[:m_y, :] = cv2.GC_BGD
        gc_mask[:, :m_x] = cv2.GC_BGD
        gc_mask[:, -m_x:] = cv2.GC_BGD
        gc_mask[-max(2, int(h * 0.015)):, :] = cv2.GC_BGD

        diff_bgr = np.linalg.norm(img.astype(np.float32) - border_median[None, None, :], axis=2)
        border_similar = (diff_bgr < 22.0)

        # Floodfill outer background from perimeter borders to avoid cutting internal car parts
        flood = border_similar.astype(np.uint8) * 255
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0)]:
            if flood[seed[1], seed[0]] == 255:
                cv2.floodFill(flood, flood_mask, seed, 128)
        exterior_bg = (flood == 128)
        gc_mask[exterior_bg] = cv2.GC_BGD

        # Definite vehicle interior (windshield, hood, headlights, grilles)
        hull_mask = np.zeros((h, w), np.uint8)
        hull_mask[int(h * 0.12):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = 255
        interior_fg = (hull_mask == 255) & (~exterior_bg)
        gc_mask[interior_fg] = cv2.GC_FGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img, gc_mask, (0, 0, 0, 0), bgdModel, fgdModel, 6, cv2.GC_INIT_WITH_MASK)
        is_fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # 4. Strip Attached Ground Shadow Skirt from vehicle base
        border_lab = cv2.cvtColor(border_median[None, None, :].astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        if border_lab[0] > 170.0:  # High-key white/grey studio floor
            lab_f = lab.astype(np.float32)
            chroma_dist = np.sqrt((lab_f[:, :, 1] - border_lab[1])**2 + (lab_f[:, :, 2] - border_lab[2])**2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 90)
            k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edge_zone = cv2.dilate(edges, k_edge, iterations=1) > 0

            # Ground shadow is in bottom area, smooth, neutral chroma, and lighter than black tire rubber
            shadow_fringe = (
                (is_fg > 0) &
                (np.arange(h)[:, None] > int(h * 0.68)) &
                (chroma_dist < 16.0) &
                (lab_f[:, :, 0] > 40.0) &
                (lab_f[:, :, 0] < border_lab[0] - 5.0) &
                (~edge_zone)
            )
            is_fg[shadow_fringe] = 0

    # 5. Morphological Cleanup & Smoothing
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    is_fg = cv2.morphologyEx(is_fg, cv2.MORPH_OPEN, k_open, iterations=1)
    is_fg = cv2.morphologyEx(is_fg, cv2.MORPH_CLOSE, k_close, iterations=2)

    # Keep largest vehicle component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(is_fg, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
        is_fg = np.where(labels == k, 255, 0).astype(np.uint8)

    # Fill internal vehicle holes (e.g., windows, windshield, grilles)
    contours, _ = cv2.findContours(is_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(is_fg, contours, -1, 255, -1)

    # 6. Anti-Aliased Subpixel Alpha Edge Smoothing
    alpha = cv2.GaussianBlur(is_fg, (3, 3), 0.8)
    return alpha


def estimate_vehicle_structural_tilt(fg_bgr: np.ndarray, alpha: np.ndarray) -> float:
    """Estimates the structural orientation and roll tilt of the vehicle for all viewing angles.
    
    Supports:
    1. Frontal & Rear Views: Aligns horizontal grille slats, license plate, bumper lip, and headlight brows.
    2. Side & 3/4 Perspective Views: Aligns physically upright cabin pillars (B-pillar, C-pillar, door shutlines).
    
    Returns the exact angle in degrees needed to rotate the vehicle so it stands perfectly straight and level.
    """
    mask = (alpha > 30).astype(np.uint8)
    if np.sum(mask > 0) < 500:
        return 0.0

    h, w = fg_bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 40 or bh < 40:
        return 0.0

    inner_mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    if np.sum(inner_mask > 0) < 300:
        inner_mask = mask

    gray = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray_enh = clahe.apply(gray)

    edges = cv2.Canny(gray_enh, 35, 110)
    edges[inner_mask == 0] = 0

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(18, int(min(bw, bh) * 0.05)),
        maxLineGap=6
    )

    v_corrections = []
    v_weights = []

    h_corrections = []
    h_weights = []

    if lines is not None and len(lines) > 0:
        for line in lines:
            lx1, ly1, lx2, ly2 = line.reshape(4)
            length = float(np.hypot(lx2 - lx1, ly2 - ly1))
            if length < max(15, int(min(bw, bh) * 0.04)):
                continue

            mid_y = (ly1 + ly2) * 0.5

            # 1. Vertical lines (B-pillar, C-pillar, door shutlines, window posts)
            if ly1 > ly2:
                lx1_v, ly1_v, lx2_v, ly2_v = lx2, ly2, lx1, ly1
            else:
                lx1_v, ly1_v, lx2_v, ly2_v = lx1, ly1, lx2, ly2
            dx_v = lx2_v - lx1_v
            dy_v = ly2_v - ly1_v
            ang_v = float(np.degrees(np.arctan2(dx_v, dy_v)))  # angle from vertical down
            if abs(ang_v) <= 15.0:
                cabin_boost = 2.0 if (y0 + bh * 0.10) <= mid_y <= (y0 + bh * 0.75) else 1.0
                v_corrections.append(ang_v)
                v_weights.append((length ** 1.6) * cabin_boost)

            # 2. Near-horizontal lines (grille, bumper line, license plate)
            if lx1 > lx2:
                lx1_h, ly1_h, lx2_h, ly2_h = lx2, ly2, lx1, ly1
            else:
                lx1_h, ly1_h, lx2_h, ly2_h = lx1, ly1, lx2, ly2
            dx_h = lx2_h - lx1_h
            dy_h = ly2_h - ly1_h
            ang_h = float(np.degrees(np.arctan2(dy_h, dx_h)))  # angle from horizontal right
            if abs(ang_h) <= 12.0:
                fascia_boost = 2.5 if mid_y >= (y0 + bh * 0.55) or mid_y <= (y0 + bh * 0.35) else 1.0
                h_corrections.append(ang_h)
                h_weights.append((length ** 1.6) * fascia_boost)

    v_tilt = None
    v_weight_total = 0.0
    if len(v_corrections) >= 2:
        v_arr = np.array(v_corrections)
        v_w = np.array(v_weights)
        v_weight_total = float(np.sum(v_w))
        bins_v = np.arange(-15.0, 15.5, 0.4)
        hist_v, _ = np.histogram(v_arr, bins=bins_v, weights=v_w)
        p_v = np.argmax(hist_v)
        c_v = (bins_v[p_v] + bins_v[p_v + 1]) * 0.5
        inliers_v = np.abs(v_arr - c_v) <= 2.2
        if np.sum(inliers_v) >= 2:
            v_tilt = float(np.average(v_arr[inliers_v], weights=v_w[inliers_v]))
        else:
            v_tilt = float(c_v)

    h_tilt = None
    h_weight_total = 0.0
    if len(h_corrections) >= 2:
        h_arr = np.array(h_corrections)
        h_w = np.array(h_weights)
        h_weight_total = float(np.sum(h_w))
        bins_h = np.arange(-12.0, 12.5, 0.4)
        hist_h, _ = np.histogram(h_arr, bins=bins_h, weights=h_w)
        p_h = np.argmax(hist_h)
        c_h = (bins_h[p_h] + bins_h[p_h + 1]) * 0.5
        inliers_h = np.abs(h_arr - c_h) <= 2.0
        if np.sum(inliers_h) >= 2:
            h_tilt = float(np.average(h_arr[inliers_h], weights=h_w[inliers_h]))
        else:
            h_tilt = float(c_h)

    # Fusion decision
    chosen_tilt = 0.0
    if h_tilt is not None and v_tilt is not None:
        if abs(h_tilt - v_tilt) <= 3.5:
            chosen_tilt = (v_tilt * v_weight_total + h_tilt * h_weight_total) / (v_weight_total + h_weight_total)
        elif h_weight_total > v_weight_total * 1.5:
            chosen_tilt = h_tilt
        else:
            chosen_tilt = v_tilt
    elif h_tilt is not None and abs(h_tilt) >= 0.35:
        chosen_tilt = h_tilt
    elif v_tilt is not None and abs(v_tilt) >= 0.35:
        chosen_tilt = v_tilt

    if abs(chosen_tilt) < 0.35:
        return 0.0

    return float(np.clip(chosen_tilt, -18.0, 18.0))


def analyze_vehicle(fg_bgr: np.ndarray, alpha: np.ndarray) -> VehicleAnalysis:
    """Analyzes vehicle geometry, contact patches, structural auto-leveling angle, and color profile."""
    h, w = fg_bgr.shape[:2]
    mask = alpha > 30
    if not np.any(mask):
        dummy_mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        dummy_std = np.array([20.0, 20.0, 20.0], dtype=np.float32)
        return VehicleAnalysis(
            bbox=(0, 0, w, h),
            contact_points=((int(w * 0.2), h), (int(w * 0.8), h)),
            base_y=h,
            mean_lab=dummy_mean,
            std_lab=dummy_std,
            has_original_shadow=False,
            original_shadow_mask=None,
            auto_rotation_deg=0.0,
            front_is_left=True,
        )

    ys, xs = np.where(mask)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    base_y = int(ys.max())
    span_w = x1 - x0

    # 1. Structural Line Tilt Detection (only if tilt correction enabled)
    structural_tilt = 0.0

    # 2. Detect Left & Right wheel contact points
    left_mask = mask & (np.arange(w)[None, :] <= x0 + int(span_w * 0.40))
    right_mask = mask & (np.arange(w)[None, :] >= x1 - int(span_w * 0.40))

    left_ys, left_xs = np.where(left_mask)
    right_ys, right_xs = np.where(right_mask)

    if len(left_ys) > 0 and len(right_ys) > 0:
        left_bottom_thresh = np.percentile(left_ys, 97.0)
        right_bottom_thresh = np.percentile(right_ys, 97.0)

        left_contact = (
            int(np.mean(left_xs[left_ys >= left_bottom_thresh])),
            int(np.max(left_ys))
        )
        right_contact = (
            int(np.mean(right_xs[right_ys >= right_bottom_thresh])),
            int(np.max(right_ys))
        )
    else:
        left_contact = (int(w * 0.3), base_y)
        right_contact = (int(w * 0.7), base_y)

    # 3. Detect Front vs Rear vehicle orientation
    l_mask_top = mask & (np.arange(w)[None, :] <= x0 + int(span_w * 0.35))
    r_mask_top = mask & (np.arange(w)[None, :] >= x1 - int(span_w * 0.35))
    l_top_ys, _ = np.where(l_mask_top)
    r_top_ys, _ = np.where(r_mask_top)
    l_top = np.percentile(l_top_ys, 5) if len(l_top_ys) else h
    r_top = np.percentile(r_top_ys, 5) if len(r_top_ys) else h
    front_is_left = bool(l_top >= r_top)

    # 4. Auto-Rotation (Disabled - keep original vehicle orientation)
    auto_rotation_deg = 0.0

    # Fast vehicle color profiling in LAB space (mean BGR -> 1-pixel LAB in <1ms)
    car_pixels_bgr = fg_bgr[mask]
    if len(car_pixels_bgr) > 0:
        mean_bgr = np.mean(car_pixels_bgr, axis=0).astype(np.uint8)
        veh_mean_lab = cv2.cvtColor(mean_bgr[None, None, :], cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        veh_std_lab = np.std(car_pixels_bgr.astype(np.float32), axis=0) + 1e-4
    else:
        veh_mean_lab = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        veh_std_lab = np.array([20.0, 20.0, 20.0], dtype=np.float32)

    return VehicleAnalysis(
        bbox=(x0, y0, x1 - x0, y1 - y0),
        contact_points=(left_contact, right_contact),
        base_y=base_y,
        mean_lab=veh_mean_lab,
        std_lab=veh_std_lab,
        has_original_shadow=False,
        original_shadow_mask=None,
        auto_rotation_deg=auto_rotation_deg,
        front_is_left=front_is_left,
    )



def reinhard_color_transfer(
    fg_bgr: np.ndarray,
    alpha: np.ndarray,
    target_mean_lab: np.ndarray,
    target_std_lab: np.ndarray,
    strength: float = 0.30,
) -> np.ndarray:
    """Harmonizes vehicle color balance and ambient color temperature to scene background."""
    if strength <= 0.01:
        return fg_bgr.copy()

    mask = alpha > 25
    if not np.any(mask):
        return fg_bgr.copy()

    fg_lab = cv2.cvtColor(fg_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    car_pixels = fg_lab[mask]
    src_mean = np.mean(car_pixels, axis=0)
    src_std = np.std(car_pixels, axis=0) + 1e-4

    # Scale standard deviation and shift mean
    adjusted = car_pixels.copy()
    # Apply color transfer primarily to a* and b* (chroma) and gently to L* (luminance)
    weight = np.array([0.25 * float(strength), float(strength), float(strength)], dtype=np.float32)
    
    for i in range(3):
        scaled = (adjusted[:, i] - src_mean[i]) * (target_std_lab[i] / src_std[i]) + target_mean_lab[i]
        adjusted[:, i] = adjusted[:, i] * (1.0 - weight[i]) + scaled * weight[i]

    adjusted = np.clip(adjusted, 0.0, 255.0)
    fg_lab[mask] = adjusted
    out_bgr = cv2.cvtColor(fg_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out_bgr


def apply_directional_sheen(
    vehicle_bgr: np.ndarray,
    vehicle_alpha: np.ndarray,
    scene: SceneAnalysis,
    sheen_strength: float = 0.35,
) -> np.ndarray:
    """Applies asymmetric ceiling-light sheen biased towards the detected key light fixture and subtle Fresnel glint."""
    if sheen_strength <= 0.01:
        return vehicle_bgr.copy()

    h, w = vehicle_bgr.shape[:2]
    mask = vehicle_alpha > 30
    if not np.any(mask):
        return vehicle_bgr.copy()

    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    vh = max(1, y1 - y0)
    vw = max(1, int(xs.max()) - int(xs.min()))
    cx = (int(xs.min()) + int(xs.max())) * 0.5

    # 1. Vertical Gradient Highlight (concentrated on roof top, windshield header, and hood)
    rel_y = np.clip((np.arange(h)[:, None] - y0) / float(vh), 0.0, 1.0)
    top_factor = (np.clip(1.0 - rel_y / 0.45, 0.0, 1.0) ** 1.6)[:, :, None]

    # 2. Key Light Asymmetric Horizontal Bias
    rel_x = np.clip((np.arange(w)[None, :] - cx) / float(vw * 0.5), -1.0, 1.0)
    key_bias = 1.0 + (rel_x * float(scene.key_light_x) * 0.40)
    key_bias = np.clip(key_bias, 0.60, 1.40)[..., None]

    veh_f = vehicle_bgr.astype(np.float32)
    ceil_color = scene.ceiling_color[None, None, :]
    sheen = (ceil_color - veh_f) * (top_factor * key_bias * float(sheen_strength) * 0.38)
    sheen = np.maximum(0.0, sheen)

    # 3. Subtle Fresnel Rim Glint on Top Outlines
    k_edge = cv2.Canny((vehicle_alpha > 120).astype(np.uint8) * 255, 100, 200)
    k_edge_top = k_edge.astype(np.float32) / 255.0
    k_edge_top[int(y0 + vh * 0.5):, :] = 0.0
    k_edge_top = cv2.GaussianBlur(k_edge_top, (5, 5), 0)[..., None]

    out = veh_f + sheen + (ceil_color * k_edge_top * float(sheen_strength) * 0.20)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_directional_lighting_and_fresnel(
    vehicle_bgr: np.ndarray,
    vehicle_alpha: np.ndarray,
    scene: SceneAnalysis,
    sheen_strength: float = 0.35,
) -> np.ndarray:
    """Compatibility wrapper for apply_directional_sheen."""
    return apply_directional_sheen(vehicle_bgr, vehicle_alpha, scene, sheen_strength)


def apply_ground_bounce_light(
    vehicle_bgr: np.ndarray,
    vehicle_alpha: np.ndarray,
    scene: SceneAnalysis,
    bounce_strength: float = 0.25,
) -> np.ndarray:
    """Applies diffuse ground bounce light from the bright garage floor onto lower 30% of the vehicle body."""
    if bounce_strength <= 0.01 or scene is None:
        return vehicle_bgr.copy()

    mask = vehicle_alpha > 30
    if not np.any(mask):
        return vehicle_bgr.copy()

    h, w = vehicle_bgr.shape[:2]
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    vh = max(1, y1 - y0)

    # Gradient active in the lower 35% of the vehicle chassis
    rel_y = np.clip((np.arange(h)[:, None] - (y1 - int(vh * 0.35))) / max(1.0, float(vh * 0.35)), 0.0, 1.0)
    bounce_factor = (rel_y ** 1.6)[:, :, None]

    floor_c = scene.floor_color[None, None, :].astype(np.float32)
    veh_f = vehicle_bgr.astype(np.float32)

    # Soft ambient upward bounce
    bounce_delta = (floor_c - veh_f) * (bounce_factor * float(bounce_strength) * 0.25)
    out = veh_f + np.maximum(-5.0, bounce_delta)
    return np.clip(out, 0, 255).astype(np.uint8)


def refine_tone_with_histogram_matching(
    vehicle_bgr: np.ndarray,
    vehicle_alpha: np.ndarray,
    floor_patch_bgr: np.ndarray,
    strength: float = 0.35,
) -> np.ndarray:
    """Matches the vehicle's lightness distribution (L* channel) against the local floor patch exposure."""
    if strength <= 0.01 or floor_patch_bgr is None or floor_patch_bgr.size == 0:
        return vehicle_bgr.copy()

    mask = vehicle_alpha > 30
    if not np.any(mask):
        return vehicle_bgr.copy()

    try:
        import skimage.exposure
        veh_lab = cv2.cvtColor(vehicle_bgr, cv2.COLOR_BGR2LAB)
        floor_lab = cv2.cvtColor(floor_patch_bgr, cv2.COLOR_BGR2LAB)

        # Match L* channel of car to floor
        matched_l = skimage.exposure.match_histograms(veh_lab[..., 0], floor_lab[..., 0])
        orig_l = veh_lab[..., 0].astype(np.float32)
        target_l = matched_l.astype(np.float32)

        # Blend with strength
        blend_l = orig_l * (1.0 - float(strength)) + target_l * float(strength)
        veh_lab[..., 0] = np.clip(blend_l, 0, 255).astype(np.uint8)

        out_bgr = cv2.cvtColor(veh_lab, cv2.COLOR_LAB2BGR)
        res = vehicle_bgr.copy()
        res[mask] = out_bgr[mask]
        return res
    except Exception:
        return vehicle_bgr.copy()


def decontaminate_edge_colors(
    vehicle_rgba: np.ndarray,
    src_bg_color: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Eliminates faint white/grey edge halos by unmixing background bleed from boundary pixels
    and inpainting genuine vehicle interior paint colors into the silhouette perimeter.
    """
    out = vehicle_rgba.copy()
    bgr = out[..., :3].astype(np.float32)
    alpha = out[..., 3].astype(np.float32)

    mask = (alpha > 15).astype(np.uint8)
    if not np.any(mask):
        return out

    # 1. Source background color estimate (default to studio white/high-key grey)
    if src_bg_color is None:
        src_bg = np.array([245.0, 245.0, 245.0], dtype=np.float32)
    else:
        src_bg = np.asarray(src_bg_color, dtype=np.float32)

    # 2. Algebraic Alpha Unmixing for semi-transparent boundary pixels (10 <= alpha < 245)
    edge_semi = (alpha >= 10) & (alpha < 245)
    if np.any(edge_semi):
        a_norm = (alpha[edge_semi] / 255.0)[:, None]
        unmixed = (bgr[edge_semi] - (1.0 - a_norm) * src_bg[None, :]) / np.maximum(0.20, a_norm)
        bgr[edge_semi] = np.clip(unmixed, 0.0, 255.0)

    # 3. Morphological Perimeter Inpainting: Dilate clean solid interior vehicle colors into perimeter
    h, w = alpha.shape[:2]
    ys, xs = np.where(alpha > 0)
    if len(ys) > 0:
        margin = 12
        y0, y1 = max(0, int(ys.min()) - margin), min(h, int(ys.max()) + margin)
        x0, x1 = max(0, int(xs.min()) - margin), min(w, int(xs.max()) + margin)

        alpha_crop = alpha[y0:y1, x0:x1]
        bgr_crop = bgr[y0:y1, x0:x1]

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_solid = cv2.erode((alpha_crop > 200).astype(np.uint8), k3, iterations=2)
        if not np.any(inner_solid):
            inner_solid = cv2.erode((alpha_crop > 120).astype(np.uint8), k3, iterations=1)
            if not np.any(inner_solid):
                inner_solid = (alpha_crop > 60).astype(np.uint8)

        fringe_band = (alpha_crop > 0) & (inner_solid == 0)
        if np.any(fringe_band):
            k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            solid_color = bgr_crop * inner_solid[..., None].astype(np.float32)
            solid_mask = inner_solid.astype(np.float32)
            
            dil_color = cv2.dilate(solid_color, k5, iterations=3)
            dil_mask = cv2.dilate(solid_mask, k5, iterations=3)[..., None]
            
            clean_fill = np.where(dil_mask > 1e-3, dil_color / np.maximum(1e-3, dil_mask), bgr_crop)
            bgr_crop[fringe_band] = bgr_crop[fringe_band] * 0.30 + clean_fill[fringe_band] * 0.70
            bgr[y0:y1, x0:x1] = bgr_crop

        # 4. Smooth anti-aliased alpha boundary
        choked_alpha = cv2.erode(alpha_crop, k3, iterations=1)
        smooth_alpha = cv2.GaussianBlur(choked_alpha, (3, 3), 0.5)
        alpha[y0:y1, x0:x1] = smooth_alpha

    out[..., :3] = np.clip(bgr, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return out


def refine_edges_and_defringe(vehicle_rgba: np.ndarray) -> np.ndarray:
    """Compatibility wrapper for decontaminate_edge_colors."""
    return decontaminate_edge_colors(vehicle_rgba)


def transform_vehicle(
    vehicle_rgba: np.ndarray,
    flip_h: bool = False,
    rotation_deg: float = 0.0,
    persp_x: float = 0.0,
    persp_y: float = 0.0,
    front_lift_px: float = 0.0,
    rear_lift_px: float = 0.0,
    front_is_left: bool = True,
    skew_x_deg: float = 0.0,
    camber_deg: float = 0.0,
    brightness: float = 0.0,
    contrast: float = 1.0,
) -> np.ndarray:
    # Applies user-controlled vehicle transformations
    out = vehicle_rgba.copy()
    h, w = out.shape[:2]

    if flip_h:
        out = cv2.flip(out, 1)

    # 1. Front & Rear Height Elevation (Move upward/downward from front or rear independently)
    if abs(front_lift_px) > 0.1 or abs(rear_lift_px) > 0.1:
        lift_l = float(front_lift_px if front_is_left else rear_lift_px)
        lift_r = float(rear_lift_px if front_is_left else front_lift_px)
        if flip_h:
            lift_l, lift_r = lift_r, lift_l

        # Positive lift moves UP (-Y), negative lift moves DOWN (+Y)
        dy_l = -lift_l
        dy_r = -lift_r

        pad_top = int(max(0, -min(dy_l, dy_r))) + 15
        pad_bot = int(max(0, max(dy_l, dy_r))) + 15
        new_h = h + pad_top + pad_bot

        src_lift = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        dst_lift = np.array([
            [0, pad_top + dy_l],
            [w - 1, pad_top + dy_r],
            [w - 1, pad_top + (h - 1) + dy_r],
            [0, pad_top + (h - 1) + dy_l]
        ], dtype=np.float32)

        M_lift = cv2.getPerspectiveTransform(src_lift, dst_lift)
        out = cv2.warpPerspective(
            out, M_lift, (w, new_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        h, w = out.shape[:2]

    # 2. Body Skew / Stance Shear (shifts upper vehicle body while preserving bottom contact)
    if abs(skew_x_deg) > 0.05:
        shear_factor = np.tan(np.radians(skew_x_deg))
        M_skew = np.array([
            [1.0, -shear_factor, shear_factor * h if shear_factor < 0 else 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)
        pad_w = int(abs(shear_factor) * h)
        new_w = w + pad_w
        if shear_factor < 0:
            M_skew[0, 2] += pad_w * 0.5
        out = cv2.warpAffine(
            out, M_skew, (new_w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        h, w = out.shape[:2]

    # 3. 3D Perspective Keystoning (3D Yaw, 3D Pitch, Camber Stance)
    if abs(persp_x) > 0.01 or abs(persp_y) > 0.01 or abs(camber_deg) > 0.01:
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        
        shift_yaw = (persp_x / 100.0) * (w * 0.28)
        shift_pitch = (persp_y / 100.0) * (h * 0.22)
        shift_camber = (camber_deg / 100.0) * (w * 0.15)

        top_l_x = max(0.0, shift_yaw + shift_camber)
        top_l_y = max(0.0, shift_pitch)
        top_r_x = min(float(w - 1), (w - 1) - max(0.0, -shift_yaw) - shift_camber)
        top_r_y = max(0.0, shift_pitch)

        bot_r_x = min(float(w - 1), (w - 1) + min(0.0, shift_yaw) + shift_camber)
        bot_r_y = min(float(h - 1), (h - 1) - max(0.0, -shift_pitch))
        bot_l_x = max(0.0, -min(0.0, shift_yaw) - shift_camber)
        bot_l_y = min(float(h - 1), (h - 1) - max(0.0, -shift_pitch))

        dst = np.array([
            [top_l_x, top_l_y],
            [top_r_x, top_r_y],
            [bot_r_x, bot_r_y],
            [bot_l_x, bot_l_y]
        ], dtype=np.float32)

        M_persp = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(
            out, M_persp, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    # 4. 2D Roll Rotation
    if abs(rotation_deg) > 0.05:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        out = cv2.warpAffine(
            out, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

    # 5. Tone adjustments (Brightness / Contrast)
    if brightness != 0 or contrast != 1.0:
        bgr = out[..., :3].astype(np.float32)
        bgr = np.clip(bgr * float(contrast) + float(brightness), 0, 255).astype(np.uint8)
        out[..., :3] = bgr

    return out


def fit_vehicle(
    vehicle_rgba: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    floor_y: int,
    offset_x: int = 0,
    offset_y: int = 0,
    scale_multiplier: float = 1.0,
    ground_contact_y_ratio: float = 0.91,
    ground_fine_offset: int = 0,
    max_w_ratio: float = 0.82,
) -> Tuple[np.ndarray, Placement]:
    """Positions vehicle firmly on the floor plane."""
    alpha = vehicle_rgba[..., 3]
    ys, xs = np.where(alpha > 30)
    if len(xs) == 0:
        sx, sy, ex, ey = 0, 0, vehicle_rgba.shape[1], vehicle_rgba.shape[0]
        contact_src_y = int(vehicle_rgba.shape[0] * 0.9)
    else:
        sx, sy, ex, ey = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
        contact_src_y = int(ys.max())
    bw, bh = ex - sx, ey - sy

    target_contact_y = int(canvas_h * float(ground_contact_y_ratio)) + ground_fine_offset
    target_top_y = int(canvas_h * 0.10)
    target_h = target_contact_y - target_top_y

    scale_h = target_h / max(1, (contact_src_y - sy))
    scale_w = (canvas_w * max_w_ratio) / max(1, bw)
    base_scale = min(scale_h, scale_w) * max(0.2, float(scale_multiplier))

    nw = max(1, round(vehicle_rgba.shape[1] * base_scale))
    nh = max(1, round(vehicle_rgba.shape[0] * base_scale))

    resized = cv2.resize(vehicle_rgba, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    ralpha = resized[..., 3]
    sys, sxs = np.where(ralpha > 30)
    if len(sxs) == 0:
        rsx, rsy, rex, rey = 0, 0, nw, nh
        rcontact = int(nh * 0.9)
    else:
        rsx, rsy, rex, rey = int(sxs.min()), int(sys.min()), int(sxs.max() + 1), int(sys.max() + 1)
        rcontact = int(sys.max())

    px = round(canvas_w * 0.5 - (rsx + rex) / 2) + offset_x
    py = round(target_contact_y - rcontact) + offset_y

    # Canvas Bottom Edge Boundary Protection:
    # Ensure vehicle bottom never extends past canvas boundary (canvas_h - 4) to avoid flat tire clipping
    actual_bottom_y = py + int(sys.max() if len(sys) > 0 else rcontact)
    if actual_bottom_y > canvas_h - 4:
        py -= (actual_bottom_y - (canvas_h - 4))

    placement = Placement(
        scale=float(base_scale),
        x=px,
        y=py,
        width=nw,
        height=nh,
        top_y=py + rsy,
        bottom_y=py + rcontact,
        source_bbox=(sx, sy, bw, bh),
        background_floor_y=floor_y,
        target_contact_y=target_contact_y,
    )
    return resized, placement


def render_effects_and_composite(
    bg_bgr: np.ndarray,
    resized_rgba: np.ndarray,
    placement: Placement,
    scene: Optional[SceneAnalysis] = None,
    original_shadow_mask: Optional[np.ndarray] = None,
    # Grounding Controls
    tyre_contact_left: int = 0,
    tyre_contact_right: int = 0,
    # Shadow Controls
    shadow_contact_strength: float = 1.0,
    shadow_underbody_strength: float = 0.8,
    shadow_blur: float = 1.0,
    shadow_spread: float = 1.0,
    shadow_offset_x: int = 0,
    shadow_offset_y: int = 0,
    shadow_direction_deg: float = 0.0,
    # Reflection Controls
    refl_opacity: float = 0.20,
    refl_blur: float = 3.0,
) -> np.ndarray:
    """Renders comprehensive multi-layer floor shadow, specular reflection, and anti-aliased composite."""
    h_bg, w_bg = bg_bgr.shape[:2]
    canvas = bg_bgr.astype(np.float32)

    px, py = placement.x, placement.y
    rw, rh = resized_rgba.shape[1], resized_rgba.shape[0]
    ralpha = resized_rgba[..., 3].astype(np.float32)
    car_rgb = resized_rgba[..., :3].astype(np.float32)

    mask = ralpha > 40
    if not np.any(mask):
        return np.clip(canvas, 0, 255).astype(np.uint8)

    ys, xs = np.where(mask)
    car_x0, car_x1 = int(xs.min()), int(xs.max() + 1)
    car_w = max(1, car_x1 - car_x0)
    contact_local_y = int(np.percentile(ys, 99.0))

    # Column-wise bottom silhouette profile of the vehicle (fast vectorized)
    y_coords = np.arange(rh)[:, None]
    col_bottoms = np.zeros(rw, dtype=np.int32)
    col_bottoms[car_x0:car_x1] = np.max(np.where(mask[:, car_x0:car_x1], y_coords, 0), axis=0)

    max_car_y = np.max(col_bottoms[car_x0:car_x1]) if car_x1 > car_x0 else contact_local_y

    # Automated Intelligent Wheel & Ground Contact Detection (Front, Rear, Side, 3/4)
    kernel_size = max(5, int(rw * 0.01)) | 1
    smoothed = cv2.GaussianBlur(col_bottoms.astype(np.float32)[:, None], (1, kernel_size), 0).ravel()
    gray_car = cv2.cvtColor(car_rgb, cv2.COLOR_BGR2GRAY)
    # Require realistic axle separation (wheelbase is at least 25% of vehicle width)
    min_dist = max(40, int(rw * 0.25))
    peaks = []

    for x in range(int(rw * 0.05), int(rw * 0.95)):
        w_half = max(5, int(rw * 0.02))
        x_min = max(0, x - w_half)
        x_max = min(rw, x + w_half + 1)
        if smoothed[x] == np.max(smoothed[x_min:x_max]) and smoothed[x] > rh * 0.35:
            y_peak = int(col_bottoms[x])
            sample_y0 = max(0, y_peak - 10)
            sample_y1 = min(rh, y_peak + 1)
            sample_x0 = max(0, x - 8)
            sample_x1 = min(rw, x + 9)
            sub_gray = gray_car[sample_y0:sample_y1, sample_x0:sample_x1]
            sub_mask = mask[sample_y0:sample_y1, sample_x0:sample_x1]
            if sub_mask.any():
                mean_gray = np.mean(sub_gray[sub_mask])
                if mean_gray < 115:
                    c_xs = np.where((np.abs(col_bottoms - y_peak) <= 5) & (np.abs(np.arange(rw) - x) <= int(rw * 0.06)))[0]
                    cw = max(20, c_xs.max() - c_xs.min()) if len(c_xs) else 25
                    peaks.append({'x': x, 'y': y_peak, 'cw': cw, 'mean_gray': mean_gray})

    peaks.sort(key=lambda p: p['y'], reverse=True)
    wheels = []
    for p in peaks:
        if all(abs(p['x'] - w['x']) > min_dist for w in wheels):
            wheels.append(p)
            if len(wheels) == 2:
                break

    # Fallback if fewer than 2 wheels detected
    if len(wheels) < 2 and (car_x1 - car_x0) > 40:
        car_w = car_x1 - car_x0
        left_range = col_bottoms[car_x0:int(car_x0 + car_w * 0.42)]
        right_range = col_bottoms[int(car_x1 - car_w * 0.42):car_x1]
        if len(left_range) and len(right_range):
            lx = car_x0 + int(np.argmax(left_range))
            rx = int(car_x1 - car_w * 0.42) + int(np.argmax(right_range))
            if not any(abs(w['x'] - lx) < min_dist for w in wheels):
                wheels.append({'x': lx, 'y': int(col_bottoms[lx]), 'cw': max(25, int(car_w * 0.07))})
            if not any(abs(w['x'] - rx) < min_dist for w in wheels):
                wheels.append({'x': rx, 'y': int(col_bottoms[rx]), 'cw': max(25, int(car_w * 0.07))})

    wheels.sort(key=lambda w: w['x'])

    if len(wheels) >= 2:
        dx = wheels[-1]['x'] - wheels[0]['x']
        dy = wheels[-1]['y'] - wheels[0]['y']
        car_angle_deg = float(np.degrees(np.arctan2(dy, dx)))
    else:
        car_angle_deg = 0.0

    # 1. Glossy Floor Specular Reflection
    if refl_opacity > 0.01:
        refl_h = min(int(rh * 0.18 * max(0.5, float(shadow_spread))), max(0, h_bg - (py + contact_local_y)))
        if refl_h > 3:
            lower_crop = resized_rgba[max(0, contact_local_y - refl_h):contact_local_y, :]
            refl_flipped = cv2.flip(lower_crop, 0)
            refl_bgr = refl_flipped[..., :3].astype(np.float32)
            refl_a = (refl_flipped[..., 3].astype(np.float32) / 255.0)

            # Quadratic vertical fade with distance from floor
            decay = (np.linspace(1.0, 0.0, refl_flipped.shape[0]) ** 2.2)[:, None]
            fade = float(refl_opacity * 0.55) * decay
            refl_alpha = refl_a * fade

            # Anisotropic floor gloss blur (wider horizontal dispersion)
            k_rx = max(3, round(refl_blur * 4.0)) | 1
            k_ry = max(1, round(refl_blur * 1.2)) | 1
            refl_bgr = cv2.GaussianBlur(refl_bgr, (k_rx, k_ry), 0)

            rx0 = max(0, px)
            ry0 = py + contact_local_y
            rx1 = min(w_bg, rx0 + rw)
            ry1 = min(h_bg, ry0 + refl_flipped.shape[0])
            sw = rx1 - rx0
            sh = ry1 - ry0
            if sw > 0 and sh > 0:
                patch_a = refl_alpha[:sh, :sw, None]
                patch_bgr = refl_bgr[:sh, :sw]
                canvas[ry0:ry1, rx0:rx1] = patch_bgr * patch_a + canvas[ry0:ry1, rx0:rx1] * (1.0 - patch_a)

    # 2. Photometric Realistic Automotive Ground Shadow Engine
    if float(shadow_contact_strength) > 0.01:
        contact_layer = np.zeros((h_bg, w_bg), dtype=np.float32)
        underbody_layer = np.zeros((h_bg, w_bg), dtype=np.float32)

        dir_rad = math.radians(float(np.clip(shadow_direction_deg, -45.0, 45.0)))
        dir_tan = math.tan(dir_rad)

        # Ground plane perspective orientation
        if len(wheels) >= 2:
            w0, w1 = wheels[0], wheels[1]
            dx = w1['x'] - w0['x']
            dy = w1['y'] - w0['y']
            slope = dy / max(1.0, float(dx))
            ground_rad = math.atan2(dy, dx)
        else:
            w0 = wheels[0] if wheels else {'x': int(car_x0 + car_w * 0.3), 'y': contact_local_y, 'cw': max(25, int(car_w * 0.07))}
            w1 = {'x': int(car_x0 + car_w * 0.7), 'y': contact_local_y, 'cw': max(25, int(car_w * 0.07))}
            slope = 0.0
            ground_rad = 0.0

        x0_c = px + w0['x'] + shadow_offset_x
        y0_c = py + w0['y'] + shadow_offset_y + (tyre_contact_left if w0['x'] < rw * 0.5 else tyre_contact_right)
        x1_c = px + w1['x'] + shadow_offset_x
        y1_c = py + w1['y'] + shadow_offset_y + (tyre_contact_left if w1['x'] < rw * 0.5 else tyre_contact_right)

        min_wc = min(x0_c, x1_c)
        max_wc = max(x0_c, x1_c)

                # 2a. Continuous Undercarriage Ambient Ground Bed with Soft Natural Floor Spread
        if float(shadow_underbody_strength) > 0.01:
            # Vehicle footprint span
            span_x0 = max(0, px + car_x0)
            span_x1 = min(w_bg, px + car_x1)

            for cx in range(span_x0, span_x1):
                lx = int(np.clip(cx - px, car_x0, car_x1 - 1))
                c_bot = py + col_bottoms[lx] + shadow_offset_y

                if cx < min_wc:
                    # Zone E: Rear Overhang Shadow under bumper & mudflap behind rear wheel
                    dist_px = float(min_wc - cx)
                    tot_rear_px = max(1.0, float(min_wc - (px + car_x0)))
                    t_rear = dist_px / tot_rear_px
                    overhang_weight = math.exp(-0.5 * (max(0.0, t_rear - 0.15) / 0.40) ** 2) if t_rear > 0.15 else 1.0
                    depth_factor = 1.0
                    gy = int(round(c_bot + (y0_c - c_bot) * max(0.0, 1.0 - t_rear * 0.85)))
                    penumbra_h = int(5 * overhang_weight * max(0.5, float(shadow_spread)))
                elif cx > max_wc:
                    # Front Overhang Shadow under front bumper/chin
                    dist_px = float(cx - max_wc)
                    overhang_weight = math.exp(-0.5 * (dist_px / 18.0) ** 2)
                    depth_factor = 0.50
                    gy = int(round(y1_c + (c_bot - y1_c) * 0.25))
                    penumbra_h = int(4 * overhang_weight * max(0.5, float(shadow_spread)))
                else:
                    # Zone B: Undercarriage Cavity between wheels
                    t = (cx - min_wc) / max(1.0, float(max_wc - min_wc))
                    gy = int(round(y0_c * (1.0 - t) + y1_c * t))
                    overhang_weight = 1.0
                    depth_factor = 1.0 - t * 0.15
                    # Natural soft spread along sill: slightly fuller in center, gently tapering towards wheels
                    t_sill = math.sin(t * math.pi)
                    penumbra_h = int((4 + 6 * t_sill * depth_factor) * max(0.5, float(shadow_spread)))

                # Undercarriage cavity spans between sill bottom and ground line
                cavity_top = max(0, min(c_bot - 8, gy))
                cavity_bot = min(h_bg - 1, max(c_bot, gy))

                # Rich, solid underbody cavity core
                core_density = min(0.96, 0.92 * depth_factor * overhang_weight * float(shadow_underbody_strength) * 1.45)
                if core_density > 0.01:
                    underbody_layer[cavity_top:cavity_bot + 1, cx] = np.maximum(
                        underbody_layer[cavity_top:cavity_bot + 1, cx],
                        core_density
                    )

                # Soft, realistic ambient penumbra: spreads a little bit out onto the floor so it is not a straight line
                penumbra_h = max(2, min(penumbra_h, 11))
                for dy_p in range(1, penumbra_h + 1):
                    y_pen = cavity_bot + dy_p
                    if 0 <= y_pen < h_bg:
                        decay = math.exp(-0.5 * (dy_p / max(1.0, penumbra_h * 0.44)) ** 2)
                        p_val = core_density * 0.58 * decay
                        if p_val > underbody_layer[y_pen, cx]:
                            underbody_layer[y_pen, cx] = p_val

            # Optical diffusion: smooth lateral blend and gentle vertical feathering
            k_ub_x = max(7, round(17 * float(shadow_blur))) | 1
            k_ub_y = max(5, round(7 * float(shadow_blur))) | 1
            underbody_layer = cv2.GaussianBlur(underbody_layer, (k_ub_x, k_ub_y), 0)

        # 2b. Realistic Ground Wheel Contact Patches with Gentle Ambient Falloff
        for w_info in wheels:
            wx = px + w_info['x'] + shadow_offset_x
            wy = py + w_info['y'] + shadow_offset_y
            cw = float(w_info['cw'])

            if w_info['x'] < rw * 0.5:
                wy += tyre_contact_left
            else:
                wy += tyre_contact_right

            # Perspective depth based on vertical position
            ground_depth = (wy - py) / max(1.0, float(contact_local_y))
            depth_w = float(np.clip(0.55 + 0.45 * ground_depth, 0.55, 1.0))

            # Directional offset along wheel axis
            wx += round(cw * 0.20 * dir_tan)

            is_rear = (wx < (min_wc + max_wc) * 0.5)

            # Ground-flat footprint
            eff_cw = max(cw * 1.25, car_w * 0.075 * depth_w) if is_rear else cw * 1.15
            rx = eff_cw * 0.65 * depth_w * max(0.5, float(shadow_spread))

            # Soft asymmetric vertical profile:
            # Under car (dy < 0): expands into undercarriage cavity B
            ry_under = max(4.5, eff_cw * 0.22 * depth_w)
            # Outer floor (dy > 0): gentle spread (3-5px) for soft natural grounding without far projection
            ry_floor = max(2.5, eff_cw * 0.09 * depth_w)

            bb_x = int(rx * 3.5)
            bb_y = int(max(ry_under, ry_floor) * 3.5)

            x_min = max(0, int(wx - bb_x))
            x_max = min(w_bg, int(wx + bb_x + 1))
            y_min = max(0, int(wy - bb_y))
            y_max = min(h_bg, int(wy + bb_y + 1))

            if x_max > x_min and y_max > y_min:
                gy, gx = np.ogrid[y_min:y_max, x_min:x_max]
                dx = gx - wx
                dy = gy - wy

                eff_ry = np.where(dy > 0, ry_floor, ry_under)

                # Tier 1: Crisp contact pinch core directly under rubber
                d_core = (dx / rx)**2 + (dy / eff_ry)**2
                core = np.exp(-0.5 * d_core) * 0.98 * float(shadow_contact_strength)

                # Tier 2: Ground penumbra halo (gentle spread around tire)
                eff_ry_halo = np.where(dy > 0, ry_floor * 1.5, ry_under * 1.5)
                d_halo = (dx / (rx * 1.35))**2 + (dy / eff_ry_halo)**2
                halo = np.exp(-0.5 * d_halo) * 0.68 * float(shadow_contact_strength)

                # Tier 3: Soft ambient falloff cushion
                eff_ry_soft = np.where(dy > 0, ry_floor * 2.0, ry_under * 1.9)
                d_soft = (dx / (rx * 1.75))**2 + (dy / eff_ry_soft)**2
                soft = np.exp(-0.5 * d_soft) * 0.30 * float(shadow_contact_strength)

                tire_patch = np.maximum(core, np.maximum(halo, soft))

                # Smooth Hann edge taper to guarantee zero rectangular boundaries
                window_x = np.clip(np.cos(0.5 * math.pi * np.abs(dx) / max(1.0, float(bb_x))), 0.0, 1.0) ** 2
                window_y = np.clip(np.cos(0.5 * math.pi * np.abs(dy) / max(1.0, float(bb_y))), 0.0, 1.0) ** 2
                tire_patch = tire_patch * (window_x * window_y)

                contact_layer[y_min:y_max, x_min:x_max] = np.maximum(
                    contact_layer[y_min:y_max, x_min:x_max],
                    tire_patch
                )

        # 2c. Combine Underbody Bed + Contact Patches with Optical Transmission Fusion
        k_c = max(5, round(7 * float(shadow_blur))) | 1
        contact_layer = cv2.GaussianBlur(contact_layer, (k_c, k_c), 0)
        ub = underbody_layer * 0.94
        ct = contact_layer
        # Physical light transmission fusion: combines shadows seamlessly into one unified bed
        total_shadow = ub + ct - (ub * ct)
        k_m = max(5, round(7 * float(shadow_blur))) | 1
        total_shadow = cv2.GaussianBlur(total_shadow, (k_m, k_m), 0)
        total_shadow = np.clip(total_shadow, 0.0, 0.96)

        # Physical multiplicative floor absorption
        shadow_factor = 1.0 - (total_shadow[..., None] * 0.94)
        canvas *= shadow_factor

    # 3. Sub-Pixel Anti-Aliased Alpha Over Compositing
    car_bgr = car_rgb.copy()
    car_alpha_norm = ralpha / 255.0
    car_alpha = car_alpha_norm[..., None]

    dx0 = max(0, px)
    dy0 = max(0, py)
    dx1 = min(w_bg, px + rw)
    dy1 = min(h_bg, py + rh)

    sx0 = dx0 - px
    sy0 = dy0 - py
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)

    if dx1 > dx0 and dy1 > dy0:
        canvas[dy0:dy1, dx0:dx1] = (
            car_bgr[sy0:sy1, sx0:sx1] * car_alpha[sy0:sy1, sx0:sx1] +
            canvas[dy0:dy1, dx0:dx1] * (1.0 - car_alpha[sy0:sy1, sx0:sx1])
        )

    return np.clip(canvas, 0, 255).astype(np.uint8)


# Default Output Dimensions & Preset Name
DEFAULT_OUTPUT_WIDTH: int = 1024
DEFAULT_OUTPUT_HEIGHT: int = 682
DEFAULT_PRESET_NAME: str = "default_grounded"

# Key translation mappings between Python snake_case and UI camelCase
SNAKE_TO_CAMEL: dict[str, str] = {
    "bg_offset_x": "bgOffsetX",
    "bg_offset_y": "bgOffsetY",
    "bg_scale": "bgScale",
    "bg_floor_y_ratio": "bgFloorY",
    "bg_tilt_deg": "bgTiltDeg",
    "bg_blur": "bgBlur",
    "bg_brightness": "bgBrightness",
    "bg_contrast": "bgContrast",
    "ground_contact_y": "groundContactY",
    "ground_fine_offset": "groundFineOffset",
    "tyre_contact_left": "tyreContactLeft",
    "tyre_contact_right": "tyreContactRight",
    "veh_brightness": "vehBrightness",
    "veh_contrast": "vehContrast",
    "veh_offset_x": "vehOffsetX",
    "veh_offset_y": "vehOffsetY",
    "veh_scale_mult": "vehScaleMult",
    "veh_rotation_deg": "vehRotationDeg",
    "veh_front_lift": "vehFrontLift",
    "veh_rear_lift": "vehRearLift",
    "veh_persp_x": "vehPerspX",
    "veh_persp_y": "vehPerspY",
    "veh_skew_x_deg": "vehSkewXDeg",
    "veh_camber_deg": "vehCamberDeg",
    "veh_flip_h": "vehFlipH",
    "shadow_contact_strength": "shadowContactStrength",
    "shadow_underbody_strength": "shadowUnderbodyStrength",
    "shadow_blur": "shadowBlur",
    "shadow_spread": "shadowSpread",
    "shadow_offset_x": "shadowOffsetX",
    "shadow_offset_y": "shadowOffsetY",
    "shadow_direction_deg": "shadowDirectionDeg",
    "refl_opacity": "reflOpacity",
    "refl_blur": "reflBlur",
}
CAMEL_TO_SNAKE: dict[str, str] = {v: k for k, v in SNAKE_TO_CAMEL.items()}

# 4 Core Studio Presets
_RAW_PRESETS: dict[str, dict] = {
    "default_grounded": {
        "name": "✨ Default Grounded Showroom",
        "bg_offset_x": 0,
        "bg_offset_y": -210,
        "bg_scale": 1.0,
        "bg_floor_y_ratio": 0.36,
        "bg_tilt_deg": 0.0,
        "bg_blur": 0.0,
        "bg_brightness": 0.0,
        "bg_contrast": 1.0,
        "ground_contact_y": 0.91,
        "ground_fine_offset": 0,
        "tyre_contact_left": 0,
        "tyre_contact_right": 0,
        "veh_brightness": 0.0,
        "veh_contrast": 1.0,
        "veh_offset_x": 0,
        "veh_offset_y": 0,
        "veh_scale_mult": 1.0,
        "veh_rotation_deg": 0.0,
        "veh_front_lift": 0.0,
        "veh_rear_lift": 0.0,
        "veh_persp_x": 0.0,
        "veh_persp_y": 0.0,
        "veh_skew_x_deg": 0.0,
        "veh_camber_deg": 0.0,
        "veh_flip_h": False,
        "shadow_contact_strength": 0.85,
        "shadow_underbody_strength": 0.60,
        "shadow_blur": 1.0,
        "shadow_spread": 1.0,
        "shadow_offset_x": 0,
        "shadow_offset_y": 0,
        "shadow_direction_deg": 0.0,
        "refl_opacity": 0.20,
        "refl_blur": 3.0,
    },
    "deep_garage": {
        "name": "🏢 Deep Garage Bay",
        "bg_offset_x": 0,
        "bg_offset_y": -225,
        "bg_scale": 1.08,
        "bg_floor_y_ratio": 0.38,
        "bg_tilt_deg": 0.0,
        "bg_blur": 2.0,
        "bg_brightness": 0.0,
        "bg_contrast": 1.0,
        "ground_contact_y": 0.91,
        "ground_fine_offset": 20,
        "tyre_contact_left": 0,
        "tyre_contact_right": 0,
        "veh_brightness": -3.0,
        "veh_contrast": 1.06,
        "veh_offset_x": 0,
        "veh_offset_y": 0,
        "veh_scale_mult": 0.88,
        "veh_rotation_deg": 0.0,
        "veh_front_lift": 0.0,
        "veh_rear_lift": 0.0,
        "veh_persp_x": 0.0,
        "veh_persp_y": 0.0,
        "veh_skew_x_deg": 0.0,
        "veh_camber_deg": 0.0,
        "veh_flip_h": False,
        "shadow_contact_strength": 0.95,
        "shadow_underbody_strength": 0.75,
        "shadow_blur": 1.0,
        "shadow_spread": 1.05,
        "shadow_offset_x": 0,
        "shadow_offset_y": 0,
        "shadow_direction_deg": 0.0,
        "refl_opacity": 0.16,
        "refl_blur": 3.0,
    },
    "subtle_matte": {
        "name": "🚗 Subtle Matte Floor",
        "bg_offset_x": 0,
        "bg_offset_y": -200,
        "bg_scale": 1.0,
        "bg_floor_y_ratio": 0.36,
        "bg_tilt_deg": 0.0,
        "bg_blur": 0.0,
        "bg_brightness": 0.0,
        "bg_contrast": 1.0,
        "ground_contact_y": 0.91,
        "ground_fine_offset": 5,
        "tyre_contact_left": 0,
        "tyre_contact_right": 0,
        "veh_brightness": 0.0,
        "veh_contrast": 1.0,
        "veh_offset_x": 0,
        "veh_offset_y": 0,
        "veh_scale_mult": 0.95,
        "veh_rotation_deg": 0.0,
        "veh_front_lift": 0.0,
        "veh_rear_lift": 0.0,
        "veh_persp_x": 0.0,
        "veh_persp_y": 0.0,
        "veh_skew_x_deg": 0.0,
        "veh_camber_deg": 0.0,
        "veh_flip_h": False,
        "shadow_contact_strength": 0.75,
        "shadow_underbody_strength": 0.50,
        "shadow_blur": 1.1,
        "shadow_spread": 0.9,
        "shadow_offset_x": 0,
        "shadow_offset_y": 0,
        "shadow_direction_deg": 0.0,
        "refl_opacity": 0.08,
        "refl_blur": 4.5,
    },
    "sport_dynamic": {
        "name": "🏎️ Sport Dynamic Perspective",
        "bg_offset_x": 0,
        "bg_offset_y": -195,
        "bg_scale": 1.05,
        "bg_floor_y_ratio": 0.63,
        "bg_tilt_deg": -2.5,
        "bg_blur": 1.0,
        "bg_brightness": 0.0,
        "bg_contrast": 1.05,
        "ground_contact_y": 0.91,
        "ground_fine_offset": 16,
        "tyre_contact_left": 2,
        "tyre_contact_right": -2,
        "veh_brightness": -1.0,
        "veh_contrast": 1.08,
        "veh_offset_x": 10,
        "veh_offset_y": 0,
        "veh_scale_mult": 0.91,
        "veh_rotation_deg": 0.0,
        "veh_front_lift": 0.0,
        "veh_rear_lift": 0.0,
        "veh_persp_x": 3.0,
        "veh_persp_y": -1.0,
        "veh_skew_x_deg": 1.0,
        "veh_camber_deg": 1.5,
        "veh_flip_h": False,
        "shadow_contact_strength": 0.90,
        "shadow_underbody_strength": 0.70,
        "shadow_blur": 0.9,
        "shadow_spread": 1.1,
        "shadow_offset_x": 4,
        "shadow_offset_y": 1,
        "shadow_direction_deg": 6.0,
        "refl_opacity": 0.22,
        "refl_blur": 2.5,
    },
}

def _normalize_preset(p: dict) -> dict:
    res = dict(p)
    for snake, camel in SNAKE_TO_CAMEL.items():
        if snake in res and camel not in res:
            res[camel] = res[snake]
        elif camel in res and snake not in res:
            res[snake] = res[camel]
    return res

BUILTIN_PRESETS: dict[str, dict] = {k: _normalize_preset(v) for k, v in _RAW_PRESETS.items()}

class VehicleCompositor:
    """
    Main composition engine supporting full dual interactive controls for Background and Vehicle.
    All presets and defaults are built-in for zero-dependency cross-project reusability.
    """

    DEFAULT_PRESET_NAME = DEFAULT_PRESET_NAME
    DEFAULT_OUTPUT_WIDTH = DEFAULT_OUTPUT_WIDTH
    DEFAULT_OUTPUT_HEIGHT = DEFAULT_OUTPUT_HEIGHT

    @classmethod
    def get_preset(cls, name: str) -> dict:
        """Returns specific preset dictionary by name (falls back to default showroom preset)."""
        import copy
        if name in BUILTIN_PRESETS:
            return copy.deepcopy(BUILTIN_PRESETS[name])
        return copy.deepcopy(BUILTIN_PRESETS[cls.DEFAULT_PRESET_NAME])

    @staticmethod
    def _load_image_bgr(img_input: Any) -> np.ndarray:
        if isinstance(img_input, (str, Path)):
            arr = cv2.imread(str(img_input), cv2.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(f"Could not load image from path: {img_input}")
            return arr
        if hasattr(img_input, "convert"):
            return cv2.cvtColor(np.array(img_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        if isinstance(img_input, np.ndarray):
            if img_input.ndim == 2:
                return cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)
            if img_input.shape[2] == 4:
                return cv2.cvtColor(img_input, cv2.COLOR_BGRA2BGR)
            return img_input.copy()
        raise TypeError(f"Unsupported image input type: {type(img_input)}")

    @classmethod
    def _load_vehicle_input(cls, fg_input: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        existing_alpha = None
        if isinstance(fg_input, (str, Path)):
            arr = cv2.imread(str(fg_input), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise FileNotFoundError(f"Could not load vehicle image from path: {fg_input}")
            if arr.ndim == 3 and arr.shape[2] == 4:
                raw_alpha = arr[..., 3]
                if np.any(raw_alpha < 250):
                    existing_alpha = raw_alpha
                return arr[..., :3].copy(), existing_alpha
            return cls._load_image_bgr(arr), None

        if hasattr(fg_input, "convert"):
            if fg_input.mode in ("RGBA", "LA") or (hasattr(fg_input, "getbands") and "A" in fg_input.getbands()):
                fg_rgba = np.array(fg_input.convert("RGBA"))
                raw_alpha = fg_rgba[..., 3]
                if np.any(raw_alpha < 250):
                    existing_alpha = raw_alpha
                fg_bgr = cv2.cvtColor(fg_rgba[..., :3], cv2.COLOR_RGB2BGR)
                return fg_bgr, existing_alpha
            return cv2.cvtColor(np.array(fg_input.convert("RGB")), cv2.COLOR_RGB2BGR), None

        if isinstance(fg_input, np.ndarray):
            if fg_input.ndim == 3 and fg_input.shape[2] == 4:
                raw_alpha = fg_input[..., 3]
                if np.any(raw_alpha < 250):
                    existing_alpha = raw_alpha
                return fg_input[..., :3].copy(), existing_alpha
            return cls._load_image_bgr(fg_input), None

        raise TypeError(f"Unsupported vehicle input type: {type(fg_input)}")

    @classmethod
    def compose(
        cls,
        bg_image: Union[Image.Image, np.ndarray, str, Path],
        fg_image: Union[Image.Image, np.ndarray, str, Path],
        output_width: int = DEFAULT_OUTPUT_WIDTH,
        output_height: int = DEFAULT_OUTPUT_HEIGHT,
        preset: Optional[str] = None,
        config: Optional[dict] = None,
        # Background Controls
        bg_offset_x: Optional[int] = None,
        bg_offset_y: Optional[int] = None,
        bg_scale: Optional[float] = None,
        bg_floor_y_ratio: Optional[float] = None,
        bg_tilt_deg: Optional[float] = None,
        bg_brightness: Optional[float] = None,
        bg_contrast: Optional[float] = None,
        bg_blur: Optional[float] = None,
        # Vehicle Controls
        veh_offset_x: Optional[int] = None,
        veh_offset_y: Optional[int] = None,
        veh_scale_mult: Optional[float] = None,
        veh_rotation_deg: Optional[float] = None,
        veh_front_lift: Optional[float] = None,
        veh_rear_lift: Optional[float] = None,
        veh_persp_x: Optional[float] = None,
        veh_persp_y: Optional[float] = None,
        veh_skew_x_deg: Optional[float] = None,
        veh_camber_deg: Optional[float] = None,
        veh_flip_h: Optional[bool] = None,
        # Grounding Controls
        ground_contact_y: Optional[float] = None,
        ground_fine_offset: Optional[int] = None,
        tyre_contact_left: Optional[int] = None,
        tyre_contact_right: Optional[int] = None,
        # Shadow Controls
        shadow_contact_strength: Optional[float] = None,
        shadow_underbody_strength: Optional[float] = None,
        shadow_blur: Optional[float] = None,
        shadow_spread: Optional[float] = None,
        shadow_offset_x: Optional[int] = None,
        shadow_offset_y: Optional[int] = None,
        shadow_direction_deg: Optional[float] = None,
        # Reflection Controls
        refl_opacity: Optional[float] = None,
        refl_blur: Optional[float] = None,
        # Lighting Controls
        veh_brightness: Optional[float] = None,
        veh_contrast: Optional[float] = None,
        **kwargs,
    ) -> Image.Image:
        """
        Complete 17-point vehicle compositing pipeline.
        Resolves parameters starting from selected preset, overlaid by optional config dict and explicit arguments.
        """
        # 0. Resolve Base Settings from Preset
        chosen_preset = preset or cls.DEFAULT_PRESET_NAME
        params = cls.get_preset(chosen_preset)

        # Overlay config dictionary if supplied
        if config:
            for k, v in config.items():
                if v is not None:
                    k_snake = CAMEL_TO_SNAKE.get(k, k)
                    params[k_snake] = v

        # Overlay explicit keyword arguments if passed
        explicit_args = {
            "bg_offset_x": bg_offset_x,
            "bg_offset_y": bg_offset_y,
            "bg_scale": bg_scale,
            "bg_floor_y_ratio": bg_floor_y_ratio,
            "bg_tilt_deg": bg_tilt_deg,
            "bg_brightness": bg_brightness,
            "bg_contrast": bg_contrast,
            "bg_blur": bg_blur,
            "veh_offset_x": veh_offset_x,
            "veh_offset_y": veh_offset_y,
            "veh_scale_mult": veh_scale_mult,
            "veh_rotation_deg": veh_rotation_deg,
            "veh_front_lift": veh_front_lift,
            "veh_rear_lift": veh_rear_lift,
            "veh_persp_x": veh_persp_x,
            "veh_persp_y": veh_persp_y,
            "veh_skew_x_deg": veh_skew_x_deg,
            "veh_camber_deg": veh_camber_deg,
            "veh_flip_h": veh_flip_h,
            "ground_contact_y": ground_contact_y,
            "ground_fine_offset": ground_fine_offset,
            "tyre_contact_left": tyre_contact_left,
            "tyre_contact_right": tyre_contact_right,
            "shadow_contact_strength": shadow_contact_strength,
            "shadow_underbody_strength": shadow_underbody_strength,
            "shadow_blur": shadow_blur,
            "shadow_spread": shadow_spread,
            "shadow_offset_x": shadow_offset_x,
            "shadow_offset_y": shadow_offset_y,
            "shadow_direction_deg": shadow_direction_deg,
            "refl_opacity": refl_opacity,
            "refl_blur": refl_blur,
            "veh_brightness": veh_brightness,
            "veh_contrast": veh_contrast,
        }
        for k, v in kwargs.items():
            k_snake = CAMEL_TO_SNAKE.get(k, k)
            explicit_args[k_snake] = v

        for k, v in explicit_args.items():
            if v is not None:
                params[k] = v

        # Extract normalized parameters
        bg_offset_x = int(params["bg_offset_x"])
        bg_offset_y = int(params["bg_offset_y"])
        bg_scale = float(params["bg_scale"])
        bg_floor_y_val = params.get("bg_floor_y_ratio")
        bg_floor_y_ratio = float(bg_floor_y_val) if (bg_floor_y_val is not None and str(bg_floor_y_val).strip() != "") else None
        bg_tilt_deg = float(params["bg_tilt_deg"])
        bg_brightness = float(params["bg_brightness"])
        bg_contrast = float(params["bg_contrast"])
        bg_blur = float(params["bg_blur"])

        veh_offset_x = int(params["veh_offset_x"])
        veh_offset_y = int(params["veh_offset_y"])
        veh_scale_mult = float(params["veh_scale_mult"])
        veh_rotation_deg = float(params.get("veh_rotation_deg", 0.0))
        veh_front_lift = float(params["veh_front_lift"])
        veh_rear_lift = float(params["veh_rear_lift"])
        veh_persp_x = float(params["veh_persp_x"])
        veh_persp_y = float(params["veh_persp_y"])
        veh_skew_x_deg = float(params["veh_skew_x_deg"])
        veh_camber_deg = float(params["veh_camber_deg"])
        veh_flip_h = bool(params["veh_flip_h"])

        ground_contact_y = float(params["ground_contact_y"])
        ground_fine_offset = int(params["ground_fine_offset"])
        tyre_contact_left = int(params["tyre_contact_left"])
        tyre_contact_right = int(params["tyre_contact_right"])

        shadow_contact_strength = float(params["shadow_contact_strength"])
        shadow_underbody_strength = float(params["shadow_underbody_strength"])
        shadow_blur = float(params["shadow_blur"])
        shadow_spread = float(params["shadow_spread"])
        shadow_offset_x = int(params["shadow_offset_x"])
        shadow_offset_y = int(params["shadow_offset_y"])
        shadow_direction_deg = float(params["shadow_direction_deg"])

        refl_opacity = float(params["refl_opacity"])
        refl_blur = float(params["refl_blur"])

        veh_brightness = float(params["veh_brightness"])
        veh_contrast = float(params["veh_contrast"])

        bg_bgr = cls._load_image_bgr(bg_image)
        fg_bgr, existing_alpha = cls._load_vehicle_input(fg_image)

        # 1. Vehicle Segmentation & Analysis
        vehicle_alpha = segment_vehicle(fg_bgr, existing_alpha=existing_alpha)
        veh_info = analyze_vehicle(fg_bgr, vehicle_alpha)

        # 2. Transform Vehicle (Flip, Front/Rear Elevation, 3D Yaw, 3D Pitch, Tone)
        total_rotation = float(veh_info.auto_rotation_deg) + float(veh_rotation_deg)
        vehicle_rgba = np.dstack([fg_bgr, vehicle_alpha])
        vehicle_rgba = decontaminate_edge_colors(vehicle_rgba)
        vehicle_rgba = transform_vehicle(
            vehicle_rgba,
            flip_h=veh_flip_h,
            rotation_deg=total_rotation,
            persp_x=veh_persp_x,
            persp_y=veh_persp_y,
            front_lift_px=veh_front_lift,
            rear_lift_px=veh_rear_lift,
            front_is_left=veh_info.front_is_left,
            skew_x_deg=veh_skew_x_deg,
            camber_deg=veh_camber_deg,
            brightness=veh_brightness,
            contrast=veh_contrast,
        )

        # 3. Initial Placement & Calculate Tyre Ground Contact Points
        resized_rgba, placement = fit_vehicle(
            vehicle_rgba,
            canvas_w=output_width,
            canvas_h=output_height,
            floor_y=int(output_height * 0.55),
            offset_x=veh_offset_x,
            offset_y=veh_offset_y,
            scale_multiplier=veh_scale_mult,
            ground_contact_y_ratio=ground_contact_y,
            ground_fine_offset=ground_fine_offset,
        )

        # Determine highest tyre contact Y in canvas coordinates
        src_bbox = placement.source_bbox
        l_pt, r_pt = veh_info.contact_points
        l_screen_y = placement.y + int((l_pt[1] - src_bbox[1]) * placement.scale)
        r_screen_y = placement.y + int((r_pt[1] - src_bbox[1]) * placement.scale)
        highest_tyre_y = min(l_screen_y, r_screen_y)

        # Calculate vehicle's ground perspective depth angle
        dx = (r_pt[0] - l_pt[0]) * placement.scale
        dy = (r_screen_y - l_screen_y)
        veh_perspective_angle_deg = float(np.degrees(np.arctan2(dy, max(1.0, abs(dx)))))

        # Background Tilt: only applied when explicitly set by user to avoid distortion
        effective_bg_tilt = float(bg_tilt_deg)

        # 4. AI Dynamic Image-Adaptive Background Y-Position Calculation:
        target_seam_y = min(float(output_height * 0.36), float(highest_tyre_y - 160))

        # Dynamic AI offset randomly selected between -190 to -230
        auto_bg_offset_y = random.randint(-230, -190)

        # Effective offset: if default (-210) or 0, use the dynamic random offset;
        # otherwise apply manual user slider adjustments within [-230, -190].
        if bg_offset_y == 0 or bg_offset_y == -210:
            effective_bg_offset_y = auto_bg_offset_y
        else:
            delta = bg_offset_y - (-210)
            effective_bg_offset_y = int(np.clip(auto_bg_offset_y + delta, -230, -190))

        # Transform Background using dynamic image-adapted negative offset
        bg = transform_background(
            bg_bgr,
            output_width,
            output_height,
            offset_x=bg_offset_x,
            offset_y=effective_bg_offset_y,
            scale=bg_scale,
            brightness=bg_brightness,
            contrast=bg_contrast,
            blur_px=bg_blur,
        )

        # 5. Guarantee Floor Line is UPPER than all vehicle tyres
        detected_slope, detected_seam = estimate_floor_line(bg)
        max_allowable_floor_seam = min(highest_tyre_y - 50, int(output_height * 0.42))
        if bg_floor_y_ratio is not None:
            target_floor_seam = float(output_height * bg_floor_y_ratio)
        else:
            target_floor_seam = float(detected_seam)

        floor_seam_y = min(target_floor_seam, float(max_allowable_floor_seam))

        bg, floor_slope, center_seam = warp_background_perspective(
            bg,
            tilt_degrees=effective_bg_tilt,
            floor_seam_y=floor_seam_y,
            floor_seam_info=(detected_slope, detected_seam),
        )

        # 6. Scene Lighting Analysis & Minor Realistic Studio Effects
        scene_info = analyze_scene(bg, floor_seam_info=(detected_slope, detected_seam))

        veh_bgr = resized_rgba[..., :3].copy()
        veh_alpha = resized_rgba[..., 3]

        # Minor, tasteful studio sheen (delicate highlight glint on roof/hood aligned with ceiling lights)
        veh_bgr = apply_directional_sheen(veh_bgr, veh_alpha, scene_info, sheen_strength=0.10)

        # Minor upward ambient bounce on lower rocker panels
        veh_bgr = apply_ground_bounce_light(veh_bgr, veh_alpha, scene_info, bounce_strength=0.08)

        resized_rgba[..., :3] = veh_bgr

        # Update placement floor context
        placement.background_floor_y = int(center_seam)

        # 10. Multi-Layer Grounding Shadows, Reflections & Final Composite
        output_bgr = render_effects_and_composite(
            bg,
            resized_rgba,
            placement,
            scene=scene_info,
            original_shadow_mask=veh_info.original_shadow_mask,
            tyre_contact_left=tyre_contact_left,
            tyre_contact_right=tyre_contact_right,
            shadow_contact_strength=shadow_contact_strength,
            shadow_underbody_strength=shadow_underbody_strength,
            shadow_blur=shadow_blur,
            shadow_spread=shadow_spread,
            shadow_offset_x=shadow_offset_x,
            shadow_offset_y=shadow_offset_y,
            shadow_direction_deg=shadow_direction_deg,
            refl_opacity=refl_opacity,
            refl_blur=refl_blur,
        )

        # Convert back to PIL Image (RGB)
        output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(output_rgb)


# ==========================================
# HIGH-LEVEL PYTHON API
# ==========================================

def compose_images(
    bg_image: Union[str, Path, Image.Image],
    fg_image: Union[str, Path, Image.Image],
    output_path: Optional[Union[str, Path]] = None,
    preset: str = VehicleCompositor.DEFAULT_PRESET_NAME,
    output_width: int = 1024,
    output_height: int = 682,
    output_format: Optional[str] = None,
    # Background controls (optional overrides)
    bg_offset_x: Optional[int] = None,
    bg_offset_y: Optional[int] = None,
    bg_scale: Optional[float] = None,
    bg_floor_y_ratio: Optional[float] = None,
    bg_tilt_deg: Optional[float] = None,
    bg_brightness: Optional[float] = None,
    bg_contrast: Optional[float] = None,
    bg_blur: Optional[float] = None,
    # Vehicle controls (optional overrides)
    veh_offset_x: Optional[int] = None,
    veh_offset_y: Optional[int] = None,
    veh_scale_mult: Optional[float] = None,
    veh_rotation_deg: Optional[float] = None,
    veh_front_lift: Optional[float] = None,
    veh_rear_lift: Optional[float] = None,
    veh_persp_x: Optional[float] = None,
    veh_persp_y: Optional[float] = None,
    veh_skew_x_deg: Optional[float] = None,
    veh_camber_deg: Optional[float] = None,
    veh_flip_h: bool = False,
    # Grounding controls (optional overrides)
    ground_contact_y: Optional[float] = None,
    ground_fine_offset: Optional[int] = None,
    tyre_contact_left: Optional[int] = None,
    tyre_contact_right: Optional[int] = None,
    # Shadow controls (optional overrides)
    shadow_contact_strength: Optional[float] = None,
    shadow_underbody_strength: Optional[float] = None,
    shadow_blur: Optional[float] = None,
    shadow_spread: Optional[float] = None,
    shadow_offset_x: Optional[int] = None,
    shadow_offset_y: Optional[int] = None,
    shadow_direction_deg: Optional[float] = None,
    # Reflection controls (optional overrides)
    refl_opacity: Optional[float] = None,
    refl_blur: Optional[float] = None,
    # Lighting controls (optional overrides)
    veh_brightness: Optional[float] = None,
    veh_contrast: Optional[float] = None,
) -> Dict[str, Any]:
    """
    High-level compositing function.
    Accepts file paths (str/Path) or PIL Image instances.
    Optionally saves output image to output_path and returns execution metadata.
    """
    t0 = time.perf_counter()

    # 1. Load Background
    if isinstance(bg_image, (str, Path)):
        bg_p = Path(bg_image)
        if not bg_p.exists():
            raise FileNotFoundError(f"Background image not found: {bg_p}")
        bg_pil = Image.open(bg_p)
    elif isinstance(bg_image, Image.Image):
        bg_pil = bg_image
    else:
        raise TypeError(f"bg_image must be a file path or PIL Image, got {type(bg_image)}")

    # 2. Load Foreground
    if isinstance(fg_image, (str, Path)):
        fg_p = Path(fg_image)
        if not fg_p.exists():
            raise FileNotFoundError(f"Foreground vehicle image not found: {fg_p}")
        fg_pil = Image.open(fg_p)
    elif isinstance(fg_image, Image.Image):
        fg_pil = fg_image
    else:
        raise TypeError(f"fg_image must be a file path or PIL Image, got {type(fg_image)}")

    # 3. Determine output format
    if output_format:
        ext = output_format.lower().lstrip('.')
    elif output_path:
        ext = Path(output_path).suffix.lower().lstrip('.') or 'png'
    else:
        ext = 'png'

    if ext in ('jpg', 'jpeg'):
        save_format = 'JPEG'
        ext = 'jpg'
    elif ext == 'webp':
        save_format = 'WEBP'
        ext = 'webp'
    else:
        save_format = 'PNG'
        ext = 'png'

    # 4. Execute Computer Vision Compositing
    composed = VehicleCompositor.compose(
        bg_image=bg_pil,
        fg_image=fg_pil,
        output_width=output_width,
        output_height=output_height,
        preset=preset,
        bg_offset_x=bg_offset_x,
        bg_offset_y=bg_offset_y,
        bg_scale=bg_scale,
        bg_floor_y_ratio=bg_floor_y_ratio,
        bg_tilt_deg=bg_tilt_deg,
        bg_brightness=bg_brightness,
        bg_contrast=bg_contrast,
        bg_blur=bg_blur,
        veh_offset_x=veh_offset_x,
        veh_offset_y=veh_offset_y,
        veh_scale_mult=veh_scale_mult,
        veh_rotation_deg=veh_rotation_deg,
        veh_front_lift=veh_front_lift,
        veh_rear_lift=veh_rear_lift,
        veh_persp_x=veh_persp_x,
        veh_persp_y=veh_persp_y,
        veh_skew_x_deg=veh_skew_x_deg,
        veh_camber_deg=veh_camber_deg,
        veh_flip_h=veh_flip_h,
        ground_contact_y=ground_contact_y,
        ground_fine_offset=ground_fine_offset,
        tyre_contact_left=tyre_contact_left,
        tyre_contact_right=tyre_contact_right,
        shadow_contact_strength=shadow_contact_strength,
        shadow_underbody_strength=shadow_underbody_strength,
        shadow_blur=shadow_blur,
        shadow_spread=shadow_spread,
        shadow_offset_x=shadow_offset_x,
        shadow_offset_y=shadow_offset_y,
        shadow_direction_deg=shadow_direction_deg,
        refl_opacity=refl_opacity,
        refl_blur=refl_blur,
        veh_brightness=veh_brightness,
        veh_contrast=veh_contrast,
    )

    # 5. Save Output
    saved_path_str: Optional[str] = None
    if output_path:
        out_p = Path(output_path)
        if out_p.suffix == '':
            out_p = out_p.with_suffix(f".{ext}")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if save_format == 'JPEG':
            composed.convert('RGB').save(out_p, format='JPEG', quality=95)
        elif save_format == 'WEBP':
            composed.save(out_p, format='WEBP', quality=95)
        else:
            composed.save(out_p, format=save_format, compress_level=3)
        saved_path_str = str(out_p.resolve())

    trim_memory()

    process_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "success": True,
        "output_path": saved_path_str,
        "width": composed.width,
        "height": composed.height,
        "format": ext,
        "preset": preset,
        "process_time_ms": process_time_ms,
        "image": composed,
    }


# ==========================================
# COMMAND LINE INTERFACE (CLI)
# ==========================================

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="python addBg.py",
        usage="python addBg.py --input <path-to-vehicle-cutout> --output <path-to-output> --background <path-to-background> [options]",
        description="Vehicle Overlay Image Compositor — addBg.py CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core Required Arguments
    parser.add_argument(
        "--input", "-i", "--fg", "--fg-image",
        dest="input_image",
        required=True,
        help="Path to vehicle cutout image with alpha transparency (PNG, WEBP)",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output",
        default=None,
        help="Path to output image destination (default: ./composite_<uuid>.<format>)",
    )
    parser.add_argument(
        "--background", "-b", "--bg", "--bg-image",
        dest="bg_image",
        required=True,
        help="Path to background scene photo (JPG, PNG, WEBP)",
    )
    parser.add_argument("--format", dest="format", choices=["png", "jpeg", "jpg", "webp"], default=None, help="Output image format (defaults to file extension in --output or png)")
    parser.add_argument("--width", "--output-width", dest="width", type=int, default=1024, help="Output width in pixels")
    parser.add_argument("--height", "--output-height", dest="height", type=int, default=682, help="Output height in pixels")

    # Preset
    parser.add_argument(
        "--preset",
        dest="preset",
        choices=["default_grounded", "deep_garage", "subtle_matte", "sport_dynamic"],
        default="default_grounded",
        help="Compositing preset style",
    )

    # Execution & formatting flags
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON to stdout (ideal for Node.js child_process)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress informational/progress logs")

    # Optional fine-tuning controls
    # Background
    parser.add_argument("--bg-offset-x", type=int, default=None, help="Shift background horizontally in pixels")
    parser.add_argument("--bg-offset-y", type=int, default=None, help="Shift background vertically in pixels")
    parser.add_argument("--bg-scale", type=float, default=None, help="Background zoom multiplier")
    parser.add_argument("--bg-brightness", type=float, default=None, help="Background brightness adjustment")
    parser.add_argument("--bg-contrast", type=float, default=None, help="Background contrast adjustment")
    parser.add_argument("--bg-blur", type=float, default=None, help="Background blur radius")

    # Vehicle
    parser.add_argument("--veh-offset-x", type=int, default=None, help="Shift vehicle horizontally in pixels")
    parser.add_argument("--veh-offset-y", type=int, default=None, help="Shift vehicle vertically in pixels")
    parser.add_argument("--veh-scale", type=float, default=None, dest="veh_scale_mult", help="Vehicle scale multiplier")
    parser.add_argument("--veh-rotation", type=float, default=None, dest="veh_rotation_deg", help="Vehicle rotation angle in degrees")
    parser.add_argument("--veh-flip-h", action="store_true", help="Mirror/flip vehicle horizontally")

    # Grounding & Shadows
    parser.add_argument("--ground-contact-y", type=float, default=None, help="Explicit ground line Y coordinate")
    parser.add_argument("--ground-offset", type=int, default=None, dest="ground_fine_offset", help="Fine ground offset adjustment in pixels")
    parser.add_argument("--shadow-contact", type=float, default=None, dest="shadow_contact_strength", help="Contact shadow strength (0.0 - 1.0)")
    parser.add_argument("--shadow-underbody", type=float, default=None, dest="shadow_underbody_strength", help="Underbody shadow strength (0.0 - 1.0)")
    parser.add_argument("--shadow-blur", type=float, default=None, help="Shadow blur radius")
    parser.add_argument("--shadow-spread", type=float, default=None, help="Shadow lateral and forward spread multiplier")
    parser.add_argument("--shadow-direction", type=float, default=None, dest="shadow_direction_deg", help="Shadow direction angle in degrees")
    parser.add_argument("--shadow-offset-x", type=int, default=None, help="Shadow horizontal offset in pixels")
    parser.add_argument("--shadow-offset-y", type=int, default=None, help="Shadow vertical offset in pixels")

    # Floor Reflections
    parser.add_argument("--refl-opacity", type=float, default=None, help="Floor reflection opacity (0.0 - 1.0)")
    parser.add_argument("--refl-blur", type=float, default=None, help="Floor reflection blur radius")

    return parser.parse_args(args)


def main():
    args = parse_args()

    # Configure logging
    if args.quiet or args.json:
        logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        # Determine output path
        output_path = args.output
        if not output_path:
            ext = args.format.lower().lstrip('.') if args.format else 'png'
            file_uuid = uuid.uuid4().hex[:8]
            output_path = f"composite_{file_uuid}.{ext}"

        if not args.quiet and not args.json:
            print(f"\n=======================================================")
            print(f"🚗 Vehicle Overlay Image Compositor — addBg.py")
            print(f"=======================================================")
            print(f"   • Input (Cutout): {args.input_image}")
            print(f"   • Background:     {args.bg_image}")
            print(f"   • Output:         {output_path}")
            print(f"   • Preset:         {args.preset}")
            print(f"   • Resolution:     {args.width}x{args.height}")
            print(f"=======================================================\n")
            print("⏳ Processing composite...")

        _rt = _ResourceTracker() if _ResourceTracker is not None else None
        _rt_ctx = _rt if _rt is not None else _NullContext()
        with _rt_ctx:
            result = compose_images(
                bg_image=args.bg_image,
                fg_image=args.input_image,
                output_path=output_path,
                preset=args.preset,
                output_width=args.width,
                output_height=args.height,
                output_format=args.format,
                bg_offset_x=args.bg_offset_x,
                bg_offset_y=args.bg_offset_y,
                bg_scale=args.bg_scale,
                bg_brightness=args.bg_brightness,
                bg_contrast=args.bg_contrast,
                bg_blur=args.bg_blur,
                veh_offset_x=args.veh_offset_x,
                veh_offset_y=args.veh_offset_y,
                veh_scale_mult=args.veh_scale_mult,
                veh_rotation_deg=args.veh_rotation_deg,
                veh_flip_h=args.veh_flip_h,
                ground_contact_y=args.ground_contact_y,
                ground_fine_offset=args.ground_fine_offset,
                shadow_contact_strength=args.shadow_contact_strength,
                shadow_underbody_strength=args.shadow_underbody_strength,
                shadow_blur=args.shadow_blur,
                shadow_spread=args.shadow_spread,
                shadow_direction_deg=args.shadow_direction_deg,
                shadow_offset_x=args.shadow_offset_x,
                shadow_offset_y=args.shadow_offset_y,
                refl_opacity=args.refl_opacity,
                refl_blur=args.refl_blur,
            )

        if args.json:
            json_output = {
                "success": True,
                "output_path": result["output_path"],
                "width": result["width"],
                "height": result["height"],
                "format": result["format"],
                "preset": result["preset"],
                "process_time_ms": result["process_time_ms"],
            }
            if _rt is not None:
                json_output["resource_metrics"] = {
                    "elapsed_sec": _rt.elapsed_sec,
                    "peak_ram_mb": _rt.peak_ram_mb,
                    "cpu_percent": _rt.cpu_percent,
                }
            print(json.dumps(json_output, indent=2))
        else:
            print(f"✅ Composite generated successfully in {result['process_time_ms']} ms!")
            print(f"📁 Saved to: {result['output_path']}\n")

        sys.exit(0)

    except Exception as e:
        _notify(
            "addBg.compose_images",
            e,
            ref_id=Path(args.input_image).stem if getattr(args, "input_image", None) else None,
            extra=f"bg={Path(args.bg_image).stem if getattr(args, 'bg_image', None) else 'unknown'}",
        )
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"\n\u274c Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()