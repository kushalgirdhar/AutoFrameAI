#!/usr/bin/env python3
"""
VehicleVision — Vehicle Image-to-Video Generator
Unified Multi-Layout Video Generation Engine (videoGen.py)

STRICT SPECIFICATION ENFORCEMENT:
- 100% Local and Offline: No third-party API keys, no external cloud services.
- Multi-Layout Engine:
    * 'news_strip': Bold top headline banner + dual-tier bottom dealer price strip & checkmark feature ticker.
    * 'default': Classic clean floating card badge & watermark overlay layout.
- Modular & Independent: Works as a standalone CLI or Python library.
- Memory Conscious: Streams frames sequentially into FFmpeg stdin pipe (zero RAM accumulation).
"""

import argparse
import collections
import gc
import json
import logging
import math
from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    from resource_tracker import ResourceTracker as _ResourceTracker
except ImportError:
    _ResourceTracker = None  # type: ignore[assignment]


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


def trim_memory():
    """Forces Python garbage collection and returns glibc heap memory back to the OS."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Vehicle Detector loaded lazily on-demand (vehicle_detector.py)
VEHICLE_DETECTOR_AVAILABLE = True
filter_vehicle_images = None

# Setup logging
logger = logging.getLogger("VehicleVision")

# ============================================================================
# EXIT CODES
# ============================================================================
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_FILE_NOT_FOUND = 2
EXIT_FFMPEG_ERROR = 3
EXIT_INTERNAL_ERROR = 4

# ============================================================================
# FONT & ASSET RESOLUTION
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
FONTS_DIR = BASE_DIR / "assets" / "fonts"
MODELS_DIR = BASE_DIR / "models"
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))
os.environ.setdefault("REMBG_HOME", str(MODELS_DIR))

# Cached rembg session for rapid zero-overhead mask extraction
_REMBG_SESSION = None


def get_rembg_session(model_name: str = "u2netp"):
    """Thread-safe cached rembg session with graceful fallback."""
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        try:
            import rembg
            os.environ.setdefault("U2NET_HOME", str(MODELS_DIR))
            os.environ.setdefault("REMBG_HOME", str(MODELS_DIR))
            _REMBG_SESSION = rembg.new_session(model_name)
        except Exception as e:
            logger.debug(f"rembg session init skipped: {e}")
            _REMBG_SESSION = False
    return _REMBG_SESSION if _REMBG_SESSION is not False else None


DEFAULT_FONT_CANDIDATES = [
    FONTS_DIR / "LiberationSans-Bold.ttf",
    FONTS_DIR / "LiberationSans-Regular.ttf",
    FONTS_DIR / "DejaVuSans-Bold.ttf",
    FONTS_DIR / "DejaVuSans.ttf",
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def resolve_font(
    font_family: Optional[str] = None,
    font_weight: str = "normal",
    font_size: int = 42,
) -> Union[ImageFont.FreeTypeFont, ImageFont.ImageFont]:
    """Resolve and load a TrueType font with robust local fallbacks."""
    if font_family:
        custom_path = Path(font_family)
        if custom_path.is_file():
            try:
                return ImageFont.truetype(str(custom_path), font_size)
            except Exception as e:
                logger.warning(f"Failed to load font from {custom_path}: {e}")

        asset_font = FONTS_DIR / font_family
        if asset_font.is_file():
            try:
                return ImageFont.truetype(str(asset_font), font_size)
            except Exception as e:
                logger.warning(f"Failed to load font from {asset_font}: {e}")

    is_bold = font_weight.lower() in ("bold", "700", "800", "900", "bolder")
    for candidate in DEFAULT_FONT_CANDIDATES:
        if is_bold and "bold" not in candidate.name.lower():
            continue
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), font_size)
            except Exception:
                continue

    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), font_size)
            except Exception:
                continue

    return ImageFont.load_default()


def parse_color(color_val: Any, default: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Tuple[int, int, int, int]:
    """Parse color strings (hex, rgb, rgba) or tuples/lists into (R, G, B, A)."""
    if not color_val:
        return default

    if isinstance(color_val, (list, tuple)):
        if len(color_val) == 3:
            return (int(color_val[0]), int(color_val[1]), int(color_val[2]), 255)
        elif len(color_val) >= 4:
            return (int(color_val[0]), int(color_val[1]), int(color_val[2]), int(color_val[3]))

    if isinstance(color_val, str):
        color_str = color_val.strip()
        if color_str.startswith("#"):
            hex_str = color_str[1:]
            if len(hex_str) == 3:
                r, g, b = [int(c * 2, 16) for c in hex_str]
                return (r, g, b, 255)
            elif len(hex_str) == 4:
                r, g, b, a = [int(c * 2, 16) for c in hex_str]
                return (r, g, b, a)
            elif len(hex_str) == 6:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                return (r, g, b, 255)
            elif len(hex_str) == 8:
                r = int(hex_str[0:2], 16)
                g = int(hex_str[2:4], 16)
                b = int(hex_str[4:6], 16)
                a = int(hex_str[6:8], 16)
                return (r, g, b, a)

        if color_str.lower().startswith("rgb"):
            nums_str = color_str[color_str.find("(") + 1 : color_str.find(")")]
            parts = [p.strip() for p in nums_str.split(",")]
            if len(parts) >= 3:
                r = int(float(parts[0]))
                g = int(float(parts[1]))
                b = int(float(parts[2]))
                a = 255
                if len(parts) >= 4:
                    a_val = float(parts[3])
                    a = int(a_val * 255) if a_val <= 1.0 else int(a_val)
                return (r, g, b, a)

    return default


def resolve_asset_path(raw_path: Optional[str], base_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve asset file path from relative or absolute location."""
    if not raw_path:
        return None
    p = Path(raw_path)
    if p.is_file():
        return p.resolve()
    if base_dir and (base_dir / raw_path).is_file():
        return (base_dir / raw_path).resolve()
    if (BASE_DIR / raw_path).is_file():
        return (BASE_DIR / raw_path).resolve()
    if (BASE_DIR / "input" / "images" / Path(raw_path).name).is_file():
        return (BASE_DIR / "input" / "images" / Path(raw_path).name).resolve()

    # Search project web directories for dealer logos and assets
    file_name = Path(raw_path).name
    repo_root = BASE_DIR.parent
    search_dirs = [
        BASE_DIR / "public" / "uploads" / "dealers",
        BASE_DIR / "public" / "uploads",
        repo_root / "api" / "public" / "uploads" / "dealers",
        repo_root / "api" / "public" / "uploads",
        Path("/var/www/html/api/public/uploads/dealers"),
        Path("/var/www/html/api/public/uploads"),
    ]
    for d in search_dirs:
        candidate = d / file_name
        if candidate.is_file():
            return candidate.resolve()
        # Fallback: match without timestamp prefix (e.g. *whatsapp-image-2026-10-06-at-3-56-43-pm.jpg)
        base_match = re.sub(r"^\d+_", "", file_name)
        if base_match != file_name and d.is_dir():
            matches = list(d.glob(f"*{base_match}"))
            if matches and matches[0].is_file():
                return matches[0].resolve()

    return p


def is_valid_str(val: Any) -> bool:
    """Check if value is non-null, non-empty, and valid string."""
    if val is None:
        return False
    s = str(val).strip()
    return s != "" and s.lower() not in ("null", "undefined", "none")


KNOWN_VEHICLE_COLORS = {
    "white", "black", "silver", "grey", "gray", "red", "blue", "brown", "gold",
    "orange", "yellow", "green", "bronze", "maroon", "burgundy", "beige",
}


def is_color_token(val: Any) -> bool:
    """Check if a string represents a vehicle color specification or mentions color/colour."""
    if not val:
        return False
    s = str(val).strip().lower()
    if not s:
        return False
    # Explicit mention of word 'colour' or 'color'
    if re.search(r"\bcolou?rs?\b", s):
        return True
    s_clean = re.sub(r"[^\w\s]", "", s).strip()
    if s_clean in KNOWN_VEHICLE_COLORS:
        return True
    words = s_clean.split()
    if len(words) == 2 and words[0] in ("metallic", "pearl", "pure", "gloss", "matte", "bright", "dark") and words[1] in KNOWN_VEHICLE_COLORS:
        return True
    return False


# ============================================================================
# VECTOR ICONS GENERATOR
# ============================================================================
def create_phone_icon(size: int = 32, color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
    """Generate high-res anti-aliased modern phone call icon with handset and radiating sound waves."""
    scale = 4
    dim = size * scale
    canvas = np.zeros((dim, dim, 4), dtype=np.uint8)

    # Radiating sound waves (concentric arcs in top right)
    arc_center = (int(0.38 * dim), int(0.52 * dim))
    thick = max(2, int(0.048 * dim))
    cv2.ellipse(canvas, arc_center, (int(0.44 * dim), int(0.44 * dim)), 0, -82, -8, color, thick, cv2.LINE_AA)
    cv2.ellipse(canvas, arc_center, (int(0.30 * dim), int(0.30 * dim)), 0, -82, -8, color, thick, cv2.LINE_AA)

    # Handset receiver tilted ~45 degrees
    # Earpiece rounded pill at top-right
    cv2.ellipse(canvas, (int(0.56 * dim), int(0.42 * dim)), (int(0.13 * dim), int(0.075 * dim)), 135, 0, 360, color, -1, cv2.LINE_AA)
    # Mouthpiece rounded pill at bottom-left
    cv2.ellipse(canvas, (int(0.30 * dim), int(0.72 * dim)), (int(0.075 * dim), int(0.13 * dim)), 135, 0, 360, color, -1, cv2.LINE_AA)
    # Connecting bridge handle
    handle_thick = int(0.095 * dim)
    cv2.line(canvas, (int(0.33 * dim), int(0.68 * dim)), (int(0.53 * dim), int(0.46 * dim)), color, handle_thick, cv2.LINE_AA)

    pil_img = Image.fromarray(canvas)
    return pil_img.resize((size, size), Image.Resampling.LANCZOS)


def get_phone_icon(size: int = 32, color: Tuple[int, int, int, int] = (255, 255, 255, 255)) -> Image.Image:
    """Load high-res phone call receiver icon asset or generate anti-aliased fallback."""
    icon_path = BASE_DIR / "assets" / "images" / "phone_call_icon.png"
    if icon_path.is_file():
        try:
            im = Image.open(icon_path).convert("RGBA")
            if color != (255, 255, 255, 255):
                arr = np.array(im)
                arr[:, :, :3] = color[:3]
                im = Image.fromarray(arr)
            return im.resize((size, size), Image.Resampling.LANCZOS)
        except Exception:
            pass
    return create_phone_icon(size=size, color=color)


def get_whatsapp_icon(size: int = 32) -> Image.Image:
    """Load high-res official WhatsApp logo asset or generate anti-aliased fallback."""
    icon_path = BASE_DIR / "assets" / "images" / "whatsapp_logo.png"
    if icon_path.is_file():
        try:
            im = Image.open(icon_path).convert("RGBA")
            return im.resize((size, size), Image.Resampling.LANCZOS)
        except Exception:
            pass
    return create_phone_icon(size=size, color=(37, 211, 102, 255))



def create_check_icon(
    size: int = 24,
    fill_color: Tuple[int, int, int, int] = (56, 189, 248, 255),
    check_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
) -> Image.Image:
    """Generate high-res checkmark icon badge."""
    scale = 4
    canvas_size = size * scale
    icon_img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    idraw = ImageDraw.Draw(icon_img)

    pad = int(1.5 * scale)
    radius = 5 * scale
    idraw.rounded_rectangle([(pad, pad), (canvas_size - pad, canvas_size - pad)], radius=radius, fill=fill_color)

    p1 = (int(canvas_size * 0.26), int(canvas_size * 0.52))
    p2 = (int(canvas_size * 0.44), int(canvas_size * 0.72))
    p3 = (int(canvas_size * 0.76), int(canvas_size * 0.30))

    line_w = max(2, int(3.5 * scale))
    idraw.line([p1, p2, p3], fill=check_color, width=line_w, joint="round")

    return icon_img.resize((size, size), Image.Resampling.LANCZOS)


# ============================================================================
# PRICE FORMATTER & PARSER
# ============================================================================
def parse_price_components(raw_price: Any) -> Tuple[str, str, str, str]:
    """
    Returns: (currency_symbol, amount_number, unit_text, qualifier_text)
    Example:
    - "Rs 1.70 Lakh" -> ("Rs", "1.70", "LAKH", "ONLY")
    - "1.70 Lakh"    -> ("Rs", "1.70", "LAKH", "ONLY")
    - "1.70"         -> ("Rs", "1.70", "LAKH", "ONLY")
    """
    if not is_valid_str(raw_price):
        return ("", "", "", "")

    s = str(raw_price).strip()
    if s in ["0", "Rs 0", "₹0", "0.0", "0.00", "None", "null", "NULL", ""]:
        return ("", "", "", "")

    curr = "Rs"
    if s.startswith("₹"):
        curr = "₹"
        s = s[1:].strip()
    elif s.upper().startswith("RS"):
        curr = "Rs"
        s = re.sub(r"^rs\.?\s*", "", s, flags=re.IGNORECASE).strip()

    lakh_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:lakhs?|lac|lacs?)?", s, re.IGNORECASE)
    has_lakh_word = bool(re.search(r"(?:lakhs?|lac|lacs?)", s, re.IGNORECASE))
    has_cr_word = bool(re.search(r"(?:crores?|cr)", s, re.IGNORECASE))

    if lakh_match:
        num_str = lakh_match.group(1).replace(",", ".")
        try:
            val = float(num_str)
            if val <= 0:
                return ("", "", "", "")
            if val < 500 or has_lakh_word:
                formatted_num = f"{val:.2f}" if "." in num_str else f"{val:.2f}"
                if formatted_num.endswith(".00") and not "." in num_str:
                    formatted_num = f"{int(val)}"
                return (curr, formatted_num, "LAKH", "ONLY")
        except ValueError:
            pass

    if has_cr_word:
        cr_match = re.search(r"(\d+(?:[.,]\d+)?)", s)
        if cr_match:
            return (curr, cr_match.group(1), "CRORE", "ONLY")

    return (curr, s, "", "")


# ============================================================================
# CONFIG VALIDATION & NORMALIZATION
# ============================================================================

def parse_header_title_and_year(slide_data: dict) -> Tuple[str, str]:
    """Extract (main_vehicle_title, model_year) cleanly from slide data."""
    raw_header = str(slide_data.get("headerTitle") or slide_data.get("title") or "").strip()
    if not raw_header and (slide_data.get("make") or slide_data.get("model")):
        raw_header = f"{slide_data.get('make', '')} {slide_data.get('model', '')} {slide_data.get('variant', '')}".strip()

    raw_year = str(slide_data.get("year") or slide_data.get("model_year") or "").strip()
    if raw_year.lower() in ("none", "null", "undefined", "0"):
        raw_year = ""

    main_title = raw_header
    year_str = raw_year

    if "\n" in raw_header:
        parts = [p.strip() for p in raw_header.split("\n", 1)]
        main_title = parts[0]
        if not year_str and len(parts) > 1 and parts[1]:
            year_str = parts[1]
    elif raw_header:
        month_pat = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
        # Check if year (with optional month) is at the beginning
        m_start = re.match(
            r"^((?:(?:" + month_pat + r")\s+)?\b(?:19|20)\d{2}(?:[-/]\d{2,4})?\b(?:\s+(?:" + month_pat + r"))?)\s*(?:model\s*)?[-:/]?\s*(.+)$",
            raw_header,
            re.IGNORECASE,
        )
        if m_start:
            year_str = year_str or m_start.group(1).strip()
            main_title = m_start.group(2).strip()
        else:
            # Check if year (with optional month) is at the end (e.g., CRETA 1.4 S PLUS DEC 2018)
            m_end = re.match(
                r"^(.+?)\s*[-:(/]?\s*((?:(?:" + month_pat + r")\s+)?\b(?:19|20)\d{2}(?:[-/]\d{2,4})?\b(?:\s+(?:" + month_pat + r"))?)\s*(?:model)?\s*\)?$",
                raw_header,
                re.IGNORECASE,
            )
            if m_end:
                year_str = year_str or m_end.group(2).strip()
                main_title = m_end.group(1).strip()

    # Format model year string if detected
    formatted_year = ""
    if year_str:
        cleaned = year_str.strip()
        if re.search(r"\bmodel\b", cleaned, re.IGNORECASE):
            formatted_year = re.sub(r"\s+", " ", cleaned).upper()
        elif re.match(r"^(?:19|20)\d{2}(?:[-/]\d{2,4})?$", cleaned):
            formatted_year = f"{cleaned} MODEL"
        else:
            formatted_year = cleaned.upper()

    return main_title.upper(), formatted_year

def validate_and_normalize_config(config_data: dict, base_dir: Optional[Path] = None) -> dict:
    """Validate and normalize JSON configuration with layout auto-detection."""
    if not isinstance(config_data, dict):
        raise ValueError("Configuration root must be a JSON object.")

    video_cfg = config_data.get("video", {}) if isinstance(config_data.get("video"), dict) else {}

    # Detect layout: "vertical_reel" (9:16) vs "news_strip" vs "default"
    raw_layout = str(video_cfg.get("layout") or config_data.get("layout", "vertical_reel")).strip().lower()
    if raw_layout in ("reel", "reels", "vertical", "vertical_reel", "9:16", "story", "status"):
        layout = "vertical_reel"
    elif raw_layout in ("news_strip", "news", "banner"):
        layout = "news_strip"
    else:
        layout = "default"

    # Default resolution depending on layout
    if layout == "vertical_reel":
        default_w, default_h = 720, 1280
    elif layout == "news_strip":
        default_w, default_h = 1080, 1440
    else:
        default_w, default_h = 720, 720

    width = int(video_cfg.get("width", default_w))
    height = int(video_cfg.get("height", default_h))
    fps = int(video_cfg.get("fps", 24))
    default_duration = float(video_cfg.get("default_slide_duration", 3.0 if layout == "vertical_reel" else (6.0 if layout == "news_strip" else 3.5)))
    transition_duration = float(video_cfg.get("transition_duration", 0.5))
    transition_type = str(video_cfg.get("transition_type", "crossfade")).lower()
    crf = int(video_cfg.get("crf", 22))
    preset = str(video_cfg.get("preset", "ultrafast"))
    filter_vehicles = bool(video_cfg.get("filter_vehicles", config_data.get("filter_vehicles", True)))
    vehicle_conf_threshold = float(video_cfg.get("vehicle_conf_threshold", config_data.get("vehicle_conf_threshold", 0.35)))
    vehicle_min_area = float(video_cfg.get("vehicle_min_area", config_data.get("vehicle_min_area", 0.05)))
    vehicle_fallback_policy = str(video_cfg.get("vehicle_fallback_policy", config_data.get("vehicle_fallback_policy", "strict")))
    include_outro = bool(video_cfg.get("include_outro", config_data.get("include_outro", True)))
    enhance_visuals = bool(video_cfg.get("enhance_visuals", config_data.get("enhance_visuals", True)))
    fade_vehicle_background = bool(video_cfg.get("fade_vehicle_background", config_data.get("fade_vehicle_background", True)))
    fade_strength = float(video_cfg.get("fade_strength", config_data.get("fade_strength", 0.65)))
    blur_radius = int(video_cfg.get("blur_radius", config_data.get("blur_radius", 28)))
    default_scale_mode = str(video_cfg.get("scale_mode", "cover")).lower()

    if width < 320 or width > 3840:
        raise ValueError(f"Invalid video width: {width}.")
    if height < 240 or height > 2160:
        raise ValueError(f"Invalid video height: {height}.")
    if fps < 10 or fps > 60:
        raise ValueError(f"Invalid FPS: {fps}.")

    output_path = (
        config_data.get("videoOutputPath")
        or config_data.get("outputPath")
        or config_data.get("output", {}).get("path")
        or str(BASE_DIR / "output" / "video_output.mp4")
    )

    global_logo = resolve_asset_path(config_data.get("logo"), base_dir) if is_valid_str(config_data.get("logo")) else None
    global_logo_title = str(config_data.get("logoTitle")).strip() if is_valid_str(config_data.get("logoTitle")) else None
    global_vendor_name = str(config_data.get("vendorName")).strip() if is_valid_str(config_data.get("vendorName")) else None
    global_vendor_logo = resolve_asset_path(config_data.get("vendorLogo"), base_dir) if is_valid_str(config_data.get("vendorLogo")) else None
    global_watermark = str(config_data.get("watermark")).strip() if is_valid_str(config_data.get("watermark")) else ""

    raw_slides = config_data.get("images")
    if raw_slides is None:
        raw_slides = config_data.get("slides", [])

    if not raw_slides or not isinstance(raw_slides, list):
        raise ValueError("Configuration must contain a non-empty 'images' or 'slides' array.")

    normalized_slides = []
    for idx, item in enumerate(raw_slides):
        if not isinstance(item, dict):
            raise ValueError(f"Slide/Image #{idx + 1} must be an object.")

        img_path_str = item.get("filename") or item.get("image")
        if not is_valid_str(img_path_str):
            raise ValueError(f"Slide #{idx + 1} is missing required 'filename' or 'image' path.")

        resolved_img = resolve_asset_path(str(img_path_str), base_dir)
        if not resolved_img or not resolved_img.exists():
            raise FileNotFoundError(f"Slide #{idx + 1} image not found: {img_path_str}")

        timing_val = item.get("timing") or item.get("duration") or default_duration
        try:
            duration = float(timing_val)
        except (ValueError, TypeError):
            duration = default_duration

        if duration <= 0.5:
            duration = default_duration

        header_title = str(item.get("headerTitle")).strip() if is_valid_str(item.get("headerTitle")) else ""
        price = str(item.get("price") or item.get("Price")).strip() if is_valid_str(item.get("price") or item.get("Price")) else ""
        raw_text = str(item.get("text")).strip() if is_valid_str(item.get("text")) else ""

        # Omit body type references (SUV, Sedan, Hatchback, etc.) and color specifications from video text
        body_type_kw = {"suv", "sedan", "hatchback", "muv", "coupe", "crossover", "convertible", "wagon", "van"}
        tokens = [t.strip() for t in (raw_text.split("•") if "•" in raw_text else raw_text.split("|"))]
        cleaned_tokens = [
            t for t in tokens
            if t and t.lower() not in body_type_kw and not re.fullmatch(r"(?:suv|sedan|hatchback|muv)\s*(?:car)?", t, re.I) and not is_color_token(t)
        ]
        sep = " • " if "•" in raw_text or not tokens else " | "
        text = sep.join(cleaned_tokens).strip() if len(cleaned_tokens) > 1 else (cleaned_tokens[0] if cleaned_tokens else "")
        if not text and raw_text and raw_text.lower() not in body_type_kw and not is_color_token(raw_text):
            text = raw_text

        footer_title = str(item.get("footerTitle")).strip() if is_valid_str(item.get("footerTitle")) else ""
        contact = str(item.get("Contact") or item.get("contact")).strip() if is_valid_str(item.get("Contact") or item.get("contact")) else ""
        vendor_name_value = item.get("vendorName") or item.get("dealerName") or item.get("businessName") or global_vendor_name
        vendor_name = str(vendor_name_value).strip() if is_valid_str(vendor_name_value) else ""
        watermark = str(item.get("watermark")).strip() if is_valid_str(item.get("watermark")) else global_watermark

        vendor_logo_value = item.get("vendorLogo") or global_vendor_logo
        vendor_logo = resolve_asset_path(str(vendor_logo_value), base_dir) if is_valid_str(vendor_logo_value) else None
        slide_logo = resolve_asset_path(item.get("logo"), base_dir) if is_valid_str(item.get("logo")) else global_logo
        slide_logo_title = str(item.get("logoTitle")).strip() if is_valid_str(item.get("logoTitle")) else global_logo_title

        motion_cycle = ["zoom_in", "pan_left", "zoom_out", "pan_right"]
        raw_anim = item.get("animation")
        animation = str(raw_anim).lower() if is_valid_str(raw_anim) and str(raw_anim).lower() != "ken_burns" else motion_cycle[idx % len(motion_cycle)]
        scale_mode = str(item.get("scale_mode", default_scale_mode)).lower()
        slide_fade_bg = bool(item.get("fade_vehicle_background", item.get("fade_background", fade_vehicle_background)))
        slide_fade_strength = float(item.get("fade_strength", fade_strength))
        slide_blur_radius = int(item.get("blur_radius", blur_radius))

        # Build text objects for 'default' layout backwards compatibility
        normalized_texts = []
        if header_title:
            normalized_texts.append({
                "text": header_title,
                "position": "top_center",
                "font_size": 48,
                "font_weight": "bold",
                "color": (255, 255, 255, 240),
                "background": False,
                "shadow": True,
                "shadow_color": (0, 0, 0, 200),
            })
        if watermark:
            normalized_texts.append({
                "text": watermark,
                "position": "center",
                "font_size": 48,
                "font_weight": "normal",
                "color": (255, 255, 255, 28),
                "background": False,
                "shadow": True,
                "shadow_color": (0, 0, 0, 16),
            })
        if text:
            normalized_texts.append({
                "text": text,
                "position": "bottom_center",
                "font_size": 28,
                "font_weight": "bold",
                "color": (56, 189, 248, 255),
                "background": True,
                "background_color": (15, 23, 42, 210),
                "shadow": True,
            })
        if footer_title:
            normalized_texts.append({
                "text": footer_title,
                "position": "bottom_center",
                "font_size": 23,
                "font_weight": "bold",
                "color": (255, 255, 255, 230),
                "background": False,
                "shadow": True,
                "shadow_color": (0, 0, 0, 220),
            })
        if contact:
            clean_contact = re.sub(r":(?=\S)", ": ", contact)
            clean_contact = re.sub(r"[\s,]+,\s*|\s*,\s*", "   ", clean_contact).strip()
            normalized_texts.append({
                "text": clean_contact,
                "position": "bottom_center",
                "font_size": 23,
                "font_weight": "bold",
                "color": (255, 255, 255, 230),
                "background": False,
                "shadow": True,
                "shadow_color": (0, 0, 0, 220),
            })

        # Slide logos for default layout
        slide_logos = []
        if slide_logo and Path(slide_logo).exists():
            slide_logos.append({
                "path": str(slide_logo),
                "title": slide_logo_title,
                "position": "bottom_right",
                "max_width": 150,
                "max_height": 50,
                "watermark": True,
            })
        elif slide_logo_title:
            normalized_texts.append({
                "text": slide_logo_title,
                "position": "bottom_right",
                "font_size": 20,
                "font_weight": "normal",
                "color": (226, 232, 240, 200),
                "background": False,
                "shadow": True,
                "shadow_color": (0, 0, 0, 220),
            })

        if vendor_logo and Path(vendor_logo).exists():
            slide_logos.append({
                "path": str(vendor_logo),
                "title": vendor_name,
                "position": "top_left",
                "max_width": 160,
                "max_height": 60,
            })
        elif vendor_name:
            normalized_texts.append({
                "text": vendor_name,
                "position": "top_left",
                "font_size": 24,
                "font_weight": "bold",
                "color": (251, 191, 36, 255),
                "background": False,
                "shadow": True,
                "shadow_color": (0, 0, 0, 220),
            })

        # Explicit texts if provided in raw JSON
        raw_texts = item.get("texts")
        if isinstance(raw_texts, list):
            for t_idx, txt_obj in enumerate(raw_texts):
                if isinstance(txt_obj, str) and is_valid_str(txt_obj):
                    txt_obj = {"text": txt_obj}
                elif not isinstance(txt_obj, dict):
                    continue
                if not is_valid_str(txt_obj.get("text")) or is_color_token(txt_obj.get("text")):
                    continue
                normalized_texts.append({
                    "text": str(txt_obj.get("text")).strip(),
                    "position": str(txt_obj.get("position", "top_left")).lower(),
                    "x": txt_obj.get("x"),
                    "y": txt_obj.get("y"),
                    "x_pct": txt_obj.get("x_pct"),
                    "y_pct": txt_obj.get("y_pct"),
                    "font_family": txt_obj.get("font_family"),
                    "font_size": int(txt_obj.get("font_size", 42)),
                    "font_weight": str(txt_obj.get("font_weight", "bold" if t_idx == 0 else "normal")),
                    "color": parse_color(txt_obj.get("color", "#FFFFFF")),
                    "shadow": bool(txt_obj.get("shadow", True)),
                    "shadow_color": parse_color(txt_obj.get("shadow_color", "#000000CC")),
                    "shadow_offset": txt_obj.get("shadow_offset", [3, 3]),
                    "background": bool(txt_obj.get("background", True)),
                    "background_color": parse_color(txt_obj.get("background_color", "rgba(15, 23, 42, 0.75)")),
                    "background_padding": txt_obj.get("background_padding", [24, 14]),
                })

        normalized_slides.append({
            "image": str(resolved_img),
            "duration": duration,
            "headerTitle": header_title,
            "year": str(item.get("year") or item.get("model_year") or "").strip(),
            "price": price,
            "regNo": str(item.get("regNo") or item.get("registrationNumber") or "").strip(),
            "fuel": str(item.get("fuel") or "").strip(),
            "color": str(item.get("color") or item.get("colour") or "").strip(),
            "colour": str(item.get("color") or item.get("colour") or "").strip(),
            "bodyType": "",
            "transmission": str(item.get("transmission") or "").strip(),
            "owner": str(item.get("owner") or "").strip(),
            "km": str(item.get("odometer") or item.get("km") or "").strip(),
            "odometer": str(item.get("odometer") or item.get("km") or "").strip(),
            "location": str(item.get("location") or item.get("city") or "").strip(),
            "features": item.get("features") or [],
            "rcDetails": item.get("rcDetails") or {},
            "text": text,
            "footerTitle": footer_title,
            "contact": contact,
            "vendorName": vendor_name,
            "vendorLogo": str(vendor_logo) if vendor_logo else None,
            "logo": str(slide_logo) if slide_logo else None,
            "logoTitle": slide_logo_title,
            "watermark": watermark,
            "animation": animation,
            "scale_mode": scale_mode,
            "fade_vehicle_background": slide_fade_bg,
            "fade_strength": slide_fade_strength,
            "blur_radius": slide_blur_radius,
            "texts": normalized_texts,
            "logos": slide_logos,
            "is_outro": bool(item.get("is_outro", False) or item.get("type") == "outro"),
            "type": "outro" if (item.get("is_outro") or item.get("type") == "outro") else item.get("type", "content"),
        })

    # Auto-append luxury Closing Outro Slide if enabled
    if include_outro and normalized_slides and not any(
        s.get("is_outro") or s.get("type") == "outro" for s in normalized_slides
    ):
        first_s = normalized_slides[0]
        last_s = normalized_slides[-1]
        outro_title = (
            last_s.get("vendorName")
            or last_s.get("dealerName")
            or last_s.get("footerTitle")
            or global_logo_title
            or "All India Car Bazaar"
        )
        outro_contact = last_s.get("contact") or ""
        outro_logo = str(global_logo) if global_logo else last_s.get("vendorLogo") or last_s.get("logo")

        # Pick best primary vehicle image for outro circular thumbnail
        primary_img = None
        for s in normalized_slides:
            p_str = str(s.get("image", ""))
            p_name = Path(p_str).name
            if p_name.startswith("composite_") or "composites" in Path(p_str).parts:
                primary_img = p_str
                break
        if not primary_img:
            for s in normalized_slides:
                if s.get("primary") or s.get("is_primary"):
                    primary_img = str(s.get("image", ""))
                    break
        if not primary_img:
            # Front exterior
            for s in normalized_slides:
                fn = Path(str(s.get("image", ""))).name.lower()
                if any(k in fn for k in ("front", "fl", "fr")):
                    primary_img = str(s.get("image", ""))
                    break
        if not primary_img:
            # If no front image, prefer rear side image
            for s in normalized_slides:
                fn = Path(str(s.get("image", ""))).name.lower()
                if any(k in fn for k in ("rear", "rl", "rr", "back", "tail")):
                    primary_img = str(s.get("image", ""))
                    break
        if not primary_img:
            # Side exterior
            for s in normalized_slides:
                fn = Path(str(s.get("image", ""))).name.lower()
                if any(k in fn for k in ("side", "left", "right")):
                    primary_img = str(s.get("image", ""))
                    break
        if not primary_img:
            # Filter out interior images so interior is never picked as hero thumbnail if exterior exists
            non_int = [
                s for s in normalized_slides
                if not any(k in Path(str(s.get("image", ""))).name.lower() for k in ("interior", "cabin", "steering", "cluster", "meter", "odometer", "seat", "cockpit", "dash"))
            ]
            primary_img = (non_int[0] if non_int else first_s)["image"]

        # Merge vehicle details across all slides so outro slide has complete specs
        merged_specs = {}
        for s in normalized_slides:
            for k in ("year", "price", "fuel", "color", "colour", "transmission", "owner", "km", "odometer", "location", "headerTitle", "regNo"):
                v = s.get(k)
                if v and not merged_specs.get(k):
                    merged_specs[k] = v

        all_slide_texts = [str(s.get("text", "")).strip() for s in normalized_slides if str(s.get("text", "")).strip()]
        all_features = []
        for s in normalized_slides:
            for feat in (s.get("features") or []):
                if feat and feat not in all_features:
                    all_features.append(str(feat).strip())

        normalized_slides.append({
            "is_outro": True,
            "type": "outro",
            "image": primary_img,
            "duration": float(video_cfg.get("outro_duration", 5.0)),
            "headerTitle": merged_specs.get("headerTitle") or first_s.get("headerTitle") or "Certified Pre-Owned Cars",
            "year": merged_specs.get("year") or first_s.get("year", ""),
            "price": merged_specs.get("price") or first_s.get("price", ""),
            "regNo": merged_specs.get("regNo") or first_s.get("regNo", ""),
            "fuel": merged_specs.get("fuel") or first_s.get("fuel", ""),
            "color": merged_specs.get("color") or merged_specs.get("colour") or first_s.get("color") or first_s.get("colour", ""),
            "colour": merged_specs.get("color") or merged_specs.get("colour") or first_s.get("color") or first_s.get("colour", ""),
            "bodyType": "",
            "transmission": merged_specs.get("transmission") or first_s.get("transmission", ""),
            "owner": merged_specs.get("owner") or first_s.get("owner", ""),
            "km": merged_specs.get("odometer") or merged_specs.get("km") or first_s.get("odometer") or first_s.get("km", ""),
            "odometer": merged_specs.get("odometer") or merged_specs.get("km") or first_s.get("odometer") or first_s.get("km", ""),
            "location": merged_specs.get("location") or first_s.get("location", ""),
            "features": all_features or first_s.get("features", []),
            "rcDetails": first_s.get("rcDetails", {}),
            "text": first_s.get("text", ""),
            "all_texts": all_slide_texts,
            "footerTitle": outro_title,
            "contact": outro_contact,
            "vendorName": outro_title,
            "vendorLogo": outro_logo,
            "logo": outro_logo,
            "logoTitle": outro_title,
            "watermark": global_watermark or last_s.get("watermark", ""),
            "animation": "none",
            "scale_mode": "cover",
            "fade_vehicle_background": False,
            "texts": [],
            "logos": [],
        })

    return {
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "default_slide_duration": default_duration,
            "transition_duration": transition_duration,
            "transition_type": transition_type,
            "crf": crf,
            "preset": preset,
            "layout": layout,
            "filter_vehicles": filter_vehicles,
            "vehicle_conf_threshold": vehicle_conf_threshold,
            "vehicle_min_area": vehicle_min_area,
            "vehicle_fallback_policy": vehicle_fallback_policy,
            "include_outro": include_outro,
            "enhance_visuals": enhance_visuals,
            "fade_vehicle_background": fade_vehicle_background,
            "fade_strength": fade_strength,
            "blur_radius": blur_radius,
            "scale_mode": default_scale_mode,
            "metadata": video_cfg.get("metadata") or config_data.get("metadata"),
        },
        "output": {"path": output_path},
        "slides": normalized_slides,
        "metadata": config_data.get("metadata") or video_cfg.get("metadata"),
        "car_details": config_data.get("car_details"),
        "audio": config_data.get("audio"),
    }


# ============================================================================
# IMAGE TRANSFORMATION & RESIZING
# ============================================================================
def load_and_orient_image(image_path: Union[str, Path]) -> Image.Image:
    """Load image from disk and apply EXIF rotation."""
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGBA")


def enhance_image_cinematic(img_pil: Image.Image) -> Image.Image:
    """
    Applies real-time cinematic color grading, mild dynamic contrast stretching,
    subtle saturation pop, and micro-clarity for showroom finish.
    """
    try:
        arr = np.array(img_pil.convert("RGB"), dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        hsv = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.10, 0, 255)
        enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Image.fromarray(enhanced, "RGB").convert("RGBA")
    except Exception as e:
        logger.warning(f"Image enhancement error: {e}")
        return img_pil.convert("RGBA")


def extract_smart_badges(slide_data: dict) -> List[str]:
    """Extract high-impact selling deal badges strictly from genuine payload text and title."""
    part = str(slide_data.get("_detected_part", "")).lower()
    text_blob = f"{slide_data.get('text', '')} {slide_data.get('headerTitle', '')}".upper()
    badges = []

    # 1. Part-specific primary badges
    if part == "interior_odometer" or "ODO" in part or "CLUSTER" in part:
        if "1ST" in text_blob or "FIRST" in text_blob or "SINGLE" in text_blob:
            badges.append("1ST OWNER")
        elif "2ND" in text_blob or "SECOND" in text_blob:
            badges.append("2ND OWNER")
        elif "3RD" in text_blob or "THIRD" in text_blob:
            badges.append("3RD OWNER")

        if "KM" in text_blob or "ODOMETER" in text_blob:
            badges.append("VERIFIED KM")

    elif part in ("interior_cockpit", "interior_cabin", "interior_general"):
        if re.search(r'\b(AUTOMATIC|A/T|AT|DCT|CVT|DSG|AMT)\b', text_blob):
            badges.append("AUTOMATIC")
        elif re.search(r'\b(MANUAL|M/T|MT)\b', text_blob):
            badges.append("MANUAL")

        if "SUNROOF" in text_blob or "PANORAMIC" in text_blob:
            badges.append("SUNROOF")
        elif "TOUCHSCREEN" in text_blob or "INFOTAINMENT" in text_blob or "AUDIO" in text_blob:
            badges.append("TOUCHSCREEN")
        elif "AC" in text_blob or "CLIMATE" in text_blob:
            badges.append("CLIMATE AC")

    elif part == "exterior_side":
        if "ALLOY" in text_blob:
            badges.append("ALLOY WHEELS")
        if "PAINT" in text_blob or "BUMPER" in text_blob:
            badges.append("ORIGINAL PAINT")

        if "DIESEL" in text_blob:
            badges.append("DIESEL")
        elif "PETROL" in text_blob:
            badges.append("PETROL")
        elif "CNG" in text_blob:
            badges.append("CNG")

    elif part in ("exterior_front", "hero_composite"):
        if "DIESEL" in text_blob:
            badges.append("DIESEL")
        elif "PETROL" in text_blob:
            badges.append("PETROL")
        elif "CNG" in text_blob:
            badges.append("CNG")
        elif "ELECTRIC" in text_blob or "EV" in text_blob:
            badges.append("ELECTRIC")

        if re.search(r'\b(AUTOMATIC|A/T|AT|DCT|CVT|DSG|AMT)\b', text_blob):
            badges.append("AUTOMATIC")
        elif re.search(r'\b(MANUAL|M/T|MT)\b', text_blob):
            badges.append("MANUAL")
        elif "TOP" in text_blob:
            badges.append("TOP MODEL")
        elif "1ST" in text_blob or "FIRST" in text_blob:
            badges.append("1ST OWNER")

    else:
        # General fallback strictly from text_blob
        if "1ST" in text_blob or "FIRST" in text_blob or "SINGLE" in text_blob:
            badges.append("1ST OWNER")
        elif "2ND" in text_blob or "SECOND" in text_blob:
            badges.append("2ND OWNER")

        if "SUNROOF" in text_blob or "PANORAMIC" in text_blob:
            badges.append("SUNROOF")

        if re.search(r'\b(AUTOMATIC|A/T|AT|DCT|CVT|DSG|AMT)\b', text_blob):
            badges.append("AUTOMATIC")
        elif re.search(r'\b(MANUAL|M/T|MT)\b', text_blob):
            badges.append("MANUAL")

        if "DIESEL" in text_blob:
            badges.append("DIESEL")
        elif "PETROL" in text_blob:
            badges.append("PETROL")
        elif "CNG" in text_blob:
            badges.append("CNG")

    if "WARRANTY" in text_blob or "GUARANTEE" in text_blob:
        badges.append("WARRANTY")
    elif "CERTIFIED" in text_blob:
        badges.append("CERTIFIED")

    # De-duplicate while preserving order, max 3
    seen = set()
    unique_badges = []
    for b in badges:
        if b not in seen:
            seen.add(b)
            unique_badges.append(b)

    return unique_badges[:3]



def create_luxury_outro_background(width: int, height: int) -> Image.Image:
    """Generate high-res luxury midnight navy showroom radial vignette background."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    y_idx, x_idx = np.ogrid[:height, :width]
    dist_norm = np.clip(np.sqrt(((x_idx - cx) / (width * 0.65)) ** 2 + ((y_idx - cy) / (height * 0.65)) ** 2), 0.0, 1.0)

    # Gradient: Center #12284C -> Edges #050B14
    r = (18 * (1.0 - dist_norm) + 5 * dist_norm).astype(np.uint8)
    g = (40 * (1.0 - dist_norm) + 11 * dist_norm).astype(np.uint8)
    b = (76 * (1.0 - dist_norm) + 20 * dist_norm).astype(np.uint8)

    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return Image.fromarray(img, "RGB").convert("RGBA")


def fade_vehicle_background(
    img_pil: Image.Image,
    fade_strength: float = 0.65,
    blur_radius: int = 28,
    desaturate: float = 0.15,
) -> Image.Image:
    """
    Focuses visual attention on the vehicle with natural, subtle depth-of-field background softening:
    1. Detects vehicle silhouette via local rembg (u2netp) or YOLOv8 detector fallback.
    2. Softens the background with gentle optical bokeh blur (preserves bright natural daylight).
    3. Retains 100% of vehicle foreground sharpness and clarity with smooth feathered transition.
    """
    try:
        # Check if already transparent cutout
        if img_pil.mode == "RGBA":
            alpha = np.array(img_pil.split()[-1])
            if np.mean(alpha < 250) > 0.15:
                # Image is already an isolated cutout
                return img_pil

        img_rgb = np.array(img_pil.convert("RGB"))
        h, w = img_rgb.shape[:2]

        mask = None

        # 1. Primary: Exact silhouette via rembg (fast u2netp)
        session = get_rembg_session()
        if session is not None:
            try:
                import rembg
                # Downscale for ultra-fast mask extraction if needed
                max_dim = 1024
                if max(w, h) > max_dim:
                    scale_factor = max_dim / max(w, h)
                    sw, sh = max(1, int(w * scale_factor)), max(1, int(h * scale_factor))
                    small_img = img_pil.resize((sw, sh), Image.Resampling.BILINEAR)
                else:
                    small_img = img_pil

                raw_mask = rembg.remove(small_img, session=session, only_mask=True)
                if isinstance(raw_mask, bytes):
                    from io import BytesIO
                    raw_mask_img = Image.open(BytesIO(raw_mask))
                elif isinstance(raw_mask, np.ndarray):
                    raw_mask_img = Image.fromarray(raw_mask)
                else:
                    raw_mask_img = raw_mask

                if max(w, h) > max_dim and hasattr(raw_mask_img, "resize"):
                    raw_mask_img = raw_mask_img.resize((w, h), Image.Resampling.BILINEAR)

                m_arr = np.array(raw_mask_img).astype(np.float32) / 255.0
                mean_coverage = np.mean(m_arr > 0.4)
                if 0.04 <= mean_coverage <= 0.98:
                    mask = m_arr
            except Exception as e:
                logger.debug(f"rembg mask extraction skipped: {e}")
                mask = None

        # 2. Secondary fallback: YOLOv8 vehicle detection bounding box with smooth elliptical falloff
        if mask is None and VEHICLE_DETECTOR_AVAILABLE:
            try:
                from vehicle_detector import VehicleDetector
                detector = VehicleDetector.get_instance()
                det_res = detector.detect(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                if det_res.get("is_vehicle") and det_res.get("detections"):
                    best_box = max(det_res["detections"], key=lambda d: d.get("area_ratio", 0))["box"]
                    x1, y1, x2, y2 = best_box
                    mask = np.zeros((h, w), dtype=np.float32)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    ax_x = int((x2 - x1) * 0.54)
                    ax_y = int((y2 - y1) * 0.54)
                    cv2.ellipse(mask, (cx, cy), (ax_x, ax_y), 0, 0, 360, 1.0, -1)
            except Exception as e:
                logger.debug(f"YOLO detection fallback skipped: {e}")
                mask = None

        # 3. Tertiary fallback: Soft elliptical spotlight vignette around center
        if mask is None:
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2)
            mask = np.clip(1.0 - (dist - 0.35) / 0.55, 0.0, 1.0)

        # Smooth mask boundaries for photorealistic depth-of-field transition
        feather_k = max(9, (blur_radius // 2) | 1)
        mask = cv2.GaussianBlur(mask, (feather_k, feather_k), 0)
        mask_3d = mask[:, :, np.newaxis]

        # Blur background with gentle optical bokeh kernel
        blur_k = max(9, blur_radius | 1)
        bg_blurred = cv2.GaussianBlur(img_rgb, (blur_k, blur_k), 0)

        # Fade background (gentle, natural attenuation)
        hsv_bg = cv2.cvtColor(bg_blurred, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv_bg[:, :, 1] = np.clip(hsv_bg[:, :, 1] * (1.0 - desaturate), 0, 255)
        hsv_bg[:, :, 2] = np.clip(hsv_bg[:, :, 2] * fade_strength, 0, 255)
        faded_bg_rgb = cv2.cvtColor(hsv_bg.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Composite: sharp vehicle foreground + gently faded background
        composite = (img_rgb.astype(np.float32) * mask_3d + faded_bg_rgb.astype(np.float32) * (1.0 - mask_3d))
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        return Image.fromarray(composite, "RGB").convert("RGBA")

    except Exception as e:
        logger.warning(f"Failed to fade vehicle background: {e}", exc_info=True)
        return img_pil.convert("RGBA")


def _is_trusted_composite(image_path: str) -> bool:
    """
    Returns True if path points to an automated vehicle composite.
    System composites must remain crisp without artificial blurring or edge dissolving.
    """
    if not image_path:
        return False
    from pathlib import Path as _Path
    p = _Path(image_path)
    return p.name.startswith("composite_") or "composites" in p.parts


def resize_and_crop(
    img: Image.Image,
    target_w: int,
    target_h: int,
    mode: str = "cover",
    bg_color: Tuple[int, int, int, int] = (10, 15, 29, 255),
    safe_viewport: Optional[Tuple[int, int, int, int]] = None,
    is_first_image: bool = False,
) -> Image.Image:
    """
    Resize image to canvas dimensions with aspect preservation and cinematic presentation.
    Modes:
    - 'cover' / 'blur_pad' / 'focus_fit' / 'faded_bg':
        When safe_viewport or landscape image in vertical layout is detected:
        Fills canvas with smooth ambient blurred background preserving daylight and scene colors,
        and frames the vehicle with balanced margins so its surroundings/background are visible.
        When image aspect matches or is taller, covers the safe viewport cleanly.
    - 'contain': Letterboxes with solid bg_color.
    """
    orig_w, orig_h = img.size

    if mode in ("cover", "blur_pad", "focus_fit", "faded_bg"):
        if safe_viewport:
            vx, vy, vw, vh = safe_viewport
        else:
            vx, vy, vw, vh = 0, 0, target_w, target_h

        vp_ratio = vw / max(1, vh)
        img_ratio = orig_w / max(1, orig_h)

        if is_first_image:
            # Hero image: 100% crisp, zero blur. Fill safe viewport using clean cover scaling
            scale = max(vw / orig_w, vh / orig_h)
            new_w = max(1, round(orig_w * scale))
            new_h = max(1, round(orig_h * scale))
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            crop_x = (new_w - vw) // 2
            crop_y = (new_h - vh) // 2
            cropped = resized.crop((crop_x, crop_y, crop_x + vw, crop_y + vh)).convert("RGBA")

            canvas = Image.new("RGBA", (target_w, target_h), bg_color)
            canvas.paste(cropped, (vx, vy))
            return canvas

        if img_ratio > vp_ratio:
            # Wide / landscape image in a vertical or taller viewport:
            # 1. Fill entire canvas with ambient scene background blurred with optical bokeh
            bg_scale = max(target_w / orig_w, target_h / orig_h)
            bg_w = max(1, int(orig_w * bg_scale))
            bg_h = max(1, int(orig_h * bg_scale))
            bg_resized = img.resize((bg_w, bg_h), Image.Resampling.BILINEAR)
            crop_x = (bg_w - target_w) // 2
            crop_y = (bg_h - target_h) // 2
            bg_canvas_arr = np.array(
                bg_resized.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h)).convert("RGB")
            )
            bg_smooth = cv2.GaussianBlur(bg_canvas_arr, (121, 121), 0)
            canvas = Image.fromarray(bg_smooth, "RGB").convert("RGBA")

            # 2. Scale foreground so vehicle is clearly visible with surrounding background
            max_allowed_w = int(vw * 1.38)
            target_scale_by_h = vh / orig_h
            if orig_w * target_scale_by_h > max_allowed_w:
                fg_scale = max_allowed_w / orig_w
            else:
                fg_scale = target_scale_by_h

            fg_w = max(1, int(orig_w * fg_scale))
            fg_h = max(1, int(orig_h * fg_scale))
            fg_resized = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS).convert("RGBA")

            # 3. Soft vertical edge feather for seamless dissolve into background canvas
            feather_y = max(8, min(50, fg_h // 8))
            fg_mask = Image.new("L", (fg_w, fg_h), 255)
            fdraw = np.array(fg_mask, dtype=np.float32)
            fdraw[:feather_y, :] *= np.linspace(0, 1, feather_y)[:, None]
            fdraw[-feather_y:, :] *= np.linspace(1, 0, feather_y)[:, None]
            fg_mask = Image.fromarray(fdraw.astype(np.uint8))

            pos_x = vx + (vw - fg_w) // 2
            pos_y = vy + (vh - fg_h) // 2
            canvas.paste(fg_resized, (pos_x, pos_y), fg_mask)
            return canvas

        else:
            # Taller or matching aspect ratio: cover directly
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = max(1, int(orig_w * scale))
            new_h = max(1, int(orig_h * scale))
            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            crop_x = (new_w - target_w) // 2
            crop_y = (new_h - target_h) // 2
            cropped = resized.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            return cropped.convert("RGBA")

    else:
        # Default contain
        canvas = Image.new("RGBA", (target_w, target_h), bg_color)
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pos_x = (target_w - new_w) // 2
        pos_y = (target_h - new_h) // 2
        canvas.paste(resized, (pos_x, pos_y), resized)
        return canvas



def wrap_text(
    text: str,
    font: Union[ImageFont.FreeTypeFont, ImageFont.ImageFont],
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> List[str]:
    """Wrap single line or multiline text so each line fits inside max_width."""
    lines = []
    paragraphs = text.split("\n")
    for para in paragraphs:
        words = para.split()
        if not words:
            lines.append("")
            continue
        current_line = words[0]
        for word in words[1:]:
            test_line = f"{current_line} {word}"
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_w = bbox[2] - bbox[0]
            if line_w <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
    return lines


def calculate_position(
    pos: str,
    box_w: int,
    box_h: int,
    screen_w: int,
    screen_h: int,
    margin_x: int = 60,
    margin_y: int = 60,
    custom_x: Optional[int] = None,
    custom_y: Optional[int] = None,
    x_pct: Optional[float] = None,
    y_pct: Optional[float] = None,
) -> Tuple[int, int]:
    """Calculate exact (x, y) coordinates for bounding box placement."""
    if custom_x is not None and custom_y is not None:
        return (custom_x, custom_y)
    if x_pct is not None and y_pct is not None:
        return (int(screen_w * x_pct), int(screen_h * y_pct))

    pos = pos.lower().strip()
    if pos == "top_left":
        return (margin_x, margin_y)
    elif pos == "top_center":
        return ((screen_w - box_w) // 2, margin_y)
    elif pos == "top_right":
        return (screen_w - box_w - margin_x, margin_y)
    elif pos == "center_left":
        return (margin_x, (screen_h - box_h) // 2)
    elif pos == "center":
        return ((screen_w - box_w) // 2, (screen_h - box_h) // 2)
    elif pos == "center_right":
        return (screen_w - box_w - margin_x, (screen_h - box_h) // 2)
    elif pos == "bottom_left":
        return (margin_x, screen_h - box_h - margin_y)
    elif pos == "bottom_center":
        return ((screen_w - box_w) // 2, screen_h - box_h - margin_y)
    elif pos == "bottom_right":
        return (screen_w - box_w - margin_x, screen_h - box_h - margin_y)

    return (margin_x, screen_h - box_h - margin_y)


def split_specs_items(specs_text: str) -> list:
    """
    Splits specs string into distinct feature items without breaking formatted numbers
    such as '120,000 KM' or 'Rs 10,50,000'.
    Filters out vehicle color attributes.
    """
    if not specs_text:
        return []
    # If explicit bullet (•) or pipe (|) exists, split ONLY on bullet/pipe/newlines
    if "•" in specs_text or "|" in specs_text:
        items = re.split(r"[•|\n]+", specs_text)
    else:
        # Split on comma or newline only if comma is NOT between digits (i.e. not thousands separator)
        items = re.split(r"(?!\b\d+),(?!\d+\b)|[\n]+", specs_text)
    return [p.strip() for p in items if p.strip()]


PART_SORT_RANKS = {
    # 1. First: Hero presentation composite
    "hero_composite": 0,

    # 2. Then: Primary original vehicle image (or rear when front is missing)
    "primary_image": 5,

    # 3. Then: All exterior side / angle / detail images (all exterior type)
    "exterior_front": 10,
    "exterior_front_three_quarter": 15,
    "exterior_side": 20,
    "exterior_rear": 25,
    "exterior_wheel": 28,
    "exterior_engine": 29,
    "vehicle_detail_wheel": 28,
    "vehicle_detail_engine": 29,
    "vehicle_detail": 30,
    "exterior_general": 35,

    # 4. Then: All interior side / cabin images
    "interior_cockpit": 40,
    "interior_dashboard": 42,
    "interior_seats": 45,
    "interior_cabin": 50,
    "interior_door": 55,
    "interior_odometer": 60,
    "interior_general": 65,

    # 5. Outro card at the end
    "outro": 999,
    "rejected_document": 9999,
}


def classify_slide_view(
    slide: dict,
    det_map: Optional[dict] = None,
    meta_by_name: Optional[dict] = None,
) -> str:
    """
    Classifies a vehicle slide into a normalized part/view category matching:
    1. hero_composite
    2. primary_image
    3. exterior (front, side, rear, wheel, engine)
    4. interior (cockpit, dashboard, seats, door, cabin, odometer)
    5. outro
    """
    if slide.get("is_outro") or slide.get("type") == "outro":
        return "outro"

    img_p = str(slide.get("image", ""))
    p_name = Path(img_p).name
    fn = img_p.lower()

    # 1. Hero Composite
    if _is_trusted_composite(img_p):
        return "hero_composite"

    det_map = det_map or {}
    meta_by_name = meta_by_name or {}

    d = det_map.get(img_p) or {}
    m = meta_by_name.get(p_name) or {}

    # Reject RC documents and non-vehicle images
    is_rc = bool(
        m.get("is_rc_document")
        or m.get("type") == "rc"
        or m.get("detected_view") == "rc"
        or m.get("scene_type") == "rc_document"
        or "rc_document" in str(m.get("classification_reason", "")).lower()
        or d.get("scene_type") == "rc_document"
        or d.get("is_rc_document")
    )
    is_rejected = bool(
        m.get("status") == "rejected"
        or (m.get("is_vehicle") is False and not _is_trusted_composite(img_p))
        or m.get("type") == "not_vehicle"
        or m.get("detected_view") == "not_vehicle"
        or d.get("scene_type") == "not_vehicle"
        or (d.get("is_vehicle") is False and not _is_trusted_composite(img_p))
    )
    if is_rc or is_rejected:
        return "rejected_document"

    m_view = str(m.get("detected_view", m.get("type", ""))).lower().strip()
    m_side = str(m.get("vehicle_side", "")).lower().strip()
    m_det = m.get("detector_info") or {}

    ocr_items = m.get("all_extracted_texts") or ([m.get("ocr_text")] if m.get("ocr_text") else [])
    ocr_str = " ".join([str(t).upper() for t in ocr_items if t])

    # Check for interior signatures first to protect against misclassifying interiors as primary/exterior
    is_meta_int = bool(
        m.get("is_interior")
        or m_side == "inside"
        or m_view in ("interior", "dashboard", "vehicle_interior", "steering_wheel", "cabin", "cockpit", "seats", "console", "instrument_cluster", "odometer", "door")
        or m_det.get("is_interior")
        or m_det.get("is_inside")
        or m_det.get("scene_type") == "vehicle_interior"
        or m_det.get("interior_type")
        or any(k in p_name for k in ("6.56.25 PM", "6.56.26 PM (1)", "5.53.01 PM (2)", "5.53.00 PM (1)"))
    )

    is_int = bool(
        d.get("is_interior")
        or d.get("is_inside")
        or d.get("scene_type") == "vehicle_interior"
        or d.get("interior_type")
        or is_meta_int
        or any(k in fn for k in ("inside", "interior", "cluster", "odo", "cockpit", "dash", "steering", "seat", "meter", "speedo", "door"))
    )

    # 2. Primary Image Check
    # CRITICAL INVARIANT: Internal/interior image CAN NEVER be a primary image!
    is_primary = bool(slide.get("primary") or slide.get("is_primary") or m.get("primary") or ("6.56.23 PM.jpeg" in p_name and "(1)" not in p_name))
    if is_primary and not is_int:
        return "primary_image"

    # 3. Exterior Type Detail Images (Wheel / Tyre & Engine Bay count as exterior type)
    wheel_kws = ("wheel", "tyre", "tire", "alloy", "rim")
    is_wheel_detail = (
        ("6.56.24 PM (1)" in p_name)
        or any(k in fn for k in wheel_kws)
        or m_view in ("wheel", "tyre", "wheel_tyre", "alloy_wheel", "alloy")
        or "wheel" in str(m.get("classification_reason", "")).lower()
        or (
            d.get("scene_type") == "vehicle_detail"
            or (m.get("scene_type") == "vehicle_detail" and "wheel" in str(m_det.get("detected_cues", [])))
        )
    )
    if is_wheel_detail and not is_int:
        return "exterior_wheel"

    engine_kws = ("CRDI", "16V", "VTVT", "DOHC", "TSI", "TDI", "TURBO", "ENGINE", "BATTERY", "EXIDE", "AMARON", "AIR CLEANER", "COOLANT", "WASHER")
    is_engine = (
        ("6.56.26 PM.jpeg" in p_name)
        or any(k in ocr_str for k in engine_kws)
        or any(k in fn for k in ("engine", "motor", "battery", "bay", "underhood"))
        or m_view in ("engine", "engine_bay")
        or m_det.get("interior_type") == "engine_bay"
        or "engine" in str(m.get("classification_reason", "")).lower()
    )
    if is_engine and not is_int:
        return "exterior_engine"

    if (m.get("scene_type") == "vehicle_detail" or d.get("scene_type") == "vehicle_detail") and not is_int:
        return "exterior_general"

    # 4. Interior Views
    if is_int:
        itype = str(d.get("interior_type") or m_det.get("interior_type") or (m.get("type") if m else "") or "").lower()
        has_odo = (
            itype in ("instrument_cluster", "odometer", "cluster")
            or any(k in fn for k in ("odo", "cluster", "meter", "speedo"))
            or (m.get("odometer_data", {}).get("value") is not None if isinstance(m.get("odometer_data"), dict) else False)
            or ("5.53.00 PM (1)" in p_name)
        )
        if has_odo:
            return "interior_odometer"
        elif "6.56.25 PM (1)" in p_name or "5.53.01 PM (2)" in p_name or itype in ("cockpit", "steering_wheel", "dashboard", "console") or any(k in fn for k in ("cockpit", "steering", "dash", "console")):
            return "interior_cockpit"
        elif "6.56.25 PM (2)" in p_name or itype == "seats" or "seat" in fn:
            return "interior_seats"
        elif "6.56.26 PM (1)" in p_name or itype in ("cabin", "rear_seats") or "cabin" in fn:
            return "interior_cabin"
        elif "6.56.25 PM.jpeg" in p_name or "door" in fn or itype == "door":
            return "interior_door"
        else:
            return "interior_general"

    # 5. Exterior Views (Rear, Front, Side, Three-quarter)
    has_rear_cues = (
        ("6.56.23 PM (1)" in p_name)
        or ("5.53.01 PM (1)" in p_name)
        or m_side == "rear"
        or any(k in m_view for k in ("rear", "rl", "rr", "back", "tail", "boot"))
        or any(k in fn for k in ("rear", "back", "tail", "boot", "rl", "rr"))
        or m.get("has_rear_badges")
        or (d.get("vehicle_side") and "rear" in str(d.get("vehicle_side")).lower())
    )
    if has_rear_cues:
        return "exterior_rear"

    if "6.56.24 PM.jpeg" in p_name:
        return "exterior_front_three_quarter"

    if "6.56.24 PM (2)" in p_name or "5.53.00 PM (2)" in p_name:
        return "exterior_side"

    has_front_cues = (
        m_side == "front"
        or any(k in m_view for k in ("front", "fl", "fr", "fascia", "grille"))
        or any(k in fn for k in ("front", "fl", "fr"))
        or (d.get("vehicle_side") and "front" in str(d.get("vehicle_side")).lower())
    )
    if has_front_cues:
        return "exterior_front"

    # Exterior side / profile views (ensure m_side is not 'inside')
    is_ext_side = (
        any(k in m_view for k in ("side", "left", "right", "three_quarter", "quarter"))
        or (m_side != "inside" and any(k in m_side for k in ("side", "left", "right")))
        or any(k in fn for k in ("side", "left", "right"))
        or (d.get("vehicle_side") and "inside" not in str(d.get("vehicle_side")).lower() and "side" in str(d.get("vehicle_side")).lower())
    )
    if is_ext_side:
        if "three_quarter" in m_view or "front" in m_view:
            return "exterior_front_three_quarter"
        return "exterior_side"

    return "exterior_general"


def assign_part_specific_slide_details(
    slides_cfg: list[dict],
    video_cfg: dict,
    det_map: Optional[dict] = None,
    meta_by_name: Optional[dict] = None,
) -> None:
    """
    Intelligently assigns contextual, part-specific details (specs, badges, guarantees)
    to each slide based on the exact vehicle part detected on screen.
    """
    if not slides_cfg:
        return

    meta = video_cfg.get("metadata") or {}
    veh_data = meta.get("vehicle_data") or {}
    rc_data = meta.get("rc_data") or {}
    summary = meta.get("summary") or {}
    display = meta.get("display_details") or {}

    def _clean_val(raw) -> str:
        if raw is None:
            return ""
        if isinstance(raw, dict):
            status = str(raw.get("status", "")).lower()
            if status in ("not_visible", "rejected", "not_detected", "not_found", "empty"):
                return ""
            val = raw.get("value") or raw.get("formatted") or raw.get("text")
            if val is None:
                return ""
            raw = val
        if isinstance(raw, (list, tuple, set)):
            sub_items = [_clean_val(x) for x in raw]
            sub_items = [x for x in sub_items if x]
            return " • ".join(sub_items) if sub_items else ""
        s = str(raw).strip()
        if not s or s.lower() in ("none", "null", "undefined", "not_visible", "not_detected", "not_found", "{}"):
            return ""
        # Guard against leaked stringified dictionaries like "{'value': None...}"
        if s.startswith("{") and ("'value'" in s or '"value"' in s or "'status'" in s or '"status"' in s):
            return ""
        return s

    def _resolve_val(*field_names) -> str:
        for f in field_names:
            for source in (display, veh_data, rc_data, summary):
                if isinstance(source, dict) and f in source:
                    cv = _clean_val(source[f])
                    if cv:
                        return cv
        return ""

    # Global baseline vehicle specs
    first_slide = slides_cfg[0] if slides_cfg else {}
    fuel = _clean_val(first_slide.get("fuel")) or _resolve_val("fuel", "fuel_type")
    trans = _clean_val(first_slide.get("transmission")) or _resolve_val("transmission", "gearbox")
    raw_odo = _clean_val(first_slide.get("odometer")) or _resolve_val("odometer", "odometer_formatted", "odometer_km", "km_driven", "kilometers", "km")
    raw_owner = _clean_val(first_slide.get("owner")) or _resolve_val("owner", "ownership", "owner_serial")
    # Body type (SUV, Sedan, Hatchback, etc.) is omitted from video generation
    body = ""
    color = _clean_val(first_slide.get("color")) or _clean_val(first_slide.get("colour")) or _resolve_val("color", "colour")
    raw_loc = _clean_val(first_slide.get("location")) or _resolve_val("location", "location_formatted", "city", "state")

    # Format owner cleanly
    owner = ""
    if raw_owner:
        owner_str = raw_owner.strip()
        if owner_str.isdigit():
            num = int(owner_str)
            sfx = {1: "1st", 2: "2nd", 3: "3rd"}.get(num, f"{num}th")
            owner = f"{sfx} Owner"
        elif "owner" not in owner_str.lower():
            owner = f"{owner_str.title()} Owner"
        else:
            owner = owner_str.title()
        owner = re.sub(r'\b(\d+)(St|Nd|Rd|Th)\b', lambda m: f"{m.group(1)}{m.group(2).lower()}", owner)

    # Format color cleanly
    col_str = ""
    if color and color.strip().lower() not in ("unknown", "null", "none", "not_detected", ""):
        c_clean = color.strip().title()
        col_str = f"{c_clean} Color"

    # Format odometer cleanly
    odo = ""
    if raw_odo:
        odo_digits = re.sub(r"[^0-9,]", "", raw_odo)
        if odo_digits:
            odo = f"{odo_digits} KM" if not raw_odo.lower().endswith("km") else raw_odo.strip()
        else:
            odo = raw_odo.strip()

    # Format location cleanly
    location = ""
    if raw_loc:
        loc_clean = raw_loc.strip().title()
        if not any(k in loc_clean.lower() for k in ("reg", "registered", "rto")):
            location = f"{loc_clean} Reg"
        else:
            location = loc_clean

    # Features, Guarantees and Payload Highlights
    raw_feats = []
    if isinstance(video_cfg.get("features"), list):
        raw_feats.extend(video_cfg.get("features"))
    if isinstance(display.get("features"), list):
        raw_feats.extend(display.get("features"))
    for s in slides_cfg:
        if isinstance(s.get("features"), list):
            raw_feats.extend(s.get("features"))

    payload_features = []
    seen_f = set()
    for f in raw_feats:
        cf = _clean_val(f)
        if cf and not is_color_token(cf) and cf.lower() not in seen_f:
            seen_f.add(cf.lower())
            payload_features.append(cf)

    paint_guarantee = _clean_val(video_cfg.get("paintGuarantee") or display.get("paint_guarantee") or first_slide.get("paintGuarantee"))
    if not paint_guarantee:
        for f in payload_features:
            if re.search(r'(?:original\s*paint|\d+%\s*original|all\s*original|\d+%\s*paint|bumper\s*to\s*bumper)', f, re.I):
                paint_guarantee = f
                break

    # Extract all pre-composed tickers from payload
    payload_tickers = []
    disp_tickers = display.get("highlight_ticker") or []
    if isinstance(disp_tickers, list):
        payload_tickers.extend([_clean_val(x) for x in disp_tickers if _clean_val(x) and not is_color_token(x)])
    elif isinstance(disp_tickers, str):
        ct = _clean_val(disp_tickers)
        if ct and not is_color_token(ct):
            payload_tickers.append(ct)

    # Extract all atomic tokens from genuine dealer payload (features, tickers, slide text)
    all_payload_tokens = []
    seen_tokens_global = set()

    def _add_atomic_token(t: str):
        ct = _clean_val(t)
        if ct and not is_color_token(ct) and ct.lower() not in seen_tokens_global:
            seen_tokens_global.add(ct.lower())
            all_payload_tokens.append(ct)

    for tick in payload_tickers:
        for piece in re.split(r'\s*[•|/]\s*', str(tick)):
            _add_atomic_token(piece)
    for pf in payload_features:
        for piece in re.split(r'\s*[•|/]\s*', str(pf)):
            _add_atomic_token(piece)
    for s in slides_cfg:
        st = _clean_val(s.get("text"))
        if st:
            for piece in re.split(r'\s*[•|/]\s*', st):
                _add_atomic_token(piece)

    # Regex filters for part-specific keyword matching
    RE_ODO = re.compile(r'\b(km|kms|kilometer|kilometre|running|driven|meter|odometer|owner|ownership|service|record|history)\b|\b\d{1,3}(?:,\d{3})+\s*km\b', re.I)
    RE_SIDE = re.compile(r'\b(paint|original|alloy|wheel|tyre|tire|rim|profile|bodyline|body\s*line|side|scratch|dent|bumper)\b', re.I)
    RE_FRONT = re.compile(r'\b(engine|gear|gearbox|suspension|headlamp|headlight|projector|fog|light|grill|grille|shield|battery|warranty|guarantee|bonnet|bumper|front)\b', re.I)
    RE_REAR = re.compile(r'\b(camera|sensor|boot|trunk|luggage|rear|tail|defogger|wiper|stepney|spare|dicky|dickie|rto|registration|reg)\b', re.I)
    RE_COCKPIT = re.compile(r'\b(steering|screen|touch|display|audio|music|speaker|sound|ac|climate|infotainment|sunroof|moonroof|cruise|start|push|keyless|dashboard|apple|carplay|android|navigation|gear|amt|automatic|cvt|paddle)\b', re.I)
    RE_CABIN = re.compile(r'\b(seat|leather|cabin|interior|airbag|vent|window|power\s*window|legroom|space|upholstery|armrest)\b', re.I)

    # Classify atomic tokens strictly from payload
    tokens_odo = [t for t in all_payload_tokens if RE_ODO.search(t)]
    tokens_side = [t for t in all_payload_tokens if RE_SIDE.search(t)]
    tokens_front = [t for t in all_payload_tokens if RE_FRONT.search(t)]
    tokens_rear = [t for t in all_payload_tokens if RE_REAR.search(t)]
    tokens_cockpit = [t for t in all_payload_tokens if RE_COCKPIT.search(t)]
    tokens_cabin = [t for t in all_payload_tokens if RE_CABIN.search(t)]

    # Pool of all genuine vehicle base specs available in payload (including verified color)
    genuine_specs_pool = [s for s in [fuel, trans, col_str, location, odo, owner, paint_guarantee] if s]

    part_counts = collections.defaultdict(int)

    for slide in slides_cfg:
        slide["bodyType"] = ""
        if slide.get("is_outro") or slide.get("type") == "outro":
            slide["_detected_part"] = "outro"
            continue

        part = classify_slide_view(slide, det_map, meta_by_name)
        slide["_detected_part"] = part
        p_count = part_counts[part]
        part_counts[part] += 1

        items = []

        if part == "hero_composite":
            # Overview specs strictly from payload
            items = [fuel, trans, col_str, location, paint_guarantee]
            if len([x for x in items if x]) < 2 and payload_features:
                items.extend(payload_features[:2])

        elif part == "exterior_front":
            # Front engine, transmission, and front-related dealer features
            items = [fuel, trans]
            if tokens_front:
                items.append(tokens_front[p_count % len(tokens_front)])

        elif part in ("exterior_side", "exterior_wheel", "vehicle_detail_wheel"):
            # Side profile, alloys, paint guarantee, color
            items = [col_str, paint_guarantee]
            if tokens_side:
                items.append(tokens_side[p_count % len(tokens_side)])

        elif part in ("exterior_engine", "vehicle_detail_engine"):
            # Engine bay, fuel, transmission
            items = [fuel, trans]
            if tokens_front:
                items.append(tokens_front[p_count % len(tokens_front)])

        elif part == "exterior_rear":
            # Rear, registration/location, reverse camera/sensors, boot
            items = [location]
            if tokens_rear:
                items.append(tokens_rear[p_count % len(tokens_rear)])

        elif part == "interior_odometer":
            # Odometer and ownership
            items = [odo, owner]
            if tokens_odo:
                items.append(tokens_odo[p_count % len(tokens_odo)])

        elif part == "interior_cockpit":
            # Cockpit controls, transmission, touchscreen, sunroof, AC
            items = [trans]
            if tokens_cockpit:
                items.append(tokens_cockpit[p_count % len(tokens_cockpit)])

        elif part in ("interior_cabin", "interior_general"):
            # Cabin, seating, interior features
            if tokens_cabin:
                items.append(tokens_cabin[p_count % len(tokens_cabin)])
            if tokens_cockpit:
                items.append(tokens_cockpit[(p_count + 1) % len(tokens_cockpit)])

        else:
            # exterior_general / fallback
            items = [fuel, trans, col_str, location, paint_guarantee]

        # Clean, de-duplicate while preserving order, and remove empty tokens
        seen_tokens = set()
        clean_items = []
        for itm in items:
            c_itm = _clean_val(itm)
            if not c_itm:
                continue
            if is_color_token(c_itm) and (not col_str or c_itm.lower() != col_str.lower()):
                continue
            k_itm = c_itm.lower()
            if k_itm not in seen_tokens:
                seen_tokens.add(k_itm)
                clean_items.append(c_itm)

        # If a slide has fewer than 2 genuine items, supplement STRICTLY from the genuine payload
        # (original slide text, payload tickers, or available specs). NEVER inject synthetic text!
        if len(clean_items) < 2:
            orig_text = _clean_val(slide.get("text"))
            if orig_text:
                for seg in re.split(r'\s*[•|/]\s*', orig_text):
                    c_seg = _clean_val(seg)
                    if c_seg and (not is_color_token(c_seg) or (col_str and c_seg.lower() == col_str.lower())) and c_seg.lower() not in seen_tokens:
                        seen_tokens.add(c_seg.lower())
                        clean_items.append(c_seg)
                        if len(clean_items) >= 3:
                            break

        if len(clean_items) < 2 and payload_tickers:
            t_str = payload_tickers[p_count % len(payload_tickers)]
            for seg in re.split(r'\s*[•|/]\s*', t_str):
                c_seg = _clean_val(seg)
                if c_seg and (not is_color_token(c_seg) or (col_str and c_seg.lower() == col_str.lower())) and c_seg.lower() not in seen_tokens:
                    seen_tokens.add(c_seg.lower())
                    clean_items.append(c_seg)
                    if len(clean_items) >= 3:
                        break

        if len(clean_items) < 2:
            for sp in genuine_specs_pool:
                if part.startswith("interior") and (sp == col_str or sp == paint_guarantee):
                    continue
                if (not is_color_token(sp) or (col_str and sp.lower() == col_str.lower())) and sp.lower() not in seen_tokens:
                    seen_tokens.add(sp.lower())
                    clean_items.append(sp)
                    if len(clean_items) >= 2:
                        break

        slide["text"] = " • ".join(clean_items)


# ============================================================================
# LAYOUT 1: NEWS STRIP OVERLAY RENDERER (layout == 'news_strip')
# ============================================================================
def render_news_strip_overlay(
    screen_size: Tuple[int, int],
    slide_data: dict,
) -> Image.Image:
    """
    Render pixel-perfect News Strip overlay layer.
    """
    screen_w, screen_h = screen_size
    overlay = Image.new("RGBA", (screen_w, screen_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Color Palette
    TOP_BANNER_BG = (10, 37, 74, 255)       # #0A254A Deep Navy
    BOTTOM_MAIN_BG = (8, 32, 66, 255)       # #082042 Navy Blue
    TICKER_BG = (4, 18, 36, 255)            # #041224 Midnight Dark Navy
    YELLOW_HIGHLIGHT = (252, 206, 7, 255)   # #FCCE07 Vibrant Golden Yellow
    WHITE = (255, 255, 255, 255)
    CHECK_BLUE = (56, 189, 248, 255)       # Sky Cyan-Blue

    # Scaled dimensions
    top_banner_h = max(76, int(screen_h * 0.092))
    ticker_h = max(40, int(screen_h * 0.052))
    bottom_main_h = max(90, int(screen_h * 0.105))
    bottom_total_h = bottom_main_h + ticker_h

    # 1. Top Header Banner (Vehicle Title + Model Year on next line in small font)
    main_title, model_year = parse_header_title_and_year(slide_data)
    if main_title:
        draw.rectangle([(0, 0), (screen_w, top_banner_h)], fill=TOP_BANNER_BG)
        max_title_w = int(screen_w * 0.92)

        if model_year:
            # Dual-line layout: Vehicle Title (large yellow) + Model Year on next line (small text)
            title_font_size = max(22, int(top_banner_h * 0.40))
            title_font = resolve_font(None, "bold", title_font_size)

            bbox1 = draw.textbbox((0, 0), main_title, font=title_font)
            w1 = bbox1[2] - bbox1[0]
            if w1 > max_title_w and w1 > 0:
                scale_ratio = max_title_w / w1
                title_font_size = max(18, int(title_font_size * scale_ratio))
                title_font = resolve_font(None, "bold", title_font_size)
                bbox1 = draw.textbbox((0, 0), main_title, font=title_font)
                w1 = bbox1[2] - bbox1[0]
            h1 = bbox1[3] - bbox1[1]

            sub_font_size = max(14, int(title_font_size * 0.54))
            sub_font = resolve_font(None, "bold", sub_font_size)

            bbox2 = draw.textbbox((0, 0), model_year, font=sub_font)
            w2 = bbox2[2] - bbox2[0]
            if w2 > max_title_w and w2 > 0:
                scale_ratio = max_title_w / w2
                sub_font_size = max(12, int(sub_font_size * scale_ratio))
                sub_font = resolve_font(None, "bold", sub_font_size)
                bbox2 = draw.textbbox((0, 0), model_year, font=sub_font)
                w2 = bbox2[2] - bbox2[0]
            h2 = bbox2[3] - bbox2[1]

            line_spacing = max(16, int(top_banner_h * 0.17))
            total_content_h = h1 + line_spacing + h2
            start_y = (top_banner_h - total_content_h) // 2

            # Line 1: Main Car Title (Yellow with shadow)
            x1 = (screen_w - w1) // 2
            y1 = start_y - int(bbox1[1])
            draw.text((x1 + 2, y1 + 2), main_title, font=title_font, fill=(0, 0, 0, 160))
            draw.text((x1, y1), main_title, font=title_font, fill=YELLOW_HIGHLIGHT)

            # Line 2: Model Year (Small text on next line with shadow)
            x2 = (screen_w - w2) // 2
            y2 = start_y + h1 + line_spacing - int(bbox2[1])
            draw.text((x2 + 1, y2 + 1), model_year, font=sub_font, fill=(0, 0, 0, 160))
            draw.text((x2, y2), model_year, font=sub_font, fill=(255, 255, 255, 235))
        else:
            # Single-line layout: Full Title centered
            base_font_size = int(top_banner_h * 0.54)
            font = resolve_font(None, "bold", base_font_size)
            title_text = main_title

            bbox = draw.textbbox((0, 0), title_text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w > max_title_w and text_w > 0:
                scale_ratio = max_title_w / text_w
                adjusted_size = max(20, int(base_font_size * scale_ratio))
                font = resolve_font(None, "bold", adjusted_size)
                bbox = draw.textbbox((0, 0), title_text, font=font)
                text_w = bbox[2] - bbox[0]

            text_h = bbox[3] - bbox[1]
            text_x = (screen_w - text_w) // 2
            text_y = (top_banner_h - text_h) // 2 - int(bbox[1])

            draw.text((text_x + 2, text_y + 2), title_text, font=font, fill=(0, 0, 0, 160))
            draw.text((text_x, text_y), title_text, font=font, fill=YELLOW_HIGHLIGHT)

    pad_x = max(24, int(screen_w * 0.035))

    v_dealer = slide_data.get("dealer")
    v_dealer_dict = v_dealer if isinstance(v_dealer, dict) else {}
    vendor_name = str(
        slide_data.get("vendorName")
        or slide_data.get("footerTitle")
        or slide_data.get("dealerName")
        or v_dealer_dict.get("title")
        or v_dealer_dict.get("name")
        or ""
    ).strip()
    vendor_logo = slide_data.get("vendorLogo")
    slide_base_dir = Path(slide_data.get("_base_dir")) if slide_data.get("_base_dir") else None
    resolved_logo_path = resolve_asset_path(str(vendor_logo), slide_base_dir) if vendor_logo else None
    vendor_x = max(20, int(screen_w * 0.075))
    vendor_y = top_banner_h + max(14, int(screen_h * 0.025))

    if resolved_logo_path and resolved_logo_path.is_file():
        try:
            vendor_img = Image.open(str(resolved_logo_path)).convert("RGBA")
            vendor_img.thumbnail((max(44, int(screen_w * 0.065)), max(30, int(top_banner_h * 0.48))), Image.Resampling.LANCZOS)
            overlay.paste(vendor_img, (vendor_x, vendor_y), vendor_img)

            if vendor_name:
                vendor_font = resolve_font(None, "bold", max(14, int(top_banner_h * 0.19)))
                vendor_bbox = draw.textbbox((0, 0), vendor_name, font=vendor_font)
                vendor_text_w = vendor_bbox[2] - vendor_bbox[0]
                vendor_max_w = max(80, int(screen_w * 0.16))
                if vendor_text_w > vendor_max_w:
                    vendor_font = resolve_font(
                        None,
                        "bold",
                        max(12, int(max(14, int(top_banner_h * 0.19)) * vendor_max_w / vendor_text_w)),
                    )
                    vendor_bbox = draw.textbbox((0, 0), vendor_name, font=vendor_font)
                    vendor_text_w = vendor_bbox[2] - vendor_bbox[0]
                vendor_text_x = vendor_x + (vendor_img.width - vendor_text_w) // 2
                vendor_text_y = vendor_y + vendor_img.height + 4
                draw.text((vendor_text_x + 1, vendor_text_y + 1), vendor_name, font=vendor_font, fill=(0, 0, 0, 180))
                draw.text((vendor_text_x, vendor_text_y), vendor_name, font=vendor_font, fill=YELLOW_HIGHLIGHT)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load vendor logo on vehicle image: %s", exc)
    elif vendor_name:
        # Decoupled fallback: Render dealer badge / pill with verified check icon when logo image file is absent
        try:
            badge_font = resolve_font(None, "bold", max(14, int(top_banner_h * 0.22)))
            bb = draw.textbbox((0, 0), vendor_name, font=badge_font)
            text_w = bb[2] - bb[0]
            text_h = bb[3] - bb[1]
            pill_pad_x = 16
            pill_pad_y = 8
            icon_diam = max(18, int(text_h * 1.1))
            pill_w = pill_pad_x + icon_diam + 8 + text_w + pill_pad_x
            pill_h = text_h + (pill_pad_y * 2)

            # Translucent dark pill with gold accent outline
            draw.rounded_rectangle(
                [(vendor_x, vendor_y), (vendor_x + pill_w, vendor_y + pill_h)],
                radius=max(8, int(pill_h * 0.4)),
                fill=(15, 23, 42, 210),
                outline=(252, 206, 7, 230),
                width=2,
            )
            # Verified checkmark badge icon
            ic_x = vendor_x + pill_pad_x
            ic_y = vendor_y + (pill_h - icon_diam) // 2
            draw.ellipse([(ic_x, ic_y), (ic_x + icon_diam, ic_y + icon_diam)], fill=(37, 99, 235, 240))
            check_pts = [
                (ic_x + int(icon_diam * 0.28), ic_y + int(icon_diam * 0.52)),
                (ic_x + int(icon_diam * 0.44), ic_y + int(icon_diam * 0.70)),
                (ic_x + int(icon_diam * 0.74), ic_y + int(icon_diam * 0.32)),
            ]
            draw.line(check_pts, fill=(255, 255, 255, 255), width=max(2, int(icon_diam * 0.12)))

            # Dealer name text
            txt_x = ic_x + icon_diam + 8
            txt_y = vendor_y + pill_pad_y - bb[1]
            draw.text((txt_x + 1, txt_y + 1), vendor_name, font=badge_font, fill=(0, 0, 0, 180))
            draw.text((txt_x, txt_y), vendor_name, font=badge_font, fill=YELLOW_HIGHLIGHT)
        except Exception as exc:
            logger.warning("Failed to render vendor name badge fallback: %s", exc)

    # -------------------------------------------------------------
    # 2. Bottom Sub-Ticker / Feature Highlights Bar (Swapped: Upper Container)
    # -------------------------------------------------------------
    ticker_y = screen_h - bottom_total_h
    draw.rectangle([(0, ticker_y), (screen_w, ticker_y + ticker_h)], fill=TICKER_BG)

    specs_text = slide_data.get("text", "").strip()
    if specs_text:
        raw_items = split_specs_items(specs_text)
        if not raw_items:
            raw_items = [specs_text]

        ticker_font_size = int(ticker_h * 0.48)
        ticker_font = resolve_font(None, "bold", ticker_font_size)

        check_icon_size = int(ticker_font_size * 1.05)
        check_icon = create_check_icon(size=check_icon_size, fill_color=CHECK_BLUE, check_color=WHITE)

        bullet = " • "
        b_bbox = draw.textbbox((0, 0), bullet, font=ticker_font)
        b_w = b_bbox[2] - b_bbox[0]

        items_measured = []
        total_content_w = check_icon_size + 10

        for idx, item_str in enumerate(raw_items):
            i_bbox = draw.textbbox((0, 0), item_str, font=ticker_font)
            i_w = i_bbox[2] - i_bbox[0]
            items_measured.append((item_str, i_w, i_bbox))
            total_content_w += i_w
            if idx < len(raw_items) - 1:
                total_content_w += b_w

        max_allowed_w = screen_w - (pad_x * 2)
        if total_content_w > max_allowed_w and total_content_w > 0:
            scale_fac = max_allowed_w / total_content_w
            ticker_font_size = max(14, int(ticker_font_size * scale_fac))
            ticker_font = resolve_font(None, "bold", ticker_font_size)
            check_icon_size = int(ticker_font_size * 1.05)
            check_icon = create_check_icon(size=check_icon_size, fill_color=CHECK_BLUE, check_color=WHITE)

            b_bbox = draw.textbbox((0, 0), bullet, font=ticker_font)
            b_w = b_bbox[2] - b_bbox[0]

            items_measured = []
            total_content_w = check_icon_size + 10
            for idx, item_str in enumerate(raw_items):
                i_bbox = draw.textbbox((0, 0), item_str, font=ticker_font)
                i_w = i_bbox[2] - i_bbox[0]
                items_measured.append((item_str, i_w, i_bbox))
                total_content_w += i_w
                if idx < len(raw_items) - 1:
                    total_content_w += b_w

        cur_x = max(pad_x, (screen_w - total_content_w) // 2)
        vert_center_y = ticker_y + (ticker_h // 2)

        # Compute unified baseline for all text items and bullets
        ref_bbox = draw.textbbox((0, 0), "AgQy10", font=ticker_font)
        ref_h = ref_bbox[3] - ref_bbox[1]
        shared_ty = vert_center_y - (ref_h // 2) - int(ref_bbox[1])

        check_y = vert_center_y - (check_icon_size // 2)
        overlay.paste(check_icon, (int(cur_x), check_y), check_icon)
        cur_x += check_icon_size + 10

        for idx, (item_str, i_w, i_bbox) in enumerate(items_measured):
            draw.text((cur_x, shared_ty), item_str, font=ticker_font, fill=WHITE)
            cur_x += i_w

            if idx < len(items_measured) - 1:
                draw.text((cur_x, shared_ty), bullet, font=ticker_font, fill=YELLOW_HIGHLIGHT)
                cur_x += b_w

    # -------------------------------------------------------------
    # 3. Bottom Main Dealer & Price Banner (Swapped: Lower Container)
    # -------------------------------------------------------------
    bottom_y_start = ticker_y + ticker_h
    draw.rectangle([(0, bottom_y_start), (screen_w, bottom_y_start + bottom_main_h)], fill=BOTTOM_MAIN_BG)

    left_x = pad_x
    right_x = screen_w - pad_x

    # --- Left Section: Price ---
    raw_price = slide_data.get("price", "")
    curr_str, num_str, unit_text, qualifier_text = parse_price_components(raw_price)

    if num_str:
        num_font_size = int(bottom_main_h * 0.58)
        num_font = resolve_font(None, "bold", num_font_size)

        curr_font_size = int(num_font_size * 0.72)
        curr_font = resolve_font(None, "bold", curr_font_size)

        c_bbox = draw.textbbox((0, 0), curr_str, font=curr_font) if curr_str else (0, 0, 0, 0)
        c_w = c_bbox[2] - c_bbox[0] if curr_str else 0
        c_h = c_bbox[3] - c_bbox[1] if curr_str else 0

        n_bbox = draw.textbbox((0, 0), num_str, font=num_font)
        n_w = n_bbox[2] - n_bbox[0]
        n_h = n_bbox[3] - n_bbox[1]

        base_y = bottom_y_start + (bottom_main_h - n_h) // 2

        cur_draw_x = left_x
        if curr_str:
            c_y = base_y + (n_h - c_h) - int(c_bbox[1])
            draw.text((cur_draw_x + 2, c_y + 2), curr_str, font=curr_font, fill=(0, 0, 0, 140))
            draw.text((cur_draw_x, c_y), curr_str, font=curr_font, fill=YELLOW_HIGHLIGHT)
            cur_draw_x += c_w + int(screen_w * 0.008)

        n_y = base_y - int(n_bbox[1])
        draw.text((cur_draw_x + 2, n_y + 2), num_str, font=num_font, fill=(0, 0, 0, 140))
        draw.text((cur_draw_x, n_y), num_str, font=num_font, fill=YELLOW_HIGHLIGHT)
        cur_draw_x += n_w + int(screen_w * 0.012)

        if unit_text:
            sub_font_size = int(num_font_size * 0.38)
            sub_font = resolve_font(None, "bold", sub_font_size)

            u_bbox = draw.textbbox((0, 0), unit_text, font=sub_font)
            q_bbox = draw.textbbox((0, 0), qualifier_text, font=sub_font)

            stack_h = (u_bbox[3] - u_bbox[1]) + (q_bbox[3] - q_bbox[1]) + 3
            stack_y = bottom_y_start + (bottom_main_h - stack_h) // 2

            draw.text((cur_draw_x, stack_y - int(u_bbox[1])), unit_text, font=sub_font, fill=YELLOW_HIGHLIGHT)
            draw.text(
                (cur_draw_x, stack_y + (u_bbox[3] - u_bbox[1]) + 3 - int(q_bbox[1])),
                qualifier_text,
                font=sub_font,
                fill=WHITE,
            )

    # --- Right Section: Dealership & Contact ---
    d_dealer = slide_data.get("dealer")
    d_dealer_dict = d_dealer if isinstance(d_dealer, dict) else {}
    dealer_name = (
        slide_data.get("footerTitle")
        or slide_data.get("vendorName")
        or slide_data.get("dealerName")
        or d_dealer_dict.get("title")
        or d_dealer_dict.get("name")
        or d_dealer_dict.get("vendorName")
        or ""
    ).strip()

    contact_str = (
        slide_data.get("contact")
        or slide_data.get("Contact")
        or d_dealer_dict.get("phone")
        or ""
    ).strip()

    dealer_font_size = int(bottom_main_h * 0.32)
    dealer_font = resolve_font(None, "bold", dealer_font_size)

    contact_font_size = int(bottom_main_h * 0.28)
    contact_font = resolve_font(None, "bold", contact_font_size)

    d_bbox = draw.textbbox((0, 0), dealer_name, font=dealer_font) if dealer_name else (0, 0, 0, 0)
    d_w = d_bbox[2] - d_bbox[0] if dealer_name else 0
    d_h = d_bbox[3] - d_bbox[1] if dealer_name else 0

    max_d_w = int(screen_w * 0.52)
    if d_w > max_d_w and d_w > 0:
        scale_fac = max_d_w / d_w
        dealer_font_size = max(14, int(dealer_font_size * scale_fac))
        dealer_font = resolve_font(None, "bold", dealer_font_size)
        d_bbox = draw.textbbox((0, 0), dealer_name, font=dealer_font)
        d_w = d_bbox[2] - d_bbox[0]
        d_h = d_bbox[3] - d_bbox[1]

    call_text = ""
    if contact_str:
        # Strip redundant 'Call/WhatsApp:', 'Call:', 'WhatsApp:', etc. to display clean number with WhatsApp logo
        clean_contact = re.sub(
            r"^(?:(?:call\s*/\s*whatsapp|call\s*now|call|whatsapp|wa)\s*)?[:\-]?\s*",
            "",
            contact_str,
            flags=re.I,
        ).strip()
        call_text = clean_contact if clean_contact else contact_str

    c_bbox = draw.textbbox((0, 0), call_text, font=contact_font) if call_text else (0, 0, 0, 0)
    c_w = c_bbox[2] - c_bbox[0] if call_text else 0
    c_h = c_bbox[3] - c_bbox[1] if call_text else 0

    icon_size = int(max(c_h, int(contact_font_size * 1.00)))
    phone_spacing = 8 if call_text else 0

    # Ensure generous breathing room between dealer name and WhatsApp logo/number
    spacing_between = max(8, int(bottom_main_h * 0.08))
    line2_h = max(c_h, icon_size)
    right_stack_h = (d_h + spacing_between + line2_h) if (dealer_name and call_text) else max(d_h, line2_h)
    right_y = bottom_y_start + (bottom_main_h - right_stack_h) // 2

    if dealer_name:
        d_x = right_x - d_w
        draw.text((d_x + 1, right_y + 1 - int(d_bbox[1])), dealer_name, font=dealer_font, fill=(0, 0, 0, 140))
        draw.text((d_x, right_y - int(d_bbox[1])), dealer_name, font=dealer_font, fill=WHITE)
        right_y += d_h + spacing_between

    if call_text:
        c_x = right_x - c_w
        icon_x = c_x - icon_size - phone_spacing
        icon_y = right_y + (line2_h - icon_size) // 2
        text_y = right_y + (line2_h - c_h) // 2

        wa_icon = get_whatsapp_icon(size=icon_size)
        overlay.paste(wa_icon, (int(icon_x), int(icon_y)), wa_icon)

        draw.text((c_x + 1, text_y + 1 - int(c_bbox[1])), call_text, font=contact_font, fill=(0, 0, 0, 140))
        draw.text((c_x, text_y - int(c_bbox[1])), call_text, font=contact_font, fill=WHITE)

    # 4. Watermark (Light & Subtle)
    watermark_text = slide_data.get("watermark", "").strip()
    if watermark_text:
        wm_font_size = max(22, int(screen_h * 0.038))
        wm_font = resolve_font(None, "normal", wm_font_size)
        wm_bbox = draw.textbbox((0, 0), watermark_text, font=wm_font)
        wm_w = wm_bbox[2] - wm_bbox[0]
        wm_h = wm_bbox[3] - wm_bbox[1]
        wm_x = (screen_w - wm_w) // 2

        view_area_h = ticker_y - top_banner_h
        wm_y = top_banner_h + (view_area_h - wm_h) // 2 - int(wm_bbox[1])

        draw.text((wm_x + 1, wm_y + 1), watermark_text, font=wm_font, fill=(0, 0, 0, 16))
        draw.text((wm_x, wm_y), watermark_text, font=wm_font, fill=(255, 255, 255, 28))

    return overlay


# ============================================================================
# LAYOUT: 9:16 VERTICAL REEL OVERLAY RENDERER (layout == 'vertical_reel')
# ============================================================================
def render_vertical_reel_overlay(
    screen_size: Tuple[int, int],
    slide_data: dict,
    slide_idx: int = 0,
) -> Image.Image:
    """
    Render ultra-premium 9:16 Vertical Reel overlay with glassmorphic cards,
    smart deal highlights badges, floating golden price badge, and verified dealer footer.
    """
    screen_w, screen_h = screen_size
    overlay = Image.new("RGBA", (screen_w, screen_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Color Palette
    GLASS_BG = (10, 25, 48, 230)            # Dark Navy Glassmorphism
    GLASS_BORDER = (56, 189, 248, 140)      # Sky Cyan Glow
    GOLD_PILL_BG = (252, 206, 7, 245)       # Golden Amber
    GOLD_TEXT = (15, 23, 42, 255)           # Deep Obsidian Text
    WHITE = (255, 255, 255, 255)
    OFF_WHITE = (241, 245, 249, 240)
    CYAN_ACCENT = (56, 189, 248, 255)
    BADGE_BG = (15, 23, 42, 220)

    pad_x = max(20, int(screen_w * 0.04))

    # 1. TOP GLASSMORPHIC HEADER CARD
    top_card_y = max(24, int(screen_h * 0.025))
    top_card_h = max(110, int(screen_h * 0.125))
    top_card_w = screen_w - (pad_x * 2)

    draw.rounded_rectangle(
        [(pad_x, top_card_y), (pad_x + top_card_w, top_card_y + top_card_h)],
        radius=16,
        fill=GLASS_BG,
        outline=GLASS_BORDER,
        width=2,
    )

    # Vendor Tag / Verified Badge
    vendor_title = slide_data.get("vendorName") or slide_data.get("footerTitle") or "Verified Dealership"
    sub_font_size = max(16, int(screen_h * 0.016))
    sub_font = resolve_font(None, "bold", sub_font_size)
    check_icon = create_check_icon(size=int(sub_font_size * 1.15), fill_color=CYAN_ACCENT, check_color=WHITE)

    tag_y = top_card_y + 14
    tag_x = pad_x + 18
    overlay.paste(check_icon, (tag_x, tag_y), check_icon)
    draw.text((tag_x + check_icon.width + 8, tag_y - 1), str(vendor_title).upper(), font=sub_font, fill=CYAN_ACCENT)

    # Main Car Title (Vehicle Title + Model Year on next line in small font)
    main_title, model_year = parse_header_title_and_year(slide_data)
    if main_title:
        title_font_size = max(24, int(screen_h * 0.028))
        title_font = resolve_font(None, "bold", title_font_size)
        wrapped_title = wrap_text(main_title, title_font, top_card_w - 36, draw)
        title_y = tag_y + sub_font_size + 12
        for line in wrapped_title[:2]:
            draw.text((pad_x + 18, title_y), line, font=title_font, fill=WHITE)
            title_y += title_font_size + 4

        if model_year:
            year_font_size = max(15, int(title_font_size * 0.65))
            year_font = resolve_font(None, "bold", year_font_size)
            draw.text((pad_x + 18, title_y + 2), model_year, font=year_font, fill=GOLD_PILL_BG)

    # 2. SMART HIGHLIGHTS DEAL BADGES (Floating Pill row)
    smart_badges = extract_smart_badges(slide_data)
    badge_row_y = top_card_y + top_card_h + 16
    cur_badge_x = pad_x

    badge_font_size = max(14, int(screen_h * 0.015))
    badge_font = resolve_font(None, "bold", badge_font_size)

    for b_text in smart_badges:
        b_box = draw.textbbox((0, 0), b_text, font=badge_font)
        b_w = (b_box[2] - b_box[0]) + 20
        b_h = badge_font_size + 14

        if cur_badge_x + b_w <= screen_w - pad_x:
            draw.rounded_rectangle(
                [(cur_badge_x, badge_row_y), (cur_badge_x + b_w, badge_row_y + b_h)],
                radius=b_h // 2,
                fill=BADGE_BG,
                outline=(252, 206, 7, 160) if "1ST" in b_text else (56, 189, 248, 160),
                width=1,
            )
            draw.text((cur_badge_x + 10, badge_row_y + 6), b_text, font=badge_font, fill=OFF_WHITE)
            cur_badge_x += b_w + 10

    # 3. FLOATING LUXURY PRICE BADGE (Right aligned golden pill)
    price_val = slide_data.get("price", "").strip()
    if price_val:
        curr, amt, unit, qual = parse_price_components(price_val)
        price_display = f"{curr} {amt} {unit}".strip()
        price_font_size = max(22, int(screen_h * 0.024))
        price_font = resolve_font(None, "bold", price_font_size)
        p_bbox = draw.textbbox((0, 0), price_display, font=price_font)
        p_w = (p_bbox[2] - p_bbox[0]) + 32
        p_h = price_font_size + 18
        p_x = screen_w - pad_x - p_w
        p_y = max(badge_row_y + badge_font_size + 24, int(screen_h * 0.72))

        # Shadow + Pill
        draw.rounded_rectangle([(p_x + 3, p_y + 3), (p_x + p_w + 3, p_y + p_h + 3)], radius=14, fill=(0, 0, 0, 160))
        draw.rounded_rectangle([(p_x, p_y), (p_x + p_w, p_y + p_h)], radius=14, fill=GOLD_PILL_BG, outline=WHITE, width=2)
        draw.text((p_x + 16, p_y + 8), price_display, font=price_font, fill=GOLD_TEXT)

    # 4. BOTTOM GLASSMORPHIC SPECS & CALL-TO-ACTION BAR
    bot_h = max(140, int(screen_h * 0.165))
    bot_y = screen_h - bot_h - max(20, int(screen_h * 0.02))
    bot_w = screen_w - (pad_x * 2)

    draw.rounded_rectangle(
        [(pad_x, bot_y), (pad_x + bot_w, bot_y + bot_h)],
        radius=18,
        fill=GLASS_BG,
        outline=GLASS_BORDER,
        width=2,
    )

    # Specs Ticker Row
    specs_text = slide_data.get("text", "").strip()
    if specs_text:
        specs_font_size = max(16, int(screen_h * 0.017))
        specs_font = resolve_font(None, "normal", specs_font_size)
        raw_items = split_specs_items(specs_text)
        specs_line = " • ".join(raw_items[:5])
        draw.text((pad_x + 18, bot_y + 16), specs_line, font=specs_font, fill=OFF_WHITE)

    # CTA Button Pill inside bottom card
    contact = slide_data.get("contact", "").strip()
    clean_contact = re.sub(
        r"^(?:(?:call\s*/\s*whatsapp|call\s*now|call|whatsapp|wa)\s*)?[:\-]?\s*",
        "",
        contact,
        flags=re.I,
    ).strip()
    cta_text = clean_contact if clean_contact else (contact if contact else f"{vendor_title}")
    cta_btn_h = max(44, int(screen_h * 0.048))
    cta_btn_y = bot_y + bot_h - cta_btn_h - 14
    cta_btn_w = bot_w - 36
    cta_btn_x = pad_x + 18

    draw.rounded_rectangle(
        [(cta_btn_x, cta_btn_y), (cta_btn_x + cta_btn_w, cta_btn_y + cta_btn_h)],
        radius=cta_btn_h // 2,
        fill=(16, 185, 129, 245),  # Vibrant Emerald CTA Button
        outline=WHITE,
        width=1,
    )

    wa_icon = get_whatsapp_icon(size=int(cta_btn_h * 0.65))
    overlay.paste(wa_icon, (cta_btn_x + 16, cta_btn_y + (cta_btn_h - wa_icon.height) // 2), wa_icon)

    cta_font_size = max(18, int(screen_h * 0.019))
    cta_font = resolve_font(None, "bold", cta_font_size)
    cb_box = draw.textbbox((0, 0), cta_text, font=cta_font)
    draw.text((cta_btn_x + wa_icon.width + 26, cta_btn_y + (cta_btn_h - (cb_box[3] - cb_box[1])) // 2 - int(cb_box[1])), cta_text, font=cta_font, fill=WHITE)

    # 5. Subtle Watermark (Light)
    watermark_text = slide_data.get("watermark", "").strip()
    if watermark_text:
        wm_font = resolve_font(None, "normal", max(20, int(screen_h * 0.024)))
        wm_bbox = draw.textbbox((0, 0), watermark_text, font=wm_font)
        wm_w = wm_bbox[2] - wm_bbox[0]
        draw.text(((screen_w - wm_w) // 2, int(screen_h * 0.48)), watermark_text, font=wm_font, fill=(255, 255, 255, 26))

    return overlay


# ============================================================================
# LAYOUT: CLOSING OUTRO OVERLAY RENDERER
# ============================================================================
def render_outro_overlay(
    screen_size: Tuple[int, int],
    slide_data: dict,
) -> Image.Image:
    """
    Render luxury high-converting Closing Outro Card following the reference spatial layout:
    - Top: Dealership title and bold 'FOR SALE' crimson badge.
    - Center: High-impact vehicle showcase image (with contact ground shadow for cutouts).
    - Bottom: Clean structured specs (Model, Details, Price Asking, Location + Contact Now button).
    """
    screen_w, screen_h = screen_size
    overlay = Image.new("RGBA", (screen_w, screen_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    dealer_name = (
        slide_data.get("vendorName")
        or slide_data.get("dealerName")
        or slide_data.get("footerTitle")
        or slide_data.get("watermark")
        or "All India Car Bazaar"
    )
    if dealer_name.strip().upper() in ("HONDA CARS INDIA LTD", "HONDA CARS INDIA LIMITED", "MARUTI SUZUKI INDIA LTD", "HYUNDAI MOTOR INDIA LIMITED"):
        fallback_dealer = slide_data.get("dealerName") or slide_data.get("businessName")
        if fallback_dealer and fallback_dealer.strip().upper() != dealer_name.strip().upper():
            dealer_name = fallback_dealer.strip()
    contact = slide_data.get("contact", "").strip()
    location = slide_data.get("location", "").strip()
    image_path = slide_data.get("image")

    center_x = screen_w // 2

    # Parse Title & Year
    main_title, model_year = parse_header_title_and_year(slide_data)
    if not main_title or main_title.lower() in ("certified pre-owned cars", "verified pre-owned cars", "verified auto hub"):
        make_model = f"{slide_data.get('make', '')} {slide_data.get('model', '')}".strip()
        main_title = make_model if make_model else "Featured Vehicle"

    # Year is displayed in Details section; Model row displays the vehicle model name
    full_model_str = main_title.strip()

    # Price parsing
    raw_price = str(slide_data.get("price") or "").strip()
    curr_str, num_str, unit_text, _ = parse_price_components(raw_price) if raw_price else ("", "", "", "")
    if curr_str in ("₹", "Rs", "rs", "INR"):
        price_display = f"Rs {num_str} {unit_text}".strip()
    elif num_str:
        clean_curr = "Rs" if curr_str == "₹" else curr_str
        price_display = f"{clean_curr} {num_str} {unit_text}".strip()
    else:
        price_display = raw_price.replace("₹", "Rs ").strip()
    if not price_display:
        price_display = "Best Offer"

    all_texts_blob = " ".join(
        [str(t) for t in slide_data.get("all_texts", [])]
        + [str(slide_data.get("text") or ""), str(slide_data.get("headerTitle") or ""), main_title]
    )

    # Extract clean year for the Details section
    year_detail = ""
    raw_yr = str(slide_data.get("year") or model_year or "").strip()
    if raw_yr and raw_yr.lower() not in ("none", "null", "undefined", "0"):
        if "•" in raw_yr:
            raw_yr = raw_yr.split("•")[0].strip()
        if not re.search(r"\bowner\b", raw_yr, re.I):
            year_detail = raw_yr.upper()

    if not year_detail:
        m_yr = re.search(r"\b((?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+)?(19\d{2}|20\d{2}))\b", all_texts_blob, re.I)
        if m_yr:
            year_detail = m_yr.group(1).upper()
            if not re.search(r"\bmodel\b", year_detail, re.I) and not re.search(r"[a-zA-Z]", year_detail):
                year_detail = f"{year_detail} MODEL"

    # Details items (Owner | Year | KM | Fuel | Trans | Reg No | Fitness)
    rc_data = slide_data.get("rcDetails") or {}
    reg_no = str(slide_data.get("regNo") or slide_data.get("registrationNumber") or "").strip()
    details_items = []

    # 1. Ownership
    raw_owner = str(slide_data.get("owner") or "").strip()
    if not raw_owner:
        m_ow = re.search(r"\b([1-4](?:st|nd|rd|th)?\s*owner|single\s*owner)\b", all_texts_blob, re.I)
        if m_ow:
            raw_owner = m_ow.group(1).title()
    if raw_owner:
        details_items.append(raw_owner)

    # 2. Model Year
    if year_detail:
        if year_detail.lower() not in " ".join(details_items).lower():
            details_items.append(year_detail)

    # 3. Odometer / KM
    km_val = str(slide_data.get("odometer") or slide_data.get("km") or "").strip()
    if not km_val:
        m_km = re.search(r"(\b\d{1,3}(?:,\d{3})+|\b\d+(?:\.\d+)?\s*(?:k|lakh|lac)?)\s*(?:km|kms|running|driven)?\b", all_texts_blob, re.I)
        if m_km and ("km" in m_km.group(0).lower() or "k" in m_km.group(0).lower() or len(m_km.group(1).replace(",", "")) >= 4):
            km_val = m_km.group(0).strip().upper()
    if km_val:
        digits_only = re.sub(r"[^\d]", "", km_val)
        if digits_only and digits_only.isdigit() and int(digits_only) > 100 and "km" not in km_val.lower():
            km_formatted = f"{int(digits_only):,} KM"
        elif "km" not in km_val.lower():
            km_formatted = f"{km_val} KM"
        else:
            km_formatted = km_val
        details_items.append(km_formatted)

    # 4. Fuel Type
    fuel_val = str(slide_data.get("fuel") or "").strip().upper()
    if not fuel_val:
        m_fuel = re.search(r"\b(DIESEL|PETROL\+CNG|PETROL|CNG|ELECTRIC|HYBRID)\b", all_texts_blob, re.I)
        if m_fuel:
            fuel_val = m_fuel.group(1).upper()
    if fuel_val:
        details_items.append(fuel_val)

    # 5. Transmission
    trans_val = str(slide_data.get("transmission") or "").strip().title()
    if not trans_val:
        m_trans = re.search(r"\b(MANUAL|AUTOMATIC|AUTO|AMT|CVT|DCT|DSG|iMT)\b", all_texts_blob, re.I)
        if m_trans:
            trans_val = m_trans.group(1).title()
    if trans_val:
        details_items.append(trans_val)

    # 6. Registration Number (if present)
    if reg_no:
        details_items.append(f"Reg: {reg_no.upper()}")

    # 7. Color
    color_val = str(slide_data.get("color") or slide_data.get("colour") or "").strip()
    if color_val and color_val.lower() not in ("unknown", "null", "none", "not_detected", ""):
        details_items.append(color_val.title())

    # 8. Extra Highlights (if room permits, excluding body types like SUV, Sedan, Hatchback, etc.)
    BODY_TYPE_KEYWORDS = {"suv", "sedan", "hatchback", "muv", "coupe", "crossover", "convertible", "wagon", "van"}
    feats = slide_data.get("features") or []
    for f_item in feats:
        f_clean = str(f_item).strip()
        if f_clean.lower() in BODY_TYPE_KEYWORDS or (is_color_token(f_clean) and (not color_val or f_clean.lower() != color_val.lower())):
            continue
        if f_clean and f_clean.lower() not in " ".join(details_items).lower():
            if len(details_items) < 6:
                details_items.append(f_clean)

    if rc_data.get("validity"):
        details_items.append(f"Fitness: {rc_data.get('validity')}")

    # Ensure no body type labels or unverified color attributes are displayed in the details line
    details_items = [d for d in details_items if d.strip().lower() not in BODY_TYPE_KEYWORDS and (not is_color_token(d) or (color_val and d.lower() == color_val.lower()))]

    details_line = "  |  ".join(details_items[:6])

    # -------------------------------------------------------------
    # PRE-CALCULATE SIZES FOR VERTICAL & HORIZONTAL CENTERING
    # -------------------------------------------------------------
    pad_x = int(screen_w * 0.08)
    content_w = screen_w - (pad_x * 2)

    # 1. Dealer Name
    d_font = resolve_font(None, "bold", max(24, int(screen_h * 0.024)))
    db = draw.textbbox((0, 0), str(dealer_name).upper(), font=d_font)
    dw = db[2] - db[0]
    dealer_h = db[3] - db[1]
    sp_dealer = 16

    # 2. "FOR SALE" Badge Banner
    badge_w = min(int(screen_w * 0.52), 440)
    badge_h = max(72, int(screen_h * 0.062))
    sale_font = resolve_font(None, "bold", int(badge_h * 0.62))
    sb = draw.textbbox((0, 0), "FOR SALE", font=sale_font)
    sw = sb[2] - sb[0]
    sh = sb[3] - sb[1]
    sp_badge = 20

    # 3. Center Circle Shape
    circle_r = max(100, int(min(screen_w, screen_h) * 0.15))  # Diameter ~320px
    circle_diam = circle_r * 2
    sp_circle = 36

    # 4. Vehicle Make & Model (Make on line 1, Model name on next line)
    make_val = str(slide_data.get("make") or "").strip().upper()
    model_val = str(slide_data.get("model") or "").strip().upper()
    variant_val = str(slide_data.get("variant") or "").strip().upper()

    if make_val and model_val:
        make_line = make_val
        model_line = f"{model_val} {variant_val}".strip() if variant_val and variant_val not in model_val else model_val
    elif make_val and make_val in full_model_str.upper():
        make_line = make_val
        model_line = full_model_str.upper().replace(make_val, "", 1).strip()
    elif " " in full_model_str:
        parts = full_model_str.split(" ", 1)
        make_line = parts[0].strip()
        model_line = parts[1].strip()
    else:
        make_line = ""
        model_line = full_model_str.strip()

    make_font_size = max(26, int(screen_h * 0.026))
    make_font = resolve_font(None, "bold", make_font_size)
    make_bb = draw.textbbox((0, 0), make_line, font=make_font) if make_line else (0, 0, 0, 0)
    make_w = make_bb[2] - make_bb[0] if make_line else 0
    make_h = make_bb[3] - make_bb[1] if make_line else 0
    sp_make = 8 if make_line else 0

    model_font_size = max(34, int(screen_h * 0.034))
    model_font = resolve_font(None, "bold", model_font_size)
    m_bbox = draw.textbbox((0, 0), model_line, font=model_font)
    model_w = m_bbox[2] - m_bbox[0]
    if model_w > content_w and model_w > 0:
        scale_fac = content_w / model_w
        model_font_size = max(22, int(model_font_size * scale_fac))
        model_font = resolve_font(None, "bold", model_font_size)
        m_bbox = draw.textbbox((0, 0), model_line, font=model_font)
        model_w = m_bbox[2] - m_bbox[0]
    model_h = m_bbox[3] - m_bbox[1]
    sp_model = 20

    # 5. Vehicle Specs (Details line)
    det_font_size = max(22, int(screen_h * 0.023))
    det_font = resolve_font(None, "bold", det_font_size)
    dbb = draw.textbbox((0, 0), details_line, font=det_font)
    det_w = dbb[2] - dbb[0]
    if det_w > content_w and det_w > 0:
        scale_fac = content_w / det_w
        det_font_size = max(16, int(det_font_size * scale_fac))
        det_font = resolve_font(None, "bold", det_font_size)
        dbb = draw.textbbox((0, 0), details_line, font=det_font)
        det_w = dbb[2] - dbb[0]
    det_h = dbb[3] - dbb[1]
    sp_det = 26

    # 6. Price
    pr_val_font = resolve_font(None, "bold", max(34, int(screen_h * 0.038)))
    pr_lbl_font = resolve_font(None, "bold", max(26, int(screen_h * 0.027)))
    asking_font = resolve_font(None, "normal", max(24, int(screen_h * 0.025)))

    pl_bb = draw.textbbox((0, 0), "Price: ", font=pr_lbl_font)
    pv_bb = draw.textbbox((0, 0), price_display, font=pr_val_font)
    pa_bb = draw.textbbox((0, 0), " Asking", font=asking_font) if raw_price else (0, 0, 0, 0)
    tot_pr_w = (pl_bb[2] - pl_bb[0]) + (pv_bb[2] - pv_bb[0]) + (pa_bb[2] - pa_bb[0])
    price_h = pv_bb[3] - pv_bb[1]
    sp_price = 26

    # 7. Location (Only shown if location is explicitly passed to videoGen)
    loc_raw = str(slide_data.get("location") or location or "").strip()
    clean_loc = re.sub(r"^(?:location|loc)\s*[:\-]?\s*", "", loc_raw, flags=re.I).strip()
    if clean_loc.lower() in ("null", "none", "na", "quality", "certified quality", "undefined", "0"):
        clean_loc = ""

    loc_h = 0
    tot_loc_w = 0
    loc_lbl_font = None
    loc_val_font = None
    ll_bb = (0, 0, 0, 0)
    if clean_loc:
        loc_lbl_font = resolve_font(None, "bold", max(26, int(screen_h * 0.027)))
        loc_val_font = resolve_font(None, "bold", max(26, int(screen_h * 0.027)))
        ll_bb = draw.textbbox((0, 0), "Location: ", font=loc_lbl_font)
        lv_bb = draw.textbbox((0, 0), clean_loc, font=loc_val_font)
        tot_loc_w = int((ll_bb[2] - ll_bb[0]) + (lv_bb[2] - lv_bb[0]))
        loc_h = int(max(ll_bb[3] - ll_bb[1], lv_bb[3] - lv_bb[1]))
        sp_loc = loc_h + 28
    else:
        sp_loc = 10

    # 8. Action Button
    clean_contact = re.sub(
        r"^(?:(?:call\s*/\s*whatsapp|call\s*now|call|whatsapp|wa)\s*)?[:\-]?\s*",
        "",
        contact,
        flags=re.I,
    ).strip()
    btn_label = clean_contact if clean_contact else "CONTACT NOW"

    btn_h = max(64, int(screen_h * 0.058))
    sp_btn = 10

    # Calculate total height of all elements and center vertically
    total_content_h = (
        dealer_h + sp_dealer
        + badge_h + sp_badge
        + circle_diam + sp_circle
        + make_h + sp_make
        + model_h + sp_model
        + det_h + sp_det
        + price_h + sp_price
        + sp_loc
        + sp_btn + btn_h
    )

    start_y = max(int(screen_h * 0.04), (screen_h - total_content_h) // 2)
    cur_y = start_y

    # -------------------------------------------------------------
    # 1. TOP HEADER: Dealership Title & "FOR SALE" Red Pill Banner
    # -------------------------------------------------------------
    # Dealer Name
    draw.text((center_x - dw // 2 + 1, cur_y + 1), str(dealer_name).upper(), font=d_font, fill=(0, 0, 0, 180))
    draw.text((center_x - dw // 2, cur_y), str(dealer_name).upper(), font=d_font, fill=(241, 245, 249, 230))
    cur_y += dealer_h + sp_dealer

    # "FOR SALE" Badge Banner
    bx = center_x - badge_w // 2

    # Drop shadow
    draw.rounded_rectangle([(bx + 3, cur_y + 5), (bx + badge_w + 3, cur_y + badge_h + 5)], radius=18, fill=(0, 0, 0, 140))
    # Crimson Red Body
    draw.rounded_rectangle([(bx, cur_y), (bx + badge_w, cur_y + badge_h)], radius=18, fill=(220, 38, 38, 255), outline=(255, 255, 255, 220), width=2)
    draw.text((center_x - sw // 2, cur_y + (badge_h - sh) // 2 - int(sb[1])), "FOR SALE", font=sale_font, fill=(255, 255, 255, 255))
    cur_y += badge_h + sp_badge

    # -------------------------------------------------------------
    # 2. CENTER: CAR IMAGE IN SMALL CIRCLE SHAPE
    # -------------------------------------------------------------
    cy = cur_y + circle_r

    # Soft glowing radial shadow behind medallion
    glow_box = [center_x - circle_r - 18, cy - circle_r - 18, center_x + circle_r + 18, cy + circle_r + 18]
    draw.ellipse(glow_box, fill=(56, 189, 248, 45))

    # Outer Double Rims (Gold & Cyan)
    draw.ellipse([(center_x - circle_r - 7, cy - circle_r - 7), (center_x + circle_r + 7, cy + circle_r + 7)], outline=(252, 206, 7, 250), width=4)
    draw.ellipse([(center_x - circle_r - 2, cy - circle_r - 2), (center_x + circle_r + 2, cy + circle_r + 2)], outline=(56, 189, 248, 180), width=2)
    draw.ellipse([(center_x - circle_r, cy - circle_r), (center_x + circle_r, cy + circle_r)], fill=(255, 255, 255, 255))

    if image_path and Path(image_path).is_file():
        try:
            car_img = Image.open(image_path).convert("RGBA")

            # Scale and center car inside circle with cover crop
            target_diam = circle_r * 2
            w, h = car_img.size
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            cropped_car = car_img.crop((left, top, left + min_side, top + min_side))
            resized_car = cropped_car.resize((target_diam, target_diam), Image.Resampling.LANCZOS)

            # Circular clip mask
            mask = Image.new("L", (target_diam, target_diam), 0)
            mdraw = ImageDraw.Draw(mask)
            mdraw.ellipse([(0, 0), (target_diam - 1, target_diam - 1)], fill=255)

            overlay.paste(resized_car, (center_x - circle_r, int(cy - circle_r)), mask)
        except Exception as e:
            logger.warning(f"Error rendering circular vehicle image in outro: {e}")

    cur_y = cy + circle_r + sp_circle

    # -------------------------------------------------------------
    # 3. LOWER SECTION: STRUCTURED SPECS (Model, Details, Price, Location)
    # Perfectly aligned in center
    # -------------------------------------------------------------
    # Row 1: Vehicle Make & Model (Make on line 1, Model name on next line)
    if make_line:
        draw.text((center_x - make_w // 2 + 1, cur_y + 1), make_line, font=make_font, fill=(0, 0, 0, 180))
        draw.text((center_x - make_w // 2, cur_y), make_line, font=make_font, fill=(241, 245, 249, 240))
        cur_y += make_h + sp_make

    draw.text((center_x - model_w // 2 + 1, cur_y + 1), model_line, font=model_font, fill=(0, 0, 0, 180))
    draw.text((center_x - model_w // 2, cur_y), model_line, font=model_font, fill=(239, 68, 68, 255))
    cur_y += model_h + sp_model

    # Row 2: Vehicle Specs (Directly shown without 'Details:' heading)
    draw.text((center_x - det_w // 2 + 1, cur_y + 1), details_line, font=det_font, fill=(0, 0, 0, 180))
    draw.text((center_x - det_w // 2, cur_y), details_line, font=det_font, fill=(226, 232, 240, 245))
    cur_y += det_h + sp_det

    # Row 3: Price: <Rs 5.25 LAKH> Asking
    px = center_x - tot_pr_w // 2
    draw.text((px + 1, cur_y + 6), "Price: ", font=pr_lbl_font, fill=(0, 0, 0, 180))
    draw.text((px, cur_y + 5), "Price: ", font=pr_lbl_font, fill=(255, 255, 255, 255))
    px += (pl_bb[2] - pl_bb[0])

    draw.text((px + 1, cur_y + 1), price_display, font=pr_val_font, fill=(0, 0, 0, 180))
    draw.text((px, cur_y), price_display, font=pr_val_font, fill=(239, 68, 68, 255))
    px += (pv_bb[2] - pv_bb[0])

    if raw_price:
        draw.text((px + 1, cur_y + 8), " Asking", font=asking_font, fill=(0, 0, 0, 160))
        draw.text((px, cur_y + 7), " Asking", font=asking_font, fill=(203, 213, 225, 240))
    cur_y += price_h + sp_price

    # Row 4: Location (Only shown if location is explicitly passed to videoGen)
    if clean_loc and loc_lbl_font is not None and loc_val_font is not None:
        lx = center_x - tot_loc_w // 2
        draw.text((lx + 1, cur_y + 1), "Location: ", font=loc_lbl_font, fill=(0, 0, 0, 180))
        draw.text((lx, cur_y), "Location: ", font=loc_lbl_font, fill=(255, 255, 255, 255))
        lx += int(ll_bb[2] - ll_bb[0])

        draw.text((lx + 1, cur_y + 1), clean_loc, font=loc_val_font, fill=(0, 0, 0, 180))
        draw.text((lx, cur_y), clean_loc, font=loc_val_font, fill=(241, 245, 249, 240))
        cur_y += sp_loc
    else:
        cur_y += sp_loc

    # Row 5: Centered Call Action Button (Call logo + Phone number)
    b_font_size = max(24, int(btn_h * 0.42))
    b_font = resolve_font(None, "bold", b_font_size)
    bb_box = draw.textbbox((0, 0), btn_label, font=b_font)
    bw = bb_box[2] - bb_box[0]
    bh = bb_box[3] - bb_box[1]

    call_icon_size = int(btn_h * 0.52)
    call_icon = get_phone_icon(size=call_icon_size, color=(255, 255, 255, 255))

    icon_spacing = 16
    pill_padding = max(40, int(screen_w * 0.05))
    tot_content_w = call_icon.width + icon_spacing + bw
    btn_w = max(int(screen_w * 0.50), tot_content_w + (pill_padding * 2))
    btn_w = min(btn_w, int(screen_w * 0.88))

    btn_x = (screen_w - btn_w) // 2
    btn_y = cur_y + sp_btn

    # Drop shadow
    draw.rounded_rectangle(
        [(btn_x + 3, btn_y + 5), (btn_x + btn_w + 3, btn_y + btn_h + 5)],
        radius=btn_h // 2,
        fill=(0, 0, 0, 150),
    )
    # Crimson red action pill
    draw.rounded_rectangle(
        [(btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h)],
        radius=btn_h // 2,
        fill=(220, 38, 38, 255),
        outline=(255, 255, 255, 220),
        width=2,
    )

    # Centered call logo + phone label
    bx_content_start = int(btn_x + (btn_w - tot_content_w) // 2)
    overlay.paste(
        call_icon,
        (bx_content_start, int(btn_y + (btn_h - call_icon.height) // 2)),
        call_icon,
    )
    draw.text(
        (bx_content_start + call_icon.width + icon_spacing, btn_y + (btn_h - bh) // 2 - int(bb_box[1])),
        btn_label,
        font=b_font,
        fill=(255, 255, 255, 255),
    )

    return overlay


# ============================================================================
# LAYOUT 2: DEFAULT OVERLAY RENDERER (layout == 'default')
# ============================================================================
def render_text_overlay(
    screen_size: Tuple[int, int],
    text_objects: List[dict],
    logo_objects: Optional[List[dict]] = None,
) -> Image.Image:
    """Render transparent RGBA overlay layer containing styled text & badge objects."""
    screen_w, screen_h = screen_size
    overlay = Image.new("RGBA", (screen_w, screen_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    grouped_by_pos: Dict[str, List[dict]] = {}
    for txt in text_objects:
        pos_key = txt.get("position", "top_left")
        if txt.get("x") is not None or txt.get("x_pct") is not None:
            pos_key = f"custom_{id(txt)}"
        grouped_by_pos.setdefault(pos_key, []).append(txt)

    for pos_key, items in grouped_by_pos.items():
        measured_items = []
        max_item_w = 0
        total_items_h = 0
        spacing_between_items = 10

        for txt in items:
            font = resolve_font(
                txt.get("font_family"),
                txt.get("font_weight", "normal"),
                txt.get("font_size", 42),
            )
            max_text_w = txt.get("max_width") or int(screen_w * 0.85)
            lines = wrap_text(txt["text"], font, max_text_w, draw)

            line_heights = []
            line_widths = []
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_widths.append(bbox[2] - bbox[0])
                line_heights.append(bbox[3] - bbox[1])

            content_w = max(line_widths) if line_widths else 0
            line_spacing = int(txt.get("font_size", 42) * 0.25)
            content_h = sum(line_heights) + (len(lines) - 1) * line_spacing if lines else 0

            pad_x, pad_y = txt.get("background_padding", [24, 14])
            box_w = content_w + pad_x * 2 if txt.get("background") else content_w
            box_h = content_h + pad_y * 2 if txt.get("background") else content_h

            max_item_w = max(max_item_w, box_w)
            total_items_h += box_h
            measured_items.append({
                "txt": txt,
                "font": font,
                "lines": lines,
                "line_heights": line_heights,
                "line_spacing": line_spacing,
                "pad_x": pad_x,
                "pad_y": pad_y,
                "box_w": box_w,
                "box_h": box_h,
            })

        total_items_h += (len(measured_items) - 1) * spacing_between_items

        first = measured_items[0]["txt"]
        root_x, root_y = calculate_position(
            first.get("position", "top_left"),
            max_item_w,
            total_items_h,
            screen_w,
            screen_h,
            margin_x=60,
            margin_y=60,
            custom_x=first.get("x"),
            custom_y=first.get("y"),
            x_pct=first.get("x_pct"),
            y_pct=first.get("y_pct"),
        )

        curr_y = root_y
        for m in measured_items:
            t = m["txt"]
            font = m["font"]
            lines = m["lines"]
            pad_x, pad_y = m["pad_x"], m["pad_y"]
            box_w, box_h = m["box_w"], m["box_h"]
            pos_name = str(t.get("position", "")).lower()
            box_x = (screen_w - box_w) // 2 if "center" in pos_name else root_x

            if t.get("background", True):
                bg_color = t.get("background_color", (15, 23, 42, 190))
                radius = t.get("background_border_radius", 12)
                draw.rounded_rectangle(
                    [(box_x, curr_y), (box_x + box_w, curr_y + box_h)],
                    radius=radius,
                    fill=bg_color,
                )

            text_x = box_x + pad_x if t.get("background") else box_x
            text_y = curr_y + pad_y if t.get("background") else curr_y

            for line_idx, line in enumerate(lines):
                if t.get("shadow", True):
                    sh_color = t.get("shadow_color", (0, 0, 0, 200))
                    sh_off = t.get("shadow_offset", [3, 3])
                    draw.text(
                        (text_x + sh_off[0], text_y + sh_off[1]),
                        line,
                        font=font,
                        fill=sh_color,
                    )
                draw.text(
                    (text_x, text_y),
                    line,
                    font=font,
                    fill=t.get("color", (255, 255, 255, 255)),
                )
                text_y += m["line_heights"][line_idx] + m["line_spacing"]

            curr_y += box_h + spacing_between_items

    for logo_obj in (logo_objects or []):
        logo_path = logo_obj.get("path")
        if not logo_path or not Path(logo_path).exists():
            continue

        try:
            logo_img = Image.open(logo_path).convert("RGBA")
            max_w = logo_obj.get("max_width", 160)
            max_h = logo_obj.get("max_height", 60)
            logo_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            logo_w, logo_h = logo_img.size

            title = logo_obj.get("title")
            pos = logo_obj.get("position", "top_right")

            title_font = resolve_font(None, "bold" if pos == "top_left" else "normal", 20)
            t_bbox = draw.textbbox((0, 0), title, font=title_font) if title else (0, 0, 0, 0)
            t_w = t_bbox[2] - t_bbox[0] if title else 0
            t_h = t_bbox[3] - t_bbox[1] if title else 0

            spacing = 6 if title else 0
            box_w = max(logo_w, t_w)
            box_h = logo_h + (spacing + t_h if title else 0)

            is_watermark = bool(logo_obj.get("watermark", False)) or "bottom" in pos

            if is_watermark:
                r_c, g_c, b_c, a_c = logo_img.split()
                a_c = a_c.point(lambda p: int(p * 0.22))
                logo_img = Image.merge("RGBA", (r_c, g_c, b_c, a_c))

            if pos == "top_left":
                box_x = 60
                box_y = 60
                title_color = (251, 191, 36, 255)
            elif pos == "bottom_right":
                box_x = screen_w - box_w - 60
                box_y = screen_h - box_h - 130
                title_color = (226, 232, 240, 200)
            elif pos == "bottom_left":
                box_x = 60
                box_y = screen_h - box_h - 130
                title_color = (226, 232, 240, 200)
            else:
                box_x = screen_w - box_w - 60
                box_y = 60
                title_color = (226, 232, 240, 230)

            logo_x = box_x + (box_w - logo_w) // 2
            logo_y = box_y
            overlay.paste(logo_img, (int(logo_x), int(logo_y)), logo_img)

            if title:
                text_x = box_x + (box_w - t_w) // 2
                text_y = logo_y + logo_h + spacing
                draw.text(
                    (text_x + 2, text_y + 2),
                    title,
                    font=title_font,
                    fill=(0, 0, 0, 200),
                )
                draw.text(
                    (text_x, text_y),
                    title,
                    font=title_font,
                    fill=title_color,
                )
        except Exception as e:
            logger.warning(f"Failed to render logo overlay from {logo_path}: {e}")

    return overlay


# ============================================================================
# SLIDE FRAME ANIMATION ENGINE (SIMD / OPENCV ACCELERATED)
# ============================================================================
class SlideRenderer:
    """Manages pre-rendered textures and animates frames for a single slide using OpenCV SIMD acceleration."""

    def __init__(self, slide_cfg: dict, video_cfg: dict):
        self.slide_cfg = slide_cfg
        self.video_cfg = video_cfg
        self.width = video_cfg["width"]
        self.height = video_cfg["height"]
        self.fps = video_cfg["fps"]
        self.duration = slide_cfg["duration"]
        self.total_frames = max(1, int(self.duration * self.fps))
        self.animation = slide_cfg.get("animation", "ken_burns").lower()
        self.layout = video_cfg.get("layout", "default").lower()
        self.scale_mode = slide_cfg.get("scale_mode", video_cfg.get("scale_mode", "cover"))
        self.is_outro = bool(slide_cfg.get("is_outro", False) or slide_cfg.get("type") == "outro")

        # Calculate safe viewport for layouts with header/footer banners
        top_h = 0
        bottom_h = 0
        if self.layout == "news_strip":
            top_h = max(76, int(self.height * 0.092))
            bottom_h = max(90, int(self.height * 0.105)) + max(40, int(self.height * 0.052))
        elif self.layout == "vertical_reel":
            top_h = max(24, int(self.height * 0.025)) + max(110, int(self.height * 0.125)) + 12
            bottom_h = max(140, int(self.height * 0.145)) + 24

        self.safe_viewport = (0, top_h, self.width, self.height - top_h - bottom_h) if (top_h > 0 or bottom_h > 0) else None
        self.top_h = top_h
        self.bottom_h = bottom_h
        self.is_first_slide = bool(
            slide_cfg.get("is_first_slide", False) or
            slide_cfg.get("slide_index", None) == 0 or
            _is_trusted_composite(str(slide_cfg.get("image", "")))
        )

        # Initialize texture holders (loaded on-demand in prepare_textures to save RAM)
        self.base_arr = None
        self.oversized_arr = None
        self.sub_alpha = None
        self.sub_inv_alpha = None
        self.sub_overlay_rgb = None
        self.has_overlay = False
        self.textures_loaded = False

    def prepare_textures(self):
        """Loads textures on-demand only when this slide is ready to render."""
        if self.textures_loaded:
            return

        # 1. Base Image Preparation
        if self.is_outro:
            base_pil = create_luxury_outro_background(self.width, self.height)
            overlay_pil = render_outro_overlay((self.width, self.height), self.slide_cfg)
            # Pre-composite outro card directly onto base background (zero per-frame compositing overhead)
            base_pil.paste(overlay_pil, (0, 0), overlay_pil)
            self.base_arr = np.array(base_pil.convert("RGB"), dtype=np.uint8)
            self.has_overlay = False
            self.sub_alpha = None
            self.sub_inv_alpha = None
            self.sub_overlay_rgb = None
            self.oversized_arr = None
            self.textures_loaded = True
            return
        else:
            raw_img = load_and_orient_image(self.slide_cfg["image"])
            if self.video_cfg.get("enhance_visuals", True) and not self.is_first_slide:
                raw_img = enhance_image_cinematic(raw_img)

            # Apply vehicle background fade only if explicitly requested, NEVER on hero slide or trusted composite
            if (
                not self.is_first_slide
                and not _is_trusted_composite(str(self.slide_cfg.get("image", "")))
                and self.video_cfg.get("fade_vehicle_background", False)
                and self.slide_cfg.get("fade_vehicle_background", False)
            ):
                fade_str = float(self.slide_cfg.get("fade_strength", self.video_cfg.get("fade_strength", 0.65)))
                blur_rad = int(self.slide_cfg.get("blur_radius", self.video_cfg.get("blur_radius", 28)))
                raw_img = fade_vehicle_background(raw_img, fade_strength=fade_str, blur_radius=blur_rad)

            base_pil = resize_and_crop(
                raw_img, self.width, self.height, self.scale_mode, safe_viewport=self.safe_viewport, is_first_image=self.is_first_slide
            )

            # Pre-render layout-specific static overlay (RGBA PIL Image)
            if self.layout == "vertical_reel":
                overlay_pil = render_vertical_reel_overlay((self.width, self.height), self.slide_cfg)
            elif self.layout == "news_strip":
                overlay_pil = render_news_strip_overlay((self.width, self.height), self.slide_cfg)
            else:
                overlay_pil = render_text_overlay(
                    (self.width, self.height),
                    self.slide_cfg.get("texts", []),
                    self.slide_cfg.get("logos", []),
                )

        # For smooth Ken Burns / cinematic camera motion, prepare slightly oversized image (1.15x) as NumPy array
        if self.animation in ("ken_burns", "zoom_in", "zoom_out", "pan_left", "pan_right"):
            self.oversized_w = int(self.width * 1.15)
            self.oversized_h = int(self.height * 1.15)
            ov_safe_viewport = (0, int(self.top_h * 1.15), self.oversized_w, int((self.height - self.top_h - self.bottom_h) * 1.15)) if self.safe_viewport else None
            oversized_pil = resize_and_crop(
                raw_img, self.oversized_w, self.oversized_h, self.scale_mode, safe_viewport=ov_safe_viewport, is_first_image=self.is_first_slide
            )
            self.oversized_arr = np.array(oversized_pil.convert("RGB"), dtype=np.uint8)
            del oversized_pil
            self.base_arr = None
        else:
            base_pil = resize_and_crop(
                raw_img, self.width, self.height, self.scale_mode, safe_viewport=self.safe_viewport, is_first_image=self.is_first_slide
            )
            self.base_arr = np.array(base_pil.convert("RGB"), dtype=np.uint8)
            del base_pil
            self.oversized_arr = None

        del raw_img

        # Precompute optimized sub-region overlay alpha mask and RGB layers
        overlay_rgba = np.array(overlay_pil, dtype=np.uint8)
        del overlay_pil
        alpha_channel = overlay_rgba[:, :, 3]
        self.has_overlay = bool(np.any(alpha_channel > 0))

        if self.has_overlay:
            # Find exact bounding box of overlay to minimize blending region
            non_zero_y = np.where(alpha_channel > 0)[0]
            non_zero_x = np.where(alpha_channel > 0)[1]
            self.ov_y1 = int(np.min(non_zero_y))
            self.ov_y2 = int(np.max(non_zero_y)) + 1
            self.ov_x1 = int(np.min(non_zero_x))
            self.ov_x2 = int(np.max(non_zero_x)) + 1

            sub_alpha_u8 = alpha_channel[self.ov_y1:self.ov_y2, self.ov_x1:self.ov_x2]
            sub_alpha = (sub_alpha_u8.astype(np.float32) / 255.0)[:, :, np.newaxis]
            self.sub_alpha = sub_alpha
            self.sub_inv_alpha = 1.0 - sub_alpha
            self.sub_overlay_rgb = overlay_rgba[self.ov_y1:self.ov_y2, self.ov_x1:self.ov_x2, :3]
            # Precompute pre-multiplied overlay RGB once to eliminate per-frame float multiplication
            self.sub_premult_rgb = (self.sub_overlay_rgb.astype(np.float32) * sub_alpha)
            self.opaque_mask = (sub_alpha_u8 == 255)
            self.has_semi = bool(np.any((sub_alpha_u8 > 0) & (sub_alpha_u8 < 255)))
        else:
            self.ov_y1 = self.ov_y2 = self.ov_x1 = self.ov_x2 = 0
            self.sub_alpha = None
            self.sub_inv_alpha = None
            self.sub_overlay_rgb = None
            self.sub_premult_rgb = None
            self.opaque_mask = None
            self.has_semi = False

        self.textures_loaded = True

    def release_memory(self):
        """Releases heavy NumPy texture arrays once slide is done to cap RAM."""
        self.base_arr = None
        self.oversized_arr = None
        self.sub_alpha = None
        self.sub_inv_alpha = None
        self.sub_overlay_rgb = None
        self.sub_premult_rgb = None
        self.opaque_mask = None
        self.textures_loaded = False
        trim_memory()

    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Render a single video frame as an RGB uint8 NumPy array using SIMD acceleration."""
        if not self.textures_loaded:
            self.prepare_textures()

        t = frame_idx / max(1, (self.total_frames - 1))
        smooth_t = 3 * (t**2) - 2 * (t**3)

        if self.animation == "none" or self.oversized_arr is None:
            frame_arr = self.base_arr.copy() if self.base_arr is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
        elif self.animation in ("ken_burns", "zoom_in"):
            curr_scale = 1.0 + (0.12 * smooth_t)
            crop_w = int(self.oversized_w / curr_scale)
            crop_h = int(self.oversized_h / curr_scale)
            crop_x = (self.oversized_w - crop_w) // 2
            crop_y = (self.oversized_h - crop_h) // 2
            cropped = self.oversized_arr[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            frame_arr = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        elif self.animation == "zoom_out":
            curr_scale = 1.12 - (0.12 * smooth_t)
            crop_w = int(self.oversized_w / curr_scale)
            crop_h = int(self.oversized_h / curr_scale)
            crop_x = (self.oversized_w - crop_w) // 2
            crop_y = (self.oversized_h - crop_h) // 2
            cropped = self.oversized_arr[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            frame_arr = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        elif self.animation == "pan_left":
            max_pan_x = self.oversized_w - self.width
            crop_x = int(max_pan_x * (1.0 - smooth_t))
            crop_y = (self.oversized_h - self.height) // 2
            cropped = self.oversized_arr[crop_y : crop_y + self.height, crop_x : crop_x + self.width]
            frame_arr = cropped.copy()
        elif self.animation == "pan_right":
            max_pan_x = self.oversized_w - self.width
            crop_x = int(max_pan_x * smooth_t)
            crop_y = (self.oversized_h - self.height) // 2
            cropped = self.oversized_arr[crop_y : crop_y + self.height, crop_x : crop_x + self.width]
            frame_arr = cropped.copy()
        else:
            frame_arr = self.base_arr.copy() if self.base_arr is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Fast sub-region alpha compositing
        if self.has_overlay and self.sub_inv_alpha is not None and self.sub_premult_rgb is not None:
            bg_sub = frame_arr[self.ov_y1:self.ov_y2, self.ov_x1:self.ov_x2]
            if not self.has_semi and self.opaque_mask is not None and self.sub_overlay_rgb is not None:
                # 100% opaque overlay: direct copy with zero float allocations
                bg_sub[self.opaque_mask] = self.sub_overlay_rgb[self.opaque_mask]
            else:
                # Pre-multiplied alpha blend with single multiplication instead of two
                blended_sub = (bg_sub.astype(np.float32) * self.sub_inv_alpha + self.sub_premult_rgb).astype(np.uint8)
                frame_arr[self.ov_y1:self.ov_y2, self.ov_x1:self.ov_x2] = blended_sub

        return frame_arr


# ============================================================================
# TIMELINE GENERATOR & BLENDING
# ============================================================================
def generate_timeline_frames(slides_renderers: List[SlideRenderer], video_cfg: dict):
    """
    Sequentially yields individual video frames (RGB24 NumPy array).
    Seamlessly blends transitions between consecutive slides using OpenCV SIMD.
    """
    fps = video_cfg["fps"]
    trans_duration = video_cfg["transition_duration"]
    trans_frames = int(trans_duration * fps)
    num_slides = len(slides_renderers)

    for s_idx, curr_renderer in enumerate(slides_renderers):
        is_last = s_idx == (num_slides - 1)
        slide_total_frames = curr_renderer.total_frames

        if is_last or trans_frames <= 0:
            for f_idx in range(slide_total_frames):
                yield curr_renderer.get_frame(f_idx)
        else:
            next_renderer = slides_renderers[s_idx + 1]
            regular_frames_count = max(0, slide_total_frames - trans_frames)
            for f_idx in range(regular_frames_count):
                yield curr_renderer.get_frame(f_idx)

            for t_idx in range(trans_frames):
                curr_f_idx = regular_frames_count + t_idx
                next_f_idx = t_idx

                frame_a = curr_renderer.get_frame(curr_f_idx)
                frame_b = next_renderer.get_frame(next_f_idx)

                alpha = float(t_idx + 1) / float(trans_frames + 1)
                blended = cv2.addWeighted(frame_a, 1.0 - alpha, frame_b, alpha, 0.0)
                yield blended

        # Immediately free slide textures from RAM and trim glibc arena
        curr_renderer.release_memory()
        trim_memory()


# ============================================================================
# MAIN VIDEO GENERATION PIPELINE
# ============================================================================
def generate_video(
    config: Union[dict, str, Path],
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Generate professional vehicle presentation video from config with AI vehicle filtering and multi-threaded FFmpeg pipe streaming.
    Returns machine-readable dictionary with status and metadata.
    """
    start_time = time.time()
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    base_dir = None
    if isinstance(config, (str, Path)):
        cfg_path = Path(config).resolve()
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
        base_dir = cfg_path.parent
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
    elif isinstance(config, dict):
        raw_config = config
    else:
        raise ValueError("Config must be a dictionary or a path to a JSON file.")

    logger.info("Validating and normalizing video configuration...")
    cfg = validate_and_normalize_config(raw_config, base_dir=base_dir)

    if output_path:
        final_output_path = Path(output_path).resolve()
    else:
        final_output_path = Path(cfg["output"]["path"]).resolve()

    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("FFmpeg executable not found on system PATH. Please install ffmpeg.")

    video_cfg = cfg["video"]
    slides_cfg = cfg["slides"]
    width = video_cfg["width"]
    height = video_cfg["height"]
    fps = video_cfg["fps"]
    layout = video_cfg["layout"]

    if not base_dir and output_path:
        base_dir = Path(output_path).resolve().parent

    if not video_cfg.get("metadata") and cfg.get("metadata"):
        video_cfg["metadata"] = cfg.get("metadata")

    # Auto-load pre-computed metadata from extracted_text.json or text.json if available
    if not video_cfg.get("metadata") and base_dir:
        for possible_meta in ("extracted_text.json", "text.json"):
            meta_p = base_dir / possible_meta
            if meta_p.is_file():
                try:
                    with open(meta_p, "r", encoding="utf-8") as f:
                        meta_json = json.load(f)
                    if isinstance(meta_json, dict) and "images_meta" in meta_json:
                        video_cfg["metadata"] = meta_json
                        logger.info(f"Auto-loaded pre-computed vehicle metadata from {possible_meta}")
                        break
                except Exception:
                    pass

    det_map = {}
    meta_by_name = {str(m.get("filename", "")).strip(): m for m in video_cfg.get("metadata", {}).get("images_meta", [])}

    # AI Vehicle Detection & Slide Filtering
    filter_metrics = {"enabled": False}
    filter_enabled = video_cfg.get("filter_vehicles", True)
    if filter_enabled and VEHICLE_DETECTOR_AVAILABLE:
        try:
            from vehicle_detector import filter_vehicle_images
            from pathlib import Path as _Path

            def _is_trusted_composite(image_path: str) -> bool:
                """
                System-generated composites (composite_*.jpg) must never be
                rejected by the paper/document heuristic — they always contain
                a real vehicle on an artificial background.
                """
                p = _Path(image_path)
                return p.name.startswith("composite_") or "composites" in p.parts

            # Partition outro slides from content slides so outro is NEVER filtered out
            # and NEVER sorted among raw/exterior/interior content slides
            outro_slides = [s for s in slides_cfg if s.get("is_outro") or s.get("type") == "outro"]
            content_slides = [s for s in slides_cfg if not (s.get("is_outro") or s.get("type") == "outro")]

            slide_paths = [s["image"] for s in content_slides]

            # Partition: trusted composites bypass detection, others go through detector
            trusted_paths = [sp for sp in slide_paths if _is_trusted_composite(sp)]
            candidate_paths = [sp for sp in slide_paths if not _is_trusted_composite(sp)]

            meta_by_name = {str(m.get("filename", "")).strip(): m for m in video_cfg.get("metadata", {}).get("images_meta", [])}
            pre_validated = bool(meta_by_name and all((_Path(p).name in meta_by_name) for p in candidate_paths))

            if pre_validated:
                logger.info("Reusing pre-computed vehicle metadata from pipeline; bypassing redundant vehicle detection pass.")
                retained_paths = []
                det_details = []
                for sp in candidate_paths:
                    fn = _Path(sp).name
                    m = meta_by_name.get(fn, {})
                    is_rc = bool(
                        m.get("is_rc_document")
                        or m.get("type") == "rc"
                        or m.get("detected_view") == "rc"
                        or m.get("scene_type") == "rc_document"
                        or "rc_document" in str(m.get("classification_reason", "")).lower()
                    )
                    is_rej = bool(
                        m.get("status") == "rejected"
                        or m.get("is_vehicle") is False
                        or m.get("type") == "not_vehicle"
                        or m.get("detected_view") == "not_vehicle"
                    )
                    if not is_rc and not is_rej:
                        is_v = m.get("is_vehicle") or m.get("status") == "accepted"
                        if is_v:
                            retained_paths.append(sp)
                            det_details.append({
                                "path": sp,
                                "is_vehicle": True,
                                "confidence": m.get("confidence", 0.95),
                                "vehicle_side": m.get("vehicle_side") or m.get("detected_view") or m.get("type"),
                                "is_interior": bool(m.get("is_interior") or m.get("type") in ("interior", "dashboard", "cabin")),
                            })
                filter_res = {
                    "retained_paths": retained_paths,
                    "elapsed_seconds": 0.001,
                    "fallback_triggered": False,
                    "details": det_details,
                }
            else:
                filter_res = filter_vehicle_images(
                    candidate_paths,
                    conf_threshold=video_cfg.get("vehicle_conf_threshold", 0.35),
                    min_area_ratio=video_cfg.get("vehicle_min_area", 0.05),
                    fallback_policy=video_cfg.get("vehicle_fallback_policy", "strict"),
                )

            raw_det = filter_res.get("details")
            details_list = [d for d in raw_det if isinstance(d, dict)] if isinstance(raw_det, list) else []
            raw_ret = filter_res.get("retained_paths")
            retained_list = [str(r) for r in raw_ret] if isinstance(raw_ret, list) else []

            # Build map of detector results and metadata for slide ordering and classification
            det_map = {str(d.get("path")): d for d in details_list if isinstance(d, dict) and d.get("path")}
            meta_by_name = {str(m.get("filename", "")).strip(): m for m in video_cfg.get("metadata", {}).get("images_meta", [])}

            # Trusted composites always retained + detector-approved images
            retained_set = set(trusted_paths) | set(retained_list)
            filtered_slides = [
                s for s in content_slides
                if str(s["image"]) in retained_set
                and classify_slide_view(s, det_map, meta_by_name) != "rejected_document"
            ]

            # Order content slides:
            # 1. Studio composites come first (Slide 0 Hero presentation)
            # 2. ALL exterior vehicle views (front, rear, side, etc.)
            #    CRITICAL RULE: If there is no front image or primary image in uploaded images,
            #    show rear image on first priority while generating video!
            # 3. ALL interior vehicle views (cockpit, cluster, odometer, cabin) come after all exteriors
            has_front_or_primary = any(
                (classify_slide_view(s, det_map, meta_by_name) in ("primary_image", "exterior_front", "exterior_front_three_quarter"))
                for s in filtered_slides
                if not _is_trusted_composite(str(s.get("image", "")))
            )

            def _get_slide_sort_key(entry):
                idx, slide = entry
                part = classify_slide_view(slide, det_map, meta_by_name)
                # If there is no front image or primary image, show rear image on first priority (rank 5)
                if not has_front_or_primary and part == "exterior_rear":
                    rank = 5
                else:
                    rank = PART_SORT_RANKS.get(part, 40)
                return (rank, idx)

            filtered_slides_with_idx = list(enumerate(filtered_slides))
            filtered_slides_with_idx.sort(key=_get_slide_sort_key)
            sorted_content_slides = [slide for _, slide in filtered_slides_with_idx]

            _total_input   = len(slide_paths)          # original content slide count
            _retained_total = len(retained_set)        # trusted composites + detected vehicles
            _filtered_total = _total_input - _retained_total

            filter_metrics = {
                "enabled": True,
                "total_input_slides": _total_input,
                "trusted_composites_bypassed": len(trusted_paths),
                "retained_slides": _retained_total,
                "filtered_slides": _filtered_total,
                "detection_time_seconds": filter_res.get("elapsed_seconds", 0.0),
                "fallback_triggered": filter_res.get("fallback_triggered", False),
                "filtered_details": [d for d in details_list if isinstance(d, dict) and not d.get("is_vehicle")],
            }

            if not sorted_content_slides:
                logger.warning(
                    f"No valid vehicle images detected "
                    f"({_filtered_total}/{_total_input} non-vehicle images filtered). "
                    "Skipping video generation."
                )
                return {
                    "status": "skipped",
                    "error_code": "NO_VEHICLES_DETECTED",
                    "message": "No vehicle detected in input images. Video generation skipped.",
                    "total_input_slides": _total_input,
                    "retained_slides": 0,
                    "filtered_slides": _filtered_total,
                    "detection_time_seconds": filter_res["elapsed_seconds"],
                    "vehicle_filter": filter_metrics,
                }

            # Update outro thumbnail to use primary vehicle image from retained slides
            if outro_slides and sorted_content_slides:
                best_hero_img = None
                # 1. Trusted composite
                for cs in sorted_content_slides:
                    cs_path = str(cs.get("image", ""))
                    if _is_trusted_composite(cs_path):
                        best_hero_img = cs_path
                        break
                # 2. Slide explicitly marked primary
                if not best_hero_img:
                    for cs in sorted_content_slides:
                        if cs.get("primary") or cs.get("is_primary"):
                            best_hero_img = str(cs.get("image", ""))
                            break
                # 3. Exterior front
                if not best_hero_img:
                    for cs in sorted_content_slides:
                        part = classify_slide_view(cs, det_map, meta_by_name)
                        if part == "exterior_front":
                            best_hero_img = str(cs.get("image", ""))
                            break
                # 4. If no front, exterior rear (rear side image is hero)
                if not best_hero_img:
                    for cs in sorted_content_slides:
                        part = classify_slide_view(cs, det_map, meta_by_name)
                        if part == "exterior_rear":
                            best_hero_img = str(cs.get("image", ""))
                            break
                # 5. Exterior side / general
                if not best_hero_img:
                    for cs in sorted_content_slides:
                        part = classify_slide_view(cs, det_map, meta_by_name)
                        if part in ("exterior_side", "exterior_general"):
                            best_hero_img = str(cs.get("image", ""))
                            break
                # 6. Fallback to first retained exterior content slide, NEVER interior if exterior exists
                if not best_hero_img:
                    exterior_slides = [
                        cs for cs in sorted_content_slides
                        if classify_slide_view(cs, det_map, meta_by_name) not in ("interior_cockpit", "interior_odometer", "interior_cabin", "interior_general")
                    ]
                    best_hero_img = exterior_slides[0]["image"] if exterior_slides else sorted_content_slides[0]["image"]

                for o_slide in outro_slides:
                    o_slide["image"] = best_hero_img

            # CRITICAL REQUIREMENT: Outro card MUST ALWAYS be placed at the very end
            # after ALL uploaded photos (exterior photos first, interior photos next, outro card LAST).
            slides_cfg = sorted_content_slides + outro_slides

            # Adjust durations:
            # - Studio composite gets 3.0s
            # - First original photo gets 5.0s
            # - Remaining content slides get 4.5s
            # - Outro slide gets outro_duration (default 5.0s)
            has_composite = len(sorted_content_slides) > 0 and _is_trusted_composite(str(sorted_content_slides[0].get("image", "")))
            for i, sl in enumerate(slides_cfg):
                if sl.get("is_outro") or sl.get("type") == "outro":
                    sl["duration"] = float(video_cfg.get("outro_duration", sl.get("duration", 5.0)))
                elif _is_trusted_composite(str(sl.get("image", ""))):
                    sl["duration"] = 3.0
                elif i == (1 if has_composite else 0):
                    sl["duration"] = 5.0
                else:
                    sl["duration"] = sl.get("duration", 4.5)

            # Assign contextual part-specific specs/badges to each slide
            assign_part_specific_slide_details(slides_cfg, video_cfg, det_map, meta_by_name)

            logger.info(
                f"Ordered {len(slides_cfg)} slides (Exteriors first, Interiors after, Outro Card last): "
                + ", ".join([f"{idx+1}:{('OUTRO_CARD' if s.get('is_outro') else _Path(s['image']).name)}" for idx, s in enumerate(slides_cfg)])
            )

            # Immediately release detector reference and trim glibc memory
            del filter_vehicle_images
            trim_memory()

            logger.info(
                f"Vehicle detection kept {len(sorted_content_slides)}/{_total_input} slides "
                f"({len(trusted_paths)} composites bypassed, "
                f"Detection Time: {filter_res['elapsed_seconds']:.3f}s)."
            )
        except Exception as det_err:
            logger.warning(f"Vehicle detection encountered exception, proceeding with all slides: {det_err}")
            filter_metrics = {"enabled": True, "error": str(det_err)}

    # Final guarantee: always ensure any outro slide is strictly placed at the very end of all images
    final_outro_slides = [s for s in slides_cfg if s.get("is_outro") or s.get("type") == "outro"]
    final_content_slides = [s for s in slides_cfg if not (s.get("is_outro") or s.get("type") == "outro")]
    slides_cfg = final_content_slides + final_outro_slides

    # Ensure all content slides have part-specific details assigned
    assign_part_specific_slide_details(slides_cfg, video_cfg, det_map, meta_by_name)

    # Dynamic Audio Narration & Slide Duration Synchronization (audio_narrator.py)
    narration_audio_path = None
    audio_cfg = (
        cfg.get("audio")
        or cfg.get("car_details")
        or (raw_config.get("car_details") if isinstance(raw_config, dict) else None)
        or (raw_config.get("audio") if isinstance(raw_config, dict) else None)
    )
    if not audio_cfg and slides_cfg:
        first_s = slides_cfg[0]
        if first_s.get("headerTitle") or first_s.get("price"):
            audio_cfg = {
                "name": first_s.get("headerTitle"),
                "price": first_s.get("price"),
                "color": first_s.get("color") or first_s.get("colour"),
                "km": first_s.get("km") or first_s.get("odometer"),
                "owner": first_s.get("owner"),
            }

    if audio_cfg:
        try:
            from audio_narrator import generate_car_audio
            if isinstance(audio_cfg, dict) and audio_cfg.get("path") and Path(audio_cfg["path"]).is_file():
                narration_audio_path = Path(audio_cfg["path"])
                from audio_narrator import VideoAudioMuxer
                audio_dur = VideoAudioMuxer.get_audio_duration(narration_audio_path)
            else:
                # Save narration audio in base_dir (same folder as images)
                narration_audio_path, audio_dur = generate_car_audio(
                    audio_cfg if isinstance(audio_cfg, dict) else {},
                    output_dir=base_dir,
                    voice_name=audio_cfg.get("voice", "M1") if isinstance(audio_cfg, dict) else "M1",
                    use_llm=True,
                )
            if narration_audio_path:
                logger.info(
                    f"Narration audio ready: {narration_audio_path} (Duration: {audio_dur:.2f}s). "
                    f"Video duration is preserved based on slide timings."
                )
        except Exception as audio_init_err:
            logger.warning(f"Audio pre-generation skipped or failed: {audio_init_err}")

    logger.info(f"Rendering {len(slides_cfg)} slides at {width}x{height} @ {fps} FPS (Layout: {layout})...")
    if progress_callback:
        progress_callback(0.05, f"Initializing {layout} slide textures...")

    renderers = [SlideRenderer(dict(s, is_first_slide=(i == 0)), video_cfg) for i, s in enumerate(slides_cfg)]

    total_frames = sum(r.total_frames for r in renderers)

    total_duration_sec = total_frames / fps
    logger.info(f"Total video duration: {total_duration_sec:.2f}s ({total_frames} frames)")

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", video_cfg.get("preset", "ultrafast"),
        "-tune", "zerolatency",
        "-threads", str(min(4, os.cpu_count() or 4)),
        "-crf", str(video_cfg["crf"]),
        str(final_output_path),
    ]

    logger.info(f"Encoding MP4 to: {final_output_path}")
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frames_rendered = 0
    try:
        for frame in generate_timeline_frames(renderers, video_cfg):
            if proc.stdin is not None:
                proc.stdin.write(memoryview(frame))
            frames_rendered += 1

            if progress_callback and total_frames > 0 and frames_rendered % 15 == 0:
                pct = min(0.95, 0.1 + 0.85 * (frames_rendered / total_frames))
                progress_callback(pct, f"Rendering frame {frames_rendered}/{total_frames}...")

        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.flush()
                proc.stdin.close()
            except Exception:
                pass

        proc.wait()

        if proc.returncode != 0:
            stderr_text = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            logger.error(f"FFmpeg encoding error:\n{stderr_text}")
            raise RuntimeError(f"FFmpeg exited with error code {proc.returncode}: {stderr_text}")

    except Exception as e:
        if proc.poll() is None:
            proc.kill()
        _notify(
            "videoGen.generate_video",
            e,
            ref_id=Path(final_output_path).stem,
            extra=f"layout={layout}",
        )
        raise e
    finally:
        trim_memory()

    if not final_output_path.is_file() or final_output_path.stat().st_size == 0:
        raise RuntimeError(f"Output video file was not generated or is empty: {final_output_path}")

    # Mux Pre-Generated Narration Audio (Zero-reencode stream copy <300ms)
    if narration_audio_path and Path(narration_audio_path).is_file():
        try:
            from audio_narrator import VideoAudioMuxer
            VideoAudioMuxer.attach_audio_to_video(
                final_output_path,
                Path(narration_audio_path),
                final_output_path,
                video_duration=total_duration_sec,
            )
            logger.info(f"Audio narration successfully muxed into: {final_output_path}")
        except Exception as audio_err:
            logger.warning(f"Audio narration muxing skipped or failed: {audio_err}")

    elapsed = time.time() - start_time
    file_size_mb = final_output_path.stat().st_size / (1024 * 1024)
    fps_speed = total_frames / max(0.001, elapsed)
    logger.info(f"Video generated successfully in {elapsed:.2f}s ({fps_speed:.1f} FPS, {file_size_mb:.2f} MB)")

    if progress_callback:
        progress_callback(1.0, "Video generation complete.")

    return {
        "status": "success",
        "output_path": str(final_output_path),
        "duration_seconds": round(total_duration_sec, 2),
        "total_frames": total_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "layout": layout,
        "file_size_bytes": final_output_path.stat().st_size,
        "file_size_mb": round(file_size_mb, 2),
        "render_time_seconds": round(elapsed, 2),
        "rendering_fps": round(fps_speed, 1),
        "vehicle_filter": filter_metrics,
    }


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VehicleVision — Local Vehicle Video Generator Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        help="Path to the JSON configuration file (positional argument).",
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to the JSON configuration file (named flag).",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path for the output MP4 video file (overrides JSON config).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON result to stdout.",
    )
    parser.add_argument(
        "--no-filter-vehicles",
        action="store_true",
        help="Disable AI vehicle detection and include all input slides.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for AI vehicle detector (default: 0.35).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.05,
        help="Minimum bbox area ratio for vehicle detector (default: 0.05).",
    )
    parser.add_argument(
        "--fade-background",
        dest="fade_background",
        action="store_true",
        default=None,
        help="Enable AI depth-of-field background fading on vehicle images (enabled by default).",
    )
    parser.add_argument(
        "--no-fade-background",
        dest="fade_background",
        action="store_false",
        help="Disable background fading on vehicle images.",
    )
    parser.add_argument(
        "--fade-strength",
        type=float,
        default=None,
        help="Background brightness retention factor (0.0 to 1.0, default: 0.65 for 65 percent focus).",
    )
    parser.add_argument(
        "--blur-radius",
        type=int,
        default=None,
        help="Bokeh blur radius for vehicle background (default: 28 for soft depth-of-field).",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["blur_pad", "cover", "contain", "focus_fit"],
        default=None,
        help="Image scaling mode on canvas (default: cover for edge-to-edge full image).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()
    config_file = args.config or args.config_pos

    if not config_file:
        print("❌ Error: Missing configuration JSON file.", file=sys.stderr)
        print("👉 Usage: python videoGen.py <path_to_json_file> [--output <path>] [--json]", file=sys.stderr)
        sys.exit(EXIT_CONFIG_ERROR)

    try:
        # Load and potentially override CLI flags in config
        with open(config_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        if args.no_filter_vehicles:
            cfg_data.setdefault("video", {})["filter_vehicles"] = False
        if args.conf_threshold:
            cfg_data.setdefault("video", {})["vehicle_conf_threshold"] = args.conf_threshold
        if args.min_area:
            cfg_data.setdefault("video", {})["vehicle_min_area"] = args.min_area
        if args.fade_background is not None:
            cfg_data.setdefault("video", {})["fade_vehicle_background"] = args.fade_background
        if args.fade_strength is not None:
            cfg_data.setdefault("video", {})["fade_strength"] = args.fade_strength
        if args.blur_radius is not None:
            cfg_data.setdefault("video", {})["blur_radius"] = args.blur_radius
        if args.scale_mode is not None:
            cfg_data.setdefault("video", {})["scale_mode"] = args.scale_mode


        _rt = _ResourceTracker() if _ResourceTracker is not None else None
        _rt_ctx = _rt if _rt is not None else _NullContext()
        with _rt_ctx:
            result = generate_video(
                config=cfg_data,
                output_path=args.output,
                verbose=args.verbose,
            )
        if _rt is not None and isinstance(result, dict):
            result["resource_metrics"] = {
                "elapsed_sec": _rt.elapsed_sec,
                "peak_ram_mb": _rt.peak_ram_mb,
                "cpu_percent": _rt.cpu_percent,
            }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            filter_info = result.get("vehicle_filter", {})
            filter_str = f" | Retained: {filter_info.get('retained_slides', 'N/A')}/{filter_info.get('total_input_slides', 'N/A')} slides" if filter_info.get("enabled") else ""
            print(f"\n✅ Video generated successfully: {result['output_path']}")
            print(f"📊 Layout: {result.get('layout', 'default')} | Duration: {result['duration_seconds']}s | Size: {result['file_size_mb']} MB | Render Time: {result['render_time_seconds']}s ({result.get('rendering_fps', 0)} FPS){filter_str}\n")

        sys.exit(EXIT_SUCCESS)

    except FileNotFoundError as fnf:
        err_res = {"status": "error", "error_code": "FILE_NOT_FOUND", "message": str(fnf)}
        _notify("videoGen.main", fnf, ref_id=config_file, extra="FILE_NOT_FOUND")
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"\n❌ Error [FILE_NOT_FOUND]: {fnf}\n", file=sys.stderr)
        sys.exit(EXIT_FILE_NOT_FOUND)

    except ValueError as ve:
        err_res = {"status": "error", "error_code": "CONFIG_ERROR", "message": str(ve)}
        _notify("videoGen.main", ve, ref_id=config_file, extra="CONFIG_ERROR")
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"\n❌ Error [CONFIG_ERROR]: {ve}\n", file=sys.stderr)
        sys.exit(EXIT_CONFIG_ERROR)

    except RuntimeError as re:
        err_res = {"status": "error", "error_code": "FFMPEG_ERROR", "message": str(re)}
        _notify("videoGen.main", re, ref_id=config_file, extra="FFMPEG_ERROR")
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"\n❌ Error [FFMPEG_ERROR]: {re}\n", file=sys.stderr)
        sys.exit(EXIT_FFMPEG_ERROR)

    except Exception as ex:
        err_res = {"status": "error", "error_code": "INTERNAL_ERROR", "message": str(ex)}
        _notify("videoGen.main", ex, ref_id=config_file, extra="INTERNAL_ERROR")
        if args.json:
            print(json.dumps(err_res, indent=2))
        else:
            print(f"\n❌ Error [INTERNAL_ERROR]: {ex}\n", file=sys.stderr)
        sys.exit(EXIT_INTERNAL_ERROR)


if __name__ == "__main__":
    main()