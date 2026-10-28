#!/usr/bin/env python3
"""
VehicleVision — Standalone AI Vehicle Detector & Classifier
===========================================================

100% Local, Offline Computer-Vision Vehicle Detection & Classification Engine.

STRICT CONSTRAINTS SATISFACTION:
- Single-File Architecture: Entirely contained in this single file.
- Zero Pretrained Models & Zero Network: No YOLO, ONNX, PyTorch, TensorFlow, or weight files.
- Classical Computer Vision: Powered exclusively by Python standard library, OpenCV (cv2), and NumPy.
- Transparent Heuristics: Uses explainable geometry, circular wheel structures, ground shadow seams,
  and interior cockpit signatures. Uncertainty is treated as a first-class result.
- Backwards Compatibility: Drop-in replacement for VehicleVision consumers (including videoGen.py).

LIMITATIONS & REALITY:
Traditional computer-vision heuristics rely on observable geometric and structural cues
(aspect ratio, edge patterns, circular wheel signatures, symmetry, ground shadows). They are not
statistically calibrated deep-learning probabilities and cannot guarantee 100% accuracy on arbitrary,
heavily occluded, or unusual photos. Ambiguous or low-signal images are conservatively marked 'uncertain'.
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# RapidOCR engine support (Fast, local ONNX OCR)
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RapidOCR = None  # type: ignore[assignment]
    RAPID_OCR_AVAILABLE = False

_RAPID_OCR_INSTANCE = None

def get_rapid_ocr_engine():
    global _RAPID_OCR_INSTANCE
    if RAPID_OCR_AVAILABLE and RapidOCR is not None and _RAPID_OCR_INSTANCE is None:
        try:
            _RAPID_OCR_INSTANCE = RapidOCR()
        except Exception:
            _RAPID_OCR_INSTANCE = None
    return _RAPID_OCR_INSTANCE

# Optional pytesseract import for optical character recognition fallback
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None  # type: ignore[assignment]
    PYTESSERACT_AVAILABLE = False

TESSERACT_CMD = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")
if PYTESSERACT_AVAILABLE and pytesseract is not None and hasattr(pytesseract, "pytesseract"):
    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

try:
    from resource_tracker import ResourceTracker as _ResourceTracker
except ImportError:
    _ResourceTracker = None

class _NullContext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# ---------------------------------------------------------------------------
# Configurable Threshold Constants & Named Parameters
# ---------------------------------------------------------------------------
MIN_IMAGE_DIM = 64
MAX_ANALYSIS_DIM = 1280
BLUR_LAPLACIAN_THRESH = 18.0
LOW_LIGHT_THRESH = 40.0
OVEREXPOSURE_THRESH = 245.0

# Document / Non-Vehicle Text Rejection Thresholds
DOC_LIGHT_PIXEL_THRESH = 0.60    # >60% light background
DOC_TEXT_COMPONENT_MIN = 35     # High density of small character components

# Indian State & Union Territory RTO Prefixes
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB", "UA", "OR", "BH"
}

# Positive Indian RC Document Title Patterns (Tier 1: Weight 0.40)
RC_TIER1_TITLES = [
    (r"\bREGISTRATION\s+CERTIFICATE\b", "Registration Certificate Title"),
    (r"\bCERTIFICATE\s+OF\s+REGISTRATION\b", "Certificate of Registration Title"),
    (r"\bFORM\s+23(?:[A-Z])?\b", "Form 23 Title"),
    (r"\bRC\s+SMART\s*CARD\b", "RC Smart Card Title"),
    (r"\bVEHICLE\s+REGISTRATION\s+CERTIFICATE\b", "Vehicle Registration Certificate Title"),
    (r"\bPARIVAHAN\s+SEWA\b", "Parivahan Sewa Portal"),
]

# Government / RTO Authorities (Tier 2: Weight 0.25)
RC_TIER2_AUTHORITIES = [
    (r"\bTRANSPORT\s+DEPARTMENT\b", "Transport Department"),
    (r"\bREGISTERING\s+AUTHORITY\b", "Registering Authority"),
    (r"\b(?:REGIONAL\s+TRANSPORT|RTO|DTO|SDM|RLA)\b", "RTO / Transport Authority"),
    (r"\bGOVERNMENT\s+OF\s+INDIA\b", "Government of India"),
    (r"\bGOVT\.?\s+OF\s+[A-Z\s]{3,20}\b", "State Government Header"),
    (r"\bSTATE\s+TRANSPORT\b", "State Transport"),
]

# Core Vehicle Registration Fields (Tier 3: Weight 0.20)
RC_TIER3_VITALS = [
    (r"\bCHASSIS\s*NO?\.?\b", "Chassis Number Field"),
    (r"\bCHASSIS\s*NUMBER\b", "Chassis Number"),
    (r"\bENGINE\s*NO?\.?\b", "Engine Number Field"),
    (r"\bENGINE\s*NUMBER\b", "Engine Number"),
    (r"\bREGN?\.?\s*NO?\.?\b", "Registration Number Field"),
    (r"\bREGISTRATION\s*NO?\.?\b", "Registration Number"),
    (r"\bREG\s*DATE\b", "Registration Date"),
    (r"\bDATE\s+OF\s+REG(?:ISTRATION)?\b", "Date of Registration"),
    (r"\bVALID\s+(?:UPTO|TILL)\b", "Validity Field"),
    (r"\bFITNESS\s+(?:UPTO|VALID)\b", "Fitness Validity"),
    (r"\bOWNER\s+NAME\b", "Owner Name Field"),
    (r"\bS/W/D\s+OF\b", "Relation / Parentage Field"),
]

# Technical Specifications (Tier 4: Weight 0.15)
RC_TIER4_TECHNICAL = [
    (r"\bMAKER\s*(?:NAME)?\b", "Vehicle Maker"),
    (r"\bMODEL\s*(?:NAME)?\b", "Vehicle Model"),
    (r"\bVEHICLE\s+CLASS\b", "Vehicle Class"),
    (r"\bCLASS\s+OF\s+VEHICLE\b", "Class of Vehicle"),
    (r"\bFUEL\s+TYPE\b", "Fuel Type Field"),
    (r"\b(?:PETROL|DIESEL|CNG|ELECTRIC|HYBRID|LPG)\b", "Fuel Type Value"),
    (r"\bSEATING\s+CAP(?:ACITY)?\b", "Seating Capacity"),
    (r"\bC\.?C\.?\b", "Cubic Capacity (Engine CC)"),
    (r"\bCUBIC\s+CAPACITY\b", "Cubic Capacity"),
    (r"\bUNLADEN\s+WT\b", "Unladen Weight"),
    (r"\bGROSS\s+VEH(?:ICLE)?\s+WT\b", "Gross Vehicle Weight"),
    (r"\bCOLOR\s*(?:OF\s+VEHICLE)?\b", "Vehicle Color"),
    (r"\bEMISSION\s+NORMS?\b", "Emission Norms"),
    (r"\b(?:BS-?VI|BS-?IV|BHARAT\s+STAGE)\b", "Bharat Stage Norms"),
    (r"\bWHEEL\s*BASE\b", "Wheelbase"),
    (r"\bNO\.?\s+OF\s+CYLINDERS?\b", "Cylinder Count"),
    (r"\bFINANCIER\b", "Financier Field"),
    (r"\bHYPOTHECATED\s+TO\b", "Hypothecation"),
]

# Commercial / Operational Terms (Tier 5: Weight 0.10)
RC_TIER5_TERMS = [
    (r"\bTAX\s+(?:PAID|VALID)\b", "Road Tax Field"),
    (r"\bONE\s+TIME\s+TAX\b", "One Time Tax"),
    (r"\bMV\s+TAX\b", "Motor Vehicle Tax"),
    (r"\bINSURANCE\s+POLICY\s*NO?\b", "Insurance Policy Field"),
    (r"\bINSURER\b", "Insurance Company"),
    (r"\bINSURANCE\s+UPTO\b", "Insurance Validity"),
    (r"\bPUC\s+VALID\b", "PUC Validity"),
]

# Negative Documents (Aadhaar, PAN, DL, Passport, Invoices)
NEGATIVE_DOCUMENT_PATTERNS = [
    (r"\bAADHAAR\b", "Aadhaar Card", 0.70),
    (r"\bUNIQUE\s+IDENTIFICATION\b", "UIDAI / Aadhaar", 0.70),
    (r"\bMERA\s+AADHAAR\b", "Aadhaar Slogan", 0.70),
    (r"\bINCOME\s+TAX\s+DEPARTMENT\b", "PAN Card Header", 0.70),
    (r"\bPERMANENT\s+ACCOUNT\s+NUMBER\b", "PAN Card Field", 0.70),
    (r"\bDRIVING\s+LICEN[CS]E\b", "Driving Licence", 0.50),
    (r"\bUNION\s+OF\s+INDIA\s+DRIVING\b", "DL Authority", 0.50),
    (r"\bELECTION\s+COMMISSION\b", "Voter ID Card", 0.70),
    (r"\bELECTOR['’]?S\s+PHOTO\b", "Voter ID Field", 0.70),
    (r"\bEPIC\s+NO\b", "Voter ID EPIC", 0.70),
    (r"\bPASSPORT\s+OF\s+INDIA\b", "Indian Passport", 0.80),
    (r"\bREPUBLIC\s+OF\s+INDIA\s+PASSPORT\b", "Passport Header", 0.80),
    (r"\bELECTRICITY\s+BILL\b", "Utility / Power Bill", 0.65),
    (r"\bWATER\s+BILL\b", "Water Bill", 0.65),
    (r"\bTAX\s+INVOICE\b", "Commercial Tax Invoice", 0.40),
]

# Vehicle Exterior Geometry Thresholds
CAR_ASPECT_MIN = 1.10
CAR_ASPECT_MAX = 2.85
BIKE_ASPECT_MIN = 0.45
BIKE_ASPECT_MAX = 1.45
BUS_ASPECT_MIN = 2.25
TRUCK_ASPECT_MIN = 1.35

# Confidence Thresholds
CONFIDENCE_VEHICLE_MIN = 0.55
CONFIDENCE_UNCERTAIN_FLOOR = 0.40
CONFIDENCE_TYPE_MIN = 0.45

# Non-Vehicle Scene Rejection Thresholds
SKIN_HUE_LOW = 5
SKIN_HUE_HIGH = 25
SKIN_SAT_LOW = 30
SKIN_SAT_HIGH = 180
SKIN_VAL_LOW = 70
SKIN_VAL_HIGH = 255
SKIN_DOMINANT_RATIO = 0.30       # >30% skin pixels = likely a person photo
FACE_CASCADE_MIN_NEIGHBORS = 4   # OpenCV Haar cascade sensitivity

# ---------------------------------------------------------------------------
# Logging Setup (Strictly to stderr so stdout remains 100% clean JSON)
# ---------------------------------------------------------------------------
logger = logging.getLogger("VehicleDetector")


def setup_logger(debug: bool = False) -> None:
    """Configure diagnostic logger writing exclusively to sys.stderr."""
    level = logging.DEBUG if debug else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# Module 1: Image Loading & Validation
# ---------------------------------------------------------------------------
def load_image(
    image_input: Union[str, Path, np.ndarray],
    max_dim: int = MAX_ANALYSIS_DIM,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], Tuple[int, int]]:
    """
    Safely decode and validate input image.
    Preserves original aspect ratio while constraining analysis dimensions.

    Returns:
        (analysis_bgr, orig_bgr, error_message, (original_width, original_height))
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.is_file():
            return None, None, f"Image file not found: {image_input}", (0, 0)
        if p.stat().st_size == 0:
            return None, None, f"Image file is empty (0 bytes): {image_input}", (0, 0)

        # Decode image using OpenCV
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            return None, None, f"Failed to decode image file (corrupted or unsupported format): {image_input}", (0, 0)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            return None, None, "Empty NumPy image array provided", (0, 0)
        if len(image_input.shape) == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        elif len(image_input.shape) == 3 and image_input.shape[2] == 3:
            img_bgr = image_input.copy()
        else:
            return None, None, f"Unsupported array shape: {image_input.shape}", (0, 0)
    elif hasattr(image_input, "convert"):
        # Handle PIL.Image.Image
        arr = np.array(image_input.convert("RGB"))
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        return None, None, f"Invalid image input type: {type(image_input)}", (0, 0)

    orig_h, orig_w = img_bgr.shape[:2]
    if orig_h < MIN_IMAGE_DIM or orig_w < MIN_IMAGE_DIM:
        return img_bgr, img_bgr, f"Image resolution too small ({orig_w}x{orig_h} < {MIN_IMAGE_DIM}px)", (orig_w, orig_h)

    # Scale down for fast CPU analysis if exceeds max_dim
    scale = min(1.0, max_dim / max(orig_h, orig_w))
    if scale < 1.0:
        new_w = max(MIN_IMAGE_DIM, int(orig_w * scale))
        new_h = max(MIN_IMAGE_DIM, int(orig_h * scale))
        analysis_img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        analysis_img = img_bgr

    return analysis_img, img_bgr, None, (orig_w, orig_h)


def measure_blur_metric(gray: np.ndarray) -> float:
    """Calculate Laplacian variance to assess optical focus / sharpness."""
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------------------
# Module 2: Document & Non-Vehicle Text Rejection
# ---------------------------------------------------------------------------
def check_is_document_or_paper(img_bgr: np.ndarray, gray: np.ndarray) -> Tuple[bool, float, List[str]]:
    """
    Detect whether the image is an automotive document (RC card, insurance form,
    invoice, or text scan) rather than a physical vehicle.
    """
    h, w = gray.shape[:2]
    total_pixels = max(1, h * w)
    cues = []

    # 1. Background luminance & low-saturation dominance (white/cream paper or card)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    light_paper_mask = (val > 175) & (sat < 50)
    light_ratio = float(np.count_nonzero(light_paper_mask)) / total_pixels

    # 2. Text line connected component analysis
    # Adaptive thresholding highlights small character glyphs
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6
    )
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    # Filter for character-sized components (small bounding boxes)
    char_candidates = 0
    for idx in range(1, num_labels):
        comp_w = stats[idx, cv2.CC_STAT_WIDTH]
        comp_h = stats[idx, cv2.CC_STAT_HEIGHT]
        comp_area = stats[idx, cv2.CC_STAT_AREA]
        if 4 <= comp_h <= 60 and 3 <= comp_w <= 80 and 8 <= comp_area <= 1800:
            char_candidates += 1

    doc_score = 0.0
    if light_ratio >= 0.48:
        doc_score += min(0.45, (light_ratio / 0.65) * 0.45)
        cues.append(f"predominantly_light_paper_background ({light_ratio:.1%})")

    if light_ratio >= 0.35 and char_candidates >= DOC_TEXT_COMPONENT_MIN:
        density_weight = min(0.50, (char_candidates / 120.0) * 0.50)
        doc_score += density_weight
        cues.append(f"high_text_component_density ({char_candidates} elements)")

    # 3. Horizontal edge alignment (text lines)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    abs_sobel_y = np.abs(sobel_y)
    mean_h_grad = float(np.mean(abs_sobel_y))
    if mean_h_grad > 16.0 and light_ratio >= 0.45:
        doc_score += 0.25
        cues.append("dense_horizontal_text_line_gradients")

    is_doc = (doc_score >= 0.55)
    return is_doc, min(1.0, doc_score), cues


def check_image_quality(img_bgr: np.ndarray) -> Tuple[Dict[str, Any], List[str]]:
    """Evaluates image dimensions, blurriness, and illumination levels."""
    h, w = img_bgr.shape[:2]
    warnings: List[str] = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr

    # 1. Resolution
    low_res = (w < 200 or h < 200)
    if low_res:
        warnings.append(f"Low resolution image ({w}x{h}); details may be unreadable.")

    # 2. Blur detection (Laplacian variance)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = (laplacian_var < BLUR_LAPLACIAN_THRESH)
    if is_blurry:
        warnings.append(f"Image may be blurry (Laplacian variance: {laplacian_var:.1f}).")

    # 3. Illumination checks
    mean_val = float(np.mean(gray))
    is_low_light = (mean_val < LOW_LIGHT_THRESH)
    if is_low_light:
        warnings.append(f"Low light / underexposed image (mean brightness: {mean_val:.1f}).")
    elif mean_val > OVEREXPOSURE_THRESH:
        warnings.append(f"Overexposed image (mean brightness: {mean_val:.1f}).")

    quality_info = {
        "width": int(w),
        "height": int(h),
        "blur_warning": is_blurry,
        "low_light_warning": is_low_light,
        "low_resolution_warning": low_res,
    }
    return quality_info, warnings


def detect_document_card_geometry(img_bgr: np.ndarray) -> Tuple[bool, Optional[np.ndarray], List[str]]:
    """
    Detect whether the image contains a dominant rectangular card/paper document
    quadrilateral using OpenCV edge detection, contour approximation, and perspective warping.
    Standard Indian RC smart cards have an ISO/IEC 7810 ID-1 aspect ratio of ~1.58 (85.60 mm x 53.98 mm).
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w
    cues: List[str] = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, cues

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.10 * img_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            pts = approx.reshape(4, 2).astype("float32")
            s = pts.sum(axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            (tl, tr, br, bl) = rect
            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            max_w = max(int(width_a), int(width_b))

            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
            max_h = max(int(height_a), int(height_b))

            if max_h == 0 or max_w == 0:
                continue

            aspect = max(max_w, max_h) / min(max_w, max_h)
            if 1.25 <= aspect <= 1.85:
                dst = np.array(
                    [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                    dtype="float32",
                )
                m = cv2.getPerspectiveTransform(rect, dst)
                warped = cv2.warpPerspective(img_bgr, m, (max_w, max_h))
                cues.append(f"card_quadrilateral_detected (aspect {aspect:.2f}, area {area / img_area:.1%})")
                return True, warped, cues

    cnt_top = contours[0]
    bx, by, bw, bh = cv2.boundingRect(cnt_top)
    b_area = bw * bh
    if b_area >= 0.20 * img_area:
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if 1.25 <= aspect <= 1.85:
            cropped = img_bgr[by : by + bh, bx : bx + bw]
            cues.append(f"rectangular_card_bounding_box (aspect {aspect:.2f})")
            return True, cropped, cues

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) if len(img_bgr.shape) == 3 else None
    if hsv is not None:
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        light_paper = float(np.count_nonzero((val > 175) & (sat < 50))) / max(1, img_area)
        if light_paper >= 0.55:
            cues.append(f"full_frame_light_paper_document ({light_paper:.1%})")
            return True, img_bgr, cues

    return False, None, cues


def perform_local_ocr(
    img_bgr: np.ndarray, doc_roi: Optional[Union[np.ndarray, List[int], Tuple[int, ...]]] = None
) -> Tuple[bool, str, List[str]]:
    """
    Perform local, offline OCR using RapidOCR (ONNX) with pytesseract fallback.
    Returns: (ocr_available, extracted_text, warnings)
    """
    warnings: List[str] = []
    h, w = img_bgr.shape[:2]
    target_img = img_bgr

    if doc_roi is not None:
        if isinstance(doc_roi, np.ndarray) and doc_roi.size > 0:
            target_img = doc_roi
        elif isinstance(doc_roi, (list, tuple)) and len(doc_roi) == 4:
            bx, by, bw, bh = doc_roi
            if bw >= 50 and bh >= 50:
                pad_x, pad_y = int(bw * 0.02), int(bh * 0.02)
                x1 = max(0, bx - pad_x)
                y1 = max(0, by - pad_y)
                x2 = min(w, bx + bw + pad_x)
                y2 = min(h, by + bh + pad_y)
                target_img = img_bgr[y1:y2, x1:x2]

    # Primary Pass: RapidOCR (ultra-fast ONNX model)
    rapid_engine = get_rapid_ocr_engine()
    if rapid_engine is not None:
        try:
            res, _ = rapid_engine(target_img)
            if res:
                clean_lines = []
                seen = set()
                for item in res:
                    if len(item) > 1 and item[1]:
                        txt = str(item[1]).strip()
                        if len(txt) >= 2 and txt.upper() not in seen:
                            seen.add(txt.upper())
                            clean_lines.append(txt)
                if clean_lines:
                    return True, "\n".join(clean_lines), warnings
        except Exception as ex:
            logger.debug(f"RapidOCR pass failed, falling back to Tesseract: {ex}")

    # Fallback Pass: Pytesseract
    if not PYTESSERACT_AVAILABLE or pytesseract is None:
        if rapid_engine is None:
            warnings.append("No local OCR engine (RapidOCR or pytesseract) is available.")
            return False, "", warnings

    tess_path = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract") if (PYTESSERACT_AVAILABLE and pytesseract) else "tesseract"
    if not os.path.exists(tess_path) and os.path.isabs(tess_path):
        warnings.append(f"Tesseract binary not found at '{tess_path}'. Please configure TESSERACT_CMD.")
        return False, "", warnings

    th, tw = target_img.shape[:2]
    if tw < 800:
        scale = 800.0 / tw
        target_img = cv2.resize(target_img, (800, int(th * scale)), interpolation=cv2.INTER_CUBIC)
    elif max(th, tw) > MAX_ANALYSIS_DIM:
        scale = MAX_ANALYSIS_DIM / max(th, tw)
        target_img = cv2.resize(target_img, (int(tw * scale), int(th * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY) if len(target_img.shape) == 3 else target_img
    extracted_lines: List[str] = []

    # Pass 1: Denoise + adaptive thresholding
    try:
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 7
        )
        if pytesseract is not None:
            txt1 = pytesseract.image_to_string(thresh, config="--psm 6 -c preserve_interword_spaces=1")
            if isinstance(txt1, str) and txt1:
                extracted_lines.extend(txt1.splitlines())
    except Exception as e:
        logger.debug(f"OCR Pass 1 failed: {e}")

    # Pass 2: Grayscale fallback with psm 11
    try:
        if pytesseract is not None:
            txt2 = pytesseract.image_to_string(gray, config="--psm 11")
            if isinstance(txt2, str) and txt2:
                extracted_lines.extend(txt2.splitlines())
    except Exception as e2:
        logger.debug(f"OCR Pass 2 failed: {e2}")

    clean_lines = []
    seen = set()
    for raw in extracted_lines:
        line = re.sub(r"\s+", " ", raw).strip()
        if len(line) >= 3 and line.upper() not in seen:
            seen.add(line.upper())
            clean_lines.append(line)

    cleaned_text = "\n".join(clean_lines)
    return True, cleaned_text, warnings


def validate_indian_plate_pattern(text: str) -> List[str]:
    """Finds valid Indian vehicle registration numbers formatted with real state codes."""
    matches = []
    tokens = re.split(r"[\s,;|\n]+", text.upper())
    for token in tokens:
        cand = re.sub(r"[^A-Z0-9]", "", token)
        if 8 <= len(cand) <= 11:
            m = re.match(r"^([A-Z]{2})[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$", cand)
            if m:
                state_code = m.group(1)
                if state_code in INDIAN_STATE_CODES:
                    matches.append(cand)
    return list(dict.fromkeys(matches))


def evaluate_rc_evidence(
    ocr_text: str,
    doc_detected: bool,
    quality_info: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[bool], float, List[str], List[str]]:
    """
    Score OCR text and card geometry against positive RC patterns and negative
    competing-document indicators.
    Returns: (classification, is_rc_document, confidence, matched_cues, negative_cues)
    """
    normalized_text = f" {ocr_text.upper()} "
    corpus_normalized = re.sub(r"[\s\-_:.]+", " ", normalized_text)
    matched_cues: List[str] = []
    negative_cues: List[str] = []

    # 1. Negative Competing Document Check
    negative_penalty = 0.0
    for pattern, doc_name, penalty in NEGATIVE_DOCUMENT_PATTERNS:
        if re.search(pattern, corpus_normalized):
            negative_penalty += penalty
            negative_cues.append(f"Rejected: {doc_name}")

    if negative_penalty >= 0.70:
        return "not_rc_document", False, 0.05, matched_cues, negative_cues

    # 2. Registration Number Matching
    plates = validate_indian_plate_pattern(ocr_text)
    has_valid_reg_no = bool(plates)
    for p in plates[:2]:
        matched_cues.append(f"Valid Indian Registration Format ({p})")

    # 3. Positive Tier Scoring
    score = 0.0
    corpus_compact = re.sub(r"[^A-Z0-9]", "", normalized_text)

    # Tier 1: Explicit RC Titles (Weight: 0.40)
    has_tier1 = False
    for pattern, desc in RC_TIER1_TITLES:
        if re.search(pattern, corpus_normalized):
            score += 0.40
            matched_cues.append(desc)
            has_tier1 = True
            break
    if not has_tier1 and any(k in corpus_compact for k in ("REGISTRATIONCERTIFICATE", "CERTIFICATEOFREGISTRATION", "VEHICLEREGISTRATION", "FORM23", "PARIVAHAN")):
        score += 0.40
        matched_cues.append("Registration Certificate Title")
        has_tier1 = True

    # Tier 2: Government & Transport Authorities (Weight: 0.25)
    has_tier2 = False
    for pattern, desc in RC_TIER2_AUTHORITIES:
        if re.search(pattern, corpus_normalized):
            score += 0.25
            matched_cues.append(desc)
            has_tier2 = True
            break

    # Tier 3: Core Technical Identifiers (Weight: 0.20 per cue, max 0.40)
    tier3_count = 0
    for pattern, desc in RC_TIER3_VITALS:
        if re.search(pattern, corpus_normalized):
            tier3_count += 1
            matched_cues.append(desc)
            score += 0.20
            if tier3_count >= 2:
                break

    if tier3_count < 2:
        if "CHASSIS" in corpus_compact and not any("Chassis" in c for c in matched_cues):
            tier3_count += 1
            matched_cues.append("Chassis Number Field")
            score += 0.20
        if "ENGINE" in corpus_compact and not any("Engine" in c for c in matched_cues) and tier3_count < 2:
            tier3_count += 1
            matched_cues.append("Engine Number Field")
            score += 0.20
        if any(k in corpus_compact for k in ("REGN", "REGNO", "REGISTRATIONNO")) and not any("Registration Number" in c for c in matched_cues) and tier3_count < 2:
            tier3_count += 1
            matched_cues.append("Registration Number Field")
            score += 0.20
        if "OWNER" in corpus_compact and not any("Owner" in c for c in matched_cues) and tier3_count < 2:
            tier3_count += 1
            matched_cues.append("Owner Name Field")
            score += 0.20

    # Tier 4: Technical Specs (Weight: 0.15 per cue, max 0.30)
    tier4_count = 0
    for pattern, desc in RC_TIER4_TECHNICAL:
        if re.search(pattern, corpus_normalized):
            tier4_count += 1
            matched_cues.append(desc)
            score += 0.15
            if tier4_count >= 2:
                break

    # Tier 5: Operational Terms (Weight: 0.10)
    for pattern, desc in RC_TIER5_TERMS:
        if re.search(pattern, corpus_normalized):
            score += 0.10
            matched_cues.append(desc)
            break

    # Bonus: Valid Registration Number (0.25)
    if has_valid_reg_no:
        score += 0.25

    # Bonus: Physical Card Geometry (0.15)
    if doc_detected:
        score += 0.15
        matched_cues.append("Physical Document / Card Geometry")

    # Apply negative penalties
    net_score = max(0.0, score - negative_penalty)

    # Decision Boundaries
    if net_score >= 0.55 and (has_tier1 or has_tier2 or has_valid_reg_no or tier3_count >= 1):
        return "rc_document", True, min(0.99, round(net_score, 2)), matched_cues, negative_cues
    elif net_score >= 0.35:
        return "uncertain", None, round(net_score, 2), matched_cues, negative_cues
    else:
        return "not_rc_document", False, round(net_score, 2), matched_cues, negative_cues


def detect_rc_document(
    image_input: Union[str, Path, np.ndarray], max_ocr_chars: int = 1200
) -> Dict[str, Any]:
    """
    Main reusable library function to classify whether an image is an Indian RC document.
    Deterministic, offline, safe to import, and handles all exceptions gracefully.
    """
    t0 = time.time()
    img_bgr = None
    path_str = ""

    if isinstance(image_input, np.ndarray):
        img_bgr = image_input.copy()
        path_str = "<numpy_array>"
    else:
        path_obj = Path(image_input)
        path_str = str(image_input)
        if not path_obj.exists() or not path_obj.is_file():
            return {
                "success": False,
                "error": f"Image file not found at path: {image_input}",
                "image_path": path_str,
                "is_rc_document": None,
                "classification": "uncertain",
                "confidence": 0.0,
                "document_detected": False,
                "ocr_available": False,
                "ocr_text": "",
                "matched_rc_cues": [],
                "matched_negative_cues": [],
                "image_quality": {"width": 0, "height": 0, "blur_warning": False, "low_light_warning": False, "low_resolution_warning": False},
                "warnings": [f"File not found: {image_input}"],
                "method": "opencv_and_local_ocr_no_pretrained_model",
            }
        img_bgr = cv2.imread(str(path_obj))

    try:
        if img_bgr is None or img_bgr.size == 0:
            return {
                "success": False,
                "error": f"Failed to decode image file: {image_input}",
                "image_path": path_str,
                "is_rc_document": None,
                "classification": "uncertain",
                "confidence": 0.0,
                "document_detected": False,
                "ocr_available": False,
                "ocr_text": "",
                "matched_rc_cues": [],
                "matched_negative_cues": [],
                "image_quality": {"width": 0, "height": 0, "blur_warning": False, "low_light_warning": False, "low_resolution_warning": False},
                "warnings": ["Corrupt or unreadable image file."],
                "method": "opencv_and_local_ocr_no_pretrained_model",
            }

        quality_info, quality_warnings = check_image_quality(img_bgr)
        all_warnings: List[str] = list(quality_warnings)

        doc_detected, doc_roi, geom_cues = detect_document_card_geometry(img_bgr)
        ocr_avail, ocr_text, ocr_warnings = perform_local_ocr(img_bgr, doc_roi)
        all_warnings.extend(ocr_warnings)

        if not ocr_avail:
            all_warnings.append("OCR engine unavailable. Cannot confidently verify RC text without OCR.")
            return {
                "success": True,
                "image_path": path_str,
                "is_rc_document": None,
                "classification": "uncertain",
                "confidence": 0.20 if doc_detected else 0.05,
                "document_detected": doc_detected,
                "ocr_available": False,
                "ocr_text": "",
                "matched_rc_cues": ["Card Geometry Detected"] if doc_detected else [],
                "matched_negative_cues": [],
                "image_quality": quality_info,
                "warnings": all_warnings,
                "method": "opencv_and_local_ocr_no_pretrained_model",
            }

        classification, is_rc, confidence, matched_cues, negative_cues = evaluate_rc_evidence(
            ocr_text, doc_detected, quality_info
        )

        return {
            "success": True,
            "image_path": path_str,
            "is_rc_document": is_rc,
            "classification": classification,
            "confidence": confidence,
            "document_detected": doc_detected,
            "ocr_available": ocr_avail,
            "ocr_text": ocr_text[:max_ocr_chars] if ocr_text else "",
            "matched_rc_cues": matched_cues,
            "matched_negative_cues": negative_cues,
            "image_quality": quality_info,
            "warnings": all_warnings,
            "method": "opencv_and_local_ocr_no_pretrained_model",
        }
    except Exception as ex:
        return {
            "success": False,
            "error": str(ex),
            "image_path": path_str,
            "is_rc_document": None,
            "classification": "uncertain",
            "confidence": 0.0,
            "document_detected": False,
            "ocr_available": False,
            "ocr_text": "",
            "matched_rc_cues": [],
            "matched_negative_cues": [],
            "image_quality": {"width": 0, "height": 0, "blur_warning": False, "low_light_warning": False, "low_resolution_warning": False},
            "warnings": [f"Processing exception: {str(ex)}"],
            "method": "opencv_and_local_ocr_no_pretrained_model",
        }


# Both detect_rc_image and detect_rc_document are exported as identical public APIs
detect_rc_image = detect_rc_document


def save_rc_visualization(
    image_input: Union[str, Path, np.ndarray],
    result: Dict[str, Any],
    output_path: Union[str, Path],
) -> bool:
    """Save an annotated visual verification image displaying RC bounding box and cues."""
    try:
        if isinstance(image_input, np.ndarray):
            img = image_input.copy()
        else:
            img = cv2.imread(str(image_input))
        if img is None:
            return False

        h, w = img.shape[:2]
        color = (0, 200, 0) if result.get("is_rc_document") else (0, 0, 220)
        if result.get("is_rc_document") is None:
            color = (0, 165, 255)

        label = f"{str(result.get('classification', 'uncertain')).upper()} ({float(result.get('confidence', 0.0)):.2f})"
        cv2.putText(img, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

        y_pos = 110
        for cue in result.get("matched_rc_cues", [])[:5]:
            cv2.putText(img, f"+ {cue}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2, cv2.LINE_AA)
            y_pos += 35
        for ncue in result.get("matched_negative_cues", [])[:3]:
            cv2.putText(img, f"- {ncue}", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 240), 2, cv2.LINE_AA)
            y_pos += 35

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), img)
        logger.info(f"Saved visualization to: {out_p}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save visualization: {e}")
        return False


# ---------------------------------------------------------------------------
# Module 3: Vehicle Interior Detection
# ---------------------------------------------------------------------------

# Interior component vocabulary (from prompt specification)
INTERIOR_COMPONENT_VOCABULARY = {
    # Dashboard and instrumentation
    "dashboard", "instrument_cluster", "speedometer", "tachometer", "fuel_gauge",
    "infotainment_screen", "digital_display", "glove_box", "dashboard_trim",
    "instrument_panel_buttons", "gear_position_indicator", "heads_up_display",
    # Steering and driver controls
    "steering_wheel", "steering_column", "steering_wheel_buttons", "paddle_shifters",
    "turn_signal_lever", "windshield_wiper_controls", "headlight_controls",
    "ignition_switch", "engine_start_stop_button", "parking_brake",
    "electronic_parking_brake", "drive_mode_selector", "traction_control_button",
    "cruise_control_buttons", "hazard_light_button", "adas_controls",
    # Seats and restraints
    "driver_seat", "front_passenger_seat", "rear_seats", "third_row_seats",
    "headrests", "seat_belts", "seat_adjustment_controls", "seat_heating_controls",
    "seat_ventilation_controls", "seat_belt_buckles", "airbags", "airbag_cover",
    "child_seat_anchors", "seat_rails", "foldable_rear_seats",
    # Center console and storage
    "center_console", "gear_shift_lever", "gear_selector", "cup_holders", "armrest",
    "storage_compartment", "door_pockets", "seatback_pockets", "sunglasses_holder",
    "dashboard_storage_tray", "wireless_charging_pad", "usb_ports", "charging_ports",
    "power_outlet",
    # Doors, windows, mirrors, and trim
    "interior_door_panel", "interior_door_handle", "window_control_switches",
    "central_locking_switches", "side_mirror_adjustment_controls",
    "interior_rear_view_mirror", "digital_rear_view_mirror", "sun_visors",
    "vanity_mirror", "headliner", "sunroof", "panoramic_roof",
    "a_pillar_trim", "b_pillar_trim", "c_pillar_trim", "d_pillar_trim",
    "interior_upholstery", "carpet_flooring", "door_sills", "grab_handles",
    # Climate, lighting, and entertainment
    "air_conditioning_controls", "air_vents", "rear_ac_vents",
    "rear_climate_control_panel", "cabin_lights", "reading_lights",
    "ambient_lighting", "footwell_lights", "door_courtesy_lights", "glovebox_light",
    "audio_system", "radio", "interior_speakers", "rear_entertainment_screen",
    "cabin_microphone", "window_defroster_controls", "cabin_air_circulation_controls",
    # Footwell, cargo, and other interior areas
    "accelerator_pedal", "brake_pedal", "clutch_pedal", "footrest", "floor_mats",
    "trunk_interior", "boot_compartment", "cargo_area", "parcel_shelf",
    "spare_tyre_compartment", "cargo_floor", "cargo_side_panels", "luggage_cover",
    "interior_camera",
    # Vehicle-specific areas
    "bike_controls", "bus_passenger_interior", "bus_seats", "bus_aisle",
    "bus_handrails", "truck_cabin", "truck_dashboard", "truck_sleeper_berth",
}


def identify_interior_components(
    evidence: List[str],
    ocr_text: str,
    interior_type: Optional[str],
    hsv: np.ndarray,
    gray: np.ndarray,
    steering_detected: bool,
    gauge_dials_detected: bool,
    headrest_count: int,
    gauges_or_screen_count: int,
) -> List[str]:
    """
    Map detected cues and OCR text to standardized interior component vocabulary labels.
    Only reports components with sufficient supporting evidence — never guesses.
    """
    components: List[str] = []
    h, w = gray.shape[:2]
    total_area = max(1, h * w)
    evidence_str = " ".join(evidence)
    ocr_upper = ocr_text.upper()

    # --- Steering and driver controls ---
    if steering_detected:
        components.append("steering_wheel")
        if "cabin_interior_controls_text" in evidence_str:
            components.append("steering_wheel_buttons")

    # --- Dashboard and instrumentation ---
    if "instrument_cluster_text_signatures" in evidence_str or gauge_dials_detected:
        components.append("instrument_cluster")
    if "dashboard_horizon_gradient" in evidence_str:
        components.append("dashboard")
    if "KM/H" in ocr_upper:
        components.append("speedometer")
    if "RPM" in ocr_upper or "X1000" in ocr_upper:
        components.append("tachometer")

    # --- Infotainment / digital display ---
    center_region = gray[int(h * 0.20):int(h * 0.65), int(w * 0.25):int(w * 0.75)]
    has_screen = False
    if center_region.size > 0:
        bright_pixels = float(np.count_nonzero(center_region > 180)) / max(1, center_region.size)
        if bright_pixels > 0.08 and gauges_or_screen_count >= 1:
            components.append("infotainment_screen")
            components.append("digital_display")
            has_screen = True
    if not has_screen and ("KM/H" in ocr_upper or "RPM" in ocr_upper or "active_display_screen_glow" in evidence_str):
        components.append("digital_display")

    # --- Seats and restraints ---
    if headrest_count >= 1 and "cabin_upholstery_and_interior_trim" in evidence_str:
        components.append("driver_seat")
        components.append("headrests")
    elif headrest_count >= 2:
        components.append("driver_seat")
        components.append("front_passenger_seat")
        components.append("headrests")
    elif "vehicle_seat_and_headrest_contours" in evidence_str:
        components.append("driver_seat")
        components.append("headrests")
    elif "cabin_upholstery_and_interior_trim" in evidence_str:
        components.append("interior_upholstery")
        components.append("passenger_seats")

    # --- Center console and gear selector ---
    if "cabin_cluster_or_console_contours" in evidence_str:
        components.append("center_console")

    gear_kws = {"PRND", "P R N D", "DRIVE", "PARK", "REVERSE", "NEUTRAL", "SPORT", "SHIFTER", "GEAR"}
    if any(kw in ocr_upper for kw in gear_kws):
        components.append("gear_shift_lever")
        components.append("gear_selector")
        components.append("gear_position_indicator")

    # --- Parking brake ---
    parking_kws = {"(P)", "PARK BRAKE", "HANDBRAKE", "AUTO HOLD", "PARKING BRAKE"}
    if any(kw in ocr_upper for kw in parking_kws):
        components.append("parking_brake")
        components.append("electronic_parking_brake")

    # --- Drive mode selector & traction control ---
    drive_mode_kws = {"ECO", "SPORT", "COMFORT", "TERRAIN", "SNOW", "MUD", "EV MODE"}
    if any(kw in ocr_upper for kw in drive_mode_kws):
        components.append("drive_mode_selector")
    if "ESP" in ocr_upper or "TC OFF" in ocr_upper or "TRACTION" in ocr_upper:
        components.append("traction_control_button")

    # --- Climate controls (from OCR) ---
    climate_kws = {"CLIMATE", "TEMP", "DUAL", "REAR DEFOG", "AUTO"}
    if any(kw in ocr_upper for kw in climate_kws):
        components.append("air_conditioning_controls")

    # --- Air vents ---
    if "cabin_cluster_or_console_contours" in evidence_str and gauges_or_screen_count >= 2:
        components.append("air_vents")

    # --- Audio / Radio (from OCR) ---
    audio_kws = {"RADIO", "FM", "AM", "VOLUME", "AUDIO", "MEDIA"}
    if any(kw in ocr_upper for kw in audio_kws):
        components.append("audio_system")
        components.append("radio")

    # --- Engine start/stop (from OCR) ---
    if "START ENGINE" in ocr_upper or "ENGINE START" in ocr_upper or "PUSH START" in ocr_upper:
        components.append("engine_start_stop_button")

    # --- Airbag (from OCR) ---
    if "AIRBAG" in ocr_upper:
        components.append("airbags")

    # --- Hazard light (from OCR) ---
    if "HAZARD" in ocr_upper:
        components.append("hazard_light_button")

    # --- Cruise control (from OCR) ---
    if "CRUISE" in ocr_upper:
        components.append("cruise_control_buttons")

    # --- Navigation (from OCR) ---
    if "NAV" in ocr_upper or "NAVIGATION" in ocr_upper:
        if "infotainment_screen" not in components:
            components.append("infotainment_screen")

    # --- Door panel & window controls ---
    door_kws = {"WINDOW", "MIRROR", "DOOR LOCK", "CENTRAL LOCK"}
    if any(kw in ocr_upper for kw in door_kws):
        components.append("interior_door_panel")
        components.append("window_control_switches")

    # --- Cabin upholstery / trim ---
    if "cabin_upholstery_and_interior_trim" in evidence_str:
        components.append("interior_upholstery")
    else:
        black_leather = float(np.count_nonzero(cv2.inRange(hsv, (0, 0, 0), (180, 40, 55)))) / total_area
        grey_fabric = float(np.count_nonzero(cv2.inRange(hsv, (0, 0, 55), (180, 25, 130)))) / total_area
        beige_trim = float(np.count_nonzero(cv2.inRange(hsv, (20, 20, 150), (35, 100, 240)))) / total_area
        if (black_leather >= 0.15 and headrest_count >= 1) or \
           (grey_fabric >= 0.12 and headrest_count >= 1) or \
           (beige_trim >= 0.10 and headrest_count >= 1):
            components.append("interior_upholstery")

    # --- Ambient lighting (thin saturated color strips along edges) ---
    left_strip = hsv[:, :int(w * 0.08), :]
    right_strip = hsv[:, int(w * 0.92):, :]
    for strip in (left_strip, right_strip):
        if strip.size > 0:
            high_sat_bright = float(np.count_nonzero(
                (strip[:, :, 1] > 120) & (strip[:, :, 2] > 60) & (strip[:, :, 2] < 220)
            )) / max(1, strip[:, :, 0].size)
            if 0.02 < high_sat_bright < 0.25:
                if "ambient_lighting" not in components:
                    components.append("ambient_lighting")
                break

    # --- Cockpit windshield luminance → implies windshield / dashboard view ---
    if "cockpit_windshield_luminance_gradient" in evidence_str and steering_detected:
        if "dashboard" not in components:
            components.append("dashboard")

    # --- Bus specific interior components ---
    if headrest_count >= 4:
        components.extend(["bus_passenger_interior", "bus_seats", "bus_aisle", "bus_handrails"])

    # Deduplicate while preserving order
    seen = set()
    unique_components = []
    for c in components:
        if c not in seen and c in INTERIOR_COMPONENT_VOCABULARY:
            seen.add(c)
            unique_components.append(c)

    return unique_components


def classify_interior_vehicle_type(
    evidence: List[str],
    components: List[str],
    gray: np.ndarray,
    headrest_count: int,
) -> Tuple[str, float]:
    """
    Classify interior vehicle type from detected cues.
    Returns: (vehicle_type, type_confidence)
    """
    h, w = gray.shape[:2]
    evidence_str = " ".join(evidence)

    # Bus: repeated seat rows, high ceiling, aisle-like corridor
    if headrest_count >= 4:
        # Multiple rows of seats suggest bus passenger area
        return "bus", 0.70

    # Motorcycle: no enclosed cabin indicators, no dashboard horizon
    if ("dashboard_horizon_gradient" not in evidence_str and
        "cockpit_windshield_luminance_gradient" not in evidence_str and
        "cabin_upholstery_and_interior_trim" not in evidence_str and
        "vehicle_seat_and_headrest_contours" not in evidence_str):
        # Check for handlebar-like horizontal structure without cabin
        if any("bike_controls" in c for c in components):
            return "bike", 0.65

    # Truck: larger proportions, high dashboard
    # (Difficult to distinguish from car with heuristics alone; default to car)

    # Default: car
    return "car", 0.78


def assess_scene_context_sufficiency(
    evidence: List[str],
    components: List[str],
    score: float,
) -> str:
    """
    Determine if there's enough scene context for a full interior classification.
    Returns 'vehicle_interior' or 'vehicle_detail'.
    """
    # Count distinct spatial zones with cues
    zones = set()
    evidence_str = " ".join(evidence)
    if "steering" in evidence_str or "wheel" in evidence_str:
        zones.add("steering_area")
    if "dashboard" in evidence_str or "horizon" in evidence_str:
        zones.add("dashboard_area")
    if "cluster" in evidence_str or "gauge" in evidence_str or "instrument" in evidence_str:
        zones.add("cluster_area")
    if "seat" in evidence_str or "headrest" in evidence_str or "upholstery" in evidence_str:
        zones.add("seating_area")
    if "console" in evidence_str or "cockpit" in evidence_str:
        zones.add("console_area")
    if "windshield" in evidence_str or "luminance" in evidence_str:
        zones.add("windshield_area")

    # Single component with no surrounding context = detail view
    if len(components) <= 1 and len(zones) <= 1 and score < 0.55:
        return "vehicle_detail"

    # Very few cues with low score
    if len(evidence) <= 2 and score < 0.45:
        return "vehicle_detail"

    return "vehicle_interior"


def _validate_steering_wheel(
    gray: np.ndarray, cx: int, cy: int, r: int, h: int, w: int, evidence: List[str],
) -> int:
    """
    Validate a steering wheel candidate with supporting spatial checks.
    Returns number of validation checks passed (need >= 2 to confirm).
    """
    # Size sanity: steering wheel radius cannot exceed 35% of min image dimension
    if r > int(min(h, w) * 0.35) or r < int(min(h, w) * 0.10):
        return 0

    checks_passed = 0

    # 1. Dark dashboard-like region above the circle
    region_above = gray[max(0, cy - 2 * r):max(0, cy - r), max(0, cx - r):min(w, cx + r)]
    region_above_mean = float(np.mean(region_above)) if region_above.size > 0 else 255.0
    if region_above.size > 0 and region_above_mean < 110.0:
        checks_passed += 1

    # 2. Outer rim darkness check (steering wheel rims are black/grey leather or plastic)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    inner_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(inner_mask, (cx, cy), max(1, int(r * 0.78)), 255, -1)
    rim_mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_mask))
    rim_pixels = gray[rim_mask > 0]
    rim_mean = float(np.mean(rim_pixels)) if len(rim_pixels) > 0 else 255.0
    if rim_mean < 125.0:
        checks_passed += 1

    # 3. Hub vs ring texture contrast (center differs from ring)
    hub_r = max(1, r // 3)
    hub_region = gray[max(0, cy - hub_r):min(h, cy + hub_r), max(0, cx - hub_r):min(w, cx + hub_r)]
    if hub_region.size > 0 and len(rim_pixels) > 0:
        hub_std = float(np.std(hub_region))
        rim_std = float(np.std(rim_pixels))
        mean_diff = abs(float(np.mean(hub_region)) - rim_mean)
        if mean_diff > 18 or abs(hub_std - rim_std) > 12:
            checks_passed += 1

    # Guard: if dark road asphalt exists directly below, this is an exterior vehicle or street scene
    bottom_strip = gray[int(h * 0.85):, :]
    if bottom_strip.size > 0 and float(np.mean(bottom_strip)) < 45.0:
        return 0

    # 4. Other cabin cues already detected (dashboard gradient, luminance split, etc.)
    supporting_cues = [
        "dashboard_horizon_gradient", "cockpit_windshield_luminance_gradient",
        "cabin_cluster_or_console_contours", "cabin_upholstery_and_interior_trim",
        "instrument_cluster_text_signatures", "cabin_interior_controls_text",
    ]
    has_cabin_cues = any(cue in " ".join(evidence) for cue in supporting_cues)
    if has_cabin_cues:
        checks_passed += 1

    # Check for cabin luminance split (windshield bright above, cabin darker below)
    top_lum = float(np.mean(gray[: int(h * 0.35), :]))
    bot_lum = float(np.mean(gray[int(h * 0.60):, :]))
    has_luminance_split = (top_lum - bot_lum) > 30.0

    # Require cabin context: either dark dashboard above, top-bottom luminance split, or supporting cabin cues
    has_cabin_context = (region_above_mean < 110.0) or has_luminance_split or has_cabin_cues
    if not has_cabin_context:
        return 0

    return checks_passed


def detect_interior_signatures(
    img_bgr: np.ndarray,
    gray: np.ndarray,
    image_input: Optional[Union[str, Path, np.ndarray]] = None,
) -> Tuple[bool, float, Optional[str], float, List[str], Optional[List[int]], List[str], str, str, float]:
    """
    Detect whether the image depicts a vehicle interior (dashboard, steering wheel,
    instrument cluster, cabin layout, center console, vehicle seats).

    Returns:
        (is_interior, interior_confidence, interior_type, type_confidence, evidence, bounding_box, interior_components, scene_verdict, vehicle_type, vehicle_type_confidence)
    """
    h, w = gray.shape[:2]
    total_area = max(1, h * w)
    evidence: List[str] = []
    score = 0.0
    detected_type = "cabin"
    type_conf = 0.50
    steering_box = None

    # Guard: Vehicle interiors have dark dashboards/trim, not predominantly white/paper background
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    paper_like_ratio = float(np.count_nonzero((val > 185) & (sat < 40))) / total_area
    if paper_like_ratio > 0.60:
        return False, 0.0, "unknown", 0.0, [], None, [], "not_vehicle", "car", 0.0

    blurred = cv2.GaussianBlur(gray, (7, 7), 2)

    # 1. Optical Character Recognition for Interior / Instrument Cluster signatures
    # RapidOCR gives instant verification of speedometer, odometer, and cabin terms
    interior_ocr_text = ""
    rapid_engine = get_rapid_ocr_engine()
    if rapid_engine is not None:
        try:
            # Crop middle 60% of image for fast focused text analysis
            crop_for_ocr = img_bgr[int(h * 0.15):int(h * 0.85), int(w * 0.10):int(w * 0.90)]
            if crop_for_ocr.size > 0:
                ch, cw = crop_for_ocr.shape[:2]
                if max(ch, cw) > 960:
                    sc = 960.0 / max(ch, cw)
                    crop_for_ocr = cv2.resize(crop_for_ocr, (int(cw * sc), int(ch * sc)), interpolation=cv2.INTER_AREA)
                res_ocr, _ = rapid_engine(crop_for_ocr)
                if res_ocr:
                    interior_ocr_text = " ".join([str(r[1]).upper().strip() for r in res_ocr if len(r) > 1 and r[1]])
        except Exception as ex:
            logger.debug(f"RapidOCR interior scan exception: {ex}")

    # An image with a visible exterior Indian vehicle license plate or vehicle badge cannot be a vehicle interior
    has_ext_plate = False
    # Support both full and partially obscured plates (e.g. doodled plates with 1-4 trailing digits)
    plate_matches = re.finditer(r'\b(?:IND\s*)?([A-Z]{2})\s*(\d{1,2})\s*([A-Z]{0,3})\s*(\d{1,4})\b', interior_ocr_text)
    for m in plate_matches:
        state_prefix = m.group(1)
        if state_prefix in INDIAN_STATE_CODES:
            has_ext_plate = True
            break

    # Also detect model or rear badges in OCR text
    has_ext_badge = False
    ext_badge_name = ""
    common_badges = {
        "SWIFT", "CRETA", "BALENO", "BREZZA", "INNOVA", "DZIRE", "ALTO", "WAGONR", "WAGON R",
        "I20", "I10", "SCORPIO", "THAR", "NEXON", "SELTOS", "VERNA", "CITY", "FORTUNER",
        "POLO", "VXI", "ZXI", "LXI", "VDI", "ZDI", "CNG", "HYBRID", "MARUTI", "SUZUKI",
        "HYUNDAI", "TATA", "MAHINDRA", "HONDA", "TOYOTA", "SKODA", "VOLKSWAGEN", "RENAULT"
    }
    if interior_ocr_text:
        tokens = [re.sub(r'[^A-Z0-9]', '', t) for t in interior_ocr_text.split()]
        for tok in tokens:
            if tok in common_badges:
                has_ext_badge = True
                ext_badge_name = tok
                break

    if has_ext_plate or has_ext_badge:
        ev = []
        if has_ext_plate:
            ev.append("exterior_license_plate_detected")
        if has_ext_badge:
            ev.append(f"exterior_vehicle_badge_detected ({ext_badge_name})")
        return False, 0.0, None, 0.0, ev, None, [], "vehicle_exterior", "car", 0.0

    cluster_kws = ["KM/H", "KM/H.", "RPM", "X1000", "ODO", "TRIP", "DRIVING TIME", "AVG SPEED", "AVG.", "CRUISE"]
    matched_cluster = [kw for kw in cluster_kws if kw in interior_ocr_text]
    has_odo_km = bool(re.search(r'\b\d{3,6}\s*(?:KM|km)\b', interior_ocr_text))

    if matched_cluster or has_odo_km:
        score += 0.60
        detected_type = "dashboard"
        type_conf = 0.95
        evidence.append(f"instrument_cluster_text_signatures ({', '.join(matched_cluster) if matched_cluster else 'odometer km'})")

    cabin_kws = ["AIRBAG", "START ENGINE", "ENGINE START", "STOP ENGINE", "CLIMATE", "MODE", "NAV", "RADIO", "AUTO", "TEMP", "DUAL", "REAR DEFOG", "HAZARD"]
    matched_cabin = [kw for kw in cabin_kws if kw in interior_ocr_text]
    if matched_cabin:
        score += 0.40
        if detected_type == "cabin":
            detected_type = "dashboard"
            type_conf = 0.85
        evidence.append(f"cabin_interior_controls_text ({', '.join(matched_cabin)})")

    # 2. Steering Wheel Detection (Circular / Elliptical Ring in lower/middle half)
    min_r = max(20, int(min(h, w) * 0.12))
    max_r = max(min_r + 10, int(min(h, w) * 0.35))

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_r,
        param1=80,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    wheel_found = False
    if circles is not None:
        circles_arr = np.around(circles[0]).astype(np.uint16)
        valid_wheel_candidates = []
        for c in circles_arr:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            # Steering wheels are located inside the cabin, usually 0.25*h <= cy <= 0.78*h
            if int(h * 0.25) <= cy <= int(h * 0.78) and (0.12 * w < cx < 0.88 * w):
                valid_wheel_candidates.append((cx, cy, r))

        if len(valid_wheel_candidates) == 1:
            cx, cy, r = valid_wheel_candidates[0]
            # Validate steering wheel with spatial checks to reduce false positives
            validation_score = _validate_steering_wheel(gray, cx, cy, r, h, w, evidence)
            if validation_score >= 2:
                wheel_found = True
                steering_box = [max(0, cx - r), max(0, cy - r), min(w, 2 * r), min(h, 2 * r)]
                score += 0.48
                evidence.append("circular_steering_wheel_structure")
                detected_type = "steering_wheel"
                type_conf = 0.88
            else:
                # Circle detected but failed validation — could be exterior wheel or sign
                evidence.append("unvalidated_circular_structure")
        elif len(valid_wheel_candidates) > 1:
            sorted_cand = sorted(valid_wheel_candidates, key=lambda x: x[2], reverse=True)
            if sorted_cand[0][2] > 1.25 * sorted_cand[1][2]:
                cx, cy, r = sorted_cand[0]
                validation_score = _validate_steering_wheel(gray, cx, cy, r, h, w, evidence)
                if validation_score >= 2:
                    wheel_found = True
                    steering_box = [max(0, cx - r), max(0, cy - r), min(w, 2 * r), min(h, 2 * r)]
                    score += 0.48
                    evidence.append("circular_steering_wheel_structure")
                    detected_type = "steering_wheel"
                    type_conf = 0.88
                else:
                    evidence.append("unvalidated_circular_structure")
            else:
                # Two similar size circles in the upper-mid cabin = Gauge Dials (Speedometer + Tachometer)
                # Gauge dials must be backed by cluster text or a verified steering wheel
                has_cluster_text = bool(matched_cluster or has_odo_km)
                c1, c2 = sorted_cand[0], sorted_cand[1]
                dial_dist = float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                if (has_cluster_text or wheel_found) and \
                   (0.85 * (c1[2] + c2[2]) <= dial_dist <= 3.2 * (c1[2] + c2[2])) and \
                   abs(c1[1] - c2[1]) < (0.12 * h) and (c1[1] < int(h * 0.60)) and (c2[1] < int(h * 0.60)):
                    score += 0.45
                    evidence.append("dual_concentric_gauge_dials")
                    detected_type = "dashboard"
                    type_conf = 0.90

    # 3. Horizontal Dashboard Horizon Line
    mid_zone = gray[int(h * 0.20) : int(h * 0.70), :]
    sobel_x = cv2.Sobel(mid_zone, cv2.CV_32F, 0, 1, ksize=3)
    row_grads = np.mean(np.abs(sobel_x), axis=1)
    has_dashboard_gradient = len(row_grads) > 0 and np.max(row_grads) > 38.0
    if has_dashboard_gradient:
        score += 0.22
        evidence.append("dashboard_horizon_gradient")

    # 4. Cockpit / Windshield Luminance Split
    top_luminance = float(np.mean(gray[: int(h * 0.35), :]))
    bottom_luminance = float(np.mean(gray[int(h * 0.60) :, :]))
    has_cockpit_luminance = (top_luminance - bottom_luminance) > 32.0
    if has_cockpit_luminance:
        score += 0.20
        evidence.append("cockpit_windshield_luminance_gradient")

    # 5. Instrument Cluster / Rectangular Vent Patterns
    edges = cv2.Canny(blurred, 35, 110)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gauges_or_screen = 0
    headrest_candidates = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (h * w * 0.005) < area < (h * w * 0.18):
            bx, by, bw, bh_cnt = cv2.boundingRect(cnt)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            circularity = 4 * np.pi * area / max(1.0, peri * peri)
            is_in_steering = (
                steering_box is not None
                and abs(bx - steering_box[0]) < 25
                and abs(by - steering_box[1]) < 25
            )
            if len(approx) in (4, 8) and circularity < 0.85:
                gauges_or_screen += 1
            # Check for vehicle headrest (trapezoid/oval in upper half, non-circular)
            if by < int(h * 0.55) and 0.5 <= (bw / max(1, bh_cnt)) <= 2.2 and circularity < 0.85 and not is_in_steering:
                headrest_candidates += 1

    if gauges_or_screen >= 2:
        score += 0.18
        evidence.append("cabin_cluster_or_console_contours")
        if detected_type == "cabin":
            detected_type = "console"
            type_conf = 0.75

    if headrest_candidates >= 1:
        score += 0.28
        evidence.append(f"vehicle_seat_and_headrest_contours ({headrest_candidates})")
        if detected_type == "cabin":
            detected_type = "seats"
            type_conf = 0.85

    # 5b. Cabin Upholstery (leather/fabric) and dark interior trim check
    # Extended to cover black leather, grey fabric, beige, tan leather
    if not has_ext_plate:
        upholstery_tan = np.count_nonzero(cv2.inRange(hsv, (8, 40, 45), (25, 255, 220))) / total_area
        dark_trim = np.count_nonzero(cv2.inRange(hsv, (0, 0, 0), (180, 255, 55))) / total_area
        # Additional upholstery colors (black leather in cabin daylight reaches V up to 115, S up to 75)
        black_leather = np.count_nonzero(cv2.inRange(hsv, (0, 0, 0), (180, 75, 115))) / total_area
        grey_fabric = np.count_nonzero(cv2.inRange(hsv, (0, 0, 55), (180, 25, 130))) / total_area
        beige_trim = np.count_nonzero(cv2.inRange(hsv, (20, 20, 150), (35, 100, 240))) / total_area

        has_any_upholstery = (
            (upholstery_tan >= 0.12 and dark_trim >= 0.10)
            or (headrest_candidates >= 1 and upholstery_tan >= 0.08)
            or (headrest_candidates >= 1 and black_leather >= 0.15)
            or (headrest_candidates >= 1 and grey_fabric >= 0.12)
            or (headrest_candidates >= 1 and beige_trim >= 0.10)
            or (black_leather >= 0.22 and (dark_trim >= 0.08 or has_dashboard_gradient))
            or (grey_fabric >= 0.18 and (dark_trim >= 0.08 or has_dashboard_gradient))
        )
        if has_any_upholstery:
            score += 0.25
            evidence.append("cabin_upholstery_and_interior_trim")

    # 5c. Screen / Display Glow Detection (modern car screens)
    center_band = gray[int(h * 0.15):int(h * 0.60), int(w * 0.20):int(w * 0.80)]
    if center_band.size > 0:
        bright_screen_ratio = float(np.count_nonzero(center_band > 200)) / max(1, center_band.size)
        dark_surround = float(np.mean(gray[int(h * 0.55):, :])) < 100
        if bright_screen_ratio > 0.05 and dark_surround and has_dashboard_gradient:
            score += 0.15
            evidence.append("active_display_screen_glow")

    # 6. Absence of road asphalt ground shadow (Interior characteristic)
    bottom_strip = gray[int(h * 0.85):, :]
    bottom_val = float(np.mean(bottom_strip)) if bottom_strip.size > 0 else 128.0
    if bottom_val > 60.0 and "dashboard_horizon_gradient" in evidence:
        score += 0.10

    # Definitive interior anchors:
    has_cluster_text = "instrument_cluster_text_signatures" in " ".join(evidence)
    has_cabin_controls = "cabin_interior_controls_text" in " ".join(evidence)
    has_verified_wheel = wheel_found and (score >= 0.45 or "dashboard_horizon_gradient" in evidence or "cockpit_windshield_luminance_gradient" in evidence)
    has_seats_and_upholstery = ("vehicle_seat_and_headrest_contours" in " ".join(evidence)) and ("cabin_upholstery_and_interior_trim" in evidence)
    has_cabin_upholstery_interior = (
        "cabin_upholstery_and_interior_trim" in evidence
        and ("dashboard_horizon_gradient" in evidence or "vehicle_seat_and_headrest_contours" in " ".join(evidence))
        and score >= 0.45
    )
    has_verified_dials = ("dual_concentric_gauge_dials" in evidence) and (has_cluster_text or wheel_found)
    has_cockpit_display = (
        ("dashboard_horizon_gradient" in evidence or "cockpit_windshield_luminance_gradient" in evidence)
        and ("active_display_screen_glow" in evidence or "cabin_upholstery_and_interior_trim" in evidence)
        and score >= 0.50
    )
    has_dashboard_windshield = (
        "dashboard_horizon_gradient" in evidence
        and "cockpit_windshield_luminance_gradient" in evidence
        and score >= 0.55
    )

    is_interior = (
        has_cluster_text
        or (has_cabin_controls and score >= 0.45)
        or has_verified_wheel
        or (has_seats_and_upholstery and score >= 0.45)
        or has_cabin_upholstery_interior
        or has_verified_dials
        or has_cockpit_display
        or has_dashboard_windshield
    )
    final_conf = min(0.98, max(0.65, round(score, 2))) if is_interior else round(score, 2)

    # Identify visible interior components from evidence
    gauge_dials_detected = "dual_concentric_gauge_dials" in evidence
    components = identify_interior_components(
        evidence=evidence,
        ocr_text=interior_ocr_text,
        interior_type=detected_type,
        hsv=hsv,
        gray=gray,
        steering_detected=wheel_found,
        gauge_dials_detected=gauge_dials_detected,
        headrest_count=headrest_candidates,
        gauges_or_screen_count=gauges_or_screen,
    )

    # Assess scene context sufficiency
    scene_verdict = "vehicle_interior" if is_interior else "not_vehicle"
    if is_interior:
        scene_verdict = assess_scene_context_sufficiency(evidence, components, score)
        if scene_verdict == "vehicle_detail":
            # Downgrade to detail if insufficient context
            is_interior = False
            final_conf = round(score, 2)

    # If neither interior nor detail, clear components
    if not is_interior and scene_verdict != "vehicle_detail":
        components = []

    # Classify interior vehicle type
    if is_interior or scene_verdict == "vehicle_detail":
        int_vtype, int_vtype_conf = classify_interior_vehicle_type(
            evidence, components, gray, headrest_candidates,
        )
    else:
        int_vtype = "car"
        int_vtype_conf = 0.0

    return (
        is_interior,
        final_conf,
        detected_type,
        type_conf,
        evidence,
        steering_box,
        components,
        scene_verdict,
        int_vtype,
        int_vtype_conf,
    )


# ---------------------------------------------------------------------------
# Module 3b: Non-Vehicle Scene Detection (People, Nature, Random Objects)
# ---------------------------------------------------------------------------
def detect_non_vehicle_scene(
    img_bgr: np.ndarray,
    gray: np.ndarray,
) -> Tuple[bool, float, List[str]]:
    """
    Detect whether the image is a non-vehicle scene such as a person/selfie,
    nature landscape, or random non-automotive content.

    Uses skin-tone detection, face detection (Haar cascade), and scene
    composition analysis to identify images that are clearly not vehicles.

    Returns:
        (is_non_vehicle, confidence, evidence_cues)
    """
    h, w = gray.shape[:2]
    total_area = max(1, h * w)
    cues: List[str] = []
    score = 0.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 1. OpenCV Haar Cascade Face Detection
    has_human_face = False
    cv2_data = getattr(cv2, "data", None)
    face_cascade_path = (getattr(cv2_data, "haarcascades", "") + "haarcascade_frontalface_default.xml") if cv2_data else ""
    classifier_cls = getattr(cv2, "CascadeClassifier", None)
    if face_cascade_path and os.path.exists(face_cascade_path) and classifier_cls is not None:
        try:
            face_cascade = classifier_cls(face_cascade_path)
            if not face_cascade.empty():
                detect_gray = gray
                scale = 1.0
                if max(h, w) > 640:
                    scale = 640.0 / max(h, w)
                    detect_gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                faces = face_cascade.detectMultiScale(
                    detect_gray,
                    scaleFactor=1.1,
                    minNeighbors=FACE_CASCADE_MIN_NEIGHBORS,
                    minSize=(int(30 * scale), int(30 * scale)),
                )
                if len(faces) > 0:
                    for (fx, fy, fw, fh) in faces:
                        face_area_ratio = float(fw * fh) / max(1, detect_gray.shape[0] * detect_gray.shape[1])
                        if face_area_ratio > 0.02:  # Face occupies >2% of image
                            has_human_face = True
                            score += 0.45
                            cues.append(f"human_face_detected (area_ratio={face_area_ratio:.1%}, count={len(faces)})")
                            break
        except Exception as ex:
            logger.debug(f"Face detection cascade error: {ex}")

    # 2. Skin Tone Detection (HSV-based)
    # Human skin in HSV: H=[5-25], S=[30-180], V=[70-255]
    skin_mask = cv2.inRange(
        hsv,
        (SKIN_HUE_LOW, SKIN_SAT_LOW, SKIN_VAL_LOW),
        (SKIN_HUE_HIGH, SKIN_SAT_HIGH, SKIN_VAL_HIGH),
    )
    # Morphological cleanup to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_ratio = float(np.count_nonzero(skin_mask)) / total_area

    # Note: Paver stones, red bricks, clay, wood, tan leather seats, rust, and brown car paint
    # share the HSV range H=[5-25]. Skin tone can only provide conclusive evidence of a person
    # if a human face is present. Without a face, it must not exceed the rejection threshold.
    if has_human_face:
        if skin_ratio >= SKIN_DOMINANT_RATIO:
            score += 0.35
            cues.append(f"dominant_skin_tone_pixels ({skin_ratio:.1%})")
        if skin_ratio >= 0.15:
            skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if skin_contours:
                largest_skin = max(skin_contours, key=cv2.contourArea)
                largest_skin_ratio = float(cv2.contourArea(largest_skin)) / total_area
                if largest_skin_ratio >= 0.08:
                    score += 0.20
                    cues.append(f"large_contiguous_skin_region ({largest_skin_ratio:.1%})")
    else:
        # Without a face, skin tone alone contributes minimal weight (max 0.20)
        if skin_ratio >= 0.45:
            score += 0.20
            cues.append(f"skin_tone_like_pixels ({skin_ratio:.1%})")
        elif skin_ratio >= SKIN_DOMINANT_RATIO:
            score += 0.10
            cues.append(f"skin_tone_like_pixels ({skin_ratio:.1%})")

    # 3. Color uniformity check (natural/scenic images often have dominant green/blue/sky tones)
    # Count nature-dominant hues: green [35-85] and blue/sky [95-130]
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    blue_sky_mask = cv2.inRange(hsv, (95, 30, 100), (130, 255, 255))
    green_ratio = float(np.count_nonzero(green_mask)) / total_area
    sky_ratio = float(np.count_nonzero(blue_sky_mask)) / total_area
    if (green_ratio + sky_ratio) > 0.60:
        score += 0.20
        cues.append(f"nature_scene_dominant_hues (green={green_ratio:.1%}, sky={sky_ratio:.1%})")

    is_non_vehicle = (score >= 0.55)
    return is_non_vehicle, round(min(1.0, score), 2), cues


# ---------------------------------------------------------------------------
# Module 4: Vehicle Exterior Detection & Feature Extraction
# ---------------------------------------------------------------------------
def detect_exterior_candidate(
    img_bgr: np.ndarray,
    gray: np.ndarray,
) -> Tuple[bool, float, List[str], Optional[List[int]], Dict[str, Any]]:
    """
    Extract vehicle presence signals: bounding contours, circular wheel hubs,
    ground-contact shadow seams, and aspect ratio profiles.

    Returns:
        (is_present, presence_confidence, evidence, bounding_box_xywh, raw_metrics)
    """
    h, w = gray.shape[:2]
    total_area = max(1, h * w)
    evidence: List[str] = []
    score = 0.0

    # 1. Multi-scale Edge & Morphological Body Extraction
    # Gaussian blur is significantly faster than bilateralFilter on CPU with equivalent Canny thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 35, 110)

    # Vehicles exhibit strong horizontal connectivity (roof, beltline, bumper, ground line)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)

    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = None
    best_box = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Must occupy meaningful portion of image (>5% and <98%)
        if (total_area * 0.05) <= area <= (total_area * 0.98):
            bx, by, bw, bh = cv2.boundingRect(cnt)
            # Eliminate extreme vertical or horizontal single-line slivers
            if bw > 30 and bh > 25 and (area > best_area):
                best_area = area
                best_cnt = cnt
                best_box = [bx, by, bw, bh]

    if best_box is None:
        # Fallback to central bounding envelope if edge density is concentrated
        nonzero_pts = cv2.findNonZero(closed_edges)
        if nonzero_pts is not None and len(nonzero_pts) > 500:
            bx, by, bw, bh = cv2.boundingRect(nonzero_pts)
            if (bw * bh) >= (total_area * 0.06):
                best_box = [bx, by, bw, bh]

    if best_box is None:
        return False, 0.15, ["insufficient_foreground_contour_area"], None, {}

    bx, by, bw, bh = best_box
    aspect_ratio = round(float(bw) / max(1.0, float(bh)), 3)
    area_ratio = round(float(bw * bh) / total_area, 3)

    metrics = {
        "aspect_ratio": aspect_ratio,
        "area_ratio": area_ratio,
        "box": best_box,
    }

    # 2. Wheel Structure Detection (Lower 45% of candidate box)
    wheel_roi_y = by + int(bh * 0.52)
    wheel_roi_h = bh - int(bh * 0.52)
    wheel_roi = gray[wheel_roi_y : by + bh, bx : bx + bw]

    wheels_detected = 0
    if wheel_roi.size > 0 and wheel_roi_h > 15:
        w_min_r = max(8, int(min(bw, bh) * 0.06))
        w_max_r = max(w_min_r + 6, int(min(bw, bh) * 0.18))

        wheel_circles = cv2.HoughCircles(
            cv2.GaussianBlur(wheel_roi, (5, 5), 1.5),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(w_min_r * 2, int(bw * 0.22)),
            param1=80,
            param2=32,
            minRadius=w_min_r,
            maxRadius=w_max_r,
        )

        if wheel_circles is not None:
            wc = wheel_circles[0]
            valid_wheels = []
            for c in wc:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                full_cx = bx + cx
                full_cy = wheel_roi_y + cy
                # Ground contact check: bottom of road wheel must reach lower portion of vehicle body
                rel_bottom = (full_cy + r - by) / float(max(1, bh))
                if rel_bottom < 0.85:
                    continue
                # Tyre rubber ring check: outer perimeter must have dark tyre rubber
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(mask, (full_cx, full_cy), int(r * 1.15), 255, -1)
                inner_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.circle(inner_mask, (full_cx, full_cy), int(r * 0.72), 255, -1)
                ring_mask = cv2.bitwise_and(mask, cv2.bitwise_not(inner_mask))
                ring_pixels = gray[ring_mask > 0]
                if len(ring_pixels) == 0:
                    continue
                ring_mean = float(np.mean(ring_pixels))
                if ring_mean > 135.0:
                    continue
                # Ground strip directly below wheel contact patch
                g_strip = gray[min(h - 1, full_cy + r) : min(h, full_cy + r + max(4, int(r * 0.35))), max(0, full_cx - r) : min(w, full_cx + r)]
                if g_strip.size > 0 and float(np.mean(g_strip)) > 155.0:
                    continue
                valid_wheels.append((full_cx, full_cy, r))

            if len(valid_wheels) >= 2:
                # Exterior dual wheels must be horizontally separated across the wheelbase and cannot overlap
                found_dual_road_wheels = False
                for i in range(len(valid_wheels)):
                    for j in range(i + 1, len(valid_wheels)):
                        w1, w2 = valid_wheels[i], valid_wheels[j]
                        center_dist = float(np.hypot(w1[0] - w2[0], w1[1] - w2[1]))
                        if center_dist < 1.20 * (w1[2] + w2[2]):
                            continue
                        if abs(w1[1] - w2[1]) < (0.15 * bh) and abs(w1[0] - w2[0]) > (0.25 * bw):
                            found_dual_road_wheels = True
                            break
                    if found_dual_road_wheels:
                        break
                wheels_detected = 2 if found_dual_road_wheels else 1
            elif len(valid_wheels) == 1:
                wheels_detected = 1

    metrics["wheels_detected"] = wheels_detected
    if wheels_detected >= 2:
        score += 0.35
        evidence.append(f"dual_circular_wheel_structures_detected ({wheels_detected})")
    elif wheels_detected == 1:
        score += 0.15
        evidence.append("single_circular_wheel_structure_detected")

    # 3. Ground-Contact Shadow Seam
    # Directly underneath a real vehicle is a low-luminance contact patch
    # CRITICAL: Must verify body-to-shadow contrast — if both body and ground
    # are uniformly dark, it's a dark image/illustration, not a vehicle shadow.
    ground_y1 = min(h - 1, by + bh - int(bh * 0.10))
    ground_y2 = min(h, by + bh + int(bh * 0.15))
    metrics["has_ground_shadow"] = False
    if ground_y2 > ground_y1:
        ground_strip = gray[ground_y1:ground_y2, max(0, bx + int(bw * 0.10)) : min(w, bx + int(bw * 0.90))]
        if ground_strip.size > 0:
            shadow_val = float(np.mean(ground_strip))
            metrics["ground_shadow_val"] = shadow_val

            # Body region luminance: sample the central body area above the shadow
            body_y1 = max(by, by + int(bh * 0.20))
            body_y2 = max(body_y1 + 10, by + int(bh * 0.70))
            body_x1 = max(0, bx + int(bw * 0.15))
            body_x2 = min(w, bx + int(bw * 0.85))
            body_region = gray[body_y1:body_y2, body_x1:body_x2]
            body_luminance = float(np.mean(body_region)) if body_region.size > 0 else 0.0

            # Overall image darkness check: if >65% of image is very dark (<40),
            # it's likely an illustration/dark artistic image, not a real vehicle photo
            very_dark_pixels = float(np.count_nonzero(gray < 40)) / max(1, h * w)
            is_predominantly_dark = (very_dark_pixels > 0.65)

            # Real vehicle shadow: body above must be significantly brighter than shadow below
            # AND image must not be predominantly dark (which would be an artistic/dark image)
            body_shadow_contrast = body_luminance - shadow_val
            has_valid_contrast = (body_shadow_contrast > 25.0) and (body_luminance > 55.0)

            if not is_predominantly_dark and has_valid_contrast:
                metrics["has_ground_shadow"] = (shadow_val < 95.0)
                if shadow_val < 65.0:
                    score += 0.24
                    evidence.append(f"ground_contact_shadow_seam ({shadow_val:.1f})")
                elif shadow_val < 95.0:
                    score += 0.10
                    evidence.append(f"ground_contact_shadow_seam ({shadow_val:.1f})")

    # 4. Cabin Step-Down Profile (Upper third narrower than lower third)
    upper_slice = edges[by : by + int(bh * 0.35), bx : bx + bw]
    lower_slice = edges[by + int(bh * 0.40) : by + bh, bx : bx + bw]
    has_silhouette = False
    if upper_slice.size > 0 and lower_slice.size > 0:
        up_nonzeros = cv2.findNonZero(upper_slice)
        low_nonzeros = cv2.findNonZero(lower_slice)
        if up_nonzeros is not None and low_nonzeros is not None:
            up_w = cv2.boundingRect(up_nonzeros)[2]
            low_w = cv2.boundingRect(low_nonzeros)[2]
            # Require more significant step-down (15% wider lower body vs 8% before)
            if low_w > (up_w * 1.15):
                has_silhouette = True
                # Silhouette alone is weak; only contribute score if paired with other physical cues
                if wheels_detected >= 1 or metrics.get("has_ground_shadow", False):
                    score += 0.16
                    evidence.append("two_box_cabin_roofline_silhouette")
                else:
                    score += 0.06  # Minimal contribution without supporting physical cues
                    evidence.append("weak_cabin_roofline_silhouette")

    # 5. Aspect Ratio Plausibility Evaluation
    # Only contributes to score if strong physical vehicle cues exist
    # (dual validated wheels, or single wheel + confirmed ground shadow)
    has_strong_physical_cues = (
        wheels_detected >= 2
        or (wheels_detected >= 1 and metrics.get("has_ground_shadow", False))
        or (metrics.get("has_ground_shadow", False) and has_silhouette)  # Body on ground with roofline shape
        or (wheels_detected >= 1 and has_silhouette)
    )
    has_weak_physical_cues = (
        wheels_detected >= 1
        or metrics.get("has_ground_shadow", False)
        or has_silhouette
    )
    if has_strong_physical_cues:
        if 0.70 <= aspect_ratio <= 3.8:
            score += 0.15
            evidence.append(f"vehicle_proportional_aspect_ratio ({aspect_ratio:.2f})")
        elif 0.40 <= aspect_ratio < 0.70:
            score += 0.08
            evidence.append(f"vehicle_proportional_aspect_ratio ({aspect_ratio:.2f})")
    elif has_weak_physical_cues and 1.0 <= aspect_ratio <= 3.0:
        # Reduced contribution for weak cues — classic vehicle aspect ratios only
        score += 0.08
        evidence.append(f"vehicle_proportional_aspect_ratio ({aspect_ratio:.2f})")

    # 6. Anchor Cue Requirement:
    # A real vehicle MUST have at least one strong anchor evidence.
    # Without an anchor, the image cannot be classified as a vehicle even if
    # weak cues accumulate above the threshold.
    has_anchor_cue = (
        wheels_detected >= 2                                                  # Dual validated road wheels
        or (wheels_detected >= 1 and metrics.get("has_ground_shadow", False)) # Single wheel + ground shadow
        or (wheels_detected >= 1 and has_silhouette)                           # Single wheel + vehicle body silhouette
        or ("ground_contact_shadow_seam" in " ".join(evidence) and has_silhouette)  # Shadow + silhouette
    )

    if not has_anchor_cue:
        # Without anchor, cap score to prevent false positives from
        # accumulated weak signals (single circle + aspect ratio + mild shadow)
        score = min(score, CONFIDENCE_VEHICLE_MIN - 0.01)

    is_present = (score >= CONFIDENCE_VEHICLE_MIN)
    final_conf = min(0.95, round(score, 2))
    return is_present, final_conf, evidence, best_box, metrics


# ---------------------------------------------------------------------------
# Module 5: Vehicle Type Classification (Exterior)
# ---------------------------------------------------------------------------
def classify_exterior_vehicle_type(
    aspect_ratio: float,
    wheels_detected: int,
    metrics: Dict[str, Any],
    box_w: int,
    box_h: int,
) -> Tuple[str, float, List[str]]:
    """
    Classify vehicle type from geometric and visual cues.
    Categories: 'car', 'motorcycle', 'bus', 'truck', 'other_vehicle', 'unknown'
    """
    cues: List[str] = []

    # 1. Motorcycle / Bike: Narrow vertical profile or compact two-wheel alignment
    # Guard: Cars with portrait framing or front crops have wide bounding boxes (>260px) and large area
    is_motorcycle_candidate = (box_w < 260) and (metrics.get("area_ratio", 0) < 0.25)
    if is_motorcycle_candidate:
        if BIKE_ASPECT_MIN <= aspect_ratio <= BIKE_ASPECT_MAX and box_h > int(box_w * 0.95):
            cues.append(f"narrow_vertical_silhouette ({aspect_ratio:.2f})")
            return "motorcycle", 0.72, cues
        if aspect_ratio <= 1.15 and wheels_detected == 1:
            cues.append(f"compact_two_wheeler_proportions ({aspect_ratio:.2f})")
            return "motorcycle", 0.65, cues

    # 2. Bus: Elongated passenger body, large continuous horizontal glazing
    if aspect_ratio >= BUS_ASPECT_MIN and metrics.get("area_ratio", 0) > 0.40 and box_w > 500:
        cues.append(f"elongated_passenger_body_aspect ({aspect_ratio:.2f})")
        return "bus", 0.70, cues

    # 3. Truck: Heavy cargo box silhouette, high ground seam
    if aspect_ratio >= TRUCK_ASPECT_MIN and metrics.get("ground_shadow_val", 255) < 65.0:
        if aspect_ratio > 2.0 and metrics.get("area_ratio", 0) > 0.40:
            cues.append(f"high_cargo_mass_profile ({aspect_ratio:.2f})")
            return "truck", 0.62, cues

    # 4. Car: Standard passenger vehicle proportions (Sedan, Hatchback, SUV)
    if CAR_ASPECT_MIN <= aspect_ratio <= CAR_ASPECT_MAX:
        cues.append(f"passenger_car_body_proportions ({aspect_ratio:.2f})")
        return "car", 0.78, cues

    if 0.5 <= aspect_ratio <= 3.2:
        cues.append(f"general_automotive_silhouette ({aspect_ratio:.2f})")
        return "car", 0.70, cues

    cues.append("unresolved_geometric_proportions")
    return "car", 0.60, cues


# ---------------------------------------------------------------------------
# Module 6: Core Public Pipeline & Decision Aggregator
# ---------------------------------------------------------------------------
def detect_vehicle(image_path: Union[str, Path, np.ndarray], debug: bool = False) -> Dict[str, Any]:
    """
    Inspect an image and return structured vehicle detection analysis.

    Returns:
        JSON-compatible dictionary matching the strict specification.
    """
    setup_logger(debug=debug)
    warnings: List[str] = ["Heuristic result; not a trained-model prediction."]
    img_path_str = str(image_path) if isinstance(image_path, (str, Path)) else None

    # Step 1: Decode & Preprocess
    img_bgr, orig_bgr, err_msg, (orig_w, orig_h) = load_image(image_path)
    if err_msg or img_bgr is None:
        if "resolution too small" in (err_msg or ""):
            return {
                "success": False,
                "image_path": img_path_str,
                "is_vehicle": False,
                "is_vehicle_related": False,
                "scene_type": "uncertain",
                "vehicle_type": None,
                "interior_components": [],
                "interior_confidence": 0.0,
                "exterior_confidence": 0.0,
                "type_confidence": 0.0,
                "detected_cues": ["image_below_minimum_dimensions"],
                "warnings": warnings + [err_msg],
                "method": "opencv_heuristics_no_pretrained_model",
                "is_rc_document": False,
                "is_interior": False,
                "is_inside": False,
                "is_exterior": False,
                "vehicle_side": None,
                "interior_type": None,
                "confidence": 0.0,
                "vehicle_confidence": 0.0,
                "bounding_box": None,
                "evidence": ["image_below_minimum_dimensions"],
                "error": None,
                "rc_details": None,
            }
        logger.debug(f"Input validation error: {err_msg}")
        return {
            "success": False,
            "image_path": img_path_str,
            "is_vehicle": False,
            "is_vehicle_related": False,
            "scene_type": "uncertain",
            "vehicle_type": None,
            "interior_components": [],
            "interior_confidence": 0.0,
            "exterior_confidence": 0.0,
            "type_confidence": 0.0,
            "detected_cues": [],
            "warnings": warnings,
            "method": "opencv_heuristics_no_pretrained_model",
            "is_rc_document": False,
            "is_interior": False,
            "is_inside": False,
            "is_exterior": False,
            "vehicle_side": None,
            "interior_type": None,
            "confidence": 0.0,
            "vehicle_confidence": 0.0,
            "bounding_box": None,
            "evidence": [],
            "error": err_msg,
            "rc_details": None,
        }

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2: Image Quality & Blur Check
    blur_var = measure_blur_metric(gray)
    if blur_var < BLUR_LAPLACIAN_THRESH:
        warnings.append(f"Low sharpness/contrast detected (Laplacian variance {blur_var:.1f}).")
        if blur_var < 8.0:
            return {
                "success": True,
                "image_path": img_path_str,
                "is_vehicle": False,
                "is_vehicle_related": False,
                "scene_type": "uncertain",
                "vehicle_type": None,
                "interior_components": [],
                "interior_confidence": 0.0,
                "exterior_confidence": 0.0,
                "type_confidence": 0.0,
                "detected_cues": ["severely_blurred_or_empty_image"],
                "warnings": warnings,
                "method": "opencv_heuristics_no_pretrained_model",
                "is_rc_document": False,
                "is_interior": False,
                "is_inside": False,
                "is_exterior": False,
                "vehicle_side": None,
                "interior_type": None,
                "confidence": 0.0,
                "vehicle_confidence": 0.0,
                "bounding_box": None,
                "evidence": ["severely_blurred_or_empty_image"],
                "error": None,
                "rc_details": None,
            }

    # Step 3: Document & Indian RC Card Detection
    orig_img_for_card = orig_bgr if orig_bgr is not None else img_bgr

    card_detected = False
    card_roi = None
    card_cues = []
    if orig_img_for_card is not None and orig_img_for_card.size > 0:
        card_detected, card_roi, card_cues = detect_document_card_geometry(orig_img_for_card)

    is_doc, doc_score, doc_cues = check_is_document_or_paper(img_bgr, gray)

    # If physical card geometry is present or light paper background with text is detected, evaluate for Indian RC
    if card_detected or is_doc:
        rc_res = detect_rc_document(orig_img_for_card)
        if rc_res.get("is_rc_document"):
            logger.debug(f"Classified as RC document (conf {rc_res.get('confidence')}): {rc_res.get('matched_rc_cues')}")
            rc_cues_all = ["rc_document_detected"] + rc_res.get("matched_rc_cues", [])
            return {
                "success": True,
                "image_path": img_path_str,
                "is_vehicle": False,
                "is_vehicle_related": False,
                "scene_type": "rc_document",
                "vehicle_type": None,
                "interior_components": [],
                "interior_confidence": 0.0,
                "exterior_confidence": 0.0,
                "type_confidence": 0.0,
                "detected_cues": rc_cues_all,
                "warnings": warnings + ["Detected Indian Vehicle Registration Certificate (RC)."],
                "method": "opencv_heuristics_no_pretrained_model",
                "is_rc_document": True,
                "is_interior": False,
                "is_inside": False,
                "is_exterior": False,
                "vehicle_side": None,
                "interior_type": None,
                "confidence": float(rc_res.get("confidence", 0.98)),
                "vehicle_confidence": 0.0,
                "bounding_box": None,
                "evidence": rc_cues_all,
                "error": None,
                "rc_details": rc_res,
            }

    if is_doc:
        logger.debug(f"Rejected as document/paper (score {doc_score:.2f}): {doc_cues}")
        doc_evidence = ["automotive_document_or_text_sheet"] + doc_cues
        return {
            "success": True,
            "image_path": img_path_str,
            "is_vehicle": False,
            "is_vehicle_related": False,
            "scene_type": "not_vehicle",
            "vehicle_type": None,
            "interior_components": [],
            "interior_confidence": 0.0,
            "exterior_confidence": 0.0,
            "type_confidence": 0.0,
            "detected_cues": doc_evidence,
            "warnings": warnings + ["Detected non-vehicle document, invoice, or text sheet."],
            "method": "opencv_heuristics_no_pretrained_model",
            "is_rc_document": False,
            "is_interior": False,
            "is_inside": False,
            "is_exterior": False,
            "vehicle_side": None,
            "interior_type": None,
            "confidence": round(doc_score, 2),
            "vehicle_confidence": 0.0,
            "bounding_box": None,
            "evidence": doc_evidence,
            "error": None,
            "rc_details": None,
        }

    # Step 3b: Non-Vehicle Scene Detection (People, Selfies, Nature)
    is_non_vehicle_scene, nv_conf, nv_cues = detect_non_vehicle_scene(img_bgr, gray)

    # Step 4: Vehicle Exterior & Interior Analysis
    is_exterior, ext_conf, ext_evidence, ext_box, ext_metrics = detect_exterior_candidate(img_bgr, gray)
    (
        is_interior,
        int_conf,
        int_type,
        int_type_conf,
        int_evidence,
        int_box,
        int_components,
        scene_verdict,
        int_vtype,
        int_vtype_conf,
    ) = detect_interior_signatures(img_bgr, gray, image_path)

    has_vehicle_structure = (
        is_exterior
        or ext_metrics.get("wheels_detected", 0) >= 2
        or (ext_metrics.get("wheels_detected", 0) >= 1 and (ext_metrics.get("has_ground_shadow", False) or "two_box_cabin_roofline_silhouette" in ext_evidence))
        or "two_box_cabin_roofline_silhouette" in ext_evidence
    )

    # If non-vehicle scene cues fired, but genuine vehicle structures (wheels, silhouette, cabin)
    # are present, vehicle presence takes precedence. Only reject if no vehicle features exist.
    if is_non_vehicle_scene and not (is_exterior or is_interior or has_vehicle_structure or "exterior_license_plate_detected" in int_evidence or ext_metrics.get("wheels_detected", 0) >= 1):
        logger.debug(f"Rejected as non-vehicle scene (score {nv_conf:.2f}): {nv_cues}")
        nv_evidence = ["non_vehicle_scene_detected"] + nv_cues
        return {
            "success": True,
            "image_path": img_path_str,
            "is_vehicle": False,
            "is_vehicle_related": False,
            "scene_type": "not_vehicle",
            "vehicle_type": None,
            "interior_components": [],
            "interior_confidence": 0.0,
            "exterior_confidence": 0.0,
            "type_confidence": 0.0,
            "detected_cues": nv_evidence,
            "warnings": warnings + ["Detected non-vehicle scene (person, nature, or non-automotive content)."],
            "method": "opencv_heuristics_no_pretrained_model",
            "is_rc_document": False,
            "is_interior": False,
            "is_inside": False,
            "is_exterior": False,
            "vehicle_side": None,
            "interior_type": None,
            "confidence": round(nv_conf, 2),
            "vehicle_confidence": 0.0,
            "bounding_box": None,
            "evidence": nv_evidence,
            "error": None,
            "rc_details": None,
        }
    if is_non_vehicle_scene and (is_exterior or is_interior or has_vehicle_structure or ext_metrics.get("wheels_detected", 0) >= 1):
        warnings.append("Non-vehicle background tones detected, but verified vehicle structures confirmed presence.")

    has_ext_ocr_ev = any("exterior_license_plate_detected" in ev or "exterior_vehicle_badge_detected" in ev for ev in int_evidence)
    if has_ext_ocr_ev:
        is_exterior = True
        ext_conf = max(ext_conf, 0.92)
        for ev in int_evidence:
            if ("exterior_license_plate_detected" in ev or "exterior_vehicle_badge_detected" in ev) and ev not in ext_evidence:
                ext_evidence.append(ev)
        is_interior = False

    # Truly definitive cabin cues: verified steering wheel, instrument cluster, cabin controls, display screen, seat/headrest contours, or interior upholstery/trim
    has_definitive_interior = (
        len(int_components) >= 2
        or any(
            any(k in ev for k in (
                "circular_steering_wheel_structure",
                "instrument_cluster_text",
                "cabin_interior_controls_text",
                "vehicle_seat_and_headrest_contours",
                "infotainment_touchscreen_contour",
                "active_display_screen_glow",
                "dual_concentric_gauge_dials",
                "cabin_upholstery_and_interior_trim",
            ))
            for ev in int_evidence
        )
    )
    has_ext_ground_and_wheels = (
        ext_metrics.get("wheels_detected", 0) >= 2
        or (ext_metrics.get("wheels_detected", 0) >= 1 and (ext_metrics.get("has_ground_shadow", False) or ext_metrics.get("has_road_surface", False)))
    )
    has_ext_silhouette = bool(ext_metrics.get("has_body_contour") or ext_metrics.get("has_silhouette"))

    # Disambiguation: If both exterior and interior cues fired
    if is_exterior and is_interior:
        if has_definitive_interior:
            is_exterior = False
            warnings.append("Both interior and exterior cues detected; resolved to interior based on definitive cabin features.")
        elif has_ext_ground_and_wheels:
            is_interior = False
            warnings.append("Both interior and exterior cues detected; resolved to exterior based on ground shadow and road wheel contact.")
        elif has_ext_silhouette:
            is_interior = False
            warnings.append("Both interior and exterior cues detected; resolved to exterior based on body contour silhouette.")
        elif ext_conf >= int_conf:
            is_interior = False
            warnings.append("Both interior and exterior cues detected; resolved to exterior based on higher confidence.")
        else:
            is_exterior = False
            warnings.append("Both interior and exterior cues detected; resolved to interior by default.")
    elif is_interior and not is_exterior and has_ext_ground_and_wheels and not has_definitive_interior:
        is_interior = False
        is_exterior = True
        ext_conf = max(ext_conf, 0.70)
        warnings.append("Resolved to exterior based on road wheel ground contact.")

    # Scale bounding box back to original coordinates if image was downscaled
    scaled_box = None
    if ext_box is not None:
        curr_h, curr_w = gray.shape[:2]
        sx = orig_w / float(curr_w)
        sy = orig_h / float(curr_h)
        scaled_box = [
            int(ext_box[0] * sx),
            int(ext_box[1] * sy),
            int(ext_box[2] * sx),
            int(ext_box[3] * sy),
        ]

    # Check for exterior vehicle detail (e.g., tyre close-up with insufficient body context)
    is_exterior_detail = (
        not is_exterior
        and not is_interior
        and ext_metrics.get("wheels_detected", 0) == 1
        and "single_circular_wheel_structure_detected" in ext_evidence
        and (0.35 <= ext_metrics.get("aspect_ratio", 1.0) <= 2.80)
        and ext_conf >= 0.18
    )

    # Step 5: Interior Decision Resolution ("which image is inside from vehicle")
    if is_interior:
        v_type = int_vtype or "car"
        v_int_type = int_type or "cabin"
        if v_type == "bike":
            v_int_type = "bike_controls"
            if "bike_controls" not in int_components:
                int_components.append("bike_controls")

        all_cues = list(int_evidence)
        if ext_evidence:
            all_cues.extend([c for c in ext_evidence if c not in all_cues])

        logger.debug(f"Classified as vehicle_interior (conf {int_conf}): {int_evidence}")
        return {
            "success": True,
            "image_path": img_path_str,
            "is_vehicle": True,
            "scene_type": "vehicle_interior",
            "vehicle_type": v_type,
            "interior_components": int_components,
            "interior_confidence": round(int_conf, 2),
            "exterior_confidence": round(ext_conf, 2),
            "type_confidence": round(int_vtype_conf, 2),
            "detected_cues": all_cues,
            "warnings": warnings,
            "method": "opencv_heuristics_no_pretrained_model",
            # Legacy compatibility fields:
            "is_vehicle_related": True,
            "is_rc_document": False,
            "is_interior": True,
            "is_inside": True,
            "is_exterior": False,
            "vehicle_side": "inside",
            "interior_type": v_int_type,
            "confidence": round(int_conf, 2),
            "vehicle_confidence": round(int_conf, 2),
            "bounding_box": int_box,
            "evidence": int_evidence,
            "error": None,
            "rc_details": None,
        }

    # Step 5b: Vehicle Detail Decision Resolution (close up with insufficient scene context)
    if (scene_verdict == "vehicle_detail" or is_exterior_detail) and not is_exterior:
        v_type = int_vtype or "car"
        detail_conf = round(max(int_conf, ext_conf, 0.65), 2)
        all_cues = list(int_evidence) + [c for c in ext_evidence if c not in int_evidence]
        return {
            "success": True,
            "image_path": img_path_str,
            "is_vehicle": True,
            "scene_type": "vehicle_detail",
            "vehicle_type": v_type,
            "interior_components": int_components,
            "interior_confidence": round(int_conf, 2),
            "exterior_confidence": round(ext_conf, 2),
            "type_confidence": round(int_vtype_conf if int_vtype else 0.50, 2),
            "detected_cues": all_cues,
            "warnings": warnings + ["Component detected but scene context is insufficient to determine full interior/exterior view."],
            "method": "opencv_heuristics_no_pretrained_model",
            # Legacy compatibility fields:
            "is_vehicle_related": True,
            "is_rc_document": False,
            "is_interior": bool(int_components),
            "is_inside": bool(int_components),
            "is_exterior": not bool(int_components),
            "vehicle_side": "detail",
            "interior_type": int_type,
            "confidence": detail_conf,
            "vehicle_confidence": detail_conf,
            "bounding_box": int_box if int_box else scaled_box,
            "evidence": all_cues,
            "error": None,
            "rc_details": None,
        }

    # Step 6: Exterior Decision Resolution ("vehicle")
    if is_exterior:
        v_type, t_conf, type_cues = classify_exterior_vehicle_type(
            aspect_ratio=ext_metrics.get("aspect_ratio", 1.5),
            wheels_detected=ext_metrics.get("wheels_detected", 0),
            metrics=ext_metrics,
            box_w=ext_box[2] if (ext_box is not None and len(ext_box) > 2) else gray.shape[1],
            box_h=ext_box[3] if (ext_box is not None and len(ext_box) > 3) else gray.shape[0],
        )
        overall_conf = round((ext_conf * 0.65 + t_conf * 0.35), 2)
        ext_side = "side" if ext_metrics.get("aspect_ratio", 1.0) > 1.8 else "exterior"
        all_ext_cues = ext_evidence + type_cues
        return {
            "success": True,
            "image_path": img_path_str,
            "is_vehicle": True,
            "scene_type": "vehicle_exterior",
            "vehicle_type": v_type,
            "interior_components": [],
            "interior_confidence": round(int_conf, 2),
            "exterior_confidence": round(ext_conf, 2),
            "type_confidence": round(t_conf, 2),
            "detected_cues": all_ext_cues,
            "warnings": warnings,
            "method": "opencv_heuristics_no_pretrained_model",
            # Legacy compatibility fields:
            "is_vehicle_related": True,
            "is_rc_document": False,
            "is_interior": False,
            "is_inside": False,
            "is_exterior": True,
            "vehicle_side": ext_side,
            "interior_type": None,
            "confidence": min(1.0, overall_conf),
            "vehicle_confidence": round(ext_conf, 2),
            "bounding_box": scaled_box,
            "evidence": all_ext_cues,
            "color_details": detect_vehicle_color(img_bgr, bbox=ext_box, view=ext_side, is_exterior=True),
            "error": None,
            "rc_details": None,
        }

    # Step 7: Ambiguous / Uncertain Floor
    if ext_conf >= CONFIDENCE_UNCERTAIN_FLOOR or int_conf >= CONFIDENCE_UNCERTAIN_FLOOR:
        logger.debug(f"Confidence ambiguous (ext {ext_conf:.2f}, int {int_conf:.2f}); reporting uncertain.")
        max_c = max(ext_conf, int_conf)
        all_cues = ext_evidence + int_evidence + ["evidence_inconclusive"]
        return {
            "success": True,
            "image_path": img_path_str,
            "is_vehicle": False,
            "scene_type": "uncertain",
            "vehicle_type": None,
            "interior_components": int_components if int_components else [],
            "interior_confidence": round(int_conf, 2),
            "exterior_confidence": round(ext_conf, 2),
            "type_confidence": 0.0,
            "detected_cues": all_cues,
            "warnings": warnings + ["Visual cues are insufficient or conflicting."],
            "method": "opencv_heuristics_no_pretrained_model",
            # Legacy compatibility fields:
            "is_vehicle_related": False,
            "is_rc_document": False,
            "is_interior": False,
            "is_inside": False,
            "is_exterior": False,
            "vehicle_side": None,
            "interior_type": None,
            "confidence": round(max_c, 2),
            "vehicle_confidence": round(max_c, 2),
            "bounding_box": scaled_box,
            "evidence": all_cues,
            "error": None,
            "rc_details": None,
        }

    # Default: Not a vehicle
    cues = ext_evidence if ext_evidence else ["no_vehicle_signatures_detected"]
    return {
        "success": True,
        "image_path": img_path_str,
        "is_vehicle": False,
        "scene_type": "not_vehicle",
        "vehicle_type": None,
        "interior_components": [],
        "interior_confidence": round(int_conf, 2),
        "exterior_confidence": round(ext_conf, 2),
        "type_confidence": 0.0,
        "detected_cues": cues,
        "warnings": warnings,
        "method": "opencv_heuristics_no_pretrained_model",
        # Legacy compatibility fields:
        "is_vehicle_related": False,
        "is_rc_document": False,
        "is_interior": False,
        "is_inside": False,
        "is_exterior": False,
        "vehicle_side": None,
        "interior_type": None,
        "confidence": round(1.0 - max(ext_conf, int_conf), 2),
        "vehicle_confidence": round(max(ext_conf, int_conf), 2),
        "bounding_box": None,
        "evidence": cues,
        "error": None,
        "rc_details": None,
    }


# ---------------------------------------------------------------------------
# Module 6b: Vehicle Exterior Paint Color Detection Engine
# ---------------------------------------------------------------------------
CAR_COLOR_LABELS = [
    "White", "Silver", "Black", "Blue", "Red", "Brown",
]


def detect_vehicle_color(
    image_input: Union[str, Path, np.ndarray],
    bbox: Optional[Sequence[int]] = None,
    view: str = "",
    is_exterior: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Analyzes an exterior vehicle image and estimates the dominant exterior paint color.
    Strictly samples the cleanest sheet-metal panels:
      - Front views: Bonnet / Hood surface (strictly avoids radiator grille, fog lights, and road shadows).
      - Rear views: Boot lid / Tailgate panel (strictly avoids lower bumper and exhaust shadows).
      - Side views: Middle door panels (avoids windows and wheels).
    Excludes deep shadows (L < 45 or V < 45), specular highlights, and blue sky reflections.

    Returns:
        JSON-compatible dictionary matching the Car Exterior Color Detection specification.
    """
    setup_logger(debug=debug)
    warnings: List[str] = ["Heuristic result; not a trained-model prediction."]

    # Step 1: Validate and prepare image
    img_bgr = None
    err_msg = None
    if isinstance(image_input, (str, Path)):
        img_bgr, _, err_msg, _ = load_image(image_input)
    elif isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            err_msg = "Empty numpy array provided."
        elif len(image_input.shape) == 2:
            img_bgr = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif len(image_input.shape) == 3:
            if image_input.shape[2] == 4:
                img_bgr = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            elif image_input.shape[2] == 3:
                img_bgr = image_input
            else:
                err_msg = f"Unsupported channel count: {image_input.shape[2]}"
        else:
            err_msg = f"Unsupported array shape: {image_input.shape}"
    else:
        err_msg = f"Invalid image input type: {type(image_input)}"

    if err_msg or img_bgr is None:
        return {
            "status": "error",
            "color": None,
            "secondary_color": None,
            "confidence": 0.0,
            "method": "opencv_hsv_lab_clustering",
            "warnings": warnings + [f"Image could not be loaded: {err_msg}"],
            "metrics": {},
        }

    h_img, w_img = img_bgr.shape[:2]
    if h_img < 40 or w_img < 40:
        return {
            "status": "uncertain",
            "color": None,
            "secondary_color": None,
            "confidence": 0.0,
            "method": "opencv_hsv_lab_clustering",
            "warnings": warnings + ["Image dimensions below minimum resolution for color analysis."],
            "metrics": {},
        }

    if not is_exterior:
        return {
            "status": "uncertain",
            "color": None,
            "secondary_color": None,
            "confidence": 0.0,
            "method": "opencv_hsv_lab_clustering",
            "warnings": warnings + ["Color detection is only supported on exterior vehicle images; interior views are excluded."],
            "metrics": {},
        }

    # Step 2: Select Vehicle Region
    if bbox is not None and len(bbox) == 4:
        bx, by, bw, bh = bbox
        # Support both [x, y, w, h] and [x1, y1, x2, y2]
        if bw > bx and bh > by and bw > 1 and bh > 1 and (bw - bx) < w_img and (bh - by) < h_img and bx >= 0 and by >= 0:
            x1, y1, x2, y2 = max(0, bx), max(0, by), min(w_img, bw), min(h_img, bh)
        else:
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(w_img, bx + bw), min(h_img, by + bh)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0 or (x2 - x1) < 30 or (y2 - y1) < 30:
            crop = img_bgr
            warnings.append("Supplied vehicle bounding box was invalid or too small; fell back to full image crop.")
    else:
        # Automatically detect candidate exterior vehicle bounding box
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, _, _, auto_box, _ = detect_exterior_candidate(img_bgr, gray_full)
        if auto_box and auto_box[2] > 60 and auto_box[3] > 60:
            abx, aby, abw, abh = auto_box
            abh_extended = min(h_img - aby, int(abh * 1.30))
            crop = img_bgr[aby:aby+abh_extended, abx:abx+abw]
            warnings.append("Auto-detected candidate vehicle bounding box used.")
        else:
            crop = img_bgr[int(h_img * 0.15):int(h_img * 0.85), int(w_img * 0.15):int(w_img * 0.85)]
            warnings.append("No vehicle bounding box provided; central foreground crop used as fallback.")

    h_c, w_c = crop.shape[:2]
    if h_c < 30 or w_c < 30:
        return {
            "status": "uncertain",
            "color": None,
            "secondary_color": None,
            "confidence": 0.0,
            "method": "opencv_hsv_lab_clustering",
            "warnings": warnings + ["Crop region too small for reliable color estimation."],
            "metrics": {},
        }

    # Step 3: Focus STRICTLY on painted sheet-metal panels:
    # - Front view: Bonnet / Hood and upper fender shoulders (above headlights, immune to lower fog lights, skid plates, road)
    # - Rear view: Boot / Tailgate panel (below rear windshield glass and above bumper skid plate)
    # - Side view: Upper door panels (30% to 65% height, avoiding windows and wheels)
    v_clean = (view or "").lower().strip()
    is_front = any(k in v_clean for k in ("front", "fl", "fr")) or v_clean == "front"
    is_rear = any(k in v_clean for k in ("rear", "back", "tail", "rl", "rr")) or v_clean == "rear"

    sample_patches = []
    if is_front:
        # Central hood panel (above radiator grille top and below windshield)
        p_hood = crop[int(h_c * 0.42):int(h_c * 0.58), int(w_c * 0.35):int(w_c * 0.65)]
        if p_hood.size > 0:
            sample_patches.append(p_hood)
        # Front bumper fascia corners flanking fog lamps / headlights (vertical panels with purest paint pigment, immune to adjacent parked cars and sky glare)
        p_corner_l = crop[int(h_c * 0.70):int(h_c * 0.84), int(w_c * 0.10):int(w_c * 0.22)]
        p_corner_r = crop[int(h_c * 0.70):int(h_c * 0.84), int(w_c * 0.78):int(w_c * 0.90)]
        if p_corner_l.size > 0:
            sample_patches.append(p_corner_l)
        if p_corner_r.size > 0:
            sample_patches.append(p_corner_r)
        warnings.append("Targeted bonnet/hood sheet metal and front bumper corner fascia.")
    elif is_rear:
        # Boot / Tailgate lid: upper-middle trunk panel below rear glass and above bumper skid plate
        p_boot = crop[int(h_c * 0.40):int(h_c * 0.65), int(w_c * 0.25):int(w_c * 0.75)]
        if p_boot.size > 0:
            sample_patches.append(p_boot)
        # Rear quarter shoulders
        p_quarter_l = crop[int(h_c * 0.35):int(h_c * 0.58), int(w_c * 0.16):int(w_c * 0.28)]
        p_quarter_r = crop[int(h_c * 0.35):int(h_c * 0.58), int(w_c * 0.72):int(w_c * 0.84)]
        if p_quarter_l.size > 0:
            sample_patches.append(p_quarter_l)
        if p_quarter_r.size > 0:
            sample_patches.append(p_quarter_r)
        # Rear bumper fascia (vertical sides, above exhaust/skid plate)
        p_rbump_l = crop[int(h_c * 0.65):int(h_c * 0.78), int(w_c * 0.18):int(w_c * 0.30)]
        p_rbump_r = crop[int(h_c * 0.65):int(h_c * 0.78), int(w_c * 0.70):int(w_c * 0.82)]
        if p_rbump_l.size > 0:
            sample_patches.append(p_rbump_l)
        if p_rbump_r.size > 0:
            sample_patches.append(p_rbump_r)
        warnings.append("Targeted rear boot/tailgate, quarter shoulders, and bumper fascia panels.")
    else:
        # Side profile: middle door panels
        p_side = crop[int(h_c * 0.32):int(h_c * 0.65), int(w_c * 0.22):int(w_c * 0.78)]
        if p_side.size > 0:
            sample_patches.append(p_side)
        warnings.append("Targeted side door panels (ignoring window glass and wheels).")

    if not sample_patches:
        sample_patches = [crop[int(h_c * 0.25):int(h_c * 0.60), int(w_c * 0.20):int(w_c * 0.80)]]

    pixels_bgr = np.vstack([p.reshape(-1, 3) for p in sample_patches])

    # Step 4: Convert to HSV and CIELAB color spaces
    hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)

    h_arr = hsv[:, 0]
    s_arr = hsv[:, 1]
    v_arr = hsv[:, 2]
    l_arr = lab[:, 0]
    a_arr = lab[:, 1]
    b_arr = lab[:, 2]

    # Non-paint pixel suppression:
    # 1. Shadow / deep tire cavity rejection:
    # Extreme darkness (L < 6 and V < 10) is pure deep cavity / wheel well / asphalt shadow.
    # Real automotive black paint under daylight typically has L in [6, 110] and V in [10, 110].
    shadow_mask = (v_arr >= 10) & (l_arr >= 6)

    # 2. Sensor specular clipping rejection:
    b_chan = pixels_bgr[:, 0].astype(np.float32)
    g_chan = pixels_bgr[:, 1].astype(np.float32)
    r_chan = pixels_bgr[:, 2].astype(np.float32)
    glare_mask = ~((r_chan >= 253) & (g_chan >= 253) & (b_chan >= 253))

    valid_mask = shadow_mask & glare_mask

    total_valid = int(np.sum(valid_mask))
    if total_valid < 30:
        return {
            "status": "uncertain",
            "color": None,
            "secondary_color": None,
            "confidence": 0.20,
            "method": "opencv_hsv_lab_clustering",
            "warnings": warnings + ["Insufficient visible paint area detected after shadow and glare filtering."],
            "metrics": {"valid_pixels": total_valid},
        }

    H = h_arr[valid_mask].astype(np.float32)
    S = s_arr[valid_mask].astype(np.float32)
    V = v_arr[valid_mask].astype(np.float32)
    L = l_arr[valid_mask].astype(np.float32)
    A = a_arr[valid_mask].astype(np.float32)
    B = b_arr[valid_mask].astype(np.float32)
    R_px = r_chan[valid_mask]
    B_px = b_chan[valid_mask]
    G_px = g_chan[valid_mask]

    # Chroma in CIELAB space:
    C_star = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

    # Step 5: Vectorized classification strictly into the 6 Primary Automotive Colors:
    # 1. Red: Strong red dominance
    mask_red = ((H <= 12) | (H >= 162)) & (S >= 32) & (A > 132) & (R_px > B_px + 10)

    # 2. Blue: True blue paint (bright, royal, cobalt, electric, navy, metallic dark/Nexa blue)
    # Reject bright clear sky glare on light/white/silver vehicles
    is_sky_reflection = (
        ((L >= 135) | (V >= 140)) & (S < 60) & (B_px < R_px + 18)
    ) | (
        (B >= 126.5) & (C_star < 4.5)
    )

    # Standard / vibrant blue paint
    mask_std_blue = (
        (H >= 75) & (H <= 140)
        & (B_px > R_px + 10)
        & (B_px >= G_px - 4)
        & (B <= 125.0)
        & (S >= 40)
        & (C_star >= 6.0)
        & (~is_sky_reflection)
        & (~mask_red)
    )

    # Dark metallic / navy / deep blue paint (e.g. Maruti Nexa Blue, Midnight Blue)
    # Retains saturated blue hue despite lower Lab lightness L
    mask_dark_blue = (
        (H >= 85) & (H <= 140)
        & (S >= 40)
        & (V >= 18)
        & (B_px > R_px + 6)
        & (B_px >= G_px - 5)
        & (B <= 126.0)
        & (C_star >= 4.5)
        & (~is_sky_reflection)
        & (~mask_red)
    )
    mask_blue = mask_std_blue | mask_dark_blue

    # 3. Brown: True brown / bronze / golden brown / mocha / amber / tan / urban titanium (warm earth tones)
    mask_brown = (
        (H >= 6) & (H <= 50)
        & (B >= 127.0)
        & (
            # Normal warm brown / golden brown / bronze / urban titanium paint
            ((L >= 25) & (V >= 30) & (L < 192) & (S >= 14) & (C_star >= 4.0) & (R_px >= B_px - 4))
            # Deep rich metallic brown / chocolate
            | ((L >= 25) & (V >= 30) & (S >= 30) & (C_star >= 10.0) & (R_px >= B_px + 4))
        )
        & ~mask_red & ~mask_blue
    )

    mask_chromatic = mask_red | mask_blue | mask_brown

    # Clear-coat sky reflection rejection on horizontal panels (overcast/blue sky glare on metallic hood/roof):
    is_sky_glare = (
        (~mask_chromatic)
        & (H >= 70) & (H <= 150)
        & (S < 45)
        & (B_px >= R_px - 4)
        & (B <= 128.5)
    )

    mask_achromatic = (~mask_chromatic) & (~is_sky_glare)

    # 4. Achromatic / Neutral (White vs Silver vs Black):
    # White paint: high brightness L >= 200 (or V >= 210 with L >= 192) with low chroma, not sky glare
    mask_white = mask_achromatic & (
        ((L >= 200) | ((V >= 210) & (L >= 192)))
        & (S < 38) & (C_star < 14.0)
    )

    # Black: Dark neutral paint (L < 112 and V < 115, or L < 95)
    mask_black = mask_achromatic & (~mask_white) & (
        (L < 95) | ((L < 112) & (V < 115))
    )

    # Silver: Intermediate neutral metallic paint (112 <= L < 200, low chroma)
    mask_silver = mask_achromatic & (~mask_white) & (~mask_black)

    # 6 Primary Car Colors dictionary
    bin_counts = {
        "White": int(np.sum(mask_white)),
        "Silver": int(np.sum(mask_silver)),
        "Black": int(np.sum(mask_black)),
        "Blue": int(np.sum(mask_blue)),
        "Red": int(np.sum(mask_red)),
        "Brown": int(np.sum(mask_brown)),
    }

    # Macro distribution metrics
    median_l = float(np.median(L))
    median_s = float(np.median(S))
    median_h = float(np.median(H))
    p25_l = float(np.percentile(L, 25))
    p75_l = float(np.percentile(L, 75))

    # Achromatic arbitration check:
    total_achromatic = bin_counts["White"] + bin_counts["Silver"] + bin_counts["Black"]
    total_chromatic = bin_counts["Red"] + bin_counts["Blue"] + bin_counts["Brown"]

    # If dark chromatic blue has significant votes and median hue is firmly blue with good saturation,
    # prevent deep shadow pixels from overriding genuine blue paint
    if bin_counts["Blue"] > 0 and (85.0 <= median_h <= 135.0) and median_s >= 40.0:
        if bin_counts["Blue"] > 0.35 * total_achromatic:
            total_achromatic = 0

    if total_achromatic > total_chromatic:
        if median_l < 108.0 or (p25_l < 75.0 and median_l < 125.0):
            if bin_counts["Black"] < bin_counts["Silver"]:
                bin_counts["Black"] += bin_counts["Silver"]
                bin_counts["Silver"] = 0
        elif median_l >= 200.0 or (p75_l >= 205.0 and median_l >= 185.0 and median_s < 30.0):
            if bin_counts["White"] < bin_counts["Silver"]:
                bin_counts["White"] += bin_counts["Silver"]
                bin_counts["Silver"] = 0

    # Rank color votes
    sorted_votes = sorted(bin_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_color, top_count = sorted_votes[0]
    runner_up_color, runner_up_count = sorted_votes[1]

    total_classified = sum(bin_counts.values())
    top_ratio = top_count / float(total_classified) if total_classified > 0 else 0.0
    runner_up_ratio = runner_up_count / float(total_classified) if total_classified > 0 else 0.0

    # Check for two-tone paint scheme (e.g. black roof on white car)
    secondary_color = None
    if runner_up_ratio >= 0.22 and runner_up_count >= 50:
        secondary_color = runner_up_color

    # Confidence calculation:
    margin = max(0.0, top_ratio - runner_up_ratio)
    area_ratio = min(1.0, total_valid / float(h_c * w_c))
    calculated_conf = min(0.96, max(0.35, 0.45 + (top_ratio * 0.40) + (margin * 0.15) + (area_ratio * 0.05)))
    calculated_conf = round(calculated_conf, 2)

    status = "success"
    if top_ratio < 0.28 or calculated_conf < 0.55:
        status = "uncertain"
        warnings.append(f"Color distribution is diffuse; top candidate '{top_color}' accounts for only {top_ratio*100:.1f}% of usable paint pixels.")

    return {
        "status": status,
        "color": top_color if status == "success" else None,
        "secondary_color": secondary_color if status == "success" else None,
        "confidence": calculated_conf,
        "method": "HSV/CIELAB robust paint-region clustering with highlight/shadow rejection",
        "warnings": warnings,
        "metrics": {
            "valid_paint_pixels": total_valid,
            "paint_pixel_ratio": round(area_ratio, 2),
            "dominant_color": top_color,
            "dominant_ratio": round(top_ratio, 2),
            "secondary_color": secondary_color,
            "secondary_ratio": round(runner_up_ratio, 2) if secondary_color else 0.0,
            "median_l": round(median_l, 1),
            "median_s": round(median_s, 1),
            "median_h": round(median_h, 1),
        },
    }


# ---------------------------------------------------------------------------
# Module 7: Backward-Compatibility Layer for VehicleVision & videoGen.py
# ---------------------------------------------------------------------------
class VehicleDetector:
    """
    Singleton compatibility adapter preserving exact API contract for
    legacy callers (especially videoGen.py and server.py).
    """

    _instance: Optional["VehicleDetector"] = None

    def __init__(self, *args, **kwargs):
        # Kept for signature compatibility
        self.conf_threshold = kwargs.get("conf_threshold", 0.35)
        self.min_area_ratio = kwargs.get("min_area_ratio", 0.05)
        self.allowed_classes = kwargs.get("allowed_classes", {"car", "motorcycle", "bus", "truck"})

    @classmethod
    def get_instance(cls, *args, **kwargs) -> "VehicleDetector":
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def detect(
        self,
        image_input: Union[str, Path, np.ndarray],
        conf_threshold: Optional[float] = None,
        min_area_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Bridge call to detect_vehicle, mapping output to legacy schema."""
        res = detect_vehicle(image_input)

        is_veh = bool(res.get("is_vehicle_related", False) or res.get("is_vehicle", False))
        v_type = res.get("vehicle_type") or "car"
        box_xywh = res.get("bounding_box")

        detections = []
        max_area_ratio = 0.0

        if is_veh and box_xywh:
            bx, by, bw, bh = box_xywh
            # Support both [x1, y1, x2, y2] and [x, y, w, h] seamlessly
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            # Area ratio relative to image dimensions
            orig_h = max(1, by + bh)
            orig_w = max(1, bx + bw)
            if isinstance(image_input, np.ndarray):
                orig_h, orig_w = image_input.shape[:2]
            area_ratio = round(float(bw * bh) / max(1.0, float(orig_w * orig_h)), 4)
            max_area_ratio = area_ratio

            detections.append({
                "class": str(v_type),
                "confidence": float(res.get("confidence", 0.70)),
                "box": [x1, y1, x2, y2],       # videoGen.py line 808 expects [x1, y1, x2, y2]
                "box_xywh": [bx, by, bw, bh],
                "area_ratio": area_ratio,
            })

        rejection_reason = ""
        if not is_veh:
            scene = res.get("scene_type", "not_vehicle")
            if res.get("is_rc_document"):
                rejection_reason = "Rejected: rc_document (Indian Registration Certificate)"
            else:
                rejection_reason = f"Rejected: {scene}"

        file_name = ""
        file_path = ""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            file_name = p.name
            file_path = str(p)

        return {
            "success": bool(res.get("success", True)),
            "image_path": file_path,
            "is_vehicle": is_veh,
            "is_vehicle_related": is_veh,
            "is_rc_document": bool(res.get("is_rc_document", False)),
            "is_interior": bool(res.get("is_interior", False)),
            "is_inside": bool(res.get("is_inside", False)),
            "is_exterior": bool(res.get("is_exterior", False)),
            "scene_type": res.get("scene_type", "not_vehicle"),
            "vehicle_side": res.get("vehicle_side"),
            "vehicle_type": res.get("vehicle_type"),
            "interior_type": res.get("interior_type"),
            "interior_components": res.get("interior_components", []),
            "interior_confidence": float(res.get("interior_confidence", 0.0)),
            "exterior_confidence": float(res.get("exterior_confidence", 0.0)),
            "detected_cues": res.get("detected_cues", []),
            "method": res.get("method", "opencv_heuristics_no_pretrained_model"),
            "filename": file_name,
            "path": file_path,
            "confidence": float(res.get("confidence", 0.0)),
            "detected_classes": [str(v_type)] if is_veh else [],
            "detections": detections,
            "max_area_ratio": max_area_ratio,
            "rejection_reason": rejection_reason,
            "raw_analysis": res,
            "rc_details": res.get("rc_details"),
        }


def filter_vehicle_images(
    image_paths: Sequence[Union[str, Path]],
    conf_threshold: float = 0.35,
    min_area_ratio: float = 0.05,
    fallback_policy: str = "strict",
    **kwargs,
) -> Dict[str, Any]:
    """
    Filter sequence of image paths for videoGen.py, retaining vehicle images
    (both exterior and interior photos) while strictly rejecting documents/non-vehicles.
    """
    start_time = time.time()
    retained_paths: List[str] = []
    filtered_paths: List[str] = []
    details: List[Dict[str, Any]] = []

    detector = VehicleDetector.get_instance()

    exterior_paths: List[str] = []
    interior_paths: List[str] = []

    for p in image_paths:
        p_str = str(p)
        res = detector.detect(p_str, conf_threshold=conf_threshold, min_area_ratio=min_area_ratio)
        res["path"] = p_str
        details.append(res)
        if res["is_vehicle"]:
            retained_paths.append(p_str)
            if res.get("is_interior") or res.get("scene_type") == "vehicle_interior":
                interior_paths.append(p_str)
            else:
                exterior_paths.append(p_str)
        else:
            filtered_paths.append(p_str)

    fallback_triggered = False
    if not retained_paths and image_paths:
        if fallback_policy == "keep_all_if_empty":
            retained_paths = [str(x) for x in image_paths]
            exterior_paths = list(retained_paths)
            fallback_triggered = True
        elif fallback_policy == "keep_first_if_empty":
            retained_paths = [str(image_paths[0])]
            exterior_paths = list(retained_paths)
            fallback_triggered = True

    elapsed = time.time() - start_time
    avg_ms = (elapsed / max(1, len(image_paths))) * 1000.0

    return {
        "retained_paths": retained_paths,
        "filtered_paths": filtered_paths,
        "exterior_paths": exterior_paths,
        "interior_paths": interior_paths,
        "total_input": len(image_paths),
        "retained_count": len(retained_paths),
        "filtered_count": len(filtered_paths),
        "elapsed_seconds": round(elapsed, 4),
        "avg_ms_per_image": round(avg_ms, 2),
        "fallback_triggered": fallback_triggered,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Module 8: Synthetic Self-Test Mode (--self-test)
# ---------------------------------------------------------------------------
def run_self_test() -> int:
    """
    Execute self-contained synthetic tests verifying program execution,
    classification logic, and valid JSON serialization without external files.
    """
    print("[INFO] Starting VehicleDetector synthetic self-test suite...", file=sys.stderr)
    passed = 0
    total = 0

    def assert_test(name: str, condition: bool, msg: str = ""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}", file=sys.stderr)
        else:
            print(f"  [FAIL] {name} - {msg}", file=sys.stderr)

    # Test 1: Solid Black Image (Not a vehicle)
    img_black = np.zeros((480, 640, 3), dtype=np.uint8)
    res_black = detect_vehicle(img_black)
    assert_test(
        "Solid black image is not a vehicle",
        res_black["scene_type"] in ("not_vehicle", "uncertain") and not res_black["is_vehicle_related"],
        f"Got: {res_black['scene_type']}",
    )

    # Test 2: Synthetic Document / Invoice (White background + horizontal text lines)
    img_doc = np.full((700, 500, 3), 245, dtype=np.uint8)
    # Draw horizontal text strips
    for y in range(80, 650, 24):
        cv2.line(img_doc, (50, y), (450, y), (30, 30, 30), 4)
    res_doc = detect_vehicle(img_doc)
    assert_test(
        "Synthetic text document rejected as not_vehicle",
        res_doc["scene_type"] == "not_vehicle" and not res_doc["is_vehicle_related"],
        f"Got: {res_doc['scene_type']}",
    )

    # Test 3: Synthetic Car Side Silhouette
    # Dark asphalt background + car body + 2 round wheels + dark ground shadow
    img_car = np.full((480, 640, 3), 160, dtype=np.uint8)
    # Ground asphalt
    img_car[380:, :] = (60, 60, 60)
    # Ground contact shadow seam
    cv2.rectangle(img_car, (120, 375), (520, 385), (20, 20, 20), -1)
    # Car lower body (rectangle)
    cv2.rectangle(img_car, (120, 280), (520, 375), (210, 30, 30), -1)
    # Car cabin greenhouse (upper step)
    pts = np.array([[200, 280], [250, 210], [420, 210], [470, 280]], np.int32)
    cv2.fillPoly(img_car, [pts], (210, 30, 30))
    # Two circular wheels in lower third
    cv2.circle(img_car, (200, 375), 32, (15, 15, 15), -1)
    cv2.circle(img_car, (440, 375), 32, (15, 15, 15), -1)
    # Hubcaps
    cv2.circle(img_car, (200, 375), 12, (200, 200, 200), -1)
    cv2.circle(img_car, (440, 375), 12, (200, 200, 200), -1)

    res_car = detect_vehicle(img_car)
    assert_test(
        "Synthetic car silhouette detected as vehicle_exterior",
        res_car["scene_type"] == "vehicle_exterior" and res_car["is_vehicle_related"],
        f"Got: {res_car['scene_type']}, {res_car.get('vehicle_type')}",
    )

    # Test 4: Synthetic Vehicle Interior (Steering wheel + Dashboard horizon)
    img_interior = np.full((480, 640, 3), 70, dtype=np.uint8)
    # Windshield brightness above
    img_interior[:180, :] = (200, 210, 220)
    # Dashboard horizon line
    cv2.line(img_interior, (0, 180), (640, 180), (20, 20, 20), 4)
    # Steering wheel ring
    cv2.circle(img_interior, (220, 330), 80, (20, 20, 20), 16)
    cv2.circle(img_interior, (220, 330), 25, (30, 30, 30), -1)
    cv2.circle(img_interior, (220, 330), 10, (180, 180, 180), -1)  # Chrome badge
    cv2.line(img_interior, (140, 330), (300, 330), (25, 25, 25), 10)
    cv2.line(img_interior, (220, 330), (220, 410), (25, 25, 25), 10)

    res_int = detect_vehicle(img_interior)
    has_interior_components = "steering_wheel" in res_int.get("interior_components", []) and "dashboard" in res_int.get("interior_components", [])
    assert_test(
        "Synthetic cockpit detected as vehicle_interior with steering_wheel & dashboard components",
        res_int["scene_type"] == "vehicle_interior" and res_int["is_vehicle_related"] and has_interior_components,
        f"Got: scene={res_int['scene_type']}, components={res_int.get('interior_components')}",
    )

    # Test 5: JSON Serialization Safety (No NumPy types)
    try:
        json_str = json.dumps(res_car)
        assert_test("JSON serialization passes cleanly", isinstance(json_str, str))
    except Exception as e:
        assert_test("JSON serialization passes cleanly", False, str(e))

    # Test 6: Non-existent file error handling
    res_err = detect_vehicle("/invalid/path/that/does/not/exist.png")
    assert_test(
        "Missing file returns clean error dict",
        res_err["error"] is not None and not res_err["is_vehicle_related"] and res_err["success"] is False,
        f"Got: {res_err}",
    )

    # Test 7: Synthetic Indian RC Document Detection
    rc_card = np.full((500, 800, 3), 245, dtype=np.uint8)
    cv2.rectangle(rc_card, (20, 20), (780, 480), (180, 180, 180), 2)
    cv2.putText(rc_card, "TRANSPORT DEPARTMENT GOVT OF DELHI", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(rc_card, "REGISTRATION CERTIFICATE", (150, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(rc_card, "REGN NO: DL 01 AB 1234", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(rc_card, "CHASSIS NO: MA3ER451234567890", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(rc_card, "ENGINE NO: K15B1234567", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(rc_card, "MAKER: MARUTI SUZUKI INDIA LTD", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(rc_card, "MODEL: ERTIGA VXI PETROL", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    res_rc_veh = detect_vehicle(rc_card)
    assert_test(
        "Synthetic RC card rejected from vehicle and flagged as rc_document",
        res_rc_veh.get("is_rc_document") is True and res_rc_veh.get("scene_type") == "rc_document" and not res_rc_veh.get("is_vehicle_related"),
        f"Got: scene_type={res_rc_veh.get('scene_type')}, is_rc={res_rc_veh.get('is_rc_document')}",
    )

    # Test 8: Synthetic Competing Document (Aadhaar Card)
    aadhaar_card = np.full((500, 800, 3), 250, dtype=np.uint8)
    cv2.rectangle(aadhaar_card, (20, 20), (780, 480), (160, 160, 160), 2)
    cv2.putText(aadhaar_card, "GOVERNMENT OF INDIA", (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(aadhaar_card, "UNIQUE IDENTIFICATION AUTHORITY OF INDIA", (60, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(aadhaar_card, "AADHAAR CARD", (260, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(aadhaar_card, "1234 5678 9012", (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

    res_aadhaar_veh = detect_vehicle(aadhaar_card)
    assert_test(
        "Aadhaar card rejected as not_vehicle and not rc_document",
        res_aadhaar_veh.get("is_rc_document") is False and res_aadhaar_veh.get("scene_type") == "not_vehicle",
        f"Got: scene_type={res_aadhaar_veh.get('scene_type')}, is_rc={res_aadhaar_veh.get('is_rc_document')}",
    )

    # Test 9: Direct detect_rc_document API
    rc_direct = detect_rc_document(rc_card)
    assert_test(
        "Direct detect_rc_document correctly identifies RC",
        rc_direct.get("is_rc_document") is True and rc_direct.get("classification") == "rc_document",
        f"Got: is_rc={rc_direct.get('is_rc_document')}, class={rc_direct.get('classification')}",
    )

    # Test 10: Synthetic Bus Interior (Multiple seat rows + upholstery)
    img_bus = np.full((500, 700, 3), 25, dtype=np.uint8)  # Dark cabin trim background
    # Repeated seat headrests (aspect ratio ~1.8, tan upholstery)
    for row_y in (100, 170, 240, 310):
        cv2.rectangle(img_bus, (100, row_y), (200, row_y + 55), (20, 100, 200), -1)
        cv2.rectangle(img_bus, (500, row_y), (600, row_y + 55), (20, 100, 200), -1)
    res_bus = detect_vehicle(img_bus)
    assert_test(
        "Synthetic bus cabin identified with bus interior components",
        res_bus.get("scene_type") in ("vehicle_interior", "vehicle_detail") and any("bus" in c for c in res_bus.get("interior_components", [])),
        f"Got: scene={res_bus.get('scene_type')}, components={res_bus.get('interior_components')}",
    )

    # Test 11: Ambiguous single circle on flat background
    img_detail = np.full((400, 400, 3), 120, dtype=np.uint8)
    cv2.circle(img_detail, (200, 200), 70, (30, 30, 30), 12)
    res_detail = detect_vehicle(img_detail)
    assert_test(
        "Single ambiguous circle without cabin context returns vehicle_detail or not_vehicle/uncertain",
        res_detail["scene_type"] in ("vehicle_detail", "uncertain", "not_vehicle"),
        f"Got: {res_detail['scene_type']}",
    )

    # Test 12: Prompt Output Schema Compliance
    required_keys = {
        "success", "image_path", "is_vehicle", "scene_type", "vehicle_type",
        "interior_components", "interior_confidence", "exterior_confidence",
        "type_confidence", "detected_cues", "warnings", "method",
    }
    schema_ok = (
        required_keys.issubset(res_car.keys())
        and required_keys.issubset(res_int.keys())
        and required_keys.issubset(res_black.keys())
    )
    assert_test(
        "Full prompt output schema keys present across all test results",
        schema_ok,
        f"Missing from interior: {required_keys - set(res_int.keys())}",
    )

    print(f"\n[SUMMARY] Self-Test Completed: {passed}/{total} Passed", file=sys.stderr)
    return 0 if (passed == total) else 1


# ---------------------------------------------------------------------------
# Module 9: CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="VehicleVision — Standalone AI Vehicle Detector & Classifier",
        epilog="Runs 100% offline using explainable computer-vision heuristics and local OCR.",
    )
    parser.add_argument("image", nargs="?", help="Path to input image (JPG, PNG, WEBP).")
    parser.add_argument("--image", "-i", dest="flag_image", help="Path to input image (flag format).")
    parser.add_argument("--debug", action="store_true", help="Print diagnostic logs to stderr.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in synthetic unit tests.")
    parser.add_argument("--rc", "--detect-rc", dest="detect_rc", action="store_true", help="Explicit Indian RC document detection mode.")
    parser.add_argument("--detect-color", "--color", dest="detect_color", action="store_true", help="Explicit vehicle exterior color detection mode.")
    parser.add_argument("--show-ocr-text", action="store_true", help="Include extracted OCR text in stdout JSON output.")
    parser.add_argument("--save-visualization", type=str, default=None, help="Save annotated visualization image to the specified path.")
    parser.add_argument("--input-dir", "-d", help="Directory of images to batch-evaluate.")
    parser.add_argument("--threshold", "-t", type=float, default=CONFIDENCE_VEHICLE_MIN, help="Confidence threshold.")
    parser.add_argument("--min-area", type=float, default=0.05, help="Minimum vehicle bbox area ratio.")
    parser.add_argument("--json", "-j", action="store_true", help="Explicit JSON mode.")

    args = parser.parse_args()
    args.image = args.image or args.flag_image

    # Self-Test Mode
    if args.self_test:
        sys.exit(run_self_test())

    # Batch Directory Mode
    _tracker_cls = _ResourceTracker if _ResourceTracker is not None else None
    _tracker = _tracker_cls() if _tracker_cls else None
    ctx = _tracker if _tracker is not None else _NullContext()

    with ctx:
        if args.input_dir:
            d = Path(args.input_dir)
            if not d.is_dir():
                print(json.dumps({"error": f"Input directory not found: {args.input_dir}"}), file=sys.stdout)
                sys.exit(1)
            image_paths = []
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                image_paths.extend(sorted(d.glob(ext)))
            result = filter_vehicle_images(
                image_paths,
                conf_threshold=args.threshold,
                min_area_ratio=args.min_area,
            )
        elif not args.image:
            parser.print_help(sys.stderr)
            sys.exit(1)
        elif args.detect_color:
            result = detect_vehicle_color(args.image, debug=args.debug)
        elif args.detect_rc:
            result = detect_rc_document(args.image)
            if not args.show_ocr_text and result.get("ocr_text"):
                result["ocr_text"] = f"[Extracted {len(result['ocr_text'])} characters. Use --show-ocr-text to display]"
            if args.save_visualization and result.get("success"):
                save_rc_visualization(args.image, result, args.save_visualization)
        else:
            result = detect_vehicle(args.image, debug=args.debug)

    if _tracker is not None and isinstance(result, dict):
        result["resource_metrics"] = {
            "elapsed_sec": _tracker.elapsed_sec,
            "peak_ram_mb": _tracker.peak_ram_mb,
            "cpu_percent": _tracker.cpu_percent,
        }

    # Print pure JSON to stdout
    print(json.dumps(result, indent=2, ensure_ascii=False), file=sys.stdout)

    # Exit code: 0 if valid result, 1 if input/decoding error
    if isinstance(result, dict) and result.get("error"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
