"""
Auto Frame AI — Server API
FastAPI Backend handling routes and logic for Vehicle Computer Vision,
Background Removal, Dealership Branding, Studio Compositing, OCR & Video Generation.
"""

import os
import sys
import time
import uuid
import shutil
import logging
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Core Pipeline Imports (Preserving existing files untouched)
from vehicle_detector import VehicleDetector, detect_vehicle
from removeBg import VehicleBackgroundRemovalPipeline
from addBg import compose_images
from getText import SUPPORTED_IMAGE_EXTENSIONS, analyze_images, analyze_rc, analyze_vehicle
from videoGen import generate_video

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("AutoFrameServer")

# Initialize Directories
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
BG_ASSETS_DIR = ASSETS_DIR / "backgrounds"
STATIC_DIR = BASE_DIR / "static"
SAMPLES_DIR = BASE_DIR / "color cars"

for directory in (UPLOADS_DIR, OUTPUTS_DIR, BG_ASSETS_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Thread pool for CPU-bound computer vision workloads
cv_executor = ThreadPoolExecutor(max_workers=max(2, (os.cpu_count() or 4) // 2))

# In-memory video rendering & pipeline jobs tracker
video_jobs: Dict[str, Dict[str, Any]] = {}
pipeline_jobs: Dict[str, Dict[str, Any]] = {}

# Cached Pipelines
_bg_pipelines: Dict[str, VehicleBackgroundRemovalPipeline] = {}


def get_bg_pipeline(model_name: str = "u2net") -> VehicleBackgroundRemovalPipeline:
    """Lazily load and cache the background removal pipeline."""
    if model_name not in _bg_pipelines:
        logger.info(f"Initializing VehicleBackgroundRemovalPipeline with model '{model_name}'...")
        _bg_pipelines[model_name] = VehicleBackgroundRemovalPipeline(model_name=model_name)
    return _bg_pipelines[model_name]


# Initialize FastAPI application
app = FastAPI(
    title="Auto Frame AI API",
    description="Automated AI Studio Framing, Background Removal, License Plate Branding, OCR, and Video Showcase Generator for Vehicles.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static File Routes
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES_DIR)), name="samples")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
async def save_upload_file(upload_file: UploadFile, target_dir: Path) -> Path:
    """Save an uploaded file safely to disk and return its path."""
    ext = Path(upload_file.filename).suffix.lower() if upload_file.filename else ".jpg"
    if not ext:
        ext = ".jpg"
    unique_name = f"{uuid.uuid4().hex[:12]}{ext}"
    target_path = target_dir / unique_name
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return target_path


def get_public_url(path: Path) -> str:
    """Derive public URL path relative to the server mount points."""
    resolved = path.resolve()
    if OUTPUTS_DIR.resolve() in resolved.parents or resolved == OUTPUTS_DIR.resolve():
        return f"/outputs/{path.name}"
    if UPLOADS_DIR.resolve() in resolved.parents or resolved == UPLOADS_DIR.resolve():
        return f"/uploads/{path.name}"
    if ASSETS_DIR.resolve() in resolved.parents or resolved == ASSETS_DIR.resolve():
        rel = resolved.relative_to(ASSETS_DIR.resolve())
        return f"/assets/{rel}"
    if SAMPLES_DIR.resolve() in resolved.parents or resolved == SAMPLES_DIR.resolve():
        return f"/samples/{path.name}"
    return f"/outputs/{path.name}"


# -----------------------------------------------------------------------------
# Root & Health Check Endpoints
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve UI index.html if present, otherwise interactive landing page."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))

    return HTMLResponse(
        content=f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Auto Frame AI — Backend Server</title>
            <style>
                :root {{
                    --bg-dark: #0b0f19;
                    --card-bg: #131b2e;
                    --accent-cyan: #06b6d4;
                    --accent-blue: #3b82f6;
                    --text-primary: #f8fafc;
                    --text-secondary: #94a3b8;
                    --border-color: #1e293b;
                }}
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background-color: var(--bg-dark);
                    color: var(--text-primary);
                    margin: 0;
                    padding: 40px 20px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .container {{
                    max-width: 900px;
                    width: 100%;
                }}
                .badge {{
                    display: inline-block;
                    background: rgba(6, 182, 212, 0.15);
                    color: var(--accent-cyan);
                    padding: 6px 14px;
                    border-radius: 9999px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    margin-bottom: 12px;
                    border: 1px solid rgba(6, 182, 212, 0.3);
                }}
                h1 {{
                    font-size: 2.5rem;
                    margin: 0 0 10px 0;
                    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                p.lead {{
                    color: var(--text-secondary);
                    font-size: 1.1rem;
                    margin-bottom: 30px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                    gap: 18px;
                    margin-bottom: 30px;
                }}
                .card {{
                    background: var(--card-bg);
                    border: 1px solid var(--border-color);
                    border-radius: 12px;
                    padding: 20px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                }}
                .card h3 {{
                    margin: 0 0 8px 0;
                    color: #e2e8f0;
                    font-size: 1.1rem;
                }}
                .card p {{
                    color: var(--text-secondary);
                    font-size: 0.9rem;
                    margin: 0 0 12px 0;
                    line-height: 1.4;
                }}
                .route-pill {{
                    display: inline-block;
                    font-family: monospace;
                    font-size: 0.8rem;
                    padding: 4px 8px;
                    border-radius: 6px;
                    background: #1e293b;
                    color: #38bdf8;
                }}
                .actions {{
                    display: flex;
                    gap: 12px;
                    margin-top: 20px;
                }}
                a.btn {{
                    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
                    color: white;
                    text-decoration: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-weight: 600;
                    font-size: 0.95rem;
                    transition: transform 0.15s ease;
                }}
                a.btn:hover {{
                    transform: translateY(-2px);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="badge">● Server Active & Operational</div>
                <h1>Auto Frame AI Engine</h1>
                <p class="lead">Vehicle Vision Pipeline: AI Cutout, Studio Showroom Compositing, License Plate Branding, OCR & Cinematic Video Generation.</p>
                
                <div class="grid">
                    <div class="card">
                        <h3>🚗 Vehicle Detection</h3>
                        <p>Classify vehicle type, camera view orientation, bounding boxes, and document check.</p>
                        <span class="route-pill">POST /api/detect</span>
                    </div>
                    <div class="card">
                        <h3>✂️ Background Removal</h3>
                        <p>AI matting with U2Net/Pure-CV and optional dealer name license plate replacement.</p>
                        <span class="route-pill">POST /api/remove-bg</span>
                    </div>
                    <div class="card">
                        <h3>🏢 Studio Compositing</h3>
                        <p>Place cutouts into showroom backdrops with realistic ground contact shadows and reflections.</p>
                        <span class="route-pill">POST /api/add-bg</span>
                    </div>
                    <div class="card">
                        <h3>⚡ AutoFrame All-In-One</h3>
                        <p>Complete end-to-end pipeline: detect, cutout, brand plate, and frame in studio showroom in one call.</p>
                        <span class="route-pill">POST /api/autoframe</span>
                    </div>
                    <div class="card">
                        <h3>📄 OCR & RC Extraction</h3>
                        <p>Extract vehicle registration numbers, make/model, fuel, and specs from photos or RC documents.</p>
                        <span class="route-pill">POST /api/ocr</span>
                    </div>
                    <div class="card">
                        <h3>🎬 Video Showcase</h3>
                        <p>Multi-slide vehicle video generation with dynamic badge overlays and transitions.</p>
                        <span class="route-pill">POST /api/video/generate</span>
                    </div>
                </div>

                <div class="actions">
                    <a href="/docs" class="btn">Open Interactive Swagger API Docs</a>
                </div>
            </div>
        </body>
        </html>
        """
    )


@app.get("/api/health")
async def health_check():
    """Health and runtime diagnostics endpoint."""
    return {
        "status": "healthy",
        "service": "Auto Frame AI Server",
        "version": "1.0.0",
        "timestamp": time.time(),
        "available_backgrounds": [p.name for p in BG_ASSETS_DIR.glob("*") if p.is_file()],
        "features": {
            "vehicle_detector": True,
            "background_removal": True,
            "background_compositing": True,
            "license_plate_branding": True,
            "ocr_text_extraction": True,
            "video_generation": True,
        }
    }


# -----------------------------------------------------------------------------
# Route: Vehicle Detection
# -----------------------------------------------------------------------------
@app.post("/api/detect")
async def route_detect_vehicle(file: UploadFile = File(...)):
    """
    Detect vehicle attributes: bounding box, category, view orientation, and confidence.
    """
    input_path = await save_upload_file(file, UPLOADS_DIR)
    
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            cv_executor,
            detect_vehicle,
            str(input_path)
        )
    except Exception as exc:
        logger.error(f"Error in detect_vehicle: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "file_url": get_public_url(input_path),
        "detection": result,
    }


# -----------------------------------------------------------------------------
# Route: Background Removal & License Plate Branding
# -----------------------------------------------------------------------------
@app.post("/api/remove-bg")
async def route_remove_background(
    file: UploadFile = File(...),
    vName: Optional[str] = Form(None, description="Optional dealer/vendor name to brand onto the license plate"),
    bg_color: Optional[str] = Form(None, description="Optional background color hex (e.g. #ffffff) or empty for transparent PNG"),
    edge_refine: str = Form("crisp", description="Edge refinement: 'crisp', 'smooth', or 'none'"),
    model_name: str = Form("u2net", description="Segmentation model: 'u2net' or 'pure-cv'"),
):
    """
    Removes background from vehicle photo and optionally replaces license plate with vendor branding.
    """
    input_path = await save_upload_file(file, UPLOADS_DIR)

    def _process():
        pipeline = get_bg_pipeline(model_name=model_name)
        with Image.open(input_path) as pil_img:
            img_rgb = pil_img.convert("RGB")
            res = pipeline.process(
                image_input=img_rgb,
                bg_color=bg_color if bg_color and bg_color.strip() else None,
                edge_refine=edge_refine,
                vName=vName.strip() if vName and vName.strip() else None,
            )

        output_filename = f"cutout_{uuid.uuid4().hex[:10]}.png"
        out_path = OUTPUTS_DIR / output_filename
        res["cutout_rgba"].save(out_path, format="PNG")
        return out_path, res

    loop = asyncio.get_event_loop()
    try:
        out_path, res = await loop.run_in_executor(cv_executor, _process)
    except Exception as exc:
        logger.error(f"Error in remove_background: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "output_url": get_public_url(out_path),
        "input_url": get_public_url(input_path),
        "vendor_name_applied": bool(vName and vName.strip()),
        "plate_replacement_meta": res.get("plate_replacement_meta", {}),
        "wall_time_ms": res.get("wall_time_ms", 0),
    }


# -----------------------------------------------------------------------------
# Route: Studio Background Compositing (addBg.py)
# -----------------------------------------------------------------------------
@app.post("/api/add-bg")
async def route_add_background(
    fg_file: UploadFile = File(..., description="Vehicle cutout with transparency"),
    bg_file: Optional[UploadFile] = File(None, description="Optional custom background image"),
    preset: str = Form("showroom", description="Studio preset: 'showroom', 'studio', 'dark', 'outdoor'"),
    bg_name: Optional[str] = Form(None, description="Name of template background in assets/backgrounds/"),
    output_format: str = Form("jpg", description="Output format: 'jpg', 'png', 'webp'"),
    veh_scale: Optional[float] = Form(None, description="Vehicle scale adjustment (e.g. 1.0)"),
    veh_offset_x: Optional[int] = Form(None, description="Vehicle horizontal offset in px"),
    veh_offset_y: Optional[int] = Form(None, description="Vehicle vertical offset in px"),
    shadow_underbody: Optional[float] = Form(None, description="Shadow strength under car (0.0 - 1.0)"),
    refl_opacity: Optional[float] = Form(None, description="Floor reflection opacity (0.0 - 1.0)"),
):
    """
    Composites vehicle cutout onto studio showroom background with realistic shadows and reflections.
    """
    fg_path = await save_upload_file(fg_file, UPLOADS_DIR)

    # Determine background source
    bg_path: Path
    if bg_file and bg_file.filename:
        bg_path = await save_upload_file(bg_file, UPLOADS_DIR)
    elif bg_name and (BG_ASSETS_DIR / bg_name).exists():
        bg_path = BG_ASSETS_DIR / bg_name
    else:
        # Default to studio_showroom.jpg if available
        default_bg = BG_ASSETS_DIR / "studio_showroom.jpg"
        if default_bg.exists():
            bg_path = default_bg
        else:
            # Fallback to any image in backgrounds
            available = list(BG_ASSETS_DIR.glob("*.jpg")) + list(BG_ASSETS_DIR.glob("*.png"))
            if available:
                bg_path = available[0]
            else:
                raise HTTPException(status_code=400, detail="No background image provided or found in assets/backgrounds.")

    out_filename = f"composite_{uuid.uuid4().hex[:10]}.{output_format}"
    out_path = OUTPUTS_DIR / out_filename

    kwargs = {
        "bg_image": str(bg_path),
        "fg_image": str(fg_path),
        "output_path": str(out_path),
        "preset": preset,
        "output_format": output_format,
    }
    if veh_scale is not None:
        kwargs["veh_scale"] = veh_scale
    if veh_offset_x is not None:
        kwargs["veh_offset_x"] = veh_offset_x
    if veh_offset_y is not None:
        kwargs["veh_offset_y"] = veh_offset_y
    if shadow_underbody is not None:
        kwargs["shadow_underbody_strength"] = shadow_underbody
    if refl_opacity is not None:
        kwargs["refl_opacity"] = refl_opacity

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            cv_executor,
            lambda: compose_images(**kwargs)
        )
    except Exception as exc:
        logger.error(f"Error in addBg compositing: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "output_url": get_public_url(out_path),
        "fg_url": get_public_url(fg_path),
        "bg_url": get_public_url(bg_path),
        "preset": preset,
        "process_time_ms": result.get("process_time_ms"),
    }


# -----------------------------------------------------------------------------
# Route: End-to-End AutoFrame (Detection + Cutout + Brand Plate + Studio Framing)
# -----------------------------------------------------------------------------
@app.post("/api/autoframe")
async def route_autoframe_pipeline(
    file: UploadFile = File(..., description="Raw vehicle photo"),
    dealer_name: Optional[str] = Form(None, description="Dealership name to brand on the license plate"),
    preset: str = Form("showroom", description="Studio preset: 'showroom', 'studio', 'dark'"),
    bg_name: Optional[str] = Form(None, description="Optional background template in assets/backgrounds/"),
):
    """
    One-click end-to-end studio pipeline:
    1. Detects vehicle
    2. Segments cutout & brands license plate with dealer name
    3. Composites onto studio showroom with realistic shadows & reflections
    """
    input_path = await save_upload_file(file, UPLOADS_DIR)

    def _execute_pipeline():
        t0 = time.time()
        
        # 1. Detection
        det = detect_vehicle(str(input_path))

        # 2. Cutout + Plate Branding
        pipeline = get_bg_pipeline("u2net")
        with Image.open(input_path) as pil_img:
            img_rgb = pil_img.convert("RGB")
            cutout_res = pipeline.process(
                image_input=img_rgb,
                vName=dealer_name.strip() if dealer_name and dealer_name.strip() else None,
                edge_refine="crisp",
            )
        
        cutout_filename = f"cutout_{uuid.uuid4().hex[:10]}.png"
        cutout_path = OUTPUTS_DIR / cutout_filename
        cutout_res["cutout_rgba"].save(cutout_path, format="PNG")

        # 3. Studio Background Compositing
        bg_path: Path
        if bg_name and (BG_ASSETS_DIR / bg_name).exists():
            bg_path = BG_ASSETS_DIR / bg_name
        else:
            default_bg = BG_ASSETS_DIR / "studio_showroom.jpg"
            if default_bg.exists():
                bg_path = default_bg
            else:
                available = list(BG_ASSETS_DIR.glob("*.jpg")) + list(BG_ASSETS_DIR.glob("*.png"))
                bg_path = available[0] if available else None

        composite_url = None
        if bg_path and bg_path.exists():
            composite_filename = f"framed_{uuid.uuid4().hex[:10]}.jpg"
            composite_path = OUTPUTS_DIR / composite_filename
            compose_images(
                bg_image=str(bg_path),
                fg_image=str(cutout_path),
                output_path=str(composite_path),
                preset=preset,
                output_format="jpg",
            )
            composite_url = get_public_url(composite_path)

        total_elapsed = round(time.time() - t0, 3)

        return {
            "input_url": get_public_url(input_path),
            "cutout_url": get_public_url(cutout_path),
            "framed_url": composite_url,
            "detection": det,
            "dealer_name": dealer_name,
            "plate_branded": bool(cutout_res.get("plate_replacement_meta", {}).get("replaced")),
            "elapsed_seconds": total_elapsed,
        }

    loop = asyncio.get_event_loop()
    try:
        pipeline_result = await loop.run_in_executor(cv_executor, _execute_pipeline)
    except Exception as exc:
        logger.error(f"Error in autoframe pipeline: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        **pipeline_result
    }


# -----------------------------------------------------------------------------
# Route: OCR & RC Extraction (getText.py)
# -----------------------------------------------------------------------------
@app.post("/api/ocr")
async def route_extract_ocr(
    files: List[UploadFile] = File(..., description="One or more vehicle or RC document images"),
    doc_type: str = Form("auto", description="Extraction type: 'auto', 'rc', or 'vehicle'"),
):
    """
    Extracts license plate, vehicle registration, and document specifications via OCR engine.
    """
    saved_paths: List[str] = []
    for f in files:
        p = await save_upload_file(f, UPLOADS_DIR)
        saved_paths.append(str(p))

    def _run_ocr():
        if doc_type == "rc":
            return analyze_rc(image_paths=saved_paths)
        elif doc_type == "vehicle":
            return analyze_vehicle(image_paths=saved_paths)
        else:
            return analyze_images(image_paths=saved_paths)

    loop = asyncio.get_event_loop()
    try:
        ocr_result = await loop.run_in_executor(cv_executor, _run_ocr)
    except Exception as exc:
        logger.error(f"Error in OCR extraction: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "status": "success",
        "file_urls": [get_public_url(Path(p)) for p in saved_paths],
        "ocr_result": ocr_result,
    }


# -----------------------------------------------------------------------------
# Route: Video Showcase Generator (videoGen.py)
# -----------------------------------------------------------------------------
class VideoGenerateRequest(BaseModel):
    slides: List[Dict[str, Any]]
    width: Optional[int] = 1920
    height: Optional[int] = 1080
    fps: Optional[int] = 30
    layout: Optional[str] = "landscape"
    filter_vehicles: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None


def _background_render_video(job_id: str, config: dict, out_file: Path):
    """Executes video generation in background with status tracking."""
    video_jobs[job_id]["status"] = "processing"
    video_jobs[job_id]["progress"] = 0.05

    def _progress(ratio: float, msg: str):
        video_jobs[job_id]["progress"] = round(ratio, 2)
        video_jobs[job_id]["message"] = msg

    try:
        result = generate_video(
            config=config,
            output_path=str(out_file),
            progress_callback=_progress,
        )
        video_jobs[job_id]["status"] = "completed"
        video_jobs[job_id]["progress"] = 1.0
        video_jobs[job_id]["output_url"] = get_public_url(out_file)
        video_jobs[job_id]["metrics"] = result
        logger.info(f"Video job {job_id} completed: {out_file}")
    except Exception as exc:
        logger.error(f"Video job {job_id} failed: {exc}", exc_info=True)
        video_jobs[job_id]["status"] = "failed"
        video_jobs[job_id]["error"] = str(exc)


@app.post("/api/video/generate")
async def route_generate_video(
    req: VideoGenerateRequest,
    background_tasks: BackgroundTasks
):
    """
    Submits an asynchronous video rendering job from image slides.
    Poll /api/video/status/{job_id} to check progress.
    """
    job_id = f"vid_{uuid.uuid4().hex[:10]}"
    output_path = OUTPUTS_DIR / f"{job_id}.mp4"

    video_config = {
        "video": {
            "width": req.width or 1920,
            "height": req.height or 1080,
            "fps": req.fps or 30,
            "layout": req.layout or "landscape",
            "filter_vehicles": req.filter_vehicles,
            "metadata": req.metadata if req.metadata else {"images_meta": []},
        },
        "metadata": req.metadata if req.metadata else {"images_meta": []},
        "slides": req.slides,
        "output": {
            "path": str(output_path),
        }
    }

    video_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Job queued for rendering...",
        "output_url": None,
        "created_at": time.time(),
    }

    background_tasks.add_task(_background_render_video, job_id, video_config, output_path)

    return {
        "status": "queued",
        "job_id": job_id,
        "status_url": f"/api/video/status/{job_id}",
    }


@app.get("/api/video/status/{job_id}")
async def route_video_status(job_id: str):
    """Check rendering progress and retrieve output URL when finished."""
    job = video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")
    return job


@app.get("/api/video/download/{job_id}")
async def route_video_download(job_id: str):
    """Download the rendered MP4 video file."""
    job = video_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed yet (status: {job['status']})")
    
    file_path = OUTPUTS_DIR / f"{job_id}.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Rendered video file not found on disk.")
    
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )


# -----------------------------------------------------------------------------
# Route: Studio Background Assets Management
# -----------------------------------------------------------------------------
@app.get("/api/backgrounds")
async def route_list_backgrounds():
    """List all available studio background templates."""
    backgrounds = []
    for f in BG_ASSETS_DIR.glob("*"):
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            backgrounds.append({
                "name": f.name,
                "url": f"/assets/backgrounds/{f.name}",
                "size_bytes": f.stat().st_size,
            })
    return {"backgrounds": backgrounds}


@app.post("/api/backgrounds/upload")
async def route_upload_background(file: UploadFile = File(...)):
    """Upload a custom background template to assets/backgrounds/."""
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    safe_name = f"bg_{uuid.uuid4().hex[:8]}{ext}"
    target = BG_ASSETS_DIR / safe_name
    with open(target, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {
        "status": "success",
        "name": safe_name,
        "url": f"/assets/backgrounds/{safe_name}",
    }


# -----------------------------------------------------------------------------
# Route: Sample Vehicle Images
# -----------------------------------------------------------------------------
@app.get("/api/samples")
async def route_get_samples():
    """Returns available sample car images from the repository for demoing."""
    samples = []
    if SAMPLES_DIR.exists():
        for f in sorted(SAMPLES_DIR.glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                samples.append({
                    "name": f.name,
                    "url": f"/samples/{f.name}",
                    "size_bytes": f.stat().st_size,
                })
    return {"samples": samples}


# -----------------------------------------------------------------------------
# End-to-End Pipeline: Upload Images -> Detect -> OCR -> Cutout & Brand -> Frame -> Video
# -----------------------------------------------------------------------------
def _run_full_pipeline(
    job_id: str,
    image_paths: List[Path],
    dealer_name: str,
    dealer_phone: str,
    price: str,
    car_title: Optional[str],
    preset: str,
    layout: str,
    slide_duration: float,
):
    try:
        pipeline_jobs[job_id]["status"] = "processing"
        pipeline_jobs[job_id]["progress"] = 0.10
        pipeline_jobs[job_id]["current_stage"] = "1/5: OCR & Specs Extraction"
        pipeline_jobs[job_id]["message"] = "Extracting vehicle specifications and reading number plate..."

        # 1. OCR Specs Extraction (getText.py)
        str_paths = [str(p) for p in image_paths]
        ocr_result = {}
        try:
            ocr_result = analyze_images(image_paths=str_paths)
            if isinstance(ocr_result, dict):
                pipeline_jobs[job_id]["specs"] = ocr_result.get("vehicle_info", {})
        except Exception as ocr_err:
            logger.warning(f"OCR Extraction error: {ocr_err}")

        specs_info = ocr_result.get("vehicle_info", {}) if isinstance(ocr_result, dict) else {}
        extracted_make = specs_info.get("make") or ""
        extracted_model = specs_info.get("model") or ""
        extracted_year = specs_info.get("year") or ""
        extracted_fuel = specs_info.get("fuel_type") or ""
        extracted_reg = specs_info.get("registration_number") or ""
        extracted_odo = specs_info.get("odometer") or ""

        auto_title = f"{extracted_year} {extracted_make} {extracted_model}".strip()
        final_title = car_title.strip() if car_title and car_title.strip() else (auto_title if auto_title else "Featured Vehicle")

        pipeline_jobs[job_id]["progress"] = 0.25
        pipeline_jobs[job_id]["current_stage"] = "2/5: AI Vehicle Detection & Orientation"
        pipeline_jobs[job_id]["message"] = "Detecting vehicles and classifying view orientations..."

        # 2. Vehicle Detection (vehicle_detector.py)
        detections = []
        for p in image_paths:
            try:
                det = detect_vehicle(str(p))
                detections.append(det)
            except Exception as det_err:
                logger.warning(f"Detection error on {p}: {det_err}")
                detections.append({"is_vehicle": True, "vehicle_type": "car", "view": "exterior_front"})

        pipeline_jobs[job_id]["progress"] = 0.45
        pipeline_jobs[job_id]["current_stage"] = "3/5: Background Removal & Dealership Plate Branding"
        pipeline_jobs[job_id]["message"] = f"Generating AI vehicle cutouts and stamping '{dealer_name}' onto plate..."

        # 3. Background Removal & Dealer Branding (removeBg.py) + 4. Compositing (addBg.py)
        bg_removal_pipeline = get_bg_pipeline("u2net")
        processed_slides = []

        total_imgs = len(image_paths)
        for idx, (p, det) in enumerate(zip(image_paths, detections)):
            view_type = det.get("detected_view") or det.get("vehicle_side") or det.get("scene_type") or "exterior"
            is_interior = bool(det.get("is_interior") or view_type in ("interior", "dashboard", "cabin"))

            orig_url = get_public_url(p)
            slide_item = {
                "original_url": orig_url,
                "view": view_type,
                "is_interior": is_interior,
                "confidence": det.get("confidence", 0.9),
            }

            if is_interior:
                slide_item["framed_url"] = orig_url
                slide_item["cutout_url"] = orig_url
                processed_slides.append({
                    "image": str(p),
                    "meta": slide_item
                })
            else:
                try:
                    with Image.open(p) as pil_img:
                        img_rgb = pil_img.convert("RGB")
                        cutout_res = bg_removal_pipeline.process(
                            image_input=img_rgb,
                            vName=dealer_name.strip() if dealer_name else None,
                            edge_refine="crisp"
                        )
                    cutout_p = OUTPUTS_DIR / f"cutout_{job_id}_{idx}.png"
                    cutout_res["cutout_rgba"].save(cutout_p, format="PNG")
                    slide_item["cutout_url"] = get_public_url(cutout_p)

                    # Studio Showroom Compositing (addBg.py)
                    bg_file = BG_ASSETS_DIR / "studio_showroom.jpg"
                    if not bg_file.exists():
                        avail = list(BG_ASSETS_DIR.glob("*.jpg")) + list(BG_ASSETS_DIR.glob("*.png"))
                        bg_file = avail[0] if avail else None

                    framed_p = OUTPUTS_DIR / f"framed_{job_id}_{idx}.jpg"
                    if bg_file and bg_file.exists():
                        compose_images(
                            bg_image=str(bg_file),
                            fg_image=str(cutout_p),
                            output_path=str(framed_p),
                            preset=preset,
                            output_format="jpg",
                        )
                        slide_item["framed_url"] = get_public_url(framed_p)
                        processed_slides.append({
                            "image": str(framed_p),
                            "meta": slide_item
                        })
                    else:
                        slide_item["framed_url"] = orig_url
                        processed_slides.append({
                            "image": str(p),
                            "meta": slide_item
                        })
                except Exception as proc_err:
                    logger.warning(f"Error processing {p}: {proc_err}")
                    slide_item["framed_url"] = orig_url
                    slide_item["cutout_url"] = orig_url
                    processed_slides.append({
                        "image": str(p),
                        "meta": slide_item
                    })

            pipeline_jobs[job_id]["slides_data"].append(slide_item)
            pct = 0.45 + (0.30 * ((idx + 1) / total_imgs))
            pipeline_jobs[job_id]["progress"] = round(pct, 2)

        pipeline_jobs[job_id]["current_stage"] = "4/5: Studio Showroom Framing Finished"
        pipeline_jobs[job_id]["progress"] = 0.75
        pipeline_jobs[job_id]["message"] = "Preparing cinematic showcase layout and transitions..."

        # 5. Build Video Configuration & Render Video (videoGen.py)
        pipeline_jobs[job_id]["current_stage"] = "5/5: Rendering Cinematic Showcase Video"
        video_out = OUTPUTS_DIR / f"video_{job_id}.mp4"

        v_slides = []
        for s in processed_slides:
            meta = s["meta"]
            view_label = meta["view"].replace("_", " ").title()
            specs_snippet = " • ".join(filter(None, [extracted_fuel, extracted_odo, extracted_reg]))

            v_slides.append({
                "image": s["image"],
                "title": final_title,
                "sub_title": view_label if view_label != "Exterior" else f"{dealer_name}",
                "text": specs_snippet or f"Verified Inspection • {dealer_name}",
                "price": price if price else None,
                "contact": f"Call/WhatsApp: {dealer_phone}" if dealer_phone else f"Visit {dealer_name}",
                "duration": slide_duration,
                "primary": True if "front" in meta["view"].lower() else False,
            })

        w, h = (720, 1280) if layout in ("vertical_reel", "9:16", "reel") else (1080, 1080)

        video_cfg = {
            "video": {
                "width": w,
                "height": h,
                "fps": 24,
                "layout": layout,
                "default_slide_duration": slide_duration,
                "transition_duration": 0.5,
                "filter_vehicles": False,
                "metadata": {"images_meta": []},
            },
            "metadata": {"images_meta": []},
            "vendorName": dealer_name,
            "logoTitle": dealer_name,
            "images": v_slides,
            "output": {
                "path": str(video_out)
            }
        }

        def _video_progress(ratio: float, msg: str):
            curr_pct = 0.75 + (0.24 * ratio)
            pipeline_jobs[job_id]["progress"] = round(curr_pct, 2)
            pipeline_jobs[job_id]["message"] = msg

        vid_res = generate_video(
            config=video_cfg,
            output_path=str(video_out),
            progress_callback=_video_progress
        )

        pipeline_jobs[job_id]["progress"] = 1.0
        pipeline_jobs[job_id]["status"] = "completed"
        pipeline_jobs[job_id]["current_stage"] = "Completed"
        pipeline_jobs[job_id]["message"] = "Cinematic video generation complete!"
        pipeline_jobs[job_id]["video_url"] = get_public_url(video_out)
        pipeline_jobs[job_id]["video_metrics"] = vid_res
        logger.info(f"Full pipeline {job_id} successfully finished: {video_out}")

    except Exception as exc:
        logger.error(f"Full pipeline {job_id} failed: {exc}", exc_info=True)
        pipeline_jobs[job_id]["status"] = "failed"
        pipeline_jobs[job_id]["error"] = str(exc)
        pipeline_jobs[job_id]["current_stage"] = f"Failed: {exc}"
        pipeline_jobs[job_id]["message"] = f"Error during pipeline execution: {exc}"


@app.post("/api/pipeline/start")
async def route_start_pipeline(
    background_tasks: BackgroundTasks,
    files: Optional[List[UploadFile]] = File(None, description="Vehicle images to process"),
    sample_names: Optional[str] = Form(None, description="Comma-separated filenames from color cars/"),
    dealer_name: str = Form("AUTOFRAME MOTORS", description="Dealership name to brand on plates & badges"),
    dealer_phone: str = Form("+91 98765 43210", description="Contact phone or WhatsApp"),
    price: Optional[str] = Form("₹ 11,50,000", description="Vehicle asking price"),
    car_title: Optional[str] = Form(None, description="Car title (or auto-detected from OCR)"),
    preset: str = Form("showroom", description="Studio preset: 'showroom', 'studio', 'dark'"),
    layout: str = Form("vertical_reel", description="Video format: 'vertical_reel' (9:16) or 'default'"),
    slide_duration: float = Form(3.0, description="Duration per slide in seconds"),
):
    """
    Kicks off the end-to-end AutoFrame AI Pipeline:
    1. OCR & specs extraction
    2. Vehicle detection & orientation classification
    3. Background removal & dealer plate stamping
    4. Studio showroom compositing
    5. Cinematic video rendering
    """
    image_paths: List[Path] = []

    # 1. Handle uploaded files
    if files:
        for f in files:
            if f.filename:
                p = await save_upload_file(f, UPLOADS_DIR)
                image_paths.append(p)

    # 2. Handle sample images if specified
    if sample_names and sample_names.strip():
        for name in sample_names.split(","):
            clean_name = name.strip()
            if clean_name:
                sample_p = SAMPLES_DIR / clean_name
                if sample_p.exists() and sample_p.is_file():
                    # Copy to uploads for consistency
                    dest = UPLOADS_DIR / f"sample_{uuid.uuid4().hex[:6]}_{sample_p.name}"
                    shutil.copyfile(sample_p, dest)
                    image_paths.append(dest)

    if not image_paths:
        raise HTTPException(
            status_code=400,
            detail="No vehicle images provided. Please upload images or select sample images."
        )

    job_id = f"pipe_{uuid.uuid4().hex[:10]}"

    pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.05,
        "current_stage": "Queued",
        "message": "Initializing pipeline workers...",
        "specs": {},
        "slides_data": [],
        "video_url": None,
        "created_at": time.time(),
        "total_images": len(image_paths),
    }

    background_tasks.add_task(
        _run_full_pipeline,
        job_id=job_id,
        image_paths=image_paths,
        dealer_name=dealer_name,
        dealer_phone=dealer_phone,
        price=price,
        car_title=car_title,
        preset=preset,
        layout=layout,
        slide_duration=slide_duration,
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "total_images": len(image_paths),
        "status_url": f"/api/pipeline/status/{job_id}",
    }


@app.get("/api/pipeline/status/{job_id}")
async def route_pipeline_status(job_id: str):
    """Retrieves real-time progress and intermediate results of the pipeline."""
    job = pipeline_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found.")
    return job


@app.get("/api/pipeline/download/{job_id}")
async def route_pipeline_download(job_id: str):
    """Download the final generated MP4 video from the pipeline."""
    job = pipeline_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Pipeline is not complete (status: {job.get('status')})")

    file_path = OUTPUTS_DIR / f"video_{job_id}.mp4"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"vehicle_showcase_{job_id}.mp4"
    )
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Starting Auto Frame AI Server on http://{host}:{port}")
    uvicorn.run("server.py:app", host=host, port=port, reload=False)
