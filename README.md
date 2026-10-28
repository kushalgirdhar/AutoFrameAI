# 🚗 Auto Frame AI

> **Next-Generation Automotive Vision Pipeline & Video Generator Engine**  
> Automated AI vehicle detection, background matting, dealership license plate branding, studio showroom compositing, OCR spec extraction, and cinematic showcase reel generation.

---

## 🌟 Key Features

- **⚡ Complete End-to-End Pipeline**: Upload raw vehicle images $\rightarrow$ detect orientation $\rightarrow$ extract specs $\rightarrow$ remove background $\rightarrow$ stamp dealer branding on license plate $\rightarrow$ composite into studio showroom $\rightarrow$ generate social-ready MP4 video.
- **📸 AI Vehicle Detection & Orientation Classifier**: Powered by YOLOv8 and computer vision to identify vehicle types, bounding boxes, and camera perspectives (exterior front, rear, side, interior cockpit).
- **✂️ Flawless Background Matting**: U2Net AI segmentation removes noisy dealer lots, street clutter, or messy backgrounds.
- **🏷️ Automated Dealership License Plate Branding**: Automatically identifies license plate contours and stamps your custom dealership / vendor name directly onto the plate in perspective!
- **🏢 Luxury Studio Showroom Compositor**: Integrates vehicles into luxury showroom and studio backdrops with realistic ground contact shadows and floor reflections.
- **📄 OCR Document & Text Extractor**: Extracts registration number, make, model, variant, year, odometer, and fuel type from vehicle photos and RC cards using RapidOCR & Tesseract.
- **🎬 Cinematic Showcase Video Reel**: Compiles vehicle slides with animated crossfades, lower-third spec badges, asking price tags, and dealership call-to-action overlays (Vertical 9:16 for Reels/Shorts/TikTok or 16:9 Landscape).
- **💻 Modern Luxury Dark Web UI**: Fully responsive glassmorphic frontend with drag-and-drop file uploader, real-time stage progress bar, built-in MP4 player, and side-by-side comparison gallery.

---

## 📁 Project Architecture

```
Auto Frame AI/
├── server.py              # FastAPI backend handling routes and orchestration
├── static/                # Luxury automotive dark UI (HTML, CSS, JavaScript)
│   ├── index.html         # Main dashboard and workflow stepper
│   ├── style.css          # Glassmorphism design system & animations
│   └── app.js             # Client state, file queue & live polling
├── vehicle_detector.py    # YOLOv8 + OpenCV vehicle detection & view classifier
├── removeBg.py            # U2Net matting & license plate vendor replacer
├── addBg.py               # Studio showroom compositor, perspective shadow & reflection
├── getText.py             # OCR text & registration certificate extraction engine
├── videoGen.py            # FFmpeg timeline frame renderer & video generator
├── constants.py           # Indian RTO state codes, vehicle makes & model taxonomy
├── resource_tracker.py    # Real-time CPU & peak RAM consumption profiling
├── assets/
│   ├── backgrounds/       # Showroom studio backdrops
│   └── images/            # Vector badge icons & phone call graphics
├── color cars/            # Sample vehicle dataset for instant demo testing
├── requirements.txt       # Python pipeline dependencies
└── .gitignore             # Git exclusion rules
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** (Python 3.12 / 3.14 fully supported)
- **FFmpeg**: Installed and accessible on your system `$PATH`
  ```bash
  # Linux (Ubuntu/Debian)
  sudo apt-get update && sudo apt-get install -y ffmpeg
  # macOS
  brew install ffmpeg
  ```

### 2. Set Up Virtual Environment
```bash
# Clone the repository
git clone https://github.com/kushalgirdhar/AutoFrameAI.git
cd AutoFrameAI

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch the Server & UI
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
👉 **`http://localhost:8000/`**

Interactive Swagger API docs are available at:
👉 **`http://localhost:8000/docs`**

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :---: | :--- | :--- |
| `GET` | `/` | Serves the interactive Web Application UI |
| `GET` | `/api/health` | Service health check and available features |
| `GET` | `/api/samples` | Lists available demo car images from `color cars/` |
| `POST` | `/api/pipeline/start` | **All-in-one pipeline**: Upload photos $\rightarrow$ Detect $\rightarrow$ Cutout $\rightarrow$ Brand Plate $\rightarrow$ Studio Frame $\rightarrow$ Render Video |
| `GET` | `/api/pipeline/status/{id}` | Real-time progress percentage (0–100%) and stage diagnostics |
| `GET` | `/api/pipeline/download/{id}`| Download the rendered MP4 showcase video |
| `POST` | `/api/detect` | Vehicle bounding box detection and view classification |
| `POST` | `/api/remove-bg` | AI background removal and license plate dealer branding |
| `POST` | `/api/add-bg` | Compositing cutouts onto studio background with reflections |
| `POST` | `/api/autoframe` | One-shot vehicle cutout + plate branding + showroom framing |
| `POST` | `/api/ocr` | OCR extraction of registration plate, make, model & year |
| `POST` | `/api/video/generate` | Custom multi-slide MP4 video generation |

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Python Multipart, AsyncIO
- **Computer Vision**: OpenCV, NumPy, SciPy, Scikit-Image, Pillow
- **AI Inference**: ONNX Runtime, YOLOv8, U2Net, Rembg
- **OCR Engine**: RapidOCR (ONNX), PyTesseract, RapidFuzz
- **Video Engine**: FFmpeg, PIL Anti-Aliasing, SIMD Crossfade Blending
- **Frontend**: Semantic HTML5, Vanilla CSS (Glassmorphism), JavaScript (ES6+)

---

## 📄 License
This project is licensed under the Apache 2.0 License.
