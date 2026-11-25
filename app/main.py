"""
Mindee docTR REST API
Exposes all docTR features: OCR, detection, recognition, KIE, and export.
"""
import base64
import io
import logging
import os
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import Response
from PIL import Image

import cv2
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Runtime tuning (env vars)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except ValueError:
        return default


# Dans un pod, os.cpu_count() renvoie les cœurs du nœud, pas le quota du pod :
# sans limite, torch/OpenMP/OpenCV créent un thread par cœur hôte (sur-souscription).
CPU_THREADS = _env_int("CPU_THREADS", _env_int("OMP_NUM_THREADS", 4))
# Nombre max d'inférences simultanées (protège la VRAM et le predictor partagé).
MAX_CONCURRENT_INFERENCE = _env_int("MAX_CONCURRENT_INFERENCE", 2)
# Résolution de rendu des PDF.
PDF_DPI = _env_int("PDF_DPI", 200)
# Si vrai, le démarrage échoue quand CUDA est absent au lieu de basculer en CPU.
REQUIRE_GPU = os.getenv("REQUIRE_GPU", "0").lower() in ("1", "true", "yes")

torch.set_num_threads(CPU_THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:  # déjà fixé si torch a été initialisé avant
    pass
cv2.setNumThreads(CPU_THREADS)

_inference_slots = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)
_load_lock = threading.Lock()

# ---------------------------------------------------------------------------
# GPU / device selection
# ---------------------------------------------------------------------------

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if REQUIRE_GPU:
        raise RuntimeError("REQUIRE_GPU=1 mais CUDA est indisponible (torch %s, cuda %s)"
                           % (torch.__version__, torch.version.cuda))
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = _resolve_device()
logger.info(
    "Using device: %s | cpu_threads=%d | max_concurrent_inference=%d | pdf_dpi=%d",
    DEVICE, CPU_THREADS, MAX_CONCURRENT_INFERENCE, PDF_DPI,
)

# ---------------------------------------------------------------------------
# Model registry – loaded once and cached per (det_arch, reco_arch) pair
# ---------------------------------------------------------------------------
from doctr.models import detection_predictor, kie_predictor, ocr_predictor, recognition_predictor

_ocr_cache: dict = {}
_det_cache: dict = {}
_reco_cache: dict = {}
_kie_cache: dict = {}

DETECTION_MODELS = [
    "db_resnet34", "db_resnet50", "db_mobilenet_v3_large",
    "linknet_resnet18", "linknet_resnet34", "linknet_resnet50",
    "fast_tiny", "fast_small", "fast_base",
]
RECOGNITION_MODELS = [
    "crnn_vgg16_bn", "crnn_mobilenet_v3_small", "crnn_mobilenet_v3_large",
    "sar_resnet31", "master", "vitstr_small", "vitstr_base", "parseq",
]

DEFAULT_DET = "db_resnet50"
DEFAULT_RECO = "crnn_vgg16_bn"


def _to_device(predictor):
    """Move a predictor's underlying model(s) to the selected device."""
    if DEVICE.type != "cpu":
        predictor.to(DEVICE)
    return predictor


def _get_cached(cache: dict, key, label: str, factory):
    """Load a predictor once (thread-safe: routes now run in a thread pool)."""
    if key in cache:
        return cache[key]
    with _load_lock:
        if key not in cache:
            logger.info("Loading %s predictor %s on %s …", label, key, DEVICE)
            try:
                cache[key] = _to_device(factory())
            except Exception:
                logger.exception("Failed to load %s predictor %s (device=%s)", label, key, DEVICE)
                raise
    return cache[key]


def get_ocr_predictor(det_arch: str, reco_arch: str):
    return _get_cached(_ocr_cache, (det_arch, reco_arch), "OCR",
                       lambda: ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True))


def get_det_predictor(det_arch: str):
    return _get_cached(_det_cache, det_arch, "detection",
                       lambda: detection_predictor(arch=det_arch, pretrained=True))


def get_reco_predictor(reco_arch: str):
    return _get_cached(_reco_cache, reco_arch, "recognition",
                       lambda: recognition_predictor(arch=reco_arch, pretrained=True))


def get_kie_predictor(det_arch: str, reco_arch: str):
    return _get_cached(_kie_cache, (det_arch, reco_arch), "KIE",
                       lambda: kie_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True))


def _infer(predictor, inputs):
    """Run inference with a bound on concurrent GPU work."""
    with _inference_slots:
        return predictor(inputs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_bytes(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


MAX_PAGE_PIXELS = 4_000_000  # ~2000×2000 – cap pour éviter les OOM sur PDFs vectoriels


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert a page array to RGB, compositing RGBA onto a white background."""
    img = Image.fromarray(arr)
    if img.mode == "RGB":
        return arr
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return np.array(bg)
    return np.array(img.convert("RGB"))


def _cap_resolution(arr: np.ndarray) -> np.ndarray:
    """Down-scale a page if it exceeds MAX_PAGE_PIXELS."""
    h, w = arr.shape[:2]
    pixels = h * w
    if pixels <= MAX_PAGE_PIXELS:
        return arr
    ratio = (MAX_PAGE_PIXELS / pixels) ** 0.5
    new_w, new_h = max(1, int(w * ratio)), max(1, int(h * ratio))
    logger.info("Capping page resolution from %dx%d to %dx%d", w, h, new_w, new_h)
    return np.array(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS))


def _read_pdf_safe(data: bytes, dpi: int = PDF_DPI) -> list[np.ndarray]:
    """Render PDF pages to RGB arrays using poppler (pdftoppm) in a subprocess.

    Running in a separate process means a segfault in the renderer
    cannot kill the API worker (the main cause of 502s on vector PDFs).
    Output is uncompressed PPM: ~2x faster than PNG (no zlib encode/decode).
    """
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "input.pdf"
        pdf_path.write_bytes(data)
        out_prefix = Path(tmp) / "page"
        result = subprocess.run(
            ["pdftoppm", "-r", str(dpi), str(pdf_path), str(out_prefix)],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"pdftoppm failed (rc={result.returncode}): {result.stderr.decode(errors='replace')}")
        page_files = sorted(Path(tmp).glob("page-*.ppm"))
        if not page_files:
            raise RuntimeError("pdftoppm produced no output pages")
        pages = []
        for pf in page_files:
            img = Image.open(pf).convert("RGB")
            pages.append(np.array(img))
        logger.info("pdftoppm rendered %d page(s) at %d dpi", len(pages), dpi)
        return pages


def _load_document(file: UploadFile) -> list[np.ndarray]:
    """Return a list of numpy arrays (one per page) from image or PDF upload."""
    from doctr.io import DocumentFile
    data = file.file.read()
    ct = file.content_type or ""
    name = (file.filename or "").lower()
    logger.info("Loading document: filename=%s, content_type=%s, size=%d bytes", file.filename, ct, len(data))
    if ct == "application/pdf" or name.endswith(".pdf"):
        raw = _read_pdf_safe(data)
        return [_cap_resolution(p) for p in raw]  # déjà RGB (convert dans _read_pdf_safe)
    return DocumentFile.from_images([data])


def _word_to_dict(word) -> dict:
    return {
        "value": word.value,
        "confidence": round(float(word.confidence), 4),
        "geometry": word.geometry,
        "crop_orientation": getattr(word, "crop_orientation", None),
        "objectness_score": getattr(word, "objectness_score", None),
    }


def _page_to_dict(page) -> dict:
    blocks = []
    for block in page.blocks:
        lines = []
        for line in block.lines:
            lines.append({
                "geometry": line.geometry,
                "words": [_word_to_dict(w) for w in line.words],
            })
        artefacts = [
            {"value": a.value, "geometry": a.geometry, "type": str(type(a).__name__)}
            for a in block.artefacts
        ]
        blocks.append({"geometry": block.geometry, "lines": lines, "artefacts": artefacts})
    return {
        "page_idx": page.page_idx,
        "dimensions": page.dimensions,
        "orientation": page.orientation,
        "language": page.language,
        "blocks": blocks,
    }


def _doc_to_dict(doc) -> dict:
    return {"pages": [_page_to_dict(p) for p in doc.pages]}


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Pre-loading default OCR predictor (%s + %s) …", DEFAULT_DET, DEFAULT_RECO)
    get_ocr_predictor(DEFAULT_DET, DEFAULT_RECO)
    logger.info("docTR API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="docTR REST API",
    description="Full Mindee docTR feature exposure via HTTP REST API.",
    version="1.0.0",
    lifespan=lifespan,
)
# Compresse les réponses JSON volumineuses (si le client envoie Accept-Encoding: gzip).
app.add_middleware(GZipMiddleware, minimum_size=1024)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    gpu_info = {}
    if DEVICE.type == "cuda":
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
        }
    return {"status": "ok", "device": str(DEVICE), **gpu_info}


@app.get("/models", tags=["system"])
def list_models():
    """List all available detection and recognition model architectures."""
    return {
        "detection": DETECTION_MODELS,
        "recognition": RECOGNITION_MODELS,
        "device": str(DEVICE),
        "defaults": {"detection": DEFAULT_DET, "recognition": DEFAULT_RECO},
        "loaded": {
            "ocr": list(_ocr_cache.keys()),
            "detection": list(_det_cache.keys()),
            "recognition": list(_reco_cache.keys()),
            "kie": list(_kie_cache.keys()),
        },
    }


# ── OCR ──────────────────────────────────────────────────────────────────────

@app.post("/ocr", tags=["ocr"])
def ocr(
    file: UploadFile = File(..., description="Image (JPEG/PNG) or PDF"),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
    export_as_straight_boxes: bool = Query(False, description="Return axis-aligned boxes"),
    include_page_images: bool = Query(False, description="Base64-encode each page image"),
):
    """
    Full end-to-end OCR pipeline.
    Returns the complete document structure: pages → blocks → lines → words,
    each with bounding box, confidence score and text value.
    """
    t0 = time.perf_counter()
    try:
        pages = _load_document(file)
    except Exception as e:
        logger.exception("OCR – failed to load document (filename=%s)", file.filename)
        raise HTTPException(400, f"Could not read file: {e}")

    try:
        predictor = get_ocr_predictor(det_arch, reco_arch)
        doc = _infer(predictor, pages)
    except Exception as e:
        logger.exception(
            "OCR – inference failed (det=%s, reco=%s, num_pages=%d, device=%s)",
            det_arch, reco_arch, len(pages), DEVICE,
        )
        raise HTTPException(500, f"OCR inference failed: {e}")

    try:
        result = _doc_to_dict(doc)
    except Exception as e:
        logger.exception("OCR – failed to serialise results (det=%s, reco=%s)", det_arch, reco_arch)
        raise HTTPException(500, f"Result serialisation failed: {e}")

    result["meta"] = {
        "det_arch": det_arch,
        "reco_arch": reco_arch,
        "num_pages": len(pages),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }

    if include_page_images:
        result["page_images_b64"] = [
            base64.b64encode(
                _numpy_to_png_bytes(p)
            ).decode()
            for p in pages
        ]

    logger.info("OCR – done in %.3fs (det=%s, reco=%s, pages=%d)", time.perf_counter() - t0, det_arch, reco_arch, len(pages))
    return result


# ── Detection ─────────────────────────────────────────────────────────────────

@app.post("/detect", tags=["detection"])
def detect(
    file: UploadFile = File(..., description="Image (JPEG/PNG) or PDF"),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    include_confidence: bool = Query(True),
):
    """
    Text **detection** only – locates word/text regions and returns bounding boxes.
    """
    t0 = time.perf_counter()
    try:
        pages = _load_document(file)
    except Exception as e:
        logger.exception("Detect – failed to load document (filename=%s)", file.filename)
        raise HTTPException(400, f"Could not read file: {e}")

    try:
        predictor = get_det_predictor(det_arch)
        result = _infer(predictor, pages)
    except Exception as e:
        logger.exception("Detect – inference failed (det=%s, num_pages=%d, device=%s)", det_arch, len(pages), DEVICE)
        raise HTTPException(500, f"Detection inference failed: {e}")

    try:
        # docTR >= 0.8 : une liste (une entrée par page) de {class_name: ndarray}.
        # Boîtes droites : (N, 5) = xmin, ymin, xmax, ymax, score (coordonnées relatives)
        # Boîtes orientées : (N, 5, 2) = 4 points + ligne score.
        pages_out = []
        for page_idx, page_result in enumerate(result):
            regions = []
            for class_name, boxes in page_result.items():
                for box in np.asarray(boxes):
                    if box.ndim == 1:
                        geometry = ((float(box[0]), float(box[1])), (float(box[2]), float(box[3])))
                        score = float(box[4])
                    else:
                        geometry = tuple((float(x), float(y)) for x, y in box[:4])
                        score = float(box[4][0])
                    region: dict = {"geometry": geometry, "class": class_name}
                    if include_confidence:
                        region["confidence"] = round(score, 4)
                    regions.append(region)
            pages_out.append({"page_idx": page_idx, "regions": regions})
    except Exception as e:
        logger.exception("Detect – failed to serialise results (det=%s)", det_arch)
        raise HTTPException(500, f"Result serialisation failed: {e}")

    logger.info("Detect – done in %.3fs (det=%s, pages=%d)", time.perf_counter() - t0, det_arch, len(pages))
    return {
        "det_arch": det_arch,
        "num_pages": len(pages),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "pages": pages_out,
    }


# ── Recognition ──────────────────────────────────────────────────────────────

@app.post("/recognize", tags=["recognition"])
def recognize(
    file: UploadFile = File(..., description="Cropped word image (JPEG/PNG)"),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
):
    """
    Text **recognition** only – expects a tightly-cropped word image.
    Returns the recognised string and confidence.
    """
    t0 = time.perf_counter()
    try:
        data = file.file.read()
        img = _load_image_bytes(data)
    except Exception as e:
        logger.exception("Recognize – failed to load image (filename=%s, size=%d bytes)", file.filename, len(data) if 'data' in dir() else -1)
        raise HTTPException(400, f"Could not read image: {e}")

    try:
        predictor = get_reco_predictor(reco_arch)
        preds = _infer(predictor, [img])  # docTR 0.10 : [(value, confidence), ...]
    except Exception as e:
        logger.exception("Recognize – inference failed (reco=%s, img_shape=%s, device=%s)", reco_arch, img.shape, DEVICE)
        raise HTTPException(500, f"Recognition inference failed: {e}")

    logger.info("Recognize – done in %.3fs (reco=%s)", time.perf_counter() - t0, reco_arch)
    return {
        "reco_arch": reco_arch,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "results": [
            {"value": w, "confidence": round(float(c), 4)}
            for w, c in preds
        ],
    }


# ── KIE (Key Information Extraction) ─────────────────────────────────────────

@app.post("/kie", tags=["kie"])
def kie(
    file: UploadFile = File(..., description="Image (JPEG/PNG) or PDF"),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
):
    """
    **Key Information Extraction** – uses the KIE predictor for multi-class
    detection and returns class-labelled word regions.
    """
    t0 = time.perf_counter()
    try:
        pages = _load_document(file)
    except Exception as e:
        logger.exception("KIE – failed to load document (filename=%s)", file.filename)
        raise HTTPException(400, f"Could not read file: {e}")

    try:
        predictor = get_kie_predictor(det_arch, reco_arch)
        doc = _infer(predictor, pages)
    except Exception as e:
        logger.exception(
            "KIE – inference failed (det=%s, reco=%s, num_pages=%d, device=%s)",
            det_arch, reco_arch, len(pages), DEVICE,
        )
        raise HTTPException(500, f"KIE inference failed: {e}")

    try:
        pages_out = []
        for page in doc.pages:
            predictions: dict[str, list] = {}
            for class_name, words in page.predictions.items():
                predictions[class_name] = [_word_to_dict(w) for w in words]
            pages_out.append({"page_idx": page.page_idx, "predictions": predictions})
    except Exception as e:
        logger.exception("KIE – failed to serialise results (det=%s, reco=%s)", det_arch, reco_arch)
        raise HTTPException(500, f"Result serialisation failed: {e}")

    logger.info("KIE – done in %.3fs (det=%s, reco=%s, pages=%d)", time.perf_counter() - t0, det_arch, reco_arch, len(pages))
    return {
        "det_arch": det_arch,
        "reco_arch": reco_arch,
        "num_pages": len(pages),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "pages": pages_out,
    }


# ── Export: plain text ────────────────────────────────────────────────────────

@app.post("/export/text", tags=["export"])
def export_text(
    file: UploadFile = File(...),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
):
    """
    Run OCR and return a plain-text representation of the document,
    preserving approximate layout.
    """
    try:
        pages = _load_document(file)
    except Exception as e:
        logger.exception("Export/text – failed to load document (filename=%s)", file.filename)
        raise HTTPException(400, f"Could not read file: {e}")

    try:
        predictor = get_ocr_predictor(det_arch, reco_arch)
        doc = _infer(predictor, pages)
        text = doc.render()
    except Exception as e:
        logger.exception(
            "Export/text – inference failed (det=%s, reco=%s, num_pages=%d, device=%s)",
            det_arch, reco_arch, len(pages), DEVICE,
        )
        raise HTTPException(500, f"Text export failed: {e}")

    logger.info("Export/text – done (det=%s, reco=%s, pages=%d)", det_arch, reco_arch, len(pages))
    return Response(content=text, media_type="text/plain")


# ── Export: hOCR (XML) ────────────────────────────────────────────────────────

@app.post("/export/hocr", tags=["export"])
def export_hocr(
    file: UploadFile = File(...),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
):
    """
    Run OCR and return an **hOCR** (XML) document – compatible with Tesseract
    tooling and searchable-PDF generators.
    """
    try:
        pages = _load_document(file)
    except Exception as e:
        logger.exception("Export/hOCR – failed to load document (filename=%s)", file.filename)
        raise HTTPException(400, f"Could not read file: {e}")

    try:
        predictor = get_ocr_predictor(det_arch, reco_arch)
        doc = _infer(predictor, pages)
        hocr = doc.export_as_xml()
    except Exception as e:
        logger.exception(
            "Export/hOCR – inference failed (det=%s, reco=%s, num_pages=%d, device=%s)",
            det_arch, reco_arch, len(pages), DEVICE,
        )
        raise HTTPException(500, f"hOCR export failed: {e}")

    logger.info("Export/hOCR – done (det=%s, reco=%s, pages=%d)", det_arch, reco_arch, len(pages))
    return Response(content=hocr, media_type="application/xml")


# ── Export: searchable PDF ────────────────────────────────────────────────────

@app.post("/export/searchable-pdf", tags=["export"])
def export_searchable_pdf(
    file: UploadFile = File(...),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
):
    """
    Run OCR on a PDF and return a **searchable PDF** with hidden text layer.
    Input must be a PDF.
    """
    try:
        pages = _load_document(file)
        # Re-read bytes for the original PDF to embed
        file.file.seek(0)
        pdf_bytes = file.file.read()
    except Exception as e:
        logger.exception("Export/searchable-pdf – failed to load document (filename=%s)", file.filename)
        raise HTTPException(400, f"Could not read file: {e}")

    try:
        predictor = get_ocr_predictor(det_arch, reco_arch)
        doc = _infer(predictor, pages)
    except Exception as e:
        logger.exception(
            "Export/searchable-pdf – inference failed (det=%s, reco=%s, num_pages=%d, device=%s)",
            det_arch, reco_arch, len(pages), DEVICE,
        )
        raise HTTPException(500, f"OCR inference failed: {e}")

    try:
        out_pdf = doc.export_as_pdf(pdf_bytes)
    except Exception as e:
        logger.exception(
            "Export/searchable-pdf – PDF generation failed (det=%s, reco=%s, num_pages=%d)",
            det_arch, reco_arch, len(pages),
        )
        raise HTTPException(500, f"PDF export failed: {e}")

    logger.info("Export/searchable-pdf – done (det=%s, reco=%s, pages=%d)", det_arch, reco_arch, len(pages))
    return Response(
        content=out_pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=searchable.pdf"},
    )


# ── Visualise ─────────────────────────────────────────────────────────────────

@app.post("/visualize", tags=["debug"])
def visualize(
    file: UploadFile = File(..., description="Single-page image (JPEG/PNG)"),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
    page_idx: int = Query(0, ge=0, description="Page to render (0-indexed)"),
):
    """
    Run OCR and return an annotated **PNG image** with bounding boxes and text overlaid.
    """
    try:
        pages = _load_document(file)
    except Exception as e:
        logger.exception("Visualize – failed to load document (filename=%s)", file.filename)
        raise HTTPException(400, f"Could not read file: {e}")

    if page_idx >= len(pages):
        raise HTTPException(400, f"page_idx {page_idx} out of range (doc has {len(pages)} pages)")

    try:
        predictor = get_ocr_predictor(det_arch, reco_arch)
        doc = _infer(predictor, pages)
    except Exception as e:
        logger.exception(
            "Visualize – inference failed (det=%s, reco=%s, num_pages=%d, device=%s)",
            det_arch, reco_arch, len(pages), DEVICE,
        )
        raise HTTPException(500, f"OCR inference failed: {e}")

    page = doc.pages[page_idx]
    try:
        from doctr.utils.visualization import visualize_page
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = visualize_page(page.export(), pages[page_idx], interactive=False, add_labels=True)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        logger.info("Visualize – done (det=%s, reco=%s, page_idx=%d)", det_arch, reco_arch, page_idx)
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        logger.exception(
            "Visualize – rendering failed (det=%s, reco=%s, page_idx=%d)",
            det_arch, reco_arch, page_idx,
        )
        raise HTTPException(500, f"Visualization failed: {e}")


# ── Debug / diagnostic ────────────────────────────────────────────────────────

@app.post("/debug/pdf", tags=["debug"])
def debug_pdf(
    file: UploadFile = File(..., description="PDF to diagnose"),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
):
    """Step-by-step PDF processing diagnostic – returns where it fails."""
    import traceback as tb

    report: dict = {"filename": file.filename, "steps": {}}
    data = file.file.read()
    report["file_size_bytes"] = len(data)

    # Step 1 – pdftoppm render
    try:
        t0 = time.perf_counter()
        raw = _read_pdf_safe(data)
        report["steps"]["1_pdftoppm"] = {
            "status": "ok",
            "num_pages": len(raw),
            "page_shapes": [p.shape for p in raw],
            "page_dtypes": [str(p.dtype) for p in raw],
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        report["steps"]["1_pdftoppm"] = {"status": "FAILED", "error": tb.format_exc()}
        return report

    # Step 2 – RGB conversion + resolution cap
    try:
        t0 = time.perf_counter()
        pages = [_cap_resolution(_ensure_rgb(p)) for p in raw]
        report["steps"]["2_rgb_cap"] = {
            "status": "ok",
            "page_shapes": [p.shape for p in pages],
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        report["steps"]["2_rgb_cap"] = {"status": "FAILED", "error": tb.format_exc()}
        return report

    # Step 3 – load model
    try:
        t0 = time.perf_counter()
        predictor = get_ocr_predictor(det_arch, reco_arch)
        report["steps"]["3_model_load"] = {
            "status": "ok",
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        report["steps"]["3_model_load"] = {"status": "FAILED", "error": tb.format_exc()}
        return report

    # Step 4 – inference
    try:
        t0 = time.perf_counter()
        doc = _infer(predictor, pages)
        report["steps"]["4_inference"] = {
            "status": "ok",
            "num_words": sum(
                len(w)
                for p in doc.pages for b in p.blocks for l in b.lines for w in [l.words]
            ),
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        report["steps"]["4_inference"] = {"status": "FAILED", "error": tb.format_exc()}
        return report

    # Step 5 – serialisation
    try:
        t0 = time.perf_counter()
        result = _doc_to_dict(doc)
        report["steps"]["5_serialise"] = {
            "status": "ok",
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
    except Exception as e:
        report["steps"]["5_serialise"] = {"status": "FAILED", "error": tb.format_exc()}
        return report

    report["overall"] = "ALL STEPS OK"
    return report


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
