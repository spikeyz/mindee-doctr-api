"""
Mindee docTR REST API
Exposes all docTR features: OCR, detection, recognition, KIE, and export.
"""
import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry – loaded once and cached per (det_arch, reco_arch) pair
# ---------------------------------------------------------------------------
from doctr.models import detection_predictor, ocr_predictor, recognition_predictor

_ocr_cache: dict = {}
_det_cache: dict = {}
_reco_cache: dict = {}

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


def get_ocr_predictor(det_arch: str, reco_arch: str):
    key = (det_arch, reco_arch)
    if key not in _ocr_cache:
        logger.info("Loading OCR predictor (%s + %s) …", det_arch, reco_arch)
        _ocr_cache[key] = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
    return _ocr_cache[key]


def get_det_predictor(det_arch: str):
    if det_arch not in _det_cache:
        logger.info("Loading detection predictor (%s) …", det_arch)
        _det_cache[det_arch] = detection_predictor(arch=det_arch, pretrained=True)
    return _det_cache[det_arch]


def get_reco_predictor(reco_arch: str):
    if reco_arch not in _reco_cache:
        logger.info("Loading recognition predictor (%s) …", reco_arch)
        _reco_cache[reco_arch] = recognition_predictor(arch=reco_arch, pretrained=True)
    return _reco_cache[reco_arch]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_image_bytes(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)


def _load_document(file: UploadFile) -> list[np.ndarray]:
    """Return a list of numpy arrays (one per page) from image or PDF upload."""
    from doctr.io import DocumentFile
    data = file.file.read()
    ct = file.content_type or ""
    name = (file.filename or "").lower()
    if ct == "application/pdf" or name.endswith(".pdf"):
        return DocumentFile.from_pdf(data)
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.get("/models", tags=["system"])
def list_models():
    """List all available detection and recognition model architectures."""
    return {
        "detection": DETECTION_MODELS,
        "recognition": RECOGNITION_MODELS,
        "defaults": {"detection": DEFAULT_DET, "recognition": DEFAULT_RECO},
        "loaded": {
            "ocr": list(_ocr_cache.keys()),
            "detection": list(_det_cache.keys()),
            "recognition": list(_reco_cache.keys()),
        },
    }


# ── OCR ──────────────────────────────────────────────────────────────────────

@app.post("/ocr", tags=["ocr"])
async def ocr(
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
        raise HTTPException(400, f"Could not read file: {e}")

    predictor = get_ocr_predictor(det_arch, reco_arch)
    doc = predictor(pages)

    result = _doc_to_dict(doc)
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

    return result


# ── Detection ─────────────────────────────────────────────────────────────────

@app.post("/detect", tags=["detection"])
async def detect(
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
        raise HTTPException(400, f"Could not read file: {e}")

    predictor = get_det_predictor(det_arch)
    result = predictor(pages)

    pages_out = []
    for page_idx, page_result in enumerate(result.pages):
        regions = []
        for block in page_result.blocks:
            region: dict = {"geometry": block.geometry}
            if include_confidence:
                region["confidence"] = round(float(block.objectness_score), 4)
            regions.append(region)
        pages_out.append({"page_idx": page_idx, "regions": regions})

    return {
        "det_arch": det_arch,
        "num_pages": len(pages),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "pages": pages_out,
    }


# ── Recognition ──────────────────────────────────────────────────────────────

@app.post("/recognize", tags=["recognition"])
async def recognize(
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
        raise HTTPException(400, f"Could not read image: {e}")

    predictor = get_reco_predictor(reco_arch)
    words, confs = predictor([img])

    return {
        "reco_arch": reco_arch,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "results": [
            {"value": w, "confidence": round(float(c), 4)}
            for w, c in zip(words, confs)
        ],
    }


# ── KIE (Key Information Extraction) ─────────────────────────────────────────

@app.post("/kie", tags=["kie"])
async def kie(
    file: UploadFile = File(..., description="Image (JPEG/PNG) or PDF"),
    det_arch: str = Query(DEFAULT_DET, enum=DETECTION_MODELS),
    reco_arch: str = Query(DEFAULT_RECO, enum=RECOGNITION_MODELS),
):
    """
    **Key Information Extraction** – uses the KIE predictor for multi-class
    detection and returns class-labelled word regions.
    """
    from doctr.models import kie_predictor

    t0 = time.perf_counter()
    try:
        pages = _load_document(file)
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")

    predictor = kie_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
    doc = predictor(pages)

    pages_out = []
    for page in doc.pages:
        predictions: dict[str, list] = {}
        for class_name, words in page.predictions.items():
            predictions[class_name] = [_word_to_dict(w) for w in words]
        pages_out.append({"page_idx": page.page_idx, "predictions": predictions})

    return {
        "det_arch": det_arch,
        "reco_arch": reco_arch,
        "num_pages": len(pages),
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "pages": pages_out,
    }


# ── Export: plain text ────────────────────────────────────────────────────────

@app.post("/export/text", tags=["export"])
async def export_text(
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
        raise HTTPException(400, f"Could not read file: {e}")

    predictor = get_ocr_predictor(det_arch, reco_arch)
    doc = predictor(pages)
    text = doc.render()
    return Response(content=text, media_type="text/plain")


# ── Export: hOCR (XML) ────────────────────────────────────────────────────────

@app.post("/export/hocr", tags=["export"])
async def export_hocr(
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
        raise HTTPException(400, f"Could not read file: {e}")

    predictor = get_ocr_predictor(det_arch, reco_arch)
    doc = predictor(pages)
    hocr = doc.export_as_xml()
    return Response(content=hocr, media_type="application/xml")


# ── Export: searchable PDF ────────────────────────────────────────────────────

@app.post("/export/searchable-pdf", tags=["export"])
async def export_searchable_pdf(
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
        raise HTTPException(400, f"Could not read file: {e}")

    predictor = get_ocr_predictor(det_arch, reco_arch)
    doc = predictor(pages)

    try:
        out_pdf = doc.export_as_pdf(pdf_bytes)
    except Exception as e:
        raise HTTPException(500, f"PDF export failed: {e}")

    return Response(
        content=out_pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=searchable.pdf"},
    )


# ── Visualise ─────────────────────────────────────────────────────────────────

@app.post("/visualize", tags=["debug"])
async def visualize(
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
        raise HTTPException(400, f"Could not read file: {e}")

    if page_idx >= len(pages):
        raise HTTPException(400, f"page_idx {page_idx} out of range (doc has {len(pages)} pages)")

    predictor = get_ocr_predictor(det_arch, reco_arch)
    doc = predictor(pages)

    # docTR's show() renders to matplotlib; use export instead
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
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
