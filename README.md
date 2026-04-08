# docTR REST API

A Docker image that wraps [Mindee's docTR](https://github.com/mindee/doctr) library and exposes all its features through an HTTP REST API built with FastAPI.

## Features

- Full end-to-end OCR (image & PDF)
- Text detection only
- Text recognition only (from word crops)
- Key Information Extraction (KIE)
- Export as plain text, hOCR (XML), or searchable PDF
- Annotated visualisation output
- Runtime model selection across 9 detection and 8 recognition architectures
- Model weights pre-downloaded at build time for fast cold starts
- CPU and GPU (CUDA) support

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List available model architectures |
| POST | `/ocr` | End-to-end OCR — image or PDF |
| POST | `/detect` | Text detection only |
| POST | `/recognize` | Text recognition only (cropped word image) |
| POST | `/kie` | Key Information Extraction |
| POST | `/export/text` | OCR → plain text |
| POST | `/export/hocr` | OCR → hOCR XML |
| POST | `/export/searchable-pdf` | OCR → searchable PDF |
| POST | `/visualize` | OCR → annotated PNG |

Interactive documentation is available at `http://localhost:8000/docs` once the container is running.

## Quick start

### CPU (default)

```bash
docker compose build
docker compose up -d
```

### GPU (NVIDIA)

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

Uncomment the `args` block in `docker-compose.yml`, then:

```bash
docker compose build --build-arg PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
docker compose up -d
```

## Usage examples

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# End-to-end OCR on an image
curl -X POST http://localhost:8000/ocr \
  -F "file=@document.jpg"

# End-to-end OCR on a PDF with a different model pair
curl -X POST "http://localhost:8000/ocr?det_arch=fast_base&reco_arch=parseq" \
  -F "file=@document.pdf"

# Detection only
curl -X POST http://localhost:8000/detect \
  -F "file=@document.jpg"

# Recognition only (cropped word image)
curl -X POST http://localhost:8000/recognize \
  -F "file=@word_crop.png"

# Key Information Extraction
curl -X POST http://localhost:8000/kie \
  -F "file=@document.jpg"

# Export as plain text
curl -X POST http://localhost:8000/export/text \
  -F "file=@document.jpg"

# Export as hOCR XML
curl -X POST http://localhost:8000/export/hocr \
  -F "file=@document.pdf" -o output.xml

# Export as searchable PDF
curl -X POST http://localhost:8000/export/searchable-pdf \
  -F "file=@document.pdf" -o searchable.pdf

# Visualise with bounding boxes (returns PNG)
curl -X POST http://localhost:8000/visualize \
  -F "file=@document.jpg" -o annotated.png
```

## Available models

### Detection

| Architecture | Notes |
|---|---|
| `db_resnet34` | Lightweight DBNet |
| `db_resnet50` | **Default** – best accuracy/speed trade-off |
| `db_mobilenet_v3_large` | Mobile-friendly |
| `linknet_resnet18` | Fast |
| `linknet_resnet34` | |
| `linknet_resnet50` | |
| `fast_tiny` | Fastest |
| `fast_small` | |
| `fast_base` | |

### Recognition

| Architecture | Notes |
|---|---|
| `crnn_vgg16_bn` | **Default** |
| `crnn_mobilenet_v3_small` | Lightweight |
| `crnn_mobilenet_v3_large` | |
| `sar_resnet31` | |
| `master` | |
| `vitstr_small` | Vision Transformer |
| `vitstr_base` | |
| `parseq` | State-of-the-art accuracy |

## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `WORKERS` | `1` | Number of uvicorn workers |
| `DOCTR_CACHE_DIR` | `/app/.cache/doctr` | Model weight cache directory |

## Project structure

```
.
├── app/
│   └── main.py          # FastAPI application
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── requirements.txt
```
