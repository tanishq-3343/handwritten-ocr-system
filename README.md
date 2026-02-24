[README.md](https://github.com/user-attachments/files/25510403/README.md)
#  Handwritten OCR System

> A comprehensive Optical Character Recognition pipeline for handwritten text recognition from scanned documents — supporting multiple handwriting styles with an intelligent ensemble approach.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Local-orange)
![OCR](https://img.shields.io/badge/OCR-Tesseract%20%2B%20EasyOCR-purple)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🔍 Overview

This project implements an end-to-end handwritten OCR system that:

- Recognizes handwritten text across **three styles**: neat, cursive, and mixed
- Uses an **ensemble of two OCR engines**: Tesseract OCR and EasyOCR
- Applies a full **image preprocessing pipeline** to maximize recognition accuracy
- Evaluates performance using industry-standard metrics (CER, WER)
- Provides rich **visualizations** and detailed error analysis

---

## ✨ Features

| Feature | Description |
|---|---|
| 🖼️ Synthetic Dataset Generator | Creates realistic handwriting images with style-specific noise and distortions |
| 🔧 Preprocessing Pipeline | Grayscale → Denoise → Binarize → Morphological cleanup → Deskew |
| 🤖 Dual OCR Engines | Tesseract (PSM 7) + EasyOCR (CRNN+LSTM) |
| ⚡ Ensemble Strategy | Picks best output per sample via content-length heuristic |
| 📊 Metrics | CER, WER, Character Accuracy, Word Accuracy, Exact Match |
| 📈 Visualizations | Dashboard with 9 plots: bar charts, box plots, radar chart, heatmap, scatter |
| 🔍 Error Analysis | Character-level substitution, deletion, and insertion error breakdown |
| 🆚 Engine Comparison | Side-by-side Tesseract vs EasyOCR performance |
| 🧪 Interactive Test | Upload your own handwritten image for live OCR |

---

## 🏗️ Architecture

```
Input Image
     │
     ▼
┌─────────────────────────────┐
│    Preprocessing Pipeline    │
│  Grayscale → Blur → Binary  │
│  → Morphology → Deskew      │
└─────────────┬───────────────┘
              │
     ┌────────┴────────┐
     ▼                 ▼
┌─────────┐      ┌──────────┐
│Tesseract│      │ EasyOCR  │
│  PSM 7  │      │CRNN+LSTM │
└────┬────┘      └────┬─────┘
     └────────┬────────┘
              ▼
     ┌────────────────┐
     │ Ensemble Logic │  ← picks best output
     └───────┬────────┘
             ▼
      Final Prediction
             │
             ▼
     ┌───────────────┐
     │  Evaluation   │
     │  CER · WER    │
     │  Char/Word Acc│
     └───────────────┘
```

---

## 🛠️ Installation

### Option A — Google Colab (Recommended)

Open the notebook directly in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/handwritten-ocr-system/blob/main/notebooks/Handwritten_OCR_System.ipynb)

All dependencies install automatically in the first cell.

### Option B — Local Setup

**Prerequisites:** Python 3.10+, Tesseract OCR installed on your system.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/handwritten-ocr-system.git
cd handwritten-ocr-system

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Tesseract (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 🚀 Usage

### Run the Full Notebook

```bash
jupyter notebook notebooks/Handwritten_OCR_System.ipynb
```

### Use Individual Modules

```python
from src.preprocessing import preprocess_image
from src.ocr_engine import ensemble_ocr
from src.metrics import compute_cer, compute_wer
import cv2

# Load your image
img = cv2.imread("my_handwriting.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess
preprocessed = preprocess_image(img_rgb)

# Run OCR
prediction, tess_pred, easy_pred, method = ensemble_ocr(preprocessed, img_rgb)
print(f"Recognized text ({method}): {prediction}")

# Evaluate against ground truth
ground_truth = "The quick brown fox"
print(f"CER: {compute_cer(ground_truth, prediction):.3f}")
print(f"WER: {compute_wer(ground_truth, prediction):.3f}")
```

### Generate a Synthetic Dataset

```python
from src.dataset_generator import generate_handwriting_image, SAMPLE_TEXTS

img = generate_handwriting_image(
    text="Hello, world!",
    style="cursive",   # "neat" | "cursive" | "mixed"
    noise_level=10
)
```

---

## 📊 Results

### Overall Performance

| Metric | Score |
|---|---|
| Mean Character Accuracy | ~varies by run |
| Mean Word Accuracy | ~varies by run |
| Mean CER | ~varies by run |
| Mean WER | ~varies by run |

> ℹ️ Actual scores depend on OCR engine availability and runtime environment. Run the notebook to get precise numbers for your setup.

### Style-wise Breakdown

| Style | Char Accuracy | Word Accuracy | CER | WER |
|---|---|---|---|---|
| Neat | Highest | Highest | Lowest | Lowest |
| Mixed | Medium | Medium | Medium | Medium |
| Cursive | Challenging | Challenging | Highest | Highest |

### Key Findings

- ✅ **Neat** handwriting is most accurately recognized
- ⚠️ **Cursive** is the hardest — connected strokes confuse both engines
- 🔄 **Ensemble** outperforms single-engine baseline
- 🔧 **Preprocessing** significantly reduces noise artifacts
- 🔤 **Common confusions**: `l↔1`, `O↔0`, `rn↔m`, space insertion/deletion

---

## 📁 Project Structure

```
handwritten-ocr-system/
│
├── 📓 notebooks/
│   └── Handwritten_OCR_System.ipynb   # Main Colab notebook
│
├── 🐍 src/
│   ├── __init__.py
│   ├── dataset_generator.py           # Synthetic handwriting image generator
│   ├── preprocessing.py               # Image preprocessing pipeline
│   ├── ocr_engine.py                  # Tesseract + EasyOCR + ensemble logic
│   ├── metrics.py                     # CER, WER, accuracy functions
│   ├── error_analysis.py              # Character-level error breakdown
│   └── visualizations.py             # All plotting utilities
│
├── 🧪 tests/
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   └── test_ocr_engine.py
│
├── 📄 docs/
│   └── methodology.md                 # Detailed methodology writeup
│
├── 📊 results/                        # Generated plots (gitignored by default)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔭 Future Work

1. **Custom CRNN Model** — Train on real datasets (IAM, CVL, EMNIST)
2. **TrOCR Integration** — Microsoft's Transformer-based OCR for higher accuracy
3. **CTC Decoder** — Connectionist Temporal Classification for sequence alignment
4. **Language Model Post-processing** — Word beam search + spell correction
5. **Active Learning Loop** — Iteratively improve on hard samples
6. **REST API** — Wrap the pipeline in a FastAPI service
7. **Web UI** — Simple drag-and-drop interface for live OCR

---

## 📚 References

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- [TrOCR (Microsoft)](https://huggingface.co/docs/transformers/model_doc/trocr)
- [jiwer — WER/CER metrics](https://github.com/jitsi/jiwer)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ | Contributions welcome — open a PR or issue!</p>
