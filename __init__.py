# Methodology

## 1. Dataset

Since real handwritten datasets (IAM, CVL) require institutional access or large downloads, this project generates **synthetic handwriting images** using Python's PIL library. Each image simulates one of three styles:

| Style | Description | Noise Level |
|---|---|---|
| Neat | Clean, evenly spaced printing | Low (σ=5) |
| Mixed | Combination of print and semi-cursive | Medium (σ=8) |
| Cursive | Connected letters with slight rotation | High (σ=12) |

A total of **15 samples** (5 per style) are generated from curated phrases.

---

## 2. Preprocessing Pipeline

Each image passes through a deterministic 5-step pipeline before being fed to the OCR engines:

### Step 1 — Grayscale Conversion
RGB → single-channel luminance using OpenCV's `COLOR_RGB2GRAY`.

### Step 2 — Gaussian Blur
A 3×3 kernel with σ=0 (auto) suppresses high-frequency pixel noise without blurring character strokes.

### Step 3 — Adaptive Thresholding
`cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C`, block size 15, and constant C=8 handles uneven illumination far better than global Otsu thresholding.

### Step 4 — Morphological Opening
A 2×2 rectangular structuring element removes isolated salt-and-pepper speckles while preserving thin strokes.

### Step 5 — Deskewing
`cv2.minAreaRect` on foreground pixels estimates the dominant text angle. Rotations within ±10° are corrected; larger angles are left unchanged to avoid flipping the image.

---

## 3. OCR Engines

### Tesseract OCR (v5)
- **Page Segmentation Mode (PSM 7):** treats the image as a single text line — optimal for our single-line samples
- **OEM 3:** LSTM + legacy engine
- Configured character whitelist to reduce symbol confusion

### EasyOCR
- Uses a **CRNN + LSTM** architecture with a CTC decoder
- Language model: English
- GPU disabled (Colab CPU runtime)

### Ensemble Strategy
For each sample, both engines run independently. EasyOCR output is preferred if its length is ≥ 70% of Tesseract's output length (indicating the engine found real content). Otherwise Tesseract wins. This simple heuristic reduces the impact of either engine returning empty or garbage output.

---

## 4. Evaluation Metrics

| Metric | Formula | Range | Lower is better? |
|---|---|---|---|
| CER | edit_distance(ref_chars, hyp_chars) / len(ref_chars) | [0, 1] | ✅ Yes |
| WER | edit_distance(ref_words, hyp_words) / len(ref_words) | [0, 1] | ✅ Yes |
| Char Accuracy | 1 − CER | [0, 1] | ❌ No (higher = better) |
| Word Accuracy | 1 − WER | [0, 1] | ❌ No |
| Exact Match | 1 if pred == ref else 0 | {0, 1} | ❌ No |

Edit distance is computed using the `editdistance` library (C extension), which runs in O(n·m) time.

---

## 5. Error Analysis

A DP traceback on the character-level edit distance matrix classifies each edit as:
- **Substitution**: reference char replaced by a different hypothesis char
- **Insertion**: extra char in hypothesis not in reference
- **Deletion**: reference char missing from hypothesis

The top-N errors per category are plotted as horizontal bar charts.

---

## 6. Known Limitations

- Synthetic images don't fully capture real handwriting variability (pen pressure, baseline drift, ligatures)
- EasyOCR model download may fail in restricted network environments
- Ensemble heuristic (length-based) is a proxy for true confidence — a calibrated confidence score would be better
- CTC decoding without a language model is more error-prone for OOV words
