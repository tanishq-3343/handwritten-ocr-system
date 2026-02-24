"""
ocr_engine.py
-------------
Wraps Tesseract OCR and EasyOCR, exposing a unified ensemble interface.
"""

import re
from PIL import Image
import numpy as np
import pytesseract

# Configure Tesseract binary path (Linux Colab default)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# EasyOCR is imported lazily so the module still loads without it
_easy_reader = None
EASYOCR_AVAILABLE = False


def init_easyocr(languages: list[str] = None, gpu: bool = False) -> bool:
    """
    Initialise the EasyOCR reader. Call once before running OCR.

    Returns True if initialisation succeeded, False otherwise.
    """
    global _easy_reader, EASYOCR_AVAILABLE
    if languages is None:
        languages = ["en"]
    try:
        import easyocr
        _easy_reader = easyocr.Reader(languages, gpu=gpu, verbose=False)
        EASYOCR_AVAILABLE = True
        print("✅ EasyOCR initialized")
        return True
    except Exception as exc:
        print(f"⚠️  EasyOCR not available: {exc}. Falling back to Tesseract only.")
        EASYOCR_AVAILABLE = False
        return False


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_OCR_ARTIFACT_RE = re.compile(r"[|\\^~`]")
_WHITESPACE_RE = re.compile(r"\s+")

def clean_ocr_text(text: str) -> str:
    """Normalise OCR output for fair metric comparison."""
    text = text.strip().lower()
    text = _WHITESPACE_RE.sub(" ", text)
    text = _OCR_ARTIFACT_RE.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Individual engines
# ---------------------------------------------------------------------------

_TESS_CONFIG = (
    "--psm 7 --oem 3 "
    '-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?\\' "'
)


def ocr_tesseract(img_np: np.ndarray) -> str:
    """Run Tesseract with single-line PSM on a preprocessed image."""
    pil_img = Image.fromarray(img_np)
    raw = pytesseract.image_to_string(pil_img, config=_TESS_CONFIG)
    return clean_ocr_text(raw)


def ocr_easyocr(img_np: np.ndarray) -> str:
    """Run EasyOCR on the original (colour) image."""
    if not EASYOCR_AVAILABLE or _easy_reader is None:
        return ""
    results = _easy_reader.readtext(img_np, detail=0, paragraph=True)
    return clean_ocr_text(" ".join(results))


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_ocr(
    preprocessed_img: np.ndarray,
    original_img: np.ndarray,
) -> tuple[str, str, str, str]:
    """
    Run both engines and pick the better result.

    Strategy: prefer EasyOCR when its output is at least 70 % as long as
    Tesseract's (indicating it found real content); otherwise use Tesseract.

    Returns
    -------
    (final_prediction, tesseract_prediction, easyocr_prediction, method_used)
    """
    tess_result = ocr_tesseract(preprocessed_img)
    easy_result = ocr_easyocr(original_img)

    if EASYOCR_AVAILABLE and len(easy_result) >= len(tess_result) * 0.7:
        return easy_result, tess_result, easy_result, "EasyOCR"
    return tess_result, tess_result, easy_result, "Tesseract"
