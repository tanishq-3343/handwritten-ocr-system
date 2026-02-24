"""
dataset_generator.py
--------------------
Generates synthetic handwriting images simulating neat, cursive,
and mixed styles using PIL + OpenCV augmentation.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Sample texts per style
# ---------------------------------------------------------------------------
SAMPLE_TEXTS = {
    "neat": [
        "The quick brown fox jumps over the lazy dog.",
        "Handwriting recognition is a challenging task.",
        "Machine learning enables computers to read text.",
        "Optical character recognition converts images to text.",
        "Deep learning models achieve high accuracy on OCR tasks.",
    ],
    "cursive": [
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck.",
        "Peter Piper picked a peck of pickled peppers.",
        "All good things must come to an end.",
        "A stitch in time saves nine.",
    ],
    "mixed": [
        "OCR accuracy depends on image quality and preprocessing.",
        "Feature extraction plays a vital role in recognition.",
        "Neural networks learn patterns from large datasets.",
        "Data augmentation improves model generalization.",
        "Evaluation metrics include CER and WER scores.",
    ],
}

# Style-specific rendering parameters
_STYLE_PARAMS = {
    "neat":    {"bg": (255, 253, 240), "color": (20, 20, 80),  "size": 22, "spacing": 4},
    "cursive": {"bg": (240, 248, 255), "color": (60, 10, 90),  "size": 24, "spacing": 6},
    "mixed":   {"bg": (255, 253, 240), "color": (10, 60, 30),  "size": 21, "spacing": 4},
}

# Fallback font search paths (Linux / macOS)
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a system TrueType font; fall back to PIL default."""
    for path in _FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def generate_handwriting_image(
    text: str,
    style: str = "neat",
    img_w: int = 640,
    img_h: int = 100,
    noise_level: int = 0,
) -> np.ndarray:
    """
    Render *text* as a synthetic handwriting-like image.

    Parameters
    ----------
    text        : String to render.
    style       : One of ``"neat"``, ``"cursive"``, ``"mixed"``.
    img_w, img_h: Image dimensions in pixels.
    noise_level : Standard deviation of Gaussian pixel noise (0 = none).

    Returns
    -------
    np.ndarray  : RGB image as uint8 array of shape (img_h, img_w, 3).
    """
    params = _STYLE_PARAMS.get(style, _STYLE_PARAMS["neat"])

    img = Image.new("RGB", (img_w, img_h), color=params["bg"])
    draw = ImageDraw.Draw(img)
    font = _load_font(params["size"])

    # Ruled lines (notebook feel)
    for y_line in range(30, img_h, 30):
        draw.line([(0, y_line), (img_w, y_line)], fill=(180, 220, 240), width=1)

    # Baseline with optional cursive wave
    base_y = img_h // 2 - params["size"] // 2
    if style == "cursive":
        base_y += int(3 * np.sin(np.pi))

    draw.text((12, base_y), text, font=font, fill=params["color"], spacing=params["spacing"])

    img_np = np.array(img)

    # Gaussian pixel noise
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight random rotation for non-neat styles
    if style in ("cursive", "mixed"):
        angle = np.random.uniform(-1.5, 1.5)
        h, w = img_np.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img_np = cv2.warpAffine(img_np, M, (w, h), borderValue=params["bg"])

    return img_np


def build_dataset(noise_map: dict | None = None) -> list[dict]:
    """
    Build the full synthetic dataset from ``SAMPLE_TEXTS``.

    Parameters
    ----------
    noise_map : dict mapping style → noise_level.
                Defaults to ``{"neat": 5, "cursive": 12, "mixed": 8}``.

    Returns
    -------
    List of dicts with keys ``image``, ``ground_truth``, ``style``.
    """
    if noise_map is None:
        noise_map = {"neat": 5, "cursive": 12, "mixed": 8}

    dataset = []
    for style, texts in SAMPLE_TEXTS.items():
        for text in texts:
            img = generate_handwriting_image(text, style=style, noise_level=noise_map[style])
            dataset.append({"image": img, "ground_truth": text, "style": style})
    return dataset
