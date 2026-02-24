"""
preprocessing.py
----------------
5-step image preprocessing pipeline:
  1. Grayscale conversion
  2. Gaussian denoising
  3. Adaptive thresholding (binarization)
  4. Morphological opening (speckle removal)
  5. Deskewing via minAreaRect
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image(img_np: np.ndarray, show_steps: bool = False) -> np.ndarray:
    """
    Run the full preprocessing pipeline on an RGB image.

    Parameters
    ----------
    img_np      : Input image as np.ndarray (H, W, 3) in RGB colour space.
    show_steps  : If True, display each intermediate step with Matplotlib.

    Returns
    -------
    np.ndarray  : Cleaned binary (grayscale) image ready for OCR.
    """
    steps: dict[str, np.ndarray] = {"Original": img_np.copy()}

    # 1 ── Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    steps["Grayscale"] = gray

    # 2 ── Gaussian blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    steps["Denoised"] = blurred

    # 3 ── Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15,
        C=8,
    )
    steps["Binarized"] = binary

    # 4 ── Morphological opening (removes small speckles)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps["Cleaned"] = cleaned

    # 5 ── Deskew
    coords = np.column_stack(np.where(cleaned < 128))
    if len(coords) > 10:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) < 10:  # only fix small skews to avoid flipping
            h, w = cleaned.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            cleaned = cv2.warpAffine(
                cleaned, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
    steps["Deskewed"] = cleaned

    # Optional visualisation
    if show_steps:
        _plot_steps(steps)

    return cleaned


def _plot_steps(steps: dict) -> None:
    """Render each preprocessing step side-by-side."""
    fig, axes = plt.subplots(1, len(steps), figsize=(18, 3))
    for ax, (name, img) in zip(axes, steps.items()):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle("🔧 Preprocessing Pipeline Steps", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
