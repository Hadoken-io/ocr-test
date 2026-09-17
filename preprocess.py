"""Image preprocessing utilities for OCR pipeline."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load an image from disk using OpenCV."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image from: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert image to grayscale."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def remove_noise(img: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur to reduce noise."""
    return cv2.GaussianBlur(img, (3, 3), 0)


def threshold_image(img: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding to binarize the image."""
    return cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )


def dilate_image(img: np.ndarray) -> np.ndarray:
    """Apply dilation to thicken text for better detection."""
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def erode_image(img: np.ndarray) -> np.ndarray:
    """Apply erosion to thin out noise."""
    kernel = np.ones((2, 2), np.uint8)
    return cv2.erode(img, kernel, iterations=1)


def deskew_image(img: np.ndarray) -> np.ndarray:
    """Correct image skew by rotating to align text horizontally."""
    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def resize_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale image to improve OCR accuracy."""
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def canny_edge(img: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection (optional, for debugging)."""
    return cv2.Canny(img, 50, 150)


def preprocess(
    img: np.ndarray,
    *,
    denoise: bool = True,
    resize: bool = True,
    scale: float = 2.0,
    deskew: bool = True,
    dilate: bool = False,
    canny: bool = False,
) -> np.ndarray:
    """Run the full preprocessing pipeline on a BGR image.

    Args:
        img: Input BGR image as numpy array.
        denoise: Apply Gaussian blur to reduce noise.
        resize: Upscale the image for better OCR accuracy.
        scale: Scaling factor when resizing.
        deskew: Correct image skew.
        dilate: Apply dilation after thresholding.
        canny: Return Canny edges instead of thresholded image.

    Returns:
        Preprocessed grayscale image ready for OCR.
    """
    gray = to_grayscale(img)

    if denoise:
        gray = remove_noise(gray)

    if resize:
        gray = resize_image(gray, scale=scale)

    if deskew:
        gray = deskew_image(gray)

    if canny:
        return canny_edge(gray)

    thresholded = threshold_image(gray)

    if dilate:
        thresholded = dilate_image(thresholded)

    return thresholded
