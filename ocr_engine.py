"""OCR extraction engine using EasyOCR."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import easyocr
import numpy as np


@dataclass
class OCRResult:
    """Structured OCR output."""

    text: str
    confidence: float
    bbox: list[list[int]]


class OCREngine:
    """Wrapper around EasyOCR for text extraction."""

    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool = False,
    ) -> None:
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=gpu, verbose=False)

    def extract(self, image: np.ndarray) -> list[OCRResult]:
        """Extract text and bounding boxes from a preprocessed image.

        Args:
            image: Preprocessed image (grayscale or thresholded).

        Returns:
            List of OCRResult objects sorted by reading order.
        """
        if image is None:
            raise ValueError("Image is None")

        if len(image.shape) == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display = image

        raw_results = self.reader.readtext(display)

        results: list[OCRResult] = []
        for bbox, text, confidence in raw_results:
            bbox_list = [[int(pt[0]), int(pt[1])] for pt in bbox]
            results.append(
                OCRResult(
                    text=text.strip(),
                    confidence=float(confidence),
                    bbox=bbox_list,
                )
            )

        return results

    def extract_text(self, image: np.ndarray, *, min_confidence: float = 0.0) -> str:
        """Extract and join all detected text lines.

        Args:
            image: Preprocessed image.
            min_confidence: Minimum confidence to include a detection.

        Returns:
            Newline-joined text string.
        """
        results = self.extract(image)
        filtered = [r for r in results if r.confidence >= min_confidence]
        return "\n".join(r.text for r in filtered)
