"""Generate a sample image for testing the OCR pipeline."""

import cv2
import numpy as np

canvas = np.ones((400, 800, 3), dtype=np.uint8) * 255
y = 50
for text in [
    "OCR Pipeline Test",
    "The quick brown fox jumps over the lazy dog.",
    "EasyOCR extracts text from images reliably.",
    "Preprocessing improves accuracy.",
    "Line 5: 12345 @#$%",
]:
    cv2.putText(canvas, text, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    y += 50
cv2.imwrite("sample_input.png", canvas)
print("sample_input.png created")
