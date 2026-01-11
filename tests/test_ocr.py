from typing import List, Tuple
import cv2
import numpy as np
import pytest
from ocr import OCR, TesseractOCR


def write_text(img: np.ndarray, loc: Tuple[int, int], text: str) -> np.ndarray:
    return cv2.putText(img, text, loc, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, 1)


@pytest.mark.parametrize("text", ["Hello, World!", "Hello!", "Hi", ""])
def test_ocr_single_line(text: str):
    ocr: OCR = TesseractOCR()
    img = 255 * np.ones((500, 500), dtype=np.uint8)
    annotated_img = write_text(img, (250, 250), text)
    assert ocr.single_line(annotated_img).strip() == text
