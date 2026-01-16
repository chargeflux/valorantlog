from PIL import Image
from typing import Protocol
import numpy as np
import tesserocr


class OCR(Protocol):
    def auto(self, image: np.ndarray) -> str: ...
    def single_line(self, image: np.ndarray) -> str: ...
    def single_word(self, image: np.ndarray) -> str: ...


class TesseractOCR:
    def __init__(self):
        pass

    def _to_text(self, image: np.ndarray, psm: tesserocr.PSM) -> str:
        return tesserocr.image_to_text(Image.fromarray(image), psm=psm).strip()

    def auto(self, image: np.ndarray) -> str:
        return self._to_text(image, tesserocr.PSM.AUTO)

    def single_word(self, image: np.ndarray) -> str:
        return self._to_text(image, tesserocr.PSM.SINGLE_WORD)

    def single_line(self, image: np.ndarray) -> str:
        return self._to_text(image, tesserocr.PSM.SINGLE_LINE)
