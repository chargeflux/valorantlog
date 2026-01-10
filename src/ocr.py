from PIL import Image
from typing import Protocol
import numpy as np
import tesserocr

class OCR(Protocol):
    def single_line(self, image: np.ndarray) -> str:
        ...

class TesseractOCR():
    def __init__(self):
        pass
    
    def _to_text(self, image: np.ndarray, psm: tesserocr.PSM) -> str:
        return tesserocr.image_to_text(Image.fromarray(image), psm=psm)

    def single_line(self, image: np.ndarray) -> str:
        return self._to_text(image, tesserocr.PSM.SINGLE_LINE)
