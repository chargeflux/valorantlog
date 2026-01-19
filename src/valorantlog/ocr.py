import logging
from typing import Protocol
import numpy as np
import tesserocr

logger = logging.getLogger(__name__)

class OCR(Protocol):
    def auto(self, image: np.ndarray) -> str: ...
    def single_line(self, image: np.ndarray) -> str: ...
    def single_word(self, image: np.ndarray) -> str: ...


class TesseractOCR:
    def __init__(self):
        self.api = tesserocr.PyTessBaseAPI()

    def _set_img_bytes(self, image: np.ndarray):
        h, w = image.shape[:2]
        bpp=1 if image.ndim == 2 else image.shape[2]
        bpl=bpp*w
        self.api.SetImageBytes(
            imagedata=np.ascontiguousarray(image.astype(np.uint8)).tobytes(), # pyright: ignore[reportArgumentType]
            height=h,
            width=w,
            bytes_per_pixel=bpp,
            bytes_per_line=bpl
        )

    def _to_text(self, image: np.ndarray, psm: tesserocr.PSM) -> str:
        logger.debug(f"Running tesseract with psm {psm}")
        self._set_img_bytes(image)
        self.api.SetPageSegMode(psm)
        text = self.api.GetUTF8Text().strip()
        return text

    def auto(self, image: np.ndarray) -> str:
        return self._to_text(image, tesserocr.PSM.AUTO)

    def single_word(self, image: np.ndarray) -> str:
        return self._to_text(image, tesserocr.PSM.SINGLE_WORD)

    def single_line(self, image: np.ndarray) -> str:
        return self._to_text(image, tesserocr.PSM.SINGLE_LINE)
    
    def __del__(self):
        self.api.End()
