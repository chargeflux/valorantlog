from enum import IntFlag, auto
import logging
from typing import Protocol
import numpy as np
import tesserocr

from valorantlog.log import TRACE

logger = logging.getLogger(__name__)


class OCRHint(IntFlag):
    NONE = 0
    INVERT = auto()
    SINGLE_WORD = auto()
    SINGLE_LINE = auto()


class OCR(Protocol):
    def read(self, image: np.ndarray, hint: OCRHint) -> str: ...


class TesseractOCR:
    def __init__(self):
        self.api = tesserocr.PyTessBaseAPI(variables={"tessedit_do_invert": "0"})

    def _set_img_bytes(self, image: np.ndarray):
        h, w = image.shape[:2]
        bpp = 1 if image.ndim == 2 else image.shape[2]
        bpl = bpp * w
        self.api.SetImageBytes(
            imagedata=np.ascontiguousarray(image).tobytes(),  # pyright: ignore[reportArgumentType]
            height=h,
            width=w,
            bytes_per_pixel=bpp,
            bytes_per_line=bpl,
        )

    def _to_text(self, image: np.ndarray, psm: tesserocr.PSM) -> str:
        logger.log(TRACE, f"Running tesseract with psm {psm}")
        self._set_img_bytes(image)
        self.api.SetPageSegMode(psm)
        text = self.api.GetUTF8Text().strip()
        return text

    def read(self, image: np.ndarray, hint: OCRHint) -> str:
        processed_image = ~image if (hint & OCRHint.INVERT) else image
        psm = tesserocr.PSM.AUTO

        if hint & OCRHint.SINGLE_WORD:
            psm = tesserocr.PSM.SINGLE_WORD
        if hint & OCRHint.SINGLE_LINE:
            psm = tesserocr.PSM.SINGLE_LINE

        return self._to_text(processed_image, psm)

    def __del__(self):
        self.api.End()
