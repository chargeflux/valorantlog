from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import re
from typing import Callable, Optional, Type, TypeVar

import numpy as np
import torch

from valorantlog.detector import Detection, DetectionLabel, Detector, Xyxy
from valorantlog.ocr import OCR

logger = logging.getLogger(__name__)

ROUND_PATTERN = re.compile(r"^ROUND\s*(\d+)$")
ObservationLike = TypeVar("ObservationLike", bound="Observation")


@dataclass
class Observation:
    text: str
    detection: Optional[Detection]
    image: np.ndarray | None

    @classmethod
    def from_observation(
        cls: Type[ObservationLike], obs: "Observation"
    ) -> ObservationLike:
        return cls(text=obs.text, detection=obs.detection, image=obs.image)

    def is_valid(self) -> bool:
        return self.text != "" and self.detection is not None


@dataclass
class Round(Observation):
    @property
    def value(self) -> int:
        return int(self.text)

    def __post_init__(self):
        match = ROUND_PATTERN.match(self.text)
        if match:
            self.text = match.group(1)
        else:
            self.text = "0"

    def is_valid(self) -> bool:
        return super().is_valid() and self.text != "0"


@dataclass
class Timer(Observation):
    @property
    def value(self) -> float:
        return float(self.text)

    def __post_init__(self):
        try:
            if self.text:
                p_time = datetime.strptime(self.text, "%M:%S")
                self.text = str(
                    timedelta(
                        minutes=p_time.minute, seconds=p_time.second
                    ).total_seconds()
                )
            else:
                self.text = "-1"
        except ValueError:
            self.text = "-1"

    def is_valid(self) -> bool:
        return super().is_valid() and self.text != "-1"


@dataclass
class Team(Observation):
    @property
    def value(self) -> str:
        return self.text


@dataclass
class GameState:
    round: Round
    timer: Timer
    l_team: Team
    r_team: Team

    @classmethod
    def from_ocr_data(cls, data: dict[DetectionLabel, Observation]) -> "GameState":
        def extract_data(label: DetectionLabel):
            return data.get(
                label,
                Observation("", None, None),
            )

        return cls(
            round=Round.from_observation(extract_data(DetectionLabel.ROUND)),
            timer=Timer.from_observation(extract_data(DetectionLabel.TIMER)),
            l_team=Team.from_observation(extract_data(DetectionLabel.L_TEAM)),
            r_team=Team.from_observation(extract_data(DetectionLabel.R_TEAM)),
        )


class GameStateExtractor:
    def __init__(self, ocr: OCR, detector: Detector):
        self.ocr = ocr
        self.detector = detector

    def _get_ocr_strategy(self, label: DetectionLabel) -> Callable:
        match label:
            case DetectionLabel.ROUND:
                return self.ocr.single_line
            case DetectionLabel.L_TEAM | DetectionLabel.R_TEAM:
                return self.ocr.single_word
            case _:
                return self.ocr.auto
    
    def _clip_xyxy(self, img_h: int, img_w: int, xyxy: Xyxy) -> bool:
        if not np.any((xyxy < 0) | (xyxy > [img_w, img_h, img_w, img_h]), axis=0):
            return False
        xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, img_w)
        xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, img_h)
        return True

    def extract(self, img: np.ndarray | torch.Tensor) -> GameState:
        detections = self.detector.detect(img)
        data: dict[DetectionLabel, Observation] = {}
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        # CHW -> HWC
        img = np.moveaxis(img, 0, -1)
        for detection in detections:
            xyxy_int = detection.xyxy.astype(int)
            logger.debug(f"Detected {detection.label_name} in xyxy region {xyxy_int}")
            clipped = self._clip_xyxy(img.shape[0], img.shape[1], xyxy_int)
            if clipped:
                logger.debug(f"Clipped boundaries {xyxy_int}")
            x_min, y_min, x_max, y_max = xyxy_int
            cropped_image = img[y_min:y_max, x_min:x_max]
            strategy = self._get_ocr_strategy(detection.label_name)
            logger.debug(f"Using OCR strategy \'{strategy.__name__}\' for label {detection.label_name}")
            text = strategy(cropped_image)
            data[detection.label_name] = Observation(
                text, detection, cropped_image.copy()
            )
        return GameState.from_ocr_data(data)
