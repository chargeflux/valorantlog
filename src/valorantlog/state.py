from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Callable, Optional, Type, TypeVar

import numpy as np
import torch

from valorantlog.detector import Detection, DetectionLabel, Detector
from valorantlog.ocr import OCR


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

    def extract(self, img: np.ndarray | torch.Tensor) -> GameState:
        detections = self.detector.detect(img)
        data: dict[DetectionLabel, Observation] = {}
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        # CHW -> HWC
        img = np.moveaxis(img, 0, -1)
        for detection in detections:
            x_min, y_min, x_max, y_max = detection.xyxy.astype(int)
            cropped_image = img[y_min:y_max, x_min:x_max]
            text = self._get_ocr_strategy(detection.label_name)(cropped_image)
            data[detection.label_name] = Observation(
                text, detection, cropped_image.copy()
            )
        return GameState.from_ocr_data(data)
