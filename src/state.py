from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from typing import Optional, Type, TypeVar

import numpy as np

from detector import Detection, DetectionLabel, Detector
from ocr import OCR


ObservationLike = TypeVar("ObservationLike", bound="Observation")

ROUND_PATTERN = re.compile(r"^ROUND\s*(\d+)$")


@dataclass
class Observation:
    text: str
    detection: Optional[Detection]

    @classmethod
    def from_observation(
        cls: Type[ObservationLike], obs: "Observation"
    ) -> ObservationLike:
        return cls(text=obs.text, detection=obs.detection)

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
        return cls(
            round=Round.from_observation(
                data.get(DetectionLabel.ROUND, Observation("", None))
            ),
            timer=Timer.from_observation(
                data.get(DetectionLabel.TIMER, Observation("", None))
            ),
            l_team=Team.from_observation(
                data.get(DetectionLabel.L_TEAM, Observation("", None))
            ),
            r_team=Team.from_observation(
                data.get(DetectionLabel.R_TEAM, Observation("", None))
            ),
        )


class GameStateExtractor:
    def __init__(self, ocr: OCR, detector: Detector):
        self.ocr = ocr
        self.detector = detector

    def extract(self, img: np.ndarray) -> GameState:
        detections = self.detector.detect(img)
        data: dict[DetectionLabel, Observation] = {}
        for detection in detections:
            x_min, y_min, x_max, y_max = detection.xyxy
            text = self.ocr.single_line(img[y_min:y_max, x_min:x_max])
            data[detection.class_label] = Observation(text, detection)
        return GameState.from_ocr_data(data)
