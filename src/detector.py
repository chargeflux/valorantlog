from dataclasses import dataclass
from enum import Enum
from typing import List, Protocol, Tuple

import numpy as np

Xyxy = Tuple[float, float, float, float]


class DetectionLabel(Enum):
    ROUND = "round"
    TIMER = "timer"
    L_TEAM = "l_team"
    R_TEAM = "r_team"

    def to_class_id(self) -> int:
        return LABEL_TO_ID[self]


LABEL_TO_ID = {
    DetectionLabel.TIMER: 0,
    DetectionLabel.L_TEAM: 1,
    DetectionLabel.R_TEAM: 2,
    DetectionLabel.ROUND: 3,
}


@dataclass
class Detection:
    xyxy: Xyxy
    confidence: float
    class_id: int
    class_label: DetectionLabel


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> List[Detection]: ...
