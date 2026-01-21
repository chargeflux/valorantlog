from dataclasses import dataclass, fields
from datetime import datetime, timedelta
import logging
import os
import re
from typing import Optional, Type, TypeVar

from PIL import Image
import numpy as np
import torch

from valorantlog.detector import Detection, DetectionLabel, Detector, Xyxy
from valorantlog.ocr import OCR, OCRHint

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
class Observed(Observation):
    def is_valid(self) -> bool:
        return self.detection is not None


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

    def _parse_time(self, text: str) -> Optional[datetime]:
        if not text:
            return None
        formats = ["%M:%S", "%S.%f"]
        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def __post_init__(self):
        try:
            if p_time := self._parse_time(self.text):
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
class Spike(Observed):
    pass


@dataclass
class Team(Observation):
    @property
    def value(self) -> str:
        return self.text


@dataclass
class Score(Observation):
    @property
    def value(self) -> int:
        return int(self.text)

    def __post_init__(self):
        if not self.text.isdigit():
            self.text = "-1"

    def is_valid(self) -> bool:
        return super().is_valid() and self.text != "-1"


@dataclass
class GameState:
    round: Round
    timer: Timer
    l_team: Team
    l_team_score: Score
    r_team: Team
    r_team_score: Score
    spike: Spike

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
            l_team_score=Score.from_observation(
                extract_data(DetectionLabel.L_TEAM_SCORE)
            ),
            r_team_score=Score.from_observation(
                extract_data(DetectionLabel.R_TEAM_SCORE)
            ),
            spike=Spike.from_observation(extract_data(DetectionLabel.SPIKE)),
        )

    def __str__(self) -> str:
        res = []
        for field in fields(self):
            obs = getattr(self, field.name)

            if hasattr(obs, "value"):
                res.append(f"{field.name} {obs.value}")
            else:
                res.append(f"{field.name} {obs.is_valid()}")

        return "\t".join(res)

    def save_img(self, dir: str):
        os.makedirs(dir, exist_ok=True)
        for field in fields(self):
            obs = getattr(self, field.name)

            if hasattr(obs, "image") and obs.image is not None:
                img = Image.fromarray(obs.image)
                filename = f"{type(obs).__name__}_{obs.detection.label_name.name}_{obs.detection.label_id}.png"
                save_path = Path(dir) / filename
                img.save(save_path)


class GameStateExtractor:
    def __init__(self, ocr: OCR, detector: Detector):
        self.ocr = ocr
        self.detector = detector

    def _get_ocr_hint(self, label: DetectionLabel) -> OCRHint:
        match label:
            case DetectionLabel.ROUND:
                return OCRHint.INVERT | OCRHint.SINGLE_LINE
            case DetectionLabel.L_TEAM | DetectionLabel.R_TEAM:
                return OCRHint.INVERT | OCRHint.SINGLE_WORD
            case DetectionLabel.L_TEAM_SCORE | DetectionLabel.R_TEAM_SCORE:
                return OCRHint.INVERT | OCRHint.SINGLE_LINE
            case _:
                return OCRHint.INVERT

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
        img = np.moveaxis(img, 0, -1).astype(np.uint8)
        for detection in detections:
            xyxy_int = detection.xyxy.astype(int)
            logger.debug(f"Detected {detection.label_name} in xyxy region {xyxy_int}")
            clipped = self._clip_xyxy(img.shape[0], img.shape[1], xyxy_int)
            if clipped:
                logger.debug(f"Clipped boundaries {xyxy_int}")
            x_min, y_min, x_max, y_max = xyxy_int
            cropped_image = img[y_min:y_max, x_min:x_max]
            hint = self._get_ocr_hint(detection.label_name)
            logger.debug(
                f"Using OCR hints '{hint.name}' for label {detection.label_name}"
            )
            text = self.ocr.read(cropped_image, hint)
            data[detection.label_name] = Observation(
                text, detection, cropped_image.copy()
            )
        return GameState.from_ocr_data(data)
