from dataclasses import dataclass, fields
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import re
from typing import List, Optional, Protocol, Tuple, Type, TypeVar

from PIL import Image
import numpy as np
import torch

from valorantlog.detector import Detection, DetectionLabel, Detector, Xyxy
from valorantlog.log import TRACE
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
        if self.text == "":
            self.text = "0"
            return
        match = ROUND_PATTERN.match(self.text)
        if match:
            self.text = match.group(1)
        else:
            if self.text:
                logger.debug(f"Failed to parse round from {self.text}")
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
        if self.text == "":
            self.text = "-1"
            return
        try:
            if p_time := self._parse_time(self.text):
                self.text = str(
                    timedelta(
                        minutes=p_time.minute, seconds=p_time.second
                    ).total_seconds()
                )
            else:
                if self.text:
                    logger.debug(f"Failed to parse time from {self.text}")
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
        if self.text == "":
            self.text = "-1"
            return
        if not self.text.isdigit():
            if self.text:
                logger.debug(f"Failed to parse score from {self.text}")
            self.text = "-1"

    def is_valid(self) -> bool:
        return super().is_valid() and self.text != "-1"


class SmoothedField(Protocol):
    def update(self, value: str) -> str: ...
    def get(self) -> Optional[str]: ...


@dataclass
class ThresholdField:
    k: int = 3
    candidate: Optional[str] = None
    candidate_count: int = 0
    value: Optional[str] = None

    def update(self, value: str) -> str:
        if self.candidate != value:
            self.candidate = value
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= self.k:
            self.value = self.candidate

        return self.value if self.value is not None else value

    def get(self) -> Optional[str]:
        return self.value


@dataclass
class DecayField:
    threshold: int = 5
    decay: int = 1
    candidate: Optional[str] = None
    value: Optional[str] = None
    score: int = 0

    def update(self, value: str) -> str:
        if self.candidate is None:
            self.candidate = value
            self.value = value
            self.score = 0
            return value
        if self.candidate == value:
            self.score = min(self.score + 1, self.threshold)
        else:
            self.candidate = value
            self.score = max(0, self.score - self.decay)

        if self.score >= self.threshold:
            self.value = self.candidate
            self.score = 0

        return self.value if self.value is not None else value

    def get(self) -> Optional[str]:
        return self.value


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

    @classmethod
    def columns(cls) -> List[str]:
        return [field.name for field in fields(cls)]

    def to_row(self, keyed: bool = False) -> List[str]:
        res = []
        for field in fields(self):
            obs = getattr(self, field.name)

            if hasattr(obs, "value"):
                value = obs.value
            else:
                value = obs.is_valid()

            cell = f"{field.name}={value}" if keyed else str(value)
            res.append(cell)
        return res

    def to_dict(self) -> dict[str, str]:
        return {name: value for name, value in zip(self.columns(), self.to_row())}

    def __str__(self) -> str:
        return "\t".join(self.to_row(True))

    def save_img(self, dir: str):
        os.makedirs(dir, exist_ok=True)
        for field in fields(self):
            obs = getattr(self, field.name)

            if hasattr(obs, "image") and obs.image is not None:
                img = Image.fromarray(obs.image)
                filename = f"{type(obs).__name__}_{obs.detection.label_name.name}_{obs.detection.label_id}.png"
                save_path = Path(dir) / filename
                img.save(save_path)


class Smoother:
    def __init__(self, smoothed_fields: dict[str, SmoothedField]):
        self.fields = {name: field for name, field in smoothed_fields.items()}

    def get(self, name: str) -> Optional[str]:
        field = self.fields.get(name)
        if field is None:
            return None
        return field.get()

    def update(self, name: str, value: str) -> str:
        field = self.fields.get(name)
        if field is None:
            return value
        return field.update(value)


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
            logger.log(
                TRACE,
                f"Detected {detection.label_name} in xyxy region {xyxy_int}",
            )
            clipped = self._clip_xyxy(img.shape[0], img.shape[1], xyxy_int)
            if clipped:
                logger.log(TRACE, f"Clipped boundaries {xyxy_int}")
            x_min, y_min, x_max, y_max = xyxy_int
            cropped_image = img[y_min:y_max, x_min:x_max]
            hint = self._get_ocr_hint(detection.label_name)
            logger.log(
                TRACE,
                f"Using OCR hints '{hint.name}' for label {detection.label_name}",
            )
            text = self.ocr.read(cropped_image, hint)
            data[detection.label_name] = Observation(
                text, detection, cropped_image.copy()
            )
        return GameState.from_ocr_data(data)

    smoothed_fields: dict[str, SmoothedField] = {
        "round": DecayField(30, 1),
        "l_team": DecayField(60, 1),
        "l_team_score": DecayField(30, 1),
        "r_team": DecayField(60, 1),
        "r_team_score": DecayField(30, 1),
        "timer": ThresholdField(1),
        "spike": ThresholdField(1),
    }

    def extract_smooth(
        self, img: np.ndarray | torch.Tensor, smoother: Optional[Smoother] = None
    ) -> Tuple[GameState, Smoother]:
        gs = self.extract(img)
        if smoother is None:
            smoother = Smoother(self.smoothed_fields)
        for field in fields(GameState):
            obs = getattr(gs, field.name)
            if not obs.is_valid():
                if value := smoother.get(field.name):
                    obs.text = value
                continue
            updated = smoother.update(field.name, obs.text)
            if obs.text != updated:
                logger.log(
                    TRACE,
                    f"Updated {obs.text} to {updated} for {field.name}",
                )
                obs.text = updated
        return (gs, smoother)
