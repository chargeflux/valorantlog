from dataclasses import fields
from typing import List, Optional
import numpy as np
import pytest
import torch
from valorantlog.detector import (
    DETECTION_LABEL_TO_ID,
    Detection,
    DetectionLabel,
    Xyxy,
)
from valorantlog.ocr import OCRHint
from valorantlog.state import (
    DecayField,
    GameState,
    GameStateExtractor,
    Observation,
    ThresholdField,
)


class MockDetector:
    def __init__(self, detections: List[Detection]):
        self.detections = detections

    def detect(self, img: np.ndarray | torch.Tensor) -> List[Detection]:
        return self.detections


class MockOCR:
    def __init__(self, text_results: list[str]):
        self.text_results = iter(text_results)
        self.cnt = len(text_results)

    def _image_to_text(self, image: np.ndarray) -> str:
        try:
            return next(self.text_results)
        except StopIteration:
            raise ValueError(f"MockOCR called more than {self.cnt} times")

    def read(self, image: np.ndarray, hint: OCRHint = OCRHint.NONE) -> str:
        return self._image_to_text(image)


def create_mock_detection(
    label: DetectionLabel, xyxy: Optional[Xyxy] = None
) -> Detection:
    return Detection(
        (xyxy if xyxy is not None else np.zeros((4,), dtype=np.float32)),
        1.0,
        label.to_label_id(),
        label,
    )


@pytest.mark.parametrize(
    "input", [["ROUND 1", "A", "B", "1:01"], ["ROUND1", "A", "B", "01:01"]]
)
def test_extractor_extract(input: list[str]):
    ocr = MockOCR(input)
    labels = [
        DetectionLabel.ROUND,
        DetectionLabel.L_TEAM,
        DetectionLabel.R_TEAM,
        DetectionLabel.TIMER,
    ]
    detector = MockDetector([create_mock_detection(l) for l in labels])
    p = GameStateExtractor(ocr, detector)
    state = p.extract(np.empty((0, 0)))
    assert state.round.value == 1
    assert state.l_team.value == "A"
    assert state.r_team.value == "B"
    assert state.timer.value == 61


@pytest.mark.parametrize("input", ["ROUND", "ROUND I"])
def test_extractor_extract_round_invalid(input: str):
    ocr = MockOCR([input])
    labels = [DetectionLabel.ROUND]
    detector = MockDetector([create_mock_detection(l) for l in labels])
    p = GameStateExtractor(ocr, detector)
    state = p.extract(np.empty((0, 0)))
    assert state.round.value == 0
    assert not state.round.is_valid()


def test_extractor_extract_team_invalid():
    ocr = MockOCR(["", ""])
    labels = [DetectionLabel.L_TEAM, DetectionLabel.R_TEAM]
    detector = MockDetector([create_mock_detection(l) for l in labels])
    p = GameStateExtractor(ocr, detector)
    state = p.extract(np.empty((0, 0)))
    assert state.l_team.value == ""
    assert not state.l_team.is_valid()

    assert state.r_team.value == ""
    assert not state.r_team.is_valid()


def test_extractor_extract_timer_invalid():
    ocr = MockOCR([""])
    labels = [DetectionLabel.TIMER]
    detector = MockDetector([create_mock_detection(l) for l in labels])
    p = GameStateExtractor(ocr, detector)
    state = p.extract(np.empty((0, 0)))
    assert state.timer.value == -1
    assert not state.timer.is_valid()


def test_thresholdfield_locked():
    updates = ["3", "3", "3", "4", "4", "4"]
    expected = [None, None, "3", "3", "3", "4"]
    fl = ThresholdField(3)
    for i, update in enumerate(updates):
        fl.update(update)
        assert fl.value == expected[i]


def test_thresholdfield_update():
    fl = ThresholdField(2)
    assert fl.update("3") == "3"
    assert fl.update("3") == "3"

    assert fl.update("5") == "3"


def test_decayfield_locked():
    updates = ["3", "4", "4", "4"]
    expected = ["3", "3", "3", "4"]
    fl = DecayField(2)
    for i, update in enumerate(updates):
        fl.update(update)
        assert fl.value == expected[i]


def test_decayfield_update():
    fl = DecayField(2)
    assert fl.update("3") == "3"
    assert fl.update("3") == "3"
    assert fl.update("3") == "3"
    # locked
    assert fl.update("5") == "3"
    assert fl.update("5") == "3"
    # unlocked
    assert fl.update("5") == "5"


def test_gamestate_columns():
    assert GameState.columns() == [field.name for field in fields(GameState)]


def test_gamestate_to_row():
    gs = GameState.from_ocr_data(
        {
            DetectionLabel.ROUND: Observation(
                "ROUND 1",
                Detection(
                    np.zeros(4, dtype=np.float32),
                    0,
                    DETECTION_LABEL_TO_ID[DetectionLabel.ROUND],
                    DetectionLabel.ROUND,
                ),
                None,
            )
        }
    )
    assert gs.to_row() == ["1", "-1.0", "", "-1", "", "-1", "False"]


def test_gamestate_to_row_keyed():
    gs = GameState.from_ocr_data(
        {
            DetectionLabel.ROUND: Observation(
                "ROUND 1",
                Detection(
                    np.zeros(4, dtype=np.float32),
                    0,
                    DETECTION_LABEL_TO_ID[DetectionLabel.ROUND],
                    DetectionLabel.ROUND,
                ),
                None,
            )
        }
    )
    assert gs.to_row(True) == [
        f"{k}={v}"
        for k, v in zip(gs.columns(), ["1", "-1.0", "", "-1", "", "-1", "False"])
    ]


def test_gamestate_to_dict():
    gs = GameState.from_ocr_data(
        {
            DetectionLabel.ROUND: Observation(
                "ROUND 1",
                Detection(
                    np.zeros(4, dtype=np.float32),
                    0,
                    DETECTION_LABEL_TO_ID[DetectionLabel.ROUND],
                    DetectionLabel.ROUND,
                ),
                None,
            )
        }
    )

    assert gs.to_dict() == {
        k: v for k, v in zip(gs.columns(), ["1", "-1.0", "", "-1", "", "-1", "False"])
    }
