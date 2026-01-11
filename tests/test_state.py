from typing import List, Optional
import numpy as np
import pytest
from sympy import false
from detector import Detection, DetectionLabel, Xyxy
from state import GameStateExtractor


class MockDetector:
    def __init__(self, detections: List[Detection]):
        self.detections = detections

    def detect(self, image: np.ndarray) -> List[Detection]:
        return self.detections


class MockOCR:
    def __init__(self, text_results: list[str]):
        self.text_results = iter(text_results)
        self.cnt = len(text_results)

    def single_line(self, image: np.ndarray) -> str:
        try:
            return next(self.text_results)
        except StopIteration:
            raise ValueError(f"MockOCR called more than {self.cnt} times")


def create_mock_detection(
    label: DetectionLabel, xyxy: Optional[Xyxy] = None
) -> Detection:
    return Detection(xyxy if xyxy else (0, 0, 0, 0), 1.0, label.to_class_id(), label)


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
    assert state.round.is_valid() == false


def test_extractor_extract_team_invalid():
    ocr = MockOCR(["", ""])
    labels = [DetectionLabel.L_TEAM, DetectionLabel.R_TEAM]
    detector = MockDetector([create_mock_detection(l) for l in labels])
    p = GameStateExtractor(ocr, detector)
    state = p.extract(np.empty((0, 0)))
    assert state.l_team.value == ""
    assert state.l_team.is_valid() == false

    assert state.r_team.value == ""
    assert state.r_team.is_valid() == false


def test_extractor_extract_timer_invalid():
    ocr = MockOCR([""])
    labels = [DetectionLabel.TIMER]
    detector = MockDetector([create_mock_detection(l) for l in labels])
    p = GameStateExtractor(ocr, detector)
    state = p.extract(np.empty((0, 0)))
    assert state.timer.value == -1
    assert state.timer.is_valid() == false
