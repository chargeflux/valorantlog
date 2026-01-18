from typing import List
import numpy as np
import pytest
from detector import Detection, DetectionLabel
from inference import Prediction

@pytest.mark.parametrize(["actual_labels", "should_raise"], [([label.value for label in DetectionLabel], False), ([DetectionLabel.R_TEAM], True)])
def test_detection_label_verify(actual_labels: List[str], should_raise: bool):
    print(actual_labels)
    if should_raise:
        with pytest.raises(ValueError):
            DetectionLabel.verify(actual_labels)
    else:
        DetectionLabel.verify(actual_labels)

@pytest.mark.parametrize("label_id",[label.to_label_id() for label in DetectionLabel])
def test_detection_from_prediction(label_id: int):
    pred = Prediction(np.array([0, 0, 10, 10]), 0.5, label_id)
    detection = Detection.from_prediction(pred)
    assert detection.confidence == pred.confidence
    assert detection.label_id == pred.label_id
    assert np.all(detection.xyxy == pred.boxes)


