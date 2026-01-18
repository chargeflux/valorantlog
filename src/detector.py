from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Iterable, List, Literal, Protocol

import numpy as np
import torch

import numpy.typing as npt

from inference import InferenceProvider, Prediction
from rfdetr import RfdetrBase


Xyxy = Annotated[npt.NDArray[np.float32], Literal[4]]


class DetectionLabel(Enum):
    TIMER = "timer"
    L_TEAM = "l_team"
    R_TEAM = "r_team"
    ROUND = "round"

    def to_label_id(self) -> int:
        return DETECTION_LABEL_TO_ID[self]
    
    @classmethod
    def from_id(cls, label_id: int) -> "DetectionLabel":
        return ID_TO_DETECTION_LABEL[label_id]
    
    @classmethod
    def verify(cls, actual_labels: List[str]):
        if len(actual_labels) != len(cls):
            raise ValueError(f"Number of actual labels {len(actual_labels)} do not match length of {cls.__name__} ({len(cls)})")
        for i, label in enumerate(cls):
            if label.value != actual_labels[i]:
                raise ValueError(f"Expected {label.value} at {i} but correct label as provided is {actual_labels[i]}")

DETECTION_LABEL_TO_ID = {label: i for i, label in enumerate(DetectionLabel)}
ID_TO_DETECTION_LABEL = {i: label for i, label in enumerate(DetectionLabel)}


@dataclass
class Detection:
    xyxy: Xyxy
    confidence: float
    label_id: int
    label_name: DetectionLabel

    @classmethod
    def from_prediction(cls, pred: Prediction) -> "Detection":
        return cls(pred.boxes, pred.confidence, pred.label_id, ID_TO_DETECTION_LABEL[pred.label_id])


class Detector(Protocol):
    def detect(self, img: np.ndarray) -> List[Detection]: ...


class RfdetrONNX(RfdetrBase):
    providers = [
        "CUDAExecutionProvider", 
        "CPUExecutionProvider"
    ]

    def __init__(self, model_path: str):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(model_path, providers=self.providers)
        input_info = self.session.get_inputs()[0]
        # B X C X H X W -> H X W
        self.input_h, self.input_w = input_info.shape[2:]
        self.input_name = input_info.name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def predict(self, img: np.ndarray) -> Iterable[Prediction]:
        processed_img = self.preprocess(torch.from_numpy(img))
        inputs = {self.input_name: processed_img.unsqueeze(0).numpy()}
        outputs = self.session.run(self.output_names, inputs)
        boxes, logits = outputs
        if not isinstance(boxes, np.ndarray):
            raise TypeError(f"Bounding boxes is not {np.ndarray.__name__}")
        if not isinstance(logits, np.ndarray):
            raise TypeError(f"Logits is not {np.ndarray.__name__}")

        return self.postprocess(
            img.shape, torch.from_numpy(boxes), torch.from_numpy(logits)
        )


class Rfdetr:
    def __init__(self, model_path: str):
        if model_path.endswith(".onnx"):
            self.provider: InferenceProvider = RfdetrONNX(model_path)

    def detect(self, img: np.ndarray) -> List[Detection]:
        detections = []
        for pred in self.provider.predict(img):
            detections.append(Detection.from_prediction(pred))
        return detections
