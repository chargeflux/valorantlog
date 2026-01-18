from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import torch


@dataclass
class Prediction:
    boxes: np.ndarray
    confidence: float
    label_id: int


class InferenceProvider(Protocol):
    def predict(self, img: np.ndarray | torch.Tensor) -> Iterable[Prediction]: ...
