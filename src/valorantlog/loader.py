from typing import Protocol

import cv2
import numpy as np
import torch


class FrameLoader(Protocol):
    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray | torch.Tensor: ...

    def fps(self) -> float: ...


class OpenCVLoader:
    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray | torch.Tensor:
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # HWC -> CHW
        return np.moveaxis(frame, -1, 0)

    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)


class TorchCodecLoader:
    def __init__(self, path: str, device: str = "cuda"):
        from torchcodec.decoders import VideoDecoder, set_cuda_backend

        if device == "cuda":
            with set_cuda_backend("beta"):
                self.decoder = VideoDecoder(
                    path, device="cuda", seek_mode="approximate", num_ffmpeg_threads=0
                )
        else:
            self.decoder = VideoDecoder(path, device=device, num_ffmpeg_threads=0)
        self._iterator = iter(self.decoder)  # type: ignore

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray | torch.Tensor:
        try:
            frame = next(self._iterator)
            return frame
        except StopIteration:
            raise StopIteration

    def fps(self) -> float:
        fps = self.decoder.metadata.average_fps
        if fps is None:
            raise ValueError("Could not compute fps")
        return fps
