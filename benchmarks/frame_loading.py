from argparse import ArgumentParser
from enum import Enum
from functools import partial
import time
from typing import Any, Callable, Iterable
import warnings

import torch
import numpy as np
from valorantlog.loader import OpenCVLoader, TorchCodecLoader

class LoaderBackend(Enum):
    OPENCV = "opencv"
    TORCHCODEC = "torchcodec"

def run_benchmark(f: Callable, num_runs: int, warmup=0) -> np.ndarray:
    for _ in range(warmup):
        f()
    runs = []
    
    for _ in range(num_runs):
        start = time.perf_counter()
        f()
        end = time.perf_counter()
        runs.append(end-start)
    return np.array(runs)

def load_frames(loader_factory: Callable[[], Iterable[Any]], is_cuda: bool):
    loader = loader_factory()
    for _ in loader:
        pass
    if is_cuda:
        torch.cuda.synchronize()

parser = ArgumentParser()
parser.add_argument("-f", "--file-path", type=str, help="path to file", required=True)
parser.add_argument("-b", "--backend", type=LoaderBackend, choices=LoaderBackend, help="backend for deocding", default=LoaderBackend.OPENCV)
parser.add_argument("-n", "--num-runs", type=int, help="number of runs", default=1)
parser.add_argument("-d", "--device", type=str, help="device for decoding", default="cpu")
parser.add_argument("-w", "--num-warmup", type=int, help="number of warmup runs", default=0)

args = parser.parse_args()

is_cuda = args.device=="cuda"

match args.backend:
    case LoaderBackend.OPENCV:
        if is_cuda == "cuda":
            raise ValueError("Device can not be 'cuda'")
        loader_factory = partial(OpenCVLoader, path=args.file_path)
    case LoaderBackend.TORCHCODEC:
        loader_factory = partial(TorchCodecLoader, path=args.file_path, device=args.device)
    case unknown:
        raise ValueError(f"Unknown backend: {unknown}")

task=partial(load_frames, loader_factory=loader_factory, is_cuda=is_cuda)

print(f"backend: {args.backend.value}, is_cuda: {is_cuda}, num_runs: {args.num_runs}, num_warmup: {args.num_warmup}")
if is_cuda and args.num_warmup == 0:
    warnings.warn("Number of warmup runs is 0 for CUDA")

runs = run_benchmark(task, args.num_runs, args.num_warmup)

print(f"mean: {runs.mean()}, std: {runs.std()}")
