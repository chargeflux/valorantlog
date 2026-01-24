# valorantlog

`valorantlog` analyzes Valorant Esports VODs and outputs real-time information about the state of the game as structured data. It is a computer vision pipeline consisting of 2 stages: object detection and OCR.

## Getting Started

### Initialize environment
```shell
uv sync
```

### Usage

```shell
valorantlog -m <model path> -i <video file> [-o tsv,csv,json]
```

The model can be found in [Releases](https://github.com/chargeflux/valorantlog/releases).

## Architecture

`valorantlog` has 3 main parts: 
1. Video Loader
    - [torchcodec](https://github.com/meta-pytorch/torchcodec)
    - OpenCV
2. Detector
    - [rf-detr](https://github.com/roboflow/rf-detr) (ONNX)
3. OCR
    - Tesseract

The pipeline is modular so each component can be swapped out, provided the protocol defining each stage is satisfied.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

Third-party dependencies and their licenses are listed in [NOTICE](./NOTICE).

