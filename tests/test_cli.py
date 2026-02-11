import pytest

from valorantlog.cli import Config, OCRType, OutputFormat, parse_args


@pytest.mark.parametrize(
    ["input", "expected"],
    [
        (
            ["-m", "inference_model.onnx", "-i", "file"],
            Config(
                "inference_model.onnx",
                "file",
                OutputFormat.TSV,
                "cuda",
                OCRType.TESSERACT,
            ),
        ),
        (
            ["-m", "inference_model.onnx", "-i", "file"],
            Config(
                "inference_model.onnx",
                "file",
                OutputFormat.TSV,
                "cuda",
                OCRType.TESSERACT,
            ),
        ),
        (
            ["-m", "inference_model.onnx", "-i", "file", "--output", "csv"],
            Config(
                "inference_model.onnx",
                "file",
                OutputFormat.CSV,
                "cuda",
                OCRType.TESSERACT,
            ),
        ),
        (
            [
                "-m",
                "inference_model.onnx",
                "-i",
                "file",
                "--output",
                "json",
            ],
            Config(
                "inference_model.onnx",
                "file",
                OutputFormat.JSON,
                "cuda",
                OCRType.TESSERACT,
            ),
        ),
        (
            ["-m", "inference_model.onnx", "-i", "file", "--device", "cpu"],
            Config(
                "inference_model.onnx",
                "file",
                OutputFormat.TSV,
                "cpu",
                OCRType.TESSERACT,
            ),
        ),
        (
            ["-m", "inference_model.onnx", "-i", "file", "--ocr", "easyocr"],
            Config(
                "inference_model.onnx",
                "file",
                OutputFormat.TSV,
                "cuda",
                OCRType.EASYOCR,
            ),
        ),
    ],
)
def test_parse_args(input, expected):
    assert parse_args(input) == expected
