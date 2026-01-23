import pytest

from valorantlog.cli import Config, OutputFormat, parse_args


@pytest.mark.parametrize(
    ["input", "expected"],
    [
        (
            ["-m", "inference_model.onnx", "-i", "file"],
            Config("inference_model.onnx", "file", OutputFormat.TSV),
        ),
        (
            ["-m", "inference_model.onnx", "-i", "file"],
            Config("inference_model.onnx", "file", OutputFormat.TSV),
        ),
        (
            ["-m", "inference_model.onnx", "-i", "file", "--output", "csv"],
            Config("inference_model.onnx", "file", OutputFormat.CSV),
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
            Config("inference_model.onnx", "file", OutputFormat.JSON),
        ),
    ],
)
def test_parse_args(input, expected):
    assert parse_args(input) == expected
