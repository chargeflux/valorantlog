from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
import json
import logging
import sys

from valorantlog.detector import Rfdetr
from valorantlog.loader import TorchCodecLoader
from valorantlog.log import configure_logging
from valorantlog.ocr import TesseractOCR
from valorantlog.state import GameState, GameStateExtractor


class OutputFormat(Enum):
    TSV = "tsv"
    CSV = "csv"
    JSON = "json"

    def delimiter(self) -> str:
        match self.value:
            case self.TSV.value:
                return "\t"
            case self.CSV.value:
                return ","
            case _:
                return ""


@dataclass
class Config:
    model_path: str
    input_file: str
    output_format: OutputFormat

    @classmethod
    def from_args(cls, parsed_args: Namespace) -> "Config":
        return Config(
            parsed_args.model_path,
            parsed_args.input_file,
            OutputFormat(parsed_args.output),
        )


def parse_args(args) -> Config:
    parser = ArgumentParser(
        "valorantlog", description="Analyze Valorant eSports VODS in real-time"
    )
    parser.add_argument("-m", "--model-path", help="path to model", required=True)
    parser.add_argument("-i", "--input-file", help="path to input file", required=True)
    parser.add_argument(
        "-o",
        "--output",
        default=OutputFormat.TSV,
        choices=[f.value for f in OutputFormat],
        help=f"Format output text: {[f.value for f in OutputFormat]}",
    )

    parsed = parser.parse_args(args)

    return Config.from_args(parsed)


def main():
    configure_logging()

    config = parse_args(sys.argv[1:])
    logging.debug(f"Parsed args: {config}")

    model = Rfdetr(config.model_path)
    ocr = TesseractOCR()
    loader = TorchCodecLoader(config.input_file)
    extractor = GameStateExtractor(ocr, model)
    delimiter = config.output_format.delimiter()
    is_tabular = config.output_format != OutputFormat.JSON

    if is_tabular:
        print(delimiter.join(GameState.columns()))

    smoother = None
    for frame in loader:
        ds, smoother = extractor.extract_smooth(frame, smoother)
        if is_tabular:
            print(delimiter.join(ds.to_row()))
        else:
            print(json.dumps(ds.to_dict()))


if __name__ == "__main__":
    main()
