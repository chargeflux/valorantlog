import logging
from os import environ


def configure_logging(log_level="INFO") -> None:
    log_level = environ.get("LOGGING_LEVEL", log_level).upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s:%(name)s] %(message)s",
    )
