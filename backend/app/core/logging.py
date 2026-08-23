"""Standard logging (CONTRACT.md §14): timestamp, level, component, message."""

import logging
import sys

COMPONENT = "BACKEND"

_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())


logger = logging.getLogger(COMPONENT)
