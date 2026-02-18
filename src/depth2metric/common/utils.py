import logging
import os

from depth2metric.common.settings import get_settings

settings = get_settings()


def get_logger(
    name: str,
    level: str | int = settings.logging_level,
    filename: str | None = None
) -> logging.Logger:
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    date_fmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename, mode="a+")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
