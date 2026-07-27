"""Logging utilities for TWEEF.

Library modules should get a logger through ``TWEEFLogger.get_logger`` and avoid
configuring global handlers at import time. Applications and demos can call
``TWEEFLogger.configure`` once at startup.
"""

import logging
import sys


class TWEEFLogger:
    DEFAULT_FORMAT = (
        "%(asctime)s | %(levelname)s | %(name)s.%(funcName)s | %(message)s"
    )
    DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

    @classmethod
    def configure(
        cls,
        level: int | str = logging.INFO,
        stream=None,
        force: bool = False,
        quiet_third_party: bool = True,
    ) -> None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logging.basicConfig(
            level=level,
            format=cls.DEFAULT_FORMAT,
            datefmt=cls.DEFAULT_DATE_FORMAT,
            stream=stream or sys.stderr,
            force=force,
        )
        if quiet_third_party:
            for logger_name in ("interpret", "aif360", "matplotlib", "sklearn"):
                logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return TWEEFLogger.get_logger(name)
