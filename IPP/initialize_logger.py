"""
Module loads config and rotate the logger.
"""

import logging
import logging.config

import yaml


def initialize_logger(logger: logging.Logger) -> None:
    """Load configuration of the logger. Backup old logger."""
    with open("logging.yaml", "r") as f:
        log_conf = yaml.safe_load(f)
        logging.config.dictConfig(log_conf)
    for handler in logger.handlers:
        if handler.__class__.__name__ == "RotatingFileHandler":
            handler.doRollover()  # type: ignore
