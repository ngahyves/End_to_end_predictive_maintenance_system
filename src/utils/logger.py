# Configuring our logger file

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
from src.utils.config_loader import load_config

def get_logger(name: str = "pipeline"):
    config = load_config()
    log_file = config["logging"]["log_file"]

    log_path = Path(log_file)
    log_path.parent.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # Rotating file handler
    fh = RotatingFileHandler(
        log_path,
        maxBytes=1_000_000,
        backupCount=5
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
