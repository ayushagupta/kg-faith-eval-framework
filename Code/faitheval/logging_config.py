import logging
from pathlib import Path

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

logger = logging.getLogger("faitheval")
logger.setLevel(logging.INFO)

if not logger.handlers: # don't re-init
    fh = logging.FileHandler(LOG_PATH / "faitheval.log")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    ))
    logger.addHandler(fh)