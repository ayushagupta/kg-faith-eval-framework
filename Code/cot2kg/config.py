import os
from pathlib import Path
from config.config import config_mini as _global

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", _global.OPENAI_API_KEY)
MODEL_NAME = os.getenv("COT2KG_MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", _global.TEMPERATURE))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", _global.MAX_TOKENS))

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
DEFAULT_OUTPUT_MODE = "compact" # otherwise "all"