"""All settings come from the .env file, so nothing is hardcoded."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "google")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash-lite")
# Second allowed model, used only if the main one keeps failing
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "gemini-3.1-flash-lite")
API_KEY = os.getenv("API_KEY", "")

# Free tier allows 15 requests/min, so we wait at least this long between calls
MIN_SECONDS_BETWEEN_CALLS = float(os.getenv("MIN_SECONDS_BETWEEN_CALLS", "4.5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))

# Every model reply is saved here, so a re-run never spends quota twice
CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))