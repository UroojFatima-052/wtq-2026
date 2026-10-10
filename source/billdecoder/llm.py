"""The only place that talks to the model.

Handles: rate limiting, retries with backoff, switching to the fallback
model, JSON parsing, and caching replies on disk.
"""
import hashlib
import io
import json
import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image, ImageOps

from . import config

log = logging.getLogger(__name__)

_client = None
_last_call = 0.0


class LLMError(Exception):
    """Raised when the model could not give a usable answer."""


def _get_client():
    global _client
    if _client is None:
        if not config.API_KEY:
            raise LLMError("API_KEY is missing. Copy .env.example to .env and add your key.")
        _client = genai.Client(api_key=config.API_KEY)
    return _client


def load_image(path: Path) -> tuple[bytes, str]:
    """Read a bill image, fix phone rotation (EXIF), and return PNG bytes."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _wait_for_rate_limit():
    global _last_call
    gap = time.time() - _last_call
    if gap < config.MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(config.MIN_SECONDS_BETWEEN_CALLS - gap)
    _last_call = time.time()


def _is_temporary(err: Exception) -> bool:
    text = str(err).lower()
    return any(s in text for s in ("429", "503", "500", "overloaded", "unavailable",
                                   "resource_exhausted", "timeout", "deadline", "ssl", "connection"))


def _parse_json(text: str) -> dict:
    text = (text or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in reply")
    return json.loads(text[start:end + 1])


def _call_once(model: str, prompt: str, image: tuple[bytes, str] | None) -> dict:
    parts = []
    if image:
        parts.append(types.Part.from_bytes(data=image[0], mime_type=image[1]))
    parts.append(prompt)
    _wait_for_rate_limit()
    response = _get_client().models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return _parse_json(response.text)


def ask_json(prompt: str, image: tuple[bytes, str] | None = None, cache_key: str | None = None) -> dict:
    """Send a prompt (plus optional image) and get a JSON dict back.

    Tries the main model with retries, then the fallback model.
    Cached by cache_key + prompt hash so identical requests are free.
    """
    cache_file = None
    if cache_key:
        digest = hashlib.sha256((prompt + str(len(image[0]) if image else "")).encode()).hexdigest()[:12]
        cache_file = config.CACHE_DIR / f"{cache_key}_{digest}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

    last_err = None
    for model in (config.MODEL_NAME, config.FALLBACK_MODEL_NAME):
        for attempt in range(1, config.MAX_RETRIES + 1):
            try:
                result = _call_once(model, prompt, image)
                if cache_file:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                return result
            except (ValueError, json.JSONDecodeError) as err:
                # Model replied but not valid JSON: just try again
                last_err = err
                log.warning("%s gave invalid JSON (attempt %d): %s", model, attempt, err)
            except Exception as err:  # network / quota / server errors
                last_err = err
                if not _is_temporary(err):
                    log.warning("%s failed with a permanent error: %s", model, err)
                    break
                wait = min(5 * 2 ** (attempt - 1), 40)
                log.warning("%s temporary error (attempt %d), waiting %ss: %s", model, attempt, wait, err)
                time.sleep(wait)
        log.warning("Switching away from %s", model)
    raise LLMError(f"Model call failed after retries: {last_err}")