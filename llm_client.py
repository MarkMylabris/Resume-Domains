"""
llm_client.py - Единый интерфейс к LLM-провайдеру.

Поддерживаемые провайдеры:
  "gemini"  - Google Gemini через google-genai SDK (по умолчанию)
  "openai"  - любой OpenAI-совместимый HTTP API: OpenAI, Ollama, OpenRouter, LM Studio

Конфигурация в config.json (секция "llm"):

  {
    "llm": {
      "provider": "gemini",
      "api_key":  "AIza...",
      "model":    "gemini-2.5-flash",
      "base_url": ""
    }
  }

Примеры переключения провайдеров:

  Ollama (локально, бесплатно):
    "provider": "openai",
    "base_url": "http://localhost:11434/v1",
    "api_key":  "ollama",
    "model":    "llama3.1"

  OpenRouter (облако, много моделей):
    "provider": "openai",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key":  "sk-or-v1-...",
    "model":    "openai/gpt-4o-mini"

  LM Studio (локально):
    "provider": "openai",
    "base_url": "http://localhost:1234/v1",
    "api_key":  "lm-studio",
    "model":    "local-model"

  OpenAI:
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "api_key":  "sk-...",
    "model":    "gpt-4o-mini"

Для обратной совместимости читается и старый формат "gemini": {...}.
"""

import os, sys, re, json, time
import urllib.request, urllib.error
from pathlib import Path

BASE = Path(__file__).parent

# ──────────────────────────────────────────────────────────────
#  ЗАГРУЗКА КОНФИГА
# ──────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Читает config.json, возвращает нормализованный словарь настроек LLM."""
    cfg_path = BASE / "config.json"
    raw = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # Новый формат: секция "llm"
    if "llm" in raw:
        llm = raw["llm"]
        return {
            "provider": llm.get("provider", "gemini"),
            "api_key":  (os.environ.get("LLM_API_KEY", "")
                         or os.environ.get("GEMINI_API_KEY", "")
                         or llm.get("api_key", "")),
            "model":    llm.get("model", "gemini-2.0-flash-lite"),
            "base_url": llm.get("base_url", ""),
        }

    # Обратная совместимость: старый формат "gemini": {...}
    g = raw.get("gemini", {})
    return {
        "provider": "gemini",
        "api_key":  (os.environ.get("GEMINI_API_KEY", "")
                     or g.get("api_key", "")),
        "model":    g.get("model", "gemini-2.0-flash-lite"),
        "base_url": "",
    }


def is_available() -> bool:
    """True, если LLM сконфигурирован и ключ задан."""
    cfg = _load_config()
    key = cfg["api_key"]
    return bool(key) and not key.startswith("<")


def provider_info() -> str:
    """Читаемая строка с описанием текущего провайдера."""
    cfg = _load_config()
    p = cfg["provider"]
    m = cfg["model"]
    url = cfg["base_url"]
    if p == "gemini":
        return f"Gemini ({m})"
    base = url or "https://api.openai.com/v1"
    return f"OpenAI-compatible ({m} @ {base})"


# ──────────────────────────────────────────────────────────────
#  ОПРЕДЕЛЕНИЕ ТИПА ОШИБКИ
# ──────────────────────────────────────────────────────────────

def _is_rate_limit(exc) -> bool:
    """Detects provider-specific rate-limit errors from exception text.

    Input:
      - exc: exception object raised by provider client.
    Output:
      - True for rate-limit style errors (429 and equivalents).
    """
    s = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None) or 0
    return (code == 429 or "429" in str(exc)
            or "quota" in s or "resource exhausted" in s
            or "too many requests" in s or "rate" in s)


def _is_server_error(exc) -> bool:
    """Detects transient 5xx server-side provider errors.

    Input:
      - exc: exception object raised by provider client.
    Output:
      - True if error appears to be transient server failure.
    """
    s = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None) or 0
    return ((isinstance(code, int) and 500 <= code < 600)
            or any(f" {c}" in str(exc) for c in ["500", "502", "503", "504"])
            or "server error" in s or "internal error" in s)


def _parse_retry_delay(exc) -> int:
    """Извлекает retryDelay из тела ошибки Gemini/OpenAI, если есть."""
    s = str(exc)
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s['\"]", s)
    if m:
        return max(1, int(float(m.group(1))) + 1)
    m = re.search(r"retry.{0,10}(\d+(?:\.\d+)?)\s*s", s, re.IGNORECASE)
    if m:
        return max(1, int(float(m.group(1))) + 1)
    return 0


# ──────────────────────────────────────────────────────────────
#  GEMINI-ПРОВАЙДЕР
# ──────────────────────────────────────────────────────────────

def _call_gemini(prompt: str, cfg: dict) -> str:
    """Calls Gemini provider and returns plain text response.

    Input:
      - prompt: request prompt text.
      - cfg: provider configuration dictionary.
    Output:
      - Model response text.
    """
    sys.path.insert(0, str(BASE / ".venv_libs"))
    from google import genai
    client = genai.Client(api_key=cfg["api_key"])
    response = client.models.generate_content(model=cfg["model"], contents=prompt)
    return response.text.strip()


# ──────────────────────────────────────────────────────────────
#  OPENAI-СОВМЕСТИМЫЙ ПРОВАЙДЕР (urllib, без доп. зависимостей)
# ──────────────────────────────────────────────────────────────

def _call_openai(prompt: str, cfg: dict) -> str:
    """Calls OpenAI-compatible provider endpoint and returns text response.

    Input:
      - prompt: request prompt text.
      - cfg: provider configuration dictionary including base_url/model/key.
    Output:
      - Model response text.
    """
    base = (cfg["base_url"] or "https://api.openai.com/v1").rstrip("/")
    url  = f"{base}/chat/completions"
    payload = json.dumps({
        "model":    cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"].strip()


# ──────────────────────────────────────────────────────────────
#  ОСНОВНОЙ ВЫЗОВ С RETRY
# ──────────────────────────────────────────────────────────────

def call(prompt: str, max_retries: int = 6, initial_wait: int = 65) -> str:
    """Отправляет prompt LLM, возвращает ответ (str).

    При ошибке 429/5xx - экспоненциальный backoff. Возбуждает RuntimeError
    если все попытки исчерпаны.
    """
    cfg  = _load_config()
    provider = cfg["provider"]
    wait = initial_wait

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "gemini":
                return _call_gemini(prompt, cfg)
            else:
                return _call_openai(prompt, cfg)

        except Exception as e:
            if _is_rate_limit(e) and attempt < max_retries:
                suggested   = _parse_retry_delay(e)
                actual_wait = max(suggested + 2, wait) if suggested else wait
                print(f"    [429] Rate limit. Попытка {attempt}/{max_retries}. Ждём {actual_wait}с...")
                deadline = time.time() + actual_wait
                while True:
                    left = int(deadline - time.time())
                    if left <= 0:
                        break
                    print(f"    > {left}с...    ", end="\r", flush=True)
                    time.sleep(min(5, left))
                print()
                wait = min(wait * 2, 300)

            elif _is_server_error(e) and attempt < max_retries:
                pause = min(wait, 30)
                print(f"    [5xx] Попытка {attempt}: {e}. Ждём {pause}с...")
                time.sleep(pause)
                wait = min(wait * 2, 300)

            else:
                print(f"    [ERR] {type(e).__name__}: {e}")
                if attempt >= max_retries:
                    raise RuntimeError(f"LLM call failed after {max_retries} attempts") from e

    raise RuntimeError("LLM call failed")
