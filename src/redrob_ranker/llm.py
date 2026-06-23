"""
llm.py — thin abstraction over FREE LLM providers.

IMPORTANT: this is used ONLY in offline steps that are NOT part of the timed,
no-network ranking run:
  * precompute.py  — (optionally) regenerate/augment config/role_spec.yaml from
                     the JD text.
  * reasoning.py   — (optionally) polish the top-100 reasoning strings into
                     natural language, fed ONLY profile-derived facts.

rank.py never calls this. The default provider is "none", so the whole project
runs with zero API keys and zero network. Enable a free provider via env vars:

  Ollama (fully local, no key):   LLM_PROVIDER=ollama  LLM_MODEL=llama3.1
  Google Gemini (free tier):      LLM_PROVIDER=gemini  GEMINI_API_KEY=...   [LLM_MODEL=gemini-2.5-flash]
  Groq (free tier):               LLM_PROVIDER=groq    GROQ_API_KEY=...     [LLM_MODEL=llama-3.1-8b-instant]

These can be set as real environment variables or placed in a local `.env`
file in the repo root (see .env.example); `load_dotenv()` reads it automatically.

All transport uses the Python standard library (urllib) so there are no extra
dependencies. Any failure returns None and callers fall back to deterministic
templates — the system degrades gracefully, never crashes.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: int = 60) -> Optional[dict]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
        return None


_DOTENV_LOADED = False


def load_dotenv(path: str = ".env") -> bool:
    """Load KEY=value pairs from a .env file into os.environ.

    Stdlib-only (no python-dotenv dependency). Variables already present in the
    real environment take precedence, so an exported var always wins over the
    file. Lines may be blank, `# comments`, or `export KEY=value`; surrounding
    quotes on the value are stripped. Returns True if a file was read.
    """
    global _DOTENV_LOADED
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return False
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
    _DOTENV_LOADED = True
    return True


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        # Pick up a local .env automatically (real env vars still take precedence).
        if not _DOTENV_LOADED:
            load_dotenv()
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "none")).lower()
        self.model = model or os.environ.get("LLM_MODEL", "")

    @property
    def enabled(self) -> bool:
        return self.provider not in ("none", "", None)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 256,
                 temperature: float = 0.3) -> Optional[str]:
        """Return generated text, or None on any failure / if disabled."""
        if not self.enabled:
            return None
        try:
            if self.provider == "ollama":
                return self._ollama(prompt, system, max_tokens, temperature)
            if self.provider == "gemini":
                return self._gemini(prompt, system, max_tokens, temperature)
            if self.provider == "groq":
                return self._groq(prompt, system, max_tokens, temperature)
        except Exception:
            return None
        return None

    # ---- providers ---------------------------------------------------------
    def _ollama(self, prompt, system, max_tokens, temperature):
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        model = self.model or "llama3.1"
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        out = _post_json(f"{host}/api/generate", payload)
        return (out or {}).get("response", "").strip() or None

    def _gemini(self, prompt, system, max_tokens, temperature):
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            return None
        model = self.model or "gemini-2.5-flash"
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        text = (system + "\n\n" + prompt) if system else prompt
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        out = _post_json(url, payload)
        try:
            return out["candidates"][0]["content"]["parts"][0]["text"].strip() or None
        except (KeyError, IndexError, TypeError):
            return None

    def _groq(self, prompt, system, max_tokens, temperature):
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            return None
        model = self.model or "llama-3.1-8b-instant"
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": msgs,
                   "temperature": temperature, "max_tokens": max_tokens}
        out = _post_json("https://api.groq.com/openai/v1/chat/completions", payload,
                         headers={"Authorization": f"Bearer {key}"})
        try:
            return out["choices"][0]["message"]["content"].strip() or None
        except (KeyError, IndexError, TypeError):
            return None
