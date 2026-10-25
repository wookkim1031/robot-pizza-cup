"""The Anthropic API key for the agent: ANTHROPIC_API_KEY, else the file the web UI saves it to.

The file lives outside the repo (~/.config/pizza-ui/anthropic_api_key, mode 600), so it is never committed.
"""

from __future__ import annotations

import os
from pathlib import Path

KEY_FILE = Path.home() / ".config" / "pizza-ui" / "anthropic_api_key"


def load() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return KEY_FILE.read_text().strip() or None
    except OSError:
        return None


def save(key: str) -> None:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.touch(mode=0o600, exist_ok=True)
    KEY_FILE.chmod(0o600)
    KEY_FILE.write_text(key.strip() + "\n")


def clear() -> None:
    KEY_FILE.unlink(missing_ok=True)


def masked(key: str | None) -> str:
    return f"{key[:7]}…{key[-4:]}" if key and len(key) > 12 else ("set" if key else "")
