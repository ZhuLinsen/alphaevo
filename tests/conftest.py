"""Global pytest fixtures ensuring tests never mutate user ~/.alphaevo state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from alphaevo.core.config import ConfigManager

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_alphaevo_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect HOME and ConfigManager user paths to an isolated temporary directory."""
    fake_home = tmp_path / "alphaevo_home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(ConfigManager, "USER_CONFIG_DIR", fake_home / ".alphaevo")
    monkeypatch.setattr(ConfigManager, "USER_CONFIG_FILE", fake_home / ".alphaevo" / "config.yaml")
