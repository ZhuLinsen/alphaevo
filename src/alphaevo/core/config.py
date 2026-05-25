"""Unified configuration management for AlphaEvo.

Priority (high → low):
  1. CLI parameters (passed as overrides)
  2. Environment variables (ALPHAEVO_*)
  3. Project config file (.alphaevo/config.yaml in cwd)
  4. User config file (~/.alphaevo/config.yaml)
  5. Built-in defaults
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    model: str = "gemini/gemini-2.0-flash"
    reflect_model: str | None = None  # None = reuse model
    base_url: str | None = None
    timeout: int = 120
    max_retries: int = 2


class DataConfig(BaseModel):
    """Data source configuration."""

    adapter: str = "yfinance"  # core: yfinance / akshare; optional bridge: dsa
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".alphaevo" / "cache")
    dsa_path: str | None = None  # optional daily_stock_analysis bridge path
    adanos_api_key: str | None = None  # optional Adanos event/news sentiment bridge
    adanos_base_url: str = "https://api.adanos.org"


class BacktestConfig(BaseModel):
    """Backtest engine configuration."""

    slippage: float = 0.001
    commission: float = 0.0003
    default_timing: str = "next_open"  # next_open / close / breakout_high
    fill_policy: str = "conservative"  # conservative / optimistic / close_first
    min_data_days: int = 30  # minimum trading days required for backtest
    walk_forward_folds: int = Field(default=3, ge=2)
    walk_forward_train_pct: float = Field(default=0.7, ge=0.5, lt=1.0)
    walk_forward_pass_gap: float = Field(default=0.10, ge=0.0, le=1.0)
    stress_window_days: int = Field(default=20, ge=5)
    stress_window_top_k: int = Field(default=3, ge=1)


class EvolutionConfig(BaseModel):
    """Evolution pipeline configuration."""

    max_rounds: int = 5
    max_changes_per_round: int = 3
    min_signal_count: int = 30  # minimum signals for valid evaluation
    complexity_limit: int = 8  # max entry conditions before forced simplification
    num_candidates: int = 3  # number of candidate experiments per reflection round
    auto_expand_samples: bool = True
    max_sample_expansions: int = 3
    sample_expansion_window_days: int = 180
    sample_expansion_symbol_step: int = 20


class AppConfig(BaseModel):
    """Root configuration for AlphaEvo.

    Assembled from multiple sources by ConfigManager.load().
    """

    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    db_path: Path = Field(default_factory=lambda: Path.home() / ".alphaevo" / "alphaevo.db")
    strategies_dir: Path = Field(default_factory=lambda: Path.home() / ".alphaevo" / "strategies")


# ── Environment variable mapping ────────────────────────────────────

_ENV_MAP: dict[str, str] = {
    "ALPHAEVO_LLM_MODEL": "llm.model",
    "ALPHAEVO_LLM_REFLECT_MODEL": "llm.reflect_model",
    "ALPHAEVO_LLM_BASE_URL": "llm.base_url",
    "ALPHAEVO_DATA_ADAPTER": "data.adapter",
    "ALPHAEVO_CACHE_DIR": "data.cache_dir",
    "ALPHAEVO_DSA_PATH": "data.dsa_path",
    "ADANOS_API_KEY": "data.adanos_api_key",
    "ALPHAEVO_ADANOS_API_KEY": "data.adanos_api_key",
    "ADANOS_API_BASE_URL": "data.adanos_base_url",
    "ALPHAEVO_ADANOS_BASE_URL": "data.adanos_base_url",
    "ALPHAEVO_DB_PATH": "db_path",
    "ALPHAEVO_BACKTEST_WALK_FORWARD_FOLDS": "backtest.walk_forward_folds",
    "ALPHAEVO_BACKTEST_WALK_FORWARD_TRAIN_PCT": "backtest.walk_forward_train_pct",
    "ALPHAEVO_BACKTEST_WALK_FORWARD_PASS_GAP": "backtest.walk_forward_pass_gap",
    "ALPHAEVO_BACKTEST_FILL_POLICY": "backtest.fill_policy",
    "ALPHAEVO_BACKTEST_STRESS_WINDOW_DAYS": "backtest.stress_window_days",
    "ALPHAEVO_BACKTEST_STRESS_WINDOW_TOP_K": "backtest.stress_window_top_k",
    "ALPHAEVO_EVOLUTION_MIN_SIGNAL_COUNT": "evolution.min_signal_count",
    "ALPHAEVO_EVOLUTION_AUTO_EXPAND_SAMPLES": "evolution.auto_expand_samples",
    "ALPHAEVO_EVOLUTION_MAX_SAMPLE_EXPANSIONS": "evolution.max_sample_expansions",
    "ALPHAEVO_EVOLUTION_SAMPLE_EXPANSION_WINDOW_DAYS": "evolution.sample_expansion_window_days",
    "ALPHAEVO_EVOLUTION_SAMPLE_EXPANSION_SYMBOL_STEP": "evolution.sample_expansion_symbol_step",
}


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using dotted key like 'llm.model'.

    Attempts numeric coercion for string values from environment variables
    so Pydantic validators (e.g. ``ge=0.5``) receive the correct type.
    """
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "on"}:
            value = True
        elif lowered in {"false", "no", "off"}:
            value = False
        else:
            # Try int first, then float — keep as string if neither works
            for converter in (int, float):
                try:
                    value = converter(value)
                    break
                except (ValueError, TypeError):
                    continue
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _read_yaml_config(path: Path) -> dict[str, Any]:
    """Read a YAML config file, return empty dict if missing/invalid."""
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError):
        return {}


def _read_env_file(path: Path) -> dict[str, str]:
    """Read a simple KEY=VALUE .env file without extra dependencies."""
    if not path.is_file():
        return {}

    env: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            env[key] = value
    except OSError:
        return {}

    return env


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base (override wins)."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ConfigManager:
    """Load and manage AlphaEvo configuration."""

    USER_CONFIG_DIR = Path.home() / ".alphaevo"
    USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"
    PROJECT_CONFIG_DIR = Path(".alphaevo")
    PROJECT_CONFIG_FILE = PROJECT_CONFIG_DIR / "config.yaml"
    PROJECT_ENV_FILE = Path(".env")

    def load(self, cli_overrides: dict[str, Any] | None = None) -> AppConfig:
        """Load config with full priority chain.

        Priority: cli_overrides > env vars > project file > user file > defaults.
        """
        # Start with empty overrides
        merged: dict[str, Any] = {}

        # Layer 1: User config file (~/.alphaevo/config.yaml)
        user_cfg = _read_yaml_config(self.USER_CONFIG_FILE)
        merged = _deep_merge(merged, user_cfg)

        # Layer 2: Project config file (.alphaevo/config.yaml)
        project_cfg = _read_yaml_config(self.PROJECT_CONFIG_FILE)
        merged = _deep_merge(merged, project_cfg)

        # Layer 3: Project .env (acts like env defaults for local development)
        dotenv_cfg = _read_env_file(self.PROJECT_ENV_FILE)
        for env_var, dotted_key in _ENV_MAP.items():
            value = dotenv_cfg.get(env_var)
            if value is not None:
                _set_nested(merged, dotted_key, value)

        # Layer 4: Environment variables
        for env_var, dotted_key in _ENV_MAP.items():
            value = os.environ.get(env_var)
            if value is not None:
                _set_nested(merged, dotted_key, value)

        # Layer 5: CLI overrides (highest priority)
        if cli_overrides:
            merged = _deep_merge(merged, cli_overrides)

        return AppConfig.model_validate(merged)

    def save_user_config(self, config: AppConfig) -> None:
        """Save config to ~/.alphaevo/config.yaml.

        Excludes sensitive fields (api_key) and defaults-only values.
        """
        self.USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = config.model_dump(mode="json", exclude_defaults=True)
        # Never persist API keys to disk
        if "llm" in data:
            data["llm"].pop("api_key", None)
        if "data" in data:
            data["data"].pop("adanos_api_key", None)
        with open(self.USER_CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def ensure_dirs(self, config: AppConfig) -> None:
        """Create necessary directories."""
        config.data.cache_dir.mkdir(parents=True, exist_ok=True)
        config.strategies_dir.mkdir(parents=True, exist_ok=True)
        config.db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_llm_api_key(self) -> str | None:
        """Get LLM API key from env vars, falling back to local project .env."""
        env_value = os.environ.get("ALPHAEVO_API_KEY")
        if env_value:
            return env_value
        return _read_env_file(self.PROJECT_ENV_FILE).get("ALPHAEVO_API_KEY")
