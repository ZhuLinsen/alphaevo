"""Tests for configuration management."""

from pathlib import Path

import pytest
import yaml

from alphaevo.core.config import (
    AppConfig,
    ConfigManager,
    _deep_merge,
    _read_env_file,
    _read_yaml_config,
    _set_nested,
)


class TestAppConfig:
    """Test AppConfig defaults and validation."""

    def test_default_config(self) -> None:
        config = AppConfig()
        assert config.llm.model == "gemini/gemini-3.8-flash"
        assert config.data.adapter == "auto"
        assert config.backtest.slippage == 0.001
        assert config.backtest.commission == 0.0003
        assert config.backtest.fill_policy == "conservative"
        assert config.backtest.walk_forward_folds == 3
        assert config.backtest.walk_forward_train_pct == 0.7
        assert config.backtest.walk_forward_pass_gap == 0.10
        assert config.backtest.stress_window_days == 20
        assert config.backtest.stress_window_top_k == 3
        assert config.data.source_failure_threshold == 3
        assert config.data.source_cooldown_seconds == 60.0
        assert config.evolution.max_rounds == 5
        assert config.evolution.max_changes_per_round == 3

    def test_custom_config(self) -> None:
        config = AppConfig(
            llm={"model": "openai/gpt-6.1-sol"},
            data={"adapter": "akshare"},
        )
        assert config.llm.model == "openai/gpt-6.1-sol"
        assert config.data.adapter == "akshare"

    def test_db_path_default(self) -> None:
        config = AppConfig()
        assert str(config.db_path).endswith(".alphaevo/alphaevo.db")

    def test_reflect_model_defaults_to_none(self) -> None:
        config = AppConfig()
        assert config.llm.reflect_model is None


class TestHelpers:
    """Test internal helper functions."""

    def test_set_nested_simple(self) -> None:
        d: dict = {}
        _set_nested(d, "llm.model", "test-model")
        assert d == {"llm": {"model": "test-model"}}

    def test_set_nested_top_level(self) -> None:
        d: dict = {}
        _set_nested(d, "db_path", "/tmp/test.db")
        assert d == {"db_path": "/tmp/test.db"}

    def test_deep_merge(self) -> None:
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}

    def test_deep_merge_no_mutation(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}  # original unchanged

    def test_read_yaml_config_missing(self, tmp_path: Path) -> None:
        result = _read_yaml_config(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_read_yaml_config_valid(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  model: test-model\n")
        result = _read_yaml_config(config_file)
        assert result == {"llm": {"model": "test-model"}}

    def test_read_yaml_config_invalid(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("{{invalid yaml")
        result = _read_yaml_config(config_file)
        assert result == {}

    def test_read_env_file_valid(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# comment\n"
            "ALPHAEVO_LLM_MODEL=gemini/gemini-3.8-flash\n"
            "export ALPHAEVO_DATA_ADAPTER=akshare\n"
            "ALPHAEVO_API_KEY='test-key'\n"
        )
        result = _read_env_file(env_file)
        assert result == {
            "ALPHAEVO_LLM_MODEL": "gemini/gemini-3.8-flash",
            "ALPHAEVO_DATA_ADAPTER": "akshare",
            "ALPHAEVO_API_KEY": "test-key",
        }


class TestConfigManager:
    """Test ConfigManager load/save."""

    def test_load_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ALPHAEVO_LLM_MODEL", raising=False)
        monkeypatch.delenv("ALPHAEVO_DATA_ADAPTER", raising=False)
        mgr = ConfigManager()
        # Prevent loading real user config from ~/.alphaevo/config.yaml
        monkeypatch.setattr(mgr, "USER_CONFIG_FILE", tmp_path / "no_such_config.yaml")
        config = mgr.load()
        assert isinstance(config, AppConfig)
        assert config.llm.model == "gemini/gemini-3.8-flash"
        assert config.data.adapter == "auto"

    def test_load_with_cli_overrides(self) -> None:
        mgr = ConfigManager()
        config = mgr.load(cli_overrides={"llm": {"model": "test/model"}})
        assert config.llm.model == "test/model"

    def test_load_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHAEVO_LLM_MODEL", "env-model")
        monkeypatch.setenv("ALPHAEVO_DATA_ADAPTER", "akshare")
        monkeypatch.setenv("ALPHAEVO_BACKTEST_WALK_FORWARD_FOLDS", "4")
        monkeypatch.setenv("ALPHAEVO_BACKTEST_WALK_FORWARD_TRAIN_PCT", "0.75")
        monkeypatch.setenv("ALPHAEVO_BACKTEST_WALK_FORWARD_PASS_GAP", "0.08")
        monkeypatch.setenv("ALPHAEVO_BACKTEST_FILL_POLICY", "close_first")
        monkeypatch.setenv("ALPHAEVO_BACKTEST_STRESS_WINDOW_DAYS", "15")
        monkeypatch.setenv("ALPHAEVO_BACKTEST_STRESS_WINDOW_TOP_K", "2")
        monkeypatch.setenv("ALPHAEVO_DATA_SOURCE_FAILURE_THRESHOLD", "2")
        monkeypatch.setenv("ALPHAEVO_DATA_SOURCE_COOLDOWN_SECONDS", "15.5")
        mgr = ConfigManager()
        config = mgr.load()
        assert config.llm.model == "env-model"
        assert config.data.adapter == "akshare"
        assert config.backtest.walk_forward_folds == 4
        assert config.backtest.walk_forward_train_pct == 0.75
        assert config.backtest.walk_forward_pass_gap == 0.08
        assert config.backtest.fill_policy == "close_first"
        assert config.backtest.stress_window_days == 15
        assert config.backtest.stress_window_top_k == 2
        assert config.data.source_failure_threshold == 2
        assert config.data.source_cooldown_seconds == 15.5

    def test_load_from_project_env_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".env").write_text(
            "ALPHAEVO_LLM_MODEL=dotenv-model\nALPHAEVO_DATA_ADAPTER=akshare\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ALPHAEVO_LLM_MODEL", raising=False)
        monkeypatch.delenv("ALPHAEVO_DATA_ADAPTER", raising=False)

        mgr = ConfigManager()
        config = mgr.load()

        assert config.llm.model == "dotenv-model"
        assert config.data.adapter == "akshare"

    def test_cli_overrides_beat_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHAEVO_LLM_MODEL", "env-model")
        mgr = ConfigManager()
        config = mgr.load(cli_overrides={"llm": {"model": "cli-model"}})
        assert config.llm.model == "cli-model"

    def test_exported_env_beats_dotenv(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".env").write_text("ALPHAEVO_LLM_MODEL=dotenv-model\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("ALPHAEVO_LLM_MODEL", "env-model")

        mgr = ConfigManager()
        config = mgr.load()

        assert config.llm.model == "env-model"

    def test_save_and_load_user_config(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mgr = ConfigManager()
        monkeypatch.setattr(ConfigManager, "USER_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(ConfigManager, "USER_CONFIG_FILE", tmp_path / "config.yaml")

        config = AppConfig(llm={"model": "saved-model"}, data={"adapter": "akshare"})
        mgr.save_user_config(config)

        # Verify file was created
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert saved["llm"]["model"] == "saved-model"
        assert saved["data"]["adapter"] == "akshare"

    def test_save_excludes_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        mgr = ConfigManager()
        monkeypatch.setattr(ConfigManager, "USER_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(ConfigManager, "USER_CONFIG_FILE", tmp_path / "config.yaml")

        # Save default config — should result in minimal file
        config = AppConfig()
        mgr.save_user_config(config)
        saved = yaml.safe_load((tmp_path / "config.yaml").read_text())
        # Default-only config should produce None or empty
        assert saved is None or saved == {}

    def test_get_llm_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHAEVO_API_KEY", "sk-test-key")
        mgr = ConfigManager()
        assert mgr.get_llm_api_key() == "sk-test-key"

    def test_get_llm_api_key_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ALPHAEVO_API_KEY", raising=False)
        mgr = ConfigManager()
        assert mgr.get_llm_api_key() is None

    def test_get_llm_api_key_from_project_env_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (tmp_path / ".env").write_text("ALPHAEVO_API_KEY=dotenv-key\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ALPHAEVO_API_KEY", raising=False)

        mgr = ConfigManager()

        assert mgr.get_llm_api_key() == "dotenv-key"

    def test_ensure_dirs(self, tmp_path: Path) -> None:
        config = AppConfig(
            data={"cache_dir": tmp_path / "cache"},
            db_path=tmp_path / "db" / "test.db",
            strategies_dir=tmp_path / "strategies",
        )
        mgr = ConfigManager()
        mgr.ensure_dirs(config)
        assert (tmp_path / "cache").is_dir()
        assert (tmp_path / "db").is_dir()
        assert (tmp_path / "strategies").is_dir()
