"""Tests for strategy DSL parser."""

from pathlib import Path

import pytest

from alphaevo.strategy.dsl.parser import StrategyParseError, StrategyParser

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
BUILTIN_DIR = Path(__file__).parent.parent.parent / "strategies" / "builtin"
RESEARCH_DIR = Path(__file__).parent.parent.parent / "strategies" / "research"
EVENT_NEWS_INDICATORS = {
    "negative_news_score",
    "news_sentiment_score",
    "days_since_event",
    "price_above_pre_event",
    "already_overreacted",
}


class TestStrategyParser:
    """Test suite for StrategyParser."""

    def setup_method(self) -> None:
        self.parser = StrategyParser()

    def test_parse_builtin_trend_pullback(self) -> None:
        path = BUILTIN_DIR / "trend_pullback_rebound.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        assert strategy.meta.id == "trend_pullback_rebound_v1"
        assert strategy.meta.category == "trend"
        assert len(strategy.entry.conditions) >= 4
        assert strategy.exit.stop_loss.type == "pct"
        assert strategy.exit.max_holding_days == 10
        assert len(strategy.params.tunable) >= 3
        tunable_targets = {param.target for param in strategy.params.tunable}
        assert "entry.conditions[indicator=relative_strength_20d].indicator" in tunable_targets
        assert "entry.conditions[indicator=volume_ratio_1d_5d].indicator" in tunable_targets

    def test_parse_builtin_mean_reversion(self) -> None:
        path = BUILTIN_DIR / "mean_reversion_oversold.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        assert strategy.meta.id == "mean_reversion_oversold_v1"
        assert strategy.meta.category == "reversal"
        assert len(strategy.entry.conditions) >= 3
        tunable_targets = {param.target for param in strategy.params.tunable}
        assert "entry.conditions[indicator=rsi_14].indicator" in tunable_targets
        assert "entry.conditions[indicator=deviation_from_ma20_pct].indicator" in tunable_targets
        assert "exit.take_profit.target" in tunable_targets

    def test_parse_all_builtin_strategies(self) -> None:
        if not BUILTIN_DIR.is_dir():
            pytest.skip("Builtin strategies directory not found")
        strategies = self.parser.parse_directory(BUILTIN_DIR)
        assert len(strategies) >= 4, f"Expected >=4 strategies, got {len(strategies)}"
        ids = [s.meta.id for s in strategies]
        assert "trend_pullback_rebound_v1" in ids
        assert "mean_reversion_oversold_v1" in ids

    def test_parse_builtin_ma_crossover_has_ma_period_tunable(self) -> None:
        path = BUILTIN_DIR / "ma_crossover.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        tunable_targets = {param.target for param in strategy.params.tunable}
        assert (
            "entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast" in tunable_targets
        )
        assert (
            "entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow" in tunable_targets
        )
        assert "entry.conditions[indicator=close_above_ma20].indicator" in tunable_targets
        assert "entry.conditions[indicator=rsi_14].indicator" in tunable_targets
        assert "exit.max_holding_days" in tunable_targets

    def test_parse_builtin_volatility_compression_breakout(self) -> None:
        path = BUILTIN_DIR / "volatility_compression_breakout.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        diagnostics = self.parser.diagnose(strategy)
        tunable_targets = {param.target for param in strategy.params.tunable}

        assert diagnostics.errors == []
        assert strategy.meta.id == "volatility_compression_breakout_v1"
        assert strategy.entry.triggers[0].indicator == "breakout_high_20d"
        assert strategy.exit.take_profit.type == "trailing"
        assert "entry.triggers[indicator=breakout_high_20d].indicator" in tunable_targets
        assert "entry.guards[indicator=price_position_120d].indicator" in tunable_targets
        assert "exit.take_profit.trigger_pct" in tunable_targets
        assert "exit.take_profit.trail_pct" in tunable_targets

    def test_research_ohlcv_trend_pullback_has_no_event_news_filters(self) -> None:
        path = RESEARCH_DIR / "trend_pullback_rebound_ohlcv_only_v13.yaml"
        if not path.exists():
            pytest.skip("Research strategy file not found")

        strategy = self.parser.parse_file(path)
        diagnostics = self.parser.diagnose(strategy)
        active_indicators = {
            condition.indicator
            for condition in [
                *strategy.entry.triggers,
                *strategy.entry.guards,
                *strategy.entry.conditions,
                *strategy.entry.filters,
            ]
        }

        assert diagnostics.errors == []
        assert strategy.meta.id == "trend_pullback_rebound_ohlcv_only_v13"
        assert active_indicators.isdisjoint(EVENT_NEWS_INDICATORS)
        assert strategy.entry.filters == []

    def test_parse_invalid_yaml(self) -> None:
        with pytest.raises(StrategyParseError):
            self.parser.parse_yaml("{{invalid yaml")

    def test_parse_missing_entry(self) -> None:
        yaml_content = """
meta:
  id: test_v1
  name: test
  version: 1
  category: trend
description: test
exit:
  stop_loss:
    type: pct
    value: 0.04
  take_profit:
    type: rr
    value: 2.0
"""
        with pytest.raises(StrategyParseError):
            self.parser.parse_yaml(yaml_content)

    def test_validate_too_many_conditions(self) -> None:
        path = BUILTIN_DIR / "trend_pullback_rebound.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        warnings = self.parser.validate(strategy)
        # Should not warn for reasonable number of conditions
        overfit_warnings = [w for w in warnings if "overfitting" in w.lower()]
        assert len(overfit_warnings) == 0

    def test_diagnose_unknown_indicator_is_error(self) -> None:
        yaml_content = """
meta:
  id: diag_bad_v1
  name: Bad Strategy
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: no_such_indicator
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == ["Unknown indicator: no_such_indicator"]
        assert diagnostics.is_valid is False

    def test_diagnose_templated_ma_indicator_is_valid(self) -> None:
        yaml_content = """
meta:
  id: ma_template_v1
  name: MA Template
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: close_above_ma180
      op: "=="
      value: true
    - indicator: ma50_above_ma180
      op: "=="
      value: true
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: target_ma, target: ma50}
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []
        assert diagnostics.is_valid is True

    def test_diagnose_invalid_target_ma_target(self) -> None:
        yaml_content = """
meta:
  id: bad_target_ma_v1
  name: Bad Target MA
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: target_ma, target: ema20}
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == ["target_ma take profit target must look like ma20 or ma180"]

    def test_tunable_indicator_period_target_is_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_ma_period_v1
  name: Tunable MA Period
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: close_above_ma60
      op: "=="
      value: true
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: target_ma, target: ma60}
params:
  tunable:
    - target: entry.conditions[indicator=close_above_ma60].indicator
      range: [20, 120]
      step: 5
    - target: exit.take_profit.target
      range: [20, 120]
      step: 5
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []

    def test_tunable_entry_triggers_and_guards_are_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_entry_v5
  name: Tunable Entry v5
  version: 1
  category: trend
description: test
entry:
  triggers:
    - indicator: rsi_14
      op: "<"
      value: 30
  guards:
    - indicator: relative_strength_20d
      op: ">"
      value: 0.08
exit:
  stop_loss:
    type: pct
    value: 0.04
  take_profit:
    type: rr
    value: 2.0
params:
  tunable:
    - target: entry.triggers[indicator=rsi_14].value
      range: [20, 40]
      step: 1
    - target: entry.guards[indicator=relative_strength_20d].indicator
      range: [10, 60]
      step: 5
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []

    def test_tunable_window_indicator_period_targets_are_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_window_period_v1
  name: Tunable Window Period
  version: 1
  category: reversal
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: "<"
      value: 30
    - indicator: volume_ratio_1d_5d
      op: ">"
      value: 1.2
    - indicator: relative_strength_20d
      op: ">"
      value: 0.05
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=rsi_14].indicator
      range: [7, 21]
      step: 1
    - target: entry.conditions[indicator=volume_ratio_1d_5d].indicator
      range: [3, 20]
      step: 1
    - target: entry.conditions[indicator=relative_strength_20d].indicator
      range: [10, 60]
      step: 5
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []

    def test_tunable_atr_and_bollinger_period_targets_are_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_volatility_period_v1
  name: Tunable Volatility Period
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: atr
      op: ">"
      value: 0.5
    - indicator: bollinger_band_width
      op: "<"
      value: 0.2
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=atr].indicator
      range: [7, 21]
      step: 1
    - target: entry.conditions[indicator=bollinger_band_width].indicator
      range: [10, 40]
      step: 5
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []

    def test_tunable_bollinger_std_target_is_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_bollinger_std_v1
  name: Tunable Bollinger Std
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: bollinger_band_width
      op: "<"
      value: 0.2
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=bollinger_band_width].indicator.std
      range: [1.0, 3.0]
      step: 0.5
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []

    def test_tunable_atr_stop_loss_period_is_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_atr_stop_v1
  name: Tunable ATR Stop
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 0
exit:
  stop_loss:
    type: atr
    multiplier: 2.0
    atr_period: 21
  take_profit:
    type: rr
    value: 2.0
params:
  tunable:
    - target: exit.stop_loss.atr_period
      range: [7, 30]
      step: 1
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []
        assert strategy.exit.stop_loss.atr_period == 21

    def test_tunable_dual_ma_period_targets_are_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_dual_ma_v1
  name: Tunable Dual MA
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: ma5_ge_ma10_or_crossing
      op: "=="
      value: true
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast
      range: [3, 8]
      step: 1
    - target: entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow
      range: [9, 20]
      step: 1
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []

    def test_tunable_macd_component_targets_are_valid(self) -> None:
        yaml_content = """
meta:
  id: tunable_macd_v1
  name: Tunable MACD
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: macd_histogram
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=macd_histogram].indicator.fast
      range: [6, 18]
      step: 1
    - target: entry.conditions[indicator=macd_histogram].indicator.slow
      range: [20, 40]
      step: 1
    - target: entry.conditions[indicator=macd_histogram].indicator.signal
      range: [5, 15]
      step: 1
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == []

    def test_tunable_indicator_period_rejects_unsupported_indicator_shape(self) -> None:
        yaml_content = """
meta:
  id: bad_tunable_ma_pair_v1
  name: Bad Tunable MA Pair
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: ma50_above_ma180
      op: "=="
      value: true
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=ma50_above_ma180].indicator
      range: [20, 120]
      step: 5
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == [
            "Tunable param 'entry.conditions[indicator=ma50_above_ma180].indicator': target does not resolve to a tunable value"
        ]

    def test_tunable_dual_ma_component_rejects_single_ma_indicator(self) -> None:
        yaml_content = """
meta:
  id: bad_dual_component_v1
  name: Bad Dual Component
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: close_above_ma20
      op: "=="
      value: true
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=close_above_ma20].indicator.fast
      range: [3, 8]
      step: 1
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == [
            "Tunable param 'entry.conditions[indicator=close_above_ma20].indicator.fast': target does not resolve to a tunable value"
        ]

    def test_tunable_macd_signal_component_rejects_non_macd_indicator(self) -> None:
        yaml_content = """
meta:
  id: bad_macd_signal_v1
  name: Bad MACD Signal
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 50
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=rsi_14].indicator.signal
      range: [5, 15]
      step: 1
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == [
            "Tunable param 'entry.conditions[indicator=rsi_14].indicator.signal': target does not resolve to a tunable value"
        ]

    def test_tunable_bollinger_std_component_rejects_non_bollinger_indicator(self) -> None:
        yaml_content = """
meta:
  id: bad_bollinger_std_v1
  name: Bad Bollinger Std
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 50
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
params:
  tunable:
    - target: entry.conditions[indicator=rsi_14].indicator.std
      range: [1.0, 3.0]
      step: 0.5
"""
        strategy = self.parser.parse_yaml(yaml_content)
        diagnostics = self.parser.diagnose(strategy)
        assert diagnostics.errors == [
            "Tunable param 'entry.conditions[indicator=rsi_14].indicator.std': target does not resolve to a tunable value"
        ]

    def test_assert_valid_strict_rejects_warnings(self) -> None:
        yaml_content = """
meta:
  id: diag_warn_v1
  name: Warn Strategy
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
  max_holding_days: 120
"""
        strategy = self.parser.parse_yaml(yaml_content)
        with pytest.raises(StrategyParseError, match="unusually long"):
            self.parser.assert_valid(strategy, strict=True)

    def test_complexity_score_computed(self) -> None:
        path = BUILTIN_DIR / "trend_pullback_rebound.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        assert 0.0 <= strategy.complexity_score <= 1.0

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            self.parser.parse_file(Path("/nonexistent/strategy.yaml"))

    def test_family_id_computed(self) -> None:
        path = BUILTIN_DIR / "trend_pullback_rebound.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        assert strategy.meta.family_id == "trend_pullback_rebound"

    # ── DSL v0.3 features ─────────────────────────────────────────────

    def test_parse_preferred_regime(self) -> None:
        path = BUILTIN_DIR / "trend_pullback_rebound.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        assert "trending_up" in strategy.meta.preferred_regime

    def test_preferred_regime_defaults_empty(self) -> None:
        yaml_content = """
meta:
  id: no_regime_v1
  name: No Regime
  version: 1
  category: trend
description: Strategy without preferred_regime
entry:
  conditions:
    - indicator: rsi_14
      op: "<"
      value: 30
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
"""
        strategy = self.parser.parse_yaml(yaml_content)
        assert strategy.meta.preferred_regime == []

    def test_parse_execution_config(self) -> None:
        path = BUILTIN_DIR / "rsi_reversion.yaml"
        if not path.exists():
            pytest.skip("English strategy template not found")
        strategy = self.parser.parse_file(path)
        assert strategy.entry.execution is not None
        assert strategy.entry.execution.timing == "next_open"

    def test_execution_config_defaults_none(self) -> None:
        yaml_content = """
meta:
  id: no_exec_v1
  name: No Execution Config
  version: 1
  category: trend
description: Strategy without execution config
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 0
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
"""
        strategy = self.parser.parse_yaml(yaml_content)
        assert strategy.entry.execution is None

    def test_parse_execution_config_with_slippage(self) -> None:
        yaml_content = """
meta:
  id: exec_slip_v1
  name: Execution With Slippage
  version: 1
  category: trend
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 0
  execution:
    timing: close
    slippage: 0.002
exit:
  stop_loss: {type: pct, value: 0.05}
  take_profit: {type: rr, value: 2.0}
"""
        strategy = self.parser.parse_yaml(yaml_content)
        assert strategy.entry.execution is not None
        assert strategy.entry.execution.timing == "close"
        assert strategy.entry.execution.slippage == 0.002

    def test_composite_stop_loss_standard_format(self) -> None:
        path = BUILTIN_DIR / "sector_rotation_leader.yaml"
        if not path.exists():
            pytest.skip("Builtin strategy file not found")
        strategy = self.parser.parse_file(path)
        assert strategy.exit.stop_loss.type == "composite"
        assert strategy.exit.stop_loss.conditions is not None
        for cond in strategy.exit.stop_loss.conditions:
            # All should be StrategyCondition, not dict
            from alphaevo.models.strategy import StrategyCondition

            assert isinstance(cond, StrategyCondition)

    def test_composite_stop_loss_legacy_format(self) -> None:
        """Legacy {type: ..., threshold: ...} format converted to StrategyCondition."""
        yaml_content = """
meta:
  id: legacy_composite_v1
  name: Legacy Composite
  version: 1
  category: rotation
description: test
entry:
  conditions:
    - indicator: rsi_14
      op: ">"
      value: 0
exit:
  stop_loss:
    type: composite
    conditions:
    - type: sector_rank_exit
      threshold: 10
    - type: price_below_ma
      threshold: 5
  take_profit: {type: pct, value: 0.10}
"""
        strategy = self.parser.parse_yaml(yaml_content)
        sl = strategy.exit.stop_loss
        assert sl.type == "composite"
        assert len(sl.conditions) == 2
        from alphaevo.models.strategy import StrategyCondition

        for cond in sl.conditions:
            assert isinstance(cond, StrategyCondition)
        assert sl.conditions[0].indicator == "sector_rank_exit"
        assert sl.conditions[0].value == 10
        assert sl.conditions[1].indicator == "price_below_ma"
        assert sl.conditions[1].value == 5

    def test_parse_english_strategies(self) -> None:
        """English strategy templates should parse cleanly."""
        for name in ["rsi_reversion.yaml", "ma_crossover.yaml"]:
            path = BUILTIN_DIR / name
            if not path.exists():
                pytest.skip(f"English template {name} not found")
            strategy = self.parser.parse_file(path)
            assert strategy.meta.market == "us"
            assert len(strategy.meta.preferred_regime) > 0
            assert strategy.entry.execution is not None

    def test_serializer_roundtrip_v03_features(self) -> None:
        """Serializer preserves v0.3 features (execution, preferred_regime)."""
        from alphaevo.strategy.dsl.serializer import StrategySerializer

        path = BUILTIN_DIR / "rsi_reversion.yaml"
        if not path.exists():
            pytest.skip("English strategy template not found")

        original = self.parser.parse_file(path)
        serializer = StrategySerializer()
        yaml_out = serializer.to_yaml(original)
        roundtripped = self.parser.parse_yaml(yaml_out)

        assert roundtripped.meta.preferred_regime == original.meta.preferred_regime
        assert roundtripped.entry.execution is not None
        assert roundtripped.entry.execution.timing == original.entry.execution.timing
