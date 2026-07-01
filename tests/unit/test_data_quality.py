"""Tests for run-level data quality diagnostics."""

from alphaevo.data.adapter import DataSourceHealth
from alphaevo.data.quality import build_data_quality_report
from alphaevo.models.execution import EventContextMetrics


def test_event_proxy_dominant_blocks_strategy_iteration() -> None:
    report = build_data_quality_report(
        event_context=EventContextMetrics(
            total_symbols=10,
            provider_symbols=1,
            proxy_symbols=9,
            provider_coverage=0.10,
            proxy_only_coverage=0.90,
            relevant_indicators=["negative_news_score"],
        ),
        data_source_health=[],
    )

    assert report.risk_level == "high"
    assert report.should_prioritize_data_quality is True
    assert report.blocks_strategy_iteration is True
    assert {finding.category for finding in report.findings} == {
        "low_event_provider_coverage",
        "proxy_dominant_event_context",
    }


def test_source_recovered_errors_trigger_data_quality_playbook_without_blocking() -> None:
    report = build_data_quality_report(
        data_source_health=[
            DataSourceHealth(
                name="tencent",
                priority=0,
                success_count=4,
                failure_count=1,
                consecutive_failures=0,
                last_error="timeout",
            ),
            DataSourceHealth(
                name="akshare",
                priority=1,
                success_count=0,
            ),
        ],
    )

    assert report.risk_level == "medium"
    assert report.should_prioritize_data_quality is True
    assert report.blocks_strategy_iteration is False
    assert report.findings[0].category == "source_recovered_errors"


def test_clean_ohlcv_run_has_low_risk() -> None:
    report = build_data_quality_report(
        data_source_health=[
            DataSourceHealth(
                name="yfinance",
                priority=0,
                success_count=3,
            )
        ]
    )

    assert report.risk_level == "low"
    assert report.findings == []
