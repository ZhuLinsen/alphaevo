"""Reporter — generates JSON and Markdown reports from evaluation results.

Supports single-strategy reports, multi-strategy comparison tables,
and evolution story reports for sharing.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, cast

from alphaevo.evaluator.maturity import (
    build_research_maturity_report,
    render_research_maturity_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path

    from alphaevo.models.execution import EvaluationReport, TradeSignal
    from alphaevo.models.strategy import Strategy
    from alphaevo.orchestrator.evolution import EvolutionResult


class Reporter:
    """Generate human-readable and machine-readable evaluation reports."""

    @staticmethod
    def _title_case_status(status: str) -> str:
        """Render status keys like ``parameter_misaligned`` for people."""
        return status.replace("_", " ").strip().title()

    @staticmethod
    def _timeout_failures_from_telemetry(telemetry: dict[str, object]) -> int:
        """Count timeout-like failures from serialized LLM telemetry."""
        calls = telemetry.get("calls", [])
        if not isinstance(calls, list):
            return 0
        total = 0
        for call in calls:
            if not isinstance(call, dict):
                continue
            if call.get("success", True):
                continue
            error = str(call.get("error", "")).lower()
            if "timeout" in error or "timed out" in error:
                total += 1
        return total

    @staticmethod
    def _coerce_llm_telemetry(value: object) -> dict[str, object]:
        """Normalize reflection or trajectory telemetry into a plain dict."""
        if value is None:
            return {}
        if hasattr(value, "model_dump"):
            return cast("dict[str, object]", value.model_dump(mode="json"))
        if isinstance(value, dict):
            return cast("dict[str, object]", value)
        return {}

    @staticmethod
    def _coerce_int(value: object, default: int = 0) -> int:
        """Best-effort integer coercion for serialized report values."""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def to_json(
        report: EvaluationReport,
        strategy: Strategy | None = None,
    ) -> str:
        """Convert report to pretty JSON string.

        If *strategy* is provided, its meta and description are included
        under a top-level ``strategy`` key.
        """
        data = report.model_dump(mode="json")
        if strategy is not None:
            data["strategy"] = {
                "id": strategy.meta.id,
                "name": strategy.meta.name,
                "version": strategy.meta.version,
                "category": strategy.meta.category.value,
                "description": strategy.description,
            }
        return json.dumps(data, indent=2, ensure_ascii=False)

    @staticmethod
    def to_markdown(
        report: EvaluationReport,
        strategy: Strategy | None = None,
    ) -> str:
        """Generate a human-readable Markdown report."""
        o = report.overall
        af = report.anti_overfit

        # Header
        name = strategy.meta.name if strategy else report.strategy_id
        sid = strategy.meta.id if strategy else report.strategy_id
        version = strategy.meta.version if strategy else "—"

        lines: list[str] = [
            f"# Strategy Evaluation Report: {name}",
            "",
            f"**Strategy ID**: {sid} | **Version**: {version}",
            f"**Confidence Score**: {report.confidence_score:.1%}",
        ]

        # Description
        if strategy and strategy.description:
            lines += ["", "## Description", "", strategy.description.strip()]

        if strategy is not None:
            hypothesis = strategy.assess_market_hypothesis(report)
            summary = hypothesis.summary
            lines += [
                "",
                "## Strategy Hypothesis",
                "",
                f"- Thesis: {summary.thesis}",
                f"- Expected Regimes: {', '.join(summary.expected_regimes) if summary.expected_regimes else 'unspecified'}",
                f"- Key Indicators: {', '.join(summary.key_indicators) if summary.key_indicators else 'none'}",
                f"- Signal Style: {summary.signal_style}",
                f"- Execution Assumption: {summary.execution_assumption}",
                f"- Risk Assumption: {summary.risk_assumption}",
                f"- Current Assessment: {Reporter._title_case_status(hypothesis.status)}",
                f"- Rationale: {hypothesis.rationale}",
                f"- Next Step: {hypothesis.next_step}",
            ]

        # Performance summary table
        lines += [
            "",
            "## Performance Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Win Rate | {o.win_rate:.1%} |",
            f"| Avg Return | {o.avg_return:.2%} |",
            f"| Avg Win | {o.avg_win_return:.2%} |",
            f"| Avg Loss | {o.avg_loss_return:.2%} |",
            f"| Median Return | {o.median_return:.2%} |",
            f"| P/L Ratio | {o.profit_loss_ratio:.2f} |",
            f"| Max Drawdown | {o.max_drawdown:.1%} |",
            f"| Sharpe Ratio | {o.sharpe_ratio:.2f} |",
            f"| Total Signals | {o.signal_count} |",
            f"| Avg Holding Days | {o.avg_holding_days:.1f} |",
            f"| Max Consecutive Loss | {o.max_consecutive_loss} |",
            f"| Total Return | {o.total_return:.2%} |",
        ]

        # Regime breakdown (if available)
        if report.by_regime:
            lines += [
                "",
                "## Performance by Market Regime",
                "",
                "| Regime | Win Rate | Avg Return | Signals |",
                "|--------|----------|------------|---------|",
            ]
            for rm in report.by_regime:
                lines.append(
                    f"| {rm.regime.value} | {rm.win_rate:.1%} "
                    f"| {rm.avg_return:.2%} | {rm.signal_count} |"
                )

        if report.by_sector:
            lines += [
                "",
                "## Performance by Sector",
                "",
                "| Sector | Win Rate | Avg Return | Signals |",
                "|--------|----------|------------|---------|",
            ]
            for sector, sector_metrics in report.by_sector.items():
                lines.append(
                    f"| {sector} | {sector_metrics.win_rate:.1%} "
                    f"| {sector_metrics.avg_return:.2%} | {sector_metrics.signal_count} |"
                )

        if report.regime_holdout and report.regime_holdout.holdouts:
            rh = report.regime_holdout
            worst_regime = rh.worst_regime.value if rh.worst_regime is not None else "—"
            lines += [
                "",
                "## Regime Holdout",
                "",
                f"- Preferred Regimes: {', '.join(rh.preferred_regimes) if rh.preferred_regimes else 'none'}",
                f"- Pass Threshold: gap <= {rh.pass_gap:.1%}",
                f"- Pass Rate: {rh.pass_rate:.1%} ({rh.total_cases} holdouts)",
                f"- Worst Holdout Gap: {rh.worst_gap:.1%} ({worst_regime})",
                "",
                "| Holdout Regime | Preferred | In-Sample WR | Holdout WR | Gap | In-Sample Signals | Holdout Signals |",
                "|----------------|-----------|--------------|------------|-----|-------------------|-----------------|",
            ]
            for holdout in rh.holdouts:
                lines.append(
                    f"| {holdout.regime.value} | {'yes' if holdout.preferred else 'no'} "
                    f"| {holdout.in_sample_win_rate:.1%} | {holdout.holdout_win_rate:.1%} "
                    f"| {holdout.gap:.1%} | {holdout.in_sample_signal_count} | {holdout.holdout_signal_count} |"
                )

        if report.top_patterns:
            lines += ["", "## Reusable Research Patterns", ""]
            for pattern in report.top_patterns:
                lines.append(f"- {pattern}")

        if report.event_context and (
            report.event_context.relevant_indicators
            or report.event_context.provider_symbols > 0
            or report.event_context.mixed_symbols > 0
        ):
            ec = report.event_context
            lines += [
                "",
                "## Event/News Context",
                "",
                f"- Relevant Indicators: {', '.join(ec.relevant_indicators) if ec.relevant_indicators else 'none'}",
                f"- Provider Coverage: {ec.provider_coverage:.1%} ({ec.provider_symbols + ec.mixed_symbols}/{ec.total_symbols} symbols)",
                f"- Proxy-Only Coverage: {ec.proxy_only_coverage:.1%} ({ec.proxy_symbols}/{ec.total_symbols} symbols)",
            ]
            if ec.mixed_symbols > 0:
                lines.append(f"- Mixed Provider+Proxy Symbols: {ec.mixed_symbols}")
            if ec.source_breakdown:
                source_text = ", ".join(
                    f"{source} x {count}" for source, count in ec.source_breakdown.items()
                )
                lines.append(f"- Source Breakdown: {source_text}")
            if ec.is_proxy_dominant:
                lines.append(
                    "- **Data Quality Caveat**: event/news indicators are proxy-dominant; "
                    "diagnose provider coverage before mutating or optimizing these filters."
                )
            elif ec.has_proxy_caveat:
                lines.append(
                    "- Caveat: some event/news values are price/volume proxy-derived, not direct provider evidence."
                )

        # Top failure cases
        if report.failure_cases:
            lines += [
                "",
                "## Top Failure Cases",
                "",
                "| Symbol | Date | Return | Exit Reason |",
                "|--------|------|--------|-------------|",
            ]
            for fc in report.failure_cases[:10]:
                reason = fc.exit_reason.value if fc.exit_reason else "—"
                lines.append(f"| {fc.symbol} | {fc.signal_date} | {fc.return_pct:.2%} | {reason} |")

        # Benchmark comparison
        if report.benchmark:
            bm = report.benchmark
            alpha_emoji = "🟢" if bm.beats_benchmark else "🔴"
            lines += [
                "",
                "## Benchmark Comparison (Buy & Hold)",
                "",
                "| Metric | Strategy | Benchmark | Diff |",
                "|--------|----------|-----------|------|",
                f"| Total Return | {bm.strategy_return:.2%} | {bm.benchmark_return:.2%} | {bm.excess_return:+.2%} {alpha_emoji} |",
                f"| Max Drawdown | {o.max_drawdown:.1%} | {bm.benchmark_max_drawdown:.1%} | — |",
                f"| Sharpe Ratio | {o.sharpe_ratio:.2f} | {bm.benchmark_sharpe:.2f} | — |",
                "",
                f"> Alpha vs Buy & Hold: **{bm.excess_return:+.2%}** across {bm.symbols_used} symbols",
            ]
            if bm.random_baseline_mean is not None:
                lines += [
                    "",
                    "### Random Baseline",
                    "",
                    f"- Mean Return: {bm.random_baseline_mean:.2%}",
                    f"- 95% CI: [{(bm.random_baseline_ci_lower or 0.0):.2%}, {(bm.random_baseline_ci_upper or 0.0):.2%}]",
                    f"- Beat Fraction: {(bm.random_baseline_beat_fraction or 0.0):.0%}",
                ]

        if report.stress_windows and report.stress_windows.windows:
            sw = report.stress_windows
            lines += [
                "",
                "## Stress-Window Benchmark",
                "",
                f"- Window Size: {sw.window_days} trading days",
                f"- Worst Windows Reviewed: {sw.total_windows}/{sw.top_k}",
                f"- Alpha Pass Threshold: {sw.alpha_pass_threshold:+.2%}",
                f"- Pass Rate: {sw.pass_rate:.1%}",
                f"- Average Alpha: {sw.average_alpha:+.2%}",
                f"- Worst Alpha: {sw.worst_alpha:+.2%}",
                "",
                "| Window | Period | Benchmark Return | Benchmark DD | Signals | Strategy WR | Strategy Return | Alpha |",
                "|--------|--------|------------------|--------------|---------|-------------|-----------------|-------|",
            ]
            for window in sw.windows:
                lines.append(
                    f"| {window.window_num} | {window.start_date} → {window.end_date} "
                    f"| {window.benchmark_return:.2%} | {window.benchmark_drawdown:.2%} "
                    f"| {window.signal_count} | {window.strategy_win_rate:.1%} "
                    f"| {window.strategy_total_return:.2%} | {window.alpha:+.2%} |"
                )

        # Anti-overfitting check
        tv_ok = "✅" if af.train_val_gap <= 0.10 else "⚠️"
        lines += [
            "",
            "## Anti-Overfitting Check",
            "",
            f"- Train-Val Gap: {af.train_val_gap:.1%} {tv_ok}",
            f"- Yearly Consistency: {af.yearly_consistency:.1%}",
            f"- Param Sensitivity: {af.param_sensitivity:.1%}",
        ]
        if af.is_overfit:
            lines.append("- **⚠️ Potential overfitting detected**")

        if report.walk_forward:
            protocol = report.walk_forward_protocol
            pass_gap = protocol.pass_gap if protocol is not None else 0.10
            lines += [
                "",
                "## Walk-Forward Validation",
                "",
            ]
            if protocol is not None:
                lines += [
                    f"- Protocol: requested {protocol.requested_folds} folds, effective {protocol.effective_folds} folds",
                    f"- Train/Test Split: {protocol.train_pct:.0%}/{protocol.test_pct:.0%}",
                    f"- Fold Pass Threshold: gap <= {protocol.pass_gap:.1%}",
                    f"- Min Signals Per Split: {protocol.min_signals_per_split}",
                    "",
                ]
            lines += [
                "| Fold | Train WR | Test WR | Gap | Train Signals | Test Signals |",
                "|------|----------|---------|-----|---------------|--------------|",
            ]
            for fold in report.walk_forward:
                lines.append(
                    f"| {fold.fold_num} | {fold.train_win_rate:.1%} | {fold.test_win_rate:.1%} "
                    f"| {fold.gap:.1%} | {fold.train_signal_count} | {fold.test_signal_count} |"
                )
            lines += [
                "",
                f"- Mean Walk-Forward Gap: {af.walk_forward_gap:.1%}",
                f"- Pass Rate (gap <= {pass_gap:.1%}): {af.walk_forward_pass_rate:.1%}",
            ]

        maturity = build_research_maturity_report(report, strategy)
        lines += render_research_maturity_markdown(maturity)

        # Footer
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines += [
            "",
            "---",
            f"*Generated by AlphaEvo | {ts}*",
            "*⚠️ This is research output, not investment advice.*",
            "",
        ]

        return "\n".join(lines)

    @staticmethod
    def to_file(
        report: EvaluationReport,
        path: Path,
        format: str = "markdown",
        strategy: Strategy | None = None,
    ) -> None:
        """Write report to file.

        Args:
            report: The evaluation report to write.
            path: Destination file path.
            format: ``"markdown"`` or ``"json"``.
            strategy: Optional strategy for enriched output.
        """
        if format == "json":
            content = Reporter.to_json(report, strategy)
        else:
            content = Reporter.to_markdown(report, strategy)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def summary_table(reports: list[tuple[str, EvaluationReport]]) -> str:
        """Generate an ASCII comparison table for multiple strategies.

        Each entry is a ``(strategy_name, EvaluationReport)`` tuple.
        Suitable for leaderboard / terminal display.
        """
        if not reports:
            return "(no strategies to compare)"

        # Column definitions: (header, width, formatter)
        cols: list[tuple[str, int, str]] = [
            ("Strategy", 24, "s"),
            ("Score", 7, "s"),
            ("WinRate", 8, "s"),
            ("AvgRet", 8, "s"),
            ("P/L", 6, "s"),
            ("MaxDD", 7, "s"),
            ("Sharpe", 7, "s"),
            ("Signals", 8, "s"),
            ("TotRet", 8, "s"),
        ]

        header_parts = [h.ljust(w) for h, w, _ in cols]
        header = "| " + " | ".join(header_parts) + " |"
        sep = "| " + " | ".join("-" * w for _, w, _ in cols) + " |"

        rows = [header, sep]

        # Sort by confidence score descending
        sorted_reports = sorted(reports, key=lambda r: r[1].confidence_score, reverse=True)

        for name, rpt in sorted_reports:
            o = rpt.overall
            values = [
                _truncate(name, 24),
                f"{rpt.confidence_score:.1%}",
                f"{o.win_rate:.1%}",
                f"{o.avg_return:.2%}",
                f"{o.profit_loss_ratio:.2f}",
                f"{o.max_drawdown:.1%}",
                f"{o.sharpe_ratio:.2f}",
                str(o.signal_count),
                f"{o.total_return:.2%}",
            ]
            row_parts = [v.ljust(w) for v, (_, w, _) in zip(values, cols, strict=False)]
            rows.append("| " + " | ".join(row_parts) + " |")

        return "\n".join(rows)

    @staticmethod
    def evolution_report(result: EvolutionResult) -> str:
        """Generate a shareable Markdown evolution story.

        Shows the strategy's self-improvement journey round-by-round,
        including changes applied, score progression, and final outcome.
        """
        if not result.rounds:
            return "# Evolution Report\n\nNo evolution rounds recorded."

        first = result.rounds[0]
        best_round = max(result.rounds, key=lambda r: r.evaluation.confidence_score)

        lines: list[str] = [
            f"# 🧬 Evolution Report: {result.original_strategy_id}",
            "",
            f"**Rounds**: {result.total_rounds} | "
            f"**Champion**: {result.champion_id or '—'} | "
            f"**Improvement**: {result.improvement:+.1%}",
            "",
        ]

        original_hypothesis = first.strategy.build_market_hypothesis()
        latest_hypothesis = best_round.strategy.assess_market_hypothesis(best_round.evaluation)
        lines += [
            "## Hypothesis Lens",
            "",
            f"- Original Thesis: {original_hypothesis.thesis}",
            f"- Expected Regimes: {', '.join(original_hypothesis.expected_regimes) if original_hypothesis.expected_regimes else 'unspecified'}",
            f"- Latest Assessment: {Reporter._title_case_status(latest_hypothesis.status)}",
            f"- Assessment Rationale: {latest_hypothesis.rationale}",
            f"- Recommended Next Step: {latest_hypothesis.next_step}",
            "",
        ]

        # Score progression table
        lines += [
            "## Score Progression",
            "",
            "| Round | Strategy | Score | Win Rate | Avg Return | Improved |",
            "|-------|----------|-------|----------|------------|----------|",
        ]
        for r in result.rounds:
            ev = r.evaluation
            o = ev.overall
            improved_icon = "✅" if r.improved else ("—" if r.round_num == 0 else "❌")
            lines.append(
                f"| {r.round_num} | {ev.strategy_id} "
                f"| {ev.confidence_score:.1%} | {o.win_rate:.1%} "
                f"| {o.avg_return:.2%} | {improved_icon} |"
            )

        # Velocity chart (ASCII sparkline)
        velocities = result.velocity
        if velocities:
            spark = " → ".join(f"{v:+.1%}" for v in velocities)
            lines += [
                "",
                "## Improvement Velocity",
                "",
                f"Per-round delta: {spark}",
            ]

        sampling_rounds = [r for r in result.rounds if getattr(r, "batch", None) is not None]
        if sampling_rounds:
            sparse_rounds = sum(
                1 for r in sampling_rounds if r.batch and r.batch.insufficient_signals
            )
            total_attempts = sum(
                r.batch.sampling_attempt if r.batch is not None else 0 for r in sampling_rounds
            )
            lines += [
                "",
                "## Sampling Adequacy",
                "",
                f"- Rounds with sampling metadata: {len(sampling_rounds)}",
                f"- Total sampling attempts: {total_attempts}",
                f"- Sparse-signal rounds blocked: {sparse_rounds}",
            ]

        meta_count = sum(len(r.meta_insights) for r in result.rounds)
        lesson_count = sum(len(r.experience_lessons) for r in result.rounds)
        pattern_count = sum(len(r.pattern_context) for r in result.rounds)
        cross_memory_count = sum(
            len(getattr(r, "cross_strategy_memory", [])) for r in result.rounds
        )
        if meta_count or lesson_count or pattern_count:
            lines += [
                "",
                "## Self-Evolution Signals",
                "",
                f"- Meta-learning insights consulted: {meta_count}",
                f"- Experience lessons reused: {lesson_count}",
                f"- Reusable patterns consulted: {pattern_count}",
            ]
            if cross_memory_count:
                lines.append(f"- Cross-strategy memory callouts: {cross_memory_count}")

        collective_memory: list[str] = []
        seen_memory: set[str] = set()
        for round_result in result.rounds:
            for item in getattr(round_result, "cross_strategy_memory", []):
                if item in seen_memory:
                    continue
                seen_memory.add(item)
                collective_memory.append(item)
        if collective_memory:
            lines += [
                "",
                "## Collective Memory",
                "",
                "AlphaEvo does not only react to the current strategy. It also surfaces what prior experiments from other strategy families already taught it:",
                "",
            ]
            for item in collective_memory[:6]:
                lines.append(f"- {item}")

        # Round-by-round details
        lines += ["", "## Round Details", ""]
        for r in result.rounds:
            ev = r.evaluation
            round_hypothesis = r.strategy.assess_market_hypothesis(ev)
            lines.append(f"### Round {r.round_num}: {ev.strategy_id}")
            lines.append(
                f"Score: **{ev.confidence_score:.1%}** | Signals: {ev.overall.signal_count}"
            )
            lines.append(
                f"Hypothesis: **{Reporter._title_case_status(round_hypothesis.status)}** — "
                f"{round_hypothesis.rationale}"
            )

            if r.batch is not None:
                lines.append("")
                lines.append("**Sampling Context:**")
                lines.append(
                    f"- Attempts: {r.batch.sampling_attempt} | "
                    f"Target Signals: {r.batch.signal_count_target or '—'} | "
                    f"Reached: {r.batch.signal_count_reached}"
                )
                lines.append(
                    f"- Final Method: {r.batch.sampling_method.value} | "
                    f"Symbols: {len(r.batch.symbols)} | "
                    f"Window: {r.batch.date_range[0]} → {r.batch.date_range[1]}"
                )
                if r.batch.insufficient_signals:
                    lines.append("- Status: sparse signals remained below the evolution gate")
                history = list(r.batch.sampling_history)
                if history:
                    lines.append("- Sampling Attempts:")
                    for attempt in history:
                        status = "accepted" if attempt.accepted else "retry"
                        lines.append(
                            "  - "
                            f"Attempt {attempt.attempt_num}: {attempt.sampling_method.value}, "
                            f"{attempt.selected_symbols} symbols, "
                            f"{attempt.signal_count} signals, "
                            f"{attempt.date_range[0]} → {attempt.date_range[1]} "
                            f"({status})"
                        )

            if (
                r.recommended_method is not None
                or r.recommended_intensity is not None
                or r.recommended_max_changes is not None
            ):
                lines.append("")
                lines.append("**Meta Recommendation:**")
                recommendation = []
                if r.recommended_method is not None:
                    recommendation.append(f"method={r.recommended_method}")
                if r.recommended_intensity is not None:
                    recommendation.append(f"intensity={r.recommended_intensity:.2f}")
                if r.recommended_max_changes is not None:
                    recommendation.append(f"max_changes={r.recommended_max_changes}")
                lines.append(f"- {', '.join(recommendation)}")

            if r.meta_insights:
                lines.append("")
                lines.append("**Meta-Learning Context:**")
                for insight in r.meta_insights:
                    lines.append(f"- {insight}")

            if r.experience_lessons:
                lines.append("")
                lines.append("**Experience Lessons Reused:**")
                for lesson in r.experience_lessons:
                    lines.append(f"- {lesson}")

            if r.pattern_context:
                lines.append("")
                lines.append("**Reusable Patterns Consulted:**")
                for pattern in r.pattern_context:
                    lines.append(f"- {pattern}")

            if getattr(r, "cross_strategy_memory", None):
                lines.append("")
                lines.append("**Cross-Strategy Memory Applied:**")
                for memory in r.cross_strategy_memory:
                    lines.append(f"- {memory}")

            if r.reflection and r.reflection.reflection_summary:
                lines.append("")
                lines.append(f"**Reflection Summary:** {r.reflection.reflection_summary}")

            if r.reflection and r.reflection.proposed_changes:
                lines.append("")
                lines.append("**Changes Applied:**")
                for ch in r.reflection.proposed_changes:
                    lines.append(
                        f"- `{ch.change_type.value}` {ch.target}: {ch.from_value} → {ch.to_value}"
                    )
                    if ch.reason:
                        lines.append(f"  - *{ch.reason}*")

            if r.reflection and r.reflection.failure_patterns:
                lines.append("")
                lines.append("**Failure Patterns Identified:**")
                for fp in r.reflection.failure_patterns[:5]:
                    lines.append(f"- {fp}")

            lines.append("")

        # Research Log: Agent thinking process
        if result.research_log is not None:
            events = result.research_log.get_events()
            if events:
                lines += ["## Research Log (Agent Thinking Process)", ""]
                _event_icons = {
                    "hypothesis": "🔬",
                    "observation": "📊",
                    "diagnosis": "🔍",
                    "insight": "💡",
                    "decision": "⚖️",
                    "experiment": "🧪",
                    "result": "🏁",
                    "reflection": "🪞",
                }
                current_round: int | None = None
                for event in events:
                    rnd = event.round_num
                    if rnd != 0 and rnd != current_round:
                        current_round = rnd
                        lines.append(f"#### Round {rnd}")
                        lines.append("")
                    icon = _event_icons.get(event.event_type, "📝")
                    lines.append(f"- {icon} **{event.event_type}**: {event.content}")
                lines.append("")

        # Summary
        if result.early_stopped:
            lines.append(f"> ⏹️ Early stopped: {result.stop_reason}")
            lines.append("")

        lines += [
            "## Summary",
            "",
            f"- Starting score: **{first.evaluation.confidence_score:.1%}**",
            f"- Best score: **{best_round.evaluation.confidence_score:.1%}** (Round {best_round.round_num})",
            f"- Total improvement: **{result.improvement:+.1%}**",
            "",
            "---",
            f"*Generated by AlphaEvo | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            "*⚠️ This is research output, not investment advice.*",
            "",
        ]

        return "\n".join(lines)

    @staticmethod
    def research_report(result: EvolutionResult) -> str:
        """Generate one combined shareable report for an evolution run."""
        story = Reporter.evolution_report(result).rstrip()
        evidence = Reporter.llm_evidence_report(result).rstrip()
        return f"{story}\n\n---\n\n## LLM Evidence Appendix\n\n{evidence}\n"

    @staticmethod
    def llm_evidence_report(result: EvolutionResult) -> str:
        """Generate a focused report showing what the LLM contributed."""
        if not result.rounds:
            return "# LLM Evidence Report\n\nNo evolution rounds recorded."

        first_round = result.rounds[0]
        trajectory = getattr(result, "trajectory", None)
        metadata = getattr(trajectory, "metadata", {}) or {}
        method = metadata.get("method", "unknown")
        strategy_family = getattr(trajectory, "strategy_family", result.original_strategy_id)
        steps = list(getattr(trajectory, "steps", []))

        total_changes = sum(
            len(getattr(round_result.reflection, "proposed_changes", []) or [])
            for round_result in result.rounds
        )
        improved_rounds = sum(1 for round_result in result.rounds if round_result.improved)
        telemetry_snapshots = [
            Reporter._coerce_llm_telemetry(
                getattr(getattr(round_result, "reflection", None), "llm_telemetry", None)
            )
            for round_result in result.rounds
        ]
        telemetry_snapshots = [snapshot for snapshot in telemetry_snapshots if snapshot]
        total_llm_calls = sum(
            len(cast("list[dict[str, object]]", snapshot.get("calls", [])))
            for snapshot in telemetry_snapshots
        )
        timeout_failures = sum(
            Reporter._timeout_failures_from_telemetry(snapshot) for snapshot in telemetry_snapshots
        )
        fallback_rounds = sum(
            1
            for snapshot in telemetry_snapshots
            if str(snapshot.get("path", "")) in {"single_step_fallback", "heuristic_fallback"}
        )
        avg_reflection_ms = (
            sum(
                Reporter._coerce_int(snapshot.get("total_duration_ms", 0))
                for snapshot in telemetry_snapshots
            )
            // len(telemetry_snapshots)
            if telemetry_snapshots
            else 0
        )

        lines = [
            f"# LLM Evidence Report: {result.original_strategy_id}",
            "",
            f"**Method**: {method} | **Strategy Family**: {strategy_family}",
            f"**Champion**: {result.champion_id or '—'} | **Rounds Attempted**: {result.total_rounds}",
            f"**Starting Score**: {first_round.evaluation.confidence_score:.1%} | "
            f"**Champion Score**: {result.champion_score:.1%} | "
            f"**Improvement**: {result.improvement:+.1%}",
            "",
            "## Summary",
            "",
            f"- Rounds recorded: {len(result.rounds)}",
            f"- LLM-driven change bundles applied: {total_changes}",
            f"- Improved rounds that survived evaluation: {improved_rounds}",
        ]
        if telemetry_snapshots:
            lines += [
                f"- Reflection calls recorded: {total_llm_calls}",
                f"- Rounds with fallback/degradation: {fallback_rounds}",
                f"- Timeout-like failures observed: {timeout_failures}",
                f"- Avg reflection latency: {avg_reflection_ms} ms",
            ]

        baseline_param_search_score = getattr(result, "baseline_param_search_score", None)
        if baseline_param_search_score is not None:
            lines.append(
                f"- Param-search baseline score (fallback comparator): "
                f"{baseline_param_search_score:.1%}"
            )
        if getattr(result, "early_stopped", False):
            lines.append(f"- Early stop: {getattr(result, 'stop_reason', '')}")

        lines += ["", "## Round-by-Round LLM Evidence", ""]

        if steps:
            for step in steps:
                lines.append(f"### Round {step.round_num}: {step.strategy_id}")
                lines.append(
                    f"- Before: score {step.score_before:.1%}, "
                    f"win rate {step.win_rate_before:.1%}, signals {step.signal_count_before}"
                )
                if step.failure_patterns:
                    lines.append(f"- Failure patterns: {', '.join(step.failure_patterns)}")
                if step.diagnosis:
                    lines.append(f"- Diagnosis: {step.diagnosis}")
                if step.hypothesis:
                    lines.append(f"- Hypothesis: {step.hypothesis}")
                if step.expected_outcome:
                    lines.append(f"- Expected outcome: {step.expected_outcome}")
                if step.changes:
                    lines.append("- Changes:")
                    for change in step.changes:
                        change_type = change.get("change_type", "?")
                        target = change.get("target", "?")
                        from_value = change.get("from_value", "?")
                        to_value = change.get("to_value", "?")
                        lines.append(f"  - `{change_type}` {target}: {from_value} → {to_value}")
                        reason = change.get("reason")
                        if reason:
                            lines.append(f"    - {reason}")
                else:
                    lines.append("- Changes: none applied")
                if step.critic_verdict:
                    lines.append(f"- Critic verdict: {step.critic_verdict}")
                telemetry = Reporter._coerce_llm_telemetry(getattr(step, "llm_telemetry", {}))
                if telemetry:
                    lines.append(
                        f"- LLM runtime: path={telemetry.get('path', '')}, "
                        f"calls={len(cast('list[dict[str, object]]', telemetry.get('calls', [])))}, "
                        f"total={Reporter._coerce_int(telemetry.get('total_duration_ms', 0))} ms"
                    )
                    timeout_count = Reporter._timeout_failures_from_telemetry(telemetry)
                    if timeout_count:
                        lines.append(f"- Timeout-like failures: {timeout_count}")
                    fallback_trigger = str(telemetry.get("fallback_trigger", "")).strip()
                    if fallback_trigger:
                        lines.append(f"- Fallback trigger: {fallback_trigger}")
                lines.append(
                    f"- Outcome: {'improved' if step.improved else 'not improved'} "
                    f"(score {step.score_before:.1%} → {step.score_after:.1%}, "
                    f"delta {step.score_delta:+.1%})"
                )
                lines.append("")
        else:
            for round_result in result.rounds:
                lines.append(f"### Round {round_result.round_num}: {round_result.strategy.meta.id}")
                lines.append(
                    f"- Score: {round_result.evaluation.confidence_score:.1%} | "
                    f"Signals: {round_result.evaluation.overall.signal_count}"
                )
                if round_result.reflection and round_result.reflection.failure_patterns:
                    lines.append(
                        f"- Failure patterns: {', '.join(round_result.reflection.failure_patterns)}"
                    )
                if round_result.reflection and round_result.reflection.reflection_summary:
                    lines.append(
                        f"- Reflection summary: {round_result.reflection.reflection_summary}"
                    )
                if round_result.reflection and round_result.reflection.proposed_changes:
                    lines.append("- Changes:")
                    for change in round_result.reflection.proposed_changes:
                        lines.append(
                            f"  - `{change.change_type.value}` {change.target}: "
                            f"{change.from_value} → {change.to_value}"
                        )
                        if change.reason:
                            lines.append(f"    - {change.reason}")
                else:
                    lines.append("- Changes: none applied")
                telemetry = Reporter._coerce_llm_telemetry(
                    getattr(round_result.reflection, "llm_telemetry", None)
                )
                if telemetry:
                    lines.append(
                        f"- LLM runtime: path={telemetry.get('path', '')}, "
                        f"calls={len(cast('list[dict[str, object]]', telemetry.get('calls', [])))}, "
                        f"total={Reporter._coerce_int(telemetry.get('total_duration_ms', 0))} ms"
                    )
                    timeout_count = Reporter._timeout_failures_from_telemetry(telemetry)
                    if timeout_count:
                        lines.append(f"- Timeout-like failures: {timeout_count}")
                    fallback_trigger = str(telemetry.get("fallback_trigger", "")).strip()
                    if fallback_trigger:
                        lines.append(f"- Fallback trigger: {fallback_trigger}")
                lines.append(
                    f"- Outcome: {'improved' if round_result.improved else 'not improved'}"
                )
                lines.append("")

        research_log = getattr(result, "research_log", None)
        if research_log is not None:
            diagnosis_events = research_log.get_events(event_type="diagnosis")
            decision_events = research_log.get_events(event_type="decision")
            result_events = research_log.get_events(event_type="result")
            if diagnosis_events or decision_events or result_events:
                lines += ["## Research Log Highlights", ""]
                for title, events in (
                    ("Diagnoses", diagnosis_events[:5]),
                    ("Decisions", decision_events[:5]),
                    ("Results", result_events[:5]),
                ):
                    if not events:
                        continue
                    lines.append(f"### {title}")
                    for event in events:
                        prefix = f"Round {event.round_num}: " if event.round_num else ""
                        lines.append(f"- {prefix}{event.content}")
                    lines.append("")

        lines += [
            "---",
            f"*Generated by AlphaEvo | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
            "*⚠️ This report highlights model-guided research steps, not guaranteed trading edge.*",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _build_equity_series(
        trades: list[TradeSignal],
    ) -> tuple[list[float], list[float]] | None:
        """Build cumulative-return series from completed trades.

        Returns ``(x_indices, cum_return_pct)`` or ``None`` if no data.
        """
        completed = [t for t in trades if t.exit_date is not None]
        if not completed:
            return None
        sorted_trades = sorted(completed, key=lambda t: cast("date", t.exit_date))
        cum = 1.0
        xs: list[float] = []
        ys: list[float] = []
        for i, t in enumerate(sorted_trades):
            cum *= 1 + t.return_pct
            xs.append(float(i + 1))
            ys.append((cum - 1) * 100)
        return xs, ys

    @staticmethod
    def plot_equity_curve(
        trades: list[TradeSignal],
        title: str = "Equity Curve",
    ) -> str:
        """Render a terminal equity curve using plotext (if available).

        Returns the rendered text, or a fallback ASCII representation.
        """
        if not trades:
            return "(no trades to plot)"

        series = Reporter._build_equity_series(trades)
        if series is None:
            return "(no completed trades)"
        xs, values = series

        try:
            import plotext as plt

            plt.clear_figure()
            plt.plot(xs, values, marker="braille")
            plt.title(title)
            plt.xlabel("Trade #")
            plt.ylabel("Cumulative Return (%)")
            plt.theme("clear")
            plt.plot_size(80, 20)
            return cast("str", plt.build())
        except ImportError:
            return Reporter._ascii_sparkline(values, title)

    @staticmethod
    def plot_return_distribution(
        trades: list[TradeSignal],
        title: str = "Return Distribution",
    ) -> str:
        """Render a win/loss return histogram using plotext."""
        completed = [t for t in trades if t.exit_date is not None]
        if not completed:
            return "(no completed trades)"

        returns = [t.return_pct * 100 for t in completed]
        try:
            import plotext as plt

            plt.clear_figure()
            plt.hist(returns, bins=min(20, max(5, len(returns) // 3)))
            plt.title(title)
            plt.xlabel("Return (%)")
            plt.ylabel("Count")
            plt.theme("clear")
            plt.plot_size(80, 15)
            return cast("str", plt.build())
        except ImportError:
            wins = sum(1 for r in returns if r > 0)
            losses = len(returns) - wins
            return f"{title}\nWins: {wins} | Losses: {losses}"

    @staticmethod
    def plot_drawdown_curve(
        trades: list[TradeSignal],
        title: str = "Drawdown Curve",
    ) -> str:
        """Render a drawdown curve from cumulative returns."""
        series = Reporter._build_equity_series(trades)
        if series is None:
            return "(no completed trades)"
        xs, cum_returns = series

        # Convert cumulative return % to equity values and compute drawdown
        peak = 0.0
        drawdowns: list[float] = []
        for ret in cum_returns:
            equity = 100 + ret  # starting at 100
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak * 100 if peak > 0 else 0.0
            drawdowns.append(dd)

        try:
            import plotext as plt

            plt.clear_figure()
            plt.plot(xs, drawdowns, marker="braille", color="red")
            plt.title(title)
            plt.xlabel("Trade #")
            plt.ylabel("Drawdown (%)")
            plt.theme("clear")
            plt.plot_size(80, 12)
            return cast("str", plt.build())
        except ImportError:
            if drawdowns:
                return f"{title}\nMax Drawdown: {min(drawdowns):.1f}%"
            return "(no data)"

    @staticmethod
    def plot_evolution_scores(
        round_scores: list[tuple[str, float]],
        title: str = "Evolution Progress",
    ) -> str:
        """Plot confidence scores across evolution rounds."""
        if not round_scores:
            return "(no evolution data)"

        labels = [label for label, _ in round_scores]
        scores = [score * 100 for _, score in round_scores]

        try:
            import plotext as plt

            plt.clear_figure()
            plt.bar(labels, scores)
            plt.title(title)
            plt.ylabel("Confidence Score (%)")
            plt.theme("clear")
            plt.plot_size(80, 15)
            return cast("str", plt.build())
        except ImportError:
            lines = [title]
            for label, score in zip(labels, scores, strict=True):
                bar_len = int(score / 2)
                lines.append(f"  {label}: {'█' * bar_len} {score:.1f}%")
            return "\n".join(lines)

    @staticmethod
    def _ascii_sparkline(values: list[float], title: str) -> str:
        """Fallback ASCII sparkline when plotext is not available."""
        if not values:
            return "(no data)"
        mn, mx = min(values), max(values)
        rng = mx - mn if mx > mn else 1.0
        bars = "▁▂▃▄▅▆▇█"
        line = ""
        step = max(1, len(values) // 60)
        for i in range(0, len(values), step):
            idx = int((values[i] - mn) / rng * (len(bars) - 1))
            idx = max(0, min(len(bars) - 1, idx))
            line += bars[idx]
        return f"{title}\n{line}\n{mn:.1f}% → {mx:.1f}%"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if it exceeds *max_len*."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
