"""Append-only artifact writer, SQLite indexer, and comparison service."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from alphaevo.models.execution import EvaluationReport, TradeSignal
from alphaevo.models.strategy import Strategy
from alphaevo.provenance.contracts import (
    ArtifactPathsInfo,
    CodeVersionInfo,
    ConfigProvenanceInfo,
    DataMutability,
    DataProvenanceInfo,
    DataReproducibility,
    DeterminismInfo,
    ProtocolName,
    RunComparisonResult,
    RunIndexEntry,
    RunKind,
    RunManifest,
    RunSummaryInfo,
    StrategyProvenanceInfo,
)
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.dsl.serializer import StrategySerializer
from alphaevo.truth.ledger import compute_strategy_hash

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.core.config import AppConfig

logger = logging.getLogger(__name__)

_KIND_DIRS: dict[RunKind, str] = {
    "run": "runs",
    "evolve": "evolutions",
    "optimize": "optimizations",
    "rsi": "rsi",
}

_IGNORED_STRATEGY_DIFF_PATHS = {
    "meta.id",
    "meta.name",
    "meta.version",
    "meta.parent_id",
    "meta.created_at",
    "meta.status",
}

_SECRET_KEYS = {
    "api_key",
    "adanos_api_key",
    "token",
    "password",
    "secret",
    "credential",
    "dsa_path",
    "base_url",
    "adanos_base_url",
    "notify",
    "feishu_webhook",
    "wecom_webhook",
    "dingtalk_webhook",
    "telegram_bot_token",
    "telegram_chat_id",
    "discord_webhook",
    "slack_webhook",
    "webhook_url",
}

_RUN_ARTIFACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_version INTEGER DEFAULT 1,
    requested_protocol TEXT NOT NULL,
    effective_protocol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    artifact_dir TEXT NOT NULL,
    score REAL DEFAULT 0.0,
    win_rate REAL DEFAULT 0.0,
    avg_return REAL DEFAULT 0.0,
    profit_loss_ratio REAL DEFAULT 0.0,
    max_drawdown REAL DEFAULT 0.0,
    sharpe_ratio REAL DEFAULT 0.0,
    signal_count INTEGER DEFAULT 0,
    code_dirty INTEGER DEFAULT 1,
    data_reproducibility TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed'
);

CREATE INDEX IF NOT EXISTS idx_run_artifacts_strategy ON run_artifacts(strategy_id);
CREATE INDEX IF NOT EXISTS idx_run_artifacts_protocol ON run_artifacts(effective_protocol);
CREATE INDEX IF NOT EXISTS idx_run_artifacts_score ON run_artifacts(score DESC);
"""


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _flatten_dict(obj: object, *, prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_dict(value, prefix=next_prefix))
        return flattened
    if isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            flattened.update(_flatten_dict(value, prefix=next_prefix))
        return flattened
    flattened[prefix] = str(obj)
    return flattened


def detect_code_version(repo_root: Path | None = None) -> CodeVersionInfo:
    """Detect git commit and clean/dirty state; fall back cleanly outside git."""
    cwd = repo_root or Path.cwd()
    try:
        commit_proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if commit_proc.returncode == 0 and commit_proc.stdout.strip():
            commit = commit_proc.stdout.strip()
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            dirty = status_proc.returncode != 0 or bool(status_proc.stdout.strip())
            diff_hash: str | None = None
            if dirty and status_proc.returncode == 0:
                diff_hash = (
                    f"sha256:{hashlib.sha256(status_proc.stdout.encode('utf-8')).hexdigest()[:16]}"
                )
            return CodeVersionInfo(source="git", commit=commit, dirty=dirty, diff_hash=diff_hash)
    except Exception:
        pass

    from alphaevo import __version__

    return CodeVersionInfo(source="package", commit=__version__, dirty=True, diff_hash=None)


def _strip_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _strip_secrets(v)
            for k, v in value.items()
            if str(k).lower() not in _SECRET_KEYS
        }
    if isinstance(value, list):
        return [_strip_secrets(v) for v in value]
    return value


def build_safe_config_snapshot(
    config: AppConfig | None,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an allowlisted config snapshot free of API keys, URLs, or credentials."""
    snapshot: dict[str, Any] = {}
    if config is not None:
        snapshot = {
            "data": {
                "adapter": config.data.adapter,
                "source_failure_threshold": config.data.source_failure_threshold,
                "source_cooldown_seconds": config.data.source_cooldown_seconds,
                "max_concurrent_fetches": config.data.max_concurrent_fetches,
            },
            "backtest": {
                "slippage": config.backtest.slippage,
                "commission": config.backtest.commission,
                "default_timing": config.backtest.default_timing,
                "fill_policy": config.backtest.fill_policy,
                "min_data_days": config.backtest.min_data_days,
                "walk_forward_folds": config.backtest.walk_forward_folds,
                "walk_forward_train_pct": config.backtest.walk_forward_train_pct,
                "walk_forward_pass_gap": config.backtest.walk_forward_pass_gap,
                "stress_window_days": config.backtest.stress_window_days,
                "stress_window_top_k": config.backtest.stress_window_top_k,
            },
            "evolution": {
                "max_rounds": config.evolution.max_rounds,
                "max_changes_per_round": config.evolution.max_changes_per_round,
                "min_signal_count": config.evolution.min_signal_count,
                "complexity_limit": config.evolution.complexity_limit,
                "num_candidates": config.evolution.num_candidates,
            },
            "llm": {
                "model": config.llm.model,
                "reflect_model": config.llm.reflect_model,
            },
        }
    if extra:
        snapshot["extra"] = _strip_secrets(extra)
    return snapshot


def build_data_fingerprint_payload(
    market_data: dict[str, pd.DataFrame],
    *,
    adapter_name: str,
    reproducibility: DataReproducibility,
    data_mutability: DataMutability,
    bundle_id: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Build detailed per-symbol OHLCV fingerprint metadata and aggregate SHA-256 digest."""
    import pandas as pd

    symbols_info: list[dict[str, Any]] = []
    all_starts: list[str] = []
    all_ends: list[str] = []

    for symbol in sorted(market_data):
        df = market_data[symbol]
        if df.empty:
            symbols_info.append({"symbol": symbol, "rows": 0, "ohlcv_sha256": ""})
            continue
        start_s = str(pd.Timestamp(df["date"].iloc[0]).date()) if "date" in df.columns else ""
        end_s = str(pd.Timestamp(df["date"].iloc[-1]).date()) if "date" in df.columns else ""
        if start_s:
            all_starts.append(start_s)
        if end_s:
            all_ends.append(end_s)

        cols = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
        hasher = hashlib.sha256()
        if cols:
            for row in df[cols].itertuples(index=False, name=None):
                hasher.update("|".join(str(x) for x in row).encode("utf-8"))
                hasher.update(b"\n")
        symbols_info.append(
            {
                "symbol": symbol,
                "rows": int(len(df)),
                "start_date": start_s,
                "end_date": end_s,
                "ohlcv_sha256": hasher.hexdigest()[:16],
            }
        )

    payload: dict[str, Any] = {
        "adapter": adapter_name,
        "reproducibility": reproducibility,
        "data_mutability": data_mutability,
        "data_bundle_id": bundle_id,
        "date_range": [min(all_starts), max(all_ends)] if all_starts and all_ends else None,
        "symbols": symbols_info,
    }
    digest = _canonical_sha256(payload)
    return payload, digest


def evaluate_protocol_gate(
    requested_protocol: ProtocolName,
    *,
    code_dirty: bool,
    data_reproducibility: DataReproducibility,
    evaluation: EvaluationReport,
    min_signals: int = 30,
) -> tuple[ProtocolName, str | None]:
    """Downgrade ``official_v1`` to ``full_research_v1`` when prerequisites fail."""
    if requested_protocol != "official_v1":
        return requested_protocol, None

    reasons: list[str] = []
    if code_dirty:
        reasons.append("code_dirty=True")
    if data_reproducibility != "replayable":
        reasons.append(f"data_reproducibility={data_reproducibility}")
    if evaluation.overall.signal_count < min_signals:
        reasons.append(f"signals={evaluation.overall.signal_count}<{min_signals}")
    if evaluation.anti_overfit.is_overfit:
        reasons.append("anti_overfit_flagged")

    if reasons:
        return "full_research_v1", "; ".join(reasons)
    return "official_v1", None


def _render_provenance_footer(manifest: RunManifest) -> str:
    local_note = ""
    if manifest.code_version.dirty or manifest.data.reproducibility != "replayable":
        local_note = (
            "\n> Local research run only (`code_dirty="
            f"{manifest.code_version.dirty}`, `reproducibility={manifest.data.reproducibility}`)"
            " — not eligible for official benchmark ranking.\n"
        )
    date_txt = (
        f"{manifest.data.date_range[0]} to {manifest.data.date_range[1]}"
        if manifest.data.date_range
        else "n/a"
    )
    lines = [
        "",
        "## Run Provenance",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Run ID | `{manifest.run_id}` |",
        f"| Kind | `{manifest.kind}` |",
        f"| Protocol | `{manifest.effective_protocol}` (requested: `{manifest.requested_protocol}`) |",
        f"| Strategy Snapshot Hash | `{manifest.strategy.snapshot_hash}` |",
        f"| Code Commit | `{manifest.code_version.commit}` (dirty={manifest.code_version.dirty}) |",
        f"| Data Reproducibility | `{manifest.data.reproducibility}` (adapter=`{manifest.data.adapter}`) |",
        f"| Date Range | {date_txt} |",
        f"| Config Hash | `{manifest.config.snapshot_hash}` |",
        f"| Artifact Dir | `{manifest.artifacts.artifact_dir}` |",
    ]
    if manifest.protocol_downgrade_reason:
        lines.append(f"| Protocol Downgrade | {manifest.protocol_downgrade_reason} |")
    return "\n".join(lines) + "\n" + local_note


def _normalize_db_path(db_path: str | Path) -> tuple[str, bool]:
    if str(db_path) == ":memory:":
        return ":memory:", True
    resolved = Path(db_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    probe = resolved.parent / f".{resolved.stem}.write_probe"
    probe.touch(exist_ok=True)
    probe.unlink(missing_ok=True)
    return str(resolved), False


class RunArtifactStore:
    """Append-only run artifact repository backed by JSON/YAML files and SQLite index."""

    def __init__(
        self,
        *,
        artifact_root: str | Path = "reports",
        db_path: str | Path = "~/.alphaevo/alphaevo.db",
    ) -> None:
        self._artifact_root = Path(artifact_root).expanduser()
        normalized_path, is_mem = _normalize_db_path(db_path)
        self._is_memory = is_mem
        if is_mem:
            self._db_path = ":memory:"
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
        else:
            self._db_path = normalized_path
            self._shared_conn = None
        self.init_db()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_RUN_ARTIFACTS_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def save_run(
        self,
        *,
        kind: RunKind,
        strategy: Strategy,
        evaluation: EvaluationReport,
        market_data: dict[str, pd.DataFrame],
        trades: Sequence[TradeSignal] | None = None,
        config: AppConfig | None = None,
        adapter_name: str = "unknown",
        requested_protocol: ProtocolName = "full_research_v1",
        data_reproducibility: DataReproducibility = "refetch_dependent",
        data_mutability: DataMutability = "mutable_provider",
        data_bundle_id: str | None = None,
        random_seed: int = 42,
        report_markdown: str | None = None,
        initial_strategy: Strategy | None = None,
        min_signals_official: int = 30,
        code_version_override: CodeVersionInfo | None = None,
        extra_config: dict[str, Any] | None = None,
    ) -> RunManifest:
        """Persist an immutable artifact directory and index its summary in SQLite."""
        created_at = datetime.now(timezone.utc)
        code_ver = code_version_override or detect_code_version()
        strat_hash = f"sha256:{compute_strategy_hash(strategy)}"
        safe_cfg = build_safe_config_snapshot(config, extra=extra_config)
        cfg_hash = _canonical_sha256(safe_cfg)
        fp_payload, fp_hash = build_data_fingerprint_payload(
            market_data,
            adapter_name=adapter_name,
            reproducibility=data_reproducibility,
            data_mutability=data_mutability,
            bundle_id=data_bundle_id,
        )

        effective_protocol, downgrade_reason = evaluate_protocol_gate(
            requested_protocol,
            code_dirty=code_ver.dirty,
            data_reproducibility=data_reproducibility,
            evaluation=evaluation,
            min_signals=min_signals_official,
        )

        raw_dr = fp_payload.get("date_range")
        date_range: tuple[str, str] | None = (
            (str(raw_dr[0]), str(raw_dr[1]))
            if isinstance(raw_dr, list) and len(raw_dr) == 2
            else None
        )

        id_seed = {
            "strategy_hash": strat_hash,
            "code_commit": code_ver.commit,
            "adapter": adapter_name,
            "date_range": date_range,
            "protocol": effective_protocol,
            "config_hash": cfg_hash,
            "data_hash": fp_hash,
        }
        hash8 = hashlib.sha256(
            json.dumps(id_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:8]
        ts = created_at.strftime("%Y%m%dT%H%M%SZ")
        base_dir = self._artifact_root / _KIND_DIRS[kind]
        run_id = f"{ts}_{strategy.meta.id}_{hash8}"
        run_dir = base_dir / run_id
        if run_dir.exists():
            run_id = f"{run_id}_{uuid.uuid4().hex[:4]}"
            run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        policy: list[str] = []
        if code_ver.dirty:
            policy.append("dirty_code")
        if data_mutability == "mutable_provider":
            policy.append("provider_mutable")
        elif data_reproducibility == "replayable" and not code_ver.dirty:
            policy.append("deterministic")
        if kind in {"evolve", "rsi"}:
            policy.append("search_iterative")

        artifacts_info = ArtifactPathsInfo(
            artifact_dir=str(run_dir),
            initial_strategy_snapshot="initial_strategy_snapshot.yaml"
            if initial_strategy is not None
            else None,
        )
        ov = evaluation.overall
        manifest = RunManifest(
            run_id=run_id,
            kind=kind,
            created_at=created_at,
            requested_protocol=requested_protocol,
            effective_protocol=effective_protocol,
            protocol_downgrade_reason=downgrade_reason,
            strategy=StrategyProvenanceInfo(
                id=strategy.meta.id,
                version=strategy.meta.version,
                family_id=strategy.meta.family_id,
                snapshot_hash=strat_hash,
            ),
            code_version=code_ver,
            data=DataProvenanceInfo(
                adapter=adapter_name,
                date_range=date_range,
                fingerprint_hash=fp_hash,
                reproducibility=data_reproducibility,
                data_mutability=data_mutability,
                data_bundle_id=data_bundle_id,
                symbol_count=len(market_data),
            ),
            config=ConfigProvenanceInfo(snapshot_hash=cfg_hash),
            determinism=DeterminismInfo(random_seed=random_seed, policy=policy),
            artifacts=artifacts_info,
            summary=RunSummaryInfo(
                score=float(evaluation.confidence_score),
                win_rate=float(ov.win_rate),
                avg_return=float(ov.avg_return),
                profit_loss_ratio=float(ov.profit_loss_ratio),
                max_drawdown=float(ov.max_drawdown),
                sharpe_ratio=float(ov.sharpe_ratio),
                signal_count=int(ov.signal_count),
                metrics_basis=str(ov.metrics_basis),
            ),
        )

        serializer = StrategySerializer()
        serializer.to_file(strategy, run_dir / "strategy_snapshot.yaml")
        if initial_strategy is not None:
            serializer.to_file(initial_strategy, run_dir / "initial_strategy_snapshot.yaml")

        (run_dir / "config_snapshot.json").write_text(
            json.dumps(safe_cfg, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "data_fingerprint.json").write_text(
            json.dumps(fp_payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        (run_dir / "evaluation.json").write_text(
            evaluation.model_dump_json(indent=2),
            encoding="utf-8",
        )

        trade_lines = [
            json.dumps(sig.model_dump(mode="json"), ensure_ascii=False) for sig in (trades or [])
        ]
        (run_dir / "trades.jsonl").write_text(
            ("\n".join(trade_lines) + "\n") if trade_lines else "",
            encoding="utf-8",
        )

        if report_markdown is None:
            from alphaevo.evaluator.reporter import Reporter

            report_markdown = Reporter.to_markdown(evaluation, strategy)
        if "## Run Provenance" not in report_markdown:
            report_markdown = report_markdown.rstrip() + "\n" + _render_provenance_footer(manifest)
        (run_dir / "report.md").write_text(report_markdown, encoding="utf-8")
        (run_dir / "manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_artifacts (
                    run_id, kind, strategy_id, strategy_version,
                    requested_protocol, effective_protocol, created_at,
                    artifact_dir, score, win_rate, avg_return,
                    profit_loss_ratio, max_drawdown, sharpe_ratio,
                    signal_count, code_dirty, data_reproducibility, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.run_id,
                    manifest.kind,
                    manifest.strategy.id,
                    manifest.strategy.version,
                    manifest.requested_protocol,
                    manifest.effective_protocol,
                    manifest.created_at.isoformat(),
                    manifest.artifacts.artifact_dir,
                    manifest.summary.score,
                    manifest.summary.win_rate,
                    manifest.summary.avg_return,
                    manifest.summary.profit_loss_ratio,
                    manifest.summary.max_drawdown,
                    manifest.summary.sharpe_ratio,
                    manifest.summary.signal_count,
                    1 if manifest.code_version.dirty else 0,
                    manifest.data.reproducibility,
                    manifest.status,
                ),
            )
            conn.commit()
        return manifest

    def list_runs(
        self,
        *,
        kind: RunKind | None = None,
        strategy_id: str | None = None,
        protocol: ProtocolName | None = None,
        official_only: bool = False,
        limit: int = 20,
    ) -> list[RunIndexEntry]:
        """Query indexed run artifacts ordered newest-first."""
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if strategy_id is not None:
            clauses.append("strategy_id = ?")
            params.append(strategy_id)
        if official_only:
            clauses.append("effective_protocol = ?")
            params.append("official_v1")
        elif protocol is not None:
            clauses.append("effective_protocol = ?")
            params.append(protocol)

        where = " AND ".join(clauses) if clauses else "1=1"
        params.append(max(1, limit))

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT run_id, kind, strategy_id, strategy_version,
                       requested_protocol, effective_protocol, created_at,
                       artifact_dir, score, win_rate, avg_return,
                       profit_loss_ratio, max_drawdown, sharpe_ratio,
                       signal_count, code_dirty, data_reproducibility, status
                FROM run_artifacts
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()

        entries: list[RunIndexEntry] = []
        for row in rows:
            entries.append(
                RunIndexEntry(
                    run_id=str(row[0]),
                    kind=cast("RunKind", row[1]),
                    strategy_id=str(row[2]),
                    strategy_version=int(row[3]),
                    requested_protocol=cast("ProtocolName", row[4]),
                    effective_protocol=cast("ProtocolName", row[5]),
                    created_at=datetime.fromisoformat(str(row[6])),
                    artifact_dir=str(row[7]),
                    score=float(row[8]),
                    win_rate=float(row[9]),
                    avg_return=float(row[10]),
                    profit_loss_ratio=float(row[11]),
                    max_drawdown=float(row[12]),
                    sharpe_ratio=float(row[13]),
                    signal_count=int(row[14]),
                    code_dirty=bool(row[15]),
                    data_reproducibility=cast("DataReproducibility", row[16]),
                    status=str(row[17]),
                )
            )
        return entries

    def get_manifest(self, run_id: str) -> RunManifest | None:
        """Load authoritative ``manifest.json`` for *run_id*."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT artifact_dir FROM run_artifacts WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        candidate_dirs: list[Path] = []
        if row and row[0]:
            candidate_dirs.append(Path(str(row[0])))
        for subdir in _KIND_DIRS.values():
            candidate_dirs.append(self._artifact_root / subdir / run_id)

        for directory in candidate_dirs:
            manifest_path = directory / "manifest.json"
            if manifest_path.is_file():
                return RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return None

    def load_strategy_snapshot(self, run_id: str) -> Strategy | None:
        """Load the frozen YAML strategy snapshot captured for *run_id*."""
        manifest = self.get_manifest(run_id)
        if manifest is None:
            return None
        yaml_path = Path(manifest.artifacts.artifact_dir) / manifest.artifacts.strategy_snapshot
        if not yaml_path.is_file():
            return None
        return StrategyParser().parse_file(yaml_path)

    def compare_runs(self, run_id_1: str, run_id_2: str) -> RunComparisonResult:
        """Produce high-signal diff between two immutable research runs."""
        m1 = self.get_manifest(run_id_1)
        m2 = self.get_manifest(run_id_2)
        if m1 is None:
            raise FileNotFoundError(f"Run manifest not found: {run_id_1}")
        if m2 is None:
            raise FileNotFoundError(f"Run manifest not found: {run_id_2}")

        reasons: list[str] = []
        if m1.effective_protocol != m2.effective_protocol:
            reasons.append(f"Protocol mismatch: {m1.effective_protocol} vs {m2.effective_protocol}")
        if m1.data.reproducibility != m2.data.reproducibility:
            reasons.append(
                f"Reproducibility mismatch: {m1.data.reproducibility} vs {m2.data.reproducibility}"
            )

        changed_paths: list[str] = []
        s1 = self.load_strategy_snapshot(run_id_1)
        s2 = self.load_strategy_snapshot(run_id_2)
        if s1 is not None and s2 is not None:
            serializer = StrategySerializer()
            d1 = _flatten_dict(serializer.to_dict(s1))
            d2 = _flatten_dict(serializer.to_dict(s2))
            for path in sorted(set(d1) | set(d2)):
                if path in _IGNORED_STRATEGY_DIFF_PATHS:
                    continue
                if d1.get(path) != d2.get(path):
                    changed_paths.append(path)

        return RunComparisonResult(
            run_id_1=run_id_1,
            run_id_2=run_id_2,
            comparable=(m1.effective_protocol == m2.effective_protocol),
            incomparability_reasons=reasons,
            protocol_1=m1.effective_protocol,
            protocol_2=m2.effective_protocol,
            strategy_hash_changed=(m1.strategy.snapshot_hash != m2.strategy.snapshot_hash),
            config_hash_changed=(m1.config.snapshot_hash != m2.config.snapshot_hash),
            data_fingerprint_changed=(m1.data.fingerprint_hash != m2.data.fingerprint_hash),
            score_delta=round(m2.summary.score - m1.summary.score, 6),
            win_rate_delta=round(m2.summary.win_rate - m1.summary.win_rate, 6),
            avg_return_delta=round(m2.summary.avg_return - m1.summary.avg_return, 6),
            max_drawdown_delta=round(m2.summary.max_drawdown - m1.summary.max_drawdown, 6),
            sharpe_ratio_delta=round(m2.summary.sharpe_ratio - m1.summary.sharpe_ratio, 6),
            signal_count_delta=m2.summary.signal_count - m1.summary.signal_count,
            changed_strategy_paths=changed_paths,
        )


__all__ = [
    "RunArtifactStore",
    "build_data_fingerprint_payload",
    "build_safe_config_snapshot",
    "detect_code_version",
    "evaluate_protocol_gate",
]
