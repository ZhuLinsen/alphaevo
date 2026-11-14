"""Smoke tests for RSI, Shadow Ledger, and Truth CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alphaevo.cli.main import app


def test_cli_rsi_shadow_and_truth_commands(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ALPHAEVO_DB_PATH", str(tmp_path / "cli_test.db"))
    monkeypatch.setenv("ALPHAEVO_SHADOW_LEDGER_DIR", str(tmp_path / "shadow"))

    runner = CliRunner()

    # 1. alphaevo rsi run
    res_rsi = runner.invoke(
        app,
        [
            "rsi",
            "run",
            "rsi_reversion_v1",
            "--rounds",
            "2",
            "--output",
            str(tmp_path / "rsi_rep"),
        ],
    )
    assert res_rsi.exit_code == 0, res_rsi.output
    assert (tmp_path / "rsi_rep" / "results.tsv").is_file()

    # 2. alphaevo truth trials
    res_trials = runner.invoke(app, ["truth", "trials", "--limit", "5"])
    assert res_trials.exit_code == 0, res_trials.output
    assert "TrialLedger" in res_trials.output

    # 3. alphaevo truth pit-probe on a clean causal factor file
    factor_py = tmp_path / "clean_factor.py"
    factor_py.write_text(
        "def compute(df, idx):\n"
        "    if idx < 5:\n"
        "        return 0.0\n"
        '    return float(df["close"].iloc[idx] / df["close"].iloc[idx - 5] - 1.0)\n',
        encoding="utf-8",
    )
    res_pit = runner.invoke(app, ["truth", "pit-probe", str(factor_py)])
    assert res_pit.exit_code == 0, res_pit.output
    assert "PASSED" in res_pit.output

    # 4. alphaevo shadow workflow
    r_reg = runner.invoke(app, ["shadow", "register", "ma_crossover_v1"])
    assert r_reg.exit_code == 0, r_reg.output

    r_emit = runner.invoke(app, ["shadow", "emit", "ma_crossover_v1"])
    assert r_emit.exit_code == 0, r_emit.output

    r_verify = runner.invoke(app, ["shadow", "verify"])
    assert r_verify.exit_code == 0, r_verify.output

    r_status = runner.invoke(app, ["shadow", "status"])
    assert r_status.exit_code == 0, r_status.output
