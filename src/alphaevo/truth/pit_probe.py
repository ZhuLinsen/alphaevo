"""Runtime Point-in-Time (PIT) Future-Noise & Truncation Probe for L2 factors."""

from __future__ import annotations

import math
import multiprocessing
from typing import Any, Literal, cast

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, Field


class PITProbeViolation(BaseModel):
    """Details of one lookahead leakage failure detected by the runtime PIT probe."""

    probe_type: Literal["future_noise", "truncation"]
    probe_index: int
    clean_value: float
    perturbed_value: float
    abs_diff: float


class PITProbeResult(BaseModel):
    """Outcome of runtime PIT invariance verification for factor code."""

    passed: bool
    checked_indices: list[int] = Field(default_factory=list)
    violations: list[PITProbeViolation] = Field(default_factory=list)
    error: str | None = None


def _corrupt_future_rows(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    """Create a deep copy of ``df`` identical on ``0..idx`` with corrupted future rows."""
    mutated = cast("pd.DataFrame", df.copy(deep=True))
    if idx + 1 >= len(mutated):
        return mutated
    future_slice = slice(idx + 1, len(mutated))
    for col, factor, offset in (
        ("close", 97.5, 999.0),
        ("open", 73.0, 555.0),
        ("high", 150.0, 1500.0),
        ("low", 0.01, 0.05),
        ("volume", 421.0, 10_000_000.0),
        ("prev_close", 88.0, 777.0),
    ):
        if col in mutated.columns:
            vals = pd.to_numeric(
                mutated.loc[mutated.index[future_slice], col], errors="coerce"
            ).fillna(1.0)
            mutated.loc[mutated.index[future_slice], col] = vals * factor + offset
    return mutated


def _pit_probe_worker(
    code: str,
    df: pd.DataFrame,
    indices: list[int],
    atol: float,
    result_queue: multiprocessing.Queue[Any],
) -> None:
    """Execute PIT clean vs future-corrupted vs truncated comparisons in isolation."""
    try:
        import math as std_math

        import numpy as np

        safe_builtins: dict[str, Any] = {
            k: v
            for k, v in (
                (
                    k,
                    __builtins__[k]
                    if isinstance(__builtins__, dict)
                    else getattr(__builtins__, k, None),
                )
                for k in (
                    "True",
                    "False",
                    "None",
                    "int",
                    "float",
                    "str",
                    "bool",
                    "list",
                    "tuple",
                    "dict",
                    "set",
                    "len",
                    "range",
                    "enumerate",
                    "zip",
                    "min",
                    "max",
                    "sum",
                    "abs",
                    "round",
                    "any",
                    "all",
                    "isinstance",
                    "ValueError",
                    "TypeError",
                    "KeyError",
                    "IndexError",
                    "ZeroDivisionError",
                    "Exception",
                )
            )
            if v is not None
        }
        allowed = {"numpy": np, "pandas": pd, "math": std_math, "np": np, "pd": pd}

        def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            if name in allowed:
                return allowed[name]
            raise ImportError(f"Import of '{name}' is not allowed")

        safe_builtins["__import__"] = _safe_import
        namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "np": np,
            "pd": pd,
            "math": std_math,
            "numpy": np,
            "pandas": pd,
        }
        exec(code, namespace)  # noqa: S102 — caller pre-validated code via AST whitelist
        compute_fn = namespace.get("compute")
        if compute_fn is None or not callable(compute_fn):
            result_queue.put({"error": "No callable 'compute' function defined"})
            return

        violations: list[dict[str, Any]] = []
        for idx in indices:
            clean_val = float(compute_fn(df, idx))
            if not std_math.isfinite(clean_val):
                clean_val = 0.0

            noisy_df = _corrupt_future_rows(df, idx)
            noisy_val = float(compute_fn(noisy_df, idx))
            if not std_math.isfinite(noisy_val):
                noisy_val = 0.0
            diff_noise = abs(clean_val - noisy_val)
            if diff_noise > atol:
                violations.append(
                    {
                        "probe_type": "future_noise",
                        "probe_index": idx,
                        "clean_value": clean_val,
                        "perturbed_value": noisy_val,
                        "abs_diff": diff_noise,
                    }
                )

            trunc_df = df.iloc[: idx + 1].copy(deep=True).reset_index(drop=True)
            trunc_val = float(compute_fn(trunc_df, idx))
            if not std_math.isfinite(trunc_val):
                trunc_val = 0.0
            diff_trunc = abs(clean_val - trunc_val)
            if diff_trunc > atol:
                violations.append(
                    {
                        "probe_type": "truncation",
                        "probe_index": idx,
                        "clean_value": clean_val,
                        "perturbed_value": trunc_val,
                        "abs_diff": diff_trunc,
                    }
                )

        result_queue.put({"violations": violations, "error": None})
    except Exception as exc:
        result_queue.put({"violations": [], "error": str(exc)})


def probe_point_in_time_invariance(
    code: str,
    df: pd.DataFrame,
    *,
    probe_indices: list[int] | None = None,
    atol: float = 1e-9,
    timeout_seconds: int = 10,
) -> PITProbeResult:
    """Verify at runtime that ``compute(df, idx)`` never depends on rows ``> idx``."""
    from alphaevo.alpha_factory.sandbox import FactorSandbox

    sandbox = FactorSandbox(timeout_seconds=timeout_seconds)
    is_safe, err = sandbox.validate_code(code)
    if not is_safe:
        return PITProbeResult(passed=False, error=f"Security: {err}")

    if len(df) < 10:
        return PITProbeResult(passed=False, error="DataFrame needs at least 10 rows for PIT probe")

    indices = probe_indices or [
        max(5, int(len(df) * 0.35)),
        max(6, int(len(df) * 0.60)),
        max(7, min(len(df) - 3, int(len(df) * 0.80))),
    ]
    indices = sorted({i for i in indices if 1 <= i <= len(df) - 2})
    if not indices:
        return PITProbeResult(passed=False, error="No valid interior probe indices")

    clean_df = df.reset_index(drop=True)
    queue: multiprocessing.Queue[Any] = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_pit_probe_worker,
        args=(code, clean_df, indices, atol, queue),
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=3)
        if proc.is_alive():
            proc.kill()
        return PITProbeResult(
            passed=False,
            checked_indices=indices,
            error=f"PIT probe timed out after {timeout_seconds}s",
        )

    if queue.empty():
        return PITProbeResult(
            passed=False,
            checked_indices=indices,
            error="PIT probe subprocess produced no output",
        )

    payload = queue.get_nowait()
    if payload.get("error"):
        return PITProbeResult(
            passed=False,
            checked_indices=indices,
            error=str(payload["error"]),
        )

    violations = [
        PITProbeViolation(
            probe_type=v["probe_type"],
            probe_index=int(v["probe_index"]),
            clean_value=round(float(v["clean_value"]), 8)
            if math.isfinite(v["clean_value"])
            else 0.0,
            perturbed_value=(
                round(float(v["perturbed_value"]), 8)
                if math.isfinite(v["perturbed_value"])
                else 0.0
            ),
            abs_diff=float(v["abs_diff"]),
        )
        for v in payload.get("violations", [])
    ]
    return PITProbeResult(
        passed=len(violations) == 0,
        checked_indices=indices,
        violations=violations,
    )


__all__ = [
    "PITProbeResult",
    "PITProbeViolation",
    "probe_point_in_time_invariance",
]
