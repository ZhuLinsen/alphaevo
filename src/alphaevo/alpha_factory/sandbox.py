"""Secure sandbox for executing LLM-generated factor code.

Runs factor computation in a subprocess with:
- AST whitelist validation (no dangerous imports/calls)
- Timeout enforcement
- Memory limits (via resource module on Linux)
"""

from __future__ import annotations

import ast
import logging
import multiprocessing
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel

if TYPE_CHECKING_IMPORT := False:
    pass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaevo.truth.pit_probe import PITProbeResult

logger = logging.getLogger(__name__)

# Modules allowed inside factor code
_ALLOWED_IMPORTS = frozenset({"numpy", "pandas", "math", "np", "pd"})

# Names that must NEVER appear in factor code
_FORBIDDEN_NAMES = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "open",
        "input",
        "print",
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "http",
        "urllib",
        "requests",
        "pickle",
        "shelve",
        "marshal",
        "importlib",
        "ctypes",
    }
)

# Dunder attributes used in reflection-based sandbox escapes
_FORBIDDEN_DUNDERS = frozenset(
    {
        "__class__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__func__",
        "__self__",
        "__module__",
        "__dict__",
        "__init_subclass__",
        "__reduce__",
        "__reduce_ex__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
    }
)


class SandboxResult(BaseModel):
    """Result of a sandboxed factor execution."""

    success: bool
    values: list[float] | None = None
    error: str | None = None
    execution_time_ms: float = 0


class SandboxSecurityError(Exception):
    """Raised when factor code fails security validation."""


class FactorSandbox:
    """Execute factor code safely.

    Example::

        sandbox = FactorSandbox(timeout_seconds=10)
        result = sandbox.execute(code_string, ohlcv_dataframe)
        if result.success:
            factor_values = result.values
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = 30,
        max_memory_mb: int = 512,
    ) -> None:
        self.timeout = timeout_seconds
        self.max_memory = max_memory_mb

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Validate factor code via AST analysis.

        Returns (is_safe, error_message).
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in _ALLOWED_IMPORTS:
                        return False, f"Forbidden import: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in _ALLOWED_IMPORTS:
                    return False, f"Forbidden import from: {node.module}"

            # Check function calls to forbidden names
            elif isinstance(node, ast.Name):
                if node.id in _FORBIDDEN_NAMES:
                    return False, f"Forbidden name: {node.id}"

            # Check attribute access for forbidden patterns
            elif isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_NAMES:
                    return False, f"Forbidden attribute: {node.attr}"
                # Block dunder reflection chains (e.g. __class__.__subclasses__)
                if node.attr in _FORBIDDEN_DUNDERS:
                    return False, f"Forbidden dunder access: {node.attr}"
                # Block system() and similar
                if node.attr in ("system", "popen", "spawn", "fork"):
                    return False, f"Forbidden system call: {node.attr}"

            # Check string-based eval/exec patterns
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _FORBIDDEN_NAMES
            ):
                return False, f"Forbidden call: {node.func.id}"

        # Check for future-data access patterns
        future_err = _check_future_data_access(tree)
        if future_err:
            return False, future_err

        # Verify the code defines a 'compute' function
        func_defs = [
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute"
        ]
        if not func_defs:
            return False, "Code must define a 'compute(df, idx) -> float' function"

        return True, ""

    def probe_pit(self, code: str, df: pd.DataFrame) -> PITProbeResult:
        """Run runtime future-noise and truncation probes to catch hidden lookahead."""
        from alphaevo.truth.pit_probe import probe_point_in_time_invariance

        return probe_point_in_time_invariance(
            code,
            df,
            timeout_seconds=self.timeout,
        )

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        verify_pit: bool = False,
    ) -> SandboxResult:
        """Execute factor code and compute values for all valid indices.

        The code is run in a child process for isolation. When ``verify_pit`` is True,
        also verifies runtime point-in-time invariance against future noise/truncation.
        """
        import time

        # Step 1: AST validation
        is_safe, err_msg = self.validate_code(code)
        if not is_safe:
            return SandboxResult(success=False, error=f"Security: {err_msg}")

        # Step 1b: Optional runtime PIT future-noise probe
        if verify_pit and len(df) >= 10:
            pit = self.probe_pit(code, df)
            if not pit.passed:
                detail = pit.error or (
                    f"{pit.violations[0].probe_type} leakage at idx={pit.violations[0].probe_index}"
                    if pit.violations
                    else "point-in-time check failed"
                )
                return SandboxResult(success=False, error=f"PIT violation: {detail}")

        # Step 2: Execute in subprocess
        start = time.monotonic()
        try:
            result_queue: multiprocessing.Queue = multiprocessing.Queue()
            proc = multiprocessing.Process(
                target=_run_in_subprocess,
                args=(code, df, result_queue, self.max_memory),
            )
            proc.start()
            proc.join(timeout=self.timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                return SandboxResult(
                    success=False,
                    error=f"Execution timed out after {self.timeout}s",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                )

            elapsed = (time.monotonic() - start) * 1000

            if result_queue.empty():
                return SandboxResult(
                    success=False,
                    error="Subprocess produced no output (may have crashed)",
                    execution_time_ms=elapsed,
                )

            result = result_queue.get_nowait()
            if isinstance(result, Exception):
                return SandboxResult(
                    success=False,
                    error=str(result),
                    execution_time_ms=elapsed,
                )

            return SandboxResult(
                success=True,
                values=result,
                execution_time_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(
                success=False,
                error=f"Sandbox error: {e}",
                execution_time_ms=elapsed,
            )


def _check_future_data_access(tree: ast.AST) -> str | None:
    """Detect code patterns that access future data relative to ``idx``.

    Catches:
    - ``df.iloc[idx + N]`` or ``df.iloc[idx + 1:]`` (positive offset)
    - ``df.shift(-N)`` (negative shift = look-ahead)
    - ``df.iloc[idx:]`` open-ended slices past current bar
    """
    for node in ast.walk(tree):
        # Pattern: df.shift(-N)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "shift"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                return "Future-data access: shift with negative periods looks ahead"
            if (
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, (int, float))
                and arg.value < 0
            ):
                return "Future-data access: shift with negative periods looks ahead"

        # Pattern: subscript with idx + positive_offset
        if isinstance(node, ast.Subscript):
            sl = node.slice
            # Simple index: iloc[idx + N]
            if (
                isinstance(sl, ast.BinOp)
                and isinstance(sl.op, ast.Add)
                and isinstance(sl.left, ast.Name)
                and sl.left.id == "idx"
                and isinstance(sl.right, ast.Constant)
                and isinstance(sl.right.value, (int, float))
                and sl.right.value > 0
            ):
                return f"Future-data access: idx + {sl.right.value} reads beyond current bar"

            # Slice: iloc[idx + N : ...] with positive N
            if isinstance(sl, ast.Slice) and sl.lower is not None:
                lower = sl.lower
                if (
                    isinstance(lower, ast.BinOp)
                    and isinstance(lower.op, ast.Add)
                    and isinstance(lower.left, ast.Name)
                    and lower.left.id == "idx"
                    and isinstance(lower.right, ast.Constant)
                    and isinstance(lower.right.value, (int, float))
                    and lower.right.value > 0
                ):
                    return f"Future-data access: slice starting at idx + {lower.right.value}"

    return None


def _run_in_subprocess(
    code: str,
    df: pd.DataFrame,
    result_queue: multiprocessing.Queue,
    max_memory_mb: int,
) -> None:
    """Worker function that runs in a child process."""
    try:
        # Set resource limits (Linux only)
        try:
            import os
            import resource

            # Memory limit (add inherited VmSize on fork so parent OpenBLAS/thread
            # arenas do not immediately exhaust RLIMIT_AS in the child)
            mem_bytes = max_memory_mb * 1024 * 1024
            try:
                with open("/proc/self/statm", encoding="utf-8") as statm_f:
                    vm_pages = int(statm_f.read().split()[0])
                current_vm = vm_pages * os.sysconf("SC_PAGE_SIZE")
                mem_bytes = max(mem_bytes, current_vm + mem_bytes)
            except (OSError, ValueError):
                pass
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except (ImportError, ValueError) as e:
            import sys

            print(
                f"[sandbox] Memory limits not available ({e}); "
                "factor code runs without memory protection",
                file=sys.stderr,
            )

        # Build execution namespace with restricted builtins.
        # Only safe, non-IO builtins are exposed to prevent sandbox escapes
        # even if a dunder chain slips past the AST validator.
        _SAFE_BUILTINS = {  # noqa: N806
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
                    "frozenset",
                    "bytes",
                    "bytearray",
                    "complex",
                    "len",
                    "range",
                    "enumerate",
                    "zip",
                    "map",
                    "filter",
                    "sorted",
                    "reversed",
                    "min",
                    "max",
                    "sum",
                    "abs",
                    "round",
                    "any",
                    "all",
                    "isinstance",
                    "issubclass",
                    "type",
                    "hasattr",
                    "id",
                    "hash",
                    "repr",
                    "ValueError",
                    "TypeError",
                    "KeyError",
                    "IndexError",
                    "RuntimeError",
                    "StopIteration",
                    "ZeroDivisionError",
                    "ArithmeticError",
                    "Exception",
                    "AttributeError",
                )
            )
            if v is not None
        }
        # Controlled __import__ that only allows whitelisted modules
        _allowed_modules = {
            "numpy": np,
            "pandas": pd,
            "math": __import__("math"),
            "np": np,
            "pd": pd,
        }

        def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in _allowed_modules:
                return _allowed_modules[name]
            raise ImportError(f"Import of '{name}' is not allowed in factor code")

        _SAFE_BUILTINS["__import__"] = _safe_import
        namespace: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "np": np,
            "pd": pd,
            "math": __import__("math"),
            "numpy": np,
            "pandas": pd,
        }

        # Execute the code to define the compute function
        exec(code, namespace)  # noqa: S102 — code has passed AST validation

        compute_fn = namespace.get("compute")
        if compute_fn is None or not callable(compute_fn):
            result_queue.put(ValueError("No callable 'compute' function defined"))
            return

        # Compute values for each index
        values = []
        for idx in range(len(df)):
            try:
                val = float(compute_fn(df, idx))
                if not np.isfinite(val):
                    val = 0.0
                values.append(val)
            except Exception:
                values.append(0.0)

        result_queue.put(values)

    except Exception as e:
        result_queue.put(e)
