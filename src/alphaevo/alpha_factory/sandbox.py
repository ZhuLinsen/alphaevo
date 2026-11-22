"""Secure sandbox for executing LLM-generated factor code.

Runs factor computation in a subprocess with:
- AST whitelist validation (no dangerous imports/calls)
- Timeout enforcement
- Memory limits (via resource module on Linux)
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import logging
import multiprocessing
import threading
import weakref
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from alphaevo.truth.pit_probe import PITProbeResult

logger = logging.getLogger(__name__)

_SPAWN_CTX = multiprocessing.get_context("spawn")
_ACTIVE_WORKER_STATES: dict[int, dict[str, Any]] = {}
_ACTIVE_WORKERS_LOCK = threading.Lock()


def _shutdown_worker_state(state: dict[str, Any]) -> None:
    """Terminate and reap a reusable sandbox worker process and its pipe."""
    with _ACTIVE_WORKERS_LOCK:
        _ACTIVE_WORKER_STATES.pop(id(state), None)
    conn = state.get("conn")
    proc = state.get("proc")
    state["conn"] = None
    state["proc"] = None
    if conn is not None:
        with contextlib.suppress(Exception):
            if not conn.closed:
                conn.send(None)
        with contextlib.suppress(Exception):
            conn.close()
    if proc is not None:
        with contextlib.suppress(Exception):
            if proc.is_alive():
                proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=1.0)


def _shutdown_all_workers() -> None:
    with _ACTIVE_WORKERS_LOCK:
        states = list(_ACTIVE_WORKER_STATES.values())
        _ACTIVE_WORKER_STATES.clear()
    for st in states:
        _shutdown_worker_state(st)


atexit.register(_shutdown_all_workers)

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
        "breakpoint",
        "help",
        "memoryview",
        "object",
        "super",
        "type",
        "hasattr",
        "id",
        "builtins",
        "__builtins__",
        "__loader__",
        "__spec__",
        "__file__",
        "__cached__",
        "__package__",
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

# Frame, coroutine, generator, traceback, and serialization/IO attributes
_FORBIDDEN_ATTRS = frozenset(
    {
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "tb_frame",
        "tb_next",
        "func_globals",
        "func_code",
        "im_func",
        "im_self",
        "im_class",
        "mro",
        "subclasses",
        "system",
        "popen",
        "spawn",
        "fork",
        "to_pickle",
        "read_pickle",
        "to_csv",
        "read_csv",
        "to_parquet",
        "read_parquet",
        "to_json",
        "read_json",
        "to_hdf",
        "read_hdf",
        "to_sql",
        "read_sql",
        "read_fwf",
        "read_table",
        "read_excel",
        "to_excel",
        "read_feather",
        "to_feather",
        "read_orc",
        "read_clipboard",
        "to_clipboard",
        "load",
        "loads",
        "dump",
        "dumps",
        "save",
        "savez",
        "fromfile",
        "tofile",
        "memmap",
        "ctypes",
        "query",
        "gi_yieldfrom",
        "cr_await",
        "ag_await",
        "tb_lasti",
        "base",
        "_mgr",
        "_values",
        "_data",
        "blocks",
        "loadtxt",
        "savetxt",
        "genfromtxt",
        "fromregex",
        "savez_compressed",
        "DataSource",
        "io",
        "ctypeslib",
        "testing",
        "distutils",
        "f2py",
        "lib",
    }
)

_SAFE_TO_ATTRS = frozenset(
    {
        "to_numpy",
        "tolist",
        "to_list",
        "to_dict",
        "to_frame",
        "to_series",
        "to_numeric",
        "to_datetime",
        "to_period",
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
    """Execute factor code safely inside an isolated ``spawn`` worker process.

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
        self._worker_lock = threading.RLock()
        self._worker_state: dict[str, Any] = {"proc": None, "conn": None}
        with _ACTIVE_WORKERS_LOCK:
            _ACTIVE_WORKER_STATES[id(self._worker_state)] = self._worker_state
        self._finalizer = weakref.finalize(self, _shutdown_worker_state, self._worker_state)

    def close(self) -> None:
        """Terminate and reap the backing sandbox worker process."""
        with self._worker_lock:
            _shutdown_worker_state(self._worker_state)

    def __enter__(self) -> FactorSandbox:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _reset_worker_locked(self) -> None:
        _shutdown_worker_state(self._worker_state)

    def _ensure_worker_locked(self) -> Connection:
        proc = self._worker_state.get("proc")
        conn = self._worker_state.get("conn")
        if proc is not None and proc.is_alive() and conn is not None and not conn.closed:
            return cast("Connection", conn)
        self._reset_worker_locked()
        with _ACTIVE_WORKERS_LOCK:
            _ACTIVE_WORKER_STATES[id(self._worker_state)] = self._worker_state
        parent_conn, child_conn = _SPAWN_CTX.Pipe(duplex=True)
        proc = _SPAWN_CTX.Process(
            target=_sandbox_worker_main,
            args=(child_conn, self.max_memory),
            daemon=True,
        )
        proc.start()
        with contextlib.suppress(Exception):
            child_conn.close()
        self._worker_state["proc"] = proc
        self._worker_state["conn"] = parent_conn
        return parent_conn

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
                    if alias.asname and (
                        alias.asname.startswith("_") or alias.asname in _FORBIDDEN_NAMES
                    ):
                        return False, f"Forbidden import alias: {alias.asname}"

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    return False, "Forbidden relative import"
                if not node.module or node.module not in _ALLOWED_IMPORTS:
                    return False, f"Forbidden import from: {node.module}"
                for alias in node.names:
                    if (
                        alias.name == "*"
                        or alias.name.startswith("_")
                        or alias.name in _FORBIDDEN_NAMES
                        or alias.name in _FORBIDDEN_ATTRS
                    ):
                        return False, f"Forbidden import symbol: {alias.name}"
                    if alias.asname and (
                        alias.asname.startswith("_") or alias.asname in _FORBIDDEN_NAMES
                    ):
                        return False, f"Forbidden import alias: {alias.asname}"

            # Check function calls and name references
            elif isinstance(node, ast.Name):
                if node.id in _FORBIDDEN_NAMES or node.id in _FORBIDDEN_DUNDERS:
                    return False, f"Forbidden name: {node.id}"

            # Check attribute access for forbidden patterns
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    return False, f"Forbidden dunder access: {node.attr}"
                if node.attr in _FORBIDDEN_NAMES:
                    return False, f"Forbidden attribute: {node.attr}"
                # Block dunder reflection chains (e.g. __class__.__subclasses__)
                if node.attr in _FORBIDDEN_DUNDERS:
                    return False, f"Forbidden dunder access: {node.attr}"
                if node.attr in _FORBIDDEN_ATTRS:
                    return False, f"Forbidden system/IO call: {node.attr}"
                if node.attr.startswith("read_") or (
                    node.attr.startswith("to_") and node.attr not in _SAFE_TO_ATTRS
                ):
                    return False, f"Forbidden IO attribute: {node.attr}"

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

    def preflight_code(self, code: str) -> tuple[bool, str]:
        """Static AST + bytecode compilation check without executing untrusted code."""
        is_safe, err = self.validate_code(code)
        if not is_safe:
            return False, err
        try:
            tree = ast.parse(code)
            compute_defs = [
                n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "compute"
            ]
            if not any(
                len(fn.args.posonlyargs) + len(fn.args.args) >= 2 or fn.args.vararg is not None
                for fn in compute_defs
            ):
                return False, "Function 'compute' must accept '(df, idx)' parameters"
            compile(code, "<factor_preflight>", "exec")
        except SyntaxError as exc:
            return False, f"Syntax error: {exc}"
        return True, ""

    def probe_pit(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        probe_indices: list[int] | None = None,
        atol: float = 1e-9,
    ) -> PITProbeResult:
        """Run runtime future-noise and truncation probes inside the hardened spawn worker."""
        import math as std_math

        from alphaevo.truth.pit_probe import PITProbeResult, PITProbeViolation

        is_safe, err_msg = self.preflight_code(code)
        if not is_safe:
            return PITProbeResult(passed=False, error=f"Security: {err_msg}")

        if len(df) < 10:
            return PITProbeResult(
                passed=False, error="DataFrame needs at least 10 rows for PIT probe"
            )

        raw_indices = (
            list(probe_indices)
            if probe_indices is not None
            else [
                max(5, int(len(df) * 0.35)),
                max(6, int(len(df) * 0.60)),
                max(7, min(len(df) - 3, int(len(df) * 0.80))),
            ]
        )
        indices = sorted({i for i in raw_indices if 1 <= i <= len(df) - 2})
        if not indices:
            return PITProbeResult(passed=False, error="No valid interior probe indices")

        clean_df = df.reset_index(drop=True)
        with self._worker_lock:
            try:
                conn = self._ensure_worker_locked()
                req = ("probe_pit", code, clean_df, indices, float(atol))
                try:
                    conn.send(req)
                except (EOFError, BrokenPipeError, OSError):
                    self._reset_worker_locked()
                    conn = self._ensure_worker_locked()
                    conn.send(req)

                if not conn.poll(self.timeout):
                    self._reset_worker_locked()
                    return PITProbeResult(
                        passed=False,
                        checked_indices=indices,
                        error=f"PIT probe timed out after {self.timeout}s",
                    )

                status, payload = conn.recv()
                if status != "ok":
                    return PITProbeResult(
                        passed=False,
                        checked_indices=indices,
                        error=str(payload),
                    )

                violations: list[PITProbeViolation] = []
                for v in payload:
                    raw_clean = float(v["clean_value"])
                    raw_pert = float(v["perturbed_value"])
                    p_type: Literal["future_noise", "truncation"] = (
                        "future_noise" if v.get("probe_type") == "future_noise" else "truncation"
                    )
                    violations.append(
                        PITProbeViolation(
                            probe_type=p_type,
                            probe_index=int(v["probe_index"]),
                            clean_value=round(raw_clean, 8)
                            if std_math.isfinite(raw_clean)
                            else 0.0,
                            perturbed_value=(
                                round(raw_pert, 8) if std_math.isfinite(raw_pert) else 0.0
                            ),
                            abs_diff=float(v["abs_diff"]),
                        )
                    )
                return PITProbeResult(
                    passed=len(violations) == 0,
                    checked_indices=indices,
                    violations=violations,
                )
            except (EOFError, BrokenPipeError):
                self._reset_worker_locked()
                return PITProbeResult(
                    passed=False,
                    checked_indices=indices,
                    error="PIT probe subprocess produced no output (may have crashed)",
                )
            except Exception as exc:
                self._reset_worker_locked()
                return PITProbeResult(
                    passed=False,
                    checked_indices=indices,
                    error=f"PIT probe error: {exc}",
                )

    def execute(
        self,
        code: str,
        df: pd.DataFrame,
        *,
        verify_pit: bool = False,
        enforce_pit_slice: bool = False,
    ) -> SandboxResult:
        """Execute factor code inside an isolated worker process.

        When ``verify_pit`` is True, also verifies runtime point-in-time invariance
        against future noise/truncation. When ``enforce_pit_slice`` is True, each bar
        ``idx`` only receives ``df.iloc[: idx + 1]``.
        """
        import time

        # Step 1: AST + preflight validation
        is_safe, err_msg = self.preflight_code(code)
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

        # Step 2: Execute in persistent spawn worker
        start = time.monotonic()
        with self._worker_lock:
            try:
                conn = self._ensure_worker_locked()
                try:
                    conn.send(("exec", code, df, enforce_pit_slice))
                except (EOFError, BrokenPipeError, OSError):
                    self._reset_worker_locked()
                    conn = self._ensure_worker_locked()
                    conn.send(("exec", code, df, enforce_pit_slice))

                if not conn.poll(self.timeout):
                    self._reset_worker_locked()
                    return SandboxResult(
                        success=False,
                        error=f"Execution timed out after {self.timeout}s",
                        execution_time_ms=(time.monotonic() - start) * 1000,
                    )

                status, payload = conn.recv()
                elapsed = (time.monotonic() - start) * 1000
                if status != "ok":
                    return SandboxResult(
                        success=False,
                        error=str(payload),
                        execution_time_ms=elapsed,
                    )
                return SandboxResult(
                    success=True,
                    values=payload,
                    execution_time_ms=elapsed,
                )
            except (EOFError, BrokenPipeError):
                self._reset_worker_locked()
                return SandboxResult(
                    success=False,
                    error="Subprocess produced no output (may have crashed)",
                    execution_time_ms=(time.monotonic() - start) * 1000,
                )
            except Exception as e:
                self._reset_worker_locked()
                return SandboxResult(
                    success=False,
                    error=f"Sandbox error: {e}",
                    execution_time_ms=(time.monotonic() - start) * 1000,
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


class _SafeModuleProxy:
    """Attribute-filtering proxy exposing computational symbols without file/IO helpers."""

    __slots__ = ("_module", "_module_name")

    def __init__(self, module: Any, module_name: str) -> None:
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_module_name", module_name)

    def __getattr__(self, name: str) -> Any:
        if (
            name.startswith("_")
            or name.startswith("read_")
            or (name.startswith("to_") and name not in _SAFE_TO_ATTRS)
            or name in _FORBIDDEN_ATTRS
            or name in _FORBIDDEN_NAMES
            or name in _FORBIDDEN_DUNDERS
        ):
            mod_name = object.__getattribute__(self, "_module_name")
            raise AttributeError(f"Access to '{mod_name}.{name}' is blocked in factor sandbox")
        return getattr(object.__getattribute__(self, "_module"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Mutating sandboxed modules is forbidden")

    def __dir__(self) -> list[str]:
        return []


_IN_FACTOR_EXEC = False
_AUDIT_HOOK_INSTALLED = False

_BLOCKED_AUDIT_EXACT = frozenset({"open"})
_BLOCKED_AUDIT_PREFIXES = (
    "os.",
    "subprocess.",
    "socket.",
    "urllib.",
    "http.client.",
    "shutil.",
    "ctypes.",
    "ftplib.",
    "smtplib.",
    "webbrowser.",
    "tempfile.",
)
_BLOCKED_IMPORT_ROOTS = frozenset(
    {
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
        "ftplib",
        "smtplib",
        "webbrowser",
        "tempfile",
        "sqlite3",
        "ssl",
        "pty",
        "posix",
        "nt",
    }
)


def _install_worker_audit_hook() -> None:
    """Register a process-wide PEP 578 audit hook blocking I/O during factor execution."""
    global _AUDIT_HOOK_INSTALLED
    if _AUDIT_HOOK_INSTALLED:
        return
    import sys

    def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
        if not _IN_FACTOR_EXEC:
            return
        if event in _BLOCKED_AUDIT_EXACT or event.startswith(_BLOCKED_AUDIT_PREFIXES):
            raise PermissionError(f"Sandbox audit hook blocked operation: {event}")
        if event == "import" and args:
            mod = str(args[0] or "")
            root = mod.split(".", 1)[0]
            if root in _BLOCKED_IMPORT_ROOTS:
                raise PermissionError(f"Sandbox audit hook blocked import: {mod}")

    sys.addaudithook(_audit_hook)
    _AUDIT_HOOK_INSTALLED = True


def build_restricted_namespace() -> dict[str, Any]:
    """Build an isolated execution namespace with safe non-I/O builtins and proxied modules."""
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
    import math as _math

    safe_np = _SafeModuleProxy(np, "numpy")
    safe_pd = _SafeModuleProxy(pd, "pandas")
    safe_math = _SafeModuleProxy(_math, "math")
    allowed_modules: dict[str, Any] = {
        "numpy": safe_np,
        "pandas": safe_pd,
        "math": safe_math,
        "np": safe_np,
        "pd": safe_pd,
    }

    def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        if name in allowed_modules:
            return allowed_modules[name]
        raise ImportError(f"Import of '{name}' is not allowed in factor code")

    safe_builtins["__import__"] = _safe_import
    return {
        "__builtins__": safe_builtins,
        "np": safe_np,
        "pd": safe_pd,
        "math": safe_math,
        "numpy": safe_np,
        "pandas": safe_pd,
    }


def _execute_in_worker_namespace(
    code: str,
    df: pd.DataFrame,
    *,
    enforce_pit_slice: bool = False,
) -> list[float]:
    """Compile and evaluate factor code inside a clean restricted namespace."""
    namespace = build_restricted_namespace()
    exec(code, namespace)  # noqa: S102 — code has passed AST + preflight validation
    compute_fn = namespace.get("compute")
    if compute_fn is None or not callable(compute_fn):
        raise ValueError("No callable 'compute' function defined")

    values: list[float] = []
    for idx in range(len(df)):
        try:
            target_df = df.iloc[: idx + 1].copy(deep=True) if enforce_pit_slice else df
            val = float(compute_fn(target_df, idx))
            if not np.isfinite(val):
                val = 0.0
            values.append(val)
        except (PermissionError, SandboxSecurityError):
            raise
        except Exception:
            values.append(0.0)
    return values


def _probe_pit_in_worker(
    code: str,
    prepared: list[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    atol: float,
) -> list[dict[str, Any]]:
    """Run clean vs future-corrupted vs truncated PIT comparisons in restricted namespace."""
    namespace = build_restricted_namespace()
    exec(code, namespace)  # noqa: S102 — code has passed AST + preflight validation
    compute_fn = namespace.get("compute")
    if compute_fn is None or not callable(compute_fn):
        raise ValueError("No callable 'compute' function defined")

    violations: list[dict[str, Any]] = []
    for idx, clean_copy, noisy_df, trunc_df in prepared:
        raw_clean = float(compute_fn(clean_copy, idx))
        clean_val = raw_clean if np.isfinite(raw_clean) else 0.0

        raw_noisy = float(compute_fn(noisy_df, idx))
        noisy_val = raw_noisy if np.isfinite(raw_noisy) else 0.0
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

        raw_trunc = float(compute_fn(trunc_df, idx))
        trunc_val = raw_trunc if np.isfinite(raw_trunc) else 0.0
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
    return violations


def _sandbox_worker_main(conn: Connection, max_memory_mb: int) -> None:
    """Persistent spawn worker main loop with BLAS thread caps, RLIMIT_AS, and audit hooks."""
    global _IN_FACTOR_EXEC
    import os

    for env_key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(env_key, "1")

    try:
        import resource

        mem_bytes = max_memory_mb * 1024 * 1024
        try:
            with open("/proc/self/statm", encoding="utf-8") as statm_f:
                vm_pages = int(statm_f.read().split()[0])
            current_vm = vm_pages * os.sysconf("SC_PAGE_SIZE")
            mem_bytes = max(mem_bytes, current_vm + mem_bytes)
        except (OSError, ValueError):
            pass
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            mem_bytes = min(mem_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))
    except (ImportError, ValueError) as e:
        import sys

        print(
            f"[sandbox] Memory limits not available ({e}); "
            "factor code runs without memory protection",
            file=sys.stderr,
        )

    from alphaevo.truth.pit_probe import _corrupt_future_rows

    _install_worker_audit_hook()

    while True:
        try:
            request = conn.recv()
        except EOFError:
            break
        if request is None:
            break
        if isinstance(request, tuple) and len(request) == 5 and request[0] == "probe_pit":
            _tag, code, clean_df, indices, atol = request
            prepared = [
                (
                    idx,
                    cast("pd.DataFrame", clean_df.copy(deep=True)),
                    _corrupt_future_rows(clean_df, idx),
                    cast(
                        "pd.DataFrame",
                        clean_df.iloc[: idx + 1].copy(deep=True).reset_index(drop=True),
                    ),
                )
                for idx in indices
            ]
            try:
                _IN_FACTOR_EXEC = True
                violations = _probe_pit_in_worker(code, prepared, float(atol))
                conn.send(("ok", violations))
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                conn.send(("err", f"{type(exc).__name__}: {exc}"))
            finally:
                _IN_FACTOR_EXEC = False
            continue

        if isinstance(request, tuple) and len(request) == 4 and request[0] == "exec":
            _tag, code, df, enforce_pit_slice = request
        else:
            code, df, enforce_pit_slice = request
        try:
            _IN_FACTOR_EXEC = True
            values = _execute_in_worker_namespace(
                code,
                df,
                enforce_pit_slice=bool(enforce_pit_slice),
            )
            conn.send(("ok", values))
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            conn.send(("err", f"{type(exc).__name__}: {exc}"))
        finally:
            _IN_FACTOR_EXEC = False


def _run_in_subprocess(
    code: str,
    df: pd.DataFrame,
    result_queue: multiprocessing.Queue,
    max_memory_mb: int,
) -> None:
    """Compatibility helper executing factor code with resource limits."""
    try:
        import os
        import resource

        mem_bytes = max_memory_mb * 1024 * 1024
        try:
            with open("/proc/self/statm", encoding="utf-8") as statm_f:
                vm_pages = int(statm_f.read().split()[0])
            current_vm = vm_pages * os.sysconf("SC_PAGE_SIZE")
            mem_bytes = max(mem_bytes, current_vm + mem_bytes)
        except (OSError, ValueError):
            pass
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            mem_bytes = min(mem_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard))
    except (ImportError, ValueError):
        pass

    try:
        result_queue.put(_execute_in_worker_namespace(code, df, enforce_pit_slice=False))
    except Exception as e:
        result_queue.put(e)
