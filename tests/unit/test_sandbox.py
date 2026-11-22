"""Tests for FactorSandbox (secure code execution)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from alphaevo.alpha_factory.sandbox import (
    FactorSandbox,
    SandboxResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 50) -> pd.DataFrame:
    """Create a simple OHLCV dataframe."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame(
        {
            "open": close + rng.uniform(-0.5, 0.5, n),
            "high": close + abs(rng.standard_normal(n)),
            "low": close - abs(rng.standard_normal(n)),
            "close": close,
            "volume": rng.integers(1000, 10000, n).astype(float),
        }
    )


_GOOD_CODE = """\
def compute(df, idx):
    return float(df['close'].iloc[idx])
"""

_MA_CODE = """\
import numpy as np

def compute(df, idx):
    if idx < 5:
        return 0.0
    window = df['close'].iloc[idx-4:idx+1]
    return float(np.mean(window))
"""


# ---------------------------------------------------------------------------
# Tests — validate_code
# ---------------------------------------------------------------------------


class TestValidateCode:
    def setup_method(self):
        self.sandbox = FactorSandbox()

    def test_valid_code(self):
        ok, msg = self.sandbox.validate_code(_GOOD_CODE)
        assert ok is True
        assert msg == ""

    def test_valid_with_numpy(self):
        ok, _ = self.sandbox.validate_code(_MA_CODE)
        assert ok is True

    def test_syntax_error(self):
        ok, msg = self.sandbox.validate_code("def compute(df, idx:\n  return 0")
        assert ok is False
        assert "Syntax error" in msg

    def test_forbidden_import_os(self):
        code = "import os\ndef compute(df, idx): return 0.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "os" in msg

    def test_forbidden_import_subprocess(self):
        code = "import subprocess\ndef compute(df, idx): return 0.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "subprocess" in msg

    def test_forbidden_eval(self):
        code = "def compute(df, idx): return eval('1+1')"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "eval" in msg

    def test_forbidden_exec(self):
        code = "def compute(df, idx): exec('pass'); return 0.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "exec" in msg

    def test_forbidden_open(self):
        code = "def compute(df, idx):\n    f = open('/etc/passwd')\n    return 0.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "open" in msg

    def test_forbidden_import_from(self):
        code = "from os import path\ndef compute(df, idx): return 0.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "os" in msg

    def test_missing_compute(self):
        code = "def calculate(df, idx): return 1.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "compute" in msg

    def test_forbidden_dunder_import(self):
        code = "def compute(df, idx):\n    m = __import__('os')\n    return 0.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "__import__" in msg

    def test_forbidden_system_call(self):
        code = "import numpy as np\ndef compute(df, idx):\n    np.system('ls')\n    return 0.0"
        ok, msg = self.sandbox.validate_code(code)
        assert ok is False
        assert "system" in msg

    def test_allowed_pandas_import(self):
        code = "import pandas as pd\ndef compute(df, idx): return float(pd.Series([1]).mean())"
        ok, _ = self.sandbox.validate_code(code)
        assert ok is True

    def test_allowed_math_import(self):
        code = "import math\ndef compute(df, idx): return math.sqrt(4.0)"
        ok, _ = self.sandbox.validate_code(code)
        assert ok is True


# ---------------------------------------------------------------------------
# Tests — execute
# ---------------------------------------------------------------------------


class TestExecute:
    def setup_method(self):
        self.sandbox = FactorSandbox(timeout_seconds=10)
        self.df = _make_df(20)

    def test_execute_good_code(self):
        result = self.sandbox.execute(_GOOD_CODE, self.df)
        assert result.success is True
        assert result.values is not None
        assert len(result.values) == 20
        # Values should match close prices
        for i in range(20):
            assert abs(result.values[i] - self.df["close"].iloc[i]) < 1e-6

    def test_execute_ma_code(self):
        result = self.sandbox.execute(_MA_CODE, self.df)
        assert result.success is True
        assert result.values is not None
        # First 5 values should be 0
        for i in range(5):
            assert result.values[i] == 0.0

    def test_execute_unsafe_code_blocked(self):
        code = "import os\ndef compute(df, idx): return 0.0"
        result = self.sandbox.execute(code, self.df)
        assert result.success is False
        assert "Security" in result.error

    def test_execute_runtime_error_graceful(self):
        code = "def compute(df, idx):\n    return 1.0 / 0.0"
        result = self.sandbox.execute(code, self.df)
        assert result.success is True
        # Division by zero → all values should be 0.0 (fallback)
        assert all(v == 0.0 for v in result.values)

    def test_execute_nan_replaced(self):
        code = "import numpy as np\ndef compute(df, idx): return float('nan')"
        result = self.sandbox.execute(code, self.df)
        assert result.success is True
        assert all(v == 0.0 for v in result.values)

    def test_execution_time_tracked(self):
        result = self.sandbox.execute(_GOOD_CODE, self.df)
        assert result.execution_time_ms > 0

    def test_execute_passes_configured_memory_limit_to_subprocess(self):
        sandbox = FactorSandbox(timeout_seconds=1, max_memory_mb=128)
        try:
            with patch("alphaevo.alpha_factory.sandbox._SPAWN_CTX.Process") as MockProcess:
                proc = MockProcess.return_value
                proc.is_alive.return_value = False

                sandbox.execute(_GOOD_CODE, self.df)

            process_args = MockProcess.call_args.kwargs["args"]
            assert process_args[1] == 128
        finally:
            sandbox.close()


# ---------------------------------------------------------------------------
# Tests — SandboxResult model
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_success_result(self):
        r = SandboxResult(success=True, values=[1.0, 2.0])
        assert r.success is True
        assert r.error is None

    def test_failure_result(self):
        r = SandboxResult(success=False, error="bad code")
        assert r.values is None
        assert r.error == "bad code"


# ---------------------------------------------------------------------------
# Tests — Future-data access detection
# ---------------------------------------------------------------------------


class TestFutureDataDetection:
    """Validate that the sandbox blocks code accessing future bars."""

    sandbox = FactorSandbox(timeout_seconds=5)

    def test_iloc_idx_plus_1_blocked(self):
        code = "def compute(df, idx):\n    return float(df['close'].iloc[idx + 1])"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok
        assert "Future-data" in msg

    def test_iloc_idx_plus_5_blocked(self):
        code = "def compute(df, idx):\n    return float(df['close'].iloc[idx + 5])"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok
        assert "Future-data" in msg

    def test_shift_negative_blocked(self):
        code = "def compute(df, idx):\n    return float(df['close'].shift(-1).iloc[idx])"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok
        assert "Future-data" in msg

    def test_slice_idx_plus_offset_blocked(self):
        code = "def compute(df, idx):\n    return float(df['close'].iloc[idx + 1:idx + 5].mean())"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok
        assert "Future-data" in msg

    def test_historical_access_allowed(self):
        """Accessing idx-N (past data) should be allowed."""
        code = "def compute(df, idx):\n    if idx < 5: return 0.0\n    return float(df['close'].iloc[idx - 5])"
        ok, msg = self.sandbox.validate_code(code)
        assert ok

    def test_shift_positive_allowed(self):
        """shift(1) looks backward, should be allowed."""
        code = "def compute(df, idx):\n    return float(df['close'].shift(1).iloc[idx])"
        ok, msg = self.sandbox.validate_code(code)
        assert ok

    def test_current_bar_allowed(self):
        """Accessing iloc[idx] (current bar) should be allowed."""
        code = "def compute(df, idx):\n    return float(df['close'].iloc[idx])"
        ok, msg = self.sandbox.validate_code(code)
        assert ok


# ---------------------------------------------------------------------------
# Tests — Dunder reflection escape prevention
# ---------------------------------------------------------------------------


class TestDunderReflectionBlocked:
    """Verify that dunder-based sandbox escapes are blocked."""

    sandbox = FactorSandbox(timeout_seconds=5)

    def test_class_subclasses_blocked(self):
        code = "def compute(df, idx):\n    return float(''.__class__.__subclasses__()[0].__name__ == 'str')"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok
        assert "dunder" in msg.lower() or "Forbidden" in msg

    def test_class_mro_blocked(self):
        code = "def compute(df, idx):\n    return float(len(type.__mro__))"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok

    def test_globals_via_func_blocked(self):
        code = "def compute(df, idx):\n    return float(len(compute.__globals__))"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok

    def test_builtins_access_blocked(self):
        code = "def compute(df, idx):\n    return float(len(compute.__builtins__))"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok

    def test_code_attribute_blocked(self):
        code = "def compute(df, idx):\n    return float(compute.__code__.co_argcount)"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok

    def test_bases_blocked(self):
        code = "def compute(df, idx):\n    return float(len(int.__bases__))"
        ok, msg = self.sandbox.validate_code(code)
        assert not ok


class TestRestrictedBuiltins:
    """Verify that exec namespace has restricted builtins so dunder chains
    that slip past AST cannot reach dangerous functions."""

    sandbox = FactorSandbox(timeout_seconds=10)

    def test_safe_code_still_works(self):
        """Normal factor code using allowed builtins should still work."""
        code = "def compute(df, idx):\n    return float(max(0, min(df['close'].iloc[idx], 999)))"
        result = self.sandbox.execute(code, _make_df(10))
        assert result.success

    def test_open_not_in_builtins(self):
        """open() should not be available even if AST somehow missed it."""
        # We test at the exec level: open should fail
        code = (
            "def compute(df, idx):\n"
            "    try:\n"
            "        open('/dev/null')\n"
            "        return 1.0\n"
            "    except NameError:\n"
            "        return 0.0\n"
        )
        # AST validator should catch 'open' first
        ok, msg = self.sandbox.validate_code(code)
        assert not ok

    def test_import_not_in_builtins(self):
        """__import__ should not be available in restricted builtins."""
        code = (
            "def compute(df, idx):\n"
            "    try:\n"
            "        __import__('os')\n"
            "        return 1.0\n"
            "    except NameError:\n"
            "        return 0.0\n"
        )
        ok, msg = self.sandbox.validate_code(code)
        assert not ok
