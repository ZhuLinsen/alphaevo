"""Local data cache with parquet-preferred persistence and pickle fallback.

Caches downloaded market data to avoid redundant network requests.
Historical data (past dates) never expires. Today's data expires
after a configurable TTL.

File-level locking prevents corruption when multiple processes write
to the same cache entry concurrently.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

try:
    import fcntl as _fcntl
except ImportError:  # Windows
    _fcntl = None  # type: ignore[assignment]

try:
    import msvcrt as _msvcrt  # type: ignore[import-not-found]
except ImportError:
    _msvcrt = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_TTL_SECONDS = 4 * 3600  # 4 hours for today's data
_SAFE_TOKEN_RE = re.compile(r"[^\w.\-]")


def _sanitize_token(token: str) -> str:
    """Sanitize a path segment while disallowing dot-only traversal names."""
    safe = _SAFE_TOKEN_RE.sub("_", token.strip())
    if not safe or safe in {".", ".."}:
        return "_"
    return safe


class DataCache:
    """Disk-backed data cache.

    Layout::

        {cache_dir}/{symbol}/{start}_{end}.parquet
        {cache_dir}/{namespace}/{symbol}/{start}_{end}.parquet  # when namespace is provided
        {cache_dir}/.../{start}_{end}.pkl                       # fallback without parquet engine
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _namespace_root(self, namespace: str | None = None) -> Path:
        root = self.cache_dir.resolve()
        if not namespace:
            return root
        safe_ns = _sanitize_token(namespace)
        target = (root / safe_ns).resolve()
        if not str(target).startswith(str(root)):
            raise ValueError(f"Invalid cache namespace (path traversal detected): {namespace}")
        return target

    def _cache_path(
        self,
        symbol: str,
        start: date,
        end: date,
        suffix: str,
        *,
        namespace: str | None = None,
    ) -> Path:
        """Compute a cache file path for a given query, suffix, and optional namespace."""
        safe_symbol = _sanitize_token(symbol)
        base_root = self._namespace_root(namespace)
        path = (base_root / safe_symbol / f"{start}_{end}{suffix}").resolve()
        if not str(path).startswith(str(base_root)):
            raise ValueError(f"Invalid symbol (path traversal detected): {symbol}")
        return path

    def _resolve_existing_path(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        namespace: str | None = None,
    ) -> Path | None:
        """Return the best on-disk cache path for the requested interval.

        Preference order:
          1. exact range match
          2. smallest cached range that fully covers the requested interval
        """
        for suffix in (".parquet", ".pkl"):
            path = self._cache_path(symbol, start, end, suffix, namespace=namespace)
            if path.is_file():
                return path

        best_candidate: tuple[int, float, int, Path] | None = None
        for path in self._iter_symbol_cache_files(symbol, namespace=namespace):
            interval = self._parse_cache_interval(path)
            if interval is None:
                continue
            cached_start, cached_end = interval
            if cached_start > start or cached_end < end:
                continue

            span_days = (cached_end - cached_start).days
            freshness = path.stat().st_mtime
            suffix_rank = 0 if path.suffix == ".parquet" else 1
            candidate = (span_days, -freshness, suffix_rank, path)
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        return None if best_candidate is None else best_candidate[3]

    def get(
        self,
        symbol: str,
        start: date,
        end: date,
        *,
        namespace: str | None = None,
    ) -> pd.DataFrame | None:
        """Read cached data if it exists and is still valid.

        Returns None on cache miss or expiry.
        """
        path = self._resolve_existing_path(symbol, start, end, namespace=namespace)
        if path is None:
            self._misses += 1
            return None

        # Check TTL — if end date is today (or later), apply TTL
        if end >= date.today():
            age = time.time() - path.stat().st_mtime
            if age > self.ttl_seconds:
                logger.debug("Cache expired for %s (%s → %s)", symbol, start, end)
                self._misses += 1
                return None

        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_pickle(path)
            self._hits += 1
            logger.debug("Cache hit for %s (%s → %s)", symbol, start, end)
            return df
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", symbol, e)
            self._misses += 1
            return None

    def put(
        self,
        symbol: str,
        start: date,
        end: date,
        df: pd.DataFrame,
        *,
        namespace: str | None = None,
    ) -> None:
        """Write data to cache (file-locked to prevent concurrent corruption)."""
        if df is None or df.empty:
            return

        parquet_path = self._cache_path(symbol, start, end, ".parquet", namespace=namespace)
        pickle_path = self._cache_path(symbol, start, end, ".pkl", namespace=namespace)
        parquet_path.parent.mkdir(parents=True, exist_ok=True)

        with self._file_lock(parquet_path):
            try:
                df.to_parquet(parquet_path, index=False)
                if pickle_path.exists():
                    pickle_path.unlink()
                logger.debug("Cached %d rows for %s (%s → %s)", len(df), symbol, start, end)
                return
            except Exception as e:
                logger.debug(
                    "Parquet cache write failed for %s, falling back to pickle: %s", symbol, e
                )

            try:
                df.to_pickle(pickle_path)
                if parquet_path.exists():
                    parquet_path.unlink()
                logger.debug(
                    "Cached %d rows for %s (%s → %s) using pickle fallback",
                    len(df),
                    symbol,
                    start,
                    end,
                )
            except Exception as e:
                logger.warning("Cache write failed for %s: %s", symbol, e)

    @staticmethod
    @contextmanager
    def _file_lock(target: Path) -> Iterator[None]:
        """Acquire a cross-platform exclusive file lock adjacent to *target*."""
        lock_path = target.with_suffix(target.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = lock_path.open("a+", encoding="utf-8")
        locked_posix = False
        locked_win = False
        try:
            if _fcntl is not None:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
                locked_posix = True
            elif _msvcrt is not None:
                if fd.tell() == 0:
                    fd.write(" ")
                    fd.flush()
                fd.seek(0)
                _msvcrt.locking(fd.fileno(), _msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
                locked_win = True
            yield
        finally:
            if locked_posix and _fcntl is not None:
                with contextlib.suppress(OSError):
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
            elif locked_win and _msvcrt is not None:
                with contextlib.suppress(OSError):
                    fd.seek(0)
                    _msvcrt.locking(fd.fileno(), _msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            fd.close()
            # Best-effort cleanup; race with other processes is harmless
            with contextlib.suppress(OSError):
                lock_path.unlink()

    def invalidate(
        self,
        symbol: str | None = None,
        *,
        namespace: str | None = None,
    ) -> int:
        """Clear cached data."""
        removed = 0
        if symbol is None:
            target = self._namespace_root(namespace)
            if not target.exists():
                return 0
            for f in self._iter_cache_files(target):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
            return removed

        safe_symbol = _sanitize_token(symbol)
        candidate_dirs: list[Path] = []
        if namespace is not None:
            candidate_dirs.append(self._namespace_root(namespace) / safe_symbol)
        else:
            candidate_dirs.append(self.cache_dir / safe_symbol)
            if self.cache_dir.exists():
                for child in self.cache_dir.iterdir():
                    if child.is_dir():
                        nested = child / safe_symbol
                        if nested.is_dir():
                            candidate_dirs.append(nested)

        for target in candidate_dirs:
            if not target.exists():
                continue
            for f in self._iter_cache_files(target):
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
            if target.is_dir():
                with contextlib.suppress(OSError):
                    target.rmdir()

        return removed

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_files = 0
        total_size = 0
        if self.cache_dir.exists():
            for f in self._iter_cache_files(self.cache_dir):
                total_files += 1
                total_size += f.stat().st_size

        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_dir": str(self.cache_dir),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
        }

    @staticmethod
    def _iter_cache_files(root: Path) -> Iterator[Path]:
        """Iterate over supported cache file formats under *root*."""
        yield from root.rglob("*.parquet")
        yield from root.rglob("*.pkl")

    def _iter_symbol_cache_files(
        self,
        symbol: str,
        *,
        namespace: str | None = None,
    ) -> Iterator[Path]:
        """Iterate cache files for a single symbol namespace."""
        symbol_dir = self._cache_path(
            symbol,
            date.min,
            date.min,
            ".parquet",
            namespace=namespace,
        ).parent
        if not symbol_dir.exists():
            return
        yield from self._iter_cache_files(symbol_dir)

    @staticmethod
    def _parse_cache_interval(path: Path) -> tuple[date, date] | None:
        """Parse a cache filename back into its cached date interval."""
        parts = path.stem.split("_", 1)
        if len(parts) != 2:
            return None
        try:
            return (date.fromisoformat(parts[0]), date.fromisoformat(parts[1]))
        except ValueError:
            return None
