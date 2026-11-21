"""Immutable versioned OHLCV data bundles with SHA-256 tamper verification."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


class DataBundleSymbolSpec(BaseModel):
    """Per-symbol manifest entry inside an immutable OHLCV data bundle."""

    symbol: str
    rows: int
    start_date: str
    end_date: str
    ohlcv_sha256: str
    file: str


class DataBundleManifest(BaseModel):
    """Top-level manifest for one immutable, hash-verified market data bundle."""

    bundle_id: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    market: str = "us"
    adapter: str = "snapshot"
    date_range: tuple[str, str]
    symbol_count: int
    total_rows: int
    bundle_hash: str
    symbols: list[DataBundleSymbolSpec] = Field(default_factory=list)
    description: str = ""


class DataBundleVerificationResult(BaseModel):
    """Integrity verification verdict for a stored :class:`DataBundleManifest`."""

    bundle_id: str
    verified: bool
    expected_hash: str
    actual_hash: str
    checked_symbols: int
    errors: list[str] = Field(default_factory=list)


def _sanitize_slug(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip()).strip("._-")
    return cleaned or "bundle"


def _compute_bundle_hash(
    *,
    market: str,
    adapter: str,
    date_range: tuple[str, str],
    symbols: list[DataBundleSymbolSpec],
) -> str:
    payload = {
        "market": market,
        "adapter": adapter,
        "date_range": list(date_range),
        "symbols": [
            {
                "symbol": s.symbol,
                "rows": s.rows,
                "start_date": s.start_date,
                "end_date": s.end_date,
                "ohlcv_sha256": s.ohlcv_sha256,
            }
            for s in sorted(symbols, key=lambda item: item.symbol)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _serialize_normalized_ohlcv(df: pd.DataFrame) -> tuple[bytes, int, str, str]:
    import pandas as pd  # type: ignore[import-untyped]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError("OHLCV DataFrame must contain 'date' and 'close' columns")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.strftime("%Y-%m-%d")
    work = (
        work.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
    )
    if work.empty:
        raise ValueError("OHLCV DataFrame has no valid dated rows")

    for col in ("open", "high", "low"):
        if col not in work.columns:
            work[col] = work["close"]
    if "volume" not in work.columns:
        work["volume"] = 0.0

    lines = ["date,open,high,low,close,volume"]
    dates_col = work["date"].tolist()
    opens = work["open"].astype(float).tolist()
    highs = work["high"].astype(float).tolist()
    lows = work["low"].astype(float).tolist()
    closes = work["close"].astype(float).tolist()
    vols = work["volume"].astype(float).tolist()
    for d_s, o_v, h_v, l_v, c_v, v_v in zip(
        dates_col, opens, highs, lows, closes, vols, strict=True
    ):
        lines.append(f"{d_s},{o_v:.6f},{h_v:.6f},{l_v:.6f},{c_v:.6f},{v_v:.6f}")

    payload = ("\n".join(lines) + "\n").encode("utf-8")
    return payload, len(dates_col), str(dates_col[0]), str(dates_col[-1])


class DataBundleStore:
    """Append-only manager for immutable, SHA-256 verified OHLCV data bundles."""

    def __init__(self, bundle_root: str | Path = "reports/bundles") -> None:
        self._root = Path(bundle_root).expanduser()

    def create_bundle(
        self,
        market_data: dict[str, pd.DataFrame],
        *,
        name: str = "bundle",
        market: str = "us",
        adapter: str = "snapshot",
        description: str = "",
    ) -> DataBundleManifest:
        """Normalize OHLCV frames, write immutable CSVs + manifest, and return bundle metadata."""
        if not market_data:
            raise ValueError("Cannot create data bundle from empty market_data")

        prepared: list[tuple[str, bytes, int, str, str]] = []
        for symbol in sorted(market_data):
            df = market_data[symbol]
            if df is None or getattr(df, "empty", True):
                continue
            csv_bytes, rows, start_s, end_s = _serialize_normalized_ohlcv(df)
            prepared.append((symbol, csv_bytes, rows, start_s, end_s))

        if not prepared:
            raise ValueError("No non-empty valid symbols in market_data")

        all_starts = [item[3] for item in prepared]
        all_ends = [item[4] for item in prepared]
        date_range = (min(all_starts), max(all_ends))

        symbol_specs: list[DataBundleSymbolSpec] = []
        for symbol, csv_bytes, rows, start_s, end_s in prepared:
            digest = hashlib.sha256(csv_bytes).hexdigest()
            rel_file = f"ohlcv/{_sanitize_slug(symbol)}.csv"
            symbol_specs.append(
                DataBundleSymbolSpec(
                    symbol=symbol,
                    rows=rows,
                    start_date=start_s,
                    end_date=end_s,
                    ohlcv_sha256=digest,
                    file=rel_file,
                )
            )

        bundle_hash = _compute_bundle_hash(
            market=market,
            adapter=adapter,
            date_range=date_range,
            symbols=symbol_specs,
        )
        hash8 = bundle_hash.split(":", 1)[1][:8]
        slug = _sanitize_slug(name)
        base_id = (
            f"{slug}_{date_range[0].replace('-', '')}_{date_range[1].replace('-', '')}_{hash8}"
        )
        bundle_id = base_id
        bundle_dir = self._root / bundle_id
        if bundle_dir.exists():
            bundle_id = f"{base_id}_{uuid.uuid4().hex[:4]}"
            bundle_dir = self._root / bundle_id

        (bundle_dir / "ohlcv").mkdir(parents=True, exist_ok=False)
        for (symbol, csv_bytes, _rows, _s, _e), spec in zip(prepared, symbol_specs, strict=True):
            assert spec.symbol == symbol
            (bundle_dir / spec.file).write_bytes(csv_bytes)

        manifest = DataBundleManifest(
            bundle_id=bundle_id,
            name=name,
            market=market,
            adapter=adapter,
            date_range=date_range,
            symbol_count=len(symbol_specs),
            total_rows=sum(s.rows for s in symbol_specs),
            bundle_hash=bundle_hash,
            symbols=symbol_specs,
            description=description,
        )
        (bundle_dir / "bundle_manifest.json").write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return manifest

    def list_bundles(self) -> list[DataBundleManifest]:
        """Return all valid bundle manifests sorted newest first."""
        if not self._root.is_dir():
            return []
        manifests: list[DataBundleManifest] = []
        for child in sorted(self._root.iterdir()):
            manifest_file = child / "bundle_manifest.json"
            if manifest_file.is_file():
                manifests.append(
                    DataBundleManifest.model_validate_json(
                        manifest_file.read_text(encoding="utf-8")
                    )
                )
        return sorted(manifests, key=lambda m: m.created_at, reverse=True)

    def get_manifest(self, bundle_id: str) -> DataBundleManifest | None:
        """Return manifest for *bundle_id*, or None if missing."""
        manifest_file = self._root / bundle_id / "bundle_manifest.json"
        if not manifest_file.is_file():
            return None
        return DataBundleManifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))

    def verify_bundle(self, bundle_id: str) -> DataBundleVerificationResult:
        """Verify every CSV file and aggregate digest against ``bundle_manifest.json``."""
        bundle_dir = self._root / bundle_id
        manifest = self.get_manifest(bundle_id)
        if manifest is None:
            return DataBundleVerificationResult(
                bundle_id=bundle_id,
                verified=False,
                expected_hash="",
                actual_hash="",
                checked_symbols=0,
                errors=[f"Bundle manifest not found for '{bundle_id}'"],
            )

        errors: list[str] = []
        actual_specs: list[DataBundleSymbolSpec] = []
        for spec in manifest.symbols:
            csv_path = bundle_dir / spec.file
            if not csv_path.is_file():
                errors.append(f"Missing OHLCV file: {spec.file}")
                continue
            raw_bytes = csv_path.read_bytes().replace(b"\r\n", b"\n")
            actual_sha = hashlib.sha256(raw_bytes).hexdigest()
            if actual_sha != spec.ohlcv_sha256:
                errors.append(
                    f"Checksum mismatch for {spec.symbol}: expected {spec.ohlcv_sha256[:12]}, got {actual_sha[:12]}"
                )
            actual_specs.append(spec.model_copy(update={"ohlcv_sha256": actual_sha}))

        actual_hash = _compute_bundle_hash(
            market=manifest.market,
            adapter=manifest.adapter,
            date_range=manifest.date_range,
            symbols=actual_specs,
        )
        if actual_hash != manifest.bundle_hash:
            errors.append(
                f"Aggregate bundle hash mismatch: expected {manifest.bundle_hash}, got {actual_hash}"
            )

        return DataBundleVerificationResult(
            bundle_id=bundle_id,
            verified=len(errors) == 0,
            expected_hash=manifest.bundle_hash,
            actual_hash=actual_hash,
            checked_symbols=len(actual_specs),
            errors=errors,
        )

    def load_bundle(
        self,
        bundle_id: str,
        *,
        verify: bool = True,
    ) -> tuple[dict[str, pd.DataFrame], DataBundleManifest]:
        """Load bundle OHLCV frames into memory after verifying checksums."""
        import pandas as pd  # type: ignore[import-untyped]

        manifest = self.get_manifest(bundle_id)
        if manifest is None:
            raise FileNotFoundError(f"Data bundle not found: {bundle_id}")

        if verify:
            verdict = self.verify_bundle(bundle_id)
            if not verdict.verified:
                raise ValueError(
                    f"Data bundle integrity verification failed for '{bundle_id}': "
                    + "; ".join(verdict.errors)
                )

        bundle_dir = self._root / bundle_id
        data: dict[str, pd.DataFrame] = {}
        for spec in manifest.symbols:
            df = pd.read_csv(bundle_dir / spec.file)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = df[col].astype(float)
            data[spec.symbol] = cast("pd.DataFrame", df.reset_index(drop=True))
        return data, manifest


def bundle_date_bounds(manifest: DataBundleManifest) -> tuple[date, date]:
    """Parse ``manifest.date_range`` into ``datetime.date`` bounds."""
    return (
        date.fromisoformat(manifest.date_range[0]),
        date.fromisoformat(manifest.date_range[1]),
    )


def build_bundle_summary_dict(manifest: DataBundleManifest) -> dict[str, Any]:
    """Return lightweight bundle metadata suitable for config/extra JSON."""
    return {
        "bundle_id": manifest.bundle_id,
        "bundle_hash": manifest.bundle_hash,
        "symbol_count": manifest.symbol_count,
        "total_rows": manifest.total_rows,
        "date_range": list(manifest.date_range),
    }


__all__ = [
    "DataBundleManifest",
    "DataBundleStore",
    "DataBundleSymbolSpec",
    "DataBundleVerificationResult",
    "build_bundle_summary_dict",
    "bundle_date_bounds",
]
