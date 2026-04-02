"""
Content-addressed caching system for the ML pipeline.

Prevents redundant computation by caching function results keyed on
SHA-256 hashes of (function_name + serialised parameters + data hash).
Supports TTL expiration, config-aware invalidation, multi-format
storage, and a decorator pattern for transparent integration.

Thread-safe via atomic writes (temp + rename) and file-based locking.

Usage
-----
    from ml.data.cache_manager import CacheManager

    cache = CacheManager(cache_dir="data/cache", config=loaded_config)

    @cache.cached(ttl_hours=24, depends_on=["garch"])
    def fit_garch(returns_df, config):
        ...

    # Manual get / set
    cache.set("my_key", df, ttl_hours=6, depends_on=["preprocessing"])
    result = cache.get("my_key")

    print(cache.stats())
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default TTLs (hours) by data category
# ---------------------------------------------------------------------------
DEFAULT_TTLS: Dict[str, float] = {
    "ohlcv": 1.0,
    "features": 6.0,
    "model": 24.0,
    "screening": 24.0,
    "backtest": 168.0,  # 7 days
}

_PICKLE_PROTOCOL = 5
_LOCK_TIMEOUT_SEC = 30
_LOCK_POLL_SEC = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return hex SHA-256 digest of a file, reading in 64 KiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_hash(config: Optional[dict]) -> str:
    """Deterministic hash of a config dict (sorted JSON)."""
    if config is None:
        return "no_config"
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return _sha256_bytes(raw)


def _config_section_hash(config: Optional[dict], key: str) -> str:
    """Hash of a single top-level config section (e.g. 'garch')."""
    if config is None or key not in config:
        return "missing"
    raw = json.dumps(config[key], sort_keys=True, default=str).encode()
    return _sha256_bytes(raw)


def _serialise_arg(obj: Any) -> str:
    """Best-effort deterministic string representation for hashing."""
    if isinstance(obj, pd.DataFrame):
        return f"df:shape={obj.shape}:hash={pd.util.hash_pandas_object(obj).sum()}"
    if isinstance(obj, pd.Series):
        return f"series:len={len(obj)}:hash={pd.util.hash_pandas_object(obj).sum()}"
    if isinstance(obj, np.ndarray):
        return f"np:shape={obj.shape}:hash={hashlib.sha256(obj.tobytes()).hexdigest()[:16]}"
    if isinstance(obj, dict):
        return json.dumps(obj, sort_keys=True, default=str)
    return str(obj)


class _FileLock:
    """
    Simple file-based lock using mkdir (atomic on all platforms).

    Avoids external dependencies while giving safe concurrent access.
    """

    def __init__(self, path: Path, timeout: float = _LOCK_TIMEOUT_SEC):
        self._lock_dir = path.with_suffix(".lock")
        self._timeout = timeout

    def acquire(self) -> None:
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                self._lock_dir.mkdir(parents=False, exist_ok=False)
                return
            except FileExistsError:
                if time.monotonic() > deadline:
                    # Stale lock -- force break
                    logger.warning("Lock timeout -- breaking stale lock %s", self._lock_dir)
                    shutil.rmtree(self._lock_dir, ignore_errors=True)
                    continue
                time.sleep(_LOCK_POLL_SEC)

    def release(self) -> None:
        shutil.rmtree(self._lock_dir, ignore_errors=True)

    def __enter__(self) -> "_FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------

class CacheManager:
    """Production-grade content-addressed cache for the ML pipeline."""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        config: Optional[dict] = None,
    ) -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._dir / "cache_manifest.json"
        self._lock = _FileLock(self._manifest_path)
        self._config = config
        self._config_hash = _config_hash(config)

        # Section-level hashes for fine-grained invalidation
        self._section_hashes: Dict[str, str] = {}
        if config:
            for key in config:
                self._section_hashes[key] = _config_section_hash(config, key)

        self._manifest = self._load_manifest()
        self._check_config_change()

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        """Load the manifest from disk, or return a fresh skeleton."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupt manifest -- starting fresh")
        return {
            "config_hash": self._config_hash,
            "section_hashes": dict(self._section_hashes),
            "entries": {},
            "stats": {
                "total_hits": 0,
                "total_misses": 0,
                "total_time_saved_sec": 0.0,
            },
        }

    def _save_manifest(self) -> None:
        """Atomically persist the manifest to disk."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._dir), suffix=".tmp", prefix=".manifest_"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(self._manifest, fh, indent=2, default=str)
            os.replace(tmp_path, str(self._manifest_path))
        except Exception:
            os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # Config-aware invalidation on init
    # ------------------------------------------------------------------

    def _check_config_change(self) -> None:
        """
        On startup, compare the stored config hash against the current one.
        If a specific config section changed, invalidate only entries that
        declared a dependency on that section.  If the overall hash differs
        but no per-section match is possible (legacy entries), invalidate
        everything that has a ``depends_on`` list.
        """
        stored_hash = self._manifest.get("config_hash", "")
        if stored_hash == self._config_hash:
            return  # nothing changed

        logger.info("Config change detected -- checking dependent caches")
        stored_sections: Dict[str, str] = self._manifest.get("section_hashes", {})
        changed_sections: set[str] = set()
        for key, new_hash in self._section_hashes.items():
            old_hash = stored_sections.get(key, "")
            if old_hash != new_hash:
                changed_sections.add(key)

        if changed_sections:
            logger.info("Changed config sections: %s", changed_sections)

        invalidated = 0
        to_remove: List[str] = []
        for key, entry in self._manifest["entries"].items():
            deps: List[str] = entry.get("depends_on") or []
            if not deps:
                continue  # no declared dependency -- leave untouched
            if changed_sections.intersection(deps):
                to_remove.append(key)
                invalidated += 1

        for key in to_remove:
            self._remove_entry(key)

        # Update stored hashes
        self._manifest["config_hash"] = self._config_hash
        self._manifest["section_hashes"] = dict(self._section_hashes)
        self._save_manifest()

        if invalidated:
            logger.info("Config change invalidated %d cache entries", invalidated)

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    def compute_key(self, func_name: str, *args: Any, **kwargs: Any) -> str:
        """
        Build a deterministic SHA-256 cache key from function identity
        and all arguments.
        """
        parts = [func_name]
        for a in args:
            parts.append(_serialise_arg(a))
        for k in sorted(kwargs):
            parts.append(f"{k}={_serialise_arg(kwargs[k])}")
        raw = "|".join(parts).encode()
        return _sha256_bytes(raw)

    # ------------------------------------------------------------------
    # Format detection + serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_format(value: Any) -> str:
        if isinstance(value, pd.DataFrame):
            return "parquet"
        if isinstance(value, np.ndarray):
            return "npy"
        if isinstance(value, (dict, list, int, float, str, bool)):
            # Only use JSON for plain JSON-serialisable types
            try:
                json.dumps(value, default=str)
                return "json"
            except (TypeError, ValueError):
                pass
        return "pkl"

    def _write_value(self, path: Path, value: Any, fmt: str) -> None:
        """Write *value* to a temp file and atomically rename to *path*."""
        # np.save appends .npy if the suffix is missing, so use the real
        # extension as the temp suffix to keep numpy (and parquet) happy.
        ext = self._extension(fmt)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._dir), suffix=ext, prefix=".cache_"
        )
        os.close(tmp_fd)
        try:
            if fmt == "parquet":
                value.to_parquet(tmp_path)
            elif fmt == "npy":
                np.save(tmp_path, value)
            elif fmt == "json":
                with open(tmp_path, "w") as fh:
                    json.dump(value, fh, default=str)
            else:
                with open(tmp_path, "wb") as fh:
                    pickle.dump(value, fh, protocol=_PICKLE_PROTOCOL)
            os.replace(tmp_path, str(path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def _read_value(path: Path, fmt: str) -> Any:
        if fmt == "parquet":
            return pd.read_parquet(path)
        if fmt == "npy":
            return np.load(path, allow_pickle=False)
        if fmt == "json":
            with open(path, "r") as fh:
                return json.load(fh)
        # pkl
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301

    @staticmethod
    def _extension(fmt: str) -> str:
        return {"parquet": ".parquet", "npy": ".npy", "json": ".json", "pkl": ".pkl"}[fmt]

    # ------------------------------------------------------------------
    # Entry helpers
    # ------------------------------------------------------------------

    def _remove_entry(self, key: str) -> None:
        """Delete a single cache entry (file + manifest record)."""
        entry = self._manifest["entries"].pop(key, None)
        if entry is None:
            return
        data_file = self._dir / entry["file"]
        if data_file.exists():
            data_file.unlink()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid(self, key: str) -> bool:
        """Return True if *key* exists and has not expired."""
        entry = self._manifest["entries"].get(key)
        if entry is None:
            return False
        created = datetime.fromisoformat(entry["created"])
        ttl_sec = entry["ttl_hours"] * 3600
        age = (datetime.now(timezone.utc) - created).total_seconds()
        if age > ttl_sec:
            return False
        data_file = self._dir / entry["file"]
        return data_file.exists()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a cached value by key.

        Returns *default* if not found or expired.
        """
        t0 = time.monotonic()
        if not self.is_valid(key):
            self._manifest["stats"]["total_misses"] += 1
            self._save_manifest()
            return default

        entry = self._manifest["entries"][key]
        data_file = self._dir / entry["file"]

        # Integrity check
        current_hash = _sha256_file(data_file)
        if current_hash != entry.get("data_hash", ""):
            logger.warning("Cache integrity failed for key %s -- discarding", key[:12])
            self._remove_entry(key)
            self._manifest["stats"]["total_misses"] += 1
            self._save_manifest()
            return default

        value = self._read_value(data_file, entry["format"])
        elapsed = time.monotonic() - t0

        entry["hit_count"] = entry.get("hit_count", 0) + 1
        self._manifest["stats"]["total_hits"] += 1
        # Estimate time saved: use the creation cost if recorded, else 1 s
        saved = entry.get("compute_sec", 1.0)
        self._manifest["stats"]["total_time_saved_sec"] += saved - elapsed
        self._save_manifest()

        logger.debug("Cache HIT  key=%s (%.3fs read, ~%.1fs saved)", key[:12], elapsed, saved)
        return value

    def set(
        self,
        key: str,
        value: Any,
        ttl_hours: Optional[float] = None,
        depends_on: Optional[List[str]] = None,
        compute_sec: float = 0.0,
    ) -> None:
        """
        Store *value* under *key* with optional TTL and config dependencies.

        Parameters
        ----------
        key : str
            Cache key (typically from ``compute_key``).
        value : Any
            Object to cache.
        ttl_hours : float, optional
            Time-to-live in hours.  Defaults to 24.
        depends_on : list[str], optional
            Config section names this entry depends on.
        compute_sec : float
            Wall-clock seconds the computation took (for stats).
        """
        if ttl_hours is None:
            ttl_hours = 24.0

        fmt = self._detect_format(value)
        filename = f"{key}{self._extension(fmt)}"
        data_path = self._dir / filename

        with self._lock:
            self._write_value(data_path, value, fmt)
            data_hash = _sha256_file(data_path)
            size_bytes = data_path.stat().st_size

            self._manifest["entries"][key] = {
                "file": filename,
                "format": fmt,
                "created": datetime.now(timezone.utc).isoformat(),
                "ttl_hours": ttl_hours,
                "depends_on": depends_on or [],
                "data_hash": data_hash,
                "size_bytes": size_bytes,
                "hit_count": 0,
                "compute_sec": compute_sec,
            }
            self._save_manifest()

        logger.debug(
            "Cache SET  key=%s fmt=%s size=%d ttl=%.1fh",
            key[:12], fmt, size_bytes, ttl_hours,
        )

    def cached(
        self,
        ttl_hours: float = 24.0,
        depends_on: Optional[List[str]] = None,
    ) -> Callable:
        """
        Decorator that transparently caches the return value of a function.

        The cache key is derived from the function name and all arguments.

        Example
        -------
        @cache_manager.cached(ttl_hours=24, depends_on=["garch"])
        def fit_garch(returns_df, config):
            ...
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                key = self.compute_key(func.__qualname__, *args, **kwargs)
                cached_val = self.get(key)
                if cached_val is not None:
                    logger.info("@cached HIT  %s -> %s", func.__qualname__, key[:12])
                    return cached_val

                logger.info("@cached MISS %s -> computing...", func.__qualname__)
                t0 = time.monotonic()
                result = func(*args, **kwargs)
                elapsed = time.monotonic() - t0

                self.set(
                    key,
                    result,
                    ttl_hours=ttl_hours,
                    depends_on=depends_on,
                    compute_sec=elapsed,
                )
                logger.info(
                    "@cached SET  %s (%.2fs) -> %s",
                    func.__qualname__, elapsed, key[:12],
                )
                return result
            return wrapper
        return decorator

    def invalidate(
        self,
        key: Optional[str] = None,
        config_key: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.

        Parameters
        ----------
        key : str, optional
            Exact cache key to remove.
        config_key : str, optional
            Remove all entries that depend on this config section.

        Returns
        -------
        int
            Number of entries invalidated.
        """
        count = 0
        with self._lock:
            if key is not None:
                if key in self._manifest["entries"]:
                    self._remove_entry(key)
                    count = 1
            elif config_key is not None:
                to_remove = [
                    k for k, v in self._manifest["entries"].items()
                    if config_key in (v.get("depends_on") or [])
                ]
                for k in to_remove:
                    self._remove_entry(k)
                count = len(to_remove)
            self._save_manifest()

        if count:
            logger.info("Invalidated %d cache entries", count)
        return count

    def clear_expired(self) -> int:
        """Remove all entries whose TTL has elapsed. Returns count removed."""
        now = datetime.now(timezone.utc)
        expired: List[str] = []
        for key, entry in self._manifest["entries"].items():
            created = datetime.fromisoformat(entry["created"])
            ttl_sec = entry["ttl_hours"] * 3600
            if (now - created).total_seconds() > ttl_sec:
                expired.append(key)

        with self._lock:
            for key in expired:
                self._remove_entry(key)
            self._save_manifest()

        if expired:
            logger.info("Cleared %d expired cache entries", len(expired))
        return len(expired)

    def clear_all(self) -> int:
        """Remove every cache entry. Returns count removed."""
        count = len(self._manifest["entries"])
        with self._lock:
            for key in list(self._manifest["entries"]):
                self._remove_entry(key)
            self._manifest["stats"] = {
                "total_hits": 0,
                "total_misses": 0,
                "total_time_saved_sec": 0.0,
            }
            self._save_manifest()

        logger.info("Cleared all %d cache entries", count)
        return count

    def stats(self) -> dict:
        """
        Return a summary of cache statistics.

        Returns
        -------
        dict
            Keys: total_entries, total_size_bytes, total_hits, total_misses,
            hit_rate, total_time_saved_sec, entries_by_format.
        """
        entries = self._manifest["entries"]
        total_size = sum(e.get("size_bytes", 0) for e in entries.values())
        fmt_counts: Dict[str, int] = {}
        for e in entries.values():
            f = e.get("format", "unknown")
            fmt_counts[f] = fmt_counts.get(f, 0) + 1

        s = self._manifest["stats"]
        total = s["total_hits"] + s["total_misses"]
        hit_rate = s["total_hits"] / total if total > 0 else 0.0

        return {
            "total_entries": len(entries),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_hits": s["total_hits"],
            "total_misses": s["total_misses"],
            "hit_rate": round(hit_rate, 4),
            "total_time_saved_sec": round(s["total_time_saved_sec"], 2),
            "entries_by_format": fmt_counts,
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"CacheManager(dir={self._dir}, entries={s['total_entries']}, "
            f"size={s['total_size_mb']}MB, hit_rate={s['hit_rate']:.1%})"
        )
