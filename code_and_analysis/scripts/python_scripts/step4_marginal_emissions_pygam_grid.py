from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Standard Library
# ────────────────────────────────────────────────────────────────────────────
import binascii
import calendar
import hashlib
import inspect
import json
import logging
import math
import os


os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "1")
os.environ["OPENBLAS_NUM_THREADS"] = os.getenv("OPENBLAS_NUM_THREADS", "2")
os.environ["MKL_NUM_THREADS"] = "1"          # MKL (Intel)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # Accelerate (macOS)
os.environ["NUMEXPR_NUM_THREADS"] = "1"      # numexpr

import random
import re
import time
from joblib import Parallel, delayed, parallel_backend
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial, reduce, wraps
from itertools import combinations, product, chain
from multiprocessing import Lock, Manager, Pool, cpu_count
from multiprocessing.pool import ThreadPool
from operator import add
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional,
    Sequence, Tuple, Union
)
from zoneinfo import ZoneInfo
from joblib import Memory
from mpi4py import MPI


# ────────────────────────────────────────────────────────────────────────────
# Core Data Handling
# ────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import polars as pl

# ────────────────────────────────────────────────────────────────────────────
# Machine Learning & Statistics
# ────────────────────────────────────────────────────────────────────────────
from feature_engine.creation import CyclicalFeatures
from pygam import LinearGAM, l, s
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import kurtosis, skew, zscore

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LinearRegression, RidgeCV, HuberRegressor, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.utils.validation import check_is_fitted

# ────────────────────────────────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import seaborn as sns

# ────────────────────────────────────────────────────────────────────────────
# Geospatial
# ────────────────────────────────────────────────────────────────────────────
import geopandas as gpd
from shapely.geometry import Point, Polygon
from shapely.wkb import loads
from pyproj import Proj, transform


# Global configuration (paths, seeds, options) — adjust as needed
RANDOM_SEED = 12
try:
    import random; random.seed(RANDOM_SEED)
except Exception:
    pass
try:
    import numpy as np; np.random.seed(RANDOM_SEED)
except Exception:
    pass


from scipy.linalg import cholesky as _scipy_cholesky
from numpy.linalg import cholesky as _numpy_cholesky

import scipy.sparse as sp
# Add .A property back to the common sparse classes if missing
if not hasattr(sp.csr_matrix, "A"):
    sp.csr_matrix.A = property(lambda self: self.toarray())
if not hasattr(sp.csc_matrix, "A"):
    sp.csc_matrix.A = property(lambda self: self.toarray())
if not hasattr(sp.coo_matrix, "A"):
    sp.coo_matrix.A = property(lambda self: self.toarray())


# DIRECTORIES
base_directory = '..'
base_data_directory = os.path.join(base_directory, 'data')
hitachi_data_directory = os.path.join(base_data_directory, "hitachi_copy")
marginal_emissions_development_directory = os.path.join(base_data_directory, "marginal_emissions_development")
marginal_emissions_results_directory = os.path.join(marginal_emissions_development_directory, "results")
marginal_emissions_logs_directory = os.path.join(marginal_emissions_development_directory, "logs")

marginal_emissions_prefix = "marginal_emissions_results"

# FILENAMES
base_file = "weather_and_grid_data_half-hourly_20250714_1401"

train_file = "marginal_emissions_estimation_20250714_1401_train_data"
validation_file = "marginal_emissions_estimation_20250714_1401_validation_data"
test_file = "marginal_emissions_estimation_20250714_1401_test_data"


# FILEPATHS
base_filepath = os.path.join(hitachi_data_directory, base_file + ".parquet")

train_filepath = os.path.join(marginal_emissions_development_directory, train_file + ".parquet")
validation_filepath = os.path.join(marginal_emissions_development_directory, validation_file + ".parquet")
test_filepath = os.path.join(marginal_emissions_development_directory, test_file + ".parquet")


# Loading
base_pldf = pl.read_parquet(base_filepath)
train_pldf = pl.read_parquet(train_filepath)
validation_pldf = pl.read_parquet(validation_filepath)
test_pldf = pl.read_parquet(test_filepath)



### Utilities
#### General
def _sum_terms(terms):
    return reduce(add, terms)
def _file_size_mb(path: Path) -> float:
    """
    Return size of `path` in MiB. If file doesn't exist, returns 0.0.

    Parameters
    ----------
    path : Path
        Path to the file.

    Returns
    -------
    float
        Size of the file in MiB.
    """
    p = Path(path)
    if not p.exists():
        return 0.0
    return p.stat().st_size / (1024 * 1024.0)

#### CSV File Handling
def _drop_hash_from_part(
        part_path: Path,
        model_hash: str,
        *,
        chunk_size: int = 200_000,
        delete_if_empty: bool = False,
) -> int:
    """
    Remove rows with model_id_hash == `model_hash` from a CSV part file.

    - Streams in chunks (no huge memory spikes)
    - Writes to a temp file, then atomically replaces the original
    - Returns number of rows dropped
    - If all rows are dropped:
        • delete the file if `delete_if_empty=True`
        • otherwise keep a header-only CSV

    Parameters
    ----------
    part_path : Path
        CSV file to edit in place.
    model_hash : str
        Value to filter out from the 'model_id_hash' column.
    chunk_size : int, default 200_000
        Pandas read_csv chunk size.
    delete_if_empty : bool, default False
        If True and all rows are removed, delete the part file.

    Returns
    -------
    int
        Number of rows removed.
    """
    part_path = Path(part_path)
    if not part_path.exists():
        return 0

    # Quick header check
    try:
        header_df = pd.read_csv(part_path, nrows=0)
    except Exception:
        # Broken file — leave as-is
        return 0
    if "model_id_hash" not in header_df.columns:
        return 0

    dropped = 0
    kept = 0
    tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")

    # Ensure no stale tmp
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    first_write = True
    try:
        for chunk in pd.read_csv(
            part_path,
            chunksize=chunk_size,
            dtype={"model_id_hash": "string"},  # force string, avoid numeric coercion
        ):
            if "model_id_hash" not in chunk.columns:
                # schema changed mid-file? abort safely
                dropped = 0
                kept = -1
                break
            mask = chunk["model_id_hash"] != model_hash
            kept_chunk = chunk.loc[mask]
            n_dropped = int((~mask).sum())
            dropped += n_dropped
            kept += int(mask.sum())

            if kept_chunk.empty:
                continue

            kept_chunk.to_csv(
                tmp_path,
                index=False,
                mode="w" if first_write else "a",
                header=first_write,
            )
            first_write = False

        # Nothing matched → no change
        if dropped == 0:
            if tmp_path.exists():
                # wrote identical content; discard temp
                try: tmp_path.unlink()
                except Exception: pass
            return 0

        # All rows removed
        if kept == 0:
            if delete_if_empty:
                # Delete original; remove temp if created
                try: part_path.unlink()
                except Exception: pass
                if tmp_path.exists():
                    try: tmp_path.unlink()
                    except Exception: pass
            else:
                # Replace with header-only CSV
                header_df.to_csv(tmp_path, index=False)
                os.replace(tmp_path, part_path)
            return dropped

        # Normal case: replace atomically
        os.replace(tmp_path, part_path)
        return dropped

    finally:
        # Best-effort cleanup
        if tmp_path.exists():
            try: os.remove(tmp_path)
            except Exception: pass

def is_model_logged_rotating_csv(
        model_hash: str,
        base_dir: str | Path,
        file_prefix: str
) -> bool:
    """
    Return True if `model_hash` appears in the rolling-log index for `file_prefix`.

    Parameters
    ----------
    model_hash : str
        The 'model_id_hash' value to look up.
    base_dir : str | Path
        Directory holding the rolling CSV parts and index.
    file_prefix : str
        Prefix of the rolling log.

    Returns
    -------
    bool
        True if present in the index; False otherwise.
    """
    idx = _read_index(_index_path(Path(base_dir), file_prefix))
    if idx.empty or "model_id_hash" not in idx.columns:
        return False
    return str(model_hash) in idx["model_id_hash"].astype("string").values
def _list_part_files(
        base_dir: Path,
        file_prefix: str,
        ext: str = "csv",
) -> list[Path]:
    """
    List existing rolling CSV parts for a given prefix, sorted by numeric part index.

    Parameters
    ----------
    base_dir : Path
        Directory to search.
    file_prefix : str
        Prefix of the rolling CSV set (e.g., 'marginal_emissions_log').
    ext : str, default 'csv'
        File extension (without dot).

    Returns
    -------
    list[Path]
        Sorted list of matching part files, e.g. [.../prefix.part000.csv, .../prefix.part001.csv, ...]
    """
    if not base_dir.exists():
        return []

    rx = re.compile(rf"^{re.escape(file_prefix)}\.part(\d+)\.{re.escape(ext)}$")
    parts: list[tuple[int, Path]] = []

    for p in base_dir.glob(f"{file_prefix}.part*.{ext}"):
        if not p.is_file():
            continue
        m = rx.match(p.name)
        if m:
            parts.append((int(m.group(1)), p))

    parts.sort(key=lambda t: t[0])
    return [p for _, p in parts]

def load_all_logs_rotating_csv(
        results_dir: str | Path = ".",
        file_prefix: str = "marginal_emissions_log",
) -> pd.DataFrame:
    """
    Read only parts referenced by the index; drop duplicate hashes (keep last).

    Parameters
    ----------
    results_dir: str | Path
        The directory containing the results.
    file_prefix: str
        The prefix of the log files to read.

    Returns
    -------
    pd.DataFrame
        The concatenated DataFrame containing the logs.
    """
    # Read the index file
    base_dir = Path(results_dir)
    idx = _read_index(_index_path(base_dir, file_prefix))
    # Check if the index is empty
    if idx.empty:
        return pd.DataFrame()
    # Get the unique parts to read
    parts = idx["part_file"].unique().tolist()
    # Read the parts into DataFrames
    dfs = [pd.read_csv(p) for p in parts if Path(p).exists()]
    # Check if any DataFrames were read
    if not dfs:
        return pd.DataFrame()
    # Concatenate the DataFrames
    out = pd.concat(dfs, ignore_index=True)
    # Drop duplicate model_id_hash entries
    if "model_id_hash" in out.columns:
        out = out.drop_duplicates(subset=["model_id_hash"], keep="last")
    return out

def _read_index(index_path: Path) -> pd.DataFrame:
    """
    Read the rolling-log index CSV (id→part mapping).

    Parameters
    ----------
    index_path : Path
        Path to '<file_prefix>_index.csv'.

    Returns
    -------
    pd.DataFrame
        Columns ['model_id_hash','part_file'] or empty frame if not found/invalid.
    """
    try:
        idx = pd.read_csv(index_path, dtype={"model_id_hash": "string", "part_file": "string"})
        if not {"model_id_hash","part_file"}.issubset(idx.columns):
            raise ValueError("Index missing required columns.")
        return idx
    except FileNotFoundError:
        return pd.DataFrame(columns=["model_id_hash","part_file"])
    except Exception:
        # Be permissive but return the expected schema
        return pd.DataFrame(columns=["model_id_hash","part_file"])

def remove_model_from_rotating_csv(
        model_hash: str,
        results_dir: str | Path = ".",
        file_prefix: str = "marginal_emissions_log",
) -> None:
    """
    Remove all rows with `model_id_hash == model_hash` from the rolling CSV set.

    Parameters
    ----------
    model_hash : str
        Identifier to remove.
    results_dir : str | Path, default "."
        Directory holding parts and index.
    file_prefix : str, default "marginal_emissions_log"
        Prefix of the rolling log files.
    """
    base_dir = _ensure_dir(Path(results_dir))
    idx_path = _index_path(base_dir, file_prefix)

    # Lock the index for the whole operation to avoid races with concurrent writers/readers
    with _file_lock(_index_lock_path(idx_path)):
        idx = _read_index(idx_path)
        if idx.empty:
            return

        # Drop from referenced part files
        for pf in idx.loc[idx["model_id_hash"] == model_hash, "part_file"].dropna().unique():
            _drop_hash_from_part(Path(pf), model_hash)

        # Update index
        idx = idx[idx["model_id_hash"] != model_hash]
        idx.to_csv(idx_path, index=False)

def save_summary_to_rotating_csv(
        summary_df: pd.DataFrame,
        results_dir: str | Path = ".",
        file_prefix: str = "marginal_emissions_log",
        max_mb: int = 95,
        force_overwrite: bool = False,
        naming: PartNaming | None = None,
        fsync: bool = False,
) -> Path:
    """
    Append a single-row summary to a rolling CSV (<prefix>.partNNN.csv) with strict rotation:
    - Per-file lock during append (prevents interleaved writes/duplicate headers)
    - Under-lock preflight ensures the write will NOT push the file over `max_mb`
      (allocates a new shard if necessary)
    - Atomic index update under lock

    Parameters
    ----------
    summary_df : pd.DataFrame
        Single-row DataFrame with at least a 'model_id_hash' column.
    results_dir : str | Path, default "."
        Directory to write parts and the index into.
    file_prefix : str, default "marginal_emissions_log"
        Prefix of the part files ('<prefix>.partNNN.csv').
    max_mb : int, default 95
        Rotate when current part would exceed this size (MiB) after the append.
    force_overwrite : bool, default False
        If True, delete existing rows with the same hash before appending.
    naming : PartNaming, optional
        Naming convention (token/width/ext). If provided, `ext` should include the dot
        (e.g., ".csv"). Internally we use the extension without the dot for matching.
    fsync : bool, default False
        If True, call fsync() on the file after writing to ensure data is flushed to disk.

    Returns
    -------
    Path
        The part file path that received the append.

    Raises
    ------
    ValueError
        If `summary_df` is empty or missing 'model_id_hash'.
    """
    if summary_df.empty:
        raise ValueError("summary_df is empty.")
    if "model_id_hash" not in summary_df.columns:
        raise ValueError("summary_df must contain 'model_id_hash'.")
    if len(summary_df) != 1:
        summary_df = summary_df.iloc[:1].copy()

    naming = naming or PartNaming()
    base_dir = _ensure_dir(Path(results_dir))
    idx_path = _index_path(base_dir, file_prefix)
    model_hash = str(summary_df["model_id_hash"].iloc[0])
    ext_nodot = naming.ext.lstrip(".")

    # Optional overwrite: remove old rows (parts + index)
    if force_overwrite:
        remove_model_from_rotating_csv(model_hash, base_dir, file_prefix)
    else:
        if is_model_logged_rotating_csv(model_hash, base_dir, file_prefix):
            print(f"[SKIP] Hash already indexed: {model_hash}")
            parts = _list_part_files(base_dir, file_prefix, ext=ext_nodot)
            return parts[-1] if parts else base_dir / naming.format(file_prefix, 0)

    # Determine candidate shard
    parts = _list_part_files(base_dir, file_prefix, ext=ext_nodot)
    if parts:
        target = parts[-1]
    else:
        target = allocate_next_part(base_dir, file_prefix, width=naming.width, ext=ext_nodot)

    threshold_bytes = int(max_mb * 1024 * 1024)

    # --- LOCK AND WRITE TO SHARD SAFELY ---
    while True:
        shard_lock = Path(str(target) + ".lock")
        with _file_lock(shard_lock):
            current_size = Path(target).stat().st_size if Path(target).exists() else 0
            write_header = (current_size == 0)
            csv_payload = summary_df.to_csv(index=False, header=write_header)
            payload_bytes = len(csv_payload.encode("utf-8"))

            if current_size + payload_bytes > threshold_bytes:
                # rotate: leave lock, allocate new shard, try again
                pass
            else:
                with open(target, "a", encoding="utf-8", newline="") as f:
                    f.write(csv_payload)
                    f.flush()
                    if fsync:
                        os.fsync(f.fileno())
                break

        target = allocate_next_part(base_dir, file_prefix, width=naming.width, ext=ext_nodot)

    # --- LOCK AND UPDATE INDEX (atomic replace + optional fsync) ---
    lock_path = _index_lock_path(idx_path)
    with _file_lock(lock_path):
        idx = _read_index(idx_path)
        already = ("model_id_hash" in idx.columns) and (model_hash in idx["model_id_hash"].astype("string").values)
        if not already:
            idx = pd.concat(
                [idx, pd.DataFrame([{"model_id_hash": model_hash, "part_file": str(target)}])],
                ignore_index=True,
            )
            tmp_idx = idx_path.with_suffix(idx_path.suffix + ".tmp")
            with open(tmp_idx, "w", encoding="utf-8", newline="") as fh:
                idx.to_csv(fh, index=False)
                fh.flush()
                if fsync:
                    os.fsync(fh.fileno())
            os.replace(tmp_idx, idx_path)
            if fsync:
                # Ensure directory entry for index is durable
                dir_fd = os.open(str(idx_path.parent), os.O_DIRECTORY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)

    print(f"[SAVE] Appended to {target}, index updated.")
    return target
#### Path and Directory Management
def _ensure_dir(
        d: str | Path,
        *,
        resolve: bool = True
) -> Path:
    """
    Ensure directory `d` exists and return it as a Path.

    - Creates parent directories as needed.
    - Raises a clear error if a non-directory already exists at `d`.
    - Optionally returns the resolved (absolute) path.

    Parameters
    ----------
    d : str | Path
        Directory path to create if missing.
    resolve : bool, default True
        If True, return Path.resolve(strict=False) to normalize/absolutize.

    Returns
    -------
    Path
        The (optionally resolved) directory path.
    """
    p = Path(d)
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"Path exists and is not a directory: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve(strict=False) if resolve else p

def _index_path(
        base_dir: Path,
        file_prefix: str
) -> Path:
    """
    Build the path to the global index CSV for a given rolling log set.

    Parameters
    ----------
    base_dir : Path
        Directory that holds the rolling CSV parts.
    file_prefix : str
        Prefix used by the rolling CSV (e.g., 'marginal_emissions_log').

    Returns
    -------
    Path
        '<base_dir>/<file_prefix>_index.csv'
    """
    return Path(base_dir) / f"{file_prefix}_index.csv"

def _index_lock_path(index_path: Path) -> Path:
    """
    Derive the lock file path for an index CSV (same directory, '.lock' suffix).

    Parameters
    ----------
    index_path : Path
        Path to the index CSV file.

    Returns
    -------
    Path
        Path to the lock file.
    """
    return index_path.with_suffix(index_path.suffix + ".lock")
def _next_csv_part_path(
        base_dir: Path,
        file_prefix: str,
        width: int = 3,
        ext: str = "csv"
) -> Path:
    """
    Return the next available rotating-CSV part path.

    Scans for files named "<file_prefix>.partNNN.<ext>" in `base_dir`, where NNN is an
    integer with zero-padding. Picks max(N) and returns the next. If none exist, returns
    "...part000.<ext>" (or the padding width you pass).

    Parameters
    ----------
    base_dir : Path
        Directory to scan for part files.
    file_prefix : str
        Prefix used before ".partNNN.<ext>".
    width : int, default 3
        Minimum zero-padding width if no files exist yet.
    ext : str, default "csv"
        File extension (without dot).

    Returns
    -------
    Path
        Path for the next part file (not created).
    """
    if width < 1:
        raise ValueError("width must be >= 1")

    base_dir = Path(base_dir)
    pattern = re.compile(rf"^{re.escape(file_prefix)}\.part(\d+)\.{re.escape(ext)}$")

    max_n = -1
    pad = width

    for p in base_dir.glob(f"{file_prefix}.part*.{ext}"):
        m = pattern.match(p.name)
        if not m:
            continue
        n_str = m.group(1)
        pad = max(pad, len(n_str))
        try:
            n = int(n_str)
        except ValueError:
            continue
        if n > max_n:
            max_n = n

    next_n = max_n + 1
    n_str = f"{next_n:0{pad}d}"
    return base_dir / f"{file_prefix}.part{n_str}.{ext}"
def _roll_if_needed(
        path: Path,
        max_mb: int,
        *,
        naming: PartNaming | None = None
) -> Path:
    """
    If `path` exists and is >= max_mb, return the *next* part filename.
    Otherwise return `path` unchanged.

    Parameters
    ----------
    path : Path
        Current part file path (e.g., 'prefix.part007.csv').
    max_mb : int
        Rotation threshold in mebibytes (MiB).
    naming : PartNaming, optional
        Naming convention (token/width/ext). Uses defaults if not provided.

    Returns
    -------
    Path
        Either `path` or a new sibling with incremented part index.
    """
    if not path.exists() or _file_size_mb(path) < float(max_mb):
        return path
    naming = naming or PartNaming()
    stem, idx = naming.split(path.name)
    next_idx = (idx or 0) + 1
    return path.with_name(naming.format(stem=stem, idx=next_idx))

#### Logging
def load_existing_hashes(
        results_dir: str | Path,
        file_prefix: str,
) -> set[str]:
    """
    Get all unique `model_id_hash` values from the rolling-log index.

    Parameters
    ----------
    results_dir : str | Path
        Directory containing the rolling CSV parts and index.
    file_prefix : str
        Prefix of the rolling log files.

    Returns
    -------
    set[str]
        Unique model_id_hash values present in the index.
    """
    idx = _read_index(_index_path(Path(results_dir), file_prefix))
    if idx.empty or "model_id_hash" not in idx.columns:
        return set()
    # Ensure NA is dropped and cast to Python strings
    return set(idx["model_id_hash"].dropna().astype(str).tolist())

def make_config_key(
        config: Mapping[str, Any],
        algo: str = "sha256"
) -> str:
    """
    Create a deterministic hash key for a configuration mapping.

    Parameters
    ----------
    config : Mapping[str, Any]
        Configuration to serialize. Keys should be stringable.
    algo : {'sha256','md5','sha1',...}, default 'sha256'
        Hash algorithm name passed to hashlib.new.

    Returns
    -------
    str
        Hex digest of the normalized, JSON-serialized configuration.
    """
    def _norm(x):
        # Order/JSON-stable normalization.
        if isinstance(x, Mapping):
            # sort by key string to be robust to non-string keys
            return {str(k): _norm(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
        if isinstance(x, (list, tuple)):
            return [_norm(v) for v in x]
        if isinstance(x, set):
            # sets are unordered; sort normalized elements
            return sorted(_norm(v) for v in x)
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
        if isinstance(x, (datetime,)):
            return x.isoformat()
        return x  # strings, ints, floats, bools, None, etc.

    payload = json.dumps(
        _norm(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,   # last-resort for odd objects
    )
    h = hashlib.new(algo)
    h.update(payload.encode("utf-8"))
    return h.hexdigest()
def signature_for_run(
        user_pipeline: Pipeline,
        x_columns: list[str],
        y: pd.Series | pd.DataFrame,
        *,
        random_state: int,
        eval_splits: tuple[str, ...] = ("train", "validation"),
        compute_test: bool = False,
        extra_info: dict | None = None,
) -> tuple[str, dict]:
    """
    Build a stable config mapping for a model run and return (hash_key, mapping).

    This just standardizes what goes into the signature so different call sites
    don’t accidentally diverge.

    Parameters
    ----------
    user_pipeline : Pipeline
        The user-defined pipeline to run.
    x_columns : list[str]
        The feature columns to use for the model.
    y : pd.Series | pd.DataFrame
        The target variable(s) for the model.
    random_state : int
        The random seed to use for the model.
    eval_splits : tuple[str, ...], default=("train", "validation")
        The data splits to evaluate the model on.
    compute_test : bool, default=False
        Whether to compute metrics on the test split.
    extra_info : dict | None, default=None
        Any extra information to include in the signature.

    Returns
    -------
    tuple[str, dict]
        The hash key and the signature mapping.
    """
    sig = {
        "pipeline_params": user_pipeline.get_params(deep=True),
        "x_columns": list(x_columns),
        "y_columns": _y_columns_for_signature(y),
        "random_state": int(random_state),
        "eval_splits": tuple(eval_splits),
        "compute_test": bool(compute_test),
        **(extra_info or {}),
    }
    return make_config_key(sig), sig
def _y_columns_for_signature(y: pd.Series | pd.DataFrame) -> list[str]:
    """
    Normalize y to a list of column names for signature purposes.

    Parameters
    ----------
    y : pd.Series | pd.DataFrame
        The target variable(s) for the model.

    Returns
    -------
    list[str]
        The list of column names for the target variable(s).
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("y must be a Series or single-column DataFrame for signature.")
        return [str(y.columns[0])]
    name = getattr(y, "name", None)
    return [str(name)] if name is not None else ["y"]

#### MPI Management
def allocate_next_part(
        base_dir: Path,
        file_prefix: str,
        width: int = 3,
        ext: str = "csv",
        max_retries: int = 32,
        jitter_ms: tuple[int, int] = (1, 40),
) -> Path:
    """
    Atomically allocate the next rotating part file by creating it exclusively.

    Uses os.open(..., O_CREAT|O_EXCL) so only one process can create a given part.
    If another process wins the race, we re-scan and try the next part number.

    Parameters
    ----------
    base_dir : Path
        Directory to write part files into (created if missing).
    file_prefix : str
        Prefix used before ".partNNN.<ext>".
    width : int, default 3
        Minimum zero-padding for part numbers if none exist.
    ext : str, default "csv"
        Extension without dot.
    max_retries : int, default 32
        Maximum attempts before giving up.
    jitter_ms : (int, int), default (1, 40)
        Random backoff (min,max) milliseconds between retries.

    Returns
    -------
    Path
        The newly created, zero-length part file path (claimed for you).

    Raises
    ------
    RuntimeError
        If a unique part file cannot be allocated within max_retries.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for _ in range(max_retries):
        path = _next_csv_part_path(base_dir, file_prefix, width=width, ext=ext)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            fd = os.open(path, flags)  # atomic claim
            os.close(fd)               # leave it for normal open() later
            return path
        except FileExistsError:
            # Someone else grabbed it; small random backoff, then try again
            time.sleep(random.uniform(*jitter_ms) / 1000.0)
            continue

    raise RuntimeError("Failed to allocate a unique part file after many attempts")

def _distribute_configs(
        configs: list[dict],
        rank: int,
        size: int,
        mode: str = "stride"
) -> list[dict]:
    """
    Distribute configurations across multiple ranks.

    Parameters
    ----------
    configs: list[dict]
        The list of configurations to distribute.
    rank: int
        The rank of the current process.
    size: int
        The total number of processes.
    mode: str
        The distribution mode ("stride" or "chunked").

    Returns
    -------
    list[dict]
        The distributed list of configurations.
    """
    # Handle single process case
    if size <= 1:
        return configs
    # Handle multi-process case
    if mode == "stride":
        return configs[rank::size]
    # chunked
    n = len(configs)
    start = (n * rank) // size
    end   = (n * (rank + 1)) // size
    return configs[start:end]
@contextmanager
def _file_lock(
        lock_path: Path,
        max_wait_s: float = 30.0,
        jitter_ms: tuple[int,int]=(2,25)
) -> None:
    """
    Simple cross-process lock using O_CREAT|O_EXCL on a lockfile.

    Parameters
    ----------
    lock_path : Path
        Path to the lock file to create.
    max_wait_s : float, default 30.0
        Maximum time to wait for the lock before raising TimeoutError.
    jitter_ms : (int,int), default (2,25)
        Randomized backoff between retries, in milliseconds.

    Yields
    ------
    None
        The lock is held for the duration of the context.
    """
    # Create the lock file
    deadline = time.time() + float(max_wait_s)
    lock_path = Path(lock_path)
    last_err = None
    # Wait for the lock to be available
    while time.time() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            try:
                yield
            finally:
                try:
                    os.unlink(lock_path)
                except FileNotFoundError:
                    pass
            return
        except FileExistsError as e:
            last_err = e
            time.sleep(random.uniform(*jitter_ms) / 1000.0)
    raise TimeoutError(f"Could not acquire lock: {lock_path}") from last_err
def _mpi_context():
    """
    Get the MPI context for distributed training.

    Returns
    -------
    Tuple[COMM, int, int]
        The MPI communicator, rank, and size.
    """
    try:
        from mpi4py import MPI  # ensures import
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except Exception:
        class _Dummy:  # single-process stub
            def bcast(self, x, root=0): return x
            def Barrier(self): pass
        return _Dummy(), 0, 1
#### Naming Conventions
@dataclass(frozen=True)
class PartNaming:
    token: str = ".part"   # separator between stem and index
    width: int = 3         # zero-pad width
    ext: str = ".csv"      # file extension, with leading dot

    def format(self,
            stem: str,
            idx: int
    ) -> str:
        """
        Format a part filename.

        Parameters
        ----------
        stem : str
            The base name of the file (without extension or part token).
        idx : int
            The part index (zero-padded).

        Returns
        -------
        str
            The formatted part filename.
        """
        return f"{stem}{self.token}{idx:0{self.width}d}{self.ext}"

    def split(self,
            name: str
    ) -> Tuple[str, int | None]:
        """
        Split a part filename into its stem and index.

        Parameters
        ----------
        name : str
            The part filename to split.

        Returns
        -------
        Tuple[str, int | None]
            The stem and index of the part filename.
        """
        # returns (stem, idx) where idx is None if no part index present
        if not name.endswith(self.ext):
            # unknown extension; treat everything before first '.' as stem
            p = Path(name)
            return (p.stem, None)
        base = name[: -len(self.ext)]
        if self.token in base:
            stem, idx_str = base.split(self.token, 1)
            if idx_str.isdigit():
                return stem, int(idx_str)
        return base, None

#### Scoring & Metrics
def _compute_group_energy_weights(
        df: pd.DataFrame,
        group_col: str,
        q_col: str,
        interval_hours: float = 0.5,
) -> pd.DataFrame:
    """
    Aggregate energy per group, and multiples by the time step.
    Outputs group_col, q_sum and energy_MWh. Allows for energy-weighted metrics
    - so big load groups count more.

    Parameters
    ----------
    df : pd.DataFrame
        Rows for a single split after preprocessing (must contain `group_col` and `q_col`).
    group_col : str
        Name of the group id column (e.g., 'median_group_id', 'quantile_group_id').
    q_col : str
        Name of the demand/quantity column used as Q in the regression (usually x_vars[0]).
    interval_hours : float, default 0.5
        Duration represented by each row in hours (half-hourly = 0.5).

    Returns
    -------
    pd.DataFrame
        Columns: [group_col, 'q_sum', 'energy_MWh']
        where energy_MWh = q_sum * interval_hours.
    """
    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' not found in df")
    if q_col not in df.columns:
        raise KeyError(f"'{q_col}' not found in df")
    if not np.issubdtype(np.asarray(df[q_col]).dtype, np.number):
        raise TypeError(f"'{q_col}' must be numeric")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be > 0")

    g = (
        df.groupby(group_col, observed=True)[q_col]
          .sum()
          .rename("q_sum")
          .reset_index()
    )
    g["energy_MWh"] = g["q_sum"] * float(interval_hours)
    return g

def finite_difference_me_metrics(
        df: pd.DataFrame,
        time_col: str = "timestamp",
        q_col: str = "demand_met",
        y_col: str = "tons_co2",
        me_col: str = "ME",
        group_keys: list[str] | tuple[str, ...] = ("city",),
        max_dt: pd.Timedelta = pd.Timedelta("2h"),
        min_abs_dq: float = 1e-6,
) -> pd.DataFrame:
    """
    Compare predicted ME to observed short-horizon slopes s = Δy/ΔQ on held-out data.

    For each group in `group_keys`:
      Δy = y_t - y_{t-1}, ΔQ = Q_t - Q_{t-1}, Δt = t - t_{t-1}
      Keep pairs with Δt ≤ max_dt and |ΔQ| ≥ min_abs_dq.
      s_t = Δy / ΔQ, ME_avg = 0.5*(ME_t + ME_{t-1})

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data to analyze.
    time_col : str
        Name of the column containing the time information.
    q_col : str
        Name of the column containing the quantity information.
    y_col : str
        Name of the column containing the target variable (e.g., emissions).
    me_col : str
        Name of the column containing the marginal emissions estimates.
    group_keys : list[str] | tuple[str, ...]
        Columns to group by when computing metrics.
    max_dt : pd.Timedelta
        Maximum time difference to consider when computing slopes.
    min_abs_dq : float
        Minimum absolute change in quantity to consider when computing slopes.

    Returns
    -------
    pd.DataFrame
        One row per group and an optional pooled 'ALL' row:
        ['pearson_r','spearman_r','rmse','mae','n_pairs', *group_keys]
    """
    if time_col not in df.columns:
        raise KeyError(f"'{time_col}' not in df")
    # ensure datetime for Δt filtering
    dt_series = pd.to_datetime(df[time_col], errors="coerce")
    if dt_series.isna().any():
        raise ValueError(f"Column '{time_col}' contains non-parseable datetimes")
    work = df.copy()
    work[time_col] = dt_series

    def _per_group(gdf: pd.DataFrame) -> dict:
        gdf = gdf.sort_values(time_col).copy()
        gdf["dt"] = gdf[time_col].diff()
        gdf["dQ"] = gdf[q_col].diff()
        gdf["dY"] = gdf[y_col].diff()
        gdf["ME_avg"] = 0.5 * (gdf[me_col] + gdf[me_col].shift(1))

        mask = (
            gdf["dt"].notna() & (gdf["dt"] <= max_dt)
            & gdf["dQ"].notna() & (np.abs(gdf["dQ"]) >= float(min_abs_dq))
            & gdf["dY"].notna() & gdf["ME_avg"].notna()
        )
        sub = gdf.loc[mask, ["dY", "dQ", "ME_avg"]]
        if sub.empty:
            return {"pearson_r": np.nan, "spearman_r": np.nan, "rmse": np.nan, "mae": np.nan, "n_pairs": 0}

        s = sub["dY"].to_numpy(dtype=float) / sub["dQ"].to_numpy(dtype=float)
        me = sub["ME_avg"].to_numpy(dtype=float)
        return {
            "pearson_r": float(pd.Series(s).corr(pd.Series(me))),
            "spearman_r": float(pd.Series(s).corr(pd.Series(me), method="spearman")),
            "rmse": float(root_mean_squared_error(s, me)),
            "mae": float(mean_absolute_error(s, me)),
            "n_pairs": int(len(sub)),
        }

    parts: list[dict] = []
    if group_keys:
        for keys, gdf in work.groupby(list(group_keys), observed=True, sort=True):
            row = _per_group(gdf)
            if isinstance(keys, tuple):
                for kname, kval in zip(group_keys, keys):
                    row[kname] = kval
            else:
                row[group_keys[0]] = keys
            parts.append(row)
    else:
        parts.append(_per_group(work) | {"group": "ALL"})

    out = pd.DataFrame(parts)

    # pooled row
    if group_keys and (not out.empty) and out["n_pairs"].sum() > 0:
        tmp = []
        for _, gdf in work.groupby(list(group_keys), observed=True, sort=True):
            gdf = gdf.sort_values(time_col).copy()
            gdf["dt"] = gdf[time_col].diff()
            gdf["dQ"] = gdf[q_col].diff()
            gdf["dY"] = gdf[y_col].diff()
            gdf["ME_avg"] = 0.5 * (gdf[me_col] + gdf[me_col].shift(1))
            mask = (
                gdf["dt"].notna() & (gdf["dt"] <= max_dt)
                & gdf["dQ"].notna() & (np.abs(gdf["dQ"]) >= float(min_abs_dq))
                & gdf["dY"].notna() & gdf["ME_avg"].notna()
            )
            sub = gdf.loc[mask, ["dY", "dQ", "ME_avg"]]
            if not sub.empty:
                tmp.append(
                    pd.DataFrame({
                        "s": sub["dY"].to_numpy(dtype=float) / sub["dQ"].to_numpy(dtype=float),
                        "ME_avg": sub["ME_avg"].to_numpy(dtype=float),
                    })
                )
        if tmp:
            pooled = pd.concat(tmp, ignore_index=True)
            pooled_row = {
                "pearson_r": float(pooled["s"].corr(pooled["ME_avg"])),
                "spearman_r": float(pooled["s"].corr(pooled["ME_avg"], method="spearman")),
                "rmse": float(root_mean_squared_error(pooled["s"], pooled["ME_avg"])),
                "mae": float(mean_absolute_error(pooled["s"], pooled["ME_avg"])),
                "n_pairs": int(len(pooled)),
            }
            for k in group_keys:
                pooled_row[k] = "ALL"
            out = pd.concat([out, pd.DataFrame([pooled_row])], ignore_index=True)

    return out

def macro_micro_means(
        df: pd.DataFrame,
        metric: str,
        weight_col: str = "n_obs"
) -> dict:
    """
    Compute macro (simple mean) and micro (weighted by `weight_col`) for a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Per-group metrics.
    metric : str
        Column name to average.
    weight_col : str, default "n_obs"
        Column to use as weights for micro average.

    Returns
    -------
    dict
        {"macro": float, "micro": float}
    """
    macro = float(np.nanmean(df[metric].to_numpy(dtype=float)))
    if (weight_col in df) and np.nansum(df[weight_col].to_numpy(dtype=float)) > 0:
        micro = float(np.average(df[metric], weights=df[weight_col]))
    else:
        micro = np.nan
    return {"macro": macro, "micro": micro}

def mean_absolute_percentage_error(
        y_true,
        y_pred,
        eps: float = 1e-6
) -> float:
    """
    Compute MAPE robustly - adding small constant to avoid division by zero.

    MAPE = mean(|(y_true - y_pred) / (|y_true| + eps)|) * 100

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.
    eps : float, default 1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Mean absolute percentage error in percent.
    """
    # true values for y
    yt = np.asarray(y_true, dtype=float)
    # predicted values for y
    yp = np.asarray(y_pred, dtype=float)
    # denominator
    denom = np.abs(yt) + float(eps)
    # compute MAPE
    m = np.abs((yt - yp) / denom)
    # return as percentage (*100)
    return float(np.nanmean(m) * 100.0)

def mean_metric(
        df: pd.DataFrame,
        metric: str
) -> float:
    """
    Compute the mean of a metric, with a special case for MSE derived from RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metric columns.
    metric : {"r2","rmse","mae","mape","n_obs","mse"}
        Metric to aggregate.

    Returns
    -------
    float
        NaN-safe mean of the requested metric.
    """
    if metric == "mse":
        if "rmse" not in df:
            raise KeyError("Cannot compute 'mse': 'rmse' column missing.")
        return float(np.nanmean(df["rmse"].to_numpy(dtype=float) ** 2))
    if metric not in df:
        raise KeyError(f"Metric '{metric}' not found in DataFrame.")
    return float(np.nanmean(df[metric].to_numpy(dtype=float)))

def pooled_co2_metrics(
        regressor,                  # fitted GroupwiseRegressor
        transformed_df: pd.DataFrame,
        y_col: str | None = None,
        group_col: str | None = None,
) -> dict:
    """
    Compute pooled (all bins together) out-of-sample metrics for CO2.

    Parameters
    ----------
    regressor : GroupwiseRegressor
        Must be fitted; `regressor.group_models_` is used per group.
    transformed_df : pd.DataFrame
        Contains features used by the regressor, the group column, and the true y.
        (Typically validation/test X after feature+binner, with y added).
    y_col : str, optional
        Target column name. Defaults to regressor.y_var.
    group_col : str, optional
        Group column name. Defaults to regressor.group_col.

    Returns
    -------
    dict
        {'r2','rmse','mae','mape','n_obs'} (NaNs if insufficient data).
    """
    y_col = y_col or regressor.y_var
    group_col = group_col or regressor.group_col
    if y_col not in transformed_df.columns:
        raise KeyError(f"'{y_col}' not found in transformed_df")
    if group_col not in transformed_df.columns:
        raise KeyError(f"'{group_col}' not found in transformed_df")

    preds = pd.Series(index=transformed_df.index, dtype=float)
    for g, gdf in transformed_df.groupby(group_col, sort=True):
        model = regressor.group_models_.get(g)
        if model is None:
            continue
        preds.loc[gdf.index] = model.predict(gdf)

    mask = preds.notna()
    n_obs = int(mask.sum())
    if n_obs == 0:
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "mape": np.nan, "n_obs": 0}

    y_true = transformed_df.loc[mask, y_col].to_numpy(dtype=float)
    y_pred = preds.loc[mask].to_numpy(dtype=float)

    # r2 can error for <2 samples or constant y
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    return {
        "r2": r2,
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "n_obs": n_obs,
    }
def summarise_metrics_logs(
        train_logs: pd.DataFrame,
        val_logs: pd.DataFrame,
        test_logs: pd.DataFrame | None = None,
        user_pipeline: Pipeline = None,
        x_columns: list | None = None,
        random_state: int = 12,
        group_col_name: str = "group",
        pooled_metrics_by_split: dict[str, dict] | None = None,
        fd_me_metrics_by_split: dict[str, dict] | None = None,
        energy_weight_col: str = "energy_MWh",
) -> pd.DataFrame:
    """
    Summarise per-split, per-group metrics and pipeline metadata into a single-row DataFrame.

    This variant allows `test_logs` to be None (can skip test during tuning).

    Parameters
    ----------
    train_logs, val_logs : pd.DataFrame
        Metrics frames for train/validation.
    test_logs : pd.DataFrame or None, default None
        Test metrics; if None, test columns are omitted from the summary.
    user_pipeline : Pipeline
        The fitted or configured pipeline (used for metadata).
    x_columns : list, optional
        Feature names used by the model.
    random_state : int, default 12
        Random seed to record.
    group_col_name : str, default "group"
        Canonical name for the group column.
    pooled_metrics_by_split, fd_me_metrics_by_split : dict, optional
        Optional extra diagnostics keyed by split.
    energy_weight_col : str, default "energy_MWh"
        Column name to use for energy-weighted micro-averages if present.

    Returns
    -------
    pd.DataFrame
        One-row summary. Only includes split columns for the splits provided.
    """
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        cols = list(df.columns)

        # If desired already present, use it
        if group_col_name in cols:
            return df

        # If a plain 'group' exists, rename it to the desired name
        if "group" in cols:
            return df.rename(columns={"group": group_col_name})

        # Known aliases we can rename from
        candidates = [
            "multi_group_id",
            "quantile_group_id",
            "median_group_id",
            "original_quantile_group_id",
            "group_id",
        ]

        # Any *_group_id pattern
        pattern_hits = [c for c in cols if c.endswith("_group_id")]

        # Prefer known aliases in order
        for c in candidates:
            if c in cols:
                return df.rename(columns={c: group_col_name})

        # If exactly one *_group_id exists, use it
        if len(pattern_hits) == 1:
            return df.rename(columns={pattern_hits[0]: group_col_name})

        # Nothing we recognize → fail loudly with context
        raise KeyError(
            f"Could not locate a group column; expected '{group_col_name}' or any of "
            f"{[c for c in candidates if c in cols] + (['group'] if 'group' in cols else []) or candidates + ['group']}. "
            f"Available columns: {cols}"
        )
    splits: dict[str, pd.DataFrame] = {
        "train": _norm(train_logs.copy()),
        "validation": _norm(val_logs.copy()),
    }
    if test_logs is not None:
        splits["test"] = _norm(test_logs.copy())

    required = {"r2", "rmse", "mae", "mape", "n_obs"}
    for name, df in splits.items():
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name} logs missing metrics: {sorted(missing)}")

    first = next(iter(splits.values()))
    model_id = first.get("model_id_hash", pd.Series([np.nan])).iloc[0]
    log_time = first.get("log_time", pd.Series([np.nan])).iloc[0]
    model_name = user_pipeline._final_estimator.__class__.__name__ if user_pipeline is not None else ""
    pipeline_steps = list(user_pipeline.named_steps.keys()) if user_pipeline is not None else []

    summary: dict[str, Any] = {
        "model_id_hash": model_id,
        "random_state": random_state,
        "params_json": json.dumps(
            user_pipeline.get_params(deep=True), sort_keys=True, separators=(",", ":"), default=str
        ) if user_pipeline is not None else "{}",
        "log_time": log_time,
        "model_name": model_name,
        "pipeline_steps": pipeline_steps,
        "pipeline_n_steps": len(pipeline_steps),
        "x_columns": x_columns or [],
        "metrics_by_group": {},
    }

    nested: dict[str, dict] = {}
    for split, df in splits.items():
        # macro means
        summary[f"r2_{split}"] = float(df["r2"].mean())
        summary[f"rmse_{split}"] = float(df["rmse"].mean())
        summary[f"mae_{split}"] = float(df["mae"].mean())
        summary[f"mape_{split}"] = float(df["mape"].mean())
        # counts should be sums, not means
        summary[f"n_obs_{split}"] = int(df["n_obs"].sum())
        summary[f"mse_{split}"] = float((df["rmse"] ** 2).mean())

        # micro by n_obs
        if df["n_obs"].sum() > 0:
            w = df["n_obs"].to_numpy(dtype=float)
            summary[f"r2_{split}_micro"] = float(np.average(df["r2"], weights=w))
            summary[f"rmse_{split}_micro"] = float(np.average(df["rmse"], weights=w))
            summary[f"mae_{split}_micro"] = float(np.average(df["mae"], weights=w))
            summary[f"mape_{split}_micro"] = float(np.average(df["mape"], weights=w))
        else:
            summary[f"r2_{split}_micro"] = np.nan
            summary[f"rmse_{split}_micro"] = np.nan
            summary[f"mae_{split}_micro"] = np.nan
            summary[f"mape_{split}_micro"] = np.nan

        # energy-weighted micro (if provided)
        if (energy_weight_col in df.columns) and (df[energy_weight_col].fillna(0).sum() > 0):
            wE = df[energy_weight_col].fillna(0).to_numpy(dtype=float)
            summary[f"r2_{split}_energy_micro"] = float(np.average(df["r2"], weights=wE))
            summary[f"rmse_{split}_energy_micro"] = float(np.average(df["rmse"], weights=wE))
            summary[f"mae_{split}_energy_micro"] = float(np.average(df["mae"], weights=wE))
            summary[f"mape_{split}_energy_micro"] = float(np.average(df["mape"], weights=wE))
            summary[f"{energy_weight_col}_{split}_total"] = float(wE.sum())
        else:
            summary[f"r2_{split}_energy_micro"] = np.nan
            summary[f"rmse_{split}_energy_micro"] = np.nan
            summary[f"mae_{split}_energy_micro"] = np.nan
            summary[f"mape_{split}_energy_micro"] = np.nan
            summary[f"{energy_weight_col}_{split}_total"] = 0.0

        cols = ["r2", "rmse", "mae", "mape", "n_obs"]
        if energy_weight_col in df.columns:
            cols.append(energy_weight_col)
        nested[split] = df.set_index(group_col_name)[cols].to_dict(orient="index")

    summary["metrics_by_group"] = nested

    pooled_metrics_by_split = pooled_metrics_by_split or {}
    fd_me_metrics_by_split = fd_me_metrics_by_split or {}
    for split in splits.keys():
        summary[f"pooled_co2_{split}"] = json.dumps(pooled_metrics_by_split.get(split, {}))
        summary[f"fd_me_{split}"] = json.dumps(fd_me_metrics_by_split.get(split, {}))

    return pd.DataFrame([summary])



#### New Development
##### Helpers
def _report_added(
        before_cols: Iterable[str],
        after_cols: Iterable[str]
) -> Tuple[int, Tuple[str, ...]]:
    """
    Helper to report how many and which columns were added.

    Parameters
    ----------
    before_cols : Iterable[str]
        The columns present before the change.
    after_cols : Iterable[str]
        The columns present after the change.

    Returns:
    -------
    Tuple[int, Tuple[str, ...]]
        The number and names of the added columns.
    """
    before, after = set(before_cols), set(after_cols)
    added = tuple(sorted(after - before))
    return len(added), added
def _require_columns(
        X: pd.DataFrame,
        cols: Iterable[str],
        who: str
) -> None:
    """
    Ensure that the required columns are present in the DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        The DataFrame to check.
    cols : Iterable[str]
        The columns to check for.
    who : str
        The name of the caller, used for error messages.

    Returns
    -------
    None
        Raises KeyError if any required columns are missing.
    """
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"{who}: missing columns: {missing}")

class VerboseMixin:
    """
    Any class that wants to log verbose messages should inherit from this mixin.
    """
    def __init__(self, verbose: bool = False):
        self.verbose = bool(verbose)

    def _log(self, msg: str):
        if getattr(self, "verbose", False):
            print(f"[{self.__class__.__name__}] {msg}")
##### Feature Transformation
class StandardizeContinuous(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Add standardized copies of columns. Choose 'standard' (mean/std) or 'robust' (median/IQR).
    Keeps originals; writes <col><suffix>. Set drop_original=True to replace (rarely needed).
    """
    def __init__(self,
                 columns = None,        # : Sequence[str],
                 suffix: str = "_std",
                 strategy: str = "standard",   # "standard" | "robust"
                 with_center: bool = True,
                 with_scale: bool = True,
                 drop_original: bool = False,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.columns = columns      #list(columns)
        self.suffix = suffix
        self.strategy = strategy
        self.with_center = with_center
        self.with_scale = with_scale
        self.drop_original = drop_original
        self.stats_ = None        # dict: {col: (center, scale)}
        self._columns_ = None     # tuple of column names used

    def fit(self, X, y=None):
        self._columns_ = tuple(self.columns)
        _require_columns(X, cols=list(self._columns_), who="StandardizeContinuous.fit")

        strategy = str(self.strategy).lower()
        if strategy not in {"standard", "robust"}:
            raise ValueError(f"strategy must be 'standard' or 'robust', got {self.strategy!r}")

        self.stats_ = {}
        for c in self._columns_:
            v = pd.to_numeric(X[c], errors="coerce").astype(float)
            if strategy == "robust":
                center = float(v.median()) if self.with_center else 0.0
                scale = float((v.quantile(0.75) - v.quantile(0.25))) if self.with_scale else 1.0
            else:
                center = float(v.mean()) if self.with_center else 0.0
                scale = float(v.std(ddof=0)) if self.with_scale else 1.0
            if scale == 0.0 or not np.isfinite(scale):
                scale = 1.0
            self.stats_[c] = (center, scale)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.stats_ is None or self._columns_ is None:
            raise RuntimeError("StandardizeContinuous not fitted. Call .fit(X) first.")
        df = X.copy()
        before = df.columns
        for c in self._columns_:
            mu, sd = self.stats_[c]
            z = (pd.to_numeric(df[c], errors="coerce").astype(float) - mu) / sd
            df[f"{c}{self.suffix}"] = z
            if self.drop_original:
                df.drop(columns=[c], inplace=True)
        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} columns: {cols}")
        return df

    # convenience
    def inverse_transform_column(self, z: pd.Series, col: str) -> pd.Series:
        if self.stats_ is None:
            raise RuntimeError("Not fitted")
        mu, sd = self.stats_[col]
        return z * sd + mu

    def get_feature_names_out(self, input_features=None):
        if self._columns_ is None:
            return np.array([])
        return np.array([f"{c}{self.suffix}" for c in self._columns_])
class Log1pTransform(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Create <col>_log1p = log1p(col_clamped).
    """
    def __init__(self,
                 columns: None,      # Sequence[str],
                 out_suffix: str = "_log1p",
                 clamp_lower =0.0,  # None = no clamp Optional[float]
                 drop_original: bool = False,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.columns = columns  # list(columns)
        self.out_suffix = out_suffix
        self.clamp_lower = clamp_lower
        self.drop_original = drop_original

    def fit(self, X, y=None):
        _require_columns(X, self.columns, "Log1pTransform.fit")
        self._columns_ = tuple(self.columns)
        self._clamp_lower_ = float(self.clamp_lower) if self.clamp_lower is not None else None
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        before = df.columns
        lo = self._clamp_lower_
        for c in self._columns_:
            s = pd.to_numeric(df[c], errors="coerce").astype(float)
            if lo is not None:
                s = s.clip(lower=lo)
            df[f"{c}{self.out_suffix}"] = np.log1p(s)
            if self.drop_original:
                df.drop(columns=[c], inplace=True)
        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} columns: {cols}")
        return df

class ZeroInflatedLogTransform(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    For zero-inflated vars (e.g., precipitation):
      - <col>_occ  : 1{col > threshold}
      - <col>_log1p: log1p(col) for col>threshold, else 0
    Keeps original by default.
    """
    def __init__(self,
                 columns = None,    # Sequence[str],
                 threshold = 0.0,
                 occ_suffix: str = "_occ",
                 log_suffix: str = "_log1p",
                 drop_original: bool = False,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.columns = columns      # list(columns)
        self.threshold = threshold      # float(threshold)
        self.occ_suffix = occ_suffix
        self.log_suffix = log_suffix
        self.drop_original = drop_original

    def fit(self, X, y=None):
        _require_columns(X, self.columns, "ZeroInflatedLogTransform.fit")
        self._columns_ = tuple(self.columns)
        self._thr_ = float(self.threshold)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        before = df.columns
        thr = self._thr_
        for c in self.columns:
            v = pd.to_numeric(df[c], errors="coerce").astype(float)
            occ = (v > thr).astype(np.int8)
            logp = np.where(occ == 1, np.log1p(v), 0.0)
            df[f"{c}{self.occ_suffix}"] = occ
            df[f"{c}{self.log_suffix}"] = logp
            if self.drop_original:
                df.drop(columns=[c], inplace=True)
        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} columns: {cols}")
        return df

class DateTimeFeatureAdder(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Adds discrete time parts: year, month, day, hour, minute, dow, doy, week, weekend flag.
    """
    def __init__(self,
                 timestamp_col: str = "timestamp",
                 add=None,      # Sequence[str] = ("month","hour","dow","doy","is_weekend")
                 drop_original: bool = False,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.timestamp_col = timestamp_col
        self.add = add
        self.drop_original = drop_original
        self.verbose = verbose


    def fit(self, X, y=None):
        _require_columns(X, cols=[self.timestamp_col], who="DateTimeFeatureAdder.fit")
        default = ("month","hour","dow","doy","is_weekend")
        self._add = tuple(self.add) if self.add is not None else default
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        t = pd.to_datetime(df[self.timestamp_col], errors="raise")
        before = df.columns

        add = getattr(self, "_add", tuple(self.add) if self.add is not None else ("month","hour","dow","doy","is_weekend"))

        if "year" in add:   df["year"] = t.dt.year.astype("Int64")
        if "month" in add:  df["month"] = t.dt.month.astype("Int64")
        if "day" in add:    df["day"] = t.dt.day.astype("Int64")
        if "hour" in add:   df["hour"] = t.dt.hour.astype("Int64")
        if "minute" in add: df["minute"] = t.dt.minute.astype("Int64")
        if "dow" in add:    df["dow"] = t.dt.dayofweek.astype("Int64") # 0=Mon
        if "doy" in add:    df["doy"] = t.dt.dayofyear.astype("Int64")
        if "week" in add:   df["week"] = t.dt.isocalendar().week.astype("Int64")
        if "is_weekend" in add: df["is_weekend"] = (t.dt.dayofweek >= 5).astype("int8")
        if self.drop_original:
            df.drop(columns=[self.timestamp_col], inplace=True)
        # n_add, cols = _report_added(before, df.columns)
        # self._log(f"added {n_add} columns: {cols}")
        if self.verbose:
            print(f"[DateTimeFeatureAdder] added: {add}")
        return df
class OneHotTimeFE(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    One-hot encode discrete time parts (or any discrete columns you pass).
    Stores training dummies and aligns at transform time.
    """
    def __init__(self, columns: Sequence[str] = ("month", "hour"),
                 drop_first: bool = True, dtype: str = "int8", verbose: bool = False):
        super().__init__(verbose=verbose)
        self.columns = list(columns)
        self.drop_first = drop_first
        self.dtype = dtype
        self.out_cols_: list[str] = []

    def fit(self, X, y=None):
        _require_columns(X, self.columns, "OneHotTimeFE.fit")
        d = pd.get_dummies(
            X[self.columns].astype("Int64"),
            columns=self.columns,
            drop_first=self.drop_first,
            dtype=self.dtype
        )
        self.out_cols_ = list(d.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        before = df.columns
        d = pd.get_dummies(
            df[self.columns].astype("Int64"),
            columns=self.columns,
            drop_first=self.drop_first,
            dtype=self.dtype
        )
        # align to training
        for c in self.out_cols_:
            if c not in d.columns:
                d[c] = 0
        d = d[self.out_cols_]
        df = pd.concat([df, d], axis=1)
        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} dummies: {cols}")
        return df

    def get_feature_names_out(self, input_features=None):
        base = [] if input_features is None else list(input_features)
        return np.array(base + self.out_cols_)

class TimeFourierAdder(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Smooth cyclic time features: hour_sin/cos, dow_sin/cos, doy_sin/cos (+ is_weekend).
    """
    def __init__(self, timestamp_col="timestamp",
                 add_month=True, add_week=True, add_doy=True,
                 add_hour=True, add_weekend=True,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.timestamp_col = timestamp_col
        self.add_month = add_month
        self.add_week = add_week
        self.add_doy = add_doy
        self.add_hour = add_hour
        self.add_weekend = add_weekend

    def fit(self, X, y=None):
        _require_columns(X, [self.timestamp_col], "TimeFourierAdder.fit")
        return self

    def transform(self, X):
        df = X.copy()
        t = pd.to_datetime(df[self.timestamp_col], errors="raise")
        before = df.columns

        if self.add_hour:
            h = t.dt.hour + t.dt.minute / 60.0
            df["hour_sin"] = np.sin(2*np.pi*h/24.0)
            df["hour_cos"] = np.cos(2*np.pi*h/24.0)
        if self.add_week:
            dow = t.dt.dayofweek.astype(float)  # 0..6
            df["dow_sin"] = np.sin(2*np.pi*dow/7.0)
            df["dow_cos"] = np.cos(2*np.pi*dow/7.0)
        if self.add_doy:
            doy = t.dt.dayofyear.astype(float)
            df["doy_sin"] = np.sin(2*np.pi*doy/365.0)
            df["doy_cos"] = np.cos(2*np.pi*doy/365.0)
        if self.add_weekend:
            df["is_weekend"] = (t.dt.dayofweek >= 5).astype("int8")
        if self.add_month:
            # fractional month index (0..12) centered within month to reduce boundary jumps
            frac = (t.dt.day - 0.5) / t.dt.days_in_month
            m = (t.dt.month - 1 + frac).astype(float)  # 0..12
            df["month_sin"] = np.sin(2*np.pi*m/12.0)
            df["month_cos"] = np.cos(2*np.pi*m/12.0)

        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} columns: {cols}")
        return df

class WindDirToCyclic(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Encode wind direction as sin/cos. Supports meteorological 'from' degrees.
    """
    def __init__(self,
                 dir_col: str = "wind_direction_meteorological",
                 out_sin: str = "wind_dir_sin",
                 out_cos: str = "wind_dir_cos",
                 convention: str = "met",     # "met" (from-degree) or "math" (0 at +x, CCW)
                 drop_original: bool = True,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.dir_col = dir_col
        self.out_sin = out_sin
        self.out_cos = out_cos
        self.convention = convention
        self.drop_original = drop_original

    def fit(self, X, y=None):
        _require_columns(X, [self.dir_col], "WindDirToCyclic.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        before = df.columns
        deg = df[self.dir_col].astype(float)
        if self.convention.lower().startswith("met"):
            # convert meteorological 'from' to math radians
            theta = np.deg2rad(270.0 - deg)  # 0° = East, CCW positive
        else:
            theta = np.deg2rad(deg)
        df[self.out_sin] = np.sin(theta)
        df[self.out_cos] = np.cos(theta)
        if self.drop_original:
            df.drop(columns=[self.dir_col], inplace=True)
        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} columns: {cols}")
        return df

    def get_feature_names_out(self, input_features=None):
        base = [] if input_features is None else list(input_features)
        if self.drop_original and self.dir_col in base:
            base.remove(self.dir_col)
        return np.array(base + [self.out_sin, self.out_cos])

class Winsorize(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Clip columns to percentile bounds. Default: [0th, 99.5th].
    """
    def __init__(self, columns: Sequence[str],
                 lower: float = 0.0,
                 upper: float = 0.995,
                 per_column_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.columns = list(columns)
        self.lower = float(lower)
        self.upper = float(upper)
        self.per_column_bounds = per_column_bounds
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X, y=None):
        _require_columns(X, self.columns, "Winsorize.fit")
        self.bounds_.clear()
        for c in self.columns:
            if self.per_column_bounds and c in self.per_column_bounds:
                lo, hi = self.per_column_bounds[c]
            else:
                arr = X[c].astype(float).to_numpy()
                lo = np.nanpercentile(arr, self.lower * 100.0)
                hi = np.nanpercentile(arr, self.upper * 100.0)
            self.bounds_[c] = (float(lo), float(hi))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            df[c] = df[c].clip(lo, hi)
        self._log(f"clipped {len(self.bounds_)} columns to learned bounds")
        return df
class BoundedToLogit(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Clamp [0,1] variables to [eps,1-eps] and create <col>_logit = log(p/(1-p)).
    Keeps original by default.
    """
    def __init__(self,
                 columns= None,    # Sequence[str],
                 eps = 1e-6,        # : float
                 out_suffix: str = "_logit",
                 drop_original: bool = False,
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.columns = columns      #list(columns)
        self.eps = eps      # float()
        self.out_suffix = out_suffix
        self.drop_original = drop_original

    def fit(self, X, y=None):
        _require_columns(X, self.columns, "BoundedToLogit.fit")
        self._columns_ = tuple(self.columns)
        self._eps_ = float(self.eps)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        before = df.columns
        e = self._eps_
        for c in self._columns_:
            p = pd.to_numeric(df[c], errors="coerce").astype(float).clip(e, 1.0 - e)
            df[f"{c}{self.out_suffix}"] = np.log(p / (1.0 - p))
            if self.drop_original:
                df.drop(columns=[c], inplace=True)
        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} columns: {cols}")
        return df

class PolynomialAndInteractions(BaseEstimator, TransformerMixin, VerboseMixin):
    """
    Build q^1..q^degree and (q × features) for passed columns.
    Useful for OLS/Ridge models that compute exact ME via derivative wrt q.
    """
    def __init__(self,
                 q_col: str,
                 degree: int = 4,
                 interaction_cols: Sequence[str] = (),
                 poly_prefix: Optional[str] = None,     # defaults to f"{q_col}_p"
                 inter_prefix: Optional[str] = None,    # defaults to f"{q_col}_x_"
                 verbose: bool = False):
        super().__init__(verbose=verbose)
        self.q_col = q_col
        self.degree = int(degree)
        self.interaction_cols = list(interaction_cols)
        self.poly_prefix = poly_prefix
        self.inter_prefix = inter_prefix

    def fit(self, X, y=None):
        _require_columns(X, [self.q_col], "PolynomialAndInteractions.fit")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        before = df.columns
        q = df[self.q_col].astype(float)

        ppref = self.poly_prefix or f"{self.q_col}_p"
        ipref = self.inter_prefix or f"{self.q_col}_x_"

        for k in range(1, self.degree + 1):
            df[f"{ppref}{k}"] = q ** k

        for c in self.interaction_cols:
            if c in df.columns:
                df[f"{ipref}{c}"] = q * df[c].astype(float)

        n_add, cols = _report_added(before, df.columns)
        self._log(f"added {n_add} columns: {cols}")
        return df

##### Models
def print_model_summary(model):
    """
    Small helper to print the most relevant knobs & learned bits for any of the below.
    """
    if hasattr(model, "describe"):
        print(model.describe())
    elif hasattr(model, "coef_"):
        n = 0 if getattr(model, "coef_", None) is None else len(model.coef_)
        print({"class": model.__class__.__name__, "n_coefs": n})
    else:
        print({"class": model.__class__.__name__})
###### GAM Models
class _BaseGAMEstimator(BaseEstimator, TransformerMixin):
    """
    Flexible base for (Q)GAM models.
    - Builds X-matrix from provided feature blocks.
    - Computes Marginal Emissions via dY/dQ (chain rule from standardized Q).
    - Prints what it’s doing when verbose=True.

    Parameters
    ----------
    y_var : str
        Target column.
    q_col : str
        Raw demand column (original units).
    q_std_col : str
        Standardized demand column (preprocess step must have created).
    linear_cols : Sequence[str] | None
        Columns to include as linear terms.
    smooth_cols : Sequence[str] | None
        Columns to include as smooth terms (NOT including q_std_col; Q is handled separately).
    fe_cols : Sequence[str] | None
        One-hot or other fixed-effect columns (treated linear).
    lam_by_col : dict[str, float] | None
        Optional per-column smoothing penalty (only applied to smooth terms).
    n_splines_by_col : dict[str, int] | None
        Optional per-column spline count (only applied to smooth terms).
    n_splines_q : int
        Splines for Q spline.
    lam_q : float
        Penalty for Q spline.
    fd_h : float
        Finite-difference step for derivative wrt q_std.
    require_all_features : bool
        If False, silently drop missing features (or warn if missing_action='warn').
    missing_action : {'ignore','warn','error'}
        Behavior when a requested feature is missing.
    verbose : bool
        Print feature assembly and sizes.
    random_state : int | None
        Seed used by pygam (controls knot placement).
    """
    def __init__(
        self,
        y_var: str = "tons_co2",
        q_col: str = "demand_met",
        q_std_col: str = "demand_met_std",
        linear_cols: Optional[Sequence[str]] = None,
        smooth_cols: Optional[Sequence[str]] = None,
        fe_cols: Optional[Sequence[str]] = None,
        lam_by_col: Optional[Dict[str, float]] = None,
        n_splines_by_col: Optional[Dict[str, int]] = None,
        n_splines_q: int = 15,
        lam_q: float = 10.0,
        fd_h: float = 1e-4,
        require_all_features: bool = False,
        missing_action: str = "warn",   # 'ignore'|'warn'|'error'
        verbose: bool = False,
        random_state: Optional[int] = 12,
    ):
        self.y_var = y_var
        self.q_col = q_col
        self.q_std_col = q_std_col
        self.linear_cols = list(linear_cols or [])
        self.smooth_cols = list(smooth_cols or [])
        self.fe_cols = list(fe_cols or [])
        self.lam_by_col = dict(lam_by_col or {})
        self.n_splines_by_col = dict(n_splines_by_col or {})
        self.n_splines_q = int(n_splines_q)
        self.lam_q = float(lam_q)
        self.fd_h = float(fd_h)
        self.require_all_features = bool(require_all_features)
        self.missing_action = missing_action
        self.verbose = bool(verbose)
        self.random_state = random_state

        # learned
        self.gam_: Optional[LinearGAM] = None
        self._x_cols_: List[str] = []     # in the exact order used to build X
        self._term_kinds_: List[str] = [] # 'q','smooth','linear'
        self._q_index_: Optional[int] = None
        self._q_std_: Optional[float] = None
        self._q_mean_: Optional[float] = None

    # --- logging helper
    def _log(self, msg: str):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")

    # --- feature assembly
    def _resolve_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        want = [self.q_std_col] + self.smooth_cols + self.linear_cols + self.fe_cols
        missing = [c for c in want if c not in df.columns]
        if missing:
            if self.require_all_features or self.missing_action == "error":
                raise KeyError(f"Missing features: {missing}")
            elif self.missing_action == "warn":
                self._log(f"WARNING: dropping missing features: {missing}")
            # silently drop
        smooth = [c for c in self.smooth_cols if c in df.columns]
        linear = [c for c in (self.linear_cols + self.fe_cols) if c in df.columns]
        return [self.q_std_col], smooth, linear

    def _build_matrix(self, df: pd.DataFrame) -> np.ndarray:
        qc, smooth, linear = self._resolve_features(df)
        cols = qc + smooth + linear
        X = df[cols].to_numpy(dtype=float)
        self._x_cols_ = cols
        # term kinds align with pygam terms list
        kinds = (["q"] * len(qc)) + (["smooth"] * len(smooth)) + (["linear"] * len(linear))
        self._term_kinds_ = kinds
        self._q_index_ = cols.index(self.q_std_col)
        self._log(f"X-matrix built with {X.shape[0]:,} rows × {X.shape[1]:,} cols")
        return X

    # --- term specification for pygam
    def _gam_terms(self) -> Any:
        """Return a TermList matching the _x_cols_/_term_kinds_ order."""
        terms: List[Any] = []
        for i, kind in enumerate(self._term_kinds_):
            if kind == "q":
                terms.append(s(i))
            elif kind == "smooth":
                terms.append(s(i))
            else:
                terms.append(l(i))
        return _sum_terms(terms)

    # --- fit/predict
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if y is None:
            if self.y_var not in X.columns:
                raise KeyError(f"y_var '{self.y_var}' not in X")
            y_arr = X[self.y_var].astype(float).to_numpy()
        else:
            y_arr = pd.Series(y).astype(float).to_numpy()

        # chain-rule scaling
        if self.q_col not in X.columns or self.q_std_col not in X.columns:
            raise KeyError(f"Expect both '{self.q_col}' and '{self.q_std_col}' in X")

        q_series = X[self.q_col].astype(float)
        self._q_mean_ = float(q_series.mean())
        q_std = float(q_series.std(ddof=0))
        self._q_std_ = 1.0 if q_std == 0.0 else q_std

        Xmat = self._build_matrix(X)
        terms = self._gam_terms()

        # Build lam vector per term
        lam_vec = []
        # Map column index -> original column name
        for i, kind in enumerate(self._term_kinds_):
            col = self._x_cols_[i]
            if i == self._q_index_:
                lam_vec.append(self.lam_q)
            elif kind == "smooth":
                lam_vec.append(self.lam_by_col.get(col, 0.0))  # default 0 unless provided
            else:
                lam_vec.append(0.0)  # linear terms ignore lam

        np.random.seed(self.random_state)
        gam = LinearGAM(terms, fit_intercept=True)
        gam.lam = lam_vec

        # set n_splines per smooth
        for i, kind in enumerate(self._term_kinds_):
            if kind in {"q", "smooth"}:
                if i == self._q_index_:
                    gam.n_splines[i] = int(self.n_splines_q)
                else:
                    name = self._x_cols_[i]
                    gam.n_splines[i] = int(self.n_splines_by_col.get(name, 20))

        self.gam_ = gam.fit(Xmat, y_arr)
        self._log(f"fit complete; effective_dof={np.sum(self.gam_.statistics_['edof']):.1f}")
        return self

    def _derivative_q_std_df(self, df: pd.DataFrame, h: Optional[float] = None) -> np.ndarray:
        assert self.gam_ is not None, "Model not fitted"
        step = float(self.fd_h if h is None else h)
        df_p = df.copy()
        df_m = df.copy()
        df_p[self.q_std_col] = df_p[self.q_std_col].astype(float) + step
        df_m[self.q_std_col] = df_m[self.q_std_col].astype(float) - step

        Xp = self._build_matrix(df_p)
        Xm = self._build_matrix(df_m)
        y_p = self.gam_.predict(Xp)
        y_m = self.gam_.predict(Xm)
        return (y_p - y_m) / (2.0 * step)

    def predict(self, X: pd.DataFrame, predict_type: str = "ME") -> np.ndarray:
        if self.gam_ is None:
            raise RuntimeError("Model not fitted")

        if predict_type.lower() == "y":
            Xmat = self._build_matrix(X)
            return self.gam_.predict(Xmat)

        d_y_d_qstd = self._derivative_q_std_df(X)
        q_sd = float(self._q_std_ or 1.0)
        return d_y_d_qstd * (1.0 / q_sd)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["y_pred"] = self.predict(df, predict_type="y")
        df["ME"] = self.predict(df, predict_type="ME")
        return df

    # nice to have
    def describe(self) -> Dict[str, Any]:
        return {
            "y_var": self.y_var,
            "q_col": self.q_col,
            "q_std_col": self.q_std_col,
            "linear_cols": self.linear_cols,
            "smooth_cols": self.smooth_cols,
            "fe_cols": self.fe_cols,
            "n_splines_q": self.n_splines_q,
            "lam_q": self.lam_q,
            "lam_by_col": self.lam_by_col,
            "n_splines_by_col": self.n_splines_by_col,
            "fd_h": self.fd_h,
        }

    def get_feature_names_out(self, input_features=None):
        return np.array(self._x_cols_)
class QSplineRegressorPyGAM(_BaseGAMEstimator):
    """
    Q spline + (all others linear).
    Pass fe_cols if have one-hots to include.
    """
    def __init__(self, **kwargs):
        super().__init__(smooth_cols=[], **kwargs)
class QGAMRegressorPyGAM(_BaseGAMEstimator):
    """
    Q spline + user-defined smooth_cols + linear_cols/fe_cols.
    Use lam_by_col/n_splines_by_col for per-term control.
    """
    pass
###### Demand - Polynomial OLS
class DemandMarginalEffectsOLS(BaseEstimator, TransformerMixin):
    """
    Polynomial in raw Q + (Q × selected features) + linear controls, fit by OLS.
    Exact ME from closed-form derivative wrt Q.

    Parameters
    ----------
    y_var : str
    q : str
    poly_degree : int
        Degree of polynomial in Q (>=1).
    interaction_features : Sequence[str]
        Features to interact with Q (use standardized variants ideally).
    linear_controls : Sequence[str] | None
        Always-in linear controls; if None, inferred: interaction_features + common time/controls.
    add_time : bool
        Expect time Fourier / flags already present; if True and found, they’ll be auto-included.
    winsor_precip_col : str | None
        If not None, cap uses the training percentile for this column before log1p (compat layer).
    precip_log1p_col : str | None
        Name to use for the log1p column. If None, no special precip handling here.
    verbose : bool
    """
    def __init__(
        self,
        y_var: str = "tons_co2",
        q: str = "demand_met",
        ts_col: str = "timestamp",
        wind_dir_col: str = "wind_direction_meteorological",
        precip_col: str = "precipitation_mm",
        to_std: list[str] = (
            "wind_speed_mps",
            "temperature_celsius",
            "surface_net_solar_radiation_kWh_per_m2",
            "precipitation_mm_log1p",
            "total_cloud_cover",
        ),
        degree: int = 4,
        # if your config sometimes passes poly_degree, you can add it too:
        # poly_degree: int | None = None,
        inter_features: list[str] | None = None,
        linear_controls: list[str] | None = None,
        winsor_upper: float = 0.995,
        add_time_feats: bool = True,
        add_wind_dir: bool = True,
        verbose: bool = False,
    ):
        # --- store EXACT names so sklearn.get_params works ---
        self.y_var = y_var
        self.q = q
        self.ts_col = ts_col
        self.wind_dir_col = wind_dir_col
        self.precip_col = precip_col
        self.to_std = list(to_std) if to_std is not None else []
        self.poly_degree = int(degree)
        self.interaction_features = list(inter_features) if inter_features is not None else []
        self.lin_controls_cfg =  None if linear_controls is None else list(linear_controls)
        self.winsor_upper = float(winsor_upper)
        self.add_time_feats = bool(add_time_feats)
        self.add_wind_dir = bool(add_wind_dir)
        self.verbose = bool(verbose)
        self.winsor_precip_col = None
        self.precip_log1p_col = None

        # --- optional internal aliases (safe) ---
        self.to_std_raw = self.to_std

        # learned later
        self.cap_hi_ = None
        self.mu_ = None
        self.sd_ = None
        self.rhs_cols_ = None
        self.coef_ = None
        self.model_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")

    # --- design construction
    def _build_design(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        out = df.copy()
        # Optional compatibility: winsor + log1p precip
        if self.winsor_precip_col and self.precip_log1p_col and (self.winsor_precip_col in out.columns):
            if self.cap_hi_ is None:
                self.cap_hi_ = float(np.nanpercentile(out[self.winsor_precip_col], self.winsor_upper*100))
            out[self.winsor_precip_col] = out[self.winsor_precip_col].clip(0.0, self.cap_hi_)
            out[self.precip_log1p_col] = np.log1p(out[self.winsor_precip_col])

        qv = out[self.q].astype(float)
        poly_cols = []
        for k in range(1, self.poly_degree + 1):
            name = f"{self.q}_p{k}"
            out[name] = qv ** k
            poly_cols.append(name)

        inter_cols = []
        for c in self.interaction_features:
            if c in out.columns:
                name = f"{self.q}_x_{c}"
                out[name] = qv * out[c].astype(float)
                inter_cols.append(name)

        # default linear controls (if not explicitly set)
        if self.lin_controls_cfg is not None:
            lin_controls = [c for c in self.lin_controls_cfg if c in out.columns]
        else:
            base_controls = list(self.interaction_features)
            if self.add_time_feats:
                base_controls += [c for c in ("hour_sin","hour_cos","dow_sin","dow_cos","doy_sin","doy_cos","is_weekend") if c in out.columns]
            # precip & cloud if present
            base_controls += [c for c in ("total_cloud_cover_std", self.precip_log1p_col) if c and c in out.columns]
            # unique & present
            seen = set()
            lin_controls = [c for c in base_controls if (c in out.columns) and (not (c in seen or seen.add(c)))]

        rhs = poly_cols + inter_cols + lin_controls
        return rhs, out

    # --- sklearn API
    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        if self.y_var in X.columns and y is None:
            y_ser = X[self.y_var]
        else:
            y_ser = pd.Series(y, index=X.index, name=self.y_var)

        rhs, D = self._build_design(X)
        self.rhs_cols_ = rhs
        self._log(f"RHS terms ({len(rhs)}): {rhs[:8]}{' ...' if len(rhs)>8 else ''}")

        Xmat = sm.add_constant(D[rhs], has_constant="add")
        self.model_ = sm.OLS(y_ser.astype(float).to_numpy(), Xmat.to_numpy(float)).fit()
        self.coef_ = pd.Series(self.model_.params, index=Xmat.columns)
        self._log("fit complete")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        D = self._transform_to_design(X)
        Xm = sm.add_constant(D[self.rhs_cols_], has_constant="add")
        return np.asarray(Xm.to_numpy(float) @ self.coef_.values, dtype=float)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        yhat = self.predict(X)
        me = self.marginal_effects(X)
        out = X.copy()
        out["y_pred"] = yhat
        out["ME"] = me
        return out

    def marginal_effects(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        D = self._features_only(X)
        qv = D[self.q].astype(float).to_numpy()
        def c(name: str) -> float: return float(self.coef_.get(name, 0.0))

        me = np.zeros_like(qv, dtype=float)
        for k in range(1, self.poly_degree + 1):
            me += k * c(f"{self.q}_p{k}") * (qv ** (k-1))
        for feat in self.interaction_features:
            col = f"{self.q}_x_{feat}"
            if (col in D.columns) and (feat in D.columns):
                me = me + c(col) * D[feat].astype(float).to_numpy()
        return me

    # --- helpers
    def _features_only(self, X: pd.DataFrame) -> pd.DataFrame:
        # No extra feature builders here: assume preprocessing already added stds/time/etc.
        df = X.copy()
        # If using precip compatibility, recompute log1p from learned cap
        if self.winsor_precip_col and self.precip_log1p_col and (self.winsor_precip_col in df.columns):
            cap = float(self.cap_hi_ or np.nanpercentile(df[self.winsor_precip_col], self.winsor_upper*100))
            df[self.winsor_precip_col] = df[self.winsor_precip_col].clip(0.0, cap)
            df[self.precip_log1p_col] = np.log1p(df[self.winsor_precip_col])
        return df

    def _transform_to_design(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        _, D = self._build_design(self._features_only(X))
        for col in self.rhs_cols_:
            if col not in D.columns:
                D[col] = 0.0
        return D

    def _check_fitted(self):
        if self.model_ is None or self.rhs_cols_ is None:
            raise RuntimeError("Estimator not fitted. Call .fit(X, y) first.")

    def score(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> dict:
        y_true = np.asarray(y, dtype=float)
        y_pred = self.predict(X)
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        return {"r2": r2_score(y_true, y_pred), "rmse": float(rmse), "mae": mean_absolute_error(y_true, y_pred), "n": int(len(y_true))}
###### Polynomial with RidgeCV
class PolynomialMERegressorRidge(BaseEstimator, TransformerMixin):
    """
    RidgeCV model on standardized Q and drivers:
      Features: [Q_std, Q_std^2..Q_std^K] + drivers_std + (Q_std × driver_std) + optional time cols.
      Provides y_pred and ME in ORIGINAL Q units via chain rule.

    Parameters
    ----------
    y_var, q_col, q_std_col : str
    driver_std_cols : Sequence[str]
    time_cols : Sequence[str]
    q_poly_degree : int
        Degree of polynomial in standardized Q (>=1).
    alphas : tuple[float]
    cv : int
    fit_intercept : bool
    verbose : bool
    """
    def __init__(
        self,
        y_var="tons_co2",
        q_col="demand_met",
        q_std_col="demand_met_std",
        driver_std_cols=(),
        time_cols=(),
        q_poly_degree: int = 2,
        alphas=(1e-3, 1e-2, 1e-1, 1, 3, 10, 30, 100),
        cv=5,
        fit_intercept=True,
        verbose: bool = False,
        random_state=12,
    ):
        self.y_var = y_var
        self.q_col = q_col
        self.q_std_col = q_std_col
        self.driver_std_cols = list(driver_std_cols)
        self.time_cols = list(time_cols)
        self.q_poly_degree = int(q_poly_degree)
        self.alphas = list(alphas)
        self.cv = int(cv)
        self.fit_intercept = bool(fit_intercept)
        self.verbose = bool(verbose)
        self.random_state = random_state

        self.model_ = None
        self.q_std_train_ = None
        self.feature_names_ = None
        self.coef_series_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        Qs = df[self.q_std_col].astype(float).to_numpy()
        blocks = []
        names = []

        # Q_std polys
        for k in range(1, self.q_poly_degree + 1):
            blocks.append((Qs**k).reshape(-1,1))
            names.append(f"Q_std^{k}")

        # drivers linear
        Z = []
        Z_names = []
        for c in self.driver_std_cols:
            if c in df.columns:
                z = df[c].astype(float).to_numpy().reshape(-1,1)
                Z.append(z); Z_names.append(c)
        if Z:
            Z = np.hstack(Z)
            blocks.append(Z)
            names += Z_names

            # interactions Q_std^1 × driver
            Qi = Qs.reshape(-1,1)  # only first-power in interaction (keeps ME simple)
            blocks.append(Z * Qi)
            names += [f"Q_std:{c}" for c in Z_names]

        # time cols
        T = []
        T_names = []
        for c in self.time_cols:
            if c in df.columns:
                T.append(df[c].astype(float).to_numpy().reshape(-1,1))
                T_names.append(c)
        if T:
            T = np.hstack(T)
            blocks.append(T)
            names += T_names

        X = np.hstack(blocks) if blocks else np.empty((len(df),0))
        self.feature_names_ = names
        return X

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            if self.y_var not in X.columns:
                raise KeyError(f"'{self.y_var}' must be present if y=None")
            y_arr = X[self.y_var].astype(float).to_numpy()
        else:
            y_arr = pd.Series(y).astype(float).to_numpy()

        if self.q_col not in X.columns or self.q_std_col not in X.columns:
            raise KeyError(f"Expect both '{self.q_col}' and '{self.q_std_col}' in X")
        self.q_std_train_ = float(X[self.q_col].astype(float).std(ddof=0)) or 1.0

        D = self._design(X)
        if len(self.alphas) == 1:
            self.model_ = Ridge(alpha=float(self.alphas[0]), fit_intercept=self.fit_intercept)
        else:
            self.model_ = RidgeCV(alphas=self.alphas, cv=self.cv, fit_intercept=self.fit_intercept, scoring="r2")
        self.model_.fit(D, y_arr)

        self.coef_series_ = pd.Series(self.model_.coef_, index=self.feature_names_)
        self._log(f"fit complete; alpha_={getattr(self.model_, 'alpha_', np.nan)}")
        return self

    def predict(self, X: pd.DataFrame, predict_type: str = "ME"):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")

        if predict_type.lower() == "y":
            D = self._design(X)
            return self.model_.predict(D)

        # dY/dQ_std = Σ k*β_{Qk} Q_std^{k-1} + Σ β_{Q_std:driver}*driver
        Qs = X[self.q_std_col].astype(float).to_numpy()
        beta = self.coef_series_
        def b(name): return float(beta.get(name, 0.0))

        dY_dQstd = 0.0
        for k in range(1, self.q_poly_degree + 1):
            dY_dQstd += k * b(f"Q_std^{k}") * (Qs ** (k-1))
        for c in self.driver_std_cols:
            dY_dQstd += b(f"Q_std:{c}") * X.get(c, 0.0)

        return dY_dQstd * (1.0 / float(self.q_std_train_))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["y_pred"] = self.predict(df, predict_type="y")
        df["ME"] = self.predict(df, predict_type="ME")
        return df

    @property
    def coef_(self) -> Optional[pd.Series]:
        return self.coef_series_.copy() if self.coef_series_ is not None else None

    @property
    def alpha_(self) -> float:
        return float(getattr(self.model_, "alpha_", np.nan))
###### Polynomial Huber Regressor
class PolynomialMERegressorHuber(BaseEstimator, TransformerMixin):
    """
    Huber-regression version of your polynomial-in-Q level model.
    Features: [Q_std, Q_std^2..Q_std^K] + drivers_std + (Q_std × driver_std) + optional time cols.
    Predicts level y and ME = dY/dQ in ORIGINAL Q units (chain rule).

    Parameters mirror your Ridge variant where possible.
    """
    def __init__(
        self,
        y_var="tons_co2",
        q_col="demand_met",
        q_std_col="demand_met_std",
        driver_std_cols=(),
        time_cols=(),
        q_poly_degree: int = 2,
        # Huber knobs
        epsilon: float = 1.35,      # outlier threshold
        alpha: float = 1e-4,        # L2 on weights (small regularization)
        fit_intercept: bool = True,
        verbose: bool = False,
        random_state: int = 12,
    ):
        self.y_var = y_var
        self.q_col = q_col
        self.q_std_col = q_std_col
        self.driver_std_cols = list(driver_std_cols)
        self.time_cols = list(time_cols)
        self.q_poly_degree = int(q_poly_degree)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.verbose = bool(verbose)
        self.random_state = int(random_state)

        self.model_ = None
        self.q_std_train_ = None
        self.feature_names_ = None
        self.coef_series_ = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        Qs = df[self.q_std_col].astype(float).to_numpy()
        blocks, names = [], []

        # Q_std polys
        for k in range(1, self.q_poly_degree + 1):
            blocks.append((Qs**k).reshape(-1,1))
            names.append(f"Q_std^{k}")

        # drivers linear + interactions (Q_std × driver)
        Z, Z_names = [], []
        for c in self.driver_std_cols:
            if c in df.columns:
                z = df[c].astype(float).to_numpy().reshape(-1,1)
                Z.append(z); Z_names.append(c)
        if Z:
            Z = np.hstack(Z); blocks.append(Z); names += Z_names
            Qi = Qs.reshape(-1,1)
            blocks.append(Z * Qi); names += [f"Q_std:{c}" for c in Z_names]

        # time cols
        T, T_names = [], []
        for c in self.time_cols:
            if c in df.columns:
                T.append(df[c].astype(float).to_numpy().reshape(-1,1)); T_names.append(c)
        if T:
            T = np.hstack(T); blocks.append(T); names += T_names

        X = np.hstack(blocks) if blocks else np.empty((len(df),0))
        self.feature_names_ = names
        return X

    def fit(self, X: pd.DataFrame, y=None):
        if y is None:
            if self.y_var not in X.columns:
                raise KeyError(f"'{self.y_var}' must be present if y=None")
            y_arr = X[self.y_var].astype(float).to_numpy()
        else:
            y_arr = pd.Series(y).astype(float).to_numpy()

        if self.q_col not in X.columns or self.q_std_col not in X.columns:
            raise KeyError(f"Expect both '{self.q_col}' and '{self.q_std_col}' in X")
        self.q_std_train_ = float(X[self.q_col].astype(float).std(ddof=0)) or 1.0

        D = self._design(X)
        self.model_ = HuberRegressor(
            epsilon=self.epsilon, alpha=self.alpha,
            fit_intercept=self.fit_intercept, max_iter=200
        )
        self.model_.fit(D, y_arr)
        self.coef_series_ = pd.Series(self.model_.coef_, index=self.feature_names_)
        self._log("fit complete")
        return self

    def predict(self, X: pd.DataFrame, predict_type: str = "ME"):
        if self.model_ is None:
            raise RuntimeError("Model not fitted")

        if predict_type.lower() == "y":
            D = self._design(X)
            return self.model_.predict(D)

        # ME = dY/dQ = (dY/dQ_std) * (1/σ_Q)
        Qs = X[self.q_std_col].astype(float).to_numpy()
        beta = self.coef_series_
        def b(name): return float(beta.get(name, 0.0))

        dY_dQstd = 0.0
        for k in range(1, self.q_poly_degree + 1):
            dY_dQstd += k * b(f"Q_std^{k}") * (Qs ** (k-1))
        for c in self.driver_std_cols:
            dY_dQstd += b(f"Q_std:{c}") * X.get(c, 0.0)

        return dY_dQstd * (1.0 / float(self.q_std_train_))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["y_pred"] = self.predict(df, predict_type="y")
        df["ME"] = self.predict(df, predict_type="ME")
        return df

    @property
    def coef_(self):
        return self.coef_series_.copy() if self.coef_series_ is not None else None

##### Building Pipelines
###### Config Classes
@dataclass
class PreprocessConfig:
    timestamp_col: str = "timestamp"
    wind_dir_col: str = "wind_direction_meteorological"
    precip_col: str = "precipitation_mm"
    use_winsorize_precip: bool = True
    winsor_lower: float = 0.0
    winsor_upper: float = 0.995
    add_log1p_precip: bool = True                 # do log1p(precip)
    include_onehot_time: bool = True              # month/hour one-hot
    onehot_time_cols: Tuple[str, ...] = ("month","hour")
    onehot_drop_first: bool = True
    include_fourier_time: bool = False            # hour/dow/doy sin/cos (set True to use instead of one-hot)
    standardize_cols: Sequence[str] = field(default_factory=lambda: [
        "demand_met",
        "wind_speed_mps",
        "temperature_celsius",
        # note: if add_log1p_precip=True we’ll standardize "precipitation_mm_log1p"
        "surface_net_solar_radiation_joules_per_m2",
        "total_cloud_cover",
    ])
    add_wind_dir_cyclic: bool = True
    verbose: bool = False

@dataclass
class ModelConfig:
    model_kind: str = "qgam"                      # 'qspline' | 'qgam' | 'ridge' | 'ols'
    y_var: str = "tons_co2"
    q_col: str = "demand_met"
    q_std_col: str = "demand_met_std"
    fe_prefixes: Tuple[str, ...] = ("month_", "hour_")  # FE discovery in preprocessed frame
    # Estimator-specific kwargs (pass through):
    # QGAM/QSpline:
    gam_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        n_splines_q=16, lam_q=10.0, random_state=12,
        smooth_cols=[],                    # e.g. ["wind_speed_mps_std", ...]
        linear_cols=[],                    # e.g. ["total_cloud_cover_std","wind_dir_sin","wind_dir_cos"]
        fe_cols=[],                        # will be filled automatically if empty
        lam_by_col={},                     # per-term smoothing penalties
        n_splines_by_col={},               # per-term spline counts
        missing_action="warn",
        verbose=False,
    ))
    # Ridge:
    ridge_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        driver_std_cols=[], time_cols=[], q_poly_degree=2, alphas=(1e-3,1e-2,1e-1,1,3,10,30,100),
        cv=5, fit_intercept=True, verbose=False, random_state=12,
    ))
    # OLS polynomial:
    ols_kwargs: Dict[str, Any] = field(default_factory=lambda: dict(
        poly_degree=4,
        interaction_features=[],           # standardized drivers to interact with Q
        linear_controls=None,              # infer if None
        add_time=True,
        winsor_precip_col=None,            # usually None (handled in preprocess)
        precip_log1p_col="precipitation_mm_log1p",
        winsor_upper=0.995,
        verbose=False,
    ))
###### Build Pre-processing pipeline
def build_preprocess_pipeline(cfg: PreprocessConfig) -> Pipeline:
    """
    Flexible preprocessing pipeline:
      - Date/time features
      - Optional Fourier time OR month/hour one-hot
      - Wind dir → sin/cos
      - Optional winsorize(precip) then log1p
      - Standardize selected continuous columns (adds *_std; keeps originals)
    """
    steps: List[Tuple[str, Any]] = []

    # (A) Add timestamp-derived features
    steps.append(("DateTime", DateTimeFeatureAdder(timestamp_col=cfg.timestamp_col, drop_original=False)))

    # (B) Time encoding: Fourier vs OneHot
    if cfg.include_fourier_time:
        steps.append(("FourierTime", TimeFourierAdder(timestamp_col=cfg.timestamp_col)))
    if cfg.include_onehot_time:
        steps.append(("OneHotTimeFE", OneHotTimeFE(columns=cfg.onehot_time_cols,
                                                   drop_first=cfg.onehot_drop_first, dtype="int8")))

    # (C) Wind direction cyclic
    if cfg.add_wind_dir_cyclic:
        steps.append(("WindDirCyclic", WindDirToCyclic(dir_col=cfg.wind_dir_col,
                                                       out_sin="wind_dir_sin", out_cos="wind_dir_cos", drop_original=True)))

    # (D) Precip treatment: winsor then log1p (if configured)
    if cfg.use_winsorize_precip:
        steps.append(("WinsorizePrecip", Winsorize(columns=[cfg.precip_col],
                                                   lower=cfg.winsor_lower, upper=cfg.winsor_upper)))
    if cfg.add_log1p_precip:
        steps.append(("Log1pPrecip", Log1pTransform(columns=[cfg.precip_col])))

    # (E) Standardize continuous
    std_cols = list(cfg.standardize_cols)
    if cfg.add_log1p_precip and "precipitation_mm_log1p" not in std_cols and cfg.precip_col in std_cols:
        # prefer std of log1p column instead of raw precip if both present
        std_cols = [("precipitation_mm_log1p" if c == cfg.precip_col else c) for c in std_cols]
    steps.append(("Standardize", StandardizeContinuous(columns=std_cols, suffix="_std")))

    if cfg.verbose:
        print("[build_preprocess_pipeline] Steps:", [name for name, _ in steps])
        print("[build_preprocess_pipeline] Standardized cols:", std_cols)

    return Pipeline(steps, verbose=False)

class AttachFENames(BaseEstimator, TransformerMixin):
    """
    Meta-transformer that discovers FE columns after preprocessing and injects them into
    a downstream estimator that has an attribute `fe_cols`.
    """
    def __init__(self, estimator: Any, fe_selector: Optional[Callable[[pd.DataFrame], List[str]]] = None, verbose: bool=False):
        self.estimator = estimator
        self.fe_selector = fe_selector
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.fe_selector is not None:
            fe_cols = self.fe_selector(X)
        else:
            fe_cols = []
        if hasattr(self.estimator, "fe_cols") and (not fe_cols == []):
            self.estimator.fe_cols = fe_cols
            if self.verbose:
                print(f"[AttachFENames] fe_cols attached ({len(fe_cols)}): {fe_cols[:6]}{' ...' if len(fe_cols)>6 else ''}")
        return self

    def transform(self, X):
        return X

def make_default_fe_selector(prefixes: Tuple[str, ...]) -> Callable[[pd.DataFrame], List[str]]:
    def _sel(df: pd.DataFrame) -> List[str]:
        cols = [c for c in df.columns if c.startswith(prefixes)]
        return sorted(cols)
    return _sel
###### Build Full Pipeline
def build_me_pipeline(
    preprocess: Pipeline,
    model_cfg: ModelConfig,
) -> Pipeline:
    """
    Build a full pipeline: [preprocess] -> estimator (QSpline | QGAM | Ridge | OLS).
    FE columns are auto-attached for GAMs if `fe_cols` is empty in gam_kwargs.
    """
    kind = model_cfg.model_kind.lower()

    # ---- choose estimator ----
    if kind == "qspline":
        # QSpline: Q spline + all others linear
        est = QSplineRegressorPyGAM(
            y_var=model_cfg.y_var,
            q_col=model_cfg.q_col,
            q_std_col=model_cfg.q_std_col,
            # map from cfg.gam_kwargs
            linear_cols=model_cfg.gam_kwargs.get("linear_cols", []),
            smooth_cols=[],  # force smooth to empty for QSpline profile
            fe_cols=model_cfg.gam_kwargs.get("fe_cols", []),
            n_splines_q=model_cfg.gam_kwargs.get("n_splines_q", 15),
            lam_q=model_cfg.gam_kwargs.get("lam_q", 10.0),
            lam_by_col=model_cfg.gam_kwargs.get("lam_by_col", {}),
            n_splines_by_col=model_cfg.gam_kwargs.get("n_splines_by_col", {}),
            missing_action=model_cfg.gam_kwargs.get("missing_action", "warn"),
            verbose=model_cfg.gam_kwargs.get("verbose", False),
            random_state=model_cfg.gam_kwargs.get("random_state", 12),
        )
    elif kind == "qgam":
        # QGAM: Q spline + smooth_cols (smoothed) + linear_cols (linear)
        est = QGAMRegressorPyGAM(
            y_var=model_cfg.y_var,
            q_col=model_cfg.q_col,
            q_std_col=model_cfg.q_std_col,
            smooth_cols=model_cfg.gam_kwargs.get("smooth_cols", []),
            linear_cols=model_cfg.gam_kwargs.get("linear_cols", []),
            fe_cols=model_cfg.gam_kwargs.get("fe_cols", []),
            n_splines_q=model_cfg.gam_kwargs.get("n_splines_q", 15),
            lam_q=model_cfg.gam_kwargs.get("lam_q", 10.0),
            lam_by_col=model_cfg.gam_kwargs.get("lam_by_col", {}),
            n_splines_by_col=model_cfg.gam_kwargs.get("n_splines_by_col", {}),
            missing_action=model_cfg.gam_kwargs.get("missing_action", "warn"),
            verbose=model_cfg.gam_kwargs.get("verbose", False),
            random_state=model_cfg.gam_kwargs.get("random_state", 12),
        )
    elif kind == "ridge":
        est = PolynomialMERegressorRidge(
            y_var=model_cfg.y_var,
            q_col=model_cfg.q_col,
            q_std_col=model_cfg.q_std_col,
            driver_std_cols=model_cfg.ridge_kwargs.get("driver_std_cols", []),
            time_cols=model_cfg.ridge_kwargs.get("time_cols", []),
            q_poly_degree=model_cfg.ridge_kwargs.get("q_poly_degree", 2),
            alphas=model_cfg.ridge_kwargs.get("alphas", (1e-3,1e-2,1e-1,1,3,10,30,100)),
            cv=model_cfg.ridge_kwargs.get("cv", 5),
            fit_intercept=model_cfg.ridge_kwargs.get("fit_intercept", True),
            verbose=model_cfg.ridge_kwargs.get("verbose", False),
            random_state=model_cfg.ridge_kwargs.get("random_state", 12),
        )
    elif kind == "ols":
        est = DemandMarginalEffectsOLS(
            y_var=model_cfg.y_var,
            q=model_cfg.q_col,
            degree=model_cfg.ols_kwargs.get("poly_degree", 4),
            inter_features=model_cfg.ols_kwargs.get("inter_features", []),
            linear_controls=model_cfg.ols_kwargs.get("lin_controls_cfg", None),
            add_time_feats=model_cfg.ols_kwargs.get("add_time_feats", True),
            precip_col=model_cfg.ols_kwargs.get("precip_col", None),
            winsor_upper=model_cfg.ols_kwargs.get("winsor_upper", 0.995),
            verbose=model_cfg.ols_kwargs.get("verbose", False),
        )
    else:
        raise ValueError("model_kind must be one of {'qspline','qgam','ridge','ols'}")

    # ---- optionally attach FE names dynamically for GAMs ----
    steps: List[Tuple[str, Any]] = [("Preprocess", preprocess)]
    if kind in {"qspline","qgam"} and len(getattr(est, "fe_cols", []) or []) == 0:
        fe_selector = make_default_fe_selector(model_cfg.fe_prefixes)
        steps.append(("AttachFE", AttachFENames(est, fe_selector=fe_selector, verbose=False)))
    steps.append((est.__class__.__name__, est))

    return Pipeline(steps, verbose=False)
##### Evaluation
###### Helpers
def _safe_rmse(y_true, y_pred) -> float:
    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def _maybe_attach_predictions(fitted_pipeline, Z: pd.DataFrame,
                              y_col: str, me_col: str = "ME") -> pd.DataFrame:
    """
    Ensure Z has 'y_pred' and (optionally) ME. If the final estimator didn't
    attach them in transform(), try calling predict() directly.
    """
    out = Z.copy()
    final_est = getattr(fitted_pipeline, "_final_estimator", None)

    if "y_pred" not in out.columns and final_est is not None:
        try:
            out["y_pred"] = final_est.predict(out, predict_type="y")
        except Exception:
            pass

    if me_col not in out.columns and final_est is not None:
        try:
            out[me_col] = final_est.predict(out, predict_type="ME")
        except Exception:
            # okay to skip ME if the model doesn't support it
            pass

    # keep y_true explicitly if provided
    if y_col in out.columns and "y_true" not in out.columns:
        out["y_true"] = out[y_col].astype(float)

    return out
def compute_point_metrics(df: pd.DataFrame,
                          y_true_col: str = "y_true",
                          y_pred_col: str = "y_pred") -> Dict[str, float]:
    yt = df[y_true_col].to_numpy(float)
    yp = df[y_pred_col].to_numpy(float)
    return {
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
        "rmse": _safe_rmse(yt, yp),
        "mae": float(mean_absolute_error(yt, yp)),
        "n_obs": int(len(yt)),
    }
def compute_group_metrics(df: pd.DataFrame,
                          group_col: str,
                          y_true_col: str = "y_true",
                          y_pred_col: str = "y_pred") -> pd.DataFrame:
    rows = []
    for g, gdf in df.groupby(group_col, observed=True, sort=True):
        if gdf[y_true_col].notna().sum() < 1 or gdf[y_pred_col].notna().sum() < 1:
            continue
        yt = gdf[y_true_col].to_numpy(float)
        yp = gdf[y_pred_col].to_numpy(float)
        rows.append({
            group_col: g,
            "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
            "rmse": _safe_rmse(yt, yp),
            "mae": float(mean_absolute_error(yt, yp)),
            "n_obs": int(len(yt)),
        })
    return pd.DataFrame(rows)
###### Main Evaluator
def evaluate_pipeline(
    fitted_pipeline,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    *,
    y_col: str = "tons_co2",
    me_col: str = "ME",
    id_cols: Sequence[str] = ("timestamp", "city"),
    include_me: bool = True,
    return_predictions: bool = True,
    eval_me_slopes: bool = False,                 # use finite-diff evaluator
    me_slope_kwargs: Optional[Dict[str, Any]] = None,  # e.g. {"time_col":"timestamp","q_col":"demand_met",...}
) -> Dict[str, Any]:
    """
    Generic evaluator for ANY of your pipelines (QSpline, QGAM, Ridge, OLS).
    - Runs pipeline.transform
    - Ensures y_pred (and ME if requested)
    - Returns overall metrics, optional per-group, and optional ME slope diagnostics
    - Optionally returns the tidy predictions frame
    """
    # Combine y if provided (so we can keep y_true alongside predictions)
    if y is not None:
        X_in = pd.concat([X, y.rename(y_col)], axis=1)
    else:
        X_in = X

    # Transform through full pipeline; many of your estimators attach y_pred/ME in transform()
    Z = fitted_pipeline.transform(X_in)

    # Make sure y_pred/ME/y_true are present
    Z = _maybe_attach_predictions(fitted_pipeline, Z, y_col=y_col, me_col=me_col)

    # Build a compact predictions frame
    keep = [c for c in id_cols if c in Z.columns]
    cols = keep + [c for c in ["y_true", "y_pred"] if c in Z.columns]
    if include_me and me_col in Z.columns:
        cols.append(me_col)
    preds = Z[cols].copy()

    # Overall metrics (if y_true/y_pred are present)
    overall = {}
    if {"y_true", "y_pred"}.issubset(preds.columns):
        overall = compute_point_metrics(preds, y_true_col="y_true", y_pred_col="y_pred")

    # Per-group metrics if the estimator defines a group column and it's present
    per_group = pd.DataFrame()
    final_est = getattr(fitted_pipeline, "_final_estimator", None)
    gcol = getattr(final_est, "group_col", None)
    if gcol and gcol in Z.columns and {"y_true", "y_pred"}.issubset(Z.columns):
        per_group = compute_group_metrics(Z[[gcol, "y_true", "y_pred"]], group_col=gcol)

    # Optional finite-difference comparison for marginal effects
    me_diag = {}
    if eval_me_slopes and me_col in Z.columns:
        kwargs = dict(
            time_col="timestamp",
            q_col="demand_met",
            y_col=y_col,
            me_col=me_col,
            group_keys=("city",),
            max_dt=pd.Timedelta("2h"),
            min_abs_dq=1e-6,
        )
        if me_slope_kwargs:
            kwargs.update(me_slope_kwargs)

        try:
            me_diag = finite_difference_me_metrics(Z, **kwargs).to_dict(orient="list")
        except Exception as e:
            me_diag = {"error": str(e)}

    out = {
        "overall": overall,
        "per_group": per_group,
    }
    if eval_me_slopes:
        out["me_slope_diagnostics"] = me_diag
    if return_predictions:
        out["predictions"] = preds
    return out


### Running Models
def apply_fitted_preprocessing(
    user_pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    stop_before: str | None = None,  # name of step to stop before; None means "before final estimator"
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Apply all *already-fitted* steps in a pipeline up to (but not including) `stop_before`
    or the final estimator if stop_before is None.

    - Preserves fitted state; does NOT refit anything.
    - Returns a DataFrame (attempts to recover column names).

    Examples
    --------
    apply_fitted_preprocessing(pipe, X)                       # all preprocessors
    apply_fitted_preprocessing(pipe, X, stop_before="Binner") # stop before a named step
    """
    steps = list(user_pipeline.steps)
    if stop_before is None and len(steps) > 0:
        steps = steps[:-1]  # drop final estimator
    elif stop_before is not None:
        names = [n for n, _ in steps]
        if stop_before not in names:
            raise KeyError(f"stop_before='{stop_before}' not found in pipeline steps: {names}")
        cut = names.index(stop_before)
        steps = steps[:cut]

    Z = X
    last_transformer = None
    for name, step in steps:
        if hasattr(step, "transform"):
            if verbose:
                print(f"[apply] {name}.transform(...)")
            Z = step.transform(Z)
            last_transformer = step

    if isinstance(Z, pd.DataFrame):
        return Z

    # Try to recover feature names
    cols = None
    try:
        cols = Pipeline(steps).get_feature_names_out()
    except Exception:
        try:
            if last_transformer is not None and hasattr(last_transformer, "get_feature_names_out"):
                cols = last_transformer.get_feature_names_out()
        except Exception:
            cols = None

    if cols is None:
        cols = getattr(X, "columns", [f"f{i}" for i in range(np.asarray(Z).shape[1])])

    return pd.DataFrame(Z, index=X.index, columns=list(cols))
def compute_predictions_for_split(
    fitted_pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    y_col: Optional[str] = None,                 # if provided and present in X, will be copied to y_true
    id_cols: Sequence[str] = ("timestamp", "city"),
    include_y_pred: bool = True,
    include_me: bool = True,
    include_params: bool = True,                 # if estimator exposes per-row params in transform
    keep_cols: Sequence[str] = (),
    split_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run a FITTED pipeline on a single split and return a tidy frame with:
      [id_cols ∩ X] + optional y_true + y_pred + ME + group_id + any keep_cols

    - Works with estimators that attach y_pred/ME in transform(), or that can
      produce them via predict(predict_type="y"/"ME").
    - If ME is unavailable, it is silently omitted.
    """
    out = fitted_pipeline.transform(X)

    final_est = getattr(fitted_pipeline, "_final_estimator", None)
    gcol = getattr(final_est, "group_col", None)

    cols: list[str] = [c for c in id_cols if c in out.columns]

    # y_true passthrough
    if y_col and (y_col in out.columns) and ("y_true" not in out.columns):
        out["y_true"] = out[y_col].astype(float)
    if "y_true" in out.columns:
        cols.append("y_true")

    # ensure y_pred
    if include_y_pred:
        if "y_pred" not in out.columns and final_est is not None:
            try:
                out["y_pred"] = final_est.predict(out, predict_type="y")
            except Exception:
                pass
        if "y_pred" in out.columns:
            cols.append("y_pred")

    # ensure ME
    if include_me:
        if "ME" not in out.columns and final_est is not None:
            try:
                out["ME"] = final_est.predict(out, predict_type="ME")
            except Exception:
                pass
        if "ME" in out.columns:
            cols.append("ME")

    # optional per-row params produced by transform (e.g., alpha1/alpha2 from groupwise models)
    if include_params:
        for c in ("alpha1", "alpha2"):
            if c in out.columns and c not in cols:
                cols.append(c)

    # group column if present
    if gcol and gcol in out.columns:
        cols.append(gcol)

    # any extras the caller wants to keep (e.g., demand_met)
    for c in keep_cols:
        if c in out.columns and c not in cols:
            cols.append(c)

    result = out[cols].copy()
    if split_name is not None:
        result["split"] = split_name
    return result
def fit_and_export_predictions(
    pipeline: Pipeline,
    x_splits: Dict[str, pd.DataFrame],
    y_splits: Optional[Dict[str, pd.Series]] = None,
    out_parquet_path: str = "predictions.parquet",
    *,
    order_splits: Sequence[str] = ("train", "validation", "test"),
    y_col: str = "tons_co2",
    id_cols: Sequence[str] = ("timestamp", "city"),
    include_y_pred: bool = True,
    include_me: bool = True,
    include_params: bool = True,
    keep_cols: Sequence[str] = ("demand_met", "tons_co2"),
    save_mode: str = "single",                # "single" | "per_split"
    compression: Optional[str] = "snappy",
    return_df: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Fit on TRAIN, generate predictions (y_pred/ME) for requested splits, and write Parquet.

    - Works with any final estimator that either:
        * attaches y_pred/ME during transform(), or
        * supports predict(..., predict_type="y"/"ME").
    - If y_splits is provided, y_train is used for fit; y for other splits is optional.
    """
    out_parquet_path = Path(out_parquet_path)

    # ---- fit on train
    X_tr = x_splits["train"]
    if y_splits is not None and "train" in y_splits and y_splits["train"] is not None:
        _ = pipeline.fit_transform(X_tr, y_splits["train"])
    else:
        _ = pipeline.fit_transform(X_tr)  # some estimators expect y inside X (already merged)

    if save_mode not in {"single", "per_split"}:
        raise ValueError("save_mode must be 'single' or 'per_split'.")

    parts: list[pd.DataFrame] = []

    # ---- per split
    for split in order_splits:
        if split not in x_splits:
            continue
        Xs = x_splits[split].copy()

        # attach y if provided (helps compute y_true downstream if transform passes it through)
        if y_splits is not None and split in y_splits and y_splits[split] is not None:
            if y_col not in Xs.columns:
                Xs = pd.concat([Xs, y_splits[split].rename(y_col)], axis=1)

        df_pred = compute_predictions_for_split(
            fitted_pipeline=pipeline,
            X=Xs,
            y_col=y_col if (y_splits is not None and split in y_splits and y_splits[split] is not None) else None,
            id_cols=id_cols,
            include_y_pred=include_y_pred,
            include_me=include_me,
            include_params=include_params,
            keep_cols=keep_cols,
            split_name=split,
        )

        if save_mode == "per_split":
            split_path = out_parquet_path.with_name(
                f"{out_parquet_path.stem}__{split}{out_parquet_path.suffix or '.parquet'}"
            )
            split_path.parent.mkdir(parents=True, exist_ok=True)
            df_pred.to_parquet(split_path, index=False, compression=compression)
            if return_df:
                parts.append(df_pred)
        else:
            parts.append(df_pred)

    # ---- write combined
    if save_mode == "single":
        final = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_parquet(out_parquet_path, index=False, compression=compression)
        print(f"[SAVE] Wrote predictions to {out_parquet_path} (rows={len(final):,})")
        return final if return_df else None
    else:
        print(f"[SAVE] Wrote per-split Parquet files next to {out_parquet_path}")
        if return_df:
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        return None
#### Runners & Orchestrators
def run_regressor_model(
    user_pipeline: Pipeline,
    x_df: pd.DataFrame,
    y_df: pd.Series | pd.DataFrame,
    split_name: str,
    extra_info: dict | None = None,
    return_model: bool = False,
    random_state: int = 12,
    interval_hours: float = 0.5,
    *,
 model_id_hash: str | None = None,
    params_json_str: str | None = None,
) -> tuple[pd.DataFrame, list[str], GroupwiseRegressor | dict]:
    """
    Run a pipeline on one split, compute per-group metrics, attach energy weights,
    and compute diagnostics (pooled CO₂ fit + finite-difference ME checks).

    Parameters
    ----------
    user_pipeline : Pipeline
        Full pipeline [FeatureAddition → (Binner) → GroupwiseRegressor].
    x_df : pd.DataFrame
        Features for the split.
    y_df : pd.Series or single-column pd.DataFrame
        Target for the split.
    split_name : {"train","validation","test"}
        Which split to run.
    extra_info : dict, optional
        Extra metadata to stamp onto the output rows.
    return_model : bool, default False
        If True, returns the final estimator as the 3rd tuple item; otherwise returns extras dict.
    random_state : int, default 12
        Random seed for reproducibility.
    interval_hours : float, default 0.5
        Duration represented by each row (half-hourly = 0.5).
    model_id_hash : str, optional
        If provided, stamp this precomputed run-level hash (recommended).
        If None, a local signature is computed (useful for ad-hoc calls).
    params_json_str : str, optional
        Pre-rendered pipeline params JSON to stamp; if None, it is computed.

    Returns
    -------
    metrics_df : pd.DataFrame
        Per-group metrics with added 'energy_MWh' and metadata columns.
    x_cols_used : list[str]
        Regressor feature names used by the GroupwiseRegressor (x_vars + fe_vars).
    model_or_extras : GroupwiseRegressor | dict
        If return_model=True → the fitted final estimator; else a dict of diagnostics.
    """
    np.random.seed(random_state)

    for col in x_df.columns:
        dt = x_df[col].dtype
        if str(dt).startswith(("uint", "UInt")):
            x_df[col] = x_df[col].astype("int64")

    if split_name not in ("train", "validation", "test"):
        raise ValueError(f"split_name must be 'train', 'validation', or 'test' (got {split_name!r})")

    X = x_df.copy()
    if isinstance(y_df, pd.DataFrame):
        if y_df.shape[1] != 1:
            raise ValueError("y_df must be a Series or single-column DataFrame.")
        y_ser = y_df.iloc[:, 0]
    else:
        y_ser = y_df

    # Use provided model_id_hash (from orchestrator) or compute a local one
    if model_id_hash is None:
        model_id_hash, _ = signature_for_run(
            user_pipeline,
            x_columns=list(X.columns),
            y=y_ser,
            random_state=random_state,
            eval_splits=(split_name,),   # local call; orchestrator passes a shared hash
            compute_test=False,
            extra_info=extra_info,
        )

    if params_json_str is None:
        params_json_str = json.dumps(
            user_pipeline.get_params(deep=True),
            sort_keys=True, separators=(",", ":"), default=str
        )

    extras: dict[str, Any] = {}

    if split_name == "train":
        # Fit → metrics from regressor
        _ = user_pipeline.fit_transform(X, y_ser)
        model = user_pipeline._final_estimator  # type: ignore[attr-defined]
        metrics_df = model.get_metrics(summarise=True).reset_index(drop=True)

        # Canonicalize group col to "group"
        if model.group_col in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={model.group_col: "group"})
        elif "group" not in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={metrics_df.columns[0]: "group"})

        # Preprocessed rows for weights & diagnostics
        x_tr = _apply_fitted_preprocessing(user_pipeline, X)
        x_tr[model.y_var] = np.asarray(y_ser, dtype=float)

        # Energy weights
        w = _compute_group_energy_weights(
            df=x_tr, group_col=model.group_col, q_col=model.x_vars[0], interval_hours=interval_hours
        ).rename(columns={model.group_col: "group"})
        metrics_df = metrics_df.merge(w, on="group", how="left")

        # Diagnostics (in-sample)
        extras["pooled_co2"] = pooled_co2_metrics(
            model, x_tr, y_col=model.y_var, group_col=model.group_col
        )
        me_df = model.transform(x_tr)
        fd_df = finite_difference_me_metrics(
            df=me_df,
            time_col="timestamp" if "timestamp" in me_df.columns else "time_id",
            q_col=model.x_vars[0],
            y_col=model.y_var,
            me_col="ME",
            group_keys=[k for k in ("city",) if k in me_df.columns],
        )
        extras["fd_me_by_city"] = fd_df.to_dict(orient="records") if not fd_df.empty else []
        extras["fd_me_pooled"] = (
            fd_df.loc[fd_df["city"] == "ALL"].iloc[0].to_dict()
            if (not fd_df.empty and "city" in fd_df.columns and "ALL" in fd_df["city"].values)
            else (fd_df.sort_values("n_pairs", ascending=False).iloc[0].to_dict() if not fd_df.empty else {})
        )

    else:
        # Use fitted preprocessing + regressor
        model = user_pipeline._final_estimator  # type: ignore[attr-defined]
        x_tr = _apply_fitted_preprocessing(user_pipeline, X)

        if model.group_col not in x_tr.columns:
            raise KeyError(
                f"Group column '{model.group_col}' is missing after transform. "
                "Ensure your binner outputs it."
            )

        x_tr[model.y_var] = np.asarray(y_ser, dtype=float)

        # Per-group metrics
        metrics_df = evaluate_on_split(model, x_tr)

        # Energy weights
        w = _compute_group_energy_weights(
            df=x_tr, group_col=model.group_col, q_col=model.x_vars[0], interval_hours=interval_hours
        ).rename(columns={model.group_col: "group"})
        metrics_df = metrics_df.merge(w, on="group", how="left")

        # Out-of-sample diagnostics
        extras["pooled_co2"] = pooled_co2_metrics(
            model, x_tr, y_col=model.y_var, group_col=model.group_col
        )
        me_df = model.transform(x_tr)
        fd_df = finite_difference_me_metrics(
            df=me_df,
            time_col="timestamp" if "timestamp" in me_df.columns else "time_id",
            q_col=model.x_vars[0],
            y_col=model.y_var,
            me_col="ME",
            group_keys=[k for k in ("city",) if k in me_df.columns],
        )
        extras["fd_me_by_city"] = fd_df.to_dict(orient="records") if not fd_df.empty else []
        extras["fd_me_pooled"] = (
            fd_df.loc[fd_df["city"] == "ALL"].iloc[0].to_dict()
            if (not fd_df.empty and "city" in fd_df.columns and "ALL" in fd_df["city"].values)
            else (fd_df.sort_values("n_pairs", ascending=False).iloc[0].to_dict() if not fd_df.empty else {})
        )

    # Stamp metadata
    metrics_df["data_split"] = split_name
    metrics_df["model_id_hash"] = model_id_hash
    metrics_df["random_state"] = random_state
    metrics_df["pipeline_params_json"] = params_json_str
    metrics_df["log_time"] = datetime.now().isoformat()

    model = user_pipeline._final_estimator  # type: ignore[attr-defined]
    metrics_df["x_columns_used"] = ",".join(model.x_vars + model.fe_vars)
    for k, v in (extra_info or {}).items():
        metrics_df[k] = v

    x_cols_used = model.x_vars + model.fe_vars
    print(f"[LOG] {len(metrics_df)} rows for split={split_name}, model_id={model_id_hash}, random_state={random_state}")

    return (metrics_df, x_cols_used, model) if return_model else (metrics_df, x_cols_used, extras)
def regressor_orchestrator(
        user_pipeline: Pipeline,
        x_splits: dict,
        y_splits: dict,
        log_csv_path: str | None = "marginal_emissions_log.csv",   # legacy
        extra_info: dict | None = None,
        force_run: bool = False,
        force_overwrite: bool = False,
        random_state: int = 12,
        group_col_name: str = "group",
        interval_hours: float = 0.5,
        eval_splits: tuple[str, ...] | None = None,
        compute_test: bool = False,
        # rotating CSV
        results_dir: str | None = None,
        file_prefix: str | None = None,
        max_log_mb: int = 95,
        fsync: bool = True,
) -> pd.DataFrame | None:
    """
    Fit/evaluate a pipeline on train/validation/test, summarise metrics, and append to a CSV log.

    Parameters
    ----------
    user_pipeline : Pipeline
        Full pipeline, typically [FeatureAddition → Binner → GroupwiseRegressor].
    x_splits : dict
        Must include "train" and "validation". Include "test" iff compute_test=True.
    y_splits : dict
        Target splits with the same keys as x_splits.  Must include "train" and "validation". Include "test" iff compute_test=True.
     log_csv_path : str, optional
        Legacy path; used only to infer default results_dir/file_prefix if those are None.
    extra_info : dict, optional
        Extra metadata to stamp onto per-split logs (propagates into `run_regressor_model`).
    force_run : bool, default=False
        If False and an identical model signature was previously logged, skip this run.
    force_overwrite : bool, default=False
        If True, allows re-logging the same model_id_hash (previous rows are NOT removed here;
        use `save_summary_to_csv(..., force_overwrite=True)` for row replacement).
    random_state : int, default=12
        Random seed recorded in the model signature and summary.
    group_col_name : str, default="group"
        Canonical group column name used by `summarise_metrics_logs` for nested metrics.

    Returns
    -------
    pd.DataFrame or None
        One-row summary DataFrame if the run executes; None if skipped due to prior identical log.

    Notes
    -----
    - The model signature (hash) is computed from pipeline parameters, feature columns, target name(s),
      random_state, and any `extra_info`. If unchanged and `force_run=False`, the run is skipped.
    - `x_columns` recorded in the summary are taken from the **train** split’s evaluation result.
    """
    # in regressor_orchestrator before signature_for_run(...)
    if eval_splits is None:
        eval_splits = ("train","validation","test") if compute_test else ("train","validation")
    compute_test = ("test" in eval_splits)  # ← keep hash consistent with actual splits

    # One signature for the whole run (based on TRAIN)
    model_key, sig = signature_for_run(
        user_pipeline,
        x_columns=list(x_splits["train"].columns),
        y=y_splits["train"],
        random_state=random_state,
        eval_splits=eval_splits,
        compute_test=compute_test,
        extra_info=extra_info,
    )

    # Resolve dir + prefix (fallback to legacy path)
    if results_dir is None or file_prefix is None:
        base = Path(log_csv_path or "marginal_emissions_log.csv")
        inferred_dir = base.parent if str(base.parent) != "" else Path(".")
        inferred_prefix = base.stem
        results_dir = results_dir or str(inferred_dir)
        file_prefix = file_prefix or inferred_prefix

    # De-dupe via index
    if not force_run and not force_overwrite:
        if is_model_logged_rotating_csv(model_key, results_dir, file_prefix):
            print(f"[SKIP] Model already logged (hash: {model_key})")
            return None

    # Precompute params JSON once (consistent across splits)
    params_json_str = json.dumps(
        user_pipeline.get_params(deep=True),
        sort_keys=True, separators=(",", ":"), default=str
    )

    logs, pooled_extras, fd_extras = {}, {}, {}
    x_cols_used: list[str] | None = None

    for split in eval_splits:
        metrics_df, x_cols_used, extras = run_regressor_model(
            user_pipeline=user_pipeline,
            x_df=x_splits[split],
            y_df=y_splits[split],
            split_name=split,
            extra_info=extra_info,
            return_model=False,
            random_state=random_state,
            interval_hours=interval_hours,
            model_id_hash=model_key,          # shared ID across splits
            params_json_str=params_json_str,  # shared params JSON
        )
        logs[split] = metrics_df
        pooled_extras[split] = extras.get("pooled_co2", {})
        fd_extras[split] = extras.get("fd_me_pooled", {})

    summary_df = summarise_metrics_logs(
        train_logs=logs["train"],
        val_logs=logs["validation"],
        test_logs=logs.get("test"),
        user_pipeline=user_pipeline,
        x_columns=x_cols_used or [],
        random_state=random_state,
        group_col_name=group_col_name,           # <- use the parameter you accept
        pooled_metrics_by_split=pooled_extras,
        fd_me_metrics_by_split=fd_extras,
    )

    save_summary_to_rotating_csv(
        summary_df,
        results_dir=results_dir,
        file_prefix=file_prefix,
        max_mb=max_log_mb,
        force_overwrite=force_overwrite,
        fsync=fsync,
    )
    return summary_df
#### Grid Search
def run_grid_search(
        base_feature_pipeline: Pipeline,
        regressor_cls,
        regressor_kwargs: dict,
        grid_config: list[dict],
        x_splits: dict,
        y_splits: dict,
        log_path: str | None,  # legacy; optional now
        global_extra_info: dict | None = None,
        force_run: bool = False,
        force_overwrite: bool = False,
        base_feature_pipeline_name: str = "BaseFeaturePipeline",
        eval_splits: tuple[str, ...] = ("train","validation"),
        results_dir: str | None = None,
        file_prefix: str | None = None,
        max_log_mb: int = 95,
        fsync: bool = True,
) -> None:
    """
    Execute a series of [features → binner → regressor] runs and log one summary row per config.

    Parameters
    ----------
    base_feature_pipeline : Pipeline
        Preprocessing steps applied before binning. This object is cloned per run to avoid state leakage.
    regressor_cls : type
        Estimator class to instantiate for the final step (e.g., GroupwiseRegressor).
    regressor_kwargs : dict
        Baseline kwargs for the regressor. Per-config overrides from `grid_config` are merged on top.
        IMPORTANT: This function will not mutate the caller's dict.
    grid_config : list of dict
        Each item should contain:
            - "binner_class": class (e.g., MultiQuantileBinner or MultiMedianBinner)
            - "binner_kwargs": dict of init args for the binner
            - "label": str label for printing/logging (optional)
            - Optional: "x_vars", "fe_vars" to override the regressor’s predictors per-config
            - Optional: anything else you want echoed into `extra_info`
    x_splits, y_splits : dict
        Dicts keyed by {"train","validation","test"} with DataFrames/Series for each split.
    log_path : str
        CSV path where each successful config appends one summary row.
    global_extra_info : dict, optional
        Extra metadata stamped into each run’s logs.
    force_run, force_overwrite : bool
        Passed through to `regressor_orchestrator`.
    base_feature_pipeline_name : str, default "BaseFeaturePipeline"
        Step name used for the features sub-pipeline.

    Returns
    -------
    None
        Prints progress and writes rows to `log_path`. Skips silently (with a message) if a config
        is already logged and `force_run=False`.

    Notes
    -----
    - We clone `base_feature_pipeline` per run to avoid cross-config state sharing.
    - If a binner provides `group_col_name` and the regressor does not specify `group_col`,
      we set the regressor’s `group_col` to match.
    - If a config provides `x_vars`/`fe_vars`, they override the baseline `regressor_kwargs`.
    """
    missing_x = [s for s in eval_splits if s not in x_splits]
    missing_y = [s for s in eval_splits if s not in y_splits]
    if missing_x or missing_y:
        raise KeyError(f"Missing splits: X{missing_x} Y{missing_y}")

    total = len(grid_config)
    for i, raw_config in enumerate(grid_config, start=1):
        config = dict(raw_config)
        binner_class = config["binner_class"]
        binner_kwargs = dict(config.get("binner_kwargs", {}))
        label = config.get("label", binner_class.__name__)

        reg_kwargs = dict(regressor_kwargs)
        if "x_vars" in config:
            reg_kwargs["x_vars"] = list(config["x_vars"])
        if "fe_vars" in config:
            reg_kwargs["fe_vars"] = list(config["fe_vars"])
        reg_kwargs["random_state"] = reg_kwargs.get("random_state", 12)

        binner_group_col = binner_kwargs.get("group_col_name")
        if binner_group_col and "group_col" not in reg_kwargs:
            reg_kwargs["group_col"] = binner_group_col

        try:
            features_step = clone(base_feature_pipeline)
        except Exception:
            features_step = base_feature_pipeline

        binner = binner_class(**binner_kwargs)
        regressor = regressor_cls(**reg_kwargs)

        full_pipeline = Pipeline([
            (base_feature_pipeline_name, features_step),
            (binner_class.__name__, binner),
            (regressor_cls.__name__, regressor),
        ])

        extra_info = {
            "binner_class": binner_class.__name__,
            "binner_params": binner_kwargs,
            "regressor_params": reg_kwargs,
            "grid_label": label,
            **(global_extra_info or {}),
        }

        rank_tag = ""
        try:
            _, rank, size = _mpi_context()
            rank_tag = f"[R{rank}/{max(size-1,0)}] "
        except Exception:
            pass
        print(f"\n{rank_tag}[GRID {i}/{total}] {label}")

        try:
            summary_df = regressor_orchestrator(
                user_pipeline=full_pipeline,
                x_splits=x_splits,
                y_splits=y_splits,
                log_csv_path=log_path,            # legacy OK
                extra_info=extra_info,
                force_run=force_run,
                force_overwrite=force_overwrite,
                random_state=reg_kwargs["random_state"],
                eval_splits=eval_splits,
                # NEW
                results_dir=results_dir,
                file_prefix=file_prefix,
                max_log_mb=max_log_mb,
                fsync=fsync,
                )
            if summary_df is not None:
                print(f"[GRID] Logged: {label}")
            else:
                print(f"[GRID] Skipped (already logged): {label}")
        except Exception as e:
            print(f"[GRID] ERROR in '{label}': {type(e).__name__}: {e}")
            continue
def run_grid_search_auto(
        base_feature_pipeline,
        regressor_cls,
        regressor_kwargs: dict,
        grid_config: list[dict],
        x_splits: dict,
        y_splits: dict,
        *,
        # logging/rotation knobs
        results_dir: str,
        file_prefix: str,
        max_log_mb: int = 95,
        fsync: bool = False,              # set True on HPC if you want durable writes
        # orchestration
        base_feature_pipeline_name: str = "FeatureAdditionPipeline",
        eval_splits: tuple[str, ...] = ("train","validation"),
        force_run: bool = False,
        force_overwrite: bool = False,
        distribute: str = "auto",         # "auto" | "mpi" | "single"
        dist_mode: str = "stride",        # "stride" | "chunked"
        seed: int = 12,
) -> None:
    """
    Single-node or MPI-parallel grid search runner.

    - Auto-detects MPI and splits `grid_config` across ranks.
    - Ensures per-rank deterministic RNG via `seed + rank`.
    - Uses rotating CSV logging with per-file & index locks.

    Parameters are passed straight to `run_grid_search`, except we slice `grid_config`.

    Parameters
    ----------
    base_feature_pipeline: Pipeline
        The base feature pipeline to use for each config.
    regressor_cls: Type[BaseEstimator]
        The regression model class to use.
    regressor_kwargs: dict
        Keyword arguments to pass to the regression model.
    grid_config: list[dict]
        The grid search configuration to use.
    x_splits: dict
        The input feature splits.
    y_splits: dict
        The target variable splits.
    results_dir: str
        The directory to save results.
    file_prefix: str
        The prefix for result files.
    max_log_mb: int
        The maximum log file size in MB.
    naming: PartNaming | None
        Optional naming scheme for output files.
    fsync: bool
        Whether to fsync log files (for durability).
    base_feature_pipeline_name: str
        The name of the base feature pipeline.
    eval_splits: tuple[str, ...]
        The evaluation splits to use.
    force_run: bool
        Whether to force re-running of existing configs.
    force_overwrite: bool
        Whether to force overwriting of existing results.
    distribute: str
        The distribution strategy to use.
    dist_mode: str
        The distribution mode to use.
    seed: int
        The random seed to use.

    Returns
    -------
    None
        Logs the results of the grid search.
    """
    comm, rank, size = _mpi_context()
    if distribute == "auto":
        distribute = "mpi" if size > 1 else "single"

    # Partition the configs
    local_configs = _distribute_configs(grid_config, rank=rank, size=size, mode=dist_mode) \
                    if distribute == "mpi" else grid_config
    if not local_configs:
        if rank == 0:
            print("[GRID] No configs assigned (empty grid or partition).")
        return

    # Per-rank RNG — override/augment existing random_state
    local_reg_kwargs = dict(regressor_kwargs)
    local_reg_kwargs["random_state"] = int(local_reg_kwargs.get("random_state", seed))

    if rank == 0 and distribute == "mpi":
        print(f"[MPI] size={size} → ~{len(grid_config)/max(size,1):.1f} configs per rank")
    else:
        if distribute == "mpi":
            print(f"[MPI] rank={rank}/{size-1} assigned {len(local_configs)} configs")

    run_grid_search(
        base_feature_pipeline=base_feature_pipeline,
        regressor_cls=regressor_cls,
        regressor_kwargs=local_reg_kwargs,
        grid_config=local_configs,
        x_splits=x_splits,
        y_splits=y_splits,
        log_path=None,  # legacy path unused when using rotating logs
        global_extra_info={"runner_rank": rank, "runner_size": size},
        force_run=force_run,
        force_overwrite=force_overwrite,
        base_feature_pipeline_name=base_feature_pipeline_name,
        eval_splits=eval_splits,
        results_dir=results_dir,
        file_prefix=file_prefix,
        max_log_mb=max_log_mb,
        fsync=fsync,
    )

    # Optional barrier for neat logs
    try:
        comm.Barrier()
    except Exception:
        pass
    if rank == 0:
        print("[GRID] Completed (all ranks).")

def all_nonempty_subsets(columns: list[str]) -> list[list[str]]:
    """All non-empty subsets preserving input order."""
    return [list(c) for i in range(1, len(columns) + 1) for c in combinations(columns, i)]
def get_fe_vars(all_cols: list[str], x_vars: list[str]) -> list[str]:
    """Complement of x_vars within all_cols."""
    xset = set(x_vars)
    return [c for c in all_cols if c not in xset]
def build_x_fe_combinations_disjoint(
    candidate_x_vars: list[str],
    candidate_fe_vars: list[str],
    x_var_length: int = 2,
    max_fe_len: int | None = None,
    *,
    allow_empty_fe: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate all disjoint non-empty combinations of x_vars and fe_vars.

    Parameters
    ----------
    candidate_x_vars : list of str
        Columns eligible to be used as predictors (x_vars).
    candidate_fe_vars : list of str
        Columns eligible to be used as fixed effects (fe_vars).
    x_var_length : int
        Number of x_vars to include in each combination.
    max_fe_len : int | None
        Maximum number of fe_vars to include in each combination.
    allow_empty_fe : bool
        Whether to allow empty fe_vars in the combinations.

    Returns
    -------
    list of dicts
        Each dict has keys: {'x_vars': [...], 'fe_vars': [...]}
    """
    if x_var_length < 1:
        raise ValueError("x_var_length must be >= 1")
    if len(candidate_x_vars) < x_var_length:
        raise ValueError("Not enough candidate_x_vars for requested x_var_length")

    results: list[dict[str, Any]] = []

    x_subsets = [list(c) for c in combinations(candidate_x_vars, x_var_length)]
    fe_pool = [list(c) for i in range(0 if allow_empty_fe else 1, len(candidate_fe_vars) + 1)
               for c in combinations(candidate_fe_vars, i)]

    for x_vars in x_subsets:
        for fe_vars in fe_pool:
            if max_fe_len is not None and len(fe_vars) > max_fe_len:
                continue
            if set(x_vars).isdisjoint(fe_vars):
                results.append({"x_vars": x_vars, "fe_vars": list(fe_vars)})
    return results
def build_quantile_grid_configs(
        candidate_binning_vars: list[str],
        candidate_bin_counts: list[int],
        candidate_x_vars: list[str],
        candidate_fe_vars: list[str],
        x_var_length: int = 2,
        binner_extra_grid: dict | list[dict] | None = None,
) -> list[dict[str, Any]]:
    """
    Produce configs for MultiQuantileBinner sweeping:
      - which vars to bin on
      - how many bins
      - x/fe combinations (disjoint from binned vars)
      - optional extra binner kwargs via dict-of-lists or list-of-dicts

    Parameters
    ----------
    candidate_binning_vars : list[str]
        Variables to be binned.
    candidate_bin_counts : list[int]
        Number of bins to create for each variable.
    candidate_x_vars : list[str]
        Variables to use as predictors (x_vars).
    candidate_fe_vars : list[str]
        Variables to use as fixed effects (fe_vars).
    x_var_length : int
        Number of x_vars to include in each combination.
    binner_extra_grid : dict | list[dict] | None
        Optional extra parameters for the binner.

    Returns
    -------
    list[dict[str, Any]]
        A list of configuration dictionaries for the binner.
    """
    if not candidate_binning_vars:
        return []
    if not candidate_bin_counts:
        return []

    def _expand(grid):
        if grid is None:
            return [dict()]
        if isinstance(grid, list):
            return [dict(d) for d in grid]
        if isinstance(grid, dict):
            keys = list(grid.keys())
            vals = [list(v) if isinstance(v, (list, tuple, set)) else [v] for v in (grid[k] for k in keys)]
            return [dict(zip(keys, combo)) for combo in product(*vals)]
        raise TypeError("binner_extra_grid must be a dict or list of dicts")

    extra_list = _expand(binner_extra_grid)
    configs: list[dict[str, Any]] = []

    # compute once (perf)
    x_fe_grid = build_x_fe_combinations_disjoint(
        candidate_x_vars, candidate_fe_vars, x_var_length=x_var_length
    )

    for bin_vars in all_nonempty_subsets(candidate_binning_vars):
        bset = set(bin_vars)
        for bin_count in candidate_bin_counts:
            if int(bin_count) < 2:
                continue
            bin_spec = {v: int(bin_count) for v in bin_vars}

            for combo in x_fe_grid:
                if not set(combo["x_vars"]).isdisjoint(bset):
                    continue
                for extra in extra_list:
                    binner_kwargs = {"bin_specs": bin_spec, **extra}

                    # label suffix for clarity in logs
                    tag_bits = []
                    pol = extra.get("oob_policy")
                    if pol: tag_bits.append(f"oob{pol}")
                    rate = extra.get("max_oob_rate")
                    if rate is not None: tag_bits.append(f"rate{float(rate):g}")
                    tag = f"__{'_'.join(tag_bits)}" if tag_bits else ""

                    configs.append({
                        "binner_class": MultiQuantileBinner,
                        "binner_kwargs": binner_kwargs,
                        "label": (
                            f"qbin_{bin_count}_{'-'.join(bin_vars)}"
                            f"__x_{'-'.join(combo['x_vars'])}"
                            f"__fe_{'-'.join(combo['fe_vars'])}{tag}"
                        ),
                        "x_vars": combo["x_vars"],
                        "fe_vars": combo["fe_vars"],
                    })
    return configs


def build_median_binner_configs(
    candidate_binning_vars: list[str],
    candidate_x_vars: list[str],
    candidate_fe_vars: list[str],
    x_var_length: int = 2,
    max_fe_len: int | None = None,
    binner_extra_grid: dict | list[dict] | None = None,
) -> list[dict[str, Any]]:
    """
    Produce configs for MultiMedianBinner sweeping subsets of variables and x/fe combos.

    Parameters
    ----------
    candidate_binning_vars : list[str]
        Variables to be binned.
    candidate_x_vars : list[str]
        Variables to use as predictors (x_vars).
    candidate_fe_vars : list[str]
        Variables to use as fixed effects (fe_vars).
    x_var_length : int
        Number of x_vars to include in each combination.
    max_fe_len : int | None
        Maximum number of fixed effects to include in each combination.
    binner_extra_grid : dict | list[dict] | None
        Optional extra parameters for the binner.

    Returns
    -------
    list[dict[str, Any]]
        A list of configuration dictionaries for the binner.
    """
    if not candidate_binning_vars:
        return []

    def _expand(grid):
        if grid is None:
            return [dict()]
        if isinstance(grid, list):
            return [dict(d) for d in grid]
        if isinstance(grid, dict):
            keys = list(grid.keys())
            vals = [ (v if isinstance(v, (list, tuple, set)) else [v]) for v in grid.values() ]
            return [dict(zip(keys, combo)) for combo in product(*vals)]
        raise TypeError("binner_extra_grid must be a dict or list of dicts")

    extra_list = _expand(binner_extra_grid)

    configs: list[dict[str, Any]] = []
    x_fe_grid = build_x_fe_combinations_disjoint(
        candidate_x_vars, candidate_fe_vars, x_var_length=x_var_length, max_fe_len=max_fe_len
    )

    for bin_vars in all_nonempty_subsets(candidate_binning_vars):
        bset = set(bin_vars)
        for combo in x_fe_grid:
            if not set(combo["x_vars"]).isdisjoint(bset):
                continue
            for extra in extra_list:
                binner_kwargs = {
                    "variables": bin_vars,
                    "group_col_name": "median_group_id",
                    "retain_flags": True,
                    **extra,
                }
                tag_bits = []
                if "retain_flags" in extra:
                    tag_bits.append(f"rf{int(bool(extra['retain_flags']))}")
                for k, v in extra.items():
                    if k == "retain_flags":
                        continue
                    tag_bits.append(f"{k}{v}")
                tag = f"__{'_'.join(tag_bits)}" if tag_bits else ""

                configs.append({
                    "binner_class": MultiMedianBinner,
                    "binner_kwargs": binner_kwargs,
                    "label": (
                        f"median_{'-'.join(bin_vars)}"
                        f"__x_{'-'.join(combo['x_vars'])}"
                        f"__fe_{'-'.join(combo['fe_vars'])}{tag}"
                    ),
                    "x_vars": combo["x_vars"],
                    "fe_vars": combo["fe_vars"],
                })
    return configs





train_pldf = pl.read_parquet(train_filepath)
validation_pldf = pl.read_parquet(validation_filepath)
test_pldf = pl.read_parquet(test_filepath)



# CREATING SHARES
train_pldf_enhanced = train_pldf.with_columns(
    (pl.col("thermal_generation") / pl.col("total_generation")).alias("thermal_share"),
    (pl.col("gas_generation") / pl.col("total_generation")).alias("gas_share"),
    (pl.col("hydro_generation") / pl.col("total_generation")).alias("hydro_share"),
    (pl.col("nuclear_generation") / pl.col("total_generation")).alias("nuclear_share"),
    (pl.col("renewable_generation") / pl.col("total_generation")).alias("renewable_share"),
)

validation_pldf_enhanced = validation_pldf.with_columns(
    (pl.col("thermal_generation") / pl.col("total_generation")).alias("thermal_share"),
    (pl.col("gas_generation") / pl.col("total_generation")).alias("gas_share"),
    (pl.col("hydro_generation") / pl.col("total_generation")).alias("hydro_share"),
    (pl.col("nuclear_generation") / pl.col("total_generation")).alias("nuclear_share"),
    (pl.col("renewable_generation") / pl.col("total_generation")).alias("renewable_share"),
)

test_pldf_enhanced = test_pldf.with_columns(
    (pl.col("thermal_generation") / pl.col("total_generation")).alias("thermal_share"),
    (pl.col("gas_generation") / pl.col("total_generation")).alias("gas_share"),
    (pl.col("hydro_generation") / pl.col("total_generation")).alias("hydro_share"),
    (pl.col("nuclear_generation") / pl.col("total_generation")).alias("nuclear_share"),
    (pl.col("renewable_generation") / pl.col("total_generation")).alias("renewable_share"),
)

# print(f"Column Count before adding shares:\t Train: {train_pldf.shape[1]}, Validation: {validation_pldf.shape[1]}, Test: {test_pldf.shape[1]}")
# print(f"Column Count after adding shares:\t Train: {train_pldf_enhanced.shape[1]}, Validation: {validation_pldf_enhanced.shape[1]}, Test: {test_pldf_enhanced.shape[1]}")
# print(f"Columns in Train PLDataFrame after adding shares: {train_pldf_enhanced.columns}")
# # CREATING DEMAND MINUS RENEWABLES
# print(f"Column Count before adding demand minus renewables:\t Train: {train_pldf_enhanced.shape[1]}, Validation: {validation_pldf_enhanced.shape[1]}, Test: {test_pldf_enhanced.shape[1]}")

train_pldf_enhanced = train_pldf_enhanced.with_columns(
    (pl.col("demand_met") - pl.col("renewable_generation")).alias("demand_minus_renewables"),
)

validation_pldf_enhanced = validation_pldf_enhanced.with_columns(
    (pl.col("demand_met") - pl.col("renewable_generation")).alias("demand_minus_renewables"),
)

test_pldf_enhanced = test_pldf_enhanced.with_columns(
    (pl.col("demand_met") - pl.col("renewable_generation")).alias("demand_minus_renewables"),
)
# print(f"Column Count after adding demand minus renewables:\t Train: {train_pldf_enhanced.shape[1]}, Validation: {validation_pldf_enhanced.shape[1]}, Test: {test_pldf_enhanced.shape[1]}")
# print(f"Columns in Train PLDataFrame after adding demand minus renewables: {train_pldf_enhanced.columns}")

# # CREATING 'IS_SUNNY' FLAG
# print(f"Column Count before adding is_sunny_flag:\t Train: {train_pldf_enhanced.shape[1]}, Validation: {validation_pldf_enhanced.shape[1]}, Test: {test_pldf_enhanced.shape[1]}")

train_pldf_enhanced = train_pldf_enhanced.with_columns(
    (pl.col("surface_net_solar_radiation_kWh_per_m2").fill_nan(0) > 1e-4).cast(pl.Int8).alias("is_sunny"),
)

validation_pldf_enhanced = validation_pldf_enhanced.with_columns(
    (pl.col("surface_net_solar_radiation_kWh_per_m2").fill_nan(0) > 1e-4).cast(pl.Int8).alias("is_sunny"),
)

test_pldf_enhanced = test_pldf_enhanced.with_columns(
    (pl.col("surface_net_solar_radiation_kWh_per_m2").fill_nan(0) > 1e-4).cast(pl.Int8).alias("is_sunny"),
)
# print(f"Column Count after adding is_sunny_flag:\t Train: {train_pldf_enhanced.shape[1]}, Validation: {validation_pldf_enhanced.shape[1]}, Test: {test_pldf_enhanced.shape[1]}")
# print(f"Columns in Train PLDataFrame after adding is_sunny_flag: {train_pldf_enhanced.columns}")
# Conversion to Pandas DataFrame for compatibility with existing code
train_df = train_pldf_enhanced.to_pandas()
validation_df = validation_pldf_enhanced.to_pandas()
test_df = test_pldf_enhanced.to_pandas()
drop_columns_not_needed = [
        # Weather Vars
        "wind_dir_cardinal_8", "wind_dir_cardinal_4", "wind_dir_cardinal_16",
        "low_cloud_cover", "medium_cloud_cover", "high_cloud_cover",
        "surface_solar_radiation_downwards_kWh_per_m2", "surface_net_solar_radiation_joules_per_m2", "surface_solar_radiation_downwards_joules_per_m2",
        # Grid Vars
        "g_co2_per_kwh", "tons_co2_per_mwh",
        "total_generation",
        ]
# print(f"Count of Columns in Train Set pre drop: {train_df.shape[1]}")

train_df_reduced = train_df.drop(columns=drop_columns_not_needed, errors='ignore')
validation_df_reduced = validation_df.drop(columns=drop_columns_not_needed, errors='ignore')
test_df_reduced = test_df.drop(columns=drop_columns_not_needed, errors='ignore')

# print(f"Remaining columns after dropping: {train_df_reduced.columns.tolist()}")
# print(f"Count of Columns in Train Set: {train_df_reduced.shape[1]}\t\tValidation Set: {validation_df_reduced.shape[1]}\t\tTest Set: {test_df_reduced.shape[1]}")
X_train = train_df_reduced.drop(columns=["tons_co2"])
y_train = train_df_reduced["tons_co2"]
X_val = validation_df_reduced.drop(columns=["tons_co2"])
y_val = validation_df_reduced["tons_co2"]
X_test = test_df_reduced.drop(columns=["tons_co2"])
y_test = test_df_reduced["tons_co2"]



# this pipeline is used to transform features into more usable formats for modelling
# the purpose of the pipeline is to generate a set of features that can be chosen for modelling
# not every feature generated witll necessarily be used in the final model
feature_engineering_standard_scaling = [
    ("dt", DateTimeFeatureAdder(timestamp_col="timestamp",
                                add=("month","week", "doy","hour",),
                                drop_original=False, verbose=False)),
    ("tfourier", TimeFourierAdder(timestamp_col="timestamp",
                                  add_month=True, add_week=True, add_doy=True,  add_hour=True, add_weekend=True,
                                  verbose=False)),
    ("wind_cyc", WindDirToCyclic(dir_col="wind_direction_meteorological",
                                 out_sin="wind_dir_sin", out_cos="wind_dir_cos",
                                 convention="met", drop_original=True, verbose=False)),
    ("precip_2part", ZeroInflatedLogTransform(columns=["precipitation_mm"], threshold=0.0,
                                              occ_suffix="_occ", log_suffix="_log1p",
                                              drop_original=False, verbose=False)),
    ("solar_log", Log1pTransform(columns=["surface_net_solar_radiation_kWh_per_m2"],
                                 out_suffix="_log1p", clamp_lower=0.0,
                                 drop_original=False, verbose=False)),
    ("cloud_logit", BoundedToLogit(columns=["total_cloud_cover"], eps=1e-4,
                                   out_suffix="_logit", drop_original=False, verbose=False)),
    # Standardize demand (for chain-rule) and all driver cols we want for linear models
    ("std", StandardizeContinuous(
        columns=[
            # demand variables
            "demand_met",
            "demand_minus_renewables",

            # weather drivers
            "temperature_celsius",
            "wind_speed_mps",
            "surface_net_solar_radiation_kWh_per_m2",
            "surface_net_solar_radiation_kWh_per_m2_log1p",
            "total_cloud_cover",
            "total_cloud_cover_logit",
            "precipitation_mm",
            "precipitation_mm_log1p",

            # Grid

            # grid generation - raw
            "thermal_generation", "gas_generation",
            "hydro_generation", "nuclear_generation",
            "renewable_generation", "non_renewable_generation",

            # grid generation - shares
            "thermal_share", "gas_share",
            "hydro_share", "nuclear_share",
            "renewable_share",
        ],
        suffix="_std", strategy="standard", with_center=True, with_scale=True,
        drop_original=False, verbose=False
    )),
    ("std_drv", StandardizeContinuous(columns=[
            # weather drivers
            "temperature_celsius",
            "wind_speed_mps",
            "surface_net_solar_radiation_kWh_per_m2",
            "surface_net_solar_radiation_kWh_per_m2_log1p",
            "total_cloud_cover",
            "total_cloud_cover_logit",
            "precipitation_mm",
            "precipitation_mm_log1p",

            # Grid

            # grid generation - raw
            "thermal_generation", "gas_generation",
            "hydro_generation", "nuclear_generation",
            "renewable_generation", "non_renewable_generation",

            # grid generation - shares
            "thermal_share", "gas_share",
            "hydro_share", "nuclear_share",
            "renewable_share",
        ],
      suffix="_rbst", strategy="robust", with_center=True, with_scale=True)),
]



# Updated QGAM model
from joblib import Parallel, delayed

# - Q spline - smooth weather - linear shares and time vars
q_col, q_std_col = "demand_minus_renewables", "demand_minus_renewables_std"

time_features = [ "doy_sin","doy_cos", "hour_sin","hour_cos", "is_weekend",]

smooth_weather = [
    "temperature_celsius_std",
    "wind_speed_mps_std",
    "surface_net_solar_radiation_kWh_per_m2_log1p_std",
]
linear_cols = [
    "hydro_share_std",
    "wind_dir_sin","wind_dir_cos",
    "is_sunny",
] + time_features

# grids
lam_q_grid    = (5,10, 20, 50)
lam_temp_grid = (1, 5,10, 20, 50)
lam_wind_grid = (1, 5,10, 20, 50)
lam_sol_grid  = (1, 5,10, 20, 50)
nsp_q_grid    = (5,10, 20, 40)
nsp_weather   = 20

min_abs_dq_100  = 100.0
min_abs_dq_1000 = 1000.0
params = list(product(nsp_q_grid, lam_q_grid, lam_temp_grid, lam_wind_grid, lam_sol_grid))
combinations = len(params)
print(combinations)


import pygam as pg
s = pg.s     # restore the 's' symbol to the pygam spline constructor
# l = pg.l   # (do this too if you also use pygam.l anywhere)

results = []

for nsp_q, lam_q, lam_temp, lam_wind, lam_solar in params:
    print(f"\tstarting {nsp_q},{lam_q},{lam_temp},{lam_wind},{lam_solar}", flush=True)

    # Build the pipeline (feature_engineering_standard_scaling must CREATE the *_std cols)
    pipe = Pipeline(steps=feature_engineering_standard_scaling + [
        ("pygam_level", QGAMRegressorPyGAM(
            y_var="tons_co2",
            q_col="demand_minus_renewables",
            q_std_col="demand_minus_renewables_std",
            # placeholders; we'll overwrite after we inspect the columns post-transform
            smooth_cols=smooth_weather,
            linear_cols=linear_cols,
            n_splines_q=int(nsp_q), lam_q=float(lam_q),
            missing_action="warn", verbose=False, random_state=12
        ))
    ], memory=None)

    # --- run just the FE/scale step to see what's actually produced
    Xtr_feat = Pipeline(pipe.steps[:-1]).fit_transform(X_train, y_train)
    present = set(Xtr_feat.columns if hasattr(Xtr_feat, "columns") else Xtr_feat.dtype.names or [])

    # keep only features that exist
    smooth_weather_present = [c for c in [
        "temperature_celsius_std",
        "wind_speed_mps_std",
        "surface_net_solar_radiation_kWh_per_m2_log1p_std",
    ] if c in present]

    linear_cols_present = [c for c in [
        "hydro_share_std",          # ensure your FE step actually creates this exact name
        "wind_dir_sin","wind_dir_cos",
        "is_sunny",
        "doy_sin","doy_cos","hour_sin","hour_cos","is_weekend",
    ] if c in present]

    # per-term settings filtered to present cols
    lam_by_col = {
        "temperature_celsius_std": lam_temp,
        "wind_speed_mps_std": lam_wind,
        "surface_net_solar_radiation_kWh_per_m2_log1p_std": lam_solar,
    }
    lam_by_col = {k: v for k, v in lam_by_col.items() if k in present}

    n_splines_by_col = {
        "temperature_celsius_std": 20,
        "wind_speed_mps_std": 20,
        "surface_net_solar_radiation_kWh_per_m2_log1p_std": 20,
    }
    n_splines_by_col = {k: v for k, v in n_splines_by_col.items() if k in present}

    # update estimator params now that we know what's available
    gam_est = pipe.named_steps["pygam_level"]
    gam_est.smooth_cols = smooth_weather_present
    gam_est.linear_cols = linear_cols_present
    gam_est.n_splines_by_col = n_splines_by_col
    gam_est.lam_by_col = lam_by_col

    missing_warn = sorted(set(
        ["temperature_celsius_std","wind_speed_mps_std",
         "surface_net_solar_radiation_kWh_per_m2_log1p_std",
         "hydro_share_std","wind_dir_sin","wind_dir_cos","is_sunny",
         "doy_sin","doy_cos","hour_sin","hour_cos","is_weekend",
         "demand_minus_renewables","demand_minus_renewables_std"]
    ) - present)
    if missing_warn:
        print("[WARN] missing engineered columns:", missing_warn, flush=True)

    # (optional) align validation columns to train’s engineered columns
    def align_df(df, cols):
        for c in cols:
            if c not in df:
                df[c] = 0.0
        return df[cols]

    # Finish fit on full pipe
    _ = pipe.fit(X_train, y_train)

    # Evaluate (pipe handles its own transforms)
    ev_tr   = evaluate_pipeline(pipe, X_train, y_train, eval_me_slopes=False)
    ev_100  = evaluate_pipeline(pipe, X_val, y_val, eval_me_slopes=True,
                                me_slope_kwargs={"q_col": "demand_minus_renewables",
                                                 "max_dt": pd.Timedelta("30min"),
                                                 "min_abs_dq": 100.0})
    ev_1000 = evaluate_pipeline(pipe, X_val, y_val, eval_me_slopes=True,
                                me_slope_kwargs={"q_col": "demand_minus_renewables",
                                                 "max_dt": pd.Timedelta("30min"),
                                                 "min_abs_dq": 1000.0})

    def pull_all(diag: dict):
        pears = diag.get("pearson_r", [])
        spears = diag.get("spearman_r", [])
        cities = diag.get("city", [])
        if cities and "ALL" in cities:
            i = cities.index("ALL")
            return float(pears[i]), float(spears[i])
        return (float(pears[-1]) if pears else float("nan"),
                float(spears[-1]) if spears else float("nan"))

    p100, s100   = pull_all(ev_100.get("me_slope_diagnostics", {}))
    p1000, s1000 = pull_all(ev_1000.get("me_slope_diagnostics", {}))

    dfp = ev_100["predictions"].dropna(subset=["y_true","y_pred"]).copy()
    if "demand_met" in dfp:
        w = dfp["demand_met"].to_numpy(float) * 0.5
        w = w / w.sum() if w.sum() > 0 else np.ones(len(dfp), float) / len(dfp)
    else:
        w = np.ones(len(dfp), float) / len(dfp)
    y  = dfp["y_true"].to_numpy(float)
    yh = dfp["y_pred"].to_numpy(float)
    ybar = (w*y).sum()
    sse = (w*(y - yh)**2).sum()
    sst = (w*(y - ybar)**2).sum()
    r2_energy_100 = 1.0 - sse/sst if sst > 0 else float("nan")

    # optional: edof to sanity-check smoothness
    try:
        gam = pipe.named_steps["pygam_level"].gam_
        edof_total = float(np.sum(gam.statistics_["edof"]))

    except Exception:
        edof_total = float("nan")

    results.append({
        "model": "QGAM",
        "q": q_col, "q_std": q_std_col,
        "n_splines_q": nsp_q,
        "lam_q": lam_q,
        "lam_temp": lam_temp,
        "lam_wind": lam_wind,
        "lam_solar": lam_solar,
        "edof_total": edof_total,
        "r2_train": ev_tr["overall"].get("r2", np.nan),
        "r2_val_100": ev_100["overall"].get("r2", np.nan),
        "rmse_val_100": ev_100["overall"].get("rmse", np.nan),
        "mae_val_100": ev_100["overall"].get("mae", np.nan),
        "r2_val_energy_100": r2_energy_100,
        "pearson_ALL_100": p100,
        "spearman_ALL_100": s100,
        "r2_val_1000": ev_1000["overall"].get("r2", np.nan),
        "rmse_val_1000": ev_1000["overall"].get("rmse", np.nan),
        "mae_val_1000": ev_1000["overall"].get("mae", np.nan),
        "pearson_ALL_1000": p1000,
        "spearman_ALL_1000": s1000,
    })
    print(f"Finished QGAM model with nsp_q={nsp_q}, lam_q={lam_q}, lam_temp={lam_temp}, lam_wind={lam_wind}, lam_solar={lam_solar}", flush=True)

qgam_results_df = pd.DataFrame(results)


# write results to CSV
qgam_results_df.to_csv(os.path.join(marginal_emissions_development_directory,"qgam_results.csv"), index=False)
