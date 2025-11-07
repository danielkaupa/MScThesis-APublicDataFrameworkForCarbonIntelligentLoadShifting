# ─────────────────────────────────────────────────────────────────────────────
# FILE: step4_marginal_emissions_logging.py
#
# PURPOSE:
#   Infrastructure utilities for experiment logging and coordination.
#   Handles rotating CSV logs, unique run signatures, safe multi-process
#   writes, and MPI-based sharding/distribution.
#
# SECTIONS IN THIS FILE
#   - General Utilities: small helpers (file size, etc.).
#   - Logging & Identification: build run signatures, model hashes, skip dups.
#   - Naming Conventions: filename/part helpers for rotating logs.
#   - Path & Directory Management: ensure dirs, build paths, roll CSV parts.
#   - CSV File Handling (rotating log): append summaries, load, drop entries.
#   - MPI Management: sharding, file locks, MPI comm helpers.
#
# RUN REQUIREMENTS:
#   - Python 3.10+ recommended
#   - Dependencies: pandas, numpy, hashlib, pathlib, logging
#   - Optional: mpi4py (for distributed runs)
#   - File system access with write permissions for log directories
#
# FUNCTION LIST:
# * _abs_join
# * _distribute_configs
# * _drop_hash_from_part
# * _file_lock
# * _file_size_mb
# * _index_lock_path
# * _index_path
# * _list_part_files
# * _mpi_context
# * _next_csv_part_path
# * _read_index
# * _roll_if_needed
# * _y_columns_for_signature
# * allocate_next_part
# * is_model_logged_rotating_csv
# * load_all_logs_rotating_csv
# * load_existing_hashes
# * make_config_key
# * remove_model_from_rotating_csv
# * save_summary_to_rotating_csv
# * shard
# * signature_for_run

# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import json
import os
import random
from datetime import datetime
import re
import hashlib
import time
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from typing import Any, Mapping, Tuple

try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK, SIZE = 0, 1


# General Utilities
# ────────────────────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────────────────────
# UTILITIES - LOGGING & IDENTIFICATION
# ────────────────────────────────────────────────────────────────────────────

# Functions that assist with logging and tracking
# - load_existing_hashes
# - make_config_key
# - signature_for_run
# - _y_columns_for_signature
# ────────────────────────────────────────────────────────────────────────────

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
            return {str(k): _norm(v) for k, v in sorted(x.items(),
                                                        key=lambda kv:
                                                        str(kv[0]))}
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
    Build a stable config mapping for a model run and return (hash_key,
    mapping).

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
            raise ValueError("y must be a Series or single-column DataFrame"
                             " for signature.")
        return [str(y.columns[0])]
    name = getattr(y, "name", None)
    return [str(name)] if name is not None else ["y"]

# ────────────────────────────────────────────────────────────────────────────
# UTILITIES - NAMING CONVENTIONS
# ────────────────────────────────────────────────────────────────────────────

# Functions that assist with naming conventions and identification of unique
# parts
# class: PartNaming
# - format
# - split
# ────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PartNaming:
    token: str = ".part"   # separator between stem and index
    width: int = 3         # zero-pad width
    ext: str = ".csv"      # file extension, with leading dot

    def format(
            self,
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

    def split(
            self,
            name: str,
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


# ────────────────────────────────────────────────────────────────────────────
# UTILITIES - PATH AND DIRECTORY MANAGEMENT
# ────────────────────────────────────────────────────────────────────────────

# Functions that assist with path and directory management as models are run
# and logged:
# * _abs_join
# * _ensure_dir
# * _index_lock_path
# * _index_path
# * _next_csv_part_path
# * _roll_if_needed
# ────────────────────────────────────────────────────────────────────────────

def _abs_join(
        root: str,
        maybe_rel: str
) -> str:
    """
    Join to root if path is relative; return as-is if already absolute.
    This is a convenience function to ensure paths are absolute

    Parameters
    ----------
    root : str
        The root directory to join with.
    maybe_rel : str
        The path to join (may be relative).

    Returns
    -------
    str
        The absolute path.
    """
    return maybe_rel if os.path.isabs(maybe_rel
                                      ) else os.path.join(root, maybe_rel)


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


def _index_lock_path(index_path: Path) -> Path:
    """
    Derive the lock file path for an index CSV (same directory,
    '.lock' suffix).

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


def _next_csv_part_path(
        base_dir: Path,
        file_prefix: str,
        width: int = 3,
        ext: str = "csv"
) -> Path:
    """
    Return the next available rotating-CSV part path.

    Scans for files named "<file_prefix>.partNNN.<ext>" in `base_dir`,
    where NNN is an integer with zero-padding. Picks max(N) and returns
    the next. If none exist, returns "...part000.<ext>" (or the padding
    width you pass).

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
    pattern = re.compile(
        rf"^{re.escape(file_prefix)}\.part(\d+)\.{re.escape(ext)}$")

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


# ────────────────────────────────────────────────────────────────────────────
# UTILITIES - CSV FILE HANDLING
# ────────────────────────────────────────────────────────────────────────────

# Functions that assist with interacting with CSV files:
# * allocate_next_part
# * _drop_hash_from_part
# * is_model_logged_rotating_csv
# * _list_part_files
# * load_all_logs_rotating_csv
# * _read_index
# * remove_model_from_rotating_csv
# * save_summary_to_rotating_csv
# ────────────────────────────────────────────────────────────────────────────


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

    Uses os.open(..., O_CREAT|O_EXCL) so only one process can create a given
    part.
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

    raise RuntimeError("Failed to allocate a unique part file "
                       "after many attempts")


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
            dtype={"model_id_hash": "string"},
            # force string, avoid numeric coercion
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
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
            return 0

        # All rows removed
        if kept == 0:
            if delete_if_empty:
                # Delete original; remove temp if created
                try:
                    part_path.unlink()
                except Exception:
                    pass
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
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
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def is_model_logged_rotating_csv(
        model_hash: str,
        base_dir: str | Path,
        file_prefix: str
) -> bool:
    """
    Return True if `model_hash` appears in the rolling-log index for
    `file_prefix`.

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
    List existing rolling CSV parts for a given prefix, sorted by numeric part
    index.

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
        Sorted list of matching part files, e.g.
        [.../prefix.part000.csv, .../prefix.part001.csv, ...]
    """
    if not base_dir.exists():
        return []

    rx = re.compile(
        rf"^{re.escape(file_prefix)}\.part(\d+)\.{re.escape(ext)}$")
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
        Columns ['model_id_hash','part_file'] or empty frame if not
        found/invalid.
    """
    try:
        idx = pd.read_csv(index_path, dtype={
            "model_id_hash": "string", "part_file": "string"})
        if not {"model_id_hash", "part_file"}.issubset(idx.columns):
            raise ValueError("Index missing required columns.")
        return idx
    except FileNotFoundError:
        return pd.DataFrame(columns=["model_id_hash", "part_file"])
    except Exception:
        # Be permissive but return the expected schema
        return pd.DataFrame(columns=["model_id_hash", "part_file"])


def remove_model_from_rotating_csv(
        model_hash: str,
        results_dir: str | Path = ".",
        file_prefix: str = "marginal_emissions_log",
) -> None:
    """
    Remove all rows with `model_id_hash == model_hash` from the rolling CSV
    set.

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

    # Lock the index for the whole operation to avoid races with concurrent
    # writers/readers
    with _file_lock(_index_lock_path(idx_path)):
        idx = _read_index(idx_path)
        if idx.empty:
            return

        # Drop from referenced part files
        for pf in idx.loc[idx["model_id_hash"] == model_hash, "part_file"
                          ].dropna().unique():
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
    Append a single-row summary to a rolling CSV (<prefix>.partNNN.csv) with
    strict rotation:
    - Per-file lock during append (prevents interleaved writes/duplicate
      headers)
    - Under-lock preflight ensures the write will NOT push the file over
     `max_mb`
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
        Naming convention (token/width/ext). If provided, `ext` should include
        the dot (e.g., ".csv"). Internally we use the extension without the
        dot for matching.
    fsync : bool, default False
        If True, call fsync() on the file after writing to ensure data is
        flushed to disk.

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
            return parts[-1] if parts else base_dir / naming.format(
                file_prefix, 0)

    # Determine candidate shard
    parts = _list_part_files(base_dir, file_prefix, ext=ext_nodot)
    if parts:
        target = parts[-1]
    else:
        target = allocate_next_part(base_dir, file_prefix,
                                    width=naming.width, ext=ext_nodot)

    threshold_bytes = int(max_mb * 1024 * 1024)

    # --- LOCK AND WRITE TO SHARD SAFELY ---
    while True:
        shard_lock = Path(str(target) + ".lock")
        with _file_lock(shard_lock):
            current_size = (
                Path(target).stat().st_size if Path(target).exists() else 0)
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

        target = allocate_next_part(base_dir, file_prefix,
                                    width=naming.width, ext=ext_nodot)

    # --- LOCK AND UPDATE INDEX (atomic replace + optional fsync) ---
    lock_path = _index_lock_path(idx_path)
    with _file_lock(lock_path):
        idx = _read_index(idx_path)
        already = (
            "model_id_hash" in idx.columns) and (
                model_hash in idx["model_id_hash"].astype("string").values)
        if not already:
            idx = pd.concat(
                [idx, pd.DataFrame([{"model_id_hash": model_hash,
                                    "part_file": str(target)}])],
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


# ────────────────────────────────────────────────────────────────────────────
# UTILITIES - MPI MANAGEMENT
# ────────────────────────────────────────────────────────────────────────────

# Functions that assist with using MPI effectively:
# * _distribute_configs
# * _file_lock
# * _mpi_context
# * shard
# ────────────────────────────────────────────────────────────────────────────


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
    end = (n * (rank + 1)) // size
    return configs[start:end]


@contextmanager
def _file_lock(
        lock_path: Path,
        max_wait_s: float = 30.0,
        jitter_ms: tuple[int, int] = (2, 25)
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


def shard(
        seq: list,
        rank: int,
        size: int,
) -> list:
    """
    Shard a sequence into smaller chunks for distributed processing.

    Parameters
    ----------
    seq : list
        The input sequence to shard.
    rank : int
        The rank of the current process.
    size : int
        The total number of processes.

    Returns
    -------
    list
        The sharded sequence for the current process.
    """
    # deterministic round-robin
    return [x for i, x in enumerate(seq) if i % size == rank]
