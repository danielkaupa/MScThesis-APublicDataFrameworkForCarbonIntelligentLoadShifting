# ─────────────────────────────────────────────────────────────────────────────
# FILE: step6_optimisation_results_analysis.py
#
# PURPOSE:
#   Combine testing result shards into single files per (policy, solver, kind)
#   using Polars’ lazy engine and streaming Parquet writes.
#
# EXPECTED FILENAMES (examples):
#   policy_1_continuous_slsqp_moves_r0002_20250820_035359.parquet
#   policy_1_greedy_metrics_r0000_20250820_035359.parquet
#   policy_1_continuous_solver_trust_constr_optimised_r0001_20250820_020725.parquet
#
# GROUP KEYS:
#   policy : integer (e.g., 1, 2)
#   solver : string between "policy_<n>_" and "_{moves|metrics|optimised}_"
#   kind   : one of {"moves","metrics","optimised"}
#
# OUTPUT:
#   One combined Parquet per (policy, solver, kind), with columns:
#   [policy, solver, kind, rank, file_ts, source_file]
#
# RUN REQUIREMENTS:
# Requires: polars>=0.20, pyarrow (as your Parquet backend).
#
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
# import glob
# from datetime import datetime
import polars as pl
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# QUICK HELPER
# ────────────────────────────────────────────────────────────────────────────


def _safe_join(root: str | Path, maybe_rel: str | Path) -> Path:
    """Join to root if relative; return absolute Path otherwise."""
    p = Path(maybe_rel)
    return p if p.is_absolute() else Path(root) / p


# ────────────────────────────────────────────────────────────────────────────
# SETTING UP DIRECTORIES
# ────────────────────────────────────────────────────────────────────────────

try:
    base_directory = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
except NameError:
    # Fallback for interactive runs
    base_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))


base_data_directory = _safe_join(base_directory, "data")
hitachi_data_directory = os.path.join(base_data_directory, "hitachi_copy")
optimisation_development_directory = os.path.join(
    base_data_directory,
    "optimisation_development")
test_results_directory = os.path.join(
    optimisation_development_directory,
    "testing_results")
full_results_directory = os.path.join(
    optimisation_development_directory,
    "full_results")
optimisation_processing_directory = os.path.join(
    optimisation_development_directory,
    "processing_files")

# ────────────────────────────────────────────────────────────────────────────
# DEFINING FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────


def parse_filename(path: Path):
    m = FNAME_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    d["policy"] = int(d["policy"])
    d["rank"] = int(d["rank"])
    d["path"] = path
    return d


def combine_one_group(paths: list[Path],
                      meta_example: dict, out_dir: Path) -> None:
    policy = meta_example["policy"]
    solver = meta_example["solver"]
    kind = meta_example["kind"]
    run_id = meta_example["run_id"]

    # Keep solver name filesystem-friendly
    safe_solver = re.sub(r"[^A-Za-z0-9._-]+", "_", solver)
    out_path = out_dir / (f"policy_{policy}_{safe_solver}_{kind}_"
                          f"run{run_id}_combined.parquet")

    print(f"  → Combining {len(paths):>3} shard(s): "
          f"policy={policy}, solver={solver}, kind={kind}, run_id={run_id}")
    for p in paths:
        print(f"      - {p.name}")

    lfs = []
    for p in paths:
        md = parse_filename(p)
        lf = (
            pl.scan_parquet(p)
            .with_columns(
                pl.lit(policy).alias("policy"),
                pl.lit(solver).alias("solver"),
                pl.lit(kind).alias("kind"),
                pl.lit(md["rank"]).alias("rank"),
                pl.lit(md["date"]).alias("file_date"),
                pl.lit(md["run_id"]).alias("run_id"),
                pl.lit(p.name).alias("source_file"),
            )
        )
        lfs.append(lf)

    # Concatenate with schema tolerance for robustness across shards
    lf_all = pl.concat(lfs, how="diagonal_relaxed")

    if DROP_DUPLICATES:
        lf_all = lf_all.unique(maintain_order=True)

    lf_all.sink_parquet(out_path, compression=COMPRESSION, statistics=True)
    print(f"    ✓ Wrote: {out_path}")


def combine_groups(
    input_dir: Path,
    output_dir: Path,
    compression: str = "snappy",
    drop_duplicates: bool = False,
    glob_pat: str = "*.parquet",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in input_dir.glob(glob_pat) if p.is_file())
    if not files:
        print(f"[WARN] No files found in {input_dir} matching {glob_pat}")
        return

    groups: dict[tuple[int, str, str, str], list[Path]] = {}
    meta_for_key: dict[tuple[int, str, str, str], dict] = {}
    skipped = 0

    for p in files:
        meta = parse_filename(p)
        if not meta:
            skipped += 1
            continue
        key = (meta["policy"], meta["solver"], meta["kind"], meta["run_id"])
        groups.setdefault(key, []).append(p)
        meta_for_key.setdefault(key, meta)

    if skipped:
        print(f"[INFO] Skipped {skipped} file(s) that did not match the "
              f"expected pattern.")
    if not groups:
        print("[WARN] No matching result files after filtering. "
              "Nothing to do.")
        return

    print(f"[INFO] Found {len(groups)} groups (policy, solver, kind, run_id)."
          f" Starting combine…")

    # Use global config from the IMPLEMENTATION section
    global COMPRESSION, DROP_DUPLICATES
    COMPRESSION = compression
    DROP_DUPLICATES = drop_duplicates

    for key, paths in sorted(groups.items()):
        combine_one_group(paths, meta_for_key[key], output_dir)

    print(f"[DONE] Combined {len(groups)} group(s) into {output_dir}")

# ────────────────────────────────────────────────────────────────────────────
# IMPLEMENTATION
# ────────────────────────────────────────────────────────────────────────────


# Regex: policy_<policy>_<solver>_(moves|metrics|optimised)_r<rank>_<YYYYMMDD>_
# <RUNID>.parquet
FNAME_RE = re.compile(
    r"""^policy_
        (?P<policy>\d+)_
        (?P<solver>.+?)_
        (?P<kind>moves|metrics|optimised)_
        r(?P<rank>\d{4})_
        (?P<date>\d{8})_
        (?P<run_id>\d{6})
        \.parquet$""",
    re.VERBOSE,
)

# Directory containing the many shard files you showed in your screenshot
INPUT_DIR = test_results_directory
# Where to write the combined outputs (will be created)
OUTPUT_DIR = optimisation_development_directory
# Parquet compression for outputs
COMPRESSION = "snappy"  # options: "uncompressed","snappy","gzip","zstd","lz4"
# Drop exact duplicate rows after concatenation? (uses more memory)
DROP_DUPLICATES = False
# Which files to read from INPUT_DIR
GLOB_PATTERN = "*.parquet"


if __name__ == "__main__":
    # Pretty console output (optional)
    try:
        pl.Config.set_tbl_rows(20)
        pl.Config.set_fmt_str_lengths(120)
    except Exception:
        pass

    input_dir = _safe_join(Path.cwd(), INPUT_DIR)
    output_dir = _safe_join(Path.cwd(), OUTPUT_DIR)

    print(f"[CONFIG] INPUT_DIR={input_dir}")
    print(f"[CONFIG] OUTPUT_DIR={output_dir}")
    print(f"[CONFIG] COMPRESSION={COMPRESSION} "
          f"DROP_DUPLICATES={DROP_DUPLICATES}")
    print(f"[CONFIG] GLOB_PATTERN={GLOB_PATTERN}")

    combine_groups(
        input_dir=input_dir,
        output_dir=output_dir,
        compression=COMPRESSION,
        drop_duplicates=DROP_DUPLICATES,
        glob_pat=GLOB_PATTERN,
    )
