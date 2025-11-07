# ─────────────────────────────────────────────────────────────────────────────
# FILE: step5_meter_emissions_full_city_week_partitioning.py
#
# PURPOSE:
# - Partition the “meter readings + emissions” dataset into
#   **city–week shards** to enable embarrassingly parallel
#   optimisation later on.
# - Normalise schema (column names, types, timezone) for downstream
#   consistency.
# - Compute convenience keys:
#     * `day` (date-only),
#     * `slot` (0–47 for 30-min slots; param via SLOT_LEN_MIN),
#     * `week_start` (ISO-style week anchor as Date).
# - Write one Parquet per (city, week_start) with streaming,
#   predicate pushdown, and MPI-based fan-out.
#
# USAGE:
# - Run as an MPI program, e.g.:
#     mpirun -n 8 python step4a_partition_city_week_shards.py
# - Each rank receives a static strided subset of unique
#   (city, week_start) keys and writes its own shard(s) to disk.
# - Intended to precede the optimisation step that consumes per-city-week
#   files.
#
# RUN REQUIREMENTS:
# - Python 3.9+.
# - Libraries: polars, mpi4py.
#   (polars compiled with SIMD; s3fs/pyarrow optional if writing to cloud FS.)
# - Environment:
#   * POLARS_MAX_THREADS can be tuned (default set here to "6").
#   * POSIX-like filesystem for output directory creation.
#
# INPUTS (expected on disk):
# - meter_readings_all_years_20250714_formatted_with_emission_factors_filled
#   .parquet
#
# WHAT THE SCRIPT DOES (step-by-step):
# 1) **Path setup:** Resolves repo root (one level above /scripts) to build
#    stable absolute paths into the `data/optimisation_development` tree.
#
# 2) **Lazy scan & schema normalisation:**
#    - Renames (if present) to canonical fields:
#        * `marginal_emissions_grams_co2_per_kWh`
#            → `marginal_emissions_factor_grams_co2_per_kWh`
#        * `average_emissions_grams_co2_per_kWh`
#          or `average_emissions_factor_co2_per_kWh`
#            → `average_emissions_factor_grams_co2_per_kWh`
#    - Ensures `date` is tz-aware in **Asia/Kolkata** (localise or convert).
#    - Ensures `city` is Utf8 (not Categorical) for robust equality filters.
#
# 3) **Derived keys for partitioning:**
#    - `day` = floor to day (local tz).
#    - `slot` = half-hour index within day (0–47) using SLOT_LEN_MIN.
#    - `week_start` = Monday anchor as **Date** (not Datetime) for equality
#       joins.
#
# 4) **Key discovery (rank 0):**
#    - Collect unique pairs of (`city`, `week_start`) and broadcast to all
#       ranks.
#    - Prints a small sample of keys and the input schema for QA.
#
# 5) **MPI fan-out:**
#    - Static strided assignment: `my_keys = keys[rank::size]`.
#    - For each (city, week_start), filter lazily and **sink_parquet** to:
#        optimisation_development/city_week_shards/
#        meter_readings_20250714_emissions_<city>_week_<YYYY-MM-DD>.parquet
#    - Predicate pushdown + streaming keeps memory small per shard.
#
# OUTPUTS (Parquet, one per shard):
# - city_week_shards/
# - /meter_readings_20250714_emissions_<sanitized-city>_week_<YYYY-MM-DD>
#   .parquet
#
# NOTES & ASSUMPTIONS:
# - Expects a `date` column of Polars **Datetime** type; raises if missing.
# - Timezone handling:
#     * If `date` is naïve → localise to Asia/Kolkata.
#     * If already tz-aware but not IST → convert to Asia/Kolkata.
# - Sanitises city names for filenames (`[a-z0-9-_]`),
#   preserving dashes/underscores.
# - `week_start` is computed as `day - weekday` (Mon=0) and stored as **Date**
#   to avoid tz drift and to make equality filters exact.
# - Tune `POLARS_MAX_THREADS` and MPI `-n` to match node resources.
# - Shard size balance depends on city/week distribution; static striding is
#   simple and usually adequate—switch to dynamic work stealing if needed.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import polars as pl
from polars.datatypes import Datetime as PLDatetime
from mpi4py import MPI
os.environ.setdefault("POLARS_MAX_THREADS", "6")

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# FUNCTIONS
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


def _abs_join(root: str, maybe_rel: str) -> str:
    """Join to root if path is relative; return as-is if already absolute."""
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(
        root, maybe_rel)


def sanitize_city(s: str) -> str:
    """Make safe-ish folder/file tokens."""
    return "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in str(s).lower()
    )


# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# DEFINING FILEPATHS AND DIRECTORIES
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


# DIRECTORIES AND PATHS
# Resolve repo root → .../code_and_analysis (one level up from scripts/)

try:
    base_directory = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
except NameError:
    # Fallback for interactive runs
    base_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))

base_data_directory = _abs_join(base_directory, "data")

optimisation_development_directory = os.path.join(
    base_data_directory, "optimisation_development")
city_week_shards_directory = os.path.join(
    optimisation_development_directory, "city_week_shards")

os.makedirs(city_week_shards_directory, exist_ok=True)

# Defining Full files
full_filename = (
    "meter_readings_all_years_20250714_formatted_with_emission_factors_filled"
)

# Defining Full Filepaths
full_filepath = os.path.join(
    optimisation_development_directory,
    full_filename + ".parquet")


# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# LOADING AND PROCESSING DATA#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


# Slot config
SLOT_LEN_MIN = 30  # minutes
PER_HOUR = 60 // SLOT_LEN_MIN

out_dir = city_week_shards_directory
lf = pl.scan_parquet(full_filepath)

# normalize column names if they exist
rename_candidates = {
    "marginal_emissions_grams_co2_per_kWh":
    "marginal_emissions_factor_grams_co2_per_kWh",
    "average_emissions_grams_co2_per_kWh":
    "average_emissions_factor_grams_co2_per_kWh",
    "average_emissions_factor_co2_per_kWh":
    "average_emissions_factor_grams_co2_per_kWh",
}

schema = lf.collect_schema()
present = [k for k in rename_candidates if k in schema.names()]
if present:
    lf = lf.rename({k: rename_candidates[k] for k in present})

# ensure date is tz-aware in Asia/Kolkata
date_dt = schema.get("date")
if isinstance(date_dt, PLDatetime):
    tz = getattr(date_dt, "tz", None)
    if tz is None:
        lf = lf.with_columns(pl.col("date").dt.replace_time_zone(
            "Asia/Kolkata"))
    elif tz != "Asia/Kolkata":
        lf = lf.with_columns(pl.col("date").dt.convert_time_zone(
            "Asia/Kolkata"))
else:
    raise ValueError("Expected a Datetime column 'date'")

# ensure city is Utf8 (not Categorical)
lf = lf.with_columns(pl.col("city").cast(pl.Utf8))

# derive day/slot/week_start (week_start as Date for simple equality)
day = pl.col("date").dt.truncate("1d")
lf = lf.with_columns([
    day.alias("day"),
    ((pl.col("date").dt.hour() * PER_HOUR) +
     (pl.col("date").dt.minute() // SLOT_LEN_MIN)).cast(
         pl.Int32).alias("slot"),
    (day - pl.duration(days=day.dt.weekday())).dt.date().alias("week_start"),
])

# ---- MPI fan-out ----
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    keys_df = lf.select(["city", "week_start"]).unique().collect(
        streaming=True)
    keys = [(c, wk) for c, wk in keys_df.iter_rows()]
    print(f"[partition] {len(keys)} shards → {out_dir}", flush=True)
    # debug: show a few keys and schema
    print("[partition] sample keys:", keys[:5], flush=True)
    print("[partition] schema:", lf.collect_schema(), flush=True)
else:
    keys = None

keys = comm.bcast(keys, root=0)
my_keys = keys[rank::size]  # static slice per rank

# confirm dtypes used in filters
wk_dtype = lf.collect_schema()["week_start"]   # pl.Date
# city is Utf8 by construction

count = 0
for city, wk in my_keys:
    # avoid Date vs Datetime mismatches
    wk_lit = pl.lit(wk).cast(wk_dtype)
    out_path = os.path.join(out_dir,
                            (f"meter_readings_20250714_emissions_"
                             f"{sanitize_city(city)}_week_"
                             f"{wk:%Y-%m-%d}.parquet"))

    (lf
     .filter((pl.col("city") == city) &
             (pl.col("week_start") == wk_lit))
     # streaming write; predicate pushdown is applied if possible
     .sink_parquet(out_path))

    count += 1
    print((f"[r{rank}/{size}] {count}/{len(my_keys)} "
           f":: {city} / {wk:%Y-%m-%d} → {out_path}"),
          flush=True)

comm.Barrier()
if rank == 0:
    print("[partition] done.", flush=True)
