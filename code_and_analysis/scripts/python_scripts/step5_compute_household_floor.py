# ─────────────────────────────────────────────────────────────────────────────
# FILE: step5_compute_household_floor.py
#
# PURPOSE:
# - Precompute a per-household minimum-usage “floor” to be enforced during
#   shifting. The floor is defined per (ca_id, hour-of-day, day-of-week)
#   as max( baseline, R% * robust_max, epsilon ), where:
#     baseline = mean or min usage by (ca_id,hour,weekday) over the full
#       dataset
#     robust_max = Pth percentile of usage by (ca_id,hour,weekday)
#     R% = fraction (e.g., 10%) and epsilon = small kWh floor to avoid zeros
# - Writes a compact lookup Parquet that can be joined into shards upstream.
#
# USAGE:
# - Update the “Paths” section if your repo/data layout differs.
# - Run:
#       python step5_compute_household_floor.py
# - The script writes a single Parquet lookup with columns:
#       ["ca_id", "hod", "dow", "floor_kwh"]
#
# RUN REQUIREMENTS:
# - Python 3.9+
# - polars (with lazy/scan support).
# - Optional but recommended: zstd compression.
# - Sufficient memory to group over (ca_id, hod, dow) across the full dataset.
#
# INPUTS (expected on disk):
# - data/optimisation_development/
#     meter_readings_all_years_20250714_formatted_with_emission_factors_filled.parquet
#
# OUTPUTS (Parquet lookup):
# - data/optimisation_development/processing_files/
#     meter_readings_all_years_20250714_household_floor_lookup.parquet
#   Columns: ["ca_id","hod","dow","floor_kwh"]

# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os
import polars as pl
from polars.datatypes import Datetime as PLDatetime
if "POLARS_MAX_THREADS" not in os.environ:
    os.environ["POLARS_MAX_THREADS"] = os.environ.get("OMP_NUM_THREADS",
                                                      str(os.cpu_count() or 1))


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
processing_files_directory = os.path.join(
    optimisation_development_directory, "processing_files")
city_week_shards_directory = os.path.join(
    optimisation_development_directory, "city_week_shards")

os.makedirs(city_week_shards_directory, exist_ok=True)
os.makedirs(processing_files_directory, exist_ok=True)

# Defining Full files
full_filename = (
    "meter_readings_all_years_20250714_formatted_with_emission_factors_filled"
)

# Defining Full Filepaths
full_filepath = os.path.join(
    optimisation_development_directory,
    full_filename + ".parquet")

output_filename = "meter_readings_all_years_20250714_household_floor_lookup"
output_filepath = os.path.join(
    processing_files_directory,
    output_filename + ".parquet"
)

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# LOADING AND PROCESSING DATA#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────

ROBUST_P = 0.95   # 95th percentile
R_PCT = 0.10   # 10%
EPS = 0.001  # 1 Wh floor

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
schema = lf.collect_schema()
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

lfh = lf.with_columns([
    pl.col("date").dt.hour().alias("hod"),
    pl.col("date").dt.weekday().alias("dow"),
])

lookup = (
    lfh
    .group_by(["ca_id", "hod", "dow"])
    .agg([
        pl.mean("value").alias("baseline_kwh"),
        pl.col("value").quantile(ROBUST_P,
                                 interpolation="nearest"
                                 ).alias("robust_max_kwh"),
    ])
    .with_columns([
        pl.max_horizontal([
            pl.col("baseline_kwh"),
            pl.col("robust_max_kwh") * R_PCT
        ]).alias("floor_kwh_raw"),
        pl.lit(EPS).alias("eps"),
    ])
    .with_columns(pl.max_horizontal(["floor_kwh_raw", "eps"]
                                    ).alias("floor_kwh"))
    .select(["ca_id", "hod", "dow", "floor_kwh"])
)
lookup.sink_parquet(output_filepath, compression="snappy", statistics=True)
print(f"Wrote {output_filepath}")
print(lookup.select(pl.len()).collect())
