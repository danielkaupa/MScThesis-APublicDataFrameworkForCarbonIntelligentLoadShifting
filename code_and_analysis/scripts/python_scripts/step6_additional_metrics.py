# ─────────────────────────────────────────────────────────────────────────────
# FILE: step6_additional_metrics.py
#
# PURPOSE:
#  - Region/household usage & emissions context metrics at daily/weekly
#    horizons with parallel collection and single-scan I/O efficiency.
# NOTES:
#  - One parquet scan; results fanned out by scope (ALL/Delhi/Mumbai).
#  - Big outputs sink to Parquet; summaries also saved to CSV.
#  - Emissions math:
#       For each period take the mean of average_emissions_grams_co2_per_kWh
#       and multiply by the period’s total kWh.
#       This is not energy-weighted; it’s a time-weighted average across
#       records in that period.
#  - Week Boundaries:
#       Week boundaries use ISO week (Mon–Sun) via %G/%V/%u, so gaps in data
#       don’t bias weekly grouping.
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import polars as pl
import numpy as np
from collections import Counter

# ────────────────────────────────────────────────────────────────────────────
# Misc Environment Configuration
# ────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("POLARS_MAX_THREADS", str(os.cpu_count() or 4))
pl.enable_string_cache()

os.environ.setdefault("POLARS_FORCE_OOC", "1")  # allow out-of-core operators

# ────────────────────────────────────────────────────────────────────────────
# DIRECTORIES, FILENAMES & FILEPATHS
# ────────────────────────────────────────────────────────────────────────────

# Base directory for data
base_data_directory = "../data"
optimisation_development_directory = os.path.join(base_data_directory,
                                                  "optimisation_development")
outputs_directory = os.path.join(base_data_directory, "outputs")
metrics_directory = os.path.join(outputs_directory, "metrics")
os.makedirs(metrics_directory, exist_ok=True)

# Filenames & Filepaths
source_filename = (
    "meter_readings_all_years_20250714_formatted_with_emission_factors_filled")
source_filepath = os.path.join(optimisation_development_directory,
                               source_filename + ".parquet")

# ────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT VARIABLES
# ────────────────────────────────────────────────────────────────────────────
SOURCE_PARQUET = source_filepath
OUTDIR = metrics_directory

# None => "ALL cities"
CITY_LIST = [None, "delhi", "mumbai"]
USAGE_COL = "value"
# average emissions factor (g/kWh)
AVG_EF_COL = "average_emissions_grams_co2_per_kWh"
ID_COL = "ca_id"
# Datetime[us, Asia/Kolkata]
TS_COL = "date"
CITY_COL = "city"

# ────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────


def _city_tag(city: str | None) -> str:
    return "all" if city is None else city.lower()


def _base_scan(city: str | None = None) -> pl.LazyFrame:
    # scan only the columns we need; keep tz as-is
    lf = (
        pl.scan_parquet(SOURCE_PARQUET)
        .select([ID_COL, TS_COL, "city", USAGE_COL, AVG_EF_COL])
    )
    if city:
        lf = lf.filter(pl.col("city") == city)
    return lf


def _add_time_cols(lf: pl.LazyFrame) -> pl.LazyFrame:
    # ISO week (Mon–Sun): %G=ISO year, %V=ISO week 01–53, %u=1..7 for weekday
    # (Mon=1)
    return lf.with_columns([
        pl.col(TS_COL).dt.date().alias("day"),
        pl.col(TS_COL).dt.truncate("1h").alias("hour_bin"),
        pl.col(TS_COL).dt.hour().alias("hour_of_day"),
        pl.col(TS_COL).dt.strftime("%G").cast(pl.Int32).alias("iso_year"),
        pl.col(TS_COL).dt.strftime("%V").cast(pl.Int32).alias("iso_week"),
        pl.col(TS_COL).dt.strftime("%u").cast(pl.Int8).alias("iso_weekday"),
        # 1=Mon ... 7=Sun
    ])


def _round_for_mode(col: str, decimals: int = 2) -> pl.Expr:
    # rounding before mode to make it meaningful for continuous variables
    return pl.col(col).round(decimals)


def _mode_expr(col: str, decimals: int = 1) -> pl.Expr:
    # compute a scalar "mode" after rounding: value with highest frequency
    # returns the FIRST most-frequent value if ties
    return (
        _round_for_mode(col, decimals).alias("_rounded")
        .pipe(lambda e: pl.struct([e]).alias("_s"))
        # placeholder; we'll do mode after collect
    )


def _save_csv(df: pl.DataFrame, filename: str):
    path = os.path.join(OUTDIR, filename)
    df.write_csv(path)
    print(f"Saved CSV -> {path}")


def _sink_parquet(lf: pl.LazyFrame, filename: str):
    path = os.path.join(OUTDIR, filename)
    lf.sink_parquet(path)   # streaming, lazy sink
    print(f"Sink Parquet -> {path}")


def _summarise_series(
    df: pl.DataFrame,
    value_col: str,
    label: str,
    mode_round_decimals: int = 1,
) -> pl.DataFrame:
    # Pull the column into NumPy
    arr = df[value_col].to_numpy()

    if arr.size == 0:
        return pl.DataFrame({
            "stat_of": [label],
            "mean": [None],
            "median": [None],
            "mode_rounded": [None],
            "n": [0],
        })

    # Mean & median fast via NumPy
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))

    # Mode after rounding
    rounded = np.round(arr, mode_round_decimals)
    counts = Counter(rounded)
    # pick the smallest value in case of tie
    mode_val = min([k for k, v in counts.items() if v == max(counts.values())])

    return pl.DataFrame({
        "stat_of": [label],
        "mean": [mean_val],
        "median": [median_val],
        "mode_rounded": [mode_val],
        "n": [arr.size],
    })


def weekday_name_expr():
    # iso_weekday: 1..7 -> Mon..Sun
    return pl.when(pl.col("iso_weekday") == 1).then(pl.lit("Mon")) \
             .when(pl.col("iso_weekday") == 2).then(pl.lit("Tue")) \
             .when(pl.col("iso_weekday") == 3).then(pl.lit("Wed")) \
             .when(pl.col("iso_weekday") == 4).then(pl.lit("Thu")) \
             .when(pl.col("iso_weekday") == 5).then(pl.lit("Fri")) \
             .when(pl.col("iso_weekday") == 6).then(pl.lit("Sat")) \
             .otherwise(pl.lit("Sun")).alias("weekday_name")


def compute_top6(city: str | None):
    tag = _city_tag(city)
    base = _add_time_cols(_base_scan(city)).sort("hour_bin")

    # Sum usage per (day, weekday, hour-of-day)  -> avoids datetime key
    hourly_totals = (
        base.group_by(["day", "iso_weekday", "hour_of_day"])
            .agg(pl.sum(USAGE_COL).alias("hourly_kwh"))
    )

    # Mean across days for each (weekday, hour-of-day)
    wk_hr_mean = (
        hourly_totals.group_by(["iso_weekday", "hour_of_day"])
                     .agg(pl.mean("hourly_kwh").alias("mean_hourly_kwh"))
                     .with_columns(weekday_name_expr())
    )

    top6 = (
        wk_hr_mean
        .with_columns(
            pl.col("mean_hourly_kwh")
              .rank("dense", descending=True)
              .over("iso_weekday")
              .alias("rank_in_weekday")
        )
        .filter(pl.col("rank_in_weekday") <= 6)
        .sort(["iso_weekday", "rank_in_weekday", "hour_of_day"])
        .with_columns(pl.lit(tag.upper()).alias("scope"))
        .select(["scope", "iso_weekday", "weekday_name", "rank_in_weekday",
                 "hour_of_day", "mean_hourly_kwh"])
    ).collect(engine="streaming")

    _save_csv(top6, f"{tag}_top6_hours_by_weekday.csv")
    _sink_parquet(pl.LazyFrame(top6), f"{tag}_top6_hours_by_weekday.parquet")
    print(f"\n=== TOP-6 HOURS PER WEEKDAY ({tag.upper()}) ===")
    print(top6)


# ────────────────────────────────────────────────────────────────────────────
# IMPLEMENTATION
# ────────────────────────────────────────────────────────────────────────────

BASE_ALL = _add_time_cols(_base_scan(None))

# ---- 1) Region-wide DAILY + WEEKLY usage & emissions (ALL, Delhi, Mumbai)
all_stats_tables = []  # to print consolidated summaries later


# for city in CITY_LIST:
#    tag = _city_tag(city)
# base = BASE_ALL if city is None else BASE_ALL.filter(pl.col(CITY_COL
# ) == city)
#
#    # DAILY region-wide (sum usage; emission factor = mean of avg EF that day;
#    # emissions = sum_kwh * mean_factor)
#    daily_lf = (
#        base.group_by("day")
#        .agg([
#            pl.sum(USAGE_COL).alias("daily_kwh"),
#            pl.mean(AVG_EF_COL).alias("daily_avg_g_per_kwh"),
#        ])
#        .with_columns((pl.col("daily_kwh") * pl.col("daily_avg_g_per_kwh")
#                       ).alias("daily_emissions_g"))
#        .sort("day")
#    )
#
#    # WEEKLY (ISO Mon-Sun)
#    weekly_lf = (
#        base.group_by(["iso_year", "iso_week"])
#        .agg([
#            pl.sum(USAGE_COL).alias("weekly_kwh"),
#            pl.mean(AVG_EF_COL).alias("weekly_avg_g_per_kwh"),
#        ])
#        .with_columns((pl.col("weekly_kwh") * pl.col("weekly_avg_g_per_kwh")
#                       ).alias("weekly_emissions_g"))
#        .sort(["iso_year", "iso_week"])
#    )
#
#    # Sinks (parquet)
#    _sink_parquet(daily_lf, f"{tag}_region_daily_usage_emissions.parquet")
#    _sink_parquet(weekly_lf, f"{tag}_region_weekly_usage_emissions.parquet")
#
#    # Collect (streaming) to compute annual/monthly equivalents and stats
#    daily_df = daily_lf.collect(engine="streaming")
#    weekly_df = weekly_lf.collect(engine="streaming")
#
#    # Annual & monthly equivalents from WEEKLY (multiply by 52; divide by 12)
#    weekly_df = weekly_df.with_columns([
#        (pl.col("weekly_kwh") * 52).alias(
#            "annual_kwh_equiv"),
#        (pl.col("weekly_emissions_g") * 52).alias(
#            "annual_emissions_g_equiv"),
#        (pl.col("weekly_kwh") * 52 / 12).alias(
#            "monthly_kwh_equiv"),
#        (pl.col("weekly_emissions_g") * 52 / 12).alias(
#            "monthly_emissions_g_equiv"),
#    ])
#
#    # Save CSVs
#    _save_csv(daily_df,  f"{tag}_region_daily_usage_emissions.csv")
#    _save_csv(weekly_df, f"{tag}_region_weekly_usage_emissions_and_equiv.csv")
#
#    # Print summaries (mean/median/mode) for each metric
#    print(f"\n=== REGION-WIDE ({tag.upper()}) ===")
#    for (df, value_col, label, dec) in [
#        (daily_df,  "daily_kwh",
#         f"{tag}_daily_kwh", 1),
#        (daily_df,  "daily_emissions_g",
#         f"{tag}_daily_emissions_g", 0),
#        (weekly_df, "weekly_kwh",
#         f"{tag}_weekly_kwh", 1),
#        (weekly_df, "weekly_emissions_g",
#         f"{tag}_weekly_emissions_g", 0),
#        (weekly_df, "annual_kwh_equiv",
#         f"{tag}_annual_kwh_equiv", 0),
#        (weekly_df, "annual_emissions_g_equiv",
#         f"{tag}_annual_emissions_g_equiv", 0),
#        (weekly_df, "monthly_kwh_equiv",
#         f"{tag}_monthly_kwh_equiv", 0),
#        (weekly_df, "monthly_emissions_g_equiv",
#         f"{tag}_monthly_emissions_g_equiv", 0),
#    ]:
#        stat = _summarise_series(df, value_col, label,
#                                 mode_round_decimals=dec)
#        print(stat)
#        all_stats_tables.append(stat)
#
# Consolidated CSV of all region-wide stats
# if all_stats_tables:
#    all_stats = pl.concat(all_stats_tables).collect() if isinstance(
#        all_stats_tables[0], pl.LazyFrame
#        ) else pl.concat(all_stats_tables)
#    _save_csv(all_stats, "region_wide_summary_stats.csv")


# # ---- 2) Household-level metrics (ALL, Delhi, Mumbai) ----
# hh_stats_tables = []

# for city in CITY_LIST:
#     tag = _city_tag(city)
# base = BASE_ALL if city is None else BASE_ALL.filter(pl.col(CITY_COL
# ) == city)

#     # 2a) Hourly by customer -> min/max hourly usage per customer
#     # Build hourly sums per customer using time-based streaming grouping
# hourly_by_ca = (
#   base.group_by_dynamic(TS_COL, every="1h", group_by=ID_COL, closed="left")
#        .agg(pl.sum(USAGE_COL).alias("hourly_kwh"))
#    # TS_COL column is the hour bin here; you don't need hour_bin downstream
#     )

#     # Spill intermediate to disk to keep memory bounded, then rescan + finish
#     _hourly_tmp = os.path.join(OUTDIR, f"{tag}_hourly_by_ca.parquet")
#     hourly_by_ca.sink_parquet(_hourly_tmp)

#     per_ca_minmax = (
#         pl.scan_parquet(_hourly_tmp)
#           .group_by(ID_COL)
#           .agg([
#               pl.min("hourly_kwh").alias("min_hourly_kwh"),
#               pl.max("hourly_kwh").alias("max_hourly_kwh"),
#           ])
#           .collect(engine="streaming")
#     )

#     # Aggregate across customers -> averages/medians/modes
#     print(f"\n=== HOUSEHOLD HOURLY EXTREMES ({tag.upper()}) ===")
#     for col, lbl in [("min_hourly_kwh", "avg_min_hourly_kwh"),
#                      ("max_hourly_kwh", "avg_max_hourly_kwh")]:
#         stat = _summarise_series(per_ca_minmax,
#                                  col,
#                                  f"{tag}_{lbl}",
#                                  mode_round_decimals=1)
#         print(stat)
#         hh_stats_tables.append(stat)

#     # 2b) Household DAILY totals & emissions (per household-day)
#     hh_daily = (
#         base.group_by([ID_COL, "day"])
#         .agg([
#             pl.sum(USAGE_COL).alias("daily_kwh"),
#             pl.mean(AVG_EF_COL).alias("daily_avg_g_per_kwh"),
#         ])
#         .with_columns((pl.col("daily_kwh") * pl.col("daily_avg_g_per_kwh")
#                        ).alias("daily_emissions_g"))
#     )

#     # Reduce to one row per household: average day over observed days
#     _hh_daily_tmp = os.path.join(OUTDIR, f"{tag}_hh_daily.parquet")
#     hh_daily.sink_parquet(_hh_daily_tmp)

#     hh_daily_mean = (
#         pl.scan_parquet(_hh_daily_tmp)
#           .group_by(ID_COL)
#           .agg([
#               pl.mean("daily_kwh").alias("mean_daily_kwh"),
#               pl.mean("daily_emissions_g").alias("mean_daily_emissions_g"),
#           ])
#           .collect(engine="streaming")
#     )

#     print(f"\n=== HOUSEHOLD DAILY ({tag.upper()}) ===")
#     for col, lbl in [("mean_daily_kwh", "household_daily_kwh"),
#                      ("mean_daily_emissions_g",
#                       "household_daily_emissions_g")]:
#         stat = _summarise_series(hh_daily_mean,
#                                  col,
#                                  f"{tag}_{lbl}",
#                                  mode_round_decimals=1 if "kwh" in col
# else 0)
#         print(stat)
#         hh_stats_tables.append(stat)

#     # 2c) Household WEEKLY totals & emissions (Mon–Sun), then annual = *52
#     hh_weekly = (
#         base.group_by([ID_COL, "iso_year", "iso_week"])
#         .agg([
#             pl.sum(USAGE_COL).alias("weekly_kwh"),
#             pl.mean(AVG_EF_COL).alias("weekly_avg_g_per_kwh"),
#         ])
#         .with_columns((pl.col("weekly_kwh") * pl.col("weekly_avg_g_per_kwh")
#                        ).alias("weekly_emissions_g"))
#     )

#     _hh_weekly_tmp = os.path.join(OUTDIR, f"{tag}_hh_weekly.parquet")
#     hh_weekly.sink_parquet(_hh_weekly_tmp)

#     hh_weekly_mean = (
#         pl.scan_parquet(_hh_weekly_tmp)
#           .group_by(ID_COL)
#           .agg([
#               pl.mean("weekly_kwh").alias("mean_weekly_kwh"),
#               pl.mean("weekly_emissions_g").alias("mean_weekly_emissions_g"),
#           ])
#           .with_columns([
#               (pl.col("mean_weekly_kwh") * 52).alias("annual_kwh_equiv"),
#               (pl.col("mean_weekly_emissions_g") * 52).alias(
#                   "annual_emissions_g_equiv"),
#           ])
#           .collect(engine="streaming")
#     )

#     print(f"\n=== HOUSEHOLD WEEKLY & ANNUAL ({tag.upper()}) ===")
#     for col, lbl in [
#         ("mean_weekly_kwh", "household_weekly_kwh"),
#         ("mean_weekly_emissions_g", "household_weekly_emissions_g"),
#         ("annual_kwh_equiv", "household_annual_kwh_equiv"),
#         ("annual_emissions_g_equiv", "household_annual_emissions_g_equiv"),
#     ]:
#         stat = _summarise_series(hh_weekly_mean,
#                                  col,
#                                  f"{tag}_{lbl}",
#                                  mode_round_decimals=1 if "kwh" in col
# else 0)
#         print(stat)
#         hh_stats_tables.append(stat)

#     # Save household-per-customer aggregates (CSV + Parquet)
#     _save_csv(hh_daily_mean,
#               f"{tag}_household_mean_daily_usage_emissions.csv")
#     _save_csv(hh_weekly_mean,
#               f"{tag}_household_mean_weekly_and_annual_usage_emissions.csv")
#     # (For Parquet sinks, rebuild as LazyFrames)
#     _sink_parquet(pl.LazyFrame(hh_daily_mean),
#                   f"{tag}_household_mean_daily_usage_emissions.parquet")
#     _sink_parquet(
#         pl.LazyFrame(
#             hh_weekly_mean),
#         f"{tag}_household_mean_weekly_and_annual_usage_emissions.parquet")

# # Consolidated household stats CSV
# if hh_stats_tables:
#     hh_stats = pl.concat(hh_stats_tables)
#     _save_csv(hh_stats, "household_summary_stats.csv")


# ---- 3) Peak hours: top-6 hours by weekday (Mon–Sun) for ALL, Delhi, Mumbai
# We compute mean hourly kWh per (weekday, hour-of-day), then take top-6 per
# weekday.

for city in CITY_LIST:
    compute_top6(city)

print("\nAll done.")
