# ─────────────────────────────────────────────────────────────────────────────
# FILE: step2_meter_readings_analysis.py

# PURPOSE: Perform analysis on meter readings files that have been retrieved
# from the 'hitachi' database. This script is meant to generate all the
# statistics and plots that would be difficult to generate on a personal
# computer with normal processing power.

# USAGE: This script is designed to run on the HPC cluster at
# Imperial College London

# RUN REQUIREMENTS:
# - Requires installation and setup of dependent libraries, directories,
#   and environment variables.
# - This script specifically requires a sh file to run it on the HPC cluster
#   'meter_readings_analysis.sh'

# OUTPUTS:
# (1) an Individual Files Stats CSV (one row per input file) with customer
#   counts, half-hour totals (and hours/days equivalents), non-zero counts,
#   and per-customer averages (overall and per city).
# (2) A Bigfile Usage Stats CSV covering totals and per-customer usage
#   for hour/week/month/year across scopes (all, Delhi, Mumbai, each year,
#   city×year), using coverage-aware day fill to extrapolate missing days,
#   and also reporting an annualized figure (weekly mean ×52).
# (3) Per-customer distributions saved as CSVs plus histograms and boxplots
#   for each scope/frequency.
# (4) Monthly totals by city: a CSV per year and double-bar charts
#   (Delhi vs Mumbai) for each month with data.
# (5) Visuals/metrics: weekday vs weekend per-customer hour curves
#   (line plot + CSV), zero-usage share, top-10% concentration of consumption,
#   and city-scope peak-hour distribution (bar chart).
# (6) A coverage table (half-hours present vs expected) per city-year,
# (7) Distance analytics per city (nearest-neighbor and sampled pairwise
#   stats) with two histograms and a distance stats CSV.

# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sys
import math
import random
import binascii
from shapely.wkb import loads as wkb_loads
from sklearn.neighbors import BallTree as SKBallTree
from typing import Any, Dict, List, Optional, Tuple
# from datetime import date    # datetime,
import numpy as np
import polars as pl
import calendar
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import seaborn as sns

# MPI
try:
    from mpi4py import MPI  # type: ignore
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:   # allow non-MPI local run (SIZE=1)
    class _DummyComm:

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def bcast(self, x, root=0):
            return x

        def barrier(self):
            return None

        def gather(self, x, root=0):
            return [x]

    COMM = _DummyComm()
    RANK = 0
    SIZE = 1

matplotlib.use(backend="Agg")

# ─────────────────────────────────────────────────────────────────────────────
# Directories, Filenames, and Filepaths
# ────────────────────────────────────────────────────────────────────────────

# Directories
base_directory = os.path.join('..', '..')
data_directory = os.path.join(base_directory, 'data')
hitachi_data_directory = os.path.join(data_directory, 'hitachi')
meter_readings_directory = os.path.join(hitachi_data_directory,
                                        'meter_primary_files')
outputs_directory = os.path.join(base_directory, 'outputs')
outputs_metrics_directory = os.path.join(outputs_directory, 'metrics')
outputs_images_directory = os.path.join(outputs_directory, 'images')
outputs_tmp_directory = os.path.join(outputs_directory, '_tmp_mpi')

# Filenames and Filepaths

meter_file_names = {
    "2021": "meter_readings_2021_20250714_2015_formatted",
    "2022": "meter_readings_2022_20250714_2324_formatted",
    "2023": "meter_readings_2023_20250714_2039_formatted",
    "delhi_2021": "meter_readings_delhi_2021_20250714_2015_formatted",
    "delhi_2022": "meter_readings_delhi_2022_20250714_2324_formatted",
    "mumbai_2022": "meter_readings_mumbai_2022_20250714_2324_formatted",
    "mumbai_2023": "meter_readings_mumbai_2023_20250714_2039_formatted",
    "all_years": "meter_readings_all_years_20250714_formatted"
}

customer_filename = "customers_20250714_1401"
customer_filepath = os.path.join(hitachi_data_directory,
                                 customer_filename + ".parquet")

meter_file_paths = {key: os.path.join(meter_readings_directory,
                                      value + ".parquet")
                    for key, value in meter_file_names.items()}


# ────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
#
#   FUNCTIONS
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem + Logging helpers
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class Logger:
    """Logger to stdout + file."""

    def __init__(self, logfile: str) -> None:
        ensure_dir(os.path.dirname(logfile))
        self._fh = open(logfile, "w", encoding="utf-8")

    def log(self, msg: str) -> None:
        line = msg + "\n"
        sys.stdout.write(line)
        self._fh.write(line)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


ensure_dir(outputs_directory)
ensure_dir(outputs_metrics_directory)
ensure_dir(outputs_images_directory)
ensure_dir(outputs_tmp_directory)


LOG = Logger(os.path.join(outputs_directory,
                          'meter_mpi_coverage_analysis.log'))


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_plot(path: str) -> str:
    """
    Save current matplotlib plot; return the path for logging.

    Parameters:
    ----------
    path: str
        The file path where the plot will be saved.

    Returns:
    -------
    str
        The path to the saved plot file.
    """
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _is_numeric_dtype(dt: pl.DataType) -> bool:
    return dt in (
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    )


def align_for_concat(frames: list[pl.DataFrame]) -> list[pl.DataFrame]:
    """
    Align DataFrames for concatenation by ensuring they have the same columns.
    This function takes a list of Polars DataFrames and aligns them by adding
    missing columns with null values, ensuring that all DataFrames have the
    same schema.

    Parameters:
    ----------
    frames: list[pl.DataFrame]
        The list of DataFrames to align.

    Returns:
    -------
    list[pl.DataFrame]
        The aligned list of DataFrames.
    """
    if not frames:
        return frames

    # 1) union of column names (preserve first-seen order)
    all_cols: list[str] = []
    seen = set()
    for f in frames:
        for c in f.columns:
            if c not in seen:
                seen.add(c)
                all_cols.append(c)

    # 2) decide a target dtype per column from the frames' schemas
    target_dtype: dict[str, pl.DataType] = {}
    for f in frames:
        for c, dt in f.schema.items():
            if c not in target_dtype:
                if dt != pl.Null:
                    target_dtype[c] = dt
            else:
                t = target_dtype[c]
                if dt == t or dt == pl.Null:
                    continue
                # numeric vs numeric -> Float64
                if _is_numeric_dtype(t) and _is_numeric_dtype(dt):
                    target_dtype[c] = pl.Float64
                # date vs datetime -> Datetime
                elif (t == pl.Date and dt == pl.Datetime) or (
                        t == pl.Datetime and dt == pl.Date):
                    target_dtype[c] = pl.Datetime
                # fall back to Utf8 for other conflicts
                else:
                    target_dtype[c] = pl.Utf8

    # columns that never had a non-null type remain Null; that’s fine if all
    # shards are Null
    # (but we’ll still cast new placeholders to that target, if present)

    # 3) align each frame: add missing cols (None -> cast target), cast
    # mismatched cols, reorder
    aligned: list[pl.DataFrame] = []
    for f in frames:
        exprs = []
        # existing columns: cast if needed
        for c in f.columns:
            tgt = target_dtype.get(c)
            if tgt is not None and f.schema[c] != tgt:
                exprs.append(pl.col(c).cast(tgt, strict=False))
        if exprs:
            f = f.with_columns(exprs)

        # missing columns: add as None cast to target dtype (or Null if
        # unknown)
        missing = [c for c in all_cols if c not in f.columns]
        if missing:
            add_exprs = []
            for c in missing:
                tgt = target_dtype.get(c, pl.Null)
                add_exprs.append(pl.lit(None).cast(tgt).alias(c))
            f = f.with_columns(add_exprs)

        # reorder
        f = f.select(all_cols)
        aligned.append(f)

    return aligned


# ─────────────────────────────────────────────────────────────────────────────
# Time Utilities
# ─────────────────────────────────────────────────────────────────────────────


def days_in_month(
        y: int,
        m: int
) -> int:
    """
    Get the number of days in a month.

    Parameters:
    ----------
    y: int
        The year.
    m: int
        The month.

    Returns:
    -------
    int
        The number of days in the month.
    """
    return calendar.monthrange(y, m)[1]


def iso_year_week(dtseries: pl.Series) -> Tuple[pl.Series, pl.Series]:
    """
    Parse ISO year and week from a datetime series.

    Parameters:
    ----------
    dtseries: pl.Series
        The datetime series to parse.

    Returns:
    -------
    Tuple[pl.Series, pl.Series]
        A tuple containing the ISO year and week as separate series.
    """
    return (
        dtseries.dt.strftime("%G").cast(pl.Int32),
        dtseries.dt.strftime("%V").cast(pl.Int32),
    )


def add_time_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add time-based columns to a LazyFrame with a 'date' column.

    Parameters:
    ----------
    lf: pl.LazyFrame
        The LazyFrame to add time-based columns to.

    Returns:
    -------
    pl.LazyFrame
        The LazyFrame with added time-based columns.
    """
    return lf.with_columns([
        pl.col('date').dt.date().alias('date_day'),
        pl.col('date').dt.hour().alias('hour'),
        pl.col('date').dt.year().alias('year'),
        pl.col('date').dt.month().alias('month'),
        pl.col('date').dt.weekday().alias('weekday'),  # Mon=0..Sun=6
        ])


def summarize_daily(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Aggregate to daily totals and presence flags.

    Parameters:
    ----------
    lf: pl.LazyFrame
        The LazyFrame to summarize.

    Returns:
    -------
    pl.DataFrame
        The summarized DataFrame.
    """
    return (add_time_columns(lf)
            .group_by(['date_day'])
            .agg([
                pl.sum('value').alias('day_total'),
                pl.len().alias('rows'),
                (pl.col('value') > 0).sum().alias('nonzero_rows'),
            ])
            .sort('date_day')
            .collect()
            )


def extrapolate_period_from_sub(
        df_sub: pl.DataFrame,
        period: str
) -> pl.DataFrame:
    """
    Generic fill-by-subperiod extrapolation.
    - If period is 'week': df_sub must have columns: date_day.
    Compute ISO week (year,week) and treat subperiods=days (7).
    - If period is 'month': df_sub must have date_day; subperiods=days (28-31).
    - If period is 'year': df_sub must have year, month totals (supply
        via df_sub),
    Accept daily df and internally roll to months then fill 12 months.

    Returns a DataFrame with columns: [key cols], present_subperiods,
    total_subperiods, coverage_ratio, est_total, in_sample_total.

    Parameters
    ----------
    df_sub: pl.DataFrame
        The subset DataFrame to extrapolate from.
    period: str
        The time period to extrapolate to ('week', 'month', 'year').

    Returns:
    -------
    pl.DataFrame
        The extrapolated DataFrame.

    """
    if 'date_day' not in df_sub.columns:
        raise ValueError("df_sub must include date_day for extrapolation")

    df = df_sub

    if period == 'week':
        # roll daily to ISO-week level
        s_year, s_week = iso_year_week(pl.Series('date_day', df['date_day']))
        tmp = df.with_columns([
            s_year.alias('iso_year'),
            s_week.alias('iso_week'),
        ])

        weekly = (
            tmp.group_by(['iso_year', 'iso_week'])
            .agg([
                pl.sum('day_total').alias('in_sample_total'),
                pl.count('date_day').alias('present_subperiods'),
            ])
        )

        # 1) add constant first
        weekly = weekly.with_columns(pl.lit(7).alias('total_subperiods'))

        # 2) now you can reference it
        weekly = weekly.with_columns(
            (pl.col('present_subperiods') / pl.col('total_subperiods'))
            .alias('coverage_ratio')
        )

        # 3) compute averages and missing subperiods robustly
        weekly = weekly.with_columns([
            pl.when(pl.col('present_subperiods') > 0)
            .then(pl.col('in_sample_total') / pl.col('present_subperiods'))
            .otherwise(0.0)
            .alias('avg_present_sub'),
            (pl.col('total_subperiods') - pl.col('present_subperiods'))
            .alias('missing_sub'),
        ])

        # 4) filled total, clean up
        weekly = weekly.with_columns(
            (pl.col('in_sample_total') + pl.col('missing_sub'
                                                ) * pl.col('avg_present_sub'))
            .alias('est_total')
        ).drop(['avg_present_sub', 'missing_sub'])

        return weekly

    if period == 'month':
        tmp = df.with_columns([
            pl.col('date_day').dt.year().alias('year'),
            pl.col('date_day').dt.month().alias('month'),
        ])
        monthly = (
            tmp.group_by(['year', 'month'])
            .agg([
                pl.sum('day_total').alias('in_sample_total'),
                pl.count('date_day').alias('present_subperiods'),
            ])
        )
        # attach total days in that month
        total_days = [days_in_month(int(y), int(m))
                      for y, m in zip(monthly['year'], monthly['month'])]
        monthly = monthly.with_columns([
            pl.Series('total_subperiods', total_days),
        ]).with_columns([
            (pl.col('present_subperiods') /
             pl.col('total_subperiods')).alias('coverage_ratio')
        ])
        monthly = monthly.with_columns([
            (pl.col('in_sample_total') / pl.col('present_subperiods')
             ).alias('avg_present_sub').fill_null(0.0),
            (pl.col('total_subperiods') - pl.col('present_subperiods')
             ).alias('missing_sub'),
        ]).with_columns([
            (pl.col('in_sample_total') + pl.col('missing_sub') * pl.col(
                'avg_present_sub')).alias('est_total')
        ]).drop('avg_present_sub', 'missing_sub')
        return monthly

    if period == 'year':
        # 1) monthly via day-fill, then 2) sum to year
        monthly = extrapolate_period_from_sub(df, period='month')
        yearly = (
            monthly.group_by('year')
            .agg([
                pl.sum('in_sample_total').alias('in_sample_total'),
                pl.sum('est_total').alias('est_total'),
                pl.sum('present_subperiods').alias('present_subperiods'),
                pl.sum('total_subperiods').alias('total_subperiods'),
            ])
            .with_columns([
                (pl.col('present_subperiods') / pl.col('total_subperiods')
                 ).alias('coverage_ratio')
            ])
        )
        return yearly

    raise ValueError(f"Unsupported period: {period}")

# ─────────────────────────────────────────────────────────────────────────────
# Individual file stats
# ─────────────────────────────────────────────────────────────────────────────


def individual_file_stats(
        path: str,
        file_key: str
) -> pl.DataFrame:
    """
    Generate individual file statistics for a given meter readings file.
    This function computes various statistics for the specified file, including
    customer counts, date ranges, and per-customer metrics.

    Parameters:
    ----------
    path: str
        The path to the Parquet file.
    file_key: str
        A unique identifier for the file.

    Returns:
    -------
    pl.DataFrame
        A DataFrame containing the individual file statistics.
    """
    lf = pl.scan_parquet(path)
    # Schema expectations: ca_id, city, date, value
    # Counts
    base = (
        lf.select([
            pl.col('ca_id').n_unique().alias('n_customers'),
            pl.len().alias('n_halfhour_rows'),
            (pl.col('value') > 0).sum().alias('n_halfhour_rows_nonzero'),
            pl.col('date').min().alias('date_min'),
            pl.col('date').max().alias('date_max'),
        ]).collect()
    )
    n_customers = int(base['n_customers'][0])
    n_rows = int(base['n_halfhour_rows'][0])
    n_rows_nz = int(base['n_halfhour_rows_nonzero'][0])
    hours_equiv = n_rows * 0.5
    days_equiv = hours_equiv / 24.0
    hours_equiv_nz = n_rows_nz * 0.5
    days_equiv_nz = hours_equiv_nz / 24.0

    # Per-customer half-hour counts
    per_cust = (
        lf.group_by('ca_id').agg(pl.len().alias('hh_count')).collect()
    )
    per_customer_halfhour_mean = float(per_cust['hh_count'].mean()
                                       ) if per_cust.height else 0.0
    per_customer_halfhour_median = float(per_cust['hh_count'].median()
                                         ) if per_cust.height else 0.0

    # Per-customer per city averages (half-hours → hours/days)
    city_per_cust = (
        lf.group_by(['city', 'ca_id']).agg(
            pl.len().alias('hh')).group_by(
                'city').agg(pl.mean('hh').alias('mean_hh')).collect()
    )

    # Base row
    row = {
        'file_key': file_key,
        'file_path': path,
        'date_min': base['date_min'][0],
        'date_max': base['date_max'][0],
        'n_customers': n_customers,
        'n_halfhour_rows': n_rows,
        'hours_equiv': hours_equiv,
        'days_equiv': days_equiv,
        'n_halfhour_rows_nonzero': n_rows_nz,
        'hours_equiv_nonzero': hours_equiv_nz,
        'days_equiv_nonzero': days_equiv_nz,
        'per_customer_halfhour_mean': per_customer_halfhour_mean,
        'per_customer_halfhour_median': per_customer_halfhour_median,
        'per_customer_hours_mean': per_customer_halfhour_mean * 0.5,
        'per_customer_hours_median': per_customer_halfhour_median * 0.5,
        'per_customer_days_mean': per_customer_halfhour_mean * 0.5 / 24.0,
        'per_customer_days_median': per_customer_halfhour_median * 0.5 / 24.0,
    }

    # Add dynamic city columns
    for c, hh_mean in zip(city_per_cust['city'].to_list(),
                          city_per_cust['mean_hh'].to_list()):
        safe = str(c).strip().lower().replace(' ', '_')
        row[f'city={safe}__per_customer_halfhour_mean'] = float(hh_mean)
        row[f'city={safe}__per_customer_hours_mean'] = float(hh_mean) * 0.5
        row[f'city={safe}__per_customer_days_mean'] = float(hh_mean
                                                            ) * 0.5 / 24.0

    return pl.DataFrame([row])

# ─────────────────────────────────────────────────────────────────────────────
# Main File Metrics
# ─────────────────────────────────────────────────────────────────────────────


def scope_filter(
        lf: pl.LazyFrame,
        city: Optional[str],
        year: Optional[int]
) -> pl.LazyFrame:
    """
    Filter the LazyFrame by city and year.

    Parameters
    ----------
    lf: pl.LazyFrame
        The LazyFrame to filter.
    city: Optional[str]
        The city to filter by.
    year: Optional[int]
        The year to filter by.

    Returns
    -------
    pl.LazyFrame
        The filtered LazyFrame.
    """
    if city:
        lf = lf.filter(pl.col('city') == city)
    if year:
        lf = lf.filter(pl.col('date').dt.year() == year)
    return lf


def compute_hour_stats(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Compute hourly statistics from the LazyFrame.

    Parameters
    ----------
    lf: pl.LazyFrame
        The LazyFrame containing the data.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the computed hourly statistics.
    """
    # Aggregate to hour totals across all customers
    hourly = (lf.with_columns(
            pl.col('date').dt.truncate('1h').alias('hour_ts')
        ).group_by('hour_ts')
        .agg(pl.sum('value').alias('hour_total'))
        .sort('hour_ts')
        .collect()
    )
    if hourly.height == 0:
        return pl.DataFrame({'mean': [None], 'median': [None],
                             'n_periods': [0]})
    return pl.DataFrame({
        'mean': [float(hourly['hour_total'].mean())],
        'median': [float(hourly['hour_total'].median())],
        'n_periods': [int(hourly.height)],
    })


def compute_week_month_year_stats(lf: pl.LazyFrame) -> Dict[str, pl.DataFrame]:
    """
    Compute weekly, monthly, and yearly statistics from the LazyFrame.

    Parameters
    ----------
    lf: pl.LazyFrame
        The LazyFrame containing the data.

    Returns
    -------
    Dict[str, pl.DataFrame]
        A dictionary containing the computed statistics.
    """
    # Daily roll-up first
    daily = summarize_daily(lf)
    # Week
    week = extrapolate_period_from_sub(daily, period='week')
    week_stats = pl.DataFrame({
        'mean_total': [float(week['est_total'].mean()
                             ) if week.height else None],
        'median_total': [float(week['est_total'].median()
                               ) if week.height else None],
        'n_periods': [int(week.height)],
        'coverage_mean': [float(week['coverage_ratio'].mean()
                                ) if week.height else None],
    })
    # Month
    month = extrapolate_period_from_sub(daily, period='month')
    month_stats = pl.DataFrame({
        'mean_total': [float(month['est_total'].mean()
                             ) if month.height else None],
        'median_total': [float(month['est_total'].median()
                               ) if month.height else None],
        'n_periods': [int(month.height)],
        'coverage_mean': [float(month['coverage_ratio'].mean()
                                ) if month.height else None],
    })
    # Year
    year = extrapolate_period_from_sub(daily, period='year')
    year_stats = pl.DataFrame({
        'mean_total': [float(year['est_total'].mean()
                             ) if year.height else None],
        'median_total': [float(year['est_total'].median()
                               ) if year.height else None],
        'n_periods': [int(year.height)],
        'coverage_mean': [float(year['coverage_ratio'].mean()
                                ) if year.height else None],
    })
    # Annualized via ISO weeks × 52
    iso_week_means = float(week['est_total'].mean()) if week.height else None
    annualized = pl.DataFrame({
        'annualized_total_mean':
        [iso_week_means * 52 if iso_week_means is not None else None],
    })
    return {
        'week_stats': week_stats,
        'month_stats': month_stats,
        'year_stats': year_stats,
        'annualized': annualized,
        'week_table': week,
        'month_table': month,
        'year_table': year,
    }


def per_customer_period_distributions(
        lf: pl.LazyFrame,
        freq: str
) -> pl.DataFrame:
    """
    Return per-customer period totals (extrapolated for week/month/year).

    Parameters
    ----------
    lf: pl.LazyFrame
        The LazyFrame containing the data.
    freq: str
        The frequency for the period ('hour', 'day', 'week', 'month', 'year').

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the per-customer period totals.
    """
    if freq == 'hour':
        # lazy -> collect at end
        return (
            lf.with_columns(pl.col('date').dt.truncate('1h').alias('period'))
              .group_by(['period', 'ca_id'])
              .agg(pl.sum('value').alias('usage'))
              .collect()
        )

    # Build daily per-customer totals eagerly once
    daily_pc = (
        add_time_columns(lf)
        .group_by(['ca_id', 'date_day'])
        .agg(pl.sum('value').alias('day_total'))
        .collect()
    )

    if freq == 'week':
        s_year, s_week = iso_year_week(pl.Series('date_day',
                                                 daily_pc['date_day']))
        tmp = daily_pc.with_columns([s_year.alias('iso_year'),
                                     s_week.alias('iso_week')])
        pc_week = (
            tmp.group_by(['ca_id', 'iso_year', 'iso_week'])
               .agg([
                   pl.sum('day_total').alias('in_sample_total'),
                   pl.count('date_day').alias('present_days'),
               ])
        )
        # add first, then use
        pc_week = pc_week.with_columns(pl.lit(7).alias('total_days'))
        pc_week = pc_week.with_columns([
            pl.when(pl.col('present_days') > 0)
              .then(pl.col('in_sample_total') / pl.col('present_days'))
              .otherwise(0.0)
              .alias('avg_present_day'),
            (pl.col('total_days') - pl.col('present_days')
             ).alias('missing_days'),
        ])
        pc_week = (
            pc_week.with_columns(
                (pl.col('in_sample_total'
                        ) + pl.col('missing_days'
                                   ) * pl.col('avg_present_day'))
                .alias('usage')
            )
            .select(['ca_id', 'iso_year', 'iso_week', 'usage'])
            .rename({'iso_year': 'year', 'iso_week': 'week'})
        )
        return pc_week

    if freq == 'month':
        # 1) add (year, month)
        tmp = daily_pc.with_columns([
            pl.col('date_day').dt.year().alias('year'),
            pl.col('date_day').dt.month().alias('month'),
        ])

        # 2) per-customer in-sample + present days
        pc_month = (
            tmp.group_by(['ca_id', 'year', 'month'])
               .agg([
                   pl.sum('day_total').alias('in_sample_total'),
                   pl.count('date_day').alias('present_days'),
               ])
        )

        # 3) build a small (year, month) -> total_days dimension and join
        ym = (
            tmp.select(['year', 'month']).unique().sort(['year', 'month'])
        )
        total_days_list = [
            days_in_month(int(y), int(m))
            for y, m in zip(ym['year'], ym['month'])
        ]
        ym = ym.with_columns(pl.Series('total_days', total_days_list,
                                       dtype=pl.Int16))

        pc_month = pc_month.join(ym, on=['year', 'month'], how='left')

        # 4) fill the missing days and compute final monthly usage
        pc_month = pc_month.with_columns([
            pl.when(pl.col('present_days') > 0)
              .then(pl.col('in_sample_total') / pl.col('present_days'))
              .otherwise(0.0)
              .alias('avg_present_day'),
            (pl.col('total_days') - pl.col('present_days')
             ).alias('missing_days'),
        ])

        pc_month = pc_month.with_columns(
            (pl.col('in_sample_total') + pl.col('missing_days'
                                                ) * pl.col('avg_present_day'))
            .alias('usage')
        ).select(['ca_id', 'year', 'month', 'usage'])

        return pc_month

    if freq == 'year':
        pc_month = per_customer_period_distributions(lf, freq='month')  # eager
        pc_year = pc_month.group_by(['ca_id', 'year']).agg(
            pl.sum('usage').alias('usage')
        )
        return pc_year

    raise ValueError(f"Unsupported freq: {freq}")


def build_bigfile_scope_metrics(
        all_path: str,
        scope_name: str,
        city: Optional[str],
        year: Optional[int]
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Compute stats for one scope; return tidy metrics rows and list
    of output file paths.

    Parameters
    ----------
    all_path: str
        Path to the input data file.
    scope_name: str
        Name of the scope to analyze.
    city: Optional[str]
        Name of the city to filter by.
    year: Optional[int]
        Year to filter by.

    Returns
    -------
    Tuple[pl.DataFrame, List[str]]
        A tuple containing the metrics DataFrame and a list of output file
        paths.
    """
    out_paths: List[str] = []
    lf = pl.scan_parquet(all_path)
    lf = scope_filter(lf, city=city, year=year)

    # 1) Hour stats (totals across customers)
    hour_stats = compute_hour_stats(lf)
    hour_rows = pl.DataFrame([{
        'scope': scope_name, 'freq': 'hour', 'metric': 'total_usage',
        'stat': 'mean', 'value': hour_stats['mean'][0],
        'n_periods': hour_stats['n_periods'][0],
        'coverage_note': 'observed hours only'
    }, {
        'scope': scope_name, 'freq': 'hour', 'metric': 'total_usage',
        'stat': 'median', 'value': hour_stats['median'][0],
        'n_periods': hour_stats['n_periods'][0],
        'coverage_note': 'observed hours only'
    }])

    # 2) Week/Month/Year totals with coverage-aware fill
    wmy = compute_week_month_year_stats(lf)
    rows = []
    for freq_key, tbl_key in [('week', 'week_stats'),
                              ('month', 'month_stats'),
                              ('year', 'year_stats')]:
        tbl = wmy[tbl_key]
        rows.extend([
            {'scope': scope_name, 'freq': freq_key, 'metric': 'total_usage',
             'stat': 'mean', 'value': tbl['mean_total'][0],
             'n_periods': tbl['n_periods'][0],
             'coverage_note': f'{freq_key} via subperiod fill'},
            {'scope': scope_name, 'freq': freq_key, 'metric': 'total_usage',
             'stat': 'median', 'value': tbl['median_total'][0],
             'n_periods': tbl['n_periods'][0],
             'coverage_note': f'{freq_key} via subperiod fill'},
        ])
    # Annualized via ISO week mean × 52
    rows.append({'scope': scope_name, 'freq': 'year_annualized',
                 'metric': 'total_usage', 'stat': 'mean',
                 'value': wmy['annualized']['annualized_total_mean'][0],
                 'n_periods': wmy['week_stats']['n_periods'][0],
                 'coverage_note': 'annualized from weekly mean ×52'})

    # 3) Per-customer period distributions → mean/median across
    # customers of period totals
    pc_rows = []
    for freq in ['hour', 'week', 'month', 'year']:
        pc = per_customer_period_distributions(lf, freq=freq)
        if pc.height == 0:
            pc_rows.extend([
                {'scope': scope_name, 'freq': freq, 'metric':
                 'per_customer_usage', 'stat': 'mean', 'value': None,
                 'n_periods': 0, 'coverage_note': 'insufficient data'},
                {'scope': scope_name, 'freq': freq, 'metric':
                 'per_customer_usage', 'stat': 'median', 'value': None,
                 'n_periods': 0, 'coverage_note': 'insufficient data'},
            ])
        else:
            # For per-customer, aggregate by period first (mean across
            # customers per period),
            # then summarize across periods.
            period_cols = [c for c in pc.columns if c not in ('usage',
                                                              'ca_id')]
            per_period = pc.group_by(
                period_cols).agg(pl.mean('usage').alias('per_customer_mean'))
            pc_rows.extend([
                {'scope': scope_name, 'freq': freq,
                 'metric': 'per_customer_usage', 'stat': 'mean',
                 'value': float(per_period['per_customer_mean'].mean()),
                 'n_periods': int(per_period.height),
                 'coverage_note': 'per-customer mean per period'},
                {'scope': scope_name, 'freq': freq,
                 'metric': 'per_customer_usage', 'stat': 'median',
                 'value': float(per_period['per_customer_mean'].median()),
                 'n_periods': int(per_period.height),
                 'coverage_note': 'per-customer mean per period'},
            ])
            # Save distribution CSVs
            distr_path = os.path.join(
                outputs_metrics_directory,
                f"{scope_name}__per_customer_usage__{freq}.csv")
            per_period.sort(per_period.columns).write_csv(distr_path)
            out_paths.append(distr_path)
            # Plots (hist + boxplot) — rank 0 only
            if RANK == 0 and per_period.height > 1:
                try:
                    plt.figure(figsize=(10, 6))
                    plt.hist(
                        per_period['per_customer_mean'].to_numpy(),
                        bins=30, edgecolor='black', alpha=0.75)
                    plt.title(
                        f"{scope_name} — per-customer usage per "
                        f"{freq} (mean across customers)")
                    plt.xlabel("kWh")
                    plt.ylabel("#periods")
                    save_plot(os.path.join(
                        outputs_images_directory,
                        f"{scope_name}__per_customer_usage__{freq}__hist.png"))

                    plt.figure(figsize=(7, 6))
                    plt.boxplot(per_period['per_customer_mean'].to_numpy(),
                                vert=True)
                    plt.title(f"{scope_name} — per-customer usage per {freq}")
                    plt.ylabel("kWh")
                    save_plot(
                        os.path.join(
                            outputs_images_directory,
                            (f"{scope_name}__per_customer_usage__"
                             f"{freq}__boxplot.png")))
                except Exception as e:
                    LOG.log(f"plot error ({scope_name}, {freq}): {e}")

    tidy = pl.DataFrame(rows)  # totals
    tidy = pl.concat([hour_rows, tidy], how='vertical_relaxed')
    tidy = pl.concat([tidy, pl.DataFrame(pc_rows)], how='vertical_relaxed')

    # Quick wins
    try:
        # Zero-usage share (overall & per city if applicable)
        z_total = lf.select([
            (pl.len()).alias('rows'),
            (pl.col('value') == 0).sum().alias('zero_rows'),
        ]).collect()
        zero_share = float(z_total['zero_rows'][0] / max(1,
                                                         z_total['rows'][0]))
        z_row = pl.DataFrame([{'scope': scope_name, 'freq': 'all',
                               'metric': 'zero_usage_share_rows',
                               'stat': 'level', 'value': zero_share,
                               'n_periods': int(z_total['rows'][0]),
                               'coverage_note': ''}])
        tidy = pl.concat([tidy, z_row], how='vertical_relaxed')
    except Exception as e:
        LOG.log(f"zero-usage quick metric error ({scope_name}): {e}")

    # Weekend/weekday split of per-customer usage by hour
    try:
        ww = (
            add_time_columns(lf)
            .with_columns((pl.col('weekday') >= 5).alias('is_weekend'))
            .group_by(['is_weekend', 'hour', 'ca_id'])
            .agg(pl.mean('value').alias('avg'))
            .group_by(['is_weekend', 'hour'])
            .agg(pl.mean('avg').alias('per_customer_hour_mean'))
            .sort(['is_weekend', 'hour'])
            .collect()
        )
        ww_path = os.path.join(
            outputs_metrics_directory,
            f"{scope_name}__weekday_weekend_per_customer_by_hour.csv")
        ww.write_csv(ww_path)
        out_paths.append(ww_path)
        if RANK == 0:
            plt.figure(figsize=(12, 6))
            for flag, label in [(False, 'Weekday'), (True, 'Weekend')]:
                sub = ww.filter(pl.col('is_weekend') == flag)
                if sub.height:
                    plt.plot(
                        sub['hour'].to_numpy(),
                        sub['per_customer_hour_mean'].to_numpy(),
                        marker='o', label=label)
            plt.xlabel('Hour of day')
            plt.ylabel('kWh (per-customer mean)')
            plt.title(
                f"{scope_name} — Weekday vs Weekend per-customer by hour")
            plt.legend()
            save_plot(os.path.join(
                outputs_images_directory,
                f"{scope_name}__weekday_weekend_per_customer_by_hour.png"))
    except Exception as e:
        LOG.log(f"weekend/weekday split error ({scope_name}): {e}")

    # Top-10% concentration
    try:
        per_cust_total = lf.group_by('ca_id').agg(
            pl.sum('value').alias('total_kwh')
            ).collect().sort('total_kwh', descending=True)
        if per_cust_total.height:
            n = per_cust_total.height
            top_n = max(1, int(math.ceil(0.10 * n)))
            top_sum = float(per_cust_total['total_kwh'][:top_n].sum())
            all_sum = float(per_cust_total['total_kwh'].sum())
            share = top_sum / all_sum if all_sum > 0 else None
            t10_row = pl.DataFrame([{'scope': scope_name, 'freq': 'all',
                                     'metric': 'top10pct_share',
                                     'stat': 'level', 'value': share,
                                     'n_periods': n, 'coverage_note': ''}])
            tidy = pl.concat([tidy, t10_row], how='vertical_relaxed')
    except Exception as e:
        LOG.log(f"top10% error ({scope_name}): {e}")

    # Peak hour spread (only for city-specific scopes)
    if city is not None:
        try:
            # For each customer & day, find hour of max usage; then
            # distribution of that hour
            daily_hours = (
                add_time_columns(lf)
                .group_by(['ca_id', 'date_day', 'hour'])
                .agg(pl.sum('value').alias('hour_total'))
                .collect()
            )
            # pick max hour per (ca_id, date_day)
            daily_peak = (
                daily_hours.sort(['ca_id', 'date_day', 'hour_total'],
                                 descending=[False, False, True])
                .group_by(['ca_id', 'date_day'])
                .agg(pl.first('hour').alias('peak_hour'))
            )
            peak_dist = daily_peak.group_by(
                'peak_hour'
                ).agg(pl.len().alias('days')).sort('peak_hour')
            peak_path = os.path.join(
                outputs_metrics_directory,
                f"{scope_name}__peak_hour_distribution.csv")
            peak_dist.write_csv(peak_path)
            out_paths.append(peak_path)
            if RANK == 0:
                plt.figure(figsize=(10, 6))
                plt.bar(peak_dist['peak_hour'].to_numpy(),
                        peak_dist['days'].to_numpy())
                plt.xlabel('Hour of day')
                plt.ylabel('#customer-days')
                plt.title(
                        f"{scope_name} — Distribution of peak hour"
                        f" (per customer-day)"
                        )
                save_plot(os.path.join(
                    outputs_images_directory,
                    f"{scope_name}__peak_hour_distribution.png"))
        except Exception as e:
            LOG.log(f"peak-hour spread error ({scope_name}): {e}")

    return tidy, out_paths


# Monthly double-bar plots per year (Delhi vs Mumbai)

def monthly_city_totals_plot(
        all_path: str,
        year: int
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate monthly total consumption plots per city for a given year.

    Parameters
    ----------
    all_path : str
        Path to the input data file.
    year : int
        The year for which to generate the plots.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        Paths to the generated plot images or None if not created.

    """
    lf = pl.scan_parquet(all_path).filter(pl.col('date').dt.year() == year)
    # compute daily totals per city
    daily = (
        add_time_columns(lf)
        .group_by(['city', 'date_day'])
        .agg(pl.sum('value').alias('day_total'))
        .collect()
    )
    if daily.height == 0:
        return None, None

    # Extrapolate months per city
    def city_month(
            df_city: pl.DataFrame,
            cname: str
    ) -> pl.DataFrame:
        """
        Extrapolate monthly totals for a specific city.

        Parameters
        ----------
        df_city : pl.DataFrame
            The DataFrame containing daily totals for the city.
        cname : str
            The name of the city.

        Returns
        -------
        pl.DataFrame
            A DataFrame with the extrapolated monthly totals.
        """
        tmp = df_city.with_columns([
            pl.col('date_day').dt.year().alias('year'),
            pl.col('date_day').dt.month().alias('month'),
        ])
        m = (
            tmp.group_by(['year', 'month'])
               .agg([
                  pl.sum('day_total').alias('in_sample_total'),
                  pl.count('date_day').alias('present_days'),
                ])
        )
        m = m.with_columns([
            pl.struct(['year', 'month']
                      ).map_elements(lambda s: days_in_month(
                          int(s['year']), int(s['month'])),
                          return_dtype=pl.Int16
                          ).alias('total_days'),
            (pl.col('in_sample_total') /
             pl.col('present_days')
             ).alias('avg_present_day').fill_null(0.0),
        ])
        m = m.with_columns([
            (pl.col('in_sample_total') + (
                pl.col('total_days')-pl.col('present_days')
                )*pl.col('avg_present_day')).alias('est_total'),
            (pl.col('present_days')/pl.col('total_days')
             ).alias('coverage_ratio'),
        ])
        m = m.with_columns(pl.lit(cname).alias('city')
                           ).select(['city', 'year', 'month',
                                     'est_total', 'coverage_ratio']
                                    ).sort('month')
        return m

    cities = daily['city'].unique().to_list()
    parts = []
    for cname in cities:
        parts.append(city_month(daily.filter(pl.col('city') == cname),
                                str(cname)))
    month_city = pl.concat(parts, how='vertical_relaxed'
                           ) if parts else pl.DataFrame([])

    # Save CSV
    csv_path = os.path.join(
        outputs_metrics_directory, f"monthly_totals_city_{year}.csv")
    month_city.write_csv(csv_path)

    # Plot: double-bar for Delhi & Mumbai if present
    if RANK == 0 and month_city.height:
        try:
            pivot = month_city.pivot(
                values='est_total', index='month',
                on='city', aggregate_function='first'
                ).sort('month')
            months = pivot['month'].to_list()
            cities_sorted = sorted([c for c in pivot.columns if c != 'month'])
            width = 0.35
            x = np.arange(len(months))
            plt.figure(figsize=(12, 6))
            for i, cname in enumerate(cities_sorted):
                vals = np.array(pivot[cname]
                                ) if cname in pivot.columns else np.zeros(
                                    len(months))
                plt.bar(x + i*width, vals, width, label=cname)
            plt.xticks(x + width * (len(cities_sorted)-1)/2,
                       [calendar.month_abbr[m] for m in months])
            plt.ylabel('kWh (est total)')
            plt.title(f'Monthly total consumption by city — {year}')
            plt.legend()
            img_path = os.path.join(outputs_images_directory,
                                    f"monthly_city_totals_{year}.png")
            save_plot(img_path)
            return csv_path, img_path
        except Exception as e:
            LOG.log(f"monthly plot error ({year}): {e}")
    return csv_path, None


# Coverage metrics per city-year (share of half-hours present vs expected)

def coverage_halfhours_city_year(all_path: str) -> pl.DataFrame:
    """
    Compute coverage metrics per city-year based on half-hourly data.

    Parameters
    ----------
    all_path : str
        Path to the input data (Parquet file).

    Returns
    -------
    pl.DataFrame
        A DataFrame with coverage metrics per city-year.
    """
    lf = pl.scan_parquet(all_path)
    # For each city-year, count half-hours present
    df = (
        add_time_columns(lf)
        .group_by([pl.col('city'),
                   pl.col('date').dt.year().alias('year'),
                   pl.col('date').dt.date().alias('date_day')])
        .agg(pl.len().alias('hh_rows'))
        .group_by(['city', 'year'])
        .agg(pl.sum('hh_rows').alias('present_halfhours'))
        .collect()
    )
    # expected = days_in_year * 48
    exp = []
    for cy, y in zip(df['city'], df['year']):
        days = 366 if calendar.isleap(int(y)) else 365
        exp.append(days * 48)
    df = df.with_columns(pl.Series('expected_halfhours', exp))
    df = df.with_columns(
        (pl.col('present_halfhours')/pl.col('expected_halfhours')
         ).alias('coverage_ratio'))
    return df


# Distance metrics (optional; needs customer WKB + shapely)

def wkb_to_coords(hex_wkb: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Convert WKB hex string to (lat, lon) coordinates.

    Parameters
    ----------
    hex_wkb : str
        The WKB hex string to convert.

    Returns
    -------
    Tuple[Optional[float], Optional[float]]
        The (lat, lon) coordinates, or (None, None) if conversion fails.
    """
    if wkb_loads is None:
        return None, None
    try:
        geom = wkb_loads(binascii.unhexlify(hex_wkb))
        # Return lat, lon if Point in lon,lat
        return float(geom.y), float(geom.x)
    except Exception:
        return None, None


def haversine_km(
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
) -> float:
    """
    Compute the Haversine distance between two (lat, lon) points in kilometers.

    Returns:
    --------
        float: The Haversine distance in kilometers.

    """
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))


def city_distance_metrics(
        all_path: str,
        customers_path: str,
        city: str,
        pair_sample: int = 200_000,
        seed: int = 42
) -> Tuple[Optional[pl.DataFrame], List[str]]:
    """
    Compute distance-based metrics for customers in a given city.

    Parameters
    ----------
    all_path : str
        Path to the input data (Parquet file).
    customers_path : str
        Path to the customers data (Parquet file).
    city : str
        The city to filter customers by.
    pair_sample : int
        The number of random pairs to sample for distance metrics.
    seed : int
        The random seed for reproducibility.

    Returns
    -------
    Tuple[Optional[pl.DataFrame], List[str]]
        A tuple containing the distance metrics DataFrame (if available) and a
        list of output paths.
    """
    out_paths: List[str] = []
    if wkb_loads is None:
        LOG.log("shapely not available; skipping distance metrics")
        return None, out_paths

    # Which customers exist in data for that city?
    ca = (
        pl.scan_parquet(all_path)
        .filter(pl.col('city') == city)
        .select(pl.col('ca_id').unique())
        .collect()
        .get_column('ca_id')
        .to_list()
    )
    if not ca:
        return None, out_paths

    # Load customer coords
    cust = pl.read_parquet(customers_path)
    if 'location' not in cust.columns:
        # try other common names
        for alt in ['wkb', 'geom', 'geometry']:
            if alt in cust.columns:
                cust = cust.rename({alt: 'geom_wkb'})
                break
    if 'location' not in cust.columns:
        LOG.log("No 'location' column in customers file; "
                "skipping distance metrics")
        return None, out_paths

    cust_city = cust.filter(pl.col('city') == city)
    # Decode coords
    coords = [wkb_to_coords(x) for x in cust_city['location'].to_list()]
    lats = [c[0] for c in coords if c[0] is not None]
    lons = [c[1] for c in coords if c[1] is not None]
    ids = [i for i, c in zip(cust_city['id'].to_list(),
                             coords) if c[0] is not None]

    if len(ids) < 2:
        return None, out_paths

    # Nearest-neighbor distances
    nn_stats = {}
    try:
        if SKBallTree is not None:
            X = np.vstack([np.radians(lats), np.radians(lons)]).T
            tree = SKBallTree(X, metric='haversine')
            dists, _ = tree.query(X, k=2)  # nearest incl self
            nn_km = dists[:, 1] * 6371.0088
        else:
            # Fallback O(n^2) (only safe for small n)
            n = len(lats)
            nn_km = []
            for i in range(n):
                best = float('inf')
                for j in range(n):
                    if i == j:
                        continue
                    d = haversine_km(lats[i], lons[i],
                                     lats[j], lons[j])
                    if d < best:
                        best = d
                nn_km.append(best)
            nn_km = np.array(nn_km)
        nn_stats = {
            'nn_mean_km': float(np.mean(nn_km)),
            'nn_median_km': float(np.median(nn_km)),
            'nn_min_km': float(np.min(nn_km)),
            'nn_max_km': float(np.max(nn_km)),
        }
        # Plot
        if RANK == 0:
            plt.figure(figsize=(10, 6))
            plt.hist(nn_km, bins=40)
            plt.title(f"{city} — Nearest-neighbor distance (km)")
            plt.xlabel('km')
            plt.ylabel('#customers')
            save_plot(os.path.join(
                outputs_images_directory, f"{city}__nn_distance_hist.png"))
    except Exception as e:
        LOG.log(f"NN distance error ({city}): {e}")

    # Random pair sample for overall distances
    pair_stats = {}
    try:
        rng = random.Random(seed)
        n = len(lats)
        m = min(pair_sample, n*(n-1)//2)
        if m > 0:
            # sample index pairs without replacement (approx by rejection)
            pairs = set()
            while len(pairs) < m:
                i = rng.randrange(0, n)
                j = rng.randrange(0, n)
                if i < j:
                    pairs.add((i, j))
            dlist = [haversine_km(lats[i],
                                  lons[i],
                                  lats[j], lons[j]) for i, j in pairs]
            arr = np.array(dlist)
            pair_stats = {
                'pair_mean_km': float(np.mean(arr)),
                'pair_median_km': float(np.median(arr)),
                'pair_min_km': float(np.min(arr)),
                'pair_max_km': float(np.max(arr)),
                'pair_sample_size': int(len(arr)),
            }
            if RANK == 0:
                plt.figure(figsize=(10, 6))
                plt.hist(arr, bins=40)
                plt.title(f"{city} — Pairwise distance sample (km)")
                plt.xlabel('km')
                plt.ylabel('#pairs')
                save_plot(os.path.join(
                    outputs_images_directory,
                    f"{city}__pair_distance_hist.png"))
    except Exception as e:
        LOG.log(f"pair distance error ({city}): {e}")

    row = {
        'city': city,
        'n_customers': len(ids),
        **nn_stats,
        **pair_stats,
    }
    return pl.DataFrame([row]), out_paths

# ─────────────────────────────────────────────────────────────────────────────
# MPI task orchestration
# ─────────────────────────────────────────────────────────────────────────────


def distribute(items: List[Any]) -> List[Any]:
    """Round-robin split across MPI ranks."""
    return [x for idx, x in enumerate(items) if idx % SIZE == RANK]


def main() -> None:
    try:
        random.seed(42)
        # 1) Individual files stats (except all_years) → single CSV
        indiv_keys = [k for k in meter_file_paths.keys() if k != 'all_years']
        indiv_tasks = [(k, meter_file_paths[k]) for k in indiv_keys]
        my_indiv = distribute(indiv_tasks)
        # indiv_rows: List[pl.DataFrame] = []
        for key, path in my_indiv:
            if not os.path.exists(path):
                LOG.log(f"missing file: {path}")
                continue
            LOG.log(f"individual stats: {key}")
            try:
                df = individual_file_stats(path, key)
                tmp_path = os.path.join(
                    outputs_tmp_directory,
                    f"indiv_{key}__rank{RANK}.parquet")
                df.write_parquet(tmp_path)
            except Exception as e:
                LOG.log(f"individual stats error ({key}): {e}")

        COMM.barrier()

        if RANK == 0:
            combined = []
            for fn in sorted(os.listdir(outputs_tmp_directory)):
                if fn.startswith('indiv_') and fn.endswith('.parquet'):
                    combined.append(pl.read_parquet(
                        os.path.join(outputs_tmp_directory, fn)))
            if combined:
                # align_for_concat is optional with parquet
                combined = align_for_concat(combined)
                final_indiv = pl.concat(combined, how='vertical')
                final_path = os.path.join(
                    outputs_metrics_directory, 'individual_files_stats.csv')
                final_indiv.write_csv(final_path)
                LOG.log(f"wrote {final_path}")

        # 2) Bigfile scopes
        all_path = meter_file_paths['all_years']
        scopes: List[Tuple[str, Optional[str], Optional[int]]] = [
            ('all', None, None),
            ('delhi', 'delhi', None),
            ('mumbai', 'mumbai', None),
            ('year_2021', None, 2021),
            ('year_2022', None, 2022),
            ('year_2023', None, 2023),
            ('delhi_2021', 'delhi', 2021),
            ('delhi_2022', 'delhi', 2022),
            ('mumbai_2022', 'mumbai', 2022),
            ('mumbai_2023', 'mumbai', 2023),
        ]
        my_scopes = distribute(scopes)
        for scope_name, city, yr in my_scopes:
            LOG.log(f"scope metrics: {scope_name}")
            try:
                tidy, _paths = build_bigfile_scope_metrics(
                    all_path, scope_name, city, yr)
                tmp_path = os.path.join(
                    outputs_tmp_directory,
                    f"big_{scope_name}__rank{RANK}.parquet")
                tidy.write_parquet(tmp_path)
            except Exception as e:
                LOG.log(f"scope error ({scope_name}): {e}")

        COMM.barrier()
        # 5) Distance metrics per city (optional)
        dist_tasks = [('delhi',), ('mumbai',)]
        my_dist = distribute(dist_tasks)
        for (city_name,) in my_dist:
            LOG.log(f"distance metrics: {city_name}")
            try:
                df_stats, _ = city_distance_metrics(
                    all_path, customer_filepath, city_name)
                if df_stats is not None:
                    tmp_path = os.path.join(
                        outputs_tmp_directory,
                        f"dist_{city_name}__rank{RANK}.parquet")
                    df_stats.write_parquet(tmp_path)
            except Exception as e:
                LOG.log(f"distance error ({city_name}): {e}")

        # ---- Distance stats gather ----
        COMM.barrier()
        if RANK == 0:
            combined = []
            for fn in sorted(os.listdir(outputs_tmp_directory)):
                if fn.startswith('dist_') and fn.endswith('.parquet'):
                    combined.append(pl.read_parquet(
                        os.path.join(outputs_tmp_directory, fn)))
            if combined:
                combined = align_for_concat(combined)
                final_dist = pl.concat(combined, how='vertical')
                outp = os.path.join(
                    outputs_metrics_directory, 'city_distance_stats.csv')
                final_dist.write_csv(outp)
                LOG.log(f"wrote {outp}")

        # ---- Bigfile usage gather ----
        COMM.barrier()
        if RANK == 0:
            combined = []
            for fn in sorted(os.listdir(outputs_tmp_directory)):
                if fn.startswith('big_') and fn.endswith('.parquet'):
                    combined.append(pl.read_parquet(
                        os.path.join(outputs_tmp_directory, fn)))
            if combined:
                combined = align_for_concat(combined)
                final_big = pl.concat(combined, how='vertical')
                final_path = os.path.join(
                    outputs_metrics_directory, 'bigfile_usage_stats.csv')
                final_big.write_csv(final_path)
                LOG.log(f"wrote {final_path}")

        # 3) Monthly double-bar plots per year (do on rank 0 to avoid clashes)
        COMM.barrier()
        if RANK == 0:
            for yr in [2021, 2022, 2023]:
                LOG.log(f"monthly city totals plot for {yr}")
                monthly_city_totals_plot(all_path, year=yr)

        # 4) Coverage metrics per city-year
        COMM.barrier()
        if RANK == 0:
            LOG.log("coverage halfhours per city-year")
            cov = coverage_halfhours_city_year(all_path)
            cov_path = os.path.join(
                outputs_metrics_directory, 'coverage_halfhours_city_year.csv')
            cov.write_csv(cov_path)
            LOG.log(f"wrote {cov_path}")

        LOG.log("=== DONE PROCESSING FILE ===")
    finally:
        LOG.close()


if __name__ == '__main__':
    # Align categorical encodings across lazy scans
    with pl.StringCache():
        main()
