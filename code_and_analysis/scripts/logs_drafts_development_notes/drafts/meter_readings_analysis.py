# ─────────────────────────────────────────────────────────────────────────────
# FILE: meter_readings_analysis.py

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
# - Summary report text files for each dataset analyzed (per-year and full
#   merged file), saved in 'data/outputs/'.
#   These include:
#     - Date range (earliest and latest 'date')
#     - Counts of unique customers and cities
#     - Total row count
#     - List of unique cities
#     - Readings per city
#     - Readings per customer
#     - Customer lifetime statistics
#
# - Plot image files saved in the top-level 'images/' directory:
#     - Daily usage patterns by weekday (per city)
#     - Weekly distribution of customer lifetimes
#     - New customers per week (bar charts)
#     - Histogram of customer lifetimes
#
# - All plots are generated for both individual yearly datasets and the
#   combined dataset 'meter_readings_all_years_YYYYMMDD_formatted.parquet'.


# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────
import os
import sys
import argparse
from typing import Optional, Any, Tuple, List, Dict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import polars as pl
from datetime import datetime

import seaborn as sns


# ────────────────────────────────────────────────────────────────────────────
# Setting up Directories, Defining File Names and Paths
# ────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


# Filesystem helpers
# ─────────────────────────────────────────────────────────────────────────────


def ensure_dir(path: str) -> None:
    """
    Create directory if it doesn't exist.

    Parameters:
    ----------
    path: str
        Path to directory to create

    Returns:
    -------
    None
    """
    os.makedirs(path, exist_ok=True)


class Logger:
    """Logger class to output to both console and file."""

    def __init__(
            self,
            logfile: str
    ) -> None:
        """
        Initialize logger with specified logfile.

        Parameters:
        ----------
        logfile: str
            Path to the log file

        Returns:
        -------
        None
        """
        ensure_dir(os.path.dirname(logfile))
        self._fh = open(logfile, "w", encoding="utf-8")

    def log(
            self,
            msg: str
    ) -> None:
        """
        Log message to both stdout and logfile.

        Parameters:
        ----------
        msg: str
            Message to log

        Returns:
        -------
        None
        """
        sys.stdout.write(msg + "\n")
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        """
        Close the log file.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        self._fh.close()


def save_df(
        df: pl.DataFrame,
        out_dir: str,
        base: str
) -> None:
    """
    Save DataFrame in multiple formats for analysis.

    Parameters:
    ----------
    df: pl.DataFrame
        Polars DataFrame to save
    out_dir: str
        Output directory
    base: str
        Base filename for the output files

    Returns:
    -------
    None
    """
    ensure_dir(out_dir)
    df.head(200).write_csv(os.path.join(out_dir, f"{base}__head.csv"))
    try:
        df.describe().write_csv(os.path.join(out_dir, f"{base}__describe.csv"))
    except Exception:
        pass


def save_plot(path: str) -> None:
    """
    Save current matplotlib plot to file with proper formatting.

    Parameters:
    ----------
    path: str
        Path where to save the plot

    Returns:
    -------
    None
        Saves plot to file
    """
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────


def get_date_range_lazy(file_path: str) -> Tuple[Any, Any] | Tuple[None, None]:
    """
    Get min and max dates from a parquet file using lazy evaluation.

    Parameters:
    ----------
    file_path: str
        Path to the Parquet file with a 'date' column

    Returns:
    -------
    Tuple[str, str] | Tuple[None, None]
        A tuple containing min_date and max_date as strings,
        or (None, None) if error
    """
    try:
        df = (
            pl.scan_parquet(file_path)
            .select([
                pl.col("date").min().alias("min_date"),
                pl.col("date").max().alias("max_date")
            ])
            .collect()
        )
        return df["min_date"][0], df["max_date"][0]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


def summarize_meter_readings_lazy(file_path: str
                                  ) -> Tuple[int, int, int, List[str]]:
    """
    Generate summary statistics for meter readings file.

    Parameters:
    ----------
    file_path: str
        Path to parquet file

    Returns:
    -------
    Tuple[int, int, int, List[str]]
        Tuple containing:
            - count of unique customer IDs
            - count of unique cities
            - total row count
            - list of unique cities
    """
    lf = pl.scan_parquet(file_path)
    summary_df = (
        lf.select([
            pl.col("ca_id").n_unique().alias("n_unique_ca_id"),
            pl.col("city").n_unique().alias("n_unique_city"),
            pl.len().alias("total_rows")
        ]).collect()
    )
    count_unique_ca_ids = summary_df["n_unique_ca_id"][0]
    count_unique_cities = summary_df["n_unique_city"][0]
    count_total_rows = summary_df["total_rows"][0]
    unique_cities_list = (
        lf.select(pl.col("city").unique())
        .collect()
        .get_column("city")
        .to_list()
    )
    return (count_unique_ca_ids,
            count_unique_cities,
            count_total_rows,
            unique_cities_list)


def analyze_readings_per_city_lazy(filepath: str) -> pl.DataFrame:
    """
    Analyze number of readings per city and calculate percentages.

    Parameters:
    ----------
    filepath: str
        Path to parquet file

    Returns:
    -------
    pl.DataFrame
        DataFrame with cities, reading counts, and percentages sorted by
        number of readings
    """
    return (
        pl.scan_parquet(filepath)
        .group_by("city")
        .agg(pl.len().alias("number_of_readings"))
        .with_columns(
            (pl.col("number_of_readings") / pl.col("number_of_readings")
             .sum() * 100)
            .round(2)
            .alias("percentage_of_readings")
        )
        .sort("number_of_readings", descending=True)
        .collect()
    )


def analyze_readings_per_customer_lazy(filepath: str) -> pl.DataFrame:
    """
    Analyze number of readings per customer and calculate percentages.

    Parameters:
    ----------
    filepath: str
        Path to parquet file

    Returns:
    -------
    pl.DataFrame
        DataFrame with customer IDs, reading counts, and percentages
        sorted by number of readings
    """
    return (
        pl.scan_parquet(filepath)
        .group_by("ca_id")
        .agg(pl.len().alias("number_of_readings"))
        .with_columns(
            (pl.col("number_of_readings") / pl.col("number_of_readings")
             .sum() * 100)
            .round(5)
            .alias("percentage_of_readings"))
        .sort("number_of_readings", descending=True)
        .collect()
    )


def analyze_customer_lifetimes_lazy(filepath: str) -> pl.DataFrame:
    """
    Calculate customer lifetimes based on first and last reading dates.

    Parameters:
    ----------
    filepath: str
        Path to parquet file

    Returns:
    -------
    pl.DataFrame
        DataFrame with customer IDs, start dates, end dates,
        and lifetimes in days
    """
    return (
        pl.scan_parquet(filepath)
        .with_columns(pl.col("date").dt.date().alias("date_day"))
        .group_by("ca_id")
        .agg([
            pl.col("date_day").min().alias("start_date"),
            pl.col("date_day").max().alias("end_date")
        ])
        .with_columns(
            (pl.col("end_date") - pl.col("start_date") + pl.duration(days=1))
            .dt.total_days()
            .alias("customer_lifetime_days")
        )
        .sort("customer_lifetime_days", descending=True)
        .collect()
    )


def analyze_and_plot_daily_patterns(
        filepath: str,
        city: str,
        save_path: str
) -> pl.DataFrame:
    """
    Analyze and plot daily consumption patterns by hour and weekday.

    Parameters:
    ----------
    filepath: str
        Path to parquet file
    city: str
        City to analyze
    save_path: str
        Path to save the resulting plot

    Returns:
    -------
    pl.DataFrame
        DataFrame with hourly usage patterns
    """
    df = (
        pl.scan_parquet(filepath)
        .filter(pl.col("city") == city)
        .with_columns([
            pl.col("date").dt.hour().alias("hour"),
            pl.col("date").dt.weekday().alias("weekday"),
            pl.col("date").dt.date().alias("date_day")
        ])
        .group_by(["hour", "weekday", "date_day"])
        .agg(pl.sum("value").alias("daily_hourly_sum"))
        .group_by(["hour", "weekday"])
        .agg(pl.mean("daily_hourly_sum").alias("avg_hourly_usage"))
        .sort(["weekday", "hour"])
        .collect()
    )

    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"]

    fig, axes = plt.subplots(7,
                             1,
                             figsize=(12, 18),
                             sharex=True,
                             sharey=True)

    for weekday in range(1, 8):
        ax = axes[weekday - 1]
        day_data = df.filter(pl.col("weekday") == weekday)
        hours = day_data["hour"].to_list()
        times = [datetime(2023, 1, 1, h) for h in hours]
        values = day_data["avg_hourly_usage"].to_list()
        ax.plot(times, values, marker='o', label=weekdays[weekday-1])
        ax.set_title(f"{weekdays[weekday-1]} Usage Pattern")
        ax.set_ylabel("Average Consumption (kWh)")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True)
        ax.legend()

    plt.xlabel("Hour of Day")
    plt.suptitle(
        (f"Average Hourly Electricity Consumption Patterns "
         f"- {city.capitalize()}"),
        y=1.02)
    save_plot(save_path)
    return df.sort("avg_hourly_usage", descending=True)


def plot_new_customers_per_week(
        df: pl.DataFrame,
        year: int,
        save_path: str
) -> None:
    """
    Plot the number of new customers per week for a given year.

    Parameters:
    ----------
    df: pl.DataFrame
        DataFrame with customer lifetime data
    year: int
        Year to analyze
    save_path: str
        Path to save the resulting plot

    Returns:
    -------
    None
        Plots and saves a graph of new customers per week
    """
    df = df.with_columns(pl.col("start_date").cast(pl.Datetime))
    first_seen = df.sort("start_date").unique(subset=["ca_id"])
    first_seen = first_seen.with_columns([
        pl.col("start_date").dt.strftime("%G").cast(pl.Int32).
        alias("iso_year"),
        pl.col("start_date").dt.strftime("%V").cast(pl.Int32)
        .alias("iso_week")
    ])
    first_seen = first_seen.filter(pl.col("iso_year") == year)
    weekly_new = first_seen.group_by("iso_week").agg(
        pl.count("ca_id").alias("new_customers")
    ).sort("iso_week")

    plt.figure(figsize=(12, 6))
    plt.bar(weekly_new["iso_week"], weekly_new["new_customers"])
    plt.title(f"New Customers per Week in {year}")
    plt.xlabel("ISO Week")
    plt.ylabel("Number of New Customers")
    plt.grid(True)
    save_plot(save_path)


def plot_customer_lifetime_histogram(
        df: pl.DataFrame,
        save_path: str,
        bins: int = 30
) -> None:
    """
    Plot histogram of customer lifetimes.

    Parameters:
    ----------
    df: pl.DataFrame
        DataFrame with customer lifetime data
    save_path: str
        Path to save the resulting plot
    bins: int, optional
        Number of bins for the histogram (default: 30)

    Returns:
    -------
    None
        Plots and saves a histogram of customer lifetimes
    """
    lifetimes = df["customer_lifetime_days"].to_numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(lifetimes, bins=bins, edgecolor="black", alpha=0.75)
    plt.title("Distribution of Customer Lifetimes")
    plt.xlabel("Customer Lifetime (days)")
    plt.ylabel("Number of Customers")
    plt.grid(True, linestyle="--", alpha=0.5)
    save_plot(save_path)


def reshape_and_sort_usage(df: pl.DataFrame) -> pl.DataFrame:
    """
    Reshape and sort daily usage patterns for easy analysis.

    Parameters:
    ----------
    df: pl.DataFrame
        DataFrame with usage patterns by hour and weekday

    Returns:
    -------
    pl.DataFrame
        Pivoted and sorted DataFrame with hours as rows and weekdays as columns
    """
    pivoted = df.pivot(values="avg_hourly_usage",
                       index="hour",
                       on="weekday",
                       aggregate_function="mean")
    weekday_order = [1, 2, 3, 4, 5, 6, 7]
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    renames = {str(i): name for i, name in zip(weekday_order, weekday_names)
               if str(i) in pivoted.columns}
    pivoted = pivoted.rename(renames)
    ordered_cols = ["hour"] + [renames[str(i)]
                               for i in weekday_order if str(i) in renames]
    pivoted = pivoted.select(ordered_cols)
    if "Mon" in pivoted.columns:
        pivoted = pivoted.sort("Mon", descending=True)
    return pivoted


# ─────────────────────────────────────────────────────────────────────────────
# Args (kept, but no main() wrapper)
# ─────────────────────────────────────────────────────────────────────────────

ap = argparse.ArgumentParser(
    description="Run meter readings analysis and save reports/plots.")
ap.add_argument("--base-dir", default=os.path.join("..", ".."))
ap.add_argument("--meter-dir", default=os.path.join("data",
                                                    "hitachi_copy",
                                                    "meter_primary_files"))
ap.add_argument("--out-dir", default=os.path.join("data", "outputs"))
ap.add_argument("--images-dir", default=os.path.join("images"))

# Year files (raw + formatted)
ap.add_argument("--files-2021", required=True)
ap.add_argument("--files-2022", required=True)
ap.add_argument("--files-2023", required=True)
ap.add_argument("--files-2021-formatted", required=True)
ap.add_argument("--files-2022-formatted", required=True)
ap.add_argument("--files-2023-formatted", required=True)

# City splits
ap.add_argument("--delhi-2021", required=True)
ap.add_argument("--delhi-2022", required=True)
ap.add_argument("--mumbai-2022", required=True)
ap.add_argument("--mumbai-2023", required=True)

# All-years formatted (full dataset)
ap.add_argument("--all-years-formatted", required=True)

args = ap.parse_args()

base_dir = os.path.abspath(args.base_dir)
meter_dir = os.path.join(base_dir, args.meter_dir)
out_dir = os.path.join(base_dir, args.out_dir)
images_dir = os.path.join(base_dir, args.images_dir)

ensure_dir(out_dir)
ensure_dir(images_dir)

log = Logger(os.path.join(out_dir, "meter_readings_analysis.log"))
L = log.log

# Resolve paths
p2021 = os.path.join(meter_dir, args.files_2021)
p2022 = os.path.join(meter_dir, args.files_2022)
p2023 = os.path.join(meter_dir, args.files_2023)
p2021f = os.path.join(meter_dir, args.files_2021_formatted)
p2022f = os.path.join(meter_dir, args.files_2022_formatted)
p2023f = os.path.join(meter_dir, args.files_2023_formatted)
pdelhi21 = os.path.join(meter_dir, args.delhi_2021)
pdelhi22 = os.path.join(meter_dir, args.delhi_2022)
pmumbai22 = os.path.join(meter_dir, args.mumbai_2022)
pmumbai23 = os.path.join(meter_dir, args.mumbai_2023)
p_all = os.path.join(meter_dir, args.all_years_formatted)

# === Data types / schema (2021 sample) ===
try:
    pldf2021 = pl.read_parquet(p2021)
    L("\nData types in 'meter_readings_2021':")
    L(str(pldf2021.dtypes))
    L(str(pldf2021.schema))
    save_df(pldf2021.head(5000),
            out_dir,
            "sample__meter_readings_2021")
except Exception as e:
    L(f"ERROR reading 2021 parquet: {e}")

# === Date ranges for key files ===
L("\n" + "-" * 120)
L("Date Range Analysis for Meter Readings Files:\n" + "-" * 120)
for path in [p2021f, p2022f, p2023f, pdelhi21,
             pdelhi22, pmumbai22, pmumbai23, p_all]:
    name = os.path.basename(path)
    min_date, max_date = get_date_range_lazy(path)
    if min_date and max_date:
        L(f"{name}: {min_date} to {max_date}")
    else:
        L(f"{name}: ERROR determining date range")

# === Yearly summaries ===
for year, p in [("2021", p2021), ("2022", p2022), ("2023", p2023)]:
    L("\n" + "-" * 120)
    L(f"Summary Analysis of {year} Data :\n" + "-" * 120)
    try:
        (unique_ids,
         unique_cities,
         total_rows,
         cities_list) = summarize_meter_readings_lazy(p)

        L(f"Unique Customer Ids: {unique_ids:,}")
        L(f"Unique Cities: {unique_cities:,}")
        L(f"Total Rows: {total_rows:,}")
        L(f"Unique Cities List: {cities_list}")
    except Exception as e:
        L(f"ERROR summarizing {year}: {e}")

# === Readings per city (by year) ===
for year, p in [("2021", p2021), ("2022", p2022), ("2023", p2023)]:
    L("\n" + "-" * 120)
    L(f"{year} Data : Readings Per City \n" + "-" * 120)
    try:
        df = analyze_readings_per_city_lazy(p)
        save_df(df, out_dir, f"readings_per_city_{year}")
    except Exception as e:
        L(f"ERROR analyze_readings_per_city {year}: {e}")

# === Readings per customer (all years — heavy, but we're on HPC) ===
for year, p in [("2021", p2021), ("2022", p2022), ("2023", p2023)]:
    L("\n" + "-" * 120)
    L(f"{year} Data : Readings Per Customer \n" + "-" * 120)
    try:
        df = analyze_readings_per_customer_lazy(p)
        save_df(df, out_dir, f"readings_per_customer_{year}")
    except Exception as e:
        L(f"ERROR analyze_readings_per_customer {year}: {e}")

# === Customer lifetimes (yearly; 2021/2023 use formatted in your notebook) ===
for year, p in [("2021", p2021f), ("2022", p2022), ("2023", p2023f)]:
    L("\n" + "-" * 120)
    L(f"{year} Data : Customer Lifetimes \n" + "-" * 120)
    try:
        df = analyze_customer_lifetimes_lazy(p)
        save_df(df, out_dir, f"customer_lifetimes_{year}")
        # Plots analogous to notebook where used
        if year == "2023":
            plot_customer_lifetime_histogram(
                df,
                save_path=os.path.join(images_dir,
                                       "customer_lifetimes_2023__hist.png")
            )
            try:
                plot_new_customers_per_week(
                    df, year=2023,
                    save_path=os.path.join(images_dir,
                                           "customer_new_per_week_2023.png")
                )
            except Exception as e:
                L(f"WARNING plotting new customers per week (2023): {e}")
    except Exception as e:
        L(f"ERROR analyze_customer_lifetimes {year}: {e}")

# === City-specific (Delhi) ===
for label, p in [("2021 Delhi", pdelhi21), ("2022 Delhi", pdelhi22)]:
    L("\n" + "-" * 120)
    L(f"Summary Analysis of {label} Data :\n" + "-" * 120)
    try:
        (unique_ids,
         unique_cities,
         total_rows,
         cities_list) = summarize_meter_readings_lazy(p)
        L(f"Unique Customer Ids: {unique_ids:,}")
        L(f"Unique Cities: {unique_cities:,}")
        L(f"Total Rows: {total_rows:,}")
        L(f"Unique Cities List: {cities_list}")
    except Exception as e:
        L(f"ERROR summarizing {label}: {e}")

for label, p in [("2021 Delhi", pdelhi21), ("2022 Delhi", pdelhi22)]:
    L("\n" + "-" * 120)
    L(f"{label} Data : Readings Per City \n" + "-" * 120)
    try:
        df = analyze_readings_per_city_lazy(p)
        save_df(df,
                out_dir,
                f"readings_per_city__{label.replace(' ', '_').lower()}")
    except Exception as e:
        L(f"ERROR analyze_readings_per_city {label}: {e}")

for label, p in [("2021 Delhi", pdelhi21), ("2022 Delhi", pdelhi22)]:
    L("\n" + "-" * 120)
    L(f"{label} Data : Readings Per Customer \n" + "-" * 120)
    try:
        df = analyze_readings_per_customer_lazy(p)
        save_df(df,
                out_dir,
                f"readings_per_customer__{label.replace(' ', '_').lower()}")
    except Exception as e:
        L(f"ERROR analyze_readings_per_customer {label}: {e}")

for label, p in [("2021 Delhi", pdelhi21), ("2022 Delhi", pdelhi22)]:
    L("\n" + "-" * 120)
    L(f"{label} Data : Customer Lifetimes \n" + "-" * 120)
    try:
        df = analyze_customer_lifetimes_lazy(p)
        save_df(df,
                out_dir,
                f"customer_lifetimes__{label.replace(' ', '_').lower()}")
    except Exception as e:
        L(f"ERROR analyze_customer_lifetimes {label}: {e}")

# === City-specific (Mumbai) ===
for label, p in [("2022 Mumbai", pmumbai22), ("2023 Mumbai", pmumbai23)]:
    L("\n" + "-" * 120)
    L(f"Summary Analysis of {label} Data :\n" + "-" * 120)
    try:
        (unique_ids,
         unique_cities,
         total_rows,
         cities_list) = summarize_meter_readings_lazy(p)
        L(f"Unique Customer Ids: {unique_ids:,}")
        L(f"Unique Cities: {unique_cities:,}")
        L(f"Total Rows: {total_rows:,}")
        L(f"Unique Cities List: {cities_list}")
    except Exception as e:
        L(f"ERROR summarizing {label}: {e}")

for label, p in [("2022 Mumbai", pmumbai22), ("2023 Mumbai", pmumbai23)]:
    L("\n" + "-" * 120)
    L(f"{label} Data : Readings Per City \n" + "-" * 120)
    try:
        df = analyze_readings_per_city_lazy(p)
        save_df(df,
                out_dir,
                f"readings_per_city__{label.replace(' ', '_').lower()}")
    except Exception as e:
        L(f"ERROR analyze_readings_per_city {label}: {e}")

for label, p in [("2022 Mumbai", pmumbai22), ("2023 Mumbai", pmumbai23)]:
    L("\n" + "-" * 120)
    L(f"{label} Data : Readings Per Customer \n" + "-" * 120)
    try:
        df = analyze_readings_per_customer_lazy(p)
        save_df(df,
                out_dir,
                f"readings_per_customer__{label.replace(' ', '_').lower()}")
    except Exception as e:
        L(f"ERROR analyze_readings_per_customer {label}: {e}")

for label, p in [("2022 Mumbai", pmumbai22), ("2023 Mumbai", pmumbai23)]:
    L("\n" + "-" * 120)
    L(f"{label} Data : Customer Lifetimes \n" + "-" * 120)
    try:
        df = analyze_customer_lifetimes_lazy(p)
        save_df(df,
                out_dir,
                f"customer_lifetimes__{label.replace(' ', '_').lower()}")

        if label == "2023 Mumbai":
            plot_customer_lifetime_histogram(
                df,
                save_path=os.path.join(
                    images_dir,
                    "mumbai_2023__customer_lifetimes_hist.png")
            )
            try:
                plot_new_customers_per_week(
                    df,
                    year=2023,
                    save_path=os.path.join(
                        images_dir,
                        "mumbai_2023__new_customers_per_week.png")
                )
            except Exception as e:
                L((f"WARNING plotting new customers per week "
                   f"(Mumbai 2023): {e}"))
    except Exception as e:
        L(f"ERROR analyze_customer_lifetimes {label}: {e}")

# === Mumbai daily patterns + peaks ===
try:
    mumbai_2022_hourly = analyze_and_plot_daily_patterns(
        filepath=pmumbai22,
        city="mumbai",
        save_path=os.path.join(images_dir, "mumbai_2022__daily_patterns.png"),
    )
    mumbai_2023_hourly = analyze_and_plot_daily_patterns(
        filepath=pmumbai23,
        city="mumbai",
        save_path=os.path.join(images_dir, "mumbai_2023__daily_patterns.png"),
    )
    mumbai_2022_hourly_reshaped = reshape_and_sort_usage(mumbai_2022_hourly)
    mumbai_2023_hourly_reshaped = reshape_and_sort_usage(mumbai_2023_hourly)

    weekdays_short = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for day in weekdays_short:
        out22 = mumbai_2022_hourly_reshaped.select(["hour",
                                                    day]
                                                   ).sort(day,
                                                          descending=True
                                                          ).head(6)
        out22.write_csv(os.path.join(out_dir,
                                     f"mumbai_2022__peaks__{day}.csv"))

        out23 = mumbai_2023_hourly_reshaped.select(["hour",
                                                    day]
                                                   ).sort(day,
                                                          descending=True
                                                          ).head(6)
        out23.write_csv(os.path.join(out_dir,
                                     f"mumbai_2023__peaks__{day}.csv"))
except Exception as e:
    L(f"ERROR daily patterns / peaks: {e}")

# === Monday-sorted heatmap for Mumbai 2023 ===
try:
    meter_readings_mumbai_2023_pldf = pl.read_parquet(pmumbai23)
    hourly_peaks = (
        meter_readings_mumbai_2023_pldf.lazy()
        .with_columns([
            pl.col("date").dt.strftime("%A").alias("day_of_week"),
            pl.col("date").dt.hour().alias("hour")
        ])
        .group_by(["city", "day_of_week", "hour"])
        .agg(pl.col("value").mean())
        .sort(["day_of_week", "hour"])
        .collect()
    )
    hourly_median = (
        hourly_peaks.lazy()
        .group_by(["day_of_week", "hour"])
        .agg(pl.col("value").median())
        .collect()
        .pivot(values="value", index="hour", columns="day_of_week")
        .sort("hour")
    )
    monday_ranking = (
        hourly_median.select(["hour", "Monday"])
        .sort("Monday", descending=True)
        .with_columns(rank=pl.col("hour").rank(descending=True))
    )
    sorted_hourly = (
        hourly_median
        .join(monday_ranking.select(["hour", "rank"]), on="hour")
        .sort("rank")
        .drop("rank")
    )

    plt.figure(figsize=(12, 8))
    days = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
    hours = sorted_hourly["hour"].cast(pl.Int32).to_list()
    im = plt.imshow(
        sorted_hourly.select(days).to_numpy(),
        cmap="YlOrRd",
        aspect="auto",
        extent=[0, 7, len(hours)-0.5, -0.5]
    )
    plt.title("Hourly Electricity Demand (Sorted by Monday's Peak Pattern)")
    plt.ylabel("Hour of Day")
    plt.xlabel("Day of Week")
    plt.xticks(np.arange(7), labels=days)
    plt.yticks(np.arange(len(hours)), labels=[f"{h:02d}:00" for h in hours])
    cbar = plt.colorbar(im, pad=0.01)
    cbar.set_label("Median Usage (kW)")
    save_plot(os.path.join(images_dir,
                           "mumbai_2023__hourly_heatmap_monday_sorted.png"))

    sorted_hourly.write_csv(
        os.path.join(out_dir,
                     "mumbai_2023__hourly_heatmap_table.csv"))
except Exception as e:
    L(f"ERROR hourly heatmap: {e}")

# === Full dataset (all years) — ALWAYS run heavy here ===
L("\n" + "-" * 120)
L("All-Years (Full Dataset) : Summary & Reports\n" + "-" * 120)

try:
    # Date range
    min_date, max_date = get_date_range_lazy(p_all)
    if min_date and max_date:
        L(f"All-years date range: {min_date} to {max_date}")
    else:
        L("All-years: ERROR determining date range")

    # Summary
    (unique_ids,
     unique_cities,
     total_rows,
     cities_list) = summarize_meter_readings_lazy(p_all)
    L(f"Unique Customer Ids: {unique_ids:,}")
    L(f"Unique Cities: {unique_cities:,}")
    L(f"Total Rows: {total_rows:,}")
    L(f"Unique Cities List: {cities_list}")

    # Readings per city
    df_city = analyze_readings_per_city_lazy(p_all)
    save_df(df_city, out_dir, "all_years__readings_per_city")

    # Heavy bits: per-customer + lifetimes
    df_cust = analyze_readings_per_customer_lazy(p_all)
    save_df(df_cust, out_dir, "all_years__readings_per_customer")

    df_life = analyze_customer_lifetimes_lazy(p_all)
    save_df(df_life, out_dir, "all_years__customer_lifetimes")

    plot_customer_lifetime_histogram(
        df_life,
        save_path=os.path.join(images_dir,
                               "all_years__customer_lifetimes_hist.png")
    )

    # Weekly new customers plot for each ISO year present
    years_present = (
        df_life.select(pl.col("start_date").dt.strftime("%G").cast(pl.Int32)
                       .alias("iso_year"))
        .unique()
        .to_series()
        .to_list()
    )
    for y in sorted(y for y in years_present if y is not None):
        try:
            plot_new_customers_per_week(
                df_life,
                year=int(y),
                save_path=os.path.join(images_dir,
                                       (f"all_years__new_customers_"
                                        f"per_week_{y}.png"))
            )
        except Exception as e:
            L(f"WARNING plotting new customers per week ({y}): {e}")

except Exception as e:
    L(f"ERROR all-years summary: {e}")

L("\n=== Completed meter_readings_analysis.py ===")
log.close()
