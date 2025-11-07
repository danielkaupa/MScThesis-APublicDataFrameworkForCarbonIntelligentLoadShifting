# ─────────────────────────────────────────────────────────────────────────────
# FILE: step5_optimisation_module.py
# PURPOSE:
# - Implements and runs multiple optimisation strategies (Greedy heuristic,
#   LP relaxation, and continuous nonlinear optimisation) for shifting
#   electricity usage across time slots to reduce emissions.
# - Provides a unified pipeline for per-customer, per-day optimisation
#   with logging of solver choices, constraints, and outcomes.
# - Supports consolidation of shard results into a single Parquet log.
#
# USAGE:
# - This module is imported by the main runner script
#   (e.g., step5_run_optimisation.py) when executed on the HPC cluster.
# - Designed to be run in parallel via MPI, where each rank processes
#   a subset of customer-day shards and writes its own results file.
# - Consolidation is automatically triggered at the end of execution.
#
# RUN REQUIREMENTS:
# - Python 3.9+ environment with the following libraries installed:
#   numpy, pandas, polars, scipy, cvxpy, mpi4py.
# - Access to shard Parquet files containing customer-day usage data.
# - Optional: specific LP solvers installed (HIGHS, GLPK, ECOS, SCS,
#   CLARABEL, OSQP). At least one open-source solver must be available
#   for LP runs to succeed.
# - For continuous optimisation, SciPy’s 'SLSQP' (default) or
#   'trust-constr' method must be available.
# - For Greedy optimisation, no external solver is required.
#
# OUTPUTS:
# - Per-rank Parquet files containing optimisation results for the
#   assigned shards. Filenames include the rank ID.
# - A consolidated Parquet file combining all rank results, saved
#   in the specified output directory.
# - Logged metadata includes:
#   * high-level solver family used (greedy / continuous / lp),
#   * low-level solver/method (e.g., HIGHS, SLSQP),
#   * constraints applied and solution metrics.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
import warnings
from typing import (
    Any, Dict, Generator, List, Optional, Tuple, Literal, Set
)
import numpy as np
import pandas as pd
import polars as pl
# LP / Continuous solvers
try:
    import cvxpy as cp
except Exception:
    cp = None  # handled at call site

try:
    from scipy.optimize import minimize, Bounds, LinearConstraint

except Exception:
    minimize = None  # handled at call site

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# CLASSES
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class PeakHoursReductionLimitConfig:
    """
    Configuration for limiting peak hours reduction.

    This class allows you to define the scope and percentage limit for
    reducing peak hours in a specific region or for a specific customer.
    The purpose for this configuration is to allow for preserving comfort
    during peak hours by not reducing the consumption too much.

    Attributes:
    -----------
    peak_hours_reduction_scope: Literal["per_customer", "per_city"]
        The scope of the reduction limit (per customer or per city).
    peak_hours_reduction_percent_limit: float
        The percentage limit for    peak hours reduction.
    peak_hours_dict: Optional[Dict[str, Dict[str, List[str]]]]
        A dictionary defining peak hours for reduction by city, day, and time.
    limit_scope: Literal["slot", "hour"]
        The scope of the limit (per slot or per hour).
    """

    peak_hours_reduction_scope: Literal["per_customer",
                                        "per_city"] = "per_city"
    peak_hours_reduction_percent_limit: float = 30.0
    # Dict structure: { delhi: { "Mon": [ 9,10,20, ...], "Tue": [9,10,11, ] } }
    peak_hours_dict: Optional[Dict[str, Dict[str, List[int]]]] = None
    limit_scope: Literal["slot", "hour"] = "hour"


@dataclass
class CustomerAdoptionBehavioralConfig:
    """
    Configuration for customer adoption behavior in the energy management
    system.

    This class allows you to define the behavioral parameters that influence
    how customers interact with the energy management system, including their
    usage patterns and preferences.

    Attributes:
    -----------
    customer_power_moves_per_day : int
        The number of shifts a customer is allowed to make per day
    customer_power_moves_per_week : int
        The number of shifts a customer is allowed to make per week.
    timezone: str
        The timezone in which the customer operates.
    day_boundaries: str
        The hours in the day which can be evaluated for shifts
        (e.g., "08:00-20:00")
    week_boundaries: str
        The boundaries of the week for the customer's schedule.
    shift_hours_window: float
        The number of hours on either side of usage to look for shifts.
    slot_length_minutes: int
        The length of each time slot in minutes.
    shift_window_inclusive: bool
        Whether the shift window is inclusive or exclusive.
    peak_hours_reduction_limit_config: Optional[PeakHoursReductionLimitConfig]
        Configuration for peak hours reduction limits.
    """

    customer_power_moves_per_day: int = 1
    customer_power_moves_per_week: int = 3
    timezone: str = "Asia/Kolkata"
    day_boundaries: str = "00:00-24:00"
    week_boundaries: Literal["Mon-Sun", "ISO"] = "Mon-Sun"
    shift_hours_window: float = 2.0
    slot_length_minutes: int = 30
    shift_window_inclusive: bool = True
    peak_hours_reduction_limit_config: Optional[
                                        PeakHoursReductionLimitConfig] = None


@dataclass
class HouseholdMinimumConsumptionLimitConfig:
    """
    Configuration for minimum energy consumption limits at the household level.

    This class allows you to define the minimum energy consumption limits
    for households, ensuring that essential energy needs are met even during
    demand response events.

    The attributes of this class will be used to configure the minimum
    consumption limits. remaining_usage ≥ max(min_baseline(customer,t),
    R% * robust_max(customer,t))

    Attributes:
    -----------
    household_minimum_baseline_period: Literal["year","month","week","day"]
        The period over which to calculate the baseline consumption.
    household_minimum_baseline_type: Literal["average","absolute_min"]
        The type of baseline calculation to use (average or absolute minimum).
    household_minimum_robust_max_percentile: float
        The percentile to use for robust maximum calculation.
    household_minimum_R_percent: float
        The percentage limit for minimum consumption
        (as a fraction of robust max).
    household_minimum_epsilon_floor_kWh: float
        A small value to avoid zero consumption limits.
    """
    household_minimum_baseline_period: Literal["year",
                                               "month",
                                               "week",
                                               "day"] = "year"
    household_minimum_baseline_type: Literal["average",
                                             "absolute_min"] = "average"
    household_minimum_robust_max_percentile: float = 95.0
    # fraction of robust max - defaults to 10
    household_minimum_R_percent: float = 10.0
    # internal: epsilon floor to avoid zeros/outages
    # small value (1 Wh) to avoid zero consumption limits
    household_minimum_epsilon_floor_kWh: float = 0.001


@dataclass
class RegionalLoadShiftingLimitConfig:
    """
    Configuration for regional load shifting capabilities.

    The purpose of this configuration is to define the upper limit for load
    shifting at a regional level, as large fluctuations in the energy demand
    can impact the overall stability of the grid.

    Attributes:
    -----------
    regional_load_shift_percent_limit: float
        The percentage limit for load shifting (per city).
    regional_total_daily_average_load_kWh: float
        The total daily average load in kWh for each city.
    """
    # per city
    regional_load_shift_percent_limit: float = 10.0
    # {city: kWh/day}
    regional_total_daily_average_load_kWh: Optional[Dict[str, float]] = None


@dataclass
class ShiftWithoutSpikeLimitConfig:
    """
    Configuration to limit the amount of energy that can be moved into
    a single time slot in order to avoid causing a spike in usage.

    Attributes:
    -----------
    city: The city for which the configuration applies.
    alpha_peak_cap_percent: The per-slot upper cap vs baseline (city level).
    """
    # The city for which the configuration applies
    city: str = "default_city"
    # per-slot upper cap vs baseline (city level)
    alpha_peak_cap_percent: float = 25.0


@dataclass
class ShiftPolicy:
    """
    Policy configuration for load shifting in the energy management system.

    This class allows you to consolidate the configurations available through
    the classes defined above in order to apply constraints and limitations to
    load shifting strategies, including behavioral, regional, household, and
    spike caps.

    Attributes:
    -----------
    behavioral: CustomerAdoptionBehavioralConfig
        Behavioral configuration for customer adoption.
    regional_cap: RegionalLoadShiftingLimitConfig
        Regional load shifting limit configuration.
    household_min: HouseholdMinimumConsumptionLimitConfig
        Household minimum consumption limit configuration.
    spike_cap: ShiftWithoutSpikeLimitConfig
        Shift without spike limit configuration.
    """
    behavioral: CustomerAdoptionBehavioralConfig = (
        field(default_factory=CustomerAdoptionBehavioralConfig)
    )
    regional_cap: Optional[RegionalLoadShiftingLimitConfig] = None
    household_min: Optional[HouseholdMinimumConsumptionLimitConfig] = None
    spike_cap: Optional[ShiftWithoutSpikeLimitConfig] = None


_SUPPORTED_LP_SOLVERS: Tuple[str, ...] = (
    "HIGHS",     # fastest & very robust for LPs
    "GLPK",      # simplex/LP – good if installed
    "SCS",       # first-order conic; good fallback
    "CLARABEL",  # modern first-order conic
    "ECOS",      # interior-point conic
    "OSQP",      # QP/LP; fine for many LPs via QP form
)


@dataclass
class SolverConfig:
    """
    Configuration for the optimization solver in the energy management system.

    This class allows you to define the parameters for the solver used in
    the optimization process, including the solver family and specific options
    for each solver type.

    Attributes:
    -----------
    solver_family : Literal["continuous", "greedy", "lp"]
        The family of the solver to use (one of {"lp","greedy","continuous"})
        - "lp": linear program on flow variables y_{t->s} within a window.
        - "greedy": heuristic swaps with caps.
        - "continuous": relaxed reallocation x (no window/flow structure).
    lp_solver : Optional[str]
        The specific LP solver to use (if family is "lp").
    lp_solver_opts : Optional[Dict[str, Any]]
        Options for the LP solver.
    greedy_min_fraction_of_day_to_move : Optional[float]
        Minimum fraction of the day to move for greedy strategies.
    """
    solver_family: Literal["continuous", "greedy", "lp"] = "greedy"
    # Continuous methods for the "continuous" solver family
    continuous_method: Optional[Literal["SLSQP",
                                        "trust-constr",
                                        "closed-form"]] = "SLSQP"
    continuous_opts: Optional[Dict[str, Any]] = None
    # LP: choose cvxpy solver name if desired
    lp_solver: Optional[Literal["HIGHS", "GLPK",
                                "ECOS", "SCS",
                                "CLARABEL", "OSQP"]] = "HIGHS"
    lp_solver_opts: Optional[Dict[str, Any]] = None
    # Heuristic knobs
    # if set, skip tiny moves
    greedy_min_fraction_of_day_to_move: Optional[float] = None


# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# FUNCTIONS
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────

# Input / Output Helpers
# ────────────────────────────────────────────────────────────────────────────


def write_parquet_fsync(
        df: pd.DataFrame,
        path: str
) -> None:
    """
    Atomically write a DataFrame to Parquet, forcing data to disk.

    This writes to `path + ".tmp"`, issues an `fsync`, then atomically
    renames the temporary file to `path`. If a crash occurs mid-write,
    the original file remains intact.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to persist.
    path : str
        Destination Parquet filepath.

    Returns
    -------
    None
        Writes the DataFrame to a Parquet file at the specified path.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        df.to_parquet(f, index=False)
        f.flush()
        os.fsync(f.fileno())  # force to disk
    os.replace(tmp, path)     # atomic rename


# Parallelisation / Threading Utilities
# ────────────────────────────────────────────────────────────────────────────

def _init_worker_singlethread():
    """
    Initialize a worker process to be single-threaded for math/BLAS libraries.

    Sets environment variables for OpenMP/BLAS to 1 and, if available,
    uses `threadpoolctl.threadpool_limits(1)` to clamp threadpools.

    Returns
    -------
    None
    """
    # Keep each worker process single-threaded for math libs
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    try:
        # This caps numpy/scipy threadpools too (if available)
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _iter_results_local(
        job_args,
        workers: int
) -> Generator[Any, None, None]:
    """
    Iterate results using a local multiprocessing Pool.

    Parameters
    ----------
    job_args : Iterable
        Iterable of argument tuples for `_solve_cityweek_worker`.
    workers : int
        Number of processes to spawn.

    Yields
    ------
    Any
        Results returned by `_solve_cityweek_worker`.

    Notes
    -----
    Uses a fork context if available, else spawn. Each worker is initialized
    via `_init_worker_singlethread()` to avoid oversubscription.
    """
    import multiprocessing as mp
    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers,
                  initializer=_init_worker_singlethread) as pool:
        for res in pool.imap(_solve_cityweek_worker,
                             job_args, chunksize=1):
            yield res


# Time and Slot Utilities
# ────────────────────────────────────────────────────────────────────────────


def add_week_start_col(
        df_pd: pd.DataFrame,
        week_boundaries: str = "Mon-Sun"
) -> pd.DataFrame:
    """
    Add a 'week_start' midnight timestamp consistent with the chosen week
    convention.

    Currently supports Monday-start weeks for both "Mon-Sun" and "ISO".

    Parameters
    ----------
    df_pd : pandas.DataFrame
        Must contain a 'day' column (datetime-like or parseable).
        'day' is treated as the calendar date; time-of-day is ignored.
    week_boundaries : {"Mon-Sun","ISO"}, optional
        Week convention. Both map to Monday-start.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with a new 'week_start' (Timestamp at 00:00).
    """
    out = df_pd.copy()
    day = pd.to_datetime(out["day"])
    if week_boundaries == "ISO":
        week_start = (day - pd.to_timedelta(day.dt.dayofweek, unit="D")
                      ).dt.normalize()
    else:
        # Mon-Sun default
        week_start = (day - pd.to_timedelta(day.dt.dayofweek, unit="D")
                      ).dt.normalize()
    out["week_start"] = week_start
    return out


@lru_cache(maxsize=256)
def cached_pairs(
        T: int,
        W_slots: int
) -> Tuple[Tuple[Tuple[int, int], ...],
           Tuple[np.ndarray, ...],
           Tuple[np.ndarray, ...]]:
    """
    Generate allowed (t, s) pairs for a given time horizon T and window
    W_slots.

    Parameters
    ----------
    T : int
        Time horizon (number of slots).
    W_slots : int
        Maximum allowed distance between t and s.

    Returns
    -------
    pairs : Tuple[(int,int), ...]
        Allowed (t, s) pairs with |t - s| ≤ W_slots.
    by_src : Tuple[np.ndarray, ...]
        by_src[t] -> indices i in 'pairs' where pairs[i][0] == t
    by_dst : Tuple[np.ndarray, ...]
        by_dst[s] -> indices i in 'pairs' where pairs[i][1] == s
    """
    pairs: List[Tuple[int, int]] = []
    for t in range(T):
        s0, s1 = max(0, t - W_slots), min(T, t + W_slots + 1)
        for s in range(s0, s1):
            pairs.append((t, s))
    pairs = tuple(pairs)
    by_src = [[] for _ in range(T)]
    by_dst = [[] for _ in range(T)]
    for i, (t, s) in enumerate(pairs):
        by_src[t].append(i)
        by_dst[s].append(i)
    by_src = tuple(np.asarray(ix, dtype=int) for ix in by_src)
    by_dst = tuple(np.asarray(ix, dtype=int) for ix in by_dst)
    return pairs, by_src, by_dst


def day_and_slot_cols(
        df: pl.DataFrame,
        slot_len_min: int = 30
) -> pl.DataFrame:
    """
    Adds "day" (midnight timestamp) and half-hour "slot" [0..47] columns to
    the DataFrame.

    Parameters:
    ----------
    df : pl.DataFrame
        Input DataFrame containing a "date" column.
    slot_len_min: int
        Length of the time slots in minutes (default is 30).

    Returns:
    --------
    pl.DataFrame
        DataFrame with added "day" and "slot" columns.
    """
    # Adds "day" (midnight ts) and half-hour "slot" [0..47] columns
    half_hour = slot_len_min == 30
    out = (
        df
        .with_columns([
            pl.col("date").dt.truncate("1d").alias("day"),
            (pl.col("date").dt.hour() * (60//slot_len_min) +
             (pl.col("date").dt.minute() // slot_len_min)).alias("slot")
        ])
    )
    if half_hour:
        # ensure slot in [0..47]
        out = out.with_columns(pl.col("slot").cast(pl.Int32))
    return out


def hours_to_slots(
        hours: float,
        slot_len_min: int = 30
) -> int:
    """
    Convert hours to the number of slots based on the slot length in minutes.

    Parameters:
    ----------
    hours : float
        The number of hours to convert.
    slot_len_min : int
        The length of each slot in minutes (default is 30).

    Returns:
    -------
    int
        The number of slots corresponding to the given hours.
    """
    return int((hours * 60) // slot_len_min)


# Baselining and Limits
# ────────────────────────────────────────────────────────────────────────────


def attach_household_floor(
        df: pl.DataFrame,
        cfg: HouseholdMinimumConsumptionLimitConfig,
) -> pl.DataFrame:
    """
    Compute and join a per-row household minimum floor 'floor_kwh'.

    The floor is:
        max( baseline(hod,dow), R% * robust_max_percentile(hod,dow), epsilon )

    where stratification is (hour-of-day, day-of-week).

    Parameters
    ----------
    df : polars.DataFrame
        Must contain 'ca_id', 'date' (Datetime), and 'value' (kWh).
    cfg : HouseholdMinimumConsumptionLimitConfig
        Baseline period/type, robust percentile, R%, and epsilon floor.

    Returns
    -------
    polars.DataFrame
        Original df with a 'floor_kwh' column joined.

    Notes
    -----
    Baseline is mean or min by (ca_id, hod, dow).
    Robust max is the p-th quantile of 'value' by (ca_id, hod, dow).
    """
    # Filter by period if you want (here we keep full df; you can parametrize)
    base = (
        df
        .with_columns([
            pl.col("date").dt.hour().alias("hod"),
            pl.col("date").dt.strftime("%A").alias("dow"),
        ])
        .group_by(["ca_id", "hod", "dow"])
    )

    # Baseline
    if cfg.household_minimum_baseline_type == "average":
        baseline = base.agg(pl.mean("value").alias("baseline_kwh"))
    else:
        baseline = base.agg(pl.min("value").alias("baseline_kwh"))

    # Robust max (percentile on 'value')
    p = cfg.household_minimum_robust_max_percentile
    robust = (
        df
        .with_columns([
            pl.col("date").dt.hour().alias("hod"),
            pl.col("date").dt.strftime("%A").alias("dow"),
        ])
        .group_by(["ca_id", "hod", "dow"])
        .agg(
            pl.col("value").quantile(
                quantile=p/100.0,
                interpolation="nearest"
            ).alias("robust_max_kwh")
        )
    )

    stats = baseline.join(robust, on=["ca_id", "hod", "dow"], how="full")
    stats = stats.with_columns([
        pl.col("baseline_kwh").fill_null(0.0),
        pl.col("robust_max_kwh").fill_null(0.0),
    ])

    stats = stats.with_columns([
        pl.max_horizontal([
            pl.col("baseline_kwh"),
            (pl.col("robust_max_kwh") *
             (cfg.household_minimum_R_percent / 100.0))
        ]).alias("floor_kwh_raw")
    ]).with_columns([
        pl.max_horizontal([pl.col("floor_kwh_raw"),
                           pl.lit(cfg.household_minimum_epsilon_floor_kWh)]
                          ).alias("floor_kwh")
    ]).select(["ca_id", "hod", "dow", "floor_kwh"])

    # Attach back to original df
    df_with_floor = (
        df
        .with_columns([
            pl.col("date").dt.hour().alias("hod"),
            pl.col("date").dt.strftime("%A").alias("dow"),
        ])
        .join(stats, on=["ca_id", "hod", "dow"], how="left")
        .drop(["hod", "dow"])
    )
    return df_with_floor


def build_arrays_for_group_pd(
        sub_df: pd.DataFrame,
        slot_len_min: int = 30,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Build dense, slot-aligned arrays for a single (ca_id, day, city)
    pandas group.

    Parameters
    ----------
    sub_df : pandas.DataFrame
        Expected columns:
        - 'slot' (int), 'value' (float, kWh)
        - 'marginal_emissions_factor_grams_co2_per_kWh' (float, gCO2/kWh)
        - 'average_emissions_factor_grams_co2_per_kWh' (float, gCO2/kWh)
        - 'floor_kwh' (optional, float)
    slot_len_min : int, optional
        Slot size in minutes. Defaults to 30 (→ 48 slots/day).

    Returns
    -------
    dict or None
        dict with keys: 'usage', 'mef', 'aef' and optional 'floor'.
        Returns None when sub_df is empty or required factors missing
        in used slots.
    """
    if sub_df.empty:
        return None
    if 60 % slot_len_min != 0:
        raise ValueError((f"slot_len_min must evenly divide 60 - "
                          f"got {slot_len_min}."))
    per_hour = 60 // slot_len_min
    T = 24 * per_hour

    usage = np.zeros(T, dtype=np.float32)
    mef = np.full(T, np.nan, dtype=np.float32)
    aef = np.full(T, np.nan, dtype=np.float32)
    floor: Optional[np.ndarray] = None

    sl = sub_df["slot"].to_numpy(dtype=int)
    if (sl < 0).any() or (sl >= T).any():
        raise ValueError((f"Found slot outside [0,{T-1}] for "
                         f"slot_len_min={slot_len_min}."))

    usage[sl] = sub_df["value"].to_numpy(dtype=np.float32)
    mef[sl] = (
        sub_df["marginal_emissions_factor_grams_co2_per_kWh"]
        .to_numpy(dtype=np.float32))
    aef[sl] = (
        sub_df["average_emissions_factor_grams_co2_per_kWh"]
        .to_numpy(dtype=np.float32))

    if "floor_kwh" in sub_df.columns:
        floor = np.zeros(T, dtype=np.float32)
        floor[sl] = sub_df["floor_kwh"].to_numpy(dtype=np.float32)

    # now handles by solvers and valid slot
    # used = usage > 0
    # if used.any() and (np.isnan(mef[used]).any() or
    #  np.isnan(aef[used]).any()):
    #     return None

    out = {"usage": usage, "mef": mef, "aef": aef}
    if floor is not None:
        out["floor"] = floor
    return out


def build_cityday_ca_order_map_pl(
        pl_df: pl.DataFrame
) -> Dict[Tuple[str, pd.Timestamp], List[str]]:
    """
    Build, for each (city, day), the customer order by descending daily kWh.

    Parameters
    ----------
    pl_df : polars.DataFrame
        Must contain 'city', 'day', 'ca_id', 'value'.

    Returns
    -------
    Dict[Tuple[str, pandas.Timestamp], List[str]]
        Mapping: (city, day_midnight) -> ordered list of ca_id.

    Notes
    -----
    Useful to precompute a stable solve order and avoid repeated group-sorts.
    """
    # total kWh per (city, day, ca_id)
    daily = (
        pl_df
        .group_by(["city", "day", "ca_id"], maintain_order=False)
        .agg(pl.col("value").sum().alias("day_kwh"))
        .sort(by=["city", "day", "day_kwh"], descending=[False, False, True])
    )
    # Build grouped lists of ca_id ordered by day_kwh desc
    # (Polars: group again and collect ca_id lists in sorted order above)
    lists = (
        daily
        .group_by(["city", "day"], maintain_order=False)
        .agg(pl.col("ca_id").alias("ca_order"))
    )

    # Materialize to Python dict
    out: Dict[Tuple[str, pd.Timestamp], List[str]] = {}
    for row in lists.iter_rows(named=True):
        city = row["city"]
        day = pd.to_datetime(row["day"])  # ensure pandas Timestamp key
        out[(city, day)] = row["ca_order"]
    return out


def cityday_baseline_by_slot(
        df_city_day: pd.DataFrame,
        slot_len_min: int = 30,
        dtype=np.float32
) -> np.ndarray:
    """
    Compute city-day baseline per slot as the sum of original usage across
    customers.

    Parameters
    ----------
    df_city_day : pandas.DataFrame
        Rows for a single (city, day). Requires 'slot' and 'value'.
    slot_len_min : int
        Slot size in minutes. Defaults to 30.

    Returns
    -------
    np.ndarray
        Baseline usage (kWh) per slot.
    """
    if 60 % slot_len_min != 0:
        raise ValueError("slot_len_min must divide 60.")
    T = 24 * (60 // slot_len_min)
    base = np.zeros(T, dtype=dtype)
    grp = df_city_day.groupby("slot", as_index=False,
                              observed=False)["value"].sum()
    sl = grp["slot"].to_numpy(dtype=int)
    base[sl] = grp["value"].to_numpy(dtype=dtype)
    return base


def city_peak_targets_for_day(
        city: str,
        day_ts,
        peak_cfg: PeakHoursReductionLimitConfig,
        slot_len_min: int = 30,
) -> Tuple[Optional[np.ndarray],
           Optional[List[List[int]]],
           Optional[float], str]:
    """
    Expand user-provided full hours (e.g. 8, 9, 10) into slot indices
    for a given day.

    Parameters
    ----------
    city : str
        City name. Must exist as a key in `peak_cfg.peak_hours_dict`.
    day_ts : datetime
        Midnight timestamp for the day being solved (group key).
    peak_cfg : PeakHoursReductionLimitConfig
        Configuration object with Z% and hourly lists per (city, weekday).
    slot_len_min : int, optional
        Slot size in minutes. Defaults to 30 (→ 48 slots).

    Returns
    -------
    mask : np.ndarray or None
        Boolean mask of length T (True where *peak source* slots are located).
        Only used when applying a single cap per hour
        (i.e., limit_scope == "hour").

    groups : List[List[int]] or None
        Each inner list is the indices of the slots forming one *hour*.
        Only used when applying a single cap per hour
        (i.e., limit_scope == "hour").
    Z : float or None
        Fractional cap (e.g., 0.30 for 30%).
    limit_scope : {"slot", "hour"}
        Scope of the cap; mirrors `peak_cfg.limit_scope`.

    Notes
    -----
    - Weekday keys must match `strftime("%a")` ("Mon".."Sun").
    - If city or weekday has no configured hours, returns
        (None, None, None, "slot").
    """
    if not peak_cfg or not peak_cfg.peak_hours_dict:
        return None, None, None, "slot"

    wk = day_ts.strftime("%a")  # "Mon".."Sun"
    hours = (peak_cfg.peak_hours_dict.get(city, {}) or {}).get(wk, [])
    if not hours:
        return None, None, None, "slot"

    if 60 % slot_len_min != 0:
        raise ValueError(f"slot_len_min must divide 60; got {slot_len_min}.")

    per_hour = 60 // slot_len_min
    T = 24 * per_hour

    mask = np.zeros(T, dtype=bool)
    groups: List[List[int]] = []
    for h in hours:
        if 0 <= h <= 23:
            start = h * per_hour
            end = start + per_hour
            groups.append(list(range(start, end)))
            mask[start:end] = True

    Z = peak_cfg.peak_hours_reduction_percent_limit / 100.0
    return mask, groups, Z, peak_cfg.limit_scope


def _valid_slot_mask(
        mef: np.ndarray,
        aef: np.ndarray
) -> np.ndarray:
    """
    A slot is valid iff both MEF and AEF are finite and strictly > 0.
    Invalid slots are never used as sources or destinations for shifting.
    Baseline usage in invalid slots remains in place.

    Parameters:
    -----------
    mef : np.ndarray
        The marginal emission factor for each time slot.
    aef : np.ndarray
        The average emission factor for each time slot.

    Returns:
    -------
    np.ndarray
        A boolean mask indicating valid slots.
    """
    mef = np.asarray(mef, dtype=float)
    aef = np.asarray(aef, dtype=float)
    return np.isfinite(mef) & np.isfinite(aef) & (mef > 0.0) & (aef > 0.0)

# Emissions Math & Operations
# ────────────────────────────────────────────────────────────────────────────


def compute_emissions_totals(
        usage: np.ndarray,
        aef: np.ndarray,
        mef: np.ndarray,
) -> Dict[str, float]:
    """
    Compute total emissions in grams and tonnes of CO2 based on usage and
    emission factors.

    Parameters:
    -----------
    usage : np.ndarray
        The energy usage for each time slot.
    aef : np.ndarray
        The average emission factor for each time slot.
    mef : np.ndarray
        The marginal emission factor for each time slot.

    Returns:
    --------
    Dict[str, float]
        A dictionary containing the total emissions in grams and tonnes of CO2
        for both average and marginal weighting.

    """
    # returns totals in grams and tCO2 for both average and marginal weighting
    u = np.asarray(usage, dtype=float)
    a = np.asarray(aef, dtype=float)
    m = np.asarray(mef, dtype=float)

    mask_a = np.isfinite(a) & (a > 0.0) & (u != 0.0)
    mask_m = np.isfinite(m) & (m > 0.0) & (u != 0.0)

    g_avg = float(np.sum(u[mask_a] * a[mask_a]))
    g_mrg = float(np.sum(u[mask_m] * m[mask_m]))

    return {
        "E_avg_g": g_avg,
        "E_avg_t": tco2_from_grams(g_avg),
        "E_marg_g": g_mrg,
        "E_marg_t": tco2_from_grams(g_mrg),
    }


def grams_co2_from_kwh_grams_per_kwh(
        kwh: float,
        grams_co2_per_kwh: float
) -> float:
    """
    Convert energy in kWh to grams co2 based on a specific emission factor.

    Parameters:
    ----------
    kwh : float
        The energy in kilowatt-hours.
    grams_co2_per_kwh : float
        The emission factor in grams co2 per kilowatt-hour.

    Returns:
    -------
    float
        The equivalent emissions in grams of co2.
    """
    return float(kwh * grams_co2_per_kwh)


def flows_table(
        ca_id: str,
        city: str,
        day_ts,
        flows: List[Tuple[int, int, float]],
        mef: np.ndarray,
        aef: np.ndarray,
        slot_len_min: int = 30,
) -> List[Dict[str, Any]]:
    """
    Generate a table of flows with associated emissions data.

    Parameters:
    ----------
    ca_id : str
        The ID of the charging station.
    city : str
        The city where the charging station is located.
    day_ts : datetime
        The timestamp for the day of the flows.
    flows : List[Tuple[int,int,float]]
        A list of tuples representing the flows, where each tuple contains
        (start_time, end_time, kwh).
    mef : np.ndarray
        The marginal emission factor for each time slot.
    aef : np.ndarray
        The average emission factor for each time slot.
    slot_len_min : int, optional
        The length of each time slot in minutes (default is 30).

    Returns:
    -------
    List[Dict[str, Any]]
        A list of dictionaries containing the flow data with emissions
        information.
    """
    rows: List[Dict[str, Any]] = []
    for (t, s, kwh) in flows:
        t_minutes = t * slot_len_min
        s_minutes = s * slot_len_min
        t_ts = day_ts + timedelta(minutes=int(t_minutes))
        s_ts = day_ts + timedelta(minutes=int(s_minutes))
        dirn = "forward" if s > t else ("backward" if s < t else "stay")

        # Per-move emissions deltas (grams) using marginal & average
        g_marg_before = grams_co2_from_kwh_grams_per_kwh(kwh, mef[t])
        g_marg_after = grams_co2_from_kwh_grams_per_kwh(kwh, mef[s])
        g_marg_delta = g_marg_before - g_marg_after

        g_avg_before = grams_co2_from_kwh_grams_per_kwh(kwh, aef[t])
        g_avg_after = grams_co2_from_kwh_grams_per_kwh(kwh, aef[s])
        g_avg_delta = g_avg_before - g_avg_after

        rows.append({
            "ca_id": ca_id,
            "city": city,
            "day": day_ts,
            "original_time": t_ts,
            "proposed_shift_time": s_ts,
            "delta_minutes": int((s - t) * slot_len_min),
            "shift_direction": dirn,
            "delta_kwh": float(kwh),
            "marginal_emissions_before_shift_grams_co2": g_marg_before,
            "marginal_emissions_after_shift_grams_co2": g_marg_after,
            "marginal_emissions_delta_grams_co2": g_marg_delta,
            "average_emissions_before_shift_grams_co2": g_avg_before,
            "average_emissions_after_shift_grams_co2": g_avg_after,
            "average_emissions_delta_grams_co2": g_avg_delta,
        })
    return rows


def tco2_from_grams(g: float) -> float:
    """
    Convert grams of CO₂ to tonnes of CO₂.

    Parameters
    ----------
    g : float
        Mass in grams.

    Returns
    -------
    float
        Mass in tonnes (tCO₂).
    """
    return g / 1_000_000.0


def weighted_median_shift_minutes(
        flows: List[Tuple[int, int, float]],
        slot_len_min: int = 30
) -> float:
    """
    Energy-weighted median of absolute shift distance, in minutes.

    Parameters
    ----------
    flows : list[tuple[int,int,float]]
        (source_slot, dest_slot, kWh) moves.
    slot_len_min : int, optional
        Slot length in minutes. Default 30.

    Returns
    -------
    float
        Weighted median(|Δslots|) * slot_len_min. Returns 0.0 if no flows.
    """
    if not flows:
        return 0.0
    steps = np.array([abs(s - t) for (t, s, _) in flows], dtype=float)
    w = np.array([kwh for (_, _, kwh) in flows], dtype=float)
    order = np.argsort(steps)
    steps, w = steps[order], w[order]
    cum = np.cumsum(w) / (w.sum() + 1e-12)
    idx = np.searchsorted(cum, 0.5, side="left")
    return float(steps[min(idx, len(steps)-1)] * slot_len_min)

# Solver Helpers
# ────────────────────────────────────────────────────────────────────────────


def _pick_cvxpy_solver(
    requested: Optional[str],
    prefer_order: Tuple[str, ...] = _SUPPORTED_LP_SOLVERS,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Pick an installed CVXPY solver, preferring `requested` if available,
    otherwise fall back through `prefer_order`. Returns a tuple:
      (solver_param_for_cvxpy, chosen_name)
    where solver_param_for_cvxpy can be either cp.<NAME> or the string NAME.

    Parameters:
    ----------
    requested : Optional[str]
        The name of the requested solver (if any).
    prefer_order : Tuple[str, ...]
        The preferred order of solvers to fall back on.

    Returns:
    -------
    Tuple[Optional[Any], Optional[str]]
        - The CVXPY solver parameter (if found).
        - The name of the chosen solver (if found).
    """
    if cp is None:
        return None, None

    installed = set(cp.installed_solvers())  # e.g. {"HIGHS","ECOS","SCS",...}

    # Build the priority list: requested first (if any), then the rest
    if requested is None:
        candidates = list(prefer_order)
    else:
        candidates = [requested] + [s for s in prefer_order if s != requested]

    for name in candidates:
        if name in installed:
            # CVXPY accepts either a string or cp.<CONST>; both are fine.
            solver_param = getattr(cp, name, name)
            if requested and name != requested:
                warnings.warn(
                    f"Requested solver '{requested}' not available; "
                    f"falling back to '{name}'.",
                    RuntimeWarning,
                )
            return solver_param, name

    # Nothing available from our supported list → let CVXPY decide
    if requested:
        warnings.warn(
            f"Requested solver '{requested}' not available and no supported "
            "fallback found. Letting CVXPY choose.",
            RuntimeWarning,
        )
    return None, None


# Core Solvers
# ────────────────────────────────────────────────────────────────────────────


_CONTINUOUS_WARNED = False


def solve_continuous(
        *,
        mef: np.ndarray,
        usage: np.ndarray,
        method: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        valid_slot_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
    """
    Relaxed reallocation: minimize mef·x subject to x >= 0
    and sum(x) = sum(usage).
    NOTE: This ignores shift windows, floors, peak caps,
    and anti-spike constraints.
    Use as a lower-bound or sanity-check solution.

    Parameters:
    -----------
    mef : np.ndarray
        Marginal emissions factors (gCO₂/kWh) for each time slot.
    usage : np.ndarray
        Baseline usage (kWh) for each time slot.
    method : Optional[str]
        Optimization method to use: "SLSQP", "trust-constr", or "closed-form"
        See docs for which one is best suited for your problem.
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-trustconstr.html
        Generally, SLSQP is quick and reliable for smooth problems,
        while Trust-Region methods can handle non-smooth cases better,
        better suited for problems with complex constraints.
    options : Optional[Dict[str, Any]]
        Additional solver options to pass to `scipy.optimize.minimize`.
    valid_slot_mask : Optional[np.ndarray]
        Mask indicating valid slots for optimization (1 for valid, 0 for
        invalid).

    Returns:
    -------
    Tuple[np.ndarray, List[Tuple[int, int, float]]]
        - Optimized usage (kWh) for each time slot.
        - List of (source_slot, dest_slot, kWh) moves (empty if no moves).
    """
    # allow closed-form path without SciPy; otherwise require SciPy
    want_closed_form = (method or "").strip().lower() in ("closed-form",
                                                          "closed",
                                                          "cf")
    # existing gate, slightly relaxed
    if (not want_closed_form) and (minimize is None):
        raise RuntimeError(
            "scipy is required for 'continuous' mode unless method="
            "'closed-form'. (e.g., pip install scipy)."
        )

    global _CONTINUOUS_WARNED
    if not _CONTINUOUS_WARNED:
        warnings.warn(
            "Using the 'continuous' relaxer: ignores windows, floors, "
            "peak caps, and anti-spike constraints. Intended for a lower-"
            "bound/sanity check only.",
            RuntimeWarning,
        )
        _CONTINUOUS_WARNED = True

    mef = np.asarray(mef, dtype=float)
    usage = np.asarray(usage, dtype=float)
    usage = np.maximum(usage, 0.0)  # clamp tiny negatives

    # Build validity mask (caller should pass _valid_slot_mask(mef, aef))
    if valid_slot_mask is None:
        vm = np.isfinite(mef) & (mef > 0.0)
    else:
        vm = valid_slot_mask.astype(bool) & np.isfinite(mef) & (mef > 0.0)

    # Nothing valid or nothing to move → baseline
    if not np.any(vm):
        return usage.astype(float, copy=True), []

    movable_total = float(np.nansum(np.where(vm, usage, 0.0)))
    if movable_total <= 1e-12:
        return usage.astype(float, copy=True), []

    if want_closed_form:
        # keep invalid slots unchanged
        x_full = usage.astype(float, copy=True)
        mef_valid = mef[vm]
        idx_min_local = int(np.argmin(mef_valid))
        # map local index back to full index
        full_indices = np.flatnonzero(vm)
        best_full = int(full_indices[idx_min_local])
        x_full[vm] = 0.0
        x_full[best_full] = movable_total
        return x_full, []

    # SciPy paths (SLSQP / trust-constr) — compact domain over valid slots
    mef_f = mef[vm]
    x0 = np.nan_to_num(usage[vm], nan=0.0, posinf=0.0, neginf=0.0)
    idx_min_local = int(np.argmin(mef_f))  # cleanest valid slot

    def obj(x: np.ndarray) -> float:
        # Linear objective; cast to float for scipy
        return float(np.dot(x, mef_f))

    def grad(x: np.ndarray) -> np.ndarray:
        # Constant gradient for linear objective
        return mef_f

    method = (method or "SLSQP").upper()
    options = options or {}

    if method == "TRUST-CONSTR":
        # trust-constr path
        n = x0.size

        def zero_hess(x):  # exact Hessian for linear objective is zero
            return np.zeros((n, n), dtype=float)

        bounds = Bounds(lb=np.zeros_like(x0), ub=np.full_like(x0, np.inf))
        A = np.ones((1, x0.size))
        lin_eq = LinearConstraint(A, lb=[movable_total], ub=[movable_total])
        res = minimize(
            obj,
            x0,
            method="trust-constr",
            jac=grad,
            hess=zero_hess,
            bounds=bounds,
            constraints=[lin_eq],
            options={"xtol": 1e-12, "gtol": 1e-12, **options},
        )
    else:
        # default SLSQP
        bounds = [(0.0, None)] * x0.size
        constraints = {
            "type": "eq",
            "fun": lambda x: np.sum(x) - movable_total,
            "jac": lambda x: np.ones_like(x)
        }
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            jac=grad,
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, **options},
        )

    # Build full 48-slot solution, with zeros on non-finite MEF slots
    x_full = usage.astype(float, copy=True)

    if res.success and np.all(np.isfinite(res.x)):
        x_f = res.x.copy()

        # Tidy small numerical negatives while preserving the equality
        neg = x_f < 0
        if np.any(neg):
            deficit = float(-x_f[neg].sum())
            x_f[neg] = 0.0
            x_f[idx_min_local] += deficit  # keep sum(x_f) == total

        # Final tiny sum repair (rare)
        s = float(x_f.sum())
        if not np.isclose(s, movable_total, rtol=0, atol=1e-9):
            x_f[idx_min_local] += (movable_total - s)

        x_full[vm] = x_f
        return x_full, []
    else:
        # Robust fallback: put everything in cleanest finite slot
        x_f = np.zeros_like(mef_f)
        x_f[idx_min_local] = movable_total
        x_full[vm] = x_f
        return x_full, []


def greedy_k_moves(
        usage: np.ndarray,
        mef: np.ndarray,
        aef: np.ndarray,
        *,
        W_slots: int,
        slot_len_min: int,
        floor_vec: Optional[np.ndarray],
        peak_mask: Optional[np.ndarray],
        peak_groups: Optional[List[List[int]]],
        Z: Optional[float],
        cap_mode: Literal["slot", "hour"] = "slot",
        dest_rem: Optional[np.ndarray],              # ← Optional
        reg_rem: Optional[float],
        max_moves: int = 1,
        enforce_distinct_sources: bool = True,
        valid_slot_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], float, float]:
    """
    Greedy heuristic: perform up to `max_moves` beneficial source→dest
    transfers.

    At each step, pick the (t→s) that yields the largest emissions reduction,
    respecting all caps (floor, comfort Z on peak slots/hours, destination
    anti-spike headroom, and regional moved-kWh budget). Optionally enforces
    distinct source slots across moves.

    Parameters
    ----------
    usage : np.ndarray
        Baseline kWh per slot (length T).
    mef : np.ndarray
        Marginal emissions (g/kWh) per slot.
    aef : np.ndarray
        Average emissions (g/kWh) per slot (used for reporting only).
    W_slots : int
        Move window in slots (|t - s| ≤ W_slots).
    slot_len_min : int
        Slot length in minutes.
    floor_vec : np.ndarray or None
        Minimum remaining usage per slot; if None, treated as zeros.
    peak_mask : np.ndarray or None
        Boolean mask of *peak source* slots; None if no comfort cap. When a
        source slot t is peak (peak_mask[t] is True), the total energy moved
        out of that slot/hour is capped by Z (per-slot or per-hour).
    peak_groups : List[List[int]] or None
        Per-hour groups of slot indices, used when cap_mode == "hour".
    Z : float or None
        Comfort reduction fraction (e.g., 0.3 for 30%)
        None disables comfort cap.
    cap_mode : {"slot","hour"}, optional
        Scope for comfort cap. Default "slot".
    dest_rem : np.ndarray
        Remaining city anti-spike headroom per destination slot (kWh).
    reg_rem : float
        Remaining regional moved-kWh budget for the day.
    max_moves : int, optional
        Max number of distinct source slots to move from. Default 1.
    enforce_distinct_sources : bool, optional
        If True, each move must use a new source t. Default True.
    valid_slot_mask : np.ndarray, optional
        A boolean mask indicating valid slots.

    Returns
    -------
    usage_opt : np.ndarray
        Updated usage after moves.
    flows : List[Tuple[int,int,float]]
        Executed moves as (t, s, q_kwh).
    used_dest_total : float
        Cumulative kWh added to destination slots.
    used_reg_total : float
        Cumulative kWh counted against regional budget.

    Notes
    -----
    - Complexity ~ O(max_moves * T * W_slots).
    - Moves will not be created if they do not reduce emissions
        (mef[s] < mef[t]).
    - If a source slot has zero (or floor-equal) usage, it is skipped.
    """
    T = len(usage)
    usage_opt = usage.copy()
    flows: List[Tuple[int, int, float]] = []
    used_dest_total = 0.0
    used_reg_total = 0.0

    # Track which sources have already been used (distinct-sources cap)
    used_sources: Set[int] = set()

    # Peak comfort remaining reducible energy (per slot or per hour)
    # for THIS customer-day
    if Z is not None and peak_mask is not None and peak_mask.any():
        if cap_mode == "slot":
            peak_rem_slot = np.zeros(T, dtype=np.float32)
            peak_rem_slot[peak_mask] = (
                np.float32(Z) * usage_opt[peak_mask].astype(np.float32,
                                                            copy=False)
                )
            peak_hour_rem: Optional[List[float]] = None
        else:
            peak_hour_rem = []
            for grp in (peak_groups or []):
                peak_hour_rem.append(Z * float(usage_opt[grp].sum()))
            peak_rem_slot = None
    else:
        peak_rem_slot = None
        peak_hour_rem = None

    if valid_slot_mask is None:
        valid_slot_mask = _valid_slot_mask(mef, aef)

    for _ in range(max_moves):
        best_gain = 0.0
        best: Optional[Tuple[int, int, float]] = None

        for t in range(T):
            # must be a valid source slot
            if not valid_slot_mask[t]:
                continue
            # Distinct sources: skip if t already used in a previous move
            if enforce_distinct_sources and (t in used_sources):
                continue
            # Skip if no movable energy
            if usage_opt[t] <= 1e-12:
                continue

            floor_t = float(floor_vec[t]) if (floor_vec is not None) else 0.0
            avail_from_t = max(0.0, usage_opt[t] - floor_t)
            if avail_from_t <= 1e-12:
                continue

            # Peak remaining allowance if moving OUT of peak
            peak_lim_t = float("inf")
            if peak_rem_slot is not None and (peak_rem_slot[t] > 0.0):
                peak_lim_t = peak_rem_slot[t]
            elif peak_hour_rem is not None and peak_groups is not None:
                # find hour group index for t (only if t is inside
                # any peak group)
                for k, grp in enumerate(peak_groups):
                    if t in grp:
                        peak_lim_t = max(0.0, peak_hour_rem[k])
                        break

            s0, s1 = max(0, t - W_slots), min(T, t + W_slots + 1)
            for s in range(s0, s1):
                if s == t:
                    continue
                # destination must be valid
                if not valid_slot_mask[s]:
                    continue
                # Only consider moves that reduce emissions
                if mef[s] >= mef[t] - 1e-12:
                    continue

                # Destination anti-spike residual
                cap_dest = (
                    float(dest_rem[s])
                    if dest_rem is not None else float("inf")
                    )
                # Regional budget
                cap_reg = (
                    float(reg_rem)
                    if reg_rem is not None else float("inf")
                    )

                q_max = min(avail_from_t, cap_dest, cap_reg)
                # If moving OUT of peak, also cap by remaining peak
                if (peak_mask is not None) and peak_mask.any(
                                                ) and peak_mask[t]:
                    q_max = min(q_max, peak_lim_t)

                if q_max <= 1e-12:
                    continue

                # emissions gain = (mef[t]-mef[s]) * q
                gain = (mef[t] - mef[s]) * q_max
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best = (t, s, q_max)

        if best is None:
            break  # no beneficial distinct-source move left

        # Apply the best move
        t, s, q = best
        usage_opt[t] -= q
        usage_opt[s] += q
        flows.append((t, s, float(q)))

        # Mark source as used if enforcing distinct sources
        if enforce_distinct_sources:
            used_sources.add(t)

        # Update shared caps
        if dest_rem is not None:
            dest_rem[s] = max(0.0, dest_rem[s] - q)
        if reg_rem is not None:
            reg_rem = max(0.0, reg_rem - q)

        used_dest_total += q
        used_reg_total += q

        # Update peak allowance if we moved OUT of peak
        if Z is not None and peak_mask is not None and peak_mask.any(
                                                    ) and peak_mask[t]:
            if peak_rem_slot is not None:
                peak_rem_slot[t] = max(0.0, peak_rem_slot[t] - q)
            elif peak_hour_rem is not None and peak_groups is not None:
                for k, grp in enumerate(peak_groups):
                    if t in grp:
                        peak_hour_rem[k] = max(0.0, peak_hour_rem[k] - q)
                        break

    return usage_opt, flows, used_dest_total, used_reg_total


def solve_lp_k(
        *,
        mef: np.ndarray,
        usage: np.ndarray,
        W_slots: int,
        floor_vec: Optional[np.ndarray],
        peak_mask: Optional[np.ndarray],
        peak_groups: Optional[List[List[int]]],
        Z: Optional[float],
        cap_mode: Literal["slot", "hour"],
        dest_upper_bounds: Optional[np.ndarray],
        moved_kwh_cap: Optional[float],
        lp_solver: Optional[str] = "HIGHS",
        lp_solver_opts: Optional[Dict[str, Any]] = None,
        valid_slot_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Tuple[int, int, float]], Optional[str]]:
    """
    LP relaxation on per-pair flows y_{t->s} within window |t-s| <= W_slots.

    NOTE: This LP **does not** enforce 'max_moves' (K distinct sources). That
    requires integer variables. All other caps mirror MILP where possible.

    Parameters:
    -----------
    mef : np.ndarray
        Marginal emissions factors (gCO₂/kWh) for each time slot.
    usage : np.ndarray
        Baseline usage (kWh) for each time slot.
    W_slots : int
        Time window size (number of slots) for the LP relaxation.
    floor_vec : Optional[np.ndarray]
        Minimum usage floors (kWh) for each time slot.
    peak_mask : Optional[np.ndarray]
        Boolean mask of *peak source* slots (length T). When True at t, the
        total energy that can be moved out of slot t (or its hour group) is
        capped by Z.
    peak_groups : Optional[List[List[int]]]
        List of groups of peak slots (for per-hour caps).
    Z : Optional[float]
        Comfort reduction fraction for peak caps (e.g., 0.30 for 30%).
    cap_mode : Literal["slot", "hour"]
        Cap mode for peak constraints.
    dest_upper_bounds : Optional[np.ndarray]
        Upper bounds on destination usage (kWh).
    moved_kwh_cap : Optional[float]
        Cap on moved kWh (if any).
    lp_solver : Optional[str] = "HIGHS"
        LP solver to use (default: HIGHS).
        HIGHS, GLPK, SCS, CLARABEL, ECOS, OSQP
    lp_solver_opts : Optional[Dict[str, Any]] = None
        Options for the LP solver.
    valid_slot_mask : Optional[np.ndarray] = None
        A boolean mask indicating valid slots.

    Returns:
    -------
    Tuple[np.ndarray, List[Tuple[int, int, float]], Optional[str]]
        - Optimized usage (kWh) for each time slot.
        - List of (source_slot, dest_slot, kWh) moves (empty if no moves).
        - Name of the LP solver used (if applicable).
    """
    if cp is None:
        raise RuntimeError(
            "cvxpy is required for LP mode. Install cvxpy and a solver "
            "(e.g., pip install cvxpy && apt-get install glpk-utils)."
        )

    T = len(usage)
    if valid_slot_mask is None:
        # MEF only here, so treat invalid if MEF is non-finite or <=0
        valid_slot_mask = np.isfinite(mef) & (mef > 0.000)
    pairs, by_src, by_dst = cached_pairs(T, W_slots)
    P = len(pairs)

    y = cp.Variable(P, nonneg=True)

    # Objective: minimize post emissions Σ_{t,s} y_{t,s} * mef[s]
    # Cost coefficients: 0 for invalid destinations (including self)
    # to avoid NaNs.
    cost_coeffs = np.array([
        float(mef[s]) if (np.isfinite(mef[s]) and mef[s] > 0.000) else 0.0
        for (_, s) in pairs
    ], dtype=float)
    cost = cp.sum(cp.multiply(y, cost_coeffs))

    cons: List[Any] = []

    # Supply conservation per source: Σ_s y_{t,s} = usage[t]
    for t in range(T):
        idx = by_src[t].tolist()
        cons.append(cp.sum(y[idx]) == float(usage[t]))

    # Source floors: Σ_{s≠t} y_{t,s} ≤ usage[t] - floor[t]
    if floor_vec is not None:
        for t in range(T):
            idx_moves = [i for i in by_src[t].tolist() if pairs[i][1] != t]
            if not idx_moves:
                continue
            ub_src = max(0.0, float(usage[t] - float(floor_vec[t])))
            cons.append(cp.sum(y[idx_moves]) <= ub_src)

    # INVALID SOURCES: forbid moving out (except staying)
    for t in range(T):
        if not valid_slot_mask[t]:
            idx_moves = [i for i in by_src[t].tolist() if pairs[i][1] != t]
            if idx_moves:
                cons.append(cp.sum(y[idx_moves]) <= 0.0)

    # Peak caps
    if (Z is not None) and (Z >= 0.0) and (peak_mask
                                           is not None) and peak_mask.any():
        if cap_mode == "slot":
            # per-slot: Σ_{s≠t} y_{t,s} ≤ Z·usage[t] for peak source slots t
            for t in np.where(peak_mask)[0].tolist():
                idx_moves = [i for i in by_src[t].tolist() if pairs[i][1] != t]
                if idx_moves:
                    cons.append(cp.sum(y[idx_moves]) <= float(Z)
                                * float(usage[t]))
        else:
            # per-hour groups: Σ_{t∈G} Σ_{s≠t} y_{t,s} ≤ Z·Σ_{t∈G} usage[t]
            if peak_groups:
                for grp in peak_groups:
                    grp_moves, base_h = [], 0.0
                    for t in grp:
                        idx_moves = (
                            [i for i in by_src[t].tolist() if pairs[i][1] != t]
                            )
                        grp_moves.extend(idx_moves)
                        base_h += float(usage[t])
                    if grp_moves and base_h > 1e-12:
                        cons.append(cp.sum(y[grp_moves]) <= float(Z) * base_h)

    # Destination caps: start from provided bounds or +inf, then clamp
    # invalid dests to 0
    # Destination anti-spike: Σ_t y_{t,s} ≤ usage[s] + dest_upper_bounds[s]

    if dest_upper_bounds is None:
        dest_cap = np.full(T, np.inf, dtype=float)
    else:
        dest_cap = np.asarray(dest_upper_bounds, dtype=float).copy()

    # For invalid destinations, forbid net inflow (Σ_t y_{t,s} ≤ usage[s] + 0)
    for s in range(T):
        cap = float(dest_cap[s])
        idx = by_dst[s].tolist()
        if not idx:
            continue
        if not valid_slot_mask[s]:
            cons.append(cp.sum(y[idx]) <= float(usage[s]) + 0.0)
        elif np.isfinite(cap):
            # Enforce anti-spike only when finite
            cons.append(cp.sum(y[idx]) <= float(usage[s]) + cap)
    # Regional moved-kWh cap: Σ_t Σ_{s≠t} y_{t,s} ≤ moved_kwh_cap
    if moved_kwh_cap is not None:
        move_idxs = [i for i, (t, s) in enumerate(pairs) if s != t]
        if move_idxs:
            cons.append(cp.sum(y[move_idxs]) <= float(moved_kwh_cap))

    prob = cp.Problem(cp.Minimize(cost), cons)

    # Choose solver
    solve_kwargs: Dict[str, Any] = {}
    if lp_solver_opts:
        solve_kwargs.update(lp_solver_opts)

    solver_param, chosen_name = _pick_cvxpy_solver(lp_solver)

    try:
        if solver_param is not None:
            prob.solve(solver=solver_param, **solve_kwargs)
        else:
            prob.solve(**solve_kwargs)  # let CVXPY pick
    except Exception as e:
        raise RuntimeError((f"LP solve failed (solver="
                            f"{chosen_name or 'auto'}): {e}"))

    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.USER_LIMIT):
        # return baseline if infeasible or failed
        return usage.astype(float), [], chosen_name

    yv = np.asarray(y.value, dtype=float)
    yv[np.isnan(yv)] = 0.0

    usage_opt = np.zeros(T, dtype=float)
    flows: List[Tuple[int, int, float]] = []
    for i, val in enumerate(yv):
        if val <= 1e-12:
            continue
        t, s = pairs[i]
        usage_opt[s] += val
        if s != t:
            flows.append((t, s, float(val)))
    return usage_opt, flows, chosen_name


def _solve_cityweek_worker(args):
    (
        city, week_ts, df_cityweek,
        policy, solver,
        slot_len, W_slots,
        cityday_ca_order,
        emit_optimised_rows
    ) = args
    """
    Solve a single (city, week) workload: iterate days and customers,
    apply solver.

    Parameters
    ----------
    args : tuple
        - city : str, city
        - week_ts : pd.Timestamp, week start timestamp
        - df_cityweek : pd.DataFrame, data for the city-week
        - policy : PolicyConfig, optimization policy
        - solver : SolverConfig, solver configuration
        - slot_len : int, slot length in minutes
        - W_slots : int, number of worker slots
        - cityday_ca_order : dict[(city, day) -> list of ca_id] or None
        - emit_optimised_rows : bool, whether to emit optimized rows

    Returns
    -------
    m_rows : List[Dict[str, Any]]
        Per-customer and per-day summary metrics.
    mv_rows : List[Dict[str, Any]]
        Per-move records (optional detail).
    o_rows : List[Dict[str, Any]]
        Per-slot optimized usage rows if `emit_optimised_rows=True`.

    Notes
    -----
    - Enforces weekly and daily move quotas via `policy.behavioral`.
    - Applies floor, peak comfort, anti-spike, and regional caps as configured.
    - Prints a simple wall-clock for this worker’s city-week.
    - max_moves  is only enforced for the "greedy" solver.
    """

    m_rows, mv_rows, o_rows = [], [], []

    # Weekly move counters per customer
    weekly_quota = dict.fromkeys(
        df_cityweek["ca_id"].unique().tolist(),
        policy.behavioral.customer_power_moves_per_week
    )
    start_time = time.perf_counter()

    for day_ts, df_city_day in df_cityweek.groupby("day",
                                                   sort=True,
                                                   group_keys=False):
        day_t0 = time.perf_counter()
        day_moved_sum = 0.0
        n_ca_solved = 0

        # float32 city-day arrays
        base_city = cityday_baseline_by_slot(df_city_day,
                                             slot_len_min=slot_len,
                                             dtype=np.float32)

        # City-level anti-spike post cap (+α% vs baseline), only if configured
        if policy.spike_cap is not None:
            alpha = np.float32(policy.spike_cap.alpha_peak_cap_percent / 100.0)
            post_city_cap = (1.0 + alpha) * base_city.astype(np.float32,
                                                             copy=False)
            post_city_used = np.zeros_like(base_city, dtype=np.float32)
        else:
            post_city_cap = None
            post_city_used = None  # disabled

        # Regional moved-kWh budget (per city-day), only if configured
        if policy.regional_cap is not None:
            P_pct = np.float32(
                policy.regional_cap.regional_load_shift_percent_limit / 100.0
            )
            city_daily_avg = np.float32(
                policy.regional_cap.regional_total_daily_average_load_kWh.get(
                    city, 0.0)
            )
            moved_budget_remaining = np.float32(P_pct * city_daily_avg)
        else:
            moved_budget_remaining = None  # disabled

        # Precomputed per-day customer order (high-usage first)
        if cityday_ca_order is not None:
            order = cityday_ca_order.get((city, day_ts))
            if order is None:
                order = df_city_day["ca_id"].drop_duplicates().tolist()
        else:
            order = (
                df_city_day.groupby("ca_id",
                                    as_index=False,
                                    observed=False)["value"].sum()
                .sort_values("value", ascending=False)["ca_id"]
                .tolist()
            )

        # PRE-SPLIT once per day: avoid repeated boolean filters
        # on Arrow strings
        by_ca = {cid: g for cid, g in df_city_day.groupby("ca_id",
                                                          sort=False,
                                                          observed=False)}

        for ca_id in order:
            if weekly_quota.get(ca_id, 0) <= 0:
                continue

            # O(1) lookup instead of df_city_day[df_city_day["ca_id"] == ca_id]
            sub = by_ca.get(ca_id)
            if sub is None or sub.empty:
                continue

            arrs = build_arrays_for_group_pd(sub, slot_len_min=slot_len)
            if arrs is None:
                continue

            usage = arrs["usage"].astype(np.float32, copy=False)
            # guard against tiny negatives from data issues
            usage = np.maximum(usage, np.float32(0.0))

            mef = arrs["mef"].astype(np.float32, copy=False)
            aef = arrs["aef"].astype(np.float32, copy=False)

            # VALID slots: MEF/AEF finite and >0. Invalid slots are never
            # sources/dests.
            valid_mask = _valid_slot_mask(mef, aef)

            floor_vec = arrs.get("floor")
            if floor_vec is not None:
                floor_vec = floor_vec.astype(np.float32, copy=False)

            peak_cfg = policy.behavioral.peak_hours_reduction_limit_config
            peak_mask, peak_groups, Z, cap_mode = city_peak_targets_for_day(
                city, day_ts, peak_cfg, slot_len_min=slot_len
            )
            if Z is not None:
                Z = np.float32(Z)

            # Note = only enforced for the "greedy" solver.
            K_sources = (
                int(max(0, min(
                            policy.behavioral.customer_power_moves_per_day,
                            weekly_quota[ca_id])))
                )

            # --- Destination headroom (anti-spike) for this customer
            if post_city_cap is not None:
                # City headroom minus this customer's baseline at each slot
                resid_cap = np.maximum(
                    np.float32(0.0),
                    post_city_cap - post_city_used - usage
                ).astype(np.float32, copy=False)
                dest_bounds = resid_cap
            else:
                dest_bounds = None  # anti-spike disabled

            # Always block inflow to invalid destinations
            if dest_bounds is None:
                safe_dest_bounds = np.full_like(usage,
                                                np.inf,
                                                dtype=np.float32)
            else:
                safe_dest_bounds = dest_bounds.copy()

            safe_dest_bounds[~valid_mask] = 0.0

            # Also block moving OUT of invalid sources
            # If greedy/LP don't accept a validity mask, emulate by
            # raising the floor.
            if floor_vec is None:
                floor_eff = np.zeros_like(usage, dtype=np.float32)
            else:
                floor_eff = floor_vec.copy()
            floor_eff[~valid_mask] = np.maximum(
                floor_eff[~valid_mask],
                usage[~valid_mask]
            )

            # Regional cap for this customer solve
            moved_kwh_cap = (
                float(moved_budget_remaining)
                if moved_budget_remaining is not None
                else None
            )

            t0_solve = time.perf_counter()

            # solve
            if solver.solver_family == "lp":
                usage_opt, flows, lp_used = solve_lp_k(
                    mef=mef,
                    usage=usage,
                    W_slots=W_slots,
                    floor_vec=floor_eff,
                    peak_mask=peak_mask,
                    peak_groups=peak_groups,
                    Z=float(Z) if Z is not None else None,
                    cap_mode=cap_mode,
                    dest_upper_bounds=safe_dest_bounds,
                    moved_kwh_cap=moved_kwh_cap,
                    lp_solver=solver.lp_solver,
                    lp_solver_opts=solver.lp_solver_opts,
                    valid_slot_mask=valid_mask,
                    # may raise TypeError if not supported
                )
                low_used = lp_used  # record GLPK/HIGHS/etc.

            elif solver.solver_family == "continuous":
                method_used = solver.continuous_method or "SLSQP"
                usage_opt, flows = solve_continuous(
                    mef=mef.astype(float),
                    usage=usage.astype(float),
                    method=method_used,
                    options=solver.continuous_opts,
                    valid_slot_mask=valid_mask,
                )
                low_used = method_used  # record SLSQP/trust-constr/etc.

            elif solver.solver_family == "greedy":
                # greedy
                usage_opt, flows, used_dest, used_reg = greedy_k_moves(
                    usage=usage,
                    mef=mef,
                    aef=aef,
                    W_slots=W_slots,
                    slot_len_min=slot_len,
                    # effective floor blocks invalid sources
                    floor_vec=floor_eff,
                    peak_mask=peak_mask,
                    peak_groups=peak_groups,
                    Z=Z,
                    cap_mode=cap_mode,
                    dest_rem=safe_dest_bounds.copy(),
                    reg_rem=moved_kwh_cap,
                    max_moves=K_sources,
                    # may raise TypeError if not supported
                    valid_slot_mask=valid_mask,
                    )
                low_used = None  # greedy has no low-level solver
            else:
                raise ValueError(f"Unknown Solver: {solver.solver_family} "
                                 f"not implemented")

            solve_wall_s = time.perf_counter() - t0_solve
            usage_opt = usage_opt.astype(np.float32, copy=False)

            # update tallies
            if post_city_used is not None:
                post_city_used += usage_opt
            moved_kwh = float(np.maximum(
                                np.float32(0.0),
                                usage - usage_opt).sum(dtype=np.float32))
            if moved_budget_remaining is not None:
                moved_budget_remaining = np.maximum(
                    np.float32(0.0),
                    moved_budget_remaining - np.float32(moved_kwh)
                )
            weekly_moves_used = len({t for (t, s, _) in flows if s != t})
            weekly_quota[ca_id] = (
                max(0, weekly_quota[ca_id] - weekly_moves_used)
                )

            # metrics row (emit as float64 for reporting)
            base = compute_emissions_totals(
                usage.astype(float),
                aef.astype(float),
                mef.astype(float)
                )
            post = compute_emissions_totals(
                usage_opt.astype(float),
                aef.astype(float),
                mef.astype(float)
                )
            median_shift = weighted_median_shift_minutes(flows, slot_len)

            m_rows.append({
                "ca_id": ca_id, "city": city, "day": day_ts,
                "solver_family_used": solver.solver_family,
                "solver_low_level_used": low_used,
                "baseline_E_avg_g": base["E_avg_g"],
                "post_E_avg_g": post["E_avg_g"],
                "delta_E_avg_g": base["E_avg_g"] - post["E_avg_g"],
                "baseline_E_marg_g": base["E_marg_g"],
                "post_E_marg_g": post["E_marg_g"],
                "delta_E_marg_g": base["E_marg_g"] - post["E_marg_g"],
                "baseline_kwh": float(np.sum(usage, dtype=np.float32)),
                "post_kwh": float(np.sum(usage_opt, dtype=np.float32)),
                "moved_kwh": moved_kwh,
                "avg_shift_minutes_energy_weighted":
                (float((sum(abs(s - t) * val
                            for (t, s, val) in flows) / max(moved_kwh,
                                                            1e-9)
                        ) * slot_len
                       ) if flows else 0.0),
                "median_shift_minutes_energy_weighted": median_shift,
                "weekly_moves_used": weekly_moves_used,
                "weekly_moves_remaining": weekly_quota[ca_id],
                "solve_wall_s": float(solve_wall_s),
            })

            # moves
            mv_rows.extend(
                flows_table(ca_id,
                            city,
                            day_ts,
                            flows,
                            mef.astype(float),
                            aef.astype(float),
                            slot_len_min=slot_len)
            )

            # optional per-slot outputs
            if emit_optimised_rows:
                for s in range(len(usage_opt)):
                    ts = day_ts + timedelta(minutes=int(s * slot_len))
                    o_rows.append({
                        "ca_id": ca_id,
                        "city": city,
                        "day": day_ts,
                        "slot": s,
                        "date": ts,
                        "optimised_value": float(usage_opt[s]),
                    })

            day_moved_sum += moved_kwh
            n_ca_solved += 1

        day_wall_s = time.perf_counter() - day_t0
        m_rows.append({
            "row_type": "day_summary",
            "ca_id": None,
            "city": city,
            "day": day_ts,
            "n_customers_solved": n_ca_solved,
            "day_wall_s": float(day_wall_s),
            "moved_kwh_day_sum": float(day_moved_sum),
            })
    end_time = time.perf_counter()
    print(f"\tsingle worker solve time: {end_time - start_time:.2f} seconds")

    return m_rows, mv_rows, o_rows


# Pipeline Orchestration
# ────────────────────────────────────────────────────────────────────────────


def run_pipeline_pandas_cityweek_budget(
        df_pd: pd.DataFrame,
        policy: ShiftPolicy,
        solver: SolverConfig,
        *,
        shuffle_high_usage_order: bool = False,   # kept for API compatibility
        emit_optimised_rows: bool = True,
        workers: Optional[int] = None,
        show_progress: bool = True,
        cityday_ca_order: Optional[Dict[Tuple[str, pd.Timestamp],
                                        List[str]]] = None,
        backend: Literal["local", "mpi"] = "mpi",
        # external MPI executor (e.g., MPICommExecutor from the runner)
        executor=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate the end-to-end solve over (city, week) groups.

    Steps:
      1) Type-optimize (categoricals), add 'week_start', compute group sizes.
      2) Order groups (LPT-like) and distribute by backend.
      3) For each (city, week) group, call `_solve_cityweek_worker`.
      4) Aggregate rows into metrics/moves/optimised DataFrames.

    Parameters
    ----------
    df_pd : pandas.DataFrame
        Input rows with at least: city, ca_id, date/day/slot, value, factors.
    policy : ShiftPolicy
        All behavioral and cap constraints.
    solver : SolverConfig
        Greedy / LP / Continuous configuration.
    shuffle_high_usage_order : bool, optional
        Kept for API compatibility (unused).
    emit_optimised_rows : bool, optional
        If True, emit per-slot outputs. Default True.
    workers : int or None, optional
        Local worker count when backend="local".
    show_progress : bool, optional
        If True, print coarse progress per rank/driver.
    cityday_ca_order : dict or None, optional
        Optional precomputed solve order per (city, day).
    backend : {"local","mpi"}, optional
        Execution mode. Default "mpi" (falls back to local if size==1).
    executor : Any, optional
        External MPI executor; when provided, disables static slicing.

    Returns
    -------
    metrics : pandas.DataFrame
        Per-customer and per-day metrics.
    moves : pandas.DataFrame
        Per-move detail rows.
    optimised : pandas.DataFrame
        Per-slot optimised usage (if requested).

    Notes
    -----
    - Prints a brief banner per rank (MPI) or per process (local).
    - Does not ship large data structures when static-slicing under MPI.
    """

    # ── Early exit ───────────────────────────────────────────────────────────
    if df_pd is None or len(df_pd) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # memory/perf: categories
    df_pd["city"] = df_pd["city"].astype("category")
    df_pd["ca_id"] = df_pd["ca_id"].astype("category")

    # derive week_start once
    df = add_week_start_col(df_pd,
                            policy.behavioral.week_boundaries)

    # size per (city, week_start) for LPT ordering
    sizes = (
        df.groupby(["city", "week_start"], sort=False, observed=False)
        .size()
        .rename("cw_rows")
        .reset_index()
    )
    df = df.merge(sizes, on=["city", "week_start"], how="left",
                  copy=False)

    # Sort so first appearance of each group is by descending size
    df = df.sort_values(
        ["cw_rows", "city", "week_start", "day", "ca_id", "slot"],
        ascending=[False, True, True, True, True, True]
    )

    # Important: keep groupby in observed order (sort=False)
    gb = df.groupby(["city", "week_start"],
                    sort=False,
                    group_keys=False,
                    observed=False)
    indices = gb.indices  # dict[(city, week_start)] -> ndarray of row indices
    keys = list(indices.keys())
    total = len(keys)
    if total == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rank = 0
    total_local = total
    t0 = time.perf_counter()

    # hoist constants (needed by the job iterator)
    slot_len = policy.behavioral.slot_length_minutes
    W_slots = hours_to_slots(policy.behavioral.shift_hours_window,
                             slot_len)

    # External executor? If so, DO NOT static-slice by rank.
    use_external_exec = (backend == "mpi" and executor is not None)

    # If user asked for mpi but world size == 1, fall back to local
    if backend == "mpi" and not use_external_exec:
        try:
            from mpi4py import MPI
            world_size = MPI.COMM_WORLD.Get_size()
        except Exception:
            world_size = 1
        if world_size <= 1:
            print("[pipeline] MPI size=1 detected → falling back to local.",
                  flush=True)
            backend = "local"

    # ── Accumulators ────────────────────────────────────────────────────────
    metrics_rows: List[Dict[str, Any]] = []
    move_rows:    List[Dict[str, Any]] = []
    opt_rows:     List[Dict[str, Any]] = []

    def job_iter():
        """ Job iterator for external MPI executor. """
        for (city, wk) in keys:
            df_cw = df.iloc[indices[(city, wk)]]
            yield (city, wk, df_cw, policy, solver,
                   slot_len, W_slots, None, emit_optimised_rows)

    def job_iter_local():
        """ Job iterator for local rank. """
        for (city, wk) in my_keys:
            df_cw = df.iloc[indices[(city, wk)]]
            yield (city, wk, df_cw, policy, solver,
                   slot_len, W_slots, None, emit_optimised_rows)

    # ── Execution strategy ───────────────────────────────────────────────────
    if backend == "mpi" and not use_external_exec:
        # Static per-rank slicing — no MPIPool, no data shipping.
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        my_keys = keys[rank::size]  # LPT order preserved per stride
        total_local = len(my_keys)

        t0 = time.perf_counter()
        print(f"[r{rank}] groups_total={total} local={total_local}",
              flush=True)

        if total_local == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Optional: local multiprocessing per rank
        # (default 1 to avoid oversubscription)
        workers_eff = workers if (workers and workers > 0) else 1
        workers_eff = max(1, min(workers_eff, total_local))
        print(f"[pipeline] rank={rank} local_workers={workers_eff}",
              flush=True)

        def _maybe_progress_local(i: int) -> None:
            """Show progress for local rank."""
            if not show_progress or total_local <= 0:
                return
            if show_progress:
                if i == total_local:
                    print(f"[r{rank}] progress {i}/{total_local} (100%)",
                          flush=True)

        if workers_eff > 1:
            iterator = _iter_results_local(job_iter_local(), workers_eff)
            for i, (m, mv, o) in enumerate(iterator, 1):
                metrics_rows.extend(m)
                move_rows.extend(mv)
                opt_rows.extend(o)
                _maybe_progress_local(i)
        else:
            for i, args in enumerate(job_iter_local(), 1):
                m, mv, o = _solve_cityweek_worker(args)
                metrics_rows.extend(m)
                move_rows.extend(mv)
                opt_rows.extend(o)
                _maybe_progress_local(i)

    elif use_external_exec:
        # External MPI executor (root rank only) – ships tasks to workers.
        print(f"[pipeline] groups={total} backend=mpi executor=external",
              flush=True)
        it = executor.map(_solve_cityweek_worker, job_iter(), chunksize=1)
        for i, (m, mv, o) in enumerate(it, 1):
            metrics_rows.extend(m)
            move_rows.extend(mv)
            opt_rows.extend(o)
            if show_progress:
                if i == total_local:
                    print(f"[r{rank}] progress {i}/{total_local} (100%)",
                          flush=True)
    else:
        # Local backend: optional multiprocessing; otherwise serial.
        workers_eff = workers if (workers and workers > 0) else max(
                                    1, (os.cpu_count() or 1))
        workers_eff = max(1, min(workers_eff, total))
        print(f"[pipeline] backend=local workers={workers_eff}",
              flush=True)

        if workers_eff > 1:
            iterator = _iter_results_local(job_iter(), workers_eff)
            for i, (m, mv, o) in enumerate(iterator, 1):
                metrics_rows.extend(m)
                move_rows.extend(mv)
                opt_rows.extend(o)
                if show_progress:
                    if i == total_local:
                        print(f"[r{rank}] progress {i}/{total_local} (100%)",
                              flush=True)
        else:
            for i, args in enumerate(job_iter(), 1):
                m, mv, o = _solve_cityweek_worker(args)
                metrics_rows.extend(m)
                move_rows.extend(mv)
                opt_rows.extend(o)
                if show_progress:
                    if i == total_local:
                        print(f"[r{rank}] progress {i}/{total_local} (100%)",
                              flush=True)

    print(f"[r{rank}] done in {time.perf_counter() - t0:.1f}s",
          flush=True)
    # ── Materialize outputs ─────────────────────────────────────────────────
    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(move_rows),
        pd.DataFrame(opt_rows),
    )


# ────────────────────────────────────────────────────────────────────────────
# ARCHIVE - UNUSED BUT KEPT FOR REFERENCE:
# ────────────────────────────────────────────────────────────────────────────


def compute_city_day_percentiles(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute energy-use percentiles for each customer within (city, day).

    Parameters
    ----------
    df : polars.DataFrame
        Must contain 'city', 'day', 'ca_id', 'value'.

    Returns
    -------
    polars.DataFrame
        One row per (city, day, ca_id) with:
        - day_kwh : total kWh for that customer/day
        - pct     : percentile in [0,1] within that city-day
        (higher = heavier user)
    """
    # 1) daily kWh per (city, day, ca_id)
    daily = (
        df
        .group_by(["city", "day", "ca_id"], maintain_order=False)
        .agg(pl.col("value").sum().alias("day_kwh"))
    )

    # 2) percentile within (city, day); highest day_kwh gets pct close to 1.0
    # Polars rank() starts at 1. We want a fractional percentile in [0,1].
    # Use "average" to mirror pandas' method="average".
    daily = daily.with_columns([
        (
            pl.col("day_kwh")
            .rank(method="average", descending=True)
            .over(["city", "day"])
            / pl.len().over(["city", "day"])
        ).alias("pct")
    ])

    return daily


def rank_customers_by_daily_kwh(
        df_city_day: pd.DataFrame
) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
    """
    Rank customers by daily kWh within a city-day and compute percentiles.

    Parameters:
    ----------
    df_city_day : pandas.DataFrame
        DataFrame containing daily kWh usage for each customer in a city.

    Returns:
    --------
    order : List[str]
        ca_id ordered by descending total daily kWh.
    pct : Dict[str, float]
        ca_id -> percentile rank (0..100).
    totals : Dict[str, float]
        ca_id -> total kWh that day.
    """
    totals = df_city_day.groupby("ca_id",
                                 as_index=False,
                                 observed=False)["value"].sum()
    totals = totals.sort_values("value", ascending=False)
    # Percentiles within this city-day
    totals["pct"] = 100.0 * totals["value"].rank(pct=True, method="average")
    order = totals["ca_id"].tolist()
    pct = dict(zip(totals["ca_id"], totals["pct"]))
    tot = dict(zip(totals["ca_id"], totals["value"]))
    return order, pct, tot
