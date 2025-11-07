# ─────────────────────────────────────────────────────────────────────────────
# FILE: step5_run_optimisation_full_pygam_cal.py
#
# PURPOSE:
# - MPI-friendly runner that consumes pre-split city–week Parquet shards
#   (see step5_meter_emissions_full_city_week_partitioning.py).
# - Processes shards sequentially per rank (static stride assignment).
# - Accumulates results per rank and writes a single Parquet per kind
#   (metrics / moves / optimised), compatible with your consolidator.
#
# USAGE:
#   mpiexec -n 3 python step5_run_optimisation_full_pygam_cal.py \
#       --solver highs --policy 2
#
# NOTES:
# - Keeps BLAS single-threaded; set HIGHS_THREADS via env.
# - If you precomputed floors, place a 'floor_kwh' column in the shard files
#   and pass --use-precomputed-floors to skip recomputing per shard.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import argparse
import re
import sys
import time
import glob
from datetime import datetime
import polars as pl
import pandas as pd
from mpi4py import MPI
from pathlib import Path

import step5_optimisation_module as opt

# --------- env knobs (safe defaults) ---------
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HIGHS_THREADS = int(os.getenv("HIGHS_THREADS", os.getenv("OMP_NUM_THREADS",
                                                         "1")))
WRITE_MOVES = os.getenv("WRITE_MOVES", "1") == "1"
# default write moves
EMIT_OPTIMISED = True
# default skip optimised rows
GROUP_TARGET_MB = int(os.getenv("GROUP_TARGET_MB", "256"))
# micro-batch target
FLUSH_EVERY = int(os.getenv("FLUSH_EVERY", "20"))
# flush cadence

# keep polars single-threaded
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# FUNCTIONS
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


def _abs_join(root: str, maybe_rel: str) -> str:
    """Join to root if path is relative; return as-is if already absolute."""
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(root,
                                                                   maybe_rel)


def add_week_hod_floor(df_pl: pl.DataFrame,
                       robust_p: float = 95.0,
                       R_percent: float = 10.0,
                       eps: float = 0.001) -> pl.DataFrame:
    """
    Compute the floor kWh for each hour of day (HOD) based on the
    robust statistics.

    Parameters:
    ----------
    df_pl : pl.DataFrame
        Input DataFrame containing the data to process.
    robust_p : float, optional
        Percentile for robust statistics (default is 95.0).
    R_percent : float, optional
        Percent for the robust max kWh (default is 10.0).
    eps : float, optional
        Small value to ensure non-negativity (default is 0.001).

    Returns:
    -------
    pl.DataFrame
        DataFrame with the added floor kWh values.
    """

    # Add join keys (types aligned with the precomputed lookup)
    w = df_pl.with_columns([
        pl.col("date").dt.hour().cast(pl.Int32).alias("hod"),
        pl.col("date").dt.weekday().cast(pl.Int32).alias("dow"),
        # 0=Mon..6=Sun
    ])

    stats = (
        w.group_by(["ca_id", "hod", "dow"])
         .agg([
             pl.mean("value").alias("baseline_kwh"),
             pl.col("value").quantile(robust_p / 100.0,
                                      interpolation="nearest")
               .alias("robust_max_kwh"),
         ]).with_columns([
             pl.col("baseline_kwh").fill_null(0.0),
             pl.col("robust_max_kwh").fill_null(0.0),
             pl.max_horizontal([
                 pl.col("baseline_kwh"),
                 pl.col("robust_max_kwh") * (R_percent / 100.0),
             ]).alias("floor_kwh_raw"),
         ]).with_columns(
             pl.max_horizontal([pl.col("floor_kwh_raw"), pl.lit(eps)])
               .alias("floor_kwh")
         ).select(["ca_id", "hod", "dow", "floor_kwh"])
    )

    # Join floor back, then tidy temporary columns
    return (
        w.join(stats, on=["ca_id", "hod", "dow"], how="left")
         .drop(["hod", "dow"])
    )


def assign_shards_balanced(
        paths: list[str],
        size: int,
        rank: int
) -> list[str]:
    """
    Assign input shards to MPI ranks in a balanced manner.

    Parameters:
    ----------
    paths : list[str]
        List of input shard paths.
    size : int
        Total number of MPI ranks.
    rank : int
        Rank of the current MPI process.

    Returns:
    -------
    list[str]
        List of assigned shard paths for the current rank.
    """
    try:
        w = [(p, os.path.getsize(p)) for p in paths]
        w.sort(key=lambda x: x[1], reverse=True)
        loads = [0] * size
        buckets = [[] for _ in range(size)]
        for p, s in w:
            i = min(range(size), key=lambda k: loads[k])
            loads[i] += s
            buckets[i].append(p)
        return buckets[rank]
    except Exception:
        return paths[rank::size]


def attach_floor_if_available(
        df_pl: pl.DataFrame,
        floor_lf: pl.LazyFrame | None
) -> pl.DataFrame:
    """
    Left-join precomputed floor_kwh onto a shard by (ca_id, hod, dow).
    - df_pl: eager Polars DataFrame with columns at least: ca_id, date, value,
    - floor_lf: LazyFrame from the floor lookup parquet, or None to skip.
    Returns an eager Polars DataFrame (df_pl if no join needed/possible).

    Parameters:
    ----------
    df_pl : pl.DataFrame
        Eager Polars DataFrame with columns at least: ca_id, date, value, ...
    floor_lf : pl.LazyFrame | None
        LazyFrame from the floor lookup parquet, or None to skip.

    Returns:
    -------
    pl.DataFrame
        Eager Polars DataFrame with floor_kwh column added if available.
    """
    if floor_lf is None or "floor_kwh" in df_pl.columns:
        return df_pl

    return (
        df_pl.lazy()
        .with_columns([
            pl.col("date").dt.hour().cast(pl.Int32).alias("hod"),
            pl.col("date").dt.weekday().cast(pl.Int32).alias("dow"),
        ])
        .join(floor_lf, on=["ca_id", "hod", "dow"], how="left")
        .drop(["hod", "dow"])
        .collect()
    )


def build_solver(name: str) -> tuple[opt.SolverConfig, str]:
    """
    Build the optimization solver based on the given name.

    Parameters:
    ----------
    name : str
        Name of the solver to build.

    Returns:
    -------
    tuple[opt.SolverConfig,str]
        The constructed solver and its name.
    """
    if name == "greedy":
        return greedy_solver, "greedy"
    if name == "cont_slsqp":
        return continuous_solver_slsqp, "cont_slsqp"
    if name == "cont_trust":
        return continuous_solver_trust_constr, "cont_trust"
    if name == "cont_closed_form":
        return continuous_solver_closed_form, "cont_closed_form"
    if name == "lp_highs":
        return lp_solver_highs, "lp_highs"
    if name == "lp_glpk":
        return lp_solver_glpk, "lp_glpk"
    if name == "lp_ecos":
        return lp_solver_ecos, "lp_ecos"
    if name == "lp_scs":
        return lp_solver_scs, "lp_scs"
    if name == "lp_clar":
        return lp_solver_clarabel, "lp_clar"
    if name == "lp_osqp":
        return lp_solver_osqp, "lp_osqp"
    raise RuntimeError("unreachable")


def build_policy(name: str) -> tuple[opt.ShiftPolicy, str]:
    """
    Build the shift policy based on the given name.

    Parameters:
    ----------
    name : str
        Name of the policy to build.

    Returns:
    -------
    tuple[opt.ShiftPolicy,str]
        The constructed policy and its name.
    """
    if name == "policy_1":
        return policy_1, "policy_1"
    if name == "policy_2":
        return policy_2, "policy_2"
    raise RuntimeError("unreachable")


def _is_valid_parquet(path: str) -> bool:
    """Quick header/footer check to skip partial/corrupt shards."""
    try:
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, 2)
            tail = f.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except Exception:
        return False


def _union_scan(paths: list[str]) -> pl.LazyFrame:
    """Read many shards with schema union (missing cols → null)."""
    lfs = []
    for p in paths:
        try:
            lf = pl.scan_parquet(
                p,
                # if your Polars has it; otherwise diagonal concat handles union anyway
                allow_missing_columns=True,
                # extra_columns="ignore",  # uncomment if available in your version
                retries=2,
            )
            lfs.append(lf)
        except Exception:
            # skip unreadable shard
            pass
    if not lfs:
        return pl.LazyFrame()  # empty
    return pl.concat(lfs, how="diagonal")


def _sink_parquet_atomic(lf: pl.LazyFrame, out_path: str) -> None:
    """Write atomically (tmp → fsync → rename) to avoid partial outputs."""
    tmp = out_path + f".tmp.{os.getpid()}"
    lf.sink_parquet(tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, out_path)


def consolidate_latest_runs(
        out_dir: str,
        kinds=("metrics", "moves", "optimised"),
        delete_shards: bool = False,
        only_ts: str | None = None,
        consolidate_all_ts: bool = True,
        verbose: bool = True,
) -> None:
    """
    Consolidate per-rank shard parquet files into single files.

    Parameters:
    ----------
    out_dir : str
        Output directory containing the shard files.
    kinds : tuple[str, ...], optional
        Types of shard files to consolidate
        (default: ("metrics", "moves", "optimised")).
    delete_shards : bool, optional
        Whether to delete the original shard files after consolidation
        (default: False).
    only_ts : str | None, optional
        If provided, only this timestamp will be consolidated
        (default: None).
        If only_ts is provided: consolidates that timestamp per (case, kind).
    consolidate_all_ts : bool, optional
        Whether to consolidate all timestamps or just the latest one
        (default: True).
        If consolidate_all_ts=True (default), consolidates *every* timestamp
        present in the folder for each (case, kind).
        If consolidate_all_ts=False and only_ts is None: consolidates only the
        latest timestamp per (case, kind).
    verbose : bool, optional
        Whether to print verbose output (default: True).

    Returns:
    -------
    None
        Removes the original shard files after consolidation.
    """
    try:
        rx = re.compile(
            r"^(?P<case>.+)_(?P<kind>metrics|moves|optimised)"
            r"(?:_(?P<tag>[A-Za-z0-9\-]+))?"       # <- optional cityweek tag
            r"(?:_r(?P<rank>\d{4})(?:\.part(?P<part>\d{3}))?)?"
            r"_(?P<ts>\d{8}_\d{4,6})\.parquet$"
        )
        all_files = [os.path.basename(p)
                     for p in glob.glob(os.path.join(out_dir, "*.parquet"))]
        parsed = []
        for f in all_files:
            m = rx.match(f)
            if m and m.group("kind") in kinds:
                d = m.groupdict()
                parsed.append((d["case"], d["kind"], d["ts"], f))

        if not parsed:
            if verbose:
                print("[consolidate] no matching shard files found.")
            return

        # group → (case, kind) -> ts -> [files]
        by_ck_ts: dict[tuple[str, str], dict[str, list[str]]] = {}
        for case, kind, ts, f in parsed:
            by_ck_ts.setdefault((case, kind), {}
                                ).setdefault(ts, []).append(f)

        for (case, kind), ts_map in sorted(by_ck_ts.items()):
            ts_list = sorted(ts_map)  # chronological
            if not consolidate_all_ts:
                # choose exactly one ts
                target_ts = only_ts if (only_ts is not None) else ts_list[-1]
                ts_iter = [target_ts] if target_ts in ts_map else []
            else:
                # do all timestamps found
                ts_iter = (
                    ts_list if only_ts is None else (
                        [only_ts] if only_ts in ts_map else [])
                )

            for ts in ts_iter:
                shard_files = ts_map.get(ts, [])
                if not shard_files:
                    continue

                out_name = f"{case}_{kind}_{ts}.parquet"
                out_path = os.path.join(out_dir, out_name)

                # If already consolidated (single file with final name), skip
                if len(shard_files) == 1 and shard_files[0] == out_name:
                    if verbose:
                        print(f"[consolidate] {case}/{kind} @{ts} "
                              f"already consolidated; skip.")
                    continue

                paths = [os.path.join(out_dir, f) for f in shard_files]
                if verbose:
                    print(f"[consolidate] {case}/{kind} @{ts}: {len(paths)} "
                          f"shard(s) → {out_name}")

                try:
                    # keep only files that exist
                    existing = [os.path.join(out_dir, f) for f in shard_files]
                    existing = [p for p in existing if os.path.exists(p)]

                    # drop corrupt/partial shards (bad footer)
                    bad = [p for p in existing if not _is_valid_parquet(p)]
                    if bad and verbose:
                        print(f"  ! skipping {len(bad)} corrupt shard(s):")
                        for b in bad[:10]:
                            print(f"    - {os.path.basename(b)}")

                    good = [p for p in existing if p not in bad]
                    if not good:
                        if verbose:
                            print("  ! no valid shards to consolidate; skip.")
                        continue

                    # union schemas across shards, then write atomically
                    lf = _union_scan(good)
                    _sink_parquet_atomic(lf, out_path)
                except Exception as e:
                    print(f"  ! failed to write {out_name}: {e}")
                    continue

                if verbose:
                    print(f"  → wrote {out_path}")

                if delete_shards:
                    for p in paths:
                        if os.path.basename(p) != out_name:
                            try:
                                os.remove(p)
                            except Exception as e:
                                print(f"  ! could not delete {p}: {e}")
    except Exception as e:
        # absolutely never fail caller
        print(f"[consolidate] unexpected error: {e}")


def discover_shards(
        shards_dir: str,
) -> list[str]:
    """
    Discover shard files in the specified directory.

    Parameters:
    -----------
    shards_dir : str
        Directory to search for shard files.

    Returns:
    --------
    list[str]
        List of discovered shard file paths.
    """
    p = Path(shards_dir).resolve()
    if not p.is_dir():
        print(f"[runner] shards_dir does not exist: {p}", flush=True)
        return []
    shards = sorted(str(x) for x in p.glob("*.parquet"))
    # or p.rglob for subdirs
    print(f"[runner] scanning {p} → {len(shards)} file(s)", flush=True)
    return shards


def group_small(
        paths,
        target_bytes=256*1024*1024
) -> list[list[str]]:
    """
    Group small files into larger batches.

    Parameters:
    -----------
    paths : list[str]
        List of input file paths.
    target_bytes : int
        Target size in bytes for each batch.

    Returns:
    --------
    list[list[str]]
        List of batches, where each batch is a list of file paths.
    """
    groups, cur, acc = [], [], 0
    for p in sorted(paths, key=os.path.getsize):
        sz = os.path.getsize(p)
        if acc + sz > target_bytes and cur:
            groups.append(cur)
            cur, acc = [], 0
        cur.append(p)
        acc += sz
    if cur:
        groups.append(cur)
    return groups


def load_shard_pl(path: str) -> pl.DataFrame:
    """
    Load a shard from the specified path as a Polars DataFrame.

    Parameters:
    -----------
    path : str
        Path to the shard file.

    Returns:
    --------
    pl.DataFrame
        Loaded Polars DataFrame.
    """
    # Shards already have day/slot/week_start; just make sure names
    # are canonical.
    df = pl.read_parquet(path)
    rename_candidates = {
        "marginal_emissions_grams_co2_per_kWh":
        "marginal_emissions_factor_grams_co2_per_kWh",
        "average_emissions_grams_co2_per_kWh":
        "average_emissions_factor_grams_co2_per_kWh",
        "average_emissions_factor_co2_per_kWh":
        "average_emissions_factor_grams_co2_per_kWh",
    }
    present = [k for k in rename_candidates if k in df.columns]
    if present:
        df = df.rename({k: rename_candidates[k] for k in present})
    # Ensure required columns exist
    req = {
        "ca_id", "city", "date", "day", "slot",
        "value",
        "marginal_emissions_factor_grams_co2_per_kWh",
        "average_emissions_factor_grams_co2_per_kWh",
    }
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"[load] {os.path.basename(path)} "
                         f"missing columns: {sorted(missing)}")
    # sort stable for reproducibility
    return df.sort(["ca_id", "day", "slot"])


def normalize_solver(s: str) -> str:
    """
    Normalize the solver string to a canonical form.

    s : str
        Solver string to normalize.

    Returns:
    --------
    str
        Normalized solver string.
    """
    s = s.strip().lower()
    mapping = {
        # Greedy
        "g": "greedy", "greedy": "greedy", "greed": "greedy", "grdy": "greedy",

        # LP: highs
        "highs": "lp_highs", "lp_highs": "lp_highs", "h": "lp_highs",
        "lph": "lp_highs", "lphighs": "lp_highs",

        # LP: glpk
        "glpk": "lp_glpk", "lp_glpk": "lp_glpk", "lpg": "lp_glpk",
        "lpgl": "lp_glpk", "lpglpk": "lp_glpk",

        # LP: ecos
        "ecos": "lp_ecos", "lp_ecos": "lp_ecos", "lpe": "lp_ecos",
        "lpecos": "lp_ecos",
        # LP: scs
        "scs": "lp_scs", "lp_scs": "lp_scs", "lpscs": "lp_scs",
        "lps": "lp_scs",

        # LP: clarabel
        "clarabel": "lp_clar", "clar": "lp_clar", "lp_clarabel": "lp_clar",
        "lp_clar": "lp_clar", "lpclar": "lp_clar", "lpc": "lp_clar",

        # LP: osqp
        "osqp": "lp_osqp", "lp_osqp": "lp_osqp", "lpos": "lp_osqp",
        "lposqp": "lp_osqp",

        # Continuous: SLSQP
        "slsqp": "cont_slsqp", "cont_slsqp": "cont_slsqp",
        "continuous_slsqp": "cont_slsqp", "contsls": "cont_slsqp",
        "sls": "cont_slsqp", "contslsqp": "cont_slsqp",
        # Continuous: trust
        "trust": "cont_trust", "trust-constr": "cont_trust",
        "cont_trust": "cont_trust", "contrust": "cont_trust",
        "conttrust": "cont_trust", "ctru": "cont_trust",
        # Continuous: closed-form
        "closed-form": "cont_closed_form", "ccf": "cont_closed_form",
        "contclosedform": "cont_closed_form", "ctcf": "cont_closed_form",
        "cont_closed_form": "cont_closed_form", "contcf": "cont_closed_form",
        "closeform": "cont_closed_form",
    }
    if s not in mapping:
        raise argparse.ArgumentTypeError(f"unknown solver '{s}'")
    return mapping[s]


def normalize_policy(p: str) -> str:
    """
    Normalize the policy string to a canonical form.

    p : str
        Policy string to normalize.

    Returns:
    --------
    str
        Normalized policy string.
    """
    p = p.strip().lower()
    if p in {"1", "p1", "policy1", "policy_1"}:
        return "policy_1"
    if p in {"2", "p2", "policy2", "policy_2"}:
        return "policy_2"
    raise argparse.ArgumentTypeError(f"unknown policy '{p}'")


def run_shard(
        df_pl: pl.DataFrame,
        policy: opt.ShiftPolicy,
        solver: opt.SolverConfig,
        use_precomputed_floors: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the optimisation pipeline on a single shard.

    Parameters:
    -----------
    df_pl : pl.DataFrame
        Polars DataFrame containing the shard data.
    policy : opt.ShiftPolicy
        Shift policy configuration.
    solver : opt.SolverConfig
        Solver configuration.
    use_precomputed_floors : bool
        Flag indicating whether to use precomputed floors.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing the metrics, moves, and optimised DataFrames.
    """
    # If no precomputed floor, compute per the policy
    if (policy.household_min is not None) and (not use_precomputed_floors):
        df_pl = add_week_hod_floor(
            df_pl,
            robust_p=(
                policy.household_min.household_minimum_robust_max_percentile),
            R_percent=policy.household_min.household_minimum_R_percent,
            eps=policy.household_min.household_minimum_epsilon_floor_kWh,
        )

    cols = [
        "ca_id", "city", "date", "day", "slot", "value",
        "marginal_emissions_factor_grams_co2_per_kWh",
        "average_emissions_factor_grams_co2_per_kWh",
    ]
    if "floor_kwh" in df_pl.columns:
        cols.append("floor_kwh")

    df_pl = df_pl.select(cols)
    order_map = opt.build_cityday_ca_order_map_pl(df_pl)

    df_pd = df_pl.to_pandas()
    df_pd["city"] = df_pd["city"].astype("category")
    df_pd["ca_id"] = df_pd["ca_id"].astype("category")

    # process locally (each shard is one (city,week) group)
    metrics, moves, opt_df = opt.run_pipeline_pandas_cityweek_budget(
        df_pd=df_pd,
        policy=policy,
        solver=solver,
        emit_optimised_rows=EMIT_OPTIMISED,    # <— obey env flag
        show_progress=False,
        cityday_ca_order=order_map,            # <— pass order map
        backend="local",
        executor=None,
    )
    return metrics, moves, opt_df


def shard_tag(path: str, pattern: re.Pattern) -> str | None:
    """
    Extract the shard tag from the given path using the provided regex pattern.

    Parameters:
    -----------
    path : str
        The file path to extract the shard tag from.
    pattern : re.Pattern
        The regex pattern to use for extraction.

    Returns:
    --------
    str | None
        The extracted shard tag, or None if not found.
    """
    m = pattern.search(os.path.basename(path))
    if not m:
        return None
    city = m.group("city").lower()
    week = m.group("week")
    # if you prefer "delhi2021-11-28" drop the underscore:
    return f"{city}_{week}"


def short_solver(s: opt.SolverConfig) -> str:
    """
    Shorten the representation of the solver configuration.

    Parameters:
    -----------
    s : SolverConfig
        Solver configuration object
        from step5_optimisation_module

    Returns:
    --------
    str
    Shortened representation of the solver configuration.
    """
    fam = s.solver_family
    if fam == "lp":
        low = s.lp_solver or "auto"
        th = (s.lp_solver_opts or {}).get("threads")
        th_txt = f"(threads={th})" if th is not None else ""
        return f"lp/{low}{th_txt}"
    elif fam == "continuous":
        meth = s.continuous_method or "SLSQP"
        return f"continuous/{meth}"
    elif fam == "greedy":
        return "greedy"
    else:
        return str(fam)


def short_policy(p):
    """
    Shorten the representation of the policy configuration.

    Parameters:
    -----------
    p : PolicyConfig
        Policy configuration object
        from step5_optimisation_module

    Returns:
    --------
    str
        Shortened representation of the policy configuration.
    """
    b = p.behavioral
    ph = b.peak_hours_reduction_limit_config
    ph_txt = (f"{ph.peak_hours_reduction_percent_limit}%/"
              f"{ph.peak_hours_reduction_scope}" if ph else "none")
    return (f"moves/day={b.customer_power_moves_per_day}, "
            f"moves/week={b.customer_power_moves_per_week}, "
            f"window={b.shift_hours_window}h, slots={b.slot_length_minutes}m, "
            f"peak={ph_txt}")


def write_and_clear(segment: int | None = None):
    """
    Write accumulated metrics, moves, and optimised DataFrames to Parquet files

    Parameters:
    -----------
    segment : int | None
        Segment number for the output files.

    Returns:
    --------
    None

    """
    suffix = "" if segment is None else f".part{segment:03d}"

    # choose a tag for this flush window
    if tags_acc:
        tag_label = next(iter(tags_acc)) if len(tags_acc) == 1 else "multi"
    else:
        tag_label = "multi"

    # compact tag: turn "delhi_2021-11-28" into "delhi2021-11-28"
    nice_tag = tag_label.replace("_", "")

    shard = f"_r{rank:04d}{suffix}"
    base = os.path.join(save_dir, case_name)

    m = pd.concat(metrics_acc,
                  ignore_index=True) if metrics_acc else pd.DataFrame()
    mv = pd.concat(moves_acc,
                   ignore_index=True) if moves_acc else pd.DataFrame()
    o = pd.concat(optimised_acc,
                  ignore_index=True) if optimised_acc else pd.DataFrame()

    # metrics
    opt.write_parquet_fsync(
        m,  f"{base}_metrics_{nice_tag}{shard}_{run_tag}.parquet"
    )
    # moves (only write if you actually accumulated any
    # OR you want the file regardless)
    if WRITE_MOVES or not mv.empty:
        opt.write_parquet_fsync(
            mv, f"{base}_moves_{nice_tag}{shard}_{run_tag}.parquet"
        )
    # optimised (respect EMIT_OPTIMISED, but also write if non-empty)
    if EMIT_OPTIMISED or not o.empty:
        opt.write_parquet_fsync(
            o,  f"{base}_optimised_{nice_tag}{shard}_{run_tag}.parquet"
        )

    metrics_acc.clear()
    moves_acc.clear()
    optimised_acc.clear()
    tags_acc.clear()  # reset tag window for next flush


def write_rank_outputs(
        save_dir: str,
        case_name: str,
        run_tag: str,
        rank: int,
        metrics: list[pd.DataFrame],
        moves: list[pd.DataFrame],
        optimised: list[pd.DataFrame]
) -> None:
    """
    Write the outputs for a specific rank to Parquet files.

    Parameters:
    -----------
    save_dir : str
        Directory to save the output files.
    case_name : str
        Name of the case being processed.
    run_tag : str
        Tag for the current run.
    rank : int
        Rank of the current output.
    metrics : list[pd.DataFrame]
        List of DataFrames containing metrics data.
    moves : list[pd.DataFrame]
        List of DataFrames containing moves data.
    optimised : list[pd.DataFrame]
        List of DataFrames containing optimised data.

    Returns:
    --------
    None
        Writes the output DataFrames to Parquet files.
    """
    prefix = os.path.join(save_dir, case_name)
    shard = f"_r{rank:04d}"
    # concat safely even if lists are empty
    m = pd.concat(metrics,
                  ignore_index=True) if metrics else pd.DataFrame()
    mv = pd.concat(moves,
                   ignore_index=True) if moves else pd.DataFrame()
    o = pd.concat(optimised,
                  ignore_index=True) if optimised else pd.DataFrame()
    opt.write_parquet_fsync(m,
                            f"{prefix}_metrics{shard}_{run_tag}.parquet")
    opt.write_parquet_fsync(mv,
                            f"{prefix}_moves{shard}_{run_tag}.parquet")
    opt.write_parquet_fsync(o,
                            f"{prefix}_optimised{shard}_{run_tag}.parquet")


# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# DEFINING FILEPATHS AND DIRECTORIES
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────

# Resolve repo root → .../code_and_analysis (one level up from scripts/)
try:
    base_directory = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
except NameError:
    # Fallback for interactive runs
    base_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))


base_data_directory = _abs_join(base_directory, "data")
hitachi_data_directory = os.path.join(base_data_directory, "hitachi_copy")
optimisation_development_directory = os.path.join(
    base_data_directory,
    "optimisation_development")
processing_files_directory = os.path.join(
    optimisation_development_directory, "processing_files")
full_results_directory = os.path.join(
    optimisation_development_directory,
    "full_results")
city_week_shards_directory = os.path.join(
    optimisation_development_directory,
    "city_week_shards")

os.makedirs(optimisation_development_directory, exist_ok=True)
os.makedirs(full_results_directory, exist_ok=True)
os.makedirs(city_week_shards_directory, exist_ok=True)


# File paths
# ALL meter_readings_all_years_20250714_formatted_with_emission_factors_filled
# TEST meter_readings_all_years_20250714_formatted
# _with_emission_factors_filled_2022-05-04_to_2022-05-18

# # FULL DATASEST RUNS
marginal_emissions_filename = (
    "meter_readings_all_years_20250714_formatted_with_pygam_cal_emission_factors_filled")
marginal_emissions_filepath = os.path.join(
    optimisation_development_directory,
    marginal_emissions_filename + ".parquet")

floor_lookup_filename = (
    "meter_readings_all_years_20250714_household_floor_lookup")
floor_lookup_filepath = os.path.join(
    processing_files_directory,
    floor_lookup_filename + ".parquet")

# extract "city_weekstart" from our shard names
SHARD_TAG_RX = re.compile(
    r"_emissions_(?P<city>[A-Za-z0-9]+)_week_(?P<week>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
    )

tags_acc: set[str] = set()

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# CONFIGURATIONS, STATIC VARIABLES, AND PATH DEFINITIONS
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────

# Save directories
# ────────────────────────────────────────────────────────────────────────────
save_dir = full_results_directory
shards_dir = city_week_shards_directory

# Regional Load
# ────────────────────────────────────────────────────────────────────────────
# Regional Load is calculated using 2023 data sourced from government reports
# https://www.ceicdata.com/en/india/electricity-consumption-utilities/electricity-consumption-utilities-delhi
# India Delhi Annual consumption (2023) 34,107 GWh
# --> Daily consumption (average) = 34,107 GWh / 365 = 93.4 GWh
# --> Average Hourly Consumption = 93.4 GWh / 24 = 3.89 GWh
# --> Average Half-Hourly Consumption = 3.89 GWh / 2 = 1.945 GWh
# ────────────────────────────────────────────────────────────────────────────

delhi_total_daily_average_load_gWh = 93.4  # GWh
maharashtra_total_daily_average_load_gWh = 17.750  # GWh
delhi_total_daily_average_load_kWh = (
    delhi_total_daily_average_load_gWh * 1_000_000)  # convert GWh to kWh
maharashtra_total_daily_average_load_kWh = (
    maharashtra_total_daily_average_load_gWh * 1_000_000)  # convert GWh to kWh


# Peak Hours Dictionary
# ────────────────────────────────────────────────────────────────────────────
# This dictionary defines the peak hours for electricity consumption in each
# city. The hours were determined based on analysis of this dataset done in
# step 2
# ────────────────────────────────────────────────────────────────────────────
peak_hours_dictionary = {"delhi": {
                                    "Mon": [10, 9, 11, 8, 12, 13],
                                    "Tue": [9, 11, 8, 10, 22, 12],
                                    "Wed": [10, 11, 9, 12, 8, 13],
                                    "Thu": [9, 10, 8, 12, 11, 21],
                                    "Fri": [9, 10, 8, 11, 12, 21],
                                    "Sat": [10, 9, 12, 11, 13, 8],
                                    "Sun": [11, 10, 12, 13, 9, 22],
                        },
                        "mumbai": {
                                    "Mon": [9, 10, 11, 8, 20, 21],
                                    "Tue": [10, 9, 21, 20, 22, 11],
                                    "Wed": [10, 9, 11, 21, 8, 22],
                                    "Thu": [10, 9, 21, 20, 11, 22],
                                    "Fri": [10, 9, 11, 21, 20, 22],
                                    "Sat": [11, 10, 12, 13, 9, 20],
                                    "Sun": [11, 12, 13, 10, 14, 20],
                        }
                        }


# Defined Solvers
# ────────────────────────────────────────────────────────────────────────────
# These are all solvers that have been implemented and should be possible to
# run, but only one should be chosen to run on the dataset at a time due
# to the computational cost
# ────────────────────────────────────────────────────────────────────────────
greedy_solver = opt.SolverConfig(solver_family="greedy")
continuous_solver_slsqp = opt.SolverConfig(
        solver_family="continuous", continuous_method="SLSQP",
        continuous_opts={"ftol": 1e-9})
continuous_solver_trust_constr = opt.SolverConfig(
        solver_family="continuous", continuous_method="trust-constr",
        continuous_opts={"gtol": 1e-9, "xtol": 1e-9})
continuous_solver_closed_form = opt.SolverConfig(
        solver_family="continuous", continuous_method="closed-form",
        continuous_opts={"gtol": 1e-9, "xtol": 1e-9})
lp_solver_highs = opt.SolverConfig(
        solver_family="lp", lp_solver="HIGHS",
        lp_solver_opts={"threads": HIGHS_THREADS} if HIGHS_THREADS else None,)
lp_solver_glpk = opt.SolverConfig(
        solver_family="lp", lp_solver="GLPK")
lp_solver_ecos = opt.SolverConfig(
        solver_family="lp", lp_solver="ECOS")
lp_solver_scs = opt.SolverConfig(
        solver_family="lp", lp_solver="SCS")
lp_solver_clarabel = opt.SolverConfig(
        solver_family="lp", lp_solver="CLARABEL")
lp_solver_osqp = opt.SolverConfig(
        solver_family="lp", lp_solver="OSQP")

# Policy Configurations
# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION 1 - Relax
# ────────────────────────────────────────────────────────────────────────────
peak_hours_reduction_limit_config_1 = opt.PeakHoursReductionLimitConfig(
        peak_hours_reduction_percent_limit=80,
        peak_hours_reduction_scope="per_city",
        peak_hours_dict=peak_hours_dictionary,
)

policy_1 = opt.ShiftPolicy(
        behavioral=opt.CustomerAdoptionBehavioralConfig(
            customer_power_moves_per_day=2,
            customer_power_moves_per_week=7,
            timezone="Asia/Kolkata",
            day_boundaries="00:00-24:00",
            week_boundaries="Mon-Sun",
            shift_hours_window=3.0,
            slot_length_minutes=30,
            peak_hours_reduction_limit_config=(
                peak_hours_reduction_limit_config_1)
        ),
        regional_cap=opt.RegionalLoadShiftingLimitConfig(
            regional_load_shift_percent_limit=10.0,
            regional_total_daily_average_load_kWh={
                "delhi": delhi_total_daily_average_load_kWh,
                "mumbai": maharashtra_total_daily_average_load_kWh
            }
        ),
        household_min=opt.HouseholdMinimumConsumptionLimitConfig(
            household_minimum_baseline_period="year",
            household_minimum_baseline_type="average",
            household_minimum_robust_max_percentile=95,
            household_minimum_R_percent=10
        ),
        spike_cap=opt.ShiftWithoutSpikeLimitConfig(alpha_peak_cap_percent=25),
)

# CONFIGURATION 2 - Stricter
# ────────────────────────────────────────────────────────────────────────────

peak_hours_reduction_limit_config_2 = opt.PeakHoursReductionLimitConfig(
        peak_hours_reduction_percent_limit=25.0,
        peak_hours_reduction_scope="per_city",
        peak_hours_dict=peak_hours_dictionary,
)

policy_2 = opt.ShiftPolicy(
        behavioral=opt.CustomerAdoptionBehavioralConfig(
            customer_power_moves_per_day=1,
            customer_power_moves_per_week=3,
            timezone="Asia/Kolkata",
            day_boundaries="00:00-24:00",
            week_boundaries="Mon-Sun",
            shift_hours_window=2.0,
            slot_length_minutes=30,
            peak_hours_reduction_limit_config=(
                peak_hours_reduction_limit_config_2
            )
        ),
        regional_cap=opt.RegionalLoadShiftingLimitConfig(
            regional_load_shift_percent_limit=10.0,
            regional_total_daily_average_load_kWh={
                "delhi": delhi_total_daily_average_load_kWh,
                "mumbai": maharashtra_total_daily_average_load_kWh
            }
        ),
        household_min=opt.HouseholdMinimumConsumptionLimitConfig(
            household_minimum_baseline_period="year",
            household_minimum_baseline_type="average",
            household_minimum_robust_max_percentile=95,
            household_minimum_R_percent=10
        ),
        spike_cap=opt.ShiftWithoutSpikeLimitConfig(alpha_peak_cap_percent=25),
)


# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# IMPLEMENTATION
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Shard runner (only solver+policy required).")
    ap.add_argument("--solver",
                    required=True,
                    type=normalize_solver,
                    help=("greedy | highs | glpk | ecos | scs | "
                          "clarabel | osqp | slsqp | trust | closed-form"))
    ap.add_argument("--policy",
                    required=True,
                    type=normalize_policy,
                    help="1|p1|policy1|policy_1  or  2|p2|policy2|policy_2")
    args = ap.parse_args()

    try:
        floor_lf = pl.scan_parquet(floor_lookup_filepath)
        # ensure join key types match what we compute on the left
        floor_lf = floor_lf.with_columns([
            pl.col("hod").cast(pl.Int32),
            pl.col("dow").cast(pl.Int32),
        ])
        USE_PRECOMPUTED_FLOORS = True
    except Exception:
        floor_lf = None
        USE_PRECOMPUTED_FLOORS = False

    try:
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    except Exception:
        size, rank, comm = 1, 0, None

        # timestamp tag once (rank 0)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if size > 1:
        run_tag = comm.bcast(run_tag, root=0)

    # Choose solver/policy + auto case name
    solver, solver_label = build_solver(args.solver)
    policy, policy_label = build_policy(args.policy)
    case_name = f"{policy_label}_{solver_label}"

    print(f"[shards_dir] = {shards_dir}")
    print(f"[runner] using shards_dir={Path(shards_dir).resolve()}",
          flush=True)
    all_shards = discover_shards(shards_dir)
    if not all_shards:
        if rank == 0:
            print(f"[runner] no shards found in {shards_dir}", flush=True)
        sys.exit(0)

    my_shards = assign_shards_balanced(all_shards, size, rank
                                       ) if size > 1 else all_shards

    if rank == 0:
        print(f"[runner] MPI world size={size} | total_shards="
              f"{len(all_shards)}", flush=True)
    print(f"[runner][r{rank}] assigned {len(my_shards)} shard(s) "
          f"| solver={short_solver(solver)} | policy=({short_policy(policy)})",
          flush=True)

    os.makedirs(save_dir, exist_ok=True)
    time.sleep(0.05 * (rank % 10))

    # micro-batch small shards to ~GROUP_TARGET_MB per batch
    groups = group_small(my_shards, target_bytes=GROUP_TARGET_MB * 1024 * 1024)

    metrics_acc: list[pd.DataFrame] = []
    moves_acc: list[pd.DataFrame] = []
    optimised_acc: list[pd.DataFrame] = []
    segment = 0
    t0 = time.perf_counter()

    for gi, grp in enumerate(groups, 1):
        try:

            # collect tags in this batch
            grp_tags = {
                t for t in (shard_tag(p, SHARD_TAG_RX) for p in grp) if t}
            if grp_tags:
                tags_acc.update(grp_tags)

            # read + concat this batch
            df_pl = pl.concat([load_shard_pl(p) for p in grp], how="vertical",
                              rechunk=True)

            if USE_PRECOMPUTED_FLOORS:
                df_pl = attach_floor_if_available(df_pl, floor_lf)

            # run this batch
            m, mv, o = run_shard(df_pl, policy, solver, USE_PRECOMPUTED_FLOORS)

            # accumulate, with optional trims
            metrics_acc.append(m)
            if WRITE_MOVES:
                moves_acc.append(mv)
            if EMIT_OPTIMISED:
                optimised_acc.append(o)

            if gi % 5 == 0:
                print(f"[runner][r{rank}] {gi}/{len(groups)} batches done...",
                      flush=True)

            # periodic flush
            if gi % FLUSH_EVERY == 0:
                segment += 1
                write_and_clear(segment=segment)

        except Exception as e:
            names = ", ".join(os.path.basename(p) for p in grp)
            print(f"[runner][r{rank}] ERROR batch={gi} ({names}): {e}",
                  flush=True)

    # final flush
    write_and_clear(segment=segment + 1)
    try:
        if size > 1:
            MPI.COMM_WORLD.Barrier()
    finally:
        if rank == 0:
            consolidate_latest_runs(save_dir,
                                    kinds=("metrics", "moves", "optimised"),
                                    delete_shards=False,
                                    only_ts=run_tag,
                                    consolidate_all_ts=False,
                                    verbose=True)
    print(f"[runner][r{rank}] finished in {time.perf_counter()-t0:.1f}s",
          flush=True)
