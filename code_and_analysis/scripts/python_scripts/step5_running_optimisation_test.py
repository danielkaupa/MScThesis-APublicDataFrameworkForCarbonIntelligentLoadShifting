# ─────────────────────────────────────────────────────────────────────────────
# FILE: step5_running_optimisation.py
#
# PURPOSE:
# - Entry point for running optimisation experiments across customer-day
#   usage shards using the optimisation methods defined in
#   step5_optimisation_module.py.
# - Distributes work across MPI ranks on the HPC cluster, ensuring
#   parallel execution of optimisation tasks.
# - Handles coordination of input shards, per-rank result writing,
#   and triggers final consolidation of outputs.
#
# USAGE:
# - This script is designed to be launched on the HPC cluster using mpirun
#   or a batch scheduler job file (e.g., step5_run_optimisation.sh).
# - Typical command (example with 16 processes):
#       mpirun -np 16 python step5_run_optimisation.py
# - The script can also be run locally for testing with a single rank.
#
# RUN REQUIREMENTS:
# - Python 3.9+ with mpi4py installed and an MPI runtime available
#   (e.g., OpenMPI, MPICH).
# - Requires step5_optimisation_module.py to be present in the same
#   directory or accessible on the PYTHONPATH.
# - Input shard Parquet files must be present in the expected
#   input directory structure.
# - The same library requirements as the optimisation module apply
#   (numpy, pandas, polars, scipy, cvxpy, etc.).
#
# OUTPUTS:
# - Per-rank Parquet result files written to the configured output directory.
#   Filenames are suffixed with the rank ID.
# - A single consolidated Parquet file containing results from all ranks,
#   produced after all ranks complete successfully.
# - Standard logging to stdout/stderr with rank IDs for traceability.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

import os
import re
import glob
from mpi4py import MPI
from datetime import datetime
import polars as pl
# import cvxpy as cp

import step5_optimisation_module as opt

# --------- env knobs (safe defaults) ---------
LOCAL_WORKERS = int(os.getenv("LOCAL_WORKERS", "1"))

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# FUNCTIONS
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


def attach_floor_if_available(
        df_pl: pl.DataFrame,
        floor_lf: pl.LazyFrame | None
) -> pl.DataFrame:
    """
    Left-join precomputed floor_kwh onto a shard by (ca_id, hod, dow).
    - df_pl: eager Polars DataFrame with columns at least: ca_id, date, value,
      ...
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


def _abs_join(root: str, maybe_rel: str) -> str:
    """Join to root if path is relative; return as-is if already absolute."""
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(root,
                                                                   maybe_rel)


def consolidate_latest_runs(
        out_dir: str,
        kinds=("metrics", "moves", "optimised"),
        delete_shards: bool = False,
        only_ts: str | None = None,     # now interpreted as run_id if provided
        consolidate_all_ts: bool = True, # now "combine all run_ids"
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
        # policy_<policy>_<solver>_(moves|metrics|optimised)_r<rank>_<YYYYMMDD>_<RUNID>.parquet
        rx = re.compile(
            r"^policy_(?P<policy>\d+)_(?P<solver>.+?)_"
            r"(?P<kind>metrics|moves|optimised)_r(?P<rank>\d{4})_"
            r"(?P<date>\d{8})_(?P<run_id>\d{6})\.parquet$"
        )

        all_files = [os.path.basename(p)
                     for p in glob.glob(os.path.join(out_dir, "*.parquet"))]

        parsed = []
        for f in all_files:
            m = rx.match(f)
            if m and m.group("kind") in kinds:
                d = m.groupdict()
                # policy, solver, kind, run_id, date, rank, filename
                parsed.append((d["policy"], d["solver"], d["kind"],
                               d["run_id"], d["date"], d["rank"], f))

        if not parsed:
            if verbose:
                print("[consolidate] no matching shard files found.")
            return

        # (policy, solver, kind) -> run_id -> {"files":[...],
        # "max_date": "..."}
        by_psk_run: dict[tuple[str, str, str], dict[str, dict]] = {}
        for policy, solver, kind, run_id, date, rank, f in parsed:
            run_map = by_psk_run.setdefault((policy, solver, kind), {})
            entry = run_map.setdefault(run_id, {"files": [],
                                                "max_date": "00000000"})
            entry["files"].append(f)
            if date > entry["max_date"]:
                entry["max_date"] = date

        for (policy, solver, kind), run_map in sorted(by_psk_run.items()):
            # choose which run_id(s) to combine
            if consolidate_all_ts:
                run_ids = (
                    sorted(run_map.keys()) if only_ts is None
                    else ([only_ts] if only_ts in run_map else [])
                )
            else:
                if only_ts is not None:
                    run_ids = [only_ts] if only_ts in run_map else []
                else:
                    # pick the run with the newest date among its shards
                    if run_map:
                        latest_rid = max(run_map.keys(),
                                         key=lambda r: run_map[r]["max_date"])
                        run_ids = [latest_rid]
                    else:
                        run_ids = []

            for rid in run_ids:
                entry = run_map.get(rid)
                if not entry:
                    continue
                shard_files = entry["files"]
                if not shard_files:
                    continue

                # output name: policy_{policy}_{safe_solver}_{kind}_
                # run{run_id}_combined.parquet
                safe_solver = re.sub(r"[^A-Za-z0-9._-]+", "_", solver)
                out_name = (f"policy_{policy}_{safe_solver}_"
                            f"{kind}_run{rid}_combined.parquet")
                out_path = os.path.join(out_dir, out_name)

                if verbose:
                    print(f"[consolidate] policy={policy}, solver={solver},"
                          f" kind={kind}, run_id={rid}: "
                          f"{len(shard_files)} shard(s) → {out_name}")
                    for s in sorted(shard_files):
                        print(f"    - {s}")

                paths = [os.path.join(out_dir, f) for f in shard_files]

                try:
                    # build annotated lazy frames and concat with schema
                    # tolerance
                    lfs = []
                    for p in paths:
                        m = rx.match(os.path.basename(p))
                        md = m.groupdict() if m else {}
                        lf = (
                            pl.scan_parquet(p)
                            .with_columns(
                                pl.lit(md.get("policy")).alias("policy"),
                                pl.lit(md.get("solver")).alias("solver"),
                                pl.lit(md.get("kind")).alias("kind"),
                                pl.lit(md.get("rank")).alias("rank"),
                                pl.lit(md.get("date")).alias("file_date"),
                                pl.lit(md.get("run_id")).alias("run_id"),
                                pl.lit(os.path.basename(p)
                                       ).alias("source_file"),
                            )
                        )
                        lfs.append(lf)

                    lf_all = pl.concat(lfs, how="diagonal_relaxed")
                    # If you want de-dup like your other func, add:
                    # lf_all = lf_all.unique(maintain_order=True)

                    lf_all.sink_parquet(out_path, statistics=True)
                except Exception as e:
                    print(f"  ! failed to write {out_name}: {e}")
                    continue

                if verbose:
                    print(f"  → wrote {out_path}")

                if delete_shards:
                    for p in paths:
                        try:
                            os.remove(p)
                        except Exception as e:
                            print(f"  ! could not delete {p}: {e}")
    except Exception as e:
        print(f"[consolidate] unexpected error: {e}")


def run_case(
        run_df_pl,
        run_save_directory,
        run_policy,
        run_solver,
        run_backend,
        run_name: str,
        run_tag: str,
        use_precomputed_floors: bool,
        run_show_progress: bool = True,
        workers_passed: int = 1
) -> None:
    df_pl = opt.day_and_slot_cols(
        run_df_pl,
        slot_len_min=run_policy.behavioral.slot_length_minutes)

    if use_precomputed_floors:
        df_pl = attach_floor_if_available(df_pl, floor_lf)
    elif run_policy.household_min is not None:
        df_pl = (
            opt.attach_household_floor(df=df_pl,
                                       cfg=run_policy.household_min))
    cols = [
        "ca_id", "city", "date", "day", "slot", "value",
        "marginal_emissions_factor_grams_co2_per_kWh",
        "average_emissions_factor_grams_co2_per_kWh",
    ]
    if "floor_kwh" in df_pl.columns:
        cols.append("floor_kwh")

    df_pl = df_pl.select(cols).sort(["ca_id", "day", "slot"])

    order_map = opt.build_cityday_ca_order_map_pl(df_pl)

    df_pd = df_pl.to_pandas()
    df_pd["city"] = df_pd["city"].astype("category")
    df_pd["ca_id"] = df_pd["ca_id"].astype("category")

    r = MPI.COMM_WORLD.Get_rank() if run_backend == "mpi" else 0
    print(f"[case][r{r}] {run_name} | "
          f"solver={short_solver(run_solver)} | "
          f"policy=({short_policy(run_policy)}) | "
          f"backend={run_backend}", flush=True)

    metrics, moves, opt_df = opt.run_pipeline_pandas_cityweek_budget(
        df_pd=df_pd,
        policy=run_policy,
        solver=run_solver,
        emit_optimised_rows=True,
        show_progress=run_show_progress,
        cityday_ca_order=order_map,
        backend=run_backend,
        executor=None,
        workers=workers_passed,
    )

    # Per-rank file shards
    rank = MPI.COMM_WORLD.Get_rank() if run_backend == "mpi" else 0
    shard = f"_r{rank:04d}" if run_backend == "mpi" else ""
    prefix = os.path.join(run_save_directory, f"{run_name}")

    opt.write_parquet_fsync(metrics,
                            f"{prefix}_metrics{shard}_{run_tag}.parquet")
    opt.write_parquet_fsync(moves,
                            f"{prefix}_moves{shard}_{run_tag}.parquet")
    opt.write_parquet_fsync(opt_df,
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
meter_save_directory = os.path.join(hitachi_data_directory,
                                    "meter_primary_files")
marginal_emissions_development_directory = os.path.join(
    base_data_directory,
    "marginal_emissions_development")
marginal_emissions_results_directory = os.path.join(
    marginal_emissions_development_directory,
    "results")
marginal_emissions_logs_directory = os.path.join(
    marginal_emissions_development_directory,
    "logs")
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

os.makedirs(optimisation_development_directory, exist_ok=True)
os.makedirs(full_results_directory, exist_ok=True)
os.makedirs(test_results_directory, exist_ok=True)


# TEST DATASET RUNS
marginal_emissions_filename = (
    "meter_readings_all_years_20250714_formatted_with_emission"
    "_factors_filled_2022-05-04_to_2022-05-18")
marginal_emissions_filepath = os.path.join(
    optimisation_development_directory,
    marginal_emissions_filename + ".parquet")
save_dir = test_results_directory

floor_lookup_filename = (
    "meter_readings_all_years_20250714_household_floor_lookup")
floor_lookup_filepath = os.path.join(
    optimisation_processing_directory, floor_lookup_filename + ".parquet"
)

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# IMPLEMENTATION
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    try:
        world_size = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    except Exception:
        world_size, rank = 1, 0

    use_mpi = world_size > 1
    run_backend = "mpi" if use_mpi else "local"

    if use_mpi:
        comm = MPI.COMM_WORLD
        run_tag = datetime.now(
                        ).strftime("%Y%m%d_%H%M%S") if rank == 0 else None
        run_tag = comm.bcast(run_tag, root=0)
        if rank == 0:
            print(f"[runner] MPI world size = {world_size}", flush=True)
    else:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data on each rank (simplest; OK on parallel FS)
    rename_candidates = {
        "marginal_emissions_grams_co2_per_kWh":
        "marginal_emissions_factor_grams_co2_per_kWh",
        "average_emissions_grams_co2_per_kWh":
        "average_emissions_factor_grams_co2_per_kWh",
        "average_emissions_factor_co2_per_kWh":
        "average_emissions_factor_grams_co2_per_kWh",
    }

    df_raw = pl.read_parquet(marginal_emissions_filepath)
    present = [k for k in rename_candidates if k in df_raw.columns]
    marginal_emissions_pldf = (
        df_raw.rename({k: rename_candidates[k] for k in present})
        .with_columns(pl.col("city").cast(pl.Utf8)
                      .str.strip_chars().str.to_lowercase())
    )

    try:
        _lf = pl.scan_parquet(floor_lookup_filepath)  # lazy, efficient
        # normalize key dtypes exactly like we’ll compute in-ram (Int32)
        floor_lf = _lf.with_columns([
            pl.col("hod").cast(pl.Int32),
            pl.col("dow").cast(pl.Int32),
        ])
        HAVE_PRECOMPUTED_FLOOR = True
    except Exception:
        floor_lf = None
        HAVE_PRECOMPUTED_FLOOR = False

    # Enable via env; falls back to on-the-fly floor if lookup missing
    USE_PRECOMPUTED_FLOORS = (
        os.getenv("USE_PRECOMPUTED_FLOORS", "1") == "1"
    ) and HAVE_PRECOMPUTED_FLOOR

    if USE_PRECOMPUTED_FLOORS:
        print("[runner] Using precomputed household floor lookup.", flush=True)
    else:
        print("[runner] Computing household floor on the fly.", flush=True)

    # ────────────────────────────────────────────────────────────────────────────
    # Define Solvers
    # ────────────────────────────────────────────────────────────────────────────

    greedy_solver = opt.SolverConfig(solver_family="greedy")

    continuous_solver_slsqp = opt.SolverConfig(
        solver_family="continuous", continuous_method="SLSQP",
        continuous_opts={"ftol": 1e-6})
    continuous_solver_trust_constr = opt.SolverConfig(
        solver_family="continuous", continuous_method="trust-constr",
        continuous_opts={"gtol": 1e-6, "xtol": 1e-6})
    continuous_solver_closed_form = opt.SolverConfig(
        solver_family="continuous", continuous_method="closed-form",
        continuous_opts={"gtol": 1e-6, "xtol": 1e-6})
    lp_solver_highs = opt.SolverConfig(
        solver_family="lp", lp_solver="HIGHS",
        lp_solver_opts={"time_limit": 5.0, "threads": 1, "presolve": "on"})
    # lp_solver_opts={"threads": HIGHS_THREADS} if HIGHS_THREADS else None,)
    lp_solver_glpk = opt.SolverConfig(
        solver_family="lp", lp_solver="GLPK",
        lp_solver_opts={"tm_limit": 5000})
    lp_solver_ecos = opt.SolverConfig(
        solver_family="lp", lp_solver="ECOS",
        lp_solver_opts={"max_iters": 1000, "abstol": 1e-6, "reltol": 1e-6,
                        "feastol": 1e-6})
    lp_solver_scs = opt.SolverConfig(
        solver_family="lp", lp_solver="SCS",
        lp_solver_opts={"max_iters": 5000,  "eps": 1e-4})
    lp_solver_clarabel = opt.SolverConfig(
        solver_family="lp", lp_solver="CLARABEL",
        lp_solver_opts={"max_iter": 10000, "tol_gap_rel": 1e-4,
                        "tol_feas": 1e-6})
    lp_solver_osqp = opt.SolverConfig(
        solver_family="lp", lp_solver="OSQP",
        lp_solver_opts={"max_iter": 20000, "eps_abs": 1e-5, "eps_rel": 1e-5,
                        "polish": True, "verbose": True})

    # Minor Env Setup things
    delhi_total_daily_average_load_gWh = 3.937  # GWh
    maharashtra_total_daily_average_load_gWh = 17.750  # GWh

    delhi_total_daily_average_load_kWh = (
        delhi_total_daily_average_load_gWh * 1_000_000
    )  # convert GWh to kWh
    maharashtra_total_daily_average_load_kWh = (
        maharashtra_total_daily_average_load_gWh * 1_000_000
    )  # convert GWh to kWh

    # ────────────────────────────────────────────────────────────────────────────
    # Define configurations and policies
    # ────────────────────────────────────────────────────────────────────────────

    # CONFIGURATION 2 - Stricter
    # ────────────────────────────────────────────────────────────────────────────

    peak_hours_reduction_limit_config_2 = opt.PeakHoursReductionLimitConfig(
        peak_hours_reduction_percent_limit=25.0,
        peak_hours_reduction_scope="per_city",
        peak_hours_dict={
            "delhi": {
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
            },
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
    # RUNS AND VARIATIONS
    # ────────────────────────────────────────────────────────────────────────────
    # Policy 2
    # ────────────────────────────────────────────────────────────────────────────
    # # ------------- GREEDY SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_2,
    #              run_solver=greedy_solver,
    #              run_backend=run_backend,
    #              run_name='policy_2_greedy',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running GREEDY solver for policy 2: {e}")

    # ------------- continuous_solver_slsqp SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=continuous_solver_slsqp,
                 run_backend=run_backend,
                 run_name='policy_2_continuous_slsqp',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running continuous solver (SLSQP) "
              f"for policy 2: {e}")

    # ------------- continuous_solver_trust_constr SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=continuous_solver_trust_constr,
                 run_backend=run_backend,
                 run_name='policy_2_continuous_solver_trust_constr',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running continuous solver (TRUST_CONSTR) "
              f"for policy 2: {e}")

    # ------------- continuous_solver_closed_form SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=continuous_solver_closed_form,
                 run_backend=run_backend,
                 run_name='policy_2_continuous_solver_closed_form',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running continuous solver (CLOSED_FORM) "
              f"for policy 2: {e}")

    # ------------- lp_solver_highs SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=lp_solver_highs,
                 run_backend=run_backend,
                 run_name='policy_2_lp_solver_highs',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running lp_solver_highs for policy 2: {e}")

    # ------------- lp_solver_glpk SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=lp_solver_glpk,
                 run_backend=run_backend,
                 run_name='policy_2_lp_solver_glpk',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running lp_solver_glpk for policy 2: {e}")

    # ------------- lp_solver_ecos SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=lp_solver_ecos,
                 run_backend=run_backend,
                 run_name='policy_2_lp_solver_ecos',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running lp_solver_ecos for policy 2: {e}")

    # ------------- lp_solver_scs SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=lp_solver_scs,
                 run_backend=run_backend,
                 run_name='policy_2_lp_solver_scs',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running lp_solver_scs for policy 2: {e}")

    # ------------- lp_solver_clarabel SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=lp_solver_clarabel,
                 run_backend=run_backend,
                 run_name='policy_2_lp_solver_clarabel',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running lp_solver_clarabel for policy 2: {e}")

    # ------------- lp_solver_osqp SOLVER ------------- #
    try:
        run_case(run_df_pl=marginal_emissions_pldf,
                 run_save_directory=save_dir,
                 run_policy=policy_2,
                 run_solver=lp_solver_osqp,
                 run_backend=run_backend,
                 run_name='policy_2_lp_solver_osqp',
                 run_tag=run_tag,
                 use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
                 run_show_progress=True,
                 workers_passed=LOCAL_WORKERS)
    except Exception as e:
        print(f"[runner] Error running lp_solver_osqp for policy 2: {e}")

    # ---------- call once finished ----------
    try:
        # best-effort barrier; ignore if MPI isn't usable
        try:
            comm = MPI.COMM_WORLD
            comm.Barrier()
            rank = comm.Get_rank()
        except Exception:
            rank = 0
    finally:
        # rank 0 consolidates, unconditionally
        if (rank == 0):
            consolidate_latest_runs(
                save_dir,
                kinds=("metrics", "moves", "optimised"),
                delete_shards=False,
                only_ts=None,               # ignore run_tag; do all by default
                consolidate_all_ts=True,    # ← merge every timestamp present
                verbose=True,
            )


# EXTRA CONFIGURATIONS - NOT USED
    # # Policy 1
    # ────────────────────────────────────────────────────────────────────────────
    # try:
    #     # ------------- GREEDY SOLVER ------------- #
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=greedy_solver,
    #              run_backend=run_backend,
    #              run_name='policy_1_greedy',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running greedy solver for policy 1: {e}")
    # # ------------- continuous_solver_slsqp SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=continuous_solver_slsqp,
    #              run_backend=run_backend,
    #              run_name='policy_1_continuous_slsqp',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running continuous solver (SLSQP) "
    #           f"for policy 1: {e}")

    # # ------------- continuous_solver_trust_constr SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=continuous_solver_trust_constr,
    #              run_backend=run_backend,
    #              run_name='policy_1_continuous_solver_trust_constr',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running continuous solver (trust-constr) "
    #           f"for policy 1: {e}")

    # # ------------- lp_solver_highs SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=lp_solver_highs,
    #              run_backend=run_backend,
    #              run_name='policy_1_lp_solver_highs',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running LP solver (HIGHS) for policy 1: {e}")

    # # ------------- lp_solver_glpk SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=lp_solver_glpk,
    #              run_backend=run_backend,
    #              run_name='policy_1_lp_solver_glpk',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running LP solver (GLPK) for policy 1: {e}")

    # # ------------- lp_solver_ecos SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=lp_solver_ecos,
    #              run_backend=run_backend,
    #              run_name='policy_1_lp_solver_ecos',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running LP solver (ECOS) for policy 1: {e}")

    # # ------------- lp_solver_scs SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=lp_solver_scs,
    #              run_backend=run_backend,
    #              run_name='policy_1_lp_solver_scs',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running LP solver (SCS) for policy 1: {e}")
    # # ------------- lp_solver_clarabel SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=lp_solver_clarabel,
    #              run_backend=run_backend,
    #              run_name='policy_1_lp_solver_clarabel',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running LP solver (CLARABEL) for "
    # policy 1: {e}")

    # # ------------- lp_solver_osqp SOLVER ------------- #
    # try:
    #     run_case(run_df_pl=marginal_emissions_pldf,
    #              run_save_directory=save_dir,
    #              run_policy=policy_1,
    #              run_solver=lp_solver_osqp,
    #              run_backend=run_backend,
    #              run_name='policy_1_lp_solver_osqp',
    #              run_tag=run_tag,
    #              use_precomputed_floors=USE_PRECOMPUTED_FLOORS,
    #              run_show_progress=True,
    #              workers_passed=LOCAL_WORKERS)
    # except Exception as e:
    #     print(f"[runner] Error running LP solver (OSQP) for policy 1: {e}")

    # # CONFIGURATION 1 - Relax
    # ────────────────────────────────────────────────────────────────────────────
    # peak_hours_reduction_limit_config_1 = opt.PeakHoursReductionLimitConfig(
    #     peak_hours_reduction_percent_limit=80,
    #     peak_hours_reduction_scope="per_city",
    #     peak_hours_dict={
    #         "delhi": {
    #             "Mon": [8, 9, 10, 11, 12, 20],
    #             "Tue": [9, 10, 11, 20, 21, 22],
    #             "Wed": [9, 10, 11, 20, 21, 22],
    #             "Thu": [9, 10, 11, 20, 21, 22],
    #             "Fri": [8, 9, 10, 11, 20, 21],
    #             "Sat": [9, 10, 11, 12, 13, 20],
    #             "Sun": [10, 11, 12, 13, 14, 20],
    #         },
    #         "mumbai": {
    #             "Mon": [8, 9, 10, 11, 12, 20],
    #             "Tue": [9, 10, 11, 20, 21, 22],
    #             "Wed": [9, 10, 11, 20, 21, 22],
    #             "Thu": [9, 10, 11, 20, 21, 22],
    #             "Fri": [8, 9, 10, 11, 20, 21],
    #             "Sat": [9, 10, 11, 12, 13, 20],
    #             "Sun": [10, 11, 12, 13, 14, 20],
    #         },
    #     },
    # )

    # policy_1 = opt.ShiftPolicy(
    #     behavioral=opt.CustomerAdoptionBehavioralConfig(
    #         customer_power_moves_per_day=2,
    #         customer_power_moves_per_week=7,
    #         timezone="Asia/Kolkata",
    #         day_boundaries="00:00-24:00",
    #         week_boundaries="Mon-Sun",
    #         shift_hours_window=3.0,
    #         slot_length_minutes=30,
    #         peak_hours_reduction_limit_config=(
    #             peak_hours_reduction_limit_config_1)
    #     ),
    #     regional_cap=opt.RegionalLoadShiftingLimitConfig(
    #         regional_load_shift_percent_limit=10.0,
    #         regional_total_daily_average_load_kWh={
    #             "delhi": delhi_total_daily_average_load_kWh,
    #             "mumbai": maharashtra_total_daily_average_load_kWh
    #         }
    #     ),
    #     household_min=opt.HouseholdMinimumConsumptionLimitConfig(
    #         household_minimum_baseline_period="year",
    #         household_minimum_baseline_type="average",
    #         household_minimum_robust_max_percentile=95,
    #         household_minimum_R_percent=10
    #     ),
    #     spike_cap=opt.ShiftWithoutSpikeLimitConfig(alpha_peak_cap_percent=25),
    # )
