# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

import os
from mpi4py import MPI
import step5_optimisation_module as opt
from datetime import datetime
import polars as pl
import re
import glob

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HIGHS_THREADS = int(os.getenv("HIGHS_THREADS",
                              os.getenv(key="OMP_NUM_THREADS",
                                            default="1")))

# ────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────


def short_solver(s):
    if s.solver_family == "milp":
        th = (s.milp_solver_opts or {}).get("threads", "auto")
        return f"milp/{s.milp_solver}(threads={th})"
    return "greedy"


def short_policy(p):
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


def consolidate_latest_runs(out_dir: str,
                            kinds=("metrics","moves","optimised"),
                            delete_shards: bool = False,
                            only_ts: str | None = None):
    rx = re.compile(
        r"^(?P<case>.+)_(?P<kind>metrics|moves|optimised)"
        r"(?:_r(?P<rank>\d{4}))?_(?P<ts>\d{8}_\d{4,6})\.parquet$"
    )

    all_files = [os.path.basename(p)
                 for p in glob.glob(os.path.join(
                     out_dir, "*.parquet"))]
    parsed = []
    for f in all_files:
        m = rx.match(f)
        if m and m.group("kind") in kinds:
            d = m.groupdict()
            parsed.append((d["case"], d["kind"], d["ts"], f))

    if not parsed:
        print("[consolidate] no matching shard files found.")
        return

    groups = {}
    for case, kind, ts, f in parsed:
        groups.setdefault((case, kind), []).append((ts, f))

    for (case, kind), vals in sorted(groups.items()):
        # choose latest or enforce only_ts
        if only_ts is not None:
            target_ts = only_ts
        else:
            target_ts = max(ts for ts, _ in vals)
        shard_files = [f for ts, f in vals if ts == target_ts]

        if not shard_files:
            continue

        out_name = f"{case}_{kind}_{target_ts}.parquet"
        out_path = os.path.join(out_dir, out_name)

        # If already merged (non-MPI single file), skip
        if len(shard_files) == 1 and shard_files[0] == out_name:
            print(f"[consolidate] {case}/{kind} already consolidated "
                  f"@ {target_ts}; skip.")
            continue

        paths = [os.path.join(out_dir, f) for f in shard_files]
        print(f"[consolidate] {case}/{kind}: "
              f"{len(paths)} shard(s) → {out_name}")

        try:
            pl.scan_parquet(paths).sink_parquet(out_path)
        except Exception as e:
            print(f"  ! failed to write {out_name}: {e}")
            continue

        print(f"  → wrote {out_path}")

        if delete_shards:
            for p in paths:
                if os.path.basename(p) != out_name:
                    try:
                        os.remove(p)
                    except Exception as e:
                        print(f"  ! could not delete {p}: {e}")


def run_case(
        run_df_pl,
        run_save_directory,
        run_policy,
        run_solver,
        run_backend,
        run_name,
        run_tag: str,
        run_show_progress=True
) -> None:
    df_pl = opt.day_and_slot_cols(
        run_df_pl,
        slot_len_min=run_policy.behavioral.slot_length_minutes)

    if run_policy.household_min is not None:
        df_pl = opt.attach_household_floor(df=df_pl,
                                           cfg=run_policy.household_min)

    cols = [
        "ca_id", "city", "date", "day", "slot", "value",
        "marginal_emissions_factor_grams_co2_per_kWh",
        "average_emissions_factor_grams_co2_per_kWh",
    ]

    if "floor_kwh" in df_pl.columns:
        cols.append("floor_kwh")

    df_pl = df_pl.select(cols).sort(["ca_id", "day", "slot"])

    # Build order map if you want (optional).
    # It is NOT shipped to workers
    order_map = opt.build_cityday_ca_order_map_pl(df_pl)

    df_pd = df_pl.to_pandas()
    df_pd["city"] = df_pd["city"].astype("category")
    df_pd["ca_id"] = df_pd["ca_id"].astype("category")
    r = MPI.COMM_WORLD.Get_rank() if run_backend == "mpi" else 0
    print(f"[case][r{r}] {run_name} | solver={short_solver(run_solver)} "
          f"| policy=({short_policy(run_policy)}) | backend={run_backend}",
          flush=True)
    # IMPORTANT: executor=None. Module will do static per-rank
    # slicing when backend=="mpi".
    metrics, moves, opt_df = opt.run_pipeline_pandas_cityweek_budget(
        df_pd=df_pd,
        policy=run_policy,
        solver=run_solver,
        emit_optimised_rows=True,
        show_progress=run_show_progress,
        cityday_ca_order=order_map,
        backend=run_backend,         # "mpi" or "local"
        executor=None,               # ← key change
    )

    # Per-rank shard filenames if running under MPI + static slicing
    rank = 0
    if run_backend == "mpi":
        try:
            rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            rank = 0

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
# DEFINING FILEPATHS AND DIRECTORIES
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


# File paths
# ALL meter_readings_all_years_20250714_formatted_with_emission_factors_filled
# TEST meter_readings_all_years_20250714_formatted_with_
# emission_factors_filled_2022-05-04_to_2022-05-18

# TEST DATASET RUNS
marginal_emissions_filename = (
    "meter_readings_all_years_20250714_formatted_with_emission"
    "_factors_filled_2022-05-04_to_2022-05-18")
marginal_emissions_filepath = os.path.join(
    optimisation_development_directory,
    marginal_emissions_filename + ".parquet")
save_dir = test_results_directory


# # FULL DATASEST RUNS
# marginal_emissions_filename = (
#     "meter_readings_all_years_20250714_formatted_with_emission_factors_filled")
# marginal_emissions_filepath = os.path.join(
#     optimisation_development_directory,
#     marginal_emissions_filename + ".parquet")
# save_dir = full_results_directory


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
    marginal_emissions_pldf = pl.read_parquet(
        marginal_emissions_filepath).rename({
                    "marginal_emissions_grams_co2_per_kWh":
                    "marginal_emissions_factor_grams_co2_per_kWh",
                    "average_emissions_grams_co2_per_kWh":
                    "average_emissions_factor_grams_co2_per_kWh", }
                )

    # Solvers: let HiGHS auto-pick threads; don't pass threads to CBC/GLPK
    greedy_solver = opt.SolverConfig(solver_family="greedy")
    milp_cbc_solver = opt.SolverConfig(solver_family="milp",
                                       milp_solver="cbc")
    milp_glpk_solver = opt.SolverConfig(solver_family="milp",
                                        milp_solver="glpk")
    milp_highs_solver = opt.SolverConfig(solver_family="milp",
                                         milp_solver="highs",
                                         milp_solver_opts={"threads":
                                                           HIGHS_THREADS},)

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

    # CONFIGURATION 1 - Relax
    # ────────────────────────────────────────────────────────────────────────────
    peak_hours_reduction_limit_config_1 = opt.PeakHoursReductionLimitConfig(
        peak_hours_reduction_percent_limit=80,
        peak_hours_reduction_scope="per_city",
        peak_hours_dict={
            "delhi": {
                "Mon": [8, 9, 10, 11, 12, 20],
                "Tue": [9, 10, 11, 20, 21, 22],
                "Wed": [9, 10, 11, 20, 21, 22],
                "Thu": [9, 10, 11, 20, 21, 22],
                "Fri": [8, 9, 10, 11, 20, 21],
                "Sat": [9, 10, 11, 12, 13, 20],
                "Sun": [10, 11, 12, 13, 14, 20],
            },
            "mumbai": {
                "Mon": [8, 9, 10, 11, 12, 20],
                "Tue": [9, 10, 11, 20, 21, 22],
                "Wed": [9, 10, 11, 20, 21, 22],
                "Thu": [9, 10, 11, 20, 21, 22],
                "Fri": [8, 9, 10, 11, 20, 21],
                "Sat": [9, 10, 11, 12, 13, 20],
                "Sun": [10, 11, 12, 13, 14, 20],
            },
        },
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
        peak_hours_dict={
            "delhi": {
                "Mon": [8, 9, 10, 11, 12, 20],
                "Tue": [9, 10, 11, 20, 21, 22],
                "Wed": [9, 10, 11, 20, 21, 22],
                "Thu": [9, 10, 11, 20, 21, 22],
                "Fri": [8, 9, 10, 11, 20, 21],
                "Sat": [9, 10, 11, 12, 13, 20],
                "Sun": [10, 11, 12, 13, 14, 20],
            },
            "mumbai": {
                "Mon": [8, 9, 10, 11, 12, 20],
                "Tue": [9, 10, 11, 20, 21, 22],
                "Wed": [9, 10, 11, 20, 21, 22],
                "Thu": [9, 10, 11, 20, 21, 22],
                "Fri": [8, 9, 10, 11, 20, 21],
                "Sat": [9, 10, 11, 12, 13, 20],
                "Sun": [10, 11, 12, 13, 14, 20],
            },
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

    # Policy 1
    # ────────────────────────────────────────────────────────────────────────────
    # ------------- GREEDY SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_1,
             run_solver=greedy_solver,
             run_backend=run_backend,
             run_name='policy_1_greedy',
             run_tag=run_tag,
             run_show_progress=True
             )

    # Policy 1
    # ────────────────────────────────────────────────────────────────────────────

    # ------------- milp_cbc_solver SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_1,
             run_solver=milp_cbc_solver,
             run_backend=run_backend,
             run_name='policy_1_milp_cbc',
             run_tag=run_tag,
             run_show_progress=True
             )

    # ------------- milp_glpk_solver SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_1,
             run_solver=milp_glpk_solver,
             run_backend=run_backend,
             run_name='policy_1_milp_glpk',
             run_tag=run_tag,
             run_show_progress=True
             )

    # ------------- milp_highs_solver SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_1,
             run_solver=milp_highs_solver,
             run_backend=run_backend,
             run_name='policy_1_milp_highs',
             run_tag=run_tag,
             run_show_progress=True
             )

    # Policy 2
    # ────────────────────────────────────────────────────────────────────────────
    # ------------- GREEDY SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_2,
             run_solver=greedy_solver,
             run_backend=run_backend,
             run_name='policy_2_greedy',
             run_tag=run_tag,
             run_show_progress=True
             )

    # ------------- milp_cbc_solver SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_2,
             run_solver=milp_cbc_solver,
             run_backend=run_backend,
             run_name='policy_2_milp_cbc',
             run_tag=run_tag,
             run_show_progress=True
             )

    # ------------- milp_glpk_solver SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_2,
             run_solver=milp_glpk_solver,
             run_backend=run_backend,
             run_name='policy_2_milp_glpk',
             run_tag=run_tag,
             run_show_progress=True
             )

    # ------------- milp_highs_solver SOLVER ------------- #
    run_case(run_df_pl=marginal_emissions_pldf,
             run_save_directory=save_dir,
             run_policy=policy_2,
             run_solver=milp_highs_solver,
             run_backend=run_backend,
             run_name='policy_2_milp_highs',
             run_tag=run_tag,
             run_show_progress=True
             )

    # ---------- call once finished ----------
    try:
        comm = MPI.COMM_WORLD
        comm.Barrier()          # <- wait for all ranks to finish writing
        rank = comm.Get_rank()
    except Exception:
        rank = 0

    if rank == 0:
        consolidate_latest_runs(
            save_dir,
            kinds=("metrics", "moves", "optimised"),
            # set True if you want to keep only merged files
            delete_shards=False,
            only_ts=run_tag
        )
