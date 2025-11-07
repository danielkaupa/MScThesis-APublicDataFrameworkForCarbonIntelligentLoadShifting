# ─────────────────────────────────────────────────────────────────────────────
# FILE: step4_marginal_emissions_model_runner.py
#
# PURPOSE:
#   Entrypoint script to run marginal-emissions experiments end-to-end:
#   - Load prepared train/validation/(optional test) splits
#   - Build feature, binning, and groupwise OLS pipelines
#   - Launch grid searches (quantile and/or median binning)
#   - Log results using the rotating CSV index
#   - (Optionally) distribute configs across MPI ranks transparently
#
# SECTIONS IN THIS FILE
#   - File Paths & Directories: resolve repo-relative paths and inputs.
#   - Loading Data: read Parquet splits (Polars) and convert to Pandas.
#   - Constructing Pipelines: feature adders, binners, regressors, pipelines.
#   - Running Quantile Grid Search: build grid, launch orchestrator.
#   - Running Median Grid Search: build grid, launch orchestrator.
#   - Finalize: write completion sentinel / message.
#
# RUN REQUIREMENTS:
#   - Python 3.10+
#   - Dependencies: pandas, numpy, polars, scikit-learn, statsmodels
#   - Logging/orchestration utils from this repo:
#       step4_marginal_emissions_feature_engineering
#       step4_marginal_emissions_models
#       step4_marginal_emissions_scoring
#       step4_marginal_emissions_logging
#       step4_marginal_emissions_orchestration
#   - Parquet readers: Polars (this script) and pyarrow/fastparquet if needed
#       elsewhere
#   - Optional: mpi4py — only required if you actually run under MPI
#
# USAGE:
#   $ python step4_marginal_emissions_experiment_runner.py
#   (Or import and call `main()` from another driver.)
#
# NOTES:
#   - MPI: Distribution is handled *inside* the orchestration layer. Do not
#     pre-shard the grids here; just pass the full grids and let the runner
#     handle rank slicing.
#   - Logging: Results are appended to a rotating CSV set with a global index.
# ─────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os
from pathlib import Path
import polars as pl
from sklearn.pipeline import Pipeline

# Repo modules
from step4_marginal_emissions_logging import _abs_join
from step4_marginal_emissions_feature_engineering import (
    DateTimeFeatureAdder,
    AnalysisFeatureAdder,
    # MultiQuantileBinner,
    # MultiMedianBinner,
    # For right now we do not need to import the binners because the
    # parameter grid setup takes care of that for us
)
from step4_marginal_emissions_models import GroupwiseRegressor
from step4_marginal_emissions_orchestration import (
    run_grid_search_auto,
    build_quantile_grid_configs,
    build_median_binner_configs,
)


def main() -> None:
    # ────────────────────────────────────────────────────────────────────────────
    # FILE PATHS AND DIRECTORIES
    # ────────────────────────────────────────────────────────────────────────────

    # Resolve repo root → .../code_and_analysis (one level up from scripts/)
    try:
        base_directory = os.path.abspath(os.path.join(
            os.path.dirname(__file__), ".."))
    except NameError:
        # Fallback for interactive runs
        base_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))

    # Define Directories
    base_data_directory = _abs_join(base_directory, "data")
    # hitachi_data_directory = os.path.join(base_data_directory,
    # "hitachi_copy")
    marginal_emissions_development_directory = os.path.join(
        base_data_directory, "marginal_emissions_development")
    # marginal_emissions_results_directory = os.path.join(
    #     marginal_emissions_development_directory, "results")
    marginal_emissions_logs_directory = os.path.join(
        marginal_emissions_development_directory, "logs")

    marginal_emissions_prefix = "marginal_emissions_results"

    # Define File names and paths
    # cleaned weather data
    # base_file = "weather_and_grid_data_half-hourly_20250714_1401"

    train_file = "marginal_emissions_estimation_20250714_1401_train_data"
    validation_file = (
        "marginal_emissions_estimation_20250714_1401_validation_data")
    test_file = "marginal_emissions_estimation_20250714_1401_test_data"

    # base_filepath = os.path.join(hitachi_data_directory,
    # base_file + ".parquet")

    train_filepath = os.path.join(
        marginal_emissions_development_directory, train_file + ".parquet")
    validation_filepath = os.path.join(
        marginal_emissions_development_directory, validation_file + ".parquet")
    test_filepath = os.path.join(
        marginal_emissions_development_directory, test_file + ".parquet")

    # ────────────────────────────────────────────────────────────────────────────
    # LOADING DATA
    # ────────────────────────────────────────────────────────────────────────────

    # base_pldf = pl.read_parquet(base_filepath)
    train_pldf = pl.read_parquet(train_filepath)
    validation_pldf = pl.read_parquet(validation_filepath)
    test_pldf = pl.read_parquet(test_filepath)

    # ────────────────────────────────────────────────────────────────────────────
    # IMPLEMENTATION
    # ────────────────────────────────────────────────────────────────────────────

    # DATA PROCESSING
    # ────────────────────────────────────────────────────────────────────────────
    # Conversion to Pandas DataFrame for compatibility with existing code
    train_df = train_pldf.to_pandas()
    validation_df = validation_pldf.to_pandas()
    test_df = test_pldf.to_pandas()

    # x_original_relevant_columns = [
    #     "demand_met", "demand_met_sqrd",
    #     "surface_net_solar_radiation_kWh_per_m2", "wind_speed_mps",
    #     "month", "hour",
    # ]
    # y_original_relevant_columns = ["tons_co2"]

    # assuming full_pipeline = Pipeline([...,"regressor", reg])
    train_pdf_x_all = train_df.drop(columns=["tons_co2"])
    train_pdf_y = train_df["tons_co2"]
    validation_pdf_x_all = validation_df.drop(columns=["tons_co2"])
    validation_pdf_y = validation_df["tons_co2"]
    test_pdf_x_all = test_df.drop(columns=["tons_co2"])
    test_pdf_y = test_df["tons_co2"]

    # CONSTRUCTING PIPELINES
    # ────────────────────────────────────────────────────────────────────────────
    # Feature addition
    feature_addition_pipeline = Pipeline([
        ("Add_Datetime_Features",
            DateTimeFeatureAdder(timestamp_col="timestamp")),
        ("Add_Original_Analysis_Features",
            AnalysisFeatureAdder(timestamp_col="timestamp",
                                 demand_met_col="demand_met",
                                 co2_col="tons_co2")),
    ])
    feature_addition_pipeline.name = "FeatureAdditionPipeline"

    # THE FOLLOWING SECTIONS ARE COMMENTED OUT BECAUSE THEY ARE NOT NEEDED
    # FOR THE GRID-SEARCH STYLE IMPLEMENTATION PERFORMED HERE, BUT THEY ARE
    # KEPT FOR REFERENCE AS THEY COULD BE USED FOR SINGULAR RUNS

    # Binners
    # original_multi_binner = MultiQuantileBinner(
    #     bin_specs={
    #         "surface_net_solar_radiation_kWh_per_m2": 5,
    #         "wind_speed_mps": 5,
    #     },
    #     group_col_name="original_quantile_group_id"
    # )
    # original_median_binner = MultiMedianBinner(
    #     variables=[
    #         "surface_net_solar_radiation_kWh_per_m2",
    #         "wind_speed_mps"
    #     ],
    #     group_col_name="median_group_id",
    # )
    # median_binner_v1 = MultiMedianBinner(
    #     variables=[
    #         "surface_net_solar_radiation_kWh_per_m2",
    #         "wind_speed_mps",
    #         "temperature_celsius",
    #     ],
    #     group_col_name="median_group_id",
    # )
    # Regressors
    # original_multi_binner_regressor = GroupwiseRegressor(
    #     y_var="tons_co2",
    #     x_vars=["demand_met", "demand_met_sqrd"],
    #     fe_vars=["month", "hour"],
    #     group_col="original_quantile_group_id",
    #     min_group_size=20,
    #     track_metrics=True,
    #     verbose=True
    # )
    # original_median_regressor = GroupwiseRegressor(
    #     y_var="tons_co2",
    #     x_vars=["demand_met", "demand_met_sqrd"],
    #     fe_vars=["month", "hour"],
    #     group_col="median_group_id",
    #     min_group_size=20,
    #     track_metrics=True,
    #     verbose=True
    # )
    # median_regressor_v1 = GroupwiseRegressor(
    #     y_var="tons_co2",
    #     x_vars=["demand_met", "demand_met_sqrd"],
    #     fe_vars=["month", "hour", "week_of_year"],
    #     group_col="median_group_id",
    #     min_group_size=20,
    #     track_metrics=True,
    #     verbose=True
    # )

    # FULL PIPELINES
    # original_multi_binner_regressor_pipeline = Pipeline([
    #         ("Feature_Addition", feature_addition_pipeline),
    #         ("Multi_Quantile_Binner", original_multi_binner),
    #         ("Groupwise_Regressor", original_multi_binner_regressor)
    # ])

    # original_median_regressor_pipeline = Pipeline([
    #     ("Feature_Addition", feature_addition_pipeline),
    #     ("Multi_Median_Binner", original_median_binner),
    #     ("Groupwise_Regressor", original_median_regressor)
    # ])

    # median_regressor_pipeline_v1 = Pipeline([
    #     ("Feature_Addition", feature_addition_pipeline),
    #     ("Multi_Median_Binner", median_binner_v1),
    #     ("Groupwise_Regressor", median_regressor_v1)
    # ])

    # RUNNING QUANTILE SEARCH
    # ────────────────────────────────────────────────────────────────────────────
    # Setup parameter grid
    multi_quantile_param_grid = build_quantile_grid_configs(
        candidate_binning_vars=["surface_net_solar_radiation_kWh_per_m2",
                                "wind_speed_mps", "temperature_celsius",
                                "precipitation_mm", "total_cloud_cover"],
        candidate_bin_counts=[3, 5, 10, 20, 50, 100],
        candidate_x_vars=["demand_met", "demand_met_sqrd"],
        candidate_fe_vars=["month", "hour", "week_of_year", "day_of_week",
                           "half_hour"],
        x_var_length=2,
        binner_extra_grid={"oob_policy": ["clip"],
                           "max_oob_rate": [0.05, 0.03, None],
                           "retain_flags": [True]}
    )

    # Setup regressor kwargs
    regressor_kwargs_q = {
        "y_var": "tons_co2",
        # default; overwritten per-config anyway
        "x_vars": ["demand_met", "demand_met_sqrd"],
        "fe_vars": ["month", "hour"],
        "group_col": "quantile_group_id",
        "min_group_size": 20,
        "track_metrics": True,
        "verbose": False,
        "random_state": 12,
    }
    # Execute search
    run_grid_search_auto(
        base_feature_pipeline=feature_addition_pipeline,
        regressor_cls=GroupwiseRegressor,
        regressor_kwargs=regressor_kwargs_q,
        grid_config=multi_quantile_param_grid,
        x_splits={"train": train_pdf_x_all,
                  "validation": validation_pdf_x_all,
                  "test": test_pdf_x_all},
        y_splits={"train": train_pdf_y,
                  "validation": validation_pdf_y,
                  "test": test_pdf_y},
        # shared filesystem path
        results_dir=marginal_emissions_logs_directory,
        # separate shard family
        file_prefix=marginal_emissions_prefix + "_quantile",
        max_log_mb=95,
        # durability on HPC
        fsync=True,
        base_feature_pipeline_name="FeatureAdditionPipeline",
        # no test during tuning
        eval_splits=("train", "validation"),
        # "mpi" if MPI present, else "single"
        distribute="auto",
        # good default
        dist_mode="stride",
        force_run=False,
        force_overwrite=False,
        seed=12,
    )

    # RUNNING MEDIAN SEARCH
    # ────────────────────────────────────────────────────────────────────────────
    # Setup parameter grid
    candidate_binning_vars = ["surface_net_solar_radiation_kWh_per_m2",
                              "wind_speed_mps", "temperature_celsius",
                              "precipitation_mm", "total_cloud_cover"]
    candidate_x_vars = ["demand_met", "demand_met_sqrd"]
    candidate_fe_vars = ["month", "hour", "week_of_year", "day_of_week",
                         "half_hour"]

    multi_median_param_grid = build_median_binner_configs(
        candidate_binning_vars=candidate_binning_vars,
        candidate_x_vars=candidate_x_vars,
        candidate_fe_vars=candidate_fe_vars,
        x_var_length=2,
    )

    # Setup regressor kwargs
    regressor_kwargs_m = {
        "y_var": "tons_co2",
        # default; grid entries can override
        "x_vars": ["demand_met", "demand_met_sqrd"],
        # default; grid entries can override
        "fe_vars": ["month", "hour"],
        # MUST match binner's group_col_name
        "group_col": "median_group_id",
        "min_group_size": 20,
        "track_metrics": True,
        "verbose": False,
        "random_state": 12
    }

    # Execute search
    run_grid_search_auto(
        base_feature_pipeline=feature_addition_pipeline,
        regressor_cls=GroupwiseRegressor,
        regressor_kwargs=regressor_kwargs_m,
        grid_config=multi_median_param_grid,
        x_splits={"train": train_pdf_x_all,
                  "validation": validation_pdf_x_all,
                  "test": test_pdf_x_all},
        y_splits={"train": train_pdf_y,
                  "validation": validation_pdf_y,
                  "test": test_pdf_y},
        results_dir=marginal_emissions_logs_directory,
        file_prefix=marginal_emissions_prefix + "_median",
        max_log_mb=95,
        fsync=True,
        base_feature_pipeline_name="FeatureAdditionPipeline",
        eval_splits=("train", "validation"),
        distribute="auto",
        dist_mode="stride",
        force_run=False,
        force_overwrite=False,
        seed=12,
    )

    # finalise
    Path(".step4_DONE").touch()
    print("ALL_COMBINATIONS_EXHAUSTED", flush=True)


if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
# This script estimates short-run marginal emission factors (MEFs)—the
#  incremental CO₂ associated with an incremental unit of electricity
# demand—while allowing those effects to vary with weather regimes.
# It engineers datetime and analysis features (e.g., Q², log(Q)) and
# partitions observations via multi-quantile or multi-median binning on
# selected weather variables. Within each bin, it fits an OLS with fixed
# effects (e.g., month/hour/week_of_year) to model CO₂ as a function of
# demand Q and Q², then derives per-row MEFs from the fitted slope:
#     ME_t = α₁ + 2·α₂·Q_t.
#
# The script evaluates model quality per bin and in aggregate using R²,
# RMSE, MAE, and MAPE, and also computes energy-weighted micro-averages (MWh).
# Diagnostics include a pooled CO₂ fit across bins and a finite-difference
# check comparing predicted MEFs to short-horizon slopes (ΔCO₂/ΔQ).
# A configurable grid search sweeps binning choices and x/FE combinations,
# appending one-row summaries to rotating, lock-safe CSV logs.
# Execution supports single-node or MPI contexts for distributed runs.
# ─────────────────────────────────────────────────────────────────────────────