# ─────────────────────────────────────────────────────────────────────────────
# FILE: step4_marginal_emissions_orchestration.py
#
# PURPOSE:
#   Orchestrate end-to-end model runs for the marginal emissions pipeline:
#     - Apply fitted preprocessing and compute marginal emissions (ME)
#     - Evaluate per-split, per-group metrics and diagnostics
#     - Fit-and-export ME to Parquet for train/val/test
#     - Coordinate single-run orchestration and grid search sweeps
#     - (Optionally) shard configs across MPI ranks for parallel runs
#
# SECTIONS IN THIS FILE:
#   - Imports
#   - Running Models — Utilities
#       Helpers that apply fitted preprocessing and compute ME on a split.
#   - Running Models — Fit & Export
#       Train on train split, transform others, and write Parquet outputs.
#   - Running Models — Runners & Orchestrators
#       End-to-end single-run orchestration with logging/rotation.
#   - Running Models — Grid Search
#       Grid execution over binners/x-fe combinations with rotating logs.
#   - Running Models — Grid Search Utilities
#       Small helpers to generate x/fe and binner grids.
#
# RUN REQUIREMENTS:
#   - Python 3.10+ recommended
#   - Dependencies:
#       numpy, pandas, scikit-learn, statsmodels
#       (for Parquet export: pyarrow or fastparquet)
#   - Optional:
#       mpi4py (for distributed grid search)
#
# USAGE:
#   from step4_marginal_emissions_orchestration import regressor_orchestrator
#   summary = regressor_orchestrator(pipeline, x_splits, y_splits, ...)
#
# FUNCTION LIST:
#   _apply_fitted_preprocessing
#   compute_me_for_split
#   evaluate_on_split
#   fit_and_export_marginal_emissions
#   run_regressor_model
#   regressor_orchestrator
#   run_grid_search
#   run_grid_search_auto
#   all_nonempty_subsets
#   get_fe_vars
#   build_x_fe_combinations_disjoint
#   build_quantile_grid_configs
#   build_median_binner_configs

# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# Stdlib
import json
from datetime import datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any

# Third-party
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

# Local modules
from step4_marginal_emissions_models import GroupwiseRegressor
from step4_marginal_emissions_feature_engineering import (
    MultiQuantileBinner,
    MultiMedianBinner,
)
from step4_marginal_emissions_scoring import (
    mean_absolute_percentage_error,
    pooled_co2_metrics,
    finite_difference_me_metrics,
    summarise_metrics_logs,
    _compute_group_energy_weights,   # if you kept it in scoring
)
from step4_marginal_emissions_logging import (
    signature_for_run,
    is_model_logged_rotating_csv,
    save_summary_to_rotating_csv,
    _mpi_context,
    _distribute_configs,
)

# ────────────────────────────────────────────────────────────────────────────
# RUNNING MODELS — Utilities
# ────────────────────────────────────────────────────────────────────────────
#
# PURPOSE:
#   Helpers that run already-fitted pipelines *without re-fitting* the final
#   estimator and compute marginal emissions for a given split.
#
# FUNCTIONS:
#   - _apply_fitted_preprocessing: run all fitted transforms (no estimator)
#   - compute_me_for_split: apply full fitted pipeline and extract ME (+ids)
#
# NOTES:
#   - Avoids "Pipeline not fitted" noise by applying steps directly.
#   - Tries to recover feature names via get_feature_names_out().
# ────────────────────────────────────────────────────────────────────────────


def _apply_fitted_preprocessing(
        user_pipeline: Pipeline,
        X: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply all *already-fitted* steps in a pipeline except the final estimator,
    without constructing a new sklearn Pipeline (avoids 'Pipeline not fitted'
    warnings).

    Parameters
    ----------
    user_pipeline : Pipeline
        A pipeline that has already been fitted (on train) and whose final step
        is the estimator (e.g., GroupwiseRegressor).
    X : pd.DataFrame
        Raw features to transform through the fitted preprocessing steps.

    Returns
    -------
    pd.DataFrame
        The transformed features as a DataFrame. If a transformer returns a
        numpy array, we try to retrieve column names via
        `get_feature_names_out()`; otherwise we fall back to the original
        column names.
    """
    Z = X
    last_transformer = None

    for _, step in user_pipeline.steps[:-1]:
        if hasattr(step, "transform"):
            Z = step.transform(Z)
            last_transformer = step

    if isinstance(Z, pd.DataFrame):
        return Z

    # Try to recover column names
    cols = None
    try:
        # type: ignore[index]
        cols = user_pipeline[:-1].get_feature_names_out()
    except Exception:
        try:
            if last_transformer is not None and hasattr(
                last_transformer,
               "get_feature_names_out"):
                # type: ignore[assignment]
                cols = last_transformer.get_feature_names_out()
        except Exception:
            cols = None

    if cols is None:
        cols = X.columns
    return pd.DataFrame(Z, index=X.index, columns=list(cols))


def compute_me_for_split(
        fitted_pipeline: Pipeline,
        X: pd.DataFrame,
        split_name: str | None = None,
        id_cols: list[str] = ("timestamp", "city"),
        include_params: bool = True,
        keep_cols: list[str] = ("demand_met", "tons_co2"),
) -> pd.DataFrame:
    """
    Use a FITTED pipeline to compute marginal emissions (ME) for a single
    features DataFrame.

    Parameters
    ----------
    fitted_pipeline : Pipeline
        A pipeline that has already been fit on the training data. Its final
        step must be GroupwiseRegressor, whose transform adds 'ME'
        (and 'alpha1','alpha2').
    X : pd.DataFrame
        Feature table to transform. Must include the columns required by the
        pipeline’s feature steps and binner (e.g., weather vars),
        plus any IDs you want to keep.
    split_name : str, optional
        If provided, a 'split' column is added with this value
        ('train'/'validation'/'test'/etc).
    id_cols : list[str], default ('timestamp','city')
        Identifier columns to carry into the output if present in `X`
        after transform.
    include_params : bool, default True
        If True, also include 'alpha1' and 'alpha2' in the output.
    keep_cols : list[str], default ('demand_met','tons_co2')
        Additional columns to include if present (useful for diagnostics).

    Returns
    -------
    pd.DataFrame
        One row per input row with at least: id_cols ∩ columns, 'ME',
        and optionally 'alpha1','alpha2', the regressor’s group column,
        keep_cols, and 'split'.
    """
    # Transform through all steps → last step (GroupwiseRegressor) computes ME
    out = fitted_pipeline.transform(X)

    # Final estimator for group column name
    reg = getattr(fitted_pipeline, "_final_estimator", None)
    gcol = getattr(reg, "group_col", None)

    # Build column list in a safe, present-only way
    cols: list[str] = [c for c in id_cols if c in out.columns]
    if "ME" not in out.columns:
        raise RuntimeError("Pipeline transform did not produce 'ME'. Was the "
                           "final estimator fitted?")
    cols.append("ME")

    if include_params:
        for c in ("alpha1", "alpha2"):
            if c in out.columns:
                cols.append(c)

    if gcol and gcol in out.columns:
        cols.append(gcol)

    for c in keep_cols:
        if c in out.columns and c not in cols:
            cols.append(c)

    result = out[cols].copy()
    if split_name is not None:
        result["split"] = split_name
    return result


def evaluate_on_split(
        regression_model: GroupwiseRegressor,
        full_df: pd.DataFrame
) -> pd.DataFrame:
    """
    After pipeline.transform → full_df with group IDs & original y_var,
    compute per‑group r2/rmse/mae/n_obs using reg.group_models_.

    Parameters
    ----------
    reg : GroupwiseRegressor
        Fitted GroupwiseRegressor instance with group_models_ populated.
    full_df : pd.DataFrame
        DataFrame containing the original y_var and group_col.

    Returns
    -------
    pd.DataFrame
        DataFrame with group metrics: r2, rmse, mae, n_obs.
    """
    df = full_df.copy()
    gcol = regression_model.group_col
    yname = regression_model.y_var

    if gcol not in df.columns or yname not in df.columns:
        missing = [c for c in (gcol, yname) if c not in df.columns]
        raise KeyError(f"Required columns missing: {missing}")

    # Use the regressor's predict to ensure FE category handling is consistent
    y_true = df[yname]
    y_pred = regression_model.predict(df, predict_type="y")

    rows = []
    for grp, idx in df.groupby(gcol).groups.items():
        yt = y_true.loc[idx]
        yp = y_pred.loc[idx].dropna()
        # align just in case
        yt = yt.loc[yp.index]
        if len(yt) == 0:
            continue
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = np.nan
        rows.append({
            "group": grp,
            "r2": r2,
            "rmse": root_mean_squared_error(yt, yp),
            "mae": mean_absolute_error(yt, yp),
            "mape": mean_absolute_percentage_error(yt, yp),
            "n_obs": int(len(yt)),
        })

    mdf = pd.DataFrame(rows)
    if mdf.empty:
        # return empty with expected columns
        mdf = pd.DataFrame(columns=["group",
                                    "r2",
                                    "rmse",
                                    "mae",
                                    "mape",
                                    "n_obs"])
    return mdf


# ────────────────────────────────────────────────────────────────────────────
# RUNNING MODELS — Fit & Export
# ────────────────────────────────────────────────────────────────────────────
#
# PURPOSE:
#   Train on the train split, compute ME on requested splits, and write
#   results to Parquet (single file or per-split files).
#
# FUNCTIONS:
#   - fit_and_export_marginal_emissions
#
# NOTES:
#   - Learns binner edges and groupwise OLS params on *train only*.
#   - Validation/test transforms use those learned parameters.
#   - For large/HPC jobs, prefer per-split writing.
# ────────────────────────────────────────────────────────────────────────────


def fit_and_export_marginal_emissions(
        pipeline: Pipeline,
        x_splits: dict,
        y_splits: dict,
        out_parquet_path: str,
        *,
        id_cols: list[str] = ("timestamp", "city"),
        include_params: bool = True,
        keep_cols: list[str] = ("demand_met", "tons_co2"),
        order_splits: list[str] = ("train", "validation", "test"),
        save_mode: str = "single",              # "single" | "per_split"
        compression: str | None = "snappy",     # passed to pandas.to_parquet
        return_df: bool = True,                 # set False on huge runs
) -> pd.DataFrame:
    """
    Fit the pipeline on the train split, compute marginal emissions (ME) for
    each split, concatenate, and save to a single Parquet file.

    Parameters
    ----------
    pipeline : Pipeline
        Your full pipeline: [FeatureAddition → Binner → GroupwiseRegressor].
    x_splits : dict
        Feature splits, e.g.
        {"train": X_train, "validation": X_val, "test": X_test}.
    y_splits : dict
        Target splits, e.g.
        {"train": y_train, "validation": y_val, "test": y_test}.
        Only the train target is used for fitting; others are not needed for
        transform.
    out_parquet_path : str
        File path for the output Parquet dataset.
    id_cols : list[str], default ('timestamp','city')
        Identifier columns to include if present.
    include_params : bool, default True
        Include 'alpha1' and 'alpha2' in the export.
    keep_cols : list[str], default ('demand_met','tons_co2')
        Additional useful columns to include if present.
    order_splits : list[str], default ('train','validation','test')
        Order in which to compute and stack splits.
    save_mode : {"single","per_split"}, default "single"
        Use "per_split" on HPC/MPI (let rank 0 write or give each rank a
        different path).
    compression : str or None, default "snappy"
        Parquet compression codec (requires pyarrow/fastparquet support).
    return_df : bool, default True
        If False, skip building the concatenated DataFrame in memory.


    Returns
    -------
    pd.DataFrame or None
        Concatenated results if return_df=True and save_mode="single";
        otherwise None.


    Notes
    -----
    - Binner quantile edges and groupwise OLS coefficients are learned on
        TRAIN only.
    - Validation and test are transformed using those learned
        edges/coefficients.
    - In MPI jobs, prefer save_mode="per_split" or call this only on rank 0.
    - Binner edges and group OLS coefs are learned on train only.
    """
    out_parquet_path = Path(out_parquet_path)

    # Fit on train
    X_tr = x_splits["train"]
    y_tr = y_splits["train"]
    _ = pipeline.fit_transform(X_tr, y_tr)

    if save_mode not in {"single", "per_split"}:
        raise ValueError("save_mode must be 'single' or 'per_split'.")

    # Compute ME for each requested split
    parts: list[pd.DataFrame] = []
    for split in order_splits:
        if split not in x_splits:
            continue
        df_me = compute_me_for_split(
            fitted_pipeline=pipeline,
            X=x_splits[split],
            split_name=split,
            id_cols=id_cols,
            include_params=include_params,
            keep_cols=keep_cols,
        )

        if save_mode == "per_split":
            split_path = out_parquet_path.with_name(
                f"{out_parquet_path.stem}__"
                f"{split}{out_parquet_path.suffix or '.parquet'}"
            )
            split_path.parent.mkdir(parents=True, exist_ok=True)
            df_me.to_parquet(split_path, index=False, compression=compression)
            # optionally avoid keeping in memory on huge runs
            if return_df:
                parts.append(df_me)
        else:
            parts.append(df_me)

    if save_mode == "single":
        final = pd.concat(parts,
                          ignore_index=True) if parts else pd.DataFrame()
        out_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        final.to_parquet(out_parquet_path, index=False,
                         compression=compression)
        print(f"[SAVE] Wrote marginal emissions to {out_parquet_path} "
              f"(rows={len(final):,})")
        return final if return_df else None
    else:
        print(f"[SAVE] Wrote per-split Parquet files next to"
              f" {out_parquet_path}")
        if return_df:
            return pd.concat(parts,
                             ignore_index=True) if parts else pd.DataFrame()
        return None


# ────────────────────────────────────────────────────────────────────────────
# RUNNING MODELS — Runners & Orchestrators
# ────────────────────────────────────────────────────────────────────────────
#
# PURPOSE:
#   End-to-end single-run execution over {train, validation, test}:
#     - Compute a run signature and de-duplicate via rotating CSV index
#     - Evaluate per-group metrics + energy weights
#     - Attach pooled CO₂ metrics and finite-difference ME diagnostics
#     - Write a one-row summary via rotating CSV logging
#
# FUNCTIONS:
#   - run_regressor_model
#   - regressor_orchestrator
#
# NOTES:
#   - The orchestrator shares one model_id_hash across all splits.
#   - Uses `save_summary_to_rotating_csv` for lock-safe, append-only logging.
# ────────────────────────────────────────────────────────────────────────────

def run_regressor_model(
        user_pipeline: Pipeline,
        x_df: pd.DataFrame,
        y_df: pd.Series | pd.DataFrame,
        split_name: str,
        extra_info: dict | None = None,
        return_model: bool = False,
        random_state: int = 12,
        interval_hours: float = 0.5,
        *,
        model_id_hash: str | None = None,
        params_json_str: str | None = None,
) -> tuple[pd.DataFrame, list[str], GroupwiseRegressor | dict]:
    """
    Run a pipeline on one split, compute per-group metrics, attach energy
    weights, and compute diagnostics (pooled CO₂ fit + finite-difference ME
    checks).

    Parameters
    ----------
    user_pipeline : Pipeline
        Full pipeline [FeatureAddition → (Binner) → GroupwiseRegressor].
    x_df : pd.DataFrame
        Features for the split.
    y_df : pd.Series or single-column pd.DataFrame
        Target for the split.
    split_name : {"train","validation","test"}
        Which split to run.
    extra_info : dict, optional
        Extra metadata to stamp onto the output rows.
    return_model : bool, default False
        If True, returns the final estimator as the 3rd tuple item; otherwise
        returns extras dict.
    random_state : int, default 12
        Random seed for reproducibility.
    interval_hours : float, default 0.5
        Duration represented by each row (half-hourly = 0.5).
    model_id_hash : str, optional
        If provided, stamp this precomputed run-level hash (recommended).
        If None, a local signature is computed (useful for ad-hoc calls).
    params_json_str : str, optional
        Pre-rendered pipeline params JSON to stamp; if None, it is computed.

    Returns
    -------
    metrics_df : pd.DataFrame
        Per-group metrics with added 'energy_MWh' and metadata columns.
    x_cols_used : list[str]
        Regressor feature names used by the GroupwiseRegressor
        (x_vars + fe_vars).
    model_or_extras : GroupwiseRegressor | dict
        If return_model=True → the fitted final estimator; else a dict of
        diagnostics.
    """
    np.random.seed(random_state)

    for col in x_df.columns:
        dt = x_df[col].dtype
        if str(dt).startswith(("uint", "UInt")):
            x_df[col] = x_df[col].astype("int64")

    if split_name not in ("train", "validation", "test"):
        raise ValueError(f"split_name must be 'train', 'validation', or "
                         f"'test' (got {split_name!r})")

    X = x_df.copy()
    if isinstance(y_df, pd.DataFrame):
        if y_df.shape[1] != 1:
            raise ValueError("y_df must be a Series or single-column "
                             "DataFrame.")
        y_ser = y_df.iloc[:, 0]
    else:
        y_ser = y_df

    # Use provided model_id_hash (from orchestrator) or compute a local one
    if model_id_hash is None:
        model_id_hash, _ = signature_for_run(
            user_pipeline,
            x_columns=list(X.columns),
            y=y_ser,
            random_state=random_state,
            # local call; orchestrator passes a shared hash
            eval_splits=(split_name,),
            compute_test=False,
            extra_info=extra_info,
        )

    if params_json_str is None:
        params_json_str = json.dumps(
            user_pipeline.get_params(deep=True),
            sort_keys=True, separators=(",", ":"), default=str
        )

    extras: dict[str, Any] = {}

    if split_name == "train":
        # Fit → metrics from regressor
        _ = user_pipeline.fit_transform(X, y_ser)
        model = user_pipeline._final_estimator  # type: ignore[attr-defined]
        metrics_df = model.get_metrics(summarise=True).reset_index(drop=True)

        # Canonicalize group col to "group"
        if model.group_col in metrics_df.columns:
            metrics_df = metrics_df.rename(
                columns={model.group_col: "group"})
        elif "group" not in metrics_df.columns:
            metrics_df = metrics_df.rename(
                columns={metrics_df.columns[0]: "group"})

        # Preprocessed rows for weights & diagnostics
        x_tr = _apply_fitted_preprocessing(user_pipeline, X)
        x_tr[model.y_var] = np.asarray(y_ser, dtype=float)

        # Energy weights
        w = _compute_group_energy_weights(
            df=x_tr,
            group_col=model.group_col,
            q_col=model.x_vars[0],
            interval_hours=interval_hours
        ).rename(columns={model.group_col: "group"})
        metrics_df = metrics_df.merge(w, on="group", how="left")

        # Diagnostics (in-sample)
        extras["pooled_co2"] = pooled_co2_metrics(
            model,
            x_tr,
            y_col=model.y_var,
            group_col=model.group_col
        )
        me_df = model.transform(x_tr)
        fd_df = finite_difference_me_metrics(
            df=me_df,
            time_col=(
                "timestamp" if "timestamp" in me_df.columns else "time_id"),
            q_col=model.x_vars[0],
            y_col=model.y_var,
            me_col="ME",
            group_keys=[k for k in ("city",) if k in me_df.columns],
        )
        extras["fd_me_by_city"] = fd_df.to_dict(
            orient="records") if not fd_df.empty else []
        extras["fd_me_pooled"] = (
            fd_df.loc[fd_df["city"] == "ALL"].iloc[0].to_dict()
            if (not fd_df.empty and "city" in fd_df.columns
                and "ALL" in fd_df["city"].values)
            else (fd_df.sort_values(
                "n_pairs", ascending=False
                ).iloc[0].to_dict() if not fd_df.empty else {})
        )

    else:
        # Use fitted preprocessing + regressor
        model = user_pipeline._final_estimator  # type: ignore[attr-defined]
        x_tr = _apply_fitted_preprocessing(user_pipeline, X)

        if model.group_col not in x_tr.columns:
            raise KeyError(
                f"Group column '{model.group_col}' is missing after transform."
                f" Ensure your binner outputs it."
            )

        x_tr[model.y_var] = np.asarray(y_ser, dtype=float)

        # Per-group metrics
        metrics_df = evaluate_on_split(model, x_tr)

        # Energy weights
        w = _compute_group_energy_weights(
            df=x_tr,
            group_col=model.group_col,
            q_col=model.x_vars[0],
            interval_hours=interval_hours
        ).rename(columns={model.group_col: "group"})
        metrics_df = metrics_df.merge(w, on="group", how="left")

        # Out-of-sample diagnostics
        extras["pooled_co2"] = pooled_co2_metrics(
            model,
            x_tr,
            y_col=model.y_var,
            group_col=model.group_col
        )
        me_df = model.transform(x_tr)
        fd_df = finite_difference_me_metrics(
            df=me_df,
            time_col=(
                "timestamp" if "timestamp" in me_df.columns else "time_id"
            ),
            q_col=model.x_vars[0],
            y_col=model.y_var,
            me_col="ME",
            group_keys=[k for k in ("city",) if k in me_df.columns],
        )
        extras["fd_me_by_city"] = fd_df.to_dict(
            orient="records") if not fd_df.empty else []
        extras["fd_me_pooled"] = (
            fd_df.loc[fd_df["city"] == "ALL"].iloc[0].to_dict()
            if (not fd_df.empty and "city" in fd_df.columns
                and "ALL" in fd_df["city"].values)
            else (fd_df.sort_values(
                "n_pairs",
                ascending=False
                ).iloc[0].to_dict() if not fd_df.empty else {})
        )

    # Stamp metadata
    metrics_df["data_split"] = split_name
    metrics_df["model_id_hash"] = model_id_hash
    metrics_df["random_state"] = random_state
    metrics_df["pipeline_params_json"] = params_json_str
    metrics_df["log_time"] = datetime.now().isoformat()

    model = user_pipeline._final_estimator  # type: ignore[attr-defined]
    metrics_df["x_columns_used"] = ",".join(model.x_vars + model.fe_vars)
    for k, v in (extra_info or {}).items():
        metrics_df[k] = v

    x_cols_used = model.x_vars + model.fe_vars
    print(f"[LOG] {len(metrics_df)} rows for split={split_name},"
          f" model_id={model_id_hash}, random_state={random_state}")

    return (metrics_df, x_cols_used, model) if return_model else (metrics_df,
                                                                  x_cols_used,
                                                                  extras)


def regressor_orchestrator(
        user_pipeline: Pipeline,
        x_splits: dict,
        y_splits: dict,
        log_csv_path: str | None = "marginal_emissions_log.csv",   # legacy
        extra_info: dict | None = None,
        force_run: bool = False,
        force_overwrite: bool = False,
        random_state: int = 12,
        group_col_name: str = "group",
        interval_hours: float = 0.5,
        eval_splits: tuple[str, ...] | None = None,
        compute_test: bool = False,
        # rotating CSV
        results_dir: str | None = None,
        file_prefix: str | None = None,
        max_log_mb: int = 95,
        fsync: bool = True,
) -> pd.DataFrame | None:
    """
    Fit/evaluate a pipeline on train/validation/test, summarise metrics, and
    append to a CSV log.

    Parameters
    ----------
    user_pipeline : Pipeline
        Full pipeline, typically [FeatureAddition → Binner →
        GroupwiseRegressor].
    x_splits : dict
        Must include "train" and "validation". Include "test" iff
        compute_test=True.
    y_splits : dict
        Target splits with the same keys as x_splits.  Must include "train"
        and "validation". Include "test" iff compute_test=True.
     log_csv_path : str, optional
        Legacy path; used only to infer default results_dir/file_prefix if
        those are None.
    extra_info : dict, optional
        Extra metadata to stamp onto per-split logs
        (propagates into `run_regressor_model`).
    force_run : bool, default=False
        If False and an identical model signature was previously logged,
        skip this run.
    force_overwrite : bool, default=False
        If True, allows re-logging the same model_id_hash
        (previous rows are NOT removed here use
        `save_summary_to_csv(..., force_overwrite=True)` for row replacement).
    random_state : int, default=12
        Random seed recorded in the model signature and summary.
    group_col_name : str, default="group"
        Canonical group column name used by `summarise_metrics_logs`
        for nested metrics.

    Returns
    -------
    pd.DataFrame or None
        One-row summary DataFrame if the run executes; None if skipped due to
        prior identical log.

    Notes
    -----
    - The model signature (hash) is computed from pipeline parameters,
    feature columns, target name(s), random_state, and any `extra_info`.
    If unchanged and `force_run=False`, the run is skipped.
    - `x_columns` recorded in the summary are taken from the **train** split’s
    evaluation result.
    """
    # in regressor_orchestrator before signature_for_run(...)
    if eval_splits is None:
        eval_splits = ("train",
                       "validation",
                       "test") if compute_test else ("train", "validation")
    compute_test = ("test" in eval_splits)
    # ← keep hash consistent with actual splits

    # One signature for the whole run (based on TRAIN)
    model_key, sig = signature_for_run(
        user_pipeline,
        x_columns=list(x_splits["train"].columns),
        y=y_splits["train"],
        random_state=random_state,
        eval_splits=eval_splits,
        compute_test=compute_test,
        extra_info=extra_info,
    )

    # Resolve dir + prefix (fallback to legacy path)
    if results_dir is None or file_prefix is None:
        base = Path(log_csv_path or "marginal_emissions_log.csv")
        inferred_dir = base.parent if str(base.parent) != "" else Path(".")
        inferred_prefix = base.stem
        results_dir = results_dir or str(inferred_dir)
        file_prefix = file_prefix or inferred_prefix

    # De-dupe via index
    if not force_run and not force_overwrite:
        if is_model_logged_rotating_csv(model_key, results_dir, file_prefix):
            print(f"[SKIP] Model already logged (hash: {model_key})")
            return None

    # Precompute params JSON once (consistent across splits)
    params_json_str = json.dumps(
        user_pipeline.get_params(deep=True),
        sort_keys=True, separators=(",", ":"), default=str
    )

    logs, pooled_extras, fd_extras = {}, {}, {}
    x_cols_used: list[str] | None = None

    for split in eval_splits:
        metrics_df, x_cols_used, extras = run_regressor_model(
            user_pipeline=user_pipeline,
            x_df=x_splits[split],
            y_df=y_splits[split],
            split_name=split,
            extra_info=extra_info,
            return_model=False,
            random_state=random_state,
            interval_hours=interval_hours,
            model_id_hash=model_key,          # shared ID across splits
            params_json_str=params_json_str,  # shared params JSON
        )
        logs[split] = metrics_df
        pooled_extras[split] = extras.get("pooled_co2", {})
        fd_extras[split] = extras.get("fd_me_pooled", {})

    summary_df = summarise_metrics_logs(
        train_logs=logs["train"],
        val_logs=logs["validation"],
        test_logs=logs.get("test"),
        user_pipeline=user_pipeline,
        x_columns=x_cols_used or [],
        random_state=random_state,
        group_col_name=group_col_name,
        pooled_metrics_by_split=pooled_extras,
        fd_me_metrics_by_split=fd_extras,
    )

    save_summary_to_rotating_csv(
        summary_df,
        results_dir=results_dir,
        file_prefix=file_prefix,
        max_mb=max_log_mb,
        force_overwrite=force_overwrite,
        fsync=fsync,
    )
    return summary_df


# ────────────────────────────────────────────────────────────────────────────
# RUNNING MODELS — Grid Search
# ────────────────────────────────────────────────────────────────────────────
#
# PURPOSE:
#   Sweep over binner choices and x/fe combinations, build pipelines, and log
#   one summary row per configuration. Supports MPI-based distribution.
#
# FUNCTIONS:
#   - run_grid_search
#   - run_grid_search_auto
#
# NOTES:
#   - `run_grid_search_auto` partitions the grid across MPI ranks (stride or
#     chunked), seeds RNGs per-rank, and reuses rotating log machinery.
# ────────────────────────────────────────────────────────────────────────────


def run_grid_search(
        base_feature_pipeline: Pipeline,
        regressor_cls,
        regressor_kwargs: dict,
        grid_config: list[dict],
        x_splits: dict,
        y_splits: dict,
        log_path: str | None,  # legacy; optional now
        global_extra_info: dict | None = None,
        force_run: bool = False,
        force_overwrite: bool = False,
        base_feature_pipeline_name: str = "BaseFeaturePipeline",
        eval_splits: tuple[str, ...] = ("train", "validation"),
        results_dir: str | None = None,
        file_prefix: str | None = None,
        max_log_mb: int = 95,
        fsync: bool = True,
) -> None:
    """
    Execute a series of [features → binner → regressor] runs and log one
    summary row per config.

    Parameters
    ----------
    base_feature_pipeline : Pipeline
        Preprocessing steps applied before binning. This object is cloned per
        run to avoid state leakage.
    regressor_cls : type
        Estimator class to instantiate for the final step
        (e.g., GroupwiseRegressor).
    regressor_kwargs : dict
        Baseline kwargs for the regressor. Per-config overrides from
        `grid_config` are merged on top.
        IMPORTANT: This function will not mutate the caller's dict.
    grid_config : list of dict
        Each item should contain:
            - "binner_class": class (e.g., MultiQuantileBinner or
            MultiMedianBinner)
            - "binner_kwargs": dict of init args for the binner
            - "label": str label for printing/logging (optional)
            - Optional: "x_vars", "fe_vars" to override the regressor’s
            predictors per-config
            - Optional: anything else you want echoed into `extra_info`
    x_splits, y_splits : dict
        Dicts keyed by {"train","validation","test"} with DataFrames/Series
        for each split.
    log_path : str
        CSV path where each successful config appends one summary row.
    global_extra_info : dict, optional
        Extra metadata stamped into each run’s logs.
    force_run, force_overwrite : bool
        Passed through to `regressor_orchestrator`.
    base_feature_pipeline_name : str, default "BaseFeaturePipeline"
        Step name used for the features sub-pipeline.

    Returns
    -------
    None
        Prints progress and writes rows to `log_path`. Skips silently
        (with a message) if a config is already logged and `force_run=False`.

    Notes
    -----
    - We clone `base_feature_pipeline` per run to avoid cross-config state
    `sharing.
    - If a binner provides `group_col_name` and the regressor does not specify
    `group_col`,
      we set the regressor’s `group_col` to match.
    - If a config provides `x_vars`/`fe_vars`, they override the baseline
    `regressor_kwargs`.
    """
    missing_x = [s for s in eval_splits if s not in x_splits]
    missing_y = [s for s in eval_splits if s not in y_splits]
    if missing_x or missing_y:
        raise KeyError(f"Missing splits: X{missing_x} Y{missing_y}")

    total = len(grid_config)
    for i, raw_config in enumerate(grid_config, start=1):
        config = dict(raw_config)
        binner_class = config["binner_class"]
        binner_kwargs = dict(config.get("binner_kwargs", {}))
        label = config.get("label", binner_class.__name__)

        reg_kwargs = dict(regressor_kwargs)
        if "x_vars" in config:
            reg_kwargs["x_vars"] = list(config["x_vars"])
        if "fe_vars" in config:
            reg_kwargs["fe_vars"] = list(config["fe_vars"])
        reg_kwargs["random_state"] = reg_kwargs.get("random_state", 12)

        binner_group_col = binner_kwargs.get("group_col_name")
        if binner_group_col and "group_col" not in reg_kwargs:
            reg_kwargs["group_col"] = binner_group_col

        try:
            features_step = clone(base_feature_pipeline)
        except Exception:
            features_step = base_feature_pipeline

        binner = binner_class(**binner_kwargs)
        regressor = regressor_cls(**reg_kwargs)

        full_pipeline = Pipeline([
            (base_feature_pipeline_name, features_step),
            (binner_class.__name__, binner),
            (regressor_cls.__name__, regressor),
        ])

        extra_info = {
            "binner_class": binner_class.__name__,
            "binner_params": binner_kwargs,
            "regressor_params": reg_kwargs,
            "grid_label": label,
            **(global_extra_info or {}),
        }

        rank_tag = ""
        try:
            _, rank, size = _mpi_context()
            rank_tag = f"[R{rank}/{max(size-1,0)}] "
        except Exception:
            pass
        print(f"\n{rank_tag}[GRID {i}/{total}] {label}")

        try:
            summary_df = regressor_orchestrator(
                user_pipeline=full_pipeline,
                x_splits=x_splits,
                y_splits=y_splits,
                log_csv_path=log_path,            # legacy OK
                extra_info=extra_info,
                force_run=force_run,
                force_overwrite=force_overwrite,
                random_state=reg_kwargs["random_state"],
                eval_splits=eval_splits,
                # NEW
                results_dir=results_dir,
                file_prefix=file_prefix,
                max_log_mb=max_log_mb,
                fsync=fsync,
                )
            if summary_df is not None:
                print(f"[GRID] Logged: {label}")
            else:
                print(f"[GRID] Skipped (already logged): {label}")
        except Exception as e:
            print(f"[GRID] ERROR in '{label}': {type(e).__name__}: {e}")
            continue


def run_grid_search_auto(
        base_feature_pipeline,
        regressor_cls,
        regressor_kwargs: dict,
        grid_config: list[dict],
        x_splits: dict,
        y_splits: dict,
        *,
        # logging/rotation knobs
        results_dir: str,
        file_prefix: str,
        max_log_mb: int = 95,
        # set True on HPC
        fsync: bool = False,
        # orchestration
        base_feature_pipeline_name: str = "FeatureAdditionPipeline",
        eval_splits: tuple[str, ...] = ("train", "validation"),
        force_run: bool = False,
        force_overwrite: bool = False,
        distribute: str = "auto",         # "auto" | "mpi" | "single"
        dist_mode: str = "stride",        # "stride" | "chunked"
        seed: int = 12,
) -> None:
    """
    Single-node or MPI-parallel grid search runner.

    - Auto-detects MPI and splits `grid_config` across ranks.
    - Ensures per-rank deterministic RNG via `seed + rank`.
    - Uses rotating CSV logging with per-file & index locks.

    Parameters are passed straight to `run_grid_search`, except we slice
    `grid_config`.

    Parameters
    ----------
    base_feature_pipeline: Pipeline
        The base feature pipeline to use for each config.
    regressor_cls: Type[BaseEstimator]
        The regression model class to use.
    regressor_kwargs: dict
        Keyword arguments to pass to the regression model.
    grid_config: list[dict]
        The grid search configuration to use.
    x_splits: dict
        The input feature splits.
    y_splits: dict
        The target variable splits.
    results_dir: str
        The directory to save results.
    file_prefix: str
        The prefix for result files.
    max_log_mb: int
        The maximum log file size in MB.
    naming: PartNaming | None
        Optional naming scheme for output files.
    fsync: bool
        Whether to fsync log files (for durability).
    base_feature_pipeline_name: str
        The name of the base feature pipeline.
    eval_splits: tuple[str, ...]
        The evaluation splits to use.
    force_run: bool
        Whether to force re-running of existing configs.
    force_overwrite: bool
        Whether to force overwriting of existing results.
    distribute: str
        The distribution strategy to use.
    dist_mode: str
        The distribution mode to use.
    seed: int
        The random seed to use.

    Returns
    -------
    None
        Logs the results of the grid search.
    """
    comm, rank, size = _mpi_context()
    if distribute == "auto":
        distribute = "mpi" if size > 1 else "single"

    # Partition the configs
    local_configs = _distribute_configs(
                        grid_config,
                        rank=rank,
                        size=size,
                        mode=dist_mode) if distribute == "mpi" else grid_config
    if not local_configs:
        if rank == 0:
            print("[GRID] No configs assigned (empty grid or partition).")
        return

    # Per-rank RNG — override/augment existing random_state
    local_reg_kwargs = dict(regressor_kwargs)
    local_reg_kwargs["random_state"] = int(local_reg_kwargs.get(
        "random_state", seed))

    if rank == 0 and distribute == "mpi":
        print(f"[MPI] size={size} → ~"
              f"{len(grid_config)/max(size,1):.1f} configs per rank")
    else:
        if distribute == "mpi":
            print(f"[MPI] rank={rank}/{size-1} assigned "
                  f"{len(local_configs)} configs")

    run_grid_search(
        base_feature_pipeline=base_feature_pipeline,
        regressor_cls=regressor_cls,
        regressor_kwargs=local_reg_kwargs,
        grid_config=local_configs,
        x_splits=x_splits,
        y_splits=y_splits,
        log_path=None,  # legacy path unused when using rotating logs
        global_extra_info={"runner_rank": rank, "runner_size": size},
        force_run=force_run,
        force_overwrite=force_overwrite,
        base_feature_pipeline_name=base_feature_pipeline_name,
        eval_splits=eval_splits,
        results_dir=results_dir,
        file_prefix=file_prefix,
        max_log_mb=max_log_mb,
        fsync=fsync,
    )

    # Optional barrier for neat logs
    try:
        comm.Barrier()
    except Exception:
        pass
    if rank == 0:
        print("[GRID] Completed (all ranks).")


# ────────────────────────────────────────────────────────────────────────────
# RUNNING MODELS — Grid Search Utilities
# ────────────────────────────────────────────────────────────────────────────
#
# PURPOSE:
#   Generate combinatorial grids for x_vars/fe_vars and binners.
#
# FUNCTIONS:
#   - all_nonempty_subsets
#   - get_fe_vars
#   - build_x_fe_combinations_disjoint
#   - build_quantile_grid_configs
#   - build_median_binner_configs
#
# NOTES:
#   - Enforces x_vars and fe_vars to be disjoint when requested.
#   - Labels (e.g., "qbin_5_varA-varB__x_Q-Q2__fe_month-hour") are designed
#     to be human-readable in logs and easy to grep.
# ────────────────────────────────────────────────────────────────────────────

def all_nonempty_subsets(columns: list[str]) -> list[list[str]]:
    """
    All non-empty subsets preserving input order.   `

    Parameters
    ----------
    columns : list[str]
        Input list of column names.

    Returns
    -------
    list[list[str]]
        List of all non-empty subsets of the input columns.
    """
    return [list(c)
            for i in range(1, len(columns) + 1)
            for c in combinations(columns, i)]


def get_fe_vars(
        all_cols: list[str],
        x_vars: list[str]
) -> list[str]:
    """
    Complement of x_vars within all_cols.

    Parameters
    ----------
    all_cols : list[str]
        List of all column names.
    x_vars : list[str]
        List of x variable names.

    Returns
    -------
    list[str]
        List of fixed effect variable names.
    """
    xset = set(x_vars)
    return [c for c in all_cols if c not in xset]


def build_x_fe_combinations_disjoint(
        candidate_x_vars: list[str],
        candidate_fe_vars: list[str],
        x_var_length: int = 2,
        max_fe_len: int | None = None,
        *,
        allow_empty_fe: bool = False,
) -> list[dict[str, Any]]:
    """
    Generate all disjoint non-empty combinations of x_vars and fe_vars.

    Parameters
    ----------
    candidate_x_vars : list of str
        Columns eligible to be used as predictors (x_vars).
    candidate_fe_vars : list of str
        Columns eligible to be used as fixed effects (fe_vars).
    x_var_length : int
        Number of x_vars to include in each combination.
    max_fe_len : int | None
        Maximum number of fe_vars to include in each combination.
    allow_empty_fe : bool
        Whether to allow empty fe_vars in the combinations.

    Returns
    -------
    list of dicts
        Each dict has keys: {'x_vars': [...], 'fe_vars': [...]}
    """
    if x_var_length < 1:
        raise ValueError("x_var_length must be >= 1")
    if len(candidate_x_vars) < x_var_length:
        raise ValueError("Not enough candidate_x_vars for requested "
                         "x_var_length")

    results: list[dict[str, Any]] = []

    x_subsets = [list(c) for c in combinations(candidate_x_vars, x_var_length)]
    fe_pool = [list(c)
               for i in range(0 if allow_empty_fe else 1,
                              len(candidate_fe_vars) + 1)
               for c in combinations(candidate_fe_vars, i)]

    for x_vars in x_subsets:
        for fe_vars in fe_pool:
            if max_fe_len is not None and len(fe_vars) > max_fe_len:
                continue
            if set(x_vars).isdisjoint(fe_vars):
                results.append({"x_vars": x_vars, "fe_vars": list(fe_vars)})
    return results


def build_quantile_grid_configs(
        candidate_binning_vars: list[str],
        candidate_bin_counts: list[int],
        candidate_x_vars: list[str],
        candidate_fe_vars: list[str],
        x_var_length: int = 2,
        binner_extra_grid: dict | list[dict] | None = None,
) -> list[dict[str, Any]]:
    """
    Produce configs for MultiQuantileBinner sweeping:
      - which vars to bin on
      - how many bins
      - x/fe combinations (disjoint from binned vars)
      - optional extra binner kwargs via dict-of-lists or list-of-dicts

    Parameters
    ----------
    candidate_binning_vars : list[str]
        Variables to be binned.
    candidate_bin_counts : list[int]
        Number of bins to create for each variable.
    candidate_x_vars : list[str]
        Variables to use as predictors (x_vars).
    candidate_fe_vars : list[str]
        Variables to use as fixed effects (fe_vars).
    x_var_length : int
        Number of x_vars to include in each combination.
    binner_extra_grid : dict | list[dict] | None
        Optional extra parameters for the binner.

    Returns
    -------
    list[dict[str, Any]]
        A list of configuration dictionaries for the binner.
    """
    if not candidate_binning_vars:
        return []
    if not candidate_bin_counts:
        return []

    def _expand(grid):
        if grid is None:
            return [dict()]
        if isinstance(grid, list):
            return [dict(d) for d in grid]
        if isinstance(grid, dict):
            keys = list(grid.keys())
            vals = [list(v) if isinstance(v,
                                          (list, tuple, set)) else [v]
                    for v in (grid[k] for k in keys)]
            return [dict(zip(keys, combo)) for combo in product(*vals)]
        raise TypeError("binner_extra_grid must be a dict or list of dicts")

    extra_list = _expand(binner_extra_grid)
    configs: list[dict[str, Any]] = []

    # compute once (perf)
    x_fe_grid = build_x_fe_combinations_disjoint(
        candidate_x_vars, candidate_fe_vars, x_var_length=x_var_length
    )

    for bin_vars in all_nonempty_subsets(candidate_binning_vars):
        bset = set(bin_vars)
        for bin_count in candidate_bin_counts:
            if int(bin_count) < 2:
                continue
            bin_spec = {v: int(bin_count) for v in bin_vars}

            for combo in x_fe_grid:
                if not set(combo["x_vars"]).isdisjoint(bset):
                    continue
                for extra in extra_list:
                    binner_kwargs = {"bin_specs": bin_spec, **extra}

                    # label suffix for clarity in logs
                    tag_bits = []
                    pol = extra.get("oob_policy")
                    if pol:
                        tag_bits.append(f"oob{pol}")
                    rate = extra.get("max_oob_rate")
                    if rate is not None:
                        tag_bits.append(f"rate{float(rate):g}")
                    tag = f"__{'_'.join(tag_bits)}" if tag_bits else ""

                    configs.append({
                        "binner_class": MultiQuantileBinner,
                        "binner_kwargs": binner_kwargs,
                        "label": (
                            f"qbin_{bin_count}_{'-'.join(bin_vars)}"
                            f"__x_{'-'.join(combo['x_vars'])}"
                            f"__fe_{'-'.join(combo['fe_vars'])}{tag}"
                        ),
                        "x_vars": combo["x_vars"],
                        "fe_vars": combo["fe_vars"],
                    })
    return configs


def build_median_binner_configs(
        candidate_binning_vars: list[str],
        candidate_x_vars: list[str],
        candidate_fe_vars: list[str],
        x_var_length: int = 2,
        max_fe_len: int | None = None,
        binner_extra_grid: dict | list[dict] | None = None,
) -> list[dict[str, Any]]:
    """
    Produce configs for MultiMedianBinner sweeping subsets of variables and
    x/fe combos.

    Parameters
    ----------
    candidate_binning_vars : list[str]
        Variables to be binned.
    candidate_x_vars : list[str]
        Variables to use as predictors (x_vars).
    candidate_fe_vars : list[str]
        Variables to use as fixed effects (fe_vars).
    x_var_length : int
        Number of x_vars to include in each combination.
    max_fe_len : int | None
        Maximum number of fixed effects to include in each combination.
    binner_extra_grid : dict | list[dict] | None
        Optional extra parameters for the binner.

    Returns
    -------
    list[dict[str, Any]]
        A list of configuration dictionaries for the binner.
    """
    if not candidate_binning_vars:
        return []

    def _expand(grid):
        if grid is None:
            return [dict()]
        if isinstance(grid, list):
            return [dict(d) for d in grid]
        if isinstance(grid, dict):
            keys = list(grid.keys())
            vals = [(v if isinstance(
                v, (list, tuple, set)) else [v])
                    for v in grid.values()]
            return [dict(zip(keys, combo)) for combo in product(*vals)]
        raise TypeError("binner_extra_grid must be a dict or list of dicts")

    extra_list = _expand(binner_extra_grid)

    configs: list[dict[str, Any]] = []
    x_fe_grid = build_x_fe_combinations_disjoint(
        candidate_x_vars,
        candidate_fe_vars,
        x_var_length=x_var_length,
        max_fe_len=max_fe_len
    )

    for bin_vars in all_nonempty_subsets(candidate_binning_vars):
        bset = set(bin_vars)
        for combo in x_fe_grid:
            if not set(combo["x_vars"]).isdisjoint(bset):
                continue
            for extra in extra_list:
                binner_kwargs = {
                    "variables": bin_vars,
                    "group_col_name": "median_group_id",
                    "retain_flags": True,
                    **extra,
                }
                tag_bits = []
                if "retain_flags" in extra:
                    tag_bits.append(f"rf{int(bool(extra['retain_flags']))}")
                for k, v in extra.items():
                    if k == "retain_flags":
                        continue
                    tag_bits.append(f"{k}{v}")
                tag = f"__{'_'.join(tag_bits)}" if tag_bits else ""

                configs.append({
                    "binner_class": MultiMedianBinner,
                    "binner_kwargs": binner_kwargs,
                    "label": (
                        f"median_{'-'.join(bin_vars)}"
                        f"__x_{'-'.join(combo['x_vars'])}"
                        f"__fe_{'-'.join(combo['fe_vars'])}{tag}"
                    ),
                    "x_vars": combo["x_vars"],
                    "fe_vars": combo["fe_vars"],
                })
    return configs