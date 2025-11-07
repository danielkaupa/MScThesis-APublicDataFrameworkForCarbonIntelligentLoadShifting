# FILE: step4_marginal_emissions_scoring.py
#
# PURPOSE:
#   Utility functions for scoring and evaluating marginal emissions models.
#   Provides tools for:
#     - Computing per-group and pooled regression metrics
#     - Comparing predicted marginal emissions (ME) against finite-difference
#       estimates on held-out data
#     - Aggregating metrics with macro, micro, and energy-weighted averaging
#     - Computing standard regression error metrics (RMSE, MAE, MAPE, R², etc.)
#     - Summarising train/val/test logs into a single consolidated record
#
# RUN REQUIREMENTS:
#   - Python 3.10+
#   - Dependencies:
#       numpy
#       pandas
#       scikit-learn
#
# FUNCTION LIST:
#   _compute_group_energy_weights   → Energy totals per group
#   finite_difference_me_metrics    → Compare ME vs. observed slopes
#   macro_micro_means               → Macro/micro averaging
#   mean_absolute_percentage_error  → Robust MAPE
#   mean_metric                     → Safe metric mean (incl. MSE)
#   pooled_co2_metrics              → Pooled CO₂ metrics for GroupwiseRegressor
#   summarise_metrics_logs          → Consolidated summary of per-split logs
#
# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from typing import Any

# ────────────────────────────────────────────────────────────────────────────
# FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────


def _compute_group_energy_weights(
        df: pd.DataFrame,
        group_col: str,
        q_col: str,
        interval_hours: float = 0.5,
) -> pd.DataFrame:
    """
    Aggregate energy weights by group.

    Parameters
    ----------
    df : pd.DataFrame
        Rows for a single split after preprocessing (must contain `group_col`
          and `q_col`).
    group_col : str
        Name of the group id column (e.g., 'median_group_id',
        'quantile_group_id').
    q_col : str
        Name of the demand/quantity column used as Q in the regression
        (usually x_vars[0]).
    interval_hours : float, default 0.5
        Duration represented by each row in hours (half-hourly = 0.5).

    Returns
    -------
    pd.DataFrame
        Columns: [group_col, 'q_sum', 'energy_MWh']
        where energy_MWh = q_sum * interval_hours.
    """
    if group_col not in df.columns:
        raise KeyError(f"'{group_col}' not found in df")
    if q_col not in df.columns:
        raise KeyError(f"'{q_col}' not found in df")
    if not np.issubdtype(np.asarray(df[q_col]).dtype, np.number):
        raise TypeError(f"'{q_col}' must be numeric")
    if interval_hours <= 0:
        raise ValueError("interval_hours must be > 0")

    g = (
        df.groupby(group_col, observed=True)[q_col]
          .sum()
          .rename("q_sum")
          .reset_index()
    )
    g["energy_MWh"] = g["q_sum"] * float(interval_hours)
    return g


def finite_difference_me_metrics(
        df: pd.DataFrame,
        time_col: str = "timestamp",
        q_col: str = "demand_met",
        y_col: str = "tons_co2",
        me_col: str = "ME",
        group_keys: list[str] | tuple[str, ...] = ("city",),
        max_dt: pd.Timedelta = pd.Timedelta("2h"),
        min_abs_dq: float = 1e-6,
) -> pd.DataFrame:
    """
    Compare predicted ME to observed short-horizon slopes s = Δy/ΔQ on
    held-out data.

    For each group in `group_keys`:
      Δy = y_t - y_{t-1}, ΔQ = Q_t - Q_{t-1}, Δt = t - t_{t-1}
      Keep pairs with Δt ≤ max_dt and |ΔQ| ≥ min_abs_dq.
      s_t = Δy / ΔQ, ME_avg = 0.5*(ME_t + ME_{t-1})

    Returns
    -------
    pd.DataFrame
        One row per group and an optional pooled 'ALL' row:
        ['pearson_r','spearman_r','rmse','mae','n_pairs', *group_keys]
    """
    if time_col not in df.columns:
        raise KeyError(f"'{time_col}' not in df")
    required = [q_col, y_col, me_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for c in required:
        if not np.issubdtype(np.asarray(df[c]).dtype, np.number):
            raise TypeError(f"'{c}' must be numeric")
    # ensure datetime for Δt filtering
    dt_series = pd.to_datetime(df[time_col], errors="coerce")
    if dt_series.isna().any():
        raise ValueError(f"Column '{time_col}' contains non-parseable"
                         f" datetimes")
    work = df.copy()
    work[time_col] = dt_series

    def _per_group(gdf: pd.DataFrame) -> dict:
        gdf = gdf.sort_values(time_col).copy()
        gdf["dt"] = gdf[time_col].diff()
        gdf["dQ"] = gdf[q_col].diff()
        gdf["dY"] = gdf[y_col].diff()
        gdf["ME_avg"] = 0.5 * (gdf[me_col] + gdf[me_col].shift(1))

        mask = (
            gdf["dt"].notna() & (gdf["dt"] <= max_dt)
            & gdf["dQ"].notna() & (np.abs(gdf["dQ"]) >= float(min_abs_dq))
            & gdf["dY"].notna() & gdf["ME_avg"].notna()
        )
        sub = gdf.loc[mask, ["dY", "dQ", "ME_avg"]]
        if sub.empty:
            return {"pearson_r": np.nan,
                    "spearman_r": np.nan,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "n_pairs": 0}

        s = sub["dY"].to_numpy(dtype=float) / sub["dQ"].to_numpy(dtype=float)
        me = sub["ME_avg"].to_numpy(dtype=float)
        return {
            "pearson_r": float(pd.Series(s).corr(pd.Series(me))),
            "spearman_r": float(pd.Series(s).corr(pd.Series(me),
                                method="spearman")),
            "rmse": float(root_mean_squared_error(s, me)),
            "mae": float(mean_absolute_error(s, me)),
            "n_pairs": int(len(sub)),
        }

    parts: list[dict] = []
    if group_keys:
        for keys, gdf in work.groupby(list(group_keys),
                                      observed=True,
                                      sort=True):
            row = _per_group(gdf)
            if isinstance(keys, tuple):
                for kname, kval in zip(group_keys, keys):
                    row[kname] = kval
            else:
                row[group_keys[0]] = keys
            parts.append(row)
    else:
        parts.append(_per_group(work) | {"group": "ALL"})

    out = pd.DataFrame(parts)

    # pooled row
    if group_keys and (not out.empty) and out["n_pairs"].sum() > 0:
        tmp = []
        for _, gdf in work.groupby(list(group_keys), observed=True, sort=True):
            gdf = gdf.sort_values(time_col).copy()
            gdf["dt"] = gdf[time_col].diff()
            gdf["dQ"] = gdf[q_col].diff()
            gdf["dY"] = gdf[y_col].diff()
            gdf["ME_avg"] = 0.5 * (gdf[me_col] + gdf[me_col].shift(1))
            mask = (
                gdf["dt"].notna() & (gdf["dt"] <= max_dt)
                & gdf["dQ"].notna() & (np.abs(gdf["dQ"]) >= float(min_abs_dq))
                & gdf["dY"].notna() & gdf["ME_avg"].notna()
            )
            sub = gdf.loc[mask, ["dY", "dQ", "ME_avg"]]
            if not sub.empty:
                tmp.append(
                    pd.DataFrame({
                        "s": (sub["dY"].to_numpy(dtype=float)
                              / sub["dQ"].to_numpy(dtype=float)),
                        "ME_avg": sub["ME_avg"].to_numpy(dtype=float),
                    })
                )
        if tmp:
            pooled = pd.concat(tmp, ignore_index=True)
            pooled_row = {
                "pearson_r": float(pooled["s"].corr(pooled["ME_avg"])),
                "spearman_r": float(pooled["s"].corr(pooled["ME_avg"],
                                                     method="spearman")),
                "rmse": float(root_mean_squared_error(pooled["s"],
                                                      pooled["ME_avg"])),
                "mae": float(mean_absolute_error(pooled["s"],
                                                 pooled["ME_avg"])),
                "n_pairs": int(len(pooled)),
            }
            for k in group_keys:
                pooled_row[k] = "ALL"
            out = pd.concat([out, pd.DataFrame([pooled_row])],
                            ignore_index=True)

    return out


def macro_micro_means(
        df: pd.DataFrame,
        metric: str,
        weight_col: str = "n_obs"
) -> dict:
    """
    Compute macro (simple mean) and micro (weighted by `weight_col`)
    for a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Per-group metrics.
    metric : str
        Column name to average.
    weight_col : str, default "n_obs"
        Column to use as weights for micro average.

    Returns
    -------
    dict
        {"macro": float, "micro": float}
    """
    macro = float(np.nanmean(df[metric].to_numpy(dtype=float)))
    if (weight_col in df) and np.nansum(df[weight_col].to_numpy(dtype=float
                                                                )) > 0:
        micro = float(np.average(df[metric], weights=df[weight_col]))
    else:
        micro = np.nan
    return {"macro": macro, "micro": micro}


def mean_absolute_percentage_error(
        y_true,
        y_pred,
        eps: float = 1e-6
) -> float:
    """
    Compute MAPE robustly - adding small constant to avoid division by zero.

    MAPE = mean(|(y_true - y_pred) / (|y_true| + eps)|) * 100

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.
    eps : float, default 1e-6
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Mean absolute percentage error in percent.
    """
    # true values for y
    yt = np.asarray(y_true, dtype=float)
    # predicted values for y
    yp = np.asarray(y_pred, dtype=float)
    # denominator
    denom = np.abs(yt) + float(eps)
    # compute MAPE
    m = np.abs((yt - yp) / denom)
    # return as percentage (*100)
    return float(np.nanmean(m) * 100.0)


def mean_metric(
        df: pd.DataFrame,
        metric: str
) -> float:
    """
    Compute the mean of a metric, with a special case for MSE derived from
    RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing metric columns.
    metric : {"r2","rmse","mae","mape","n_obs","mse"}
        Metric to aggregate.

    Returns
    -------
    float
        NaN-safe mean of the requested metric.

    Raises
    ------
    KeyError
        If required columns are missing.
    """
    if metric == "mse":
        if "rmse" not in df:
            raise KeyError("Cannot compute 'mse': 'rmse' column missing.")
        return float(np.nanmean(df["rmse"].to_numpy(dtype=float) ** 2))
    if metric not in df:
        raise KeyError(f"Metric '{metric}' not found in DataFrame.")
    return float(np.nanmean(df[metric].to_numpy(dtype=float)))


def pooled_co2_metrics(
        regressor,                  # fitted GroupwiseRegressor
        transformed_df: pd.DataFrame,
        y_col: str | None = None,
        group_col: str | None = None,
) -> dict:
    """
    Compute pooled (all bins together) out-of-sample metrics for CO2.

    Parameters
    ----------
    regressor : GroupwiseRegressor
        Must be fitted; `regressor.group_models_` is used per group.
    transformed_df : pd.DataFrame
        Contains features used by the regressor, the group column, and the
        true y. (Typically validation/test X after feature+binner, with y
        added).
    y_col : str, optional
        Target column name. Defaults to regressor.y_var.
    group_col : str, optional
        Group column name. Defaults to regressor.group_col.

    Returns
    -------
    dict
        {'r2','rmse','mae','mape','n_obs'} (NaNs if insufficient data).
    """
    y_col = y_col or regressor.y_var
    group_col = group_col or regressor.group_col
    if y_col not in transformed_df.columns:
        raise KeyError(f"'{y_col}' not found in transformed_df")
    if group_col not in transformed_df.columns:
        raise KeyError(f"'{group_col}' not found in transformed_df")

    preds = pd.Series(index=transformed_df.index, dtype=float)
    for g, gdf in transformed_df.groupby(group_col, sort=True):
        model = regressor.group_models_.get(g)
        if model is None:
            continue
        preds.loc[gdf.index] = model.predict(gdf)

    mask = preds.notna()
    n_obs = int(mask.sum())
    if n_obs == 0:
        return {"r2": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "n_obs": 0}

    y_true = transformed_df.loc[mask, y_col].to_numpy(dtype=float)
    y_pred = preds.loc[mask].to_numpy(dtype=float)

    # r2 can error for <2 samples or constant y
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan

    return {
        "r2": r2,
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "n_obs": n_obs,
    }


def summarise_metrics_logs(
        train_logs: pd.DataFrame,
        val_logs: pd.DataFrame,
        test_logs: pd.DataFrame | None = None,
        user_pipeline: Pipeline = None,
        x_columns: list | None = None,
        random_state: int = 12,
        group_col_name: str = "group",
        pooled_metrics_by_split: dict[str, dict] | None = None,
        fd_me_metrics_by_split: dict[str, dict] | None = None,
        energy_weight_col: str = "energy_MWh",
) -> pd.DataFrame:
    """
    Summarise per-split, per-group metrics and pipeline metadata into a
    single-row DataFrame.
    This variant allows `test_logs` to be None (can skip test during tuning).

    Parameters
    ----------
    train_logs, val_logs : pd.DataFrame
        Metrics frames for train/validation.
    test_logs : pd.DataFrame or None, default None
        Test metrics; if None, test columns are omitted from the summary.
    user_pipeline : Pipeline
        The fitted or configured pipeline (used for metadata).
    x_columns : list, optional
        Feature names used by the model.
    random_state : int, default 12
        Random seed to record.
    group_col_name : str, default "group"
        Canonical name for the group column.
    pooled_metrics_by_split, fd_me_metrics_by_split : dict, optional
        Optional extra diagnostics keyed by split.
    energy_weight_col : str, default "energy_MWh"
        Column name to use for energy-weighted micro-averages if present.

    Returns
    -------
    pd.DataFrame
        One-row summary. Only includes split columns for the splits provided.
    """
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        cols = list(df.columns)

        # If desired already present, use it
        if group_col_name in cols:
            return df

        # If a plain 'group' exists, rename it to the desired name
        if "group" in cols:
            return df.rename(columns={"group": group_col_name})

        # Known aliases we can rename from
        candidates = [
            "multi_group_id",
            "quantile_group_id",
            "median_group_id",
            "original_quantile_group_id",
            "group_id",
        ]

        # Any *_group_id pattern
        pattern_hits = [c for c in cols if c.endswith("_group_id")]

        # Prefer known aliases in order
        for c in candidates:
            if c in cols:
                return df.rename(columns={c: group_col_name})

        # If exactly one *_group_id exists, use it
        if len(pattern_hits) == 1:
            return df.rename(columns={pattern_hits[0]: group_col_name})

        # Nothing we recognize → fail loudly with context
        found_aliases = [c for c in candidates if c in cols]
        maybe_group = ['group'] if 'group' in cols else []
        expctd_cols = (found_aliases + maybe_group) or (candidates + ['group'])
        raise KeyError(
            f"Could not locate a group column; expected '{group_col_name}'"
            f"or any of {expctd_cols}. Available columns: {cols}"
        )
    splits: dict[str, pd.DataFrame] = {
        "train": _norm(train_logs.copy()),
        "validation": _norm(val_logs.copy()),
    }
    if test_logs is not None:
        splits["test"] = _norm(test_logs.copy())

    required = {"r2", "rmse", "mae", "mape", "n_obs"}
    for name, df in splits.items():
        if df is None or df.empty:
            # allow empty; means NaNs below
            continue
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{name} logs missing metrics: {sorted(missing)}")

    # Choose first non-empty split to source model_id / log_time
    first_non_empty = next(
        (d for d in splits.values() if d is not None and not d.empty),
        next(iter(splits.values()))
    )
    # If still None/empty, create placeholders
    if first_non_empty is None or first_non_empty.empty:
        first_non_empty = pd.DataFrame(
            {"model_id_hash": [np.nan], "log_time": [np.nan]}
        )

    model_id = first_non_empty.get("model_id_hash",
                                   pd.Series([np.nan])).iloc[0]
    log_time = first_non_empty.get("log_time",
                                   pd.Series([np.nan])).iloc[0]
    model_name = (
        user_pipeline._final_estimator.__class__.__name__
        if user_pipeline is not None else ""
    )
    pipeline_steps = (
        list(user_pipeline.named_steps.keys())
        if user_pipeline is not None else []
    )

    summary: dict[str, Any] = {
        "model_id_hash": model_id,
        "random_state": random_state,
        "params_json": json.dumps(
            user_pipeline.get_params(deep=True),
            sort_keys=True,
            separators=(",", ":"),
            default=str
        ) if user_pipeline is not None else "{}",
        "log_time": log_time,
        "model_name": model_name,
        "pipeline_steps": pipeline_steps,
        "pipeline_n_steps": len(pipeline_steps),
        "x_columns": x_columns or [],
        "metrics_by_group": {},
    }

    def _weighted_avg_nan_safe(
            values: np.ndarray,
            weights: np.ndarray
    ) -> float:
        """Compute weighted average ignoring NaNs and nonpositive weights."""
        v = np.asarray(values, dtype=float)
        w = np.asarray(weights, dtype=float)
        mask = (~np.isnan(v)) & (~np.isnan(w)) & (w > 0)
        if not mask.any():
            return float(np.nan)
        return float(np.average(v[mask], weights=w[mask]))

    nested: dict[str, dict] = {}
    for split, df in splits.items():
        if df is None or df.empty:
            # Macro metrics from empty → NaNs; counts → 0
            summary[f"r2_{split}"] = np.nan
            summary[f"rmse_{split}"] = np.nan
            summary[f"mae_{split}"] = np.nan
            summary[f"mape_{split}"] = np.nan
            summary[f"n_obs_{split}"] = 0
            summary[f"mse_{split}"] = np.nan

            summary[f"r2_{split}_micro"] = np.nan
            summary[f"rmse_{split}_micro"] = np.nan
            summary[f"mae_{split}_micro"] = np.nan
            summary[f"mape_{split}_micro"] = np.nan

            summary[f"r2_{split}_energy_micro"] = np.nan
            summary[f"rmse_{split}_energy_micro"] = np.nan
            summary[f"mae_{split}_energy_micro"] = np.nan
            summary[f"mape_{split}_energy_micro"] = np.nan
            summary[f"{energy_weight_col}_{split}_total"] = 0.0

            nested[split] = {}
            continue

        # Macro means (NaN-safe via pandas mean)
        summary[f"r2_{split}"] = float(df["r2"].mean())
        summary[f"rmse_{split}"] = float(df["rmse"].mean())
        summary[f"mae_{split}"] = float(df["mae"].mean())
        summary[f"mape_{split}"] = float(df["mape"].mean())
        summary[f"n_obs_{split}"] = int(df["n_obs"].sum())
        summary[f"mse_{split}"] = float((df["rmse"] ** 2).mean())

        # Micro by n_obs (NaN-robust)
        if df["n_obs"].sum() > 0:
            w = df["n_obs"].to_numpy(dtype=float)
            summary[f"r2_{split}_micro"] = _weighted_avg_nan_safe(
                df["r2"].to_numpy(dtype=float),   w)
            summary[f"rmse_{split}_micro"] = _weighted_avg_nan_safe(
                df["rmse"].to_numpy(dtype=float), w)
            summary[f"mae_{split}_micro"] = _weighted_avg_nan_safe(
                df["mae"].to_numpy(dtype=float),  w)
            summary[f"mape_{split}_micro"] = _weighted_avg_nan_safe(
                df["mape"].to_numpy(dtype=float), w)
        else:
            summary[f"r2_{split}_micro"] = np.nan
            summary[f"rmse_{split}_micro"] = np.nan
            summary[f"mae_{split}_micro"] = np.nan
            summary[f"mape_{split}_micro"] = np.nan

        # Energy-weighted micro (if provided)
        if (energy_weight_col in df.columns) and (
            df[energy_weight_col].fillna(0).sum() > 0
        ):
            wE = df[energy_weight_col].fillna(0).to_numpy(dtype=float)
            summary[f"r2_{split}_energy_micro"] = _weighted_avg_nan_safe(
                df["r2"].to_numpy(dtype=float),   wE)
            summary[f"rmse_{split}_energy_micro"] = _weighted_avg_nan_safe(
                df["rmse"].to_numpy(dtype=float), wE)
            summary[f"mae_{split}_energy_micro"] = _weighted_avg_nan_safe(
                df["mae"].to_numpy(dtype=float),  wE)
            summary[f"mape_{split}_energy_micro"] = _weighted_avg_nan_safe(
                df["mape"].to_numpy(dtype=float), wE)
            summary[f"{energy_weight_col}_{split}_total"] = float(wE.sum())
        else:
            summary[f"r2_{split}_energy_micro"] = np.nan
            summary[f"rmse_{split}_energy_micro"] = np.nan
            summary[f"mae_{split}_energy_micro"] = np.nan
            summary[f"mape_{split}_energy_micro"] = np.nan
            summary[f"{energy_weight_col}_{split}_total"] = 0.0

        # Nest per-group metrics
        cols = ["r2", "rmse", "mae", "mape", "n_obs"]
        if energy_weight_col in df.columns:
            cols.append(energy_weight_col)
        nested[split] = df.set_index(
            group_col_name)[cols].to_dict(orient="index")

    summary["metrics_by_group"] = nested

    # Attach optional diagnostics
    pooled_metrics_by_split = pooled_metrics_by_split or {}
    fd_me_metrics_by_split = fd_me_metrics_by_split or {}
    for split in splits.keys():
        summary[f"pooled_co2_{split}"] = json.dumps(
            pooled_metrics_by_split.get(split, {}))
        summary[f"fd_me_{split}"] = json.dumps(
            fd_me_metrics_by_split.get(split, {}))

    return pd.DataFrame([summary])
