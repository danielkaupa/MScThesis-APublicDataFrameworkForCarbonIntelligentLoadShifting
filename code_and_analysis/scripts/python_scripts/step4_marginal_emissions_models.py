# ─────────────────────────────────────────────────────────────────────────────
# FILE: step4_marginal_emissions_models.py
#
# PURPOSE:
#   Defines custom regression models used in the marginal emissions pipeline.
#   Currently includes a scikit-learn–compatible GroupwiseRegressor, which
#   estimates marginal emission factors via per-group OLS regressions.
#
#
# SECTIONS IN THIS FILE:
#   - Imports
#   - Custom Regressors:
#       * GroupwiseRegressor
#
# RUN REQUIREMENTS:
#   - Python 3.10+ recommended
#   - Dependencies:
#       pandas, numpy, polars, scikit-learn, statsmodels
#
# USAGE:
#   from step4_marginal_emissions_models import GroupwiseRegressor
#   model = GroupwiseRegressor(y_var="tons_co2", x_vars=["Q", "Q_sq"],
#   group_col="k")
#   model.fit(X, y)
#   me = model.predict(X, predict_type="ME")
#
# CLASS LIST:
#   - GroupwiseRegressor
#
# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

# Standard library
import logging
from typing import Any, List, Optional, Union

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import statsmodels.formula.api as smf

import step4_marginal_emissions_scoring as score_util


# ────────────────────────────────────────────────────────────────────────────
# CUSTOM REGRESSORS
# ────────────────────────────────────────────────────────────────────────────
#
# GroupwiseRegressor
# ------------------
# Runs separate OLS regressions per group to estimate marginal emission
# factors. Compatible with scikit-learn’s fit/transform/predict API,
# using statsmodels under the hood.
#
# Main methods:
#   - fit(X, y)          → Train per-group OLS regressions
#   - transform(X)       → Append α1, α2, and marginal effects (ME) to data
#   - predict(X, type)   → Predict marginal effects (default) or CO₂
#   - get_metrics()      → Return per-group performance metrics
#
# Notes:
#   - Expects pandas DataFrame input
#   - Handles categorical fixed effects (month, hour, etc.) explicitly
#   - Tracks group-level metrics if enabled
# ────────────────────────────────────────────────────────────────────────────

class GroupwiseRegressor(BaseEstimator, TransformerMixin):
    """
    Runs separate OLS regressions in each group and computes marginal emission
    factors.

    For each group k, we fit:
        y_t = α₁ₖ · x₁_t + α₂ₖ · x₂_t + Σ β_i·C(f_i)_t + ε_t
    and compute the marginal effect:
        ME_t = ∂y_t/∂x₁_t = α₁ₖ + 2·α₂ₖ·x₁_t.

    Parameters
    ----------
    y_var : str
        Target column name (e.g. 'tons_co2').
    x_vars : List[str]
        Predictor columns; first is Q, second is Q².
    fe_vars : List[str], optional
        Categorical fixed-effect columns.
    group_col : str
        Column with integer group IDs.
    min_group_size : int
        Minimum observations per group to run regression.
    track_metrics : bool
        If True, store per-group models and metrics.
    verbose : bool
        If True, log progress and metrics.

    Attributes
    ----------
    group_models_ : dict
        Fitted statsmodels results per group (if track_metrics=True).
    group_metrics_ : dict
        Computed metrics per group (if track_metrics=True).
    """
    def __init__(
        self,
        y_var: str = "tons_co2",
        x_vars: List[str] = ["total_generation", "total_generation_sqrd"],
        fe_vars: Optional[List[str]] = None,
        group_col: str = "k",
        min_group_size: int = 10,
        track_metrics: bool = True,
        verbose: bool = True,
        random_state: int | None = 12,
    ):
        """
        Initialize the GroupwiseRegressor.

        Parameters
        ----------
        y_var : str
            Target column name (e.g. 'tons_co2').
        x_vars : List[str]
            Predictor columns; first is Q, second is Q².
        fe_vars : List[str], optional
            Categorical fixed-effect columns.
        group_col : str
            Column with integer group IDs.
        min_group_size : int
            Minimum observations per group to run regression.
        track_metrics : bool
            If True, store per-group models and metrics.
        verbose : bool
            If True, log progress and metrics.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        None
        """
        if not isinstance(y_var, str):
            raise TypeError("y_var must be a string")
        if not isinstance(
            x_vars,
            list
        ) or not x_vars or not all(isinstance(
            v,
            str
        ) for v in x_vars):
            raise TypeError("x_vars must be a non-empty list of strings")
        if fe_vars is not None and (not isinstance(
            fe_vars,
            list) or not all(isinstance(
                v, str) for v in fe_vars)):
            raise TypeError("fe_vars must be a list of strings or None")
        if not isinstance(group_col, str):
            raise TypeError("group_col must be a string")
        if not isinstance(min_group_size,
                          int) or min_group_size < 1:
            raise ValueError("min_group_size must be a positive integer")
        if not isinstance(track_metrics, bool):
            raise TypeError("track_metrics must be a boolean")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")

        self.y_var = y_var
        self.x_vars = x_vars
        self.fe_vars = fe_vars or []
        self.group_col = group_col
        self.min_group_size = min_group_size
        self.track_metrics = track_metrics
        self.verbose = verbose
        self.random_state = random_state

        # Always allocate models so transform/predict work
        # regardless of tracking
        self.group_models_: dict[Any, Any] = {}
        self.group_metrics_: dict[Any, dict[str, float]] = {}

    def fit(
            self: "GroupwiseRegressor",
            X: pd.DataFrame,
            y: pd.Series | None = None
    ) -> GroupwiseRegressor:
        """
        Fit the regressor to the data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series | None
            Target vector.

        Returns
        -------
        self : GroupwiseRegressor
            Fitted regressor instance.

        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if y is None:
            raise ValueError("y must be provided for fitting")
        if len(X) != len(y):
            raise ValueError(f"X and y have different lengths: "
                             f"{len(X)} != {len(y)}")

        df = X.copy()
        df[self.y_var] = np.asarray(y).reshape(-1)

        # Validate required columns BEFORE any casting/grouping
        required = set(self.x_vars + self.fe_vars + [self.group_col])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for fit: {missing}")

        # Avoid uints in formula design matrix
        uint_cols = [c for c in df.columns if str(df[c].dtype
                                                  ).startswith(
                                                      ("uint", "UInt"))]
        if uint_cols:
            df[uint_cols] = df[uint_cols].astype("int64")

        # Cast common FE columns (only if they exist)
        if "month" in self.fe_vars and "month" in df.columns:
            df["month"] = pd.Categorical(df["month"].astype(int),
                                         categories=range(1, 13),
                                         ordered=True)
        if "hour" in self.fe_vars and "hour" in df.columns:
            df["hour"] = pd.Categorical(df["hour"].astype(int),
                                        categories=range(24),
                                        ordered=True)
        if "day_of_week" in self.fe_vars and "day_of_week" in df.columns:
            df["day_of_week"] = pd.Categorical(df["day_of_week"].astype(int),
                                               categories=range(1, 8),
                                               ordered=True)
        if "week_of_year" in self.fe_vars and "week_of_year" in df.columns:
            df["week_of_year"] = pd.Categorical(df["week_of_year"].astype(int),
                                                categories=range(1, 54),
                                                ordered=True)
        if "half_hour" in self.fe_vars and "half_hour" in df.columns:
            df["half_hour"] = pd.Categorical(df["half_hour"].astype(int),
                                             categories=range(0, 48),
                                             ordered=True)

        # Reset state
        self.group_models_.clear()
        self.group_metrics_.clear()
        self._fitted_groups: list[Any] = []

        # Fit per group
        for grp, df_grp in df.groupby(self.group_col, sort=True):
            n = len(df_grp)
            if n < self.min_group_size:
                if self.verbose:
                    logging.warning(f"Skipping group {grp!r}: only "
                                    f"{n} < {self.min_group_size}")
                continue

            reg = " + ".join(self.x_vars)
            fe = " + ".join(f"C({f})" for f in self.fe_vars)
            formula = f"{self.y_var} ~ {reg}" + (f" + {fe}" if fe else "")

            model = smf.ols(formula, data=df_grp).fit()
            self.group_models_[grp] = model
            self._fitted_groups.append(grp)

            if self.track_metrics:
                preds = model.predict(df_grp)
                rmse = float(np.sqrt(np.mean(
                    (preds - df_grp[self.y_var]) ** 2)))
                mae = float(np.mean(np.abs(preds - df_grp[self.y_var])))
                mape = float(score_util.mean_absolute_percentage_error(
                    df_grp[self.y_var], preds))
                self.group_metrics_[grp] = {
                    "r2": float(model.rsquared),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "n_obs": int(n),
                }

        if not self._fitted_groups:
            raise ValueError("No valid groups found for fitting.")
        return self

    def _cast_fe_like_fit(self, df: pd.DataFrame) -> pd.DataFrame:
        # Mirror FE casting used in fit (only where columns exist)
        if "month" in self.fe_vars and "month" in df.columns:
            df["month"] = pd.Categorical(df["month"].astype(int),
                                         categories=range(1, 13),
                                         ordered=True)
        if "hour" in self.fe_vars and "hour" in df.columns:
            df["hour"] = pd.Categorical(df["hour"].astype(int),
                                        categories=range(24),
                                        ordered=True)

        if "day_of_week" in self.fe_vars and "day_of_week" in df.columns:
            df["day_of_week"] = pd.Categorical(df["day_of_week"].astype(int),
                                               categories=range(1, 8),
                                               ordered=True)
        if "week_of_year" in self.fe_vars and "week_of_year" in df.columns:
            df["week_of_year"] = pd.Categorical(df["week_of_year"].astype(int),
                                                categories=range(1, 54),
                                                ordered=True)
        if "half_hour" in self.fe_vars and "half_hour" in df.columns:
            df["half_hour"] = pd.Categorical(df["half_hour"].astype(int),
                                             categories=range(0, 48),
                                             ordered=True)
        return df

    def transform(
            self: GroupwiseRegressor,
            X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply groupwise OLS and compute marginal effects ME_t.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain y_var, x_vars, fe_vars, and group_col.

        Returns
        -------
        pd.DataFrame
            Original rows plus 'alpha1', 'alpha2', and 'ME'.

        Raises
        ------
        TypeError
            If X is not a pandas DataFrame.
        ValueError
            If required columns missing or no group qualifies.
        """
        if not self.group_models_:
            raise RuntimeError("GroupwiseRegressor must be fit "
                               "before transform/predict.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame")

        df = X.copy()
        req = set(self.x_vars + self.fe_vars + [self.group_col])
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in transform input: {missing}")

        # Basic numeric sanity for Q (x_vars[0]) and optionally x_vars[1]
        for c in self.x_vars[:2]:
            if c in df.columns and not np.issubdtype(
                    np.asarray(df[c]).dtype, np.number):
                raise TypeError(f"Column '{c}' must be numeric for ME"
                                f" computation.")

        df = self._cast_fe_like_fit(df)

        df["alpha1"] = np.nan
        df["alpha2"] = np.nan
        df["ME"] = np.nan

        for grp, df_grp in df.groupby(self.group_col, sort=True):
            model = self.group_models_.get(grp)
            if model is None:
                continue
            a1 = model.params.get(self.x_vars[0], np.nan)
            a2 = model.params.get(self.x_vars[1], 0.0)
            idx = df_grp.index
            df.loc[idx, "alpha1"] = a1
            df.loc[idx, "alpha2"] = a2
            df.loc[idx, "ME"] = a1 + 2.0 * a2 * df_grp[self.x_vars[0]]

        return df

    def predict(
            self: GroupwiseRegressor,
            X: pd.DataFrame,
            predict_type: str = "ME"
    ) -> pd.Series:
        """
        Predict marginal effects (default) or CO2 for each row in X using
        fitted group models.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain x_vars, fe_vars, and group_col.
        predict_type : {"ME","y"}, default "ME"
            "ME": return α1 + 2*α2*Q
            "y" : return model.predict(...) (CO2)

        Returns
        -------
        pd.Series
            Predictions aligned to X.index.
        """

        if not self.group_models_:
            raise RuntimeError("GroupwiseRegressor must be fit before "
                               "predict().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if predict_type not in {"ME", "y"}:
            raise ValueError("predict_type must be 'ME' or 'y'.")

        df = X.copy()
        req = set(self.x_vars + self.fe_vars + [self.group_col])
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in predict input: {missing}")

        if predict_type == "ME":
            q = self.x_vars[0]
            if not np.issubdtype(np.asarray(df[q]).dtype, np.number):
                raise TypeError(f"Column '{q}' must be numeric to compute ME.")

        df = self._cast_fe_like_fit(df)

        out = pd.Series(index=df.index, dtype=float)
        for grp, df_grp in df.groupby(self.group_col, sort=True):
            model = self.group_models_.get(grp)
            if model is None:
                continue
            if predict_type == "y":
                preds = model.predict(df_grp)
            else:
                a1 = model.params.get(self.x_vars[0], np.nan)
                a2 = model.params.get(self.x_vars[1], 0.0)
                preds = a1 + 2.0 * a2 * df_grp[self.x_vars[0]]
            out.loc[df_grp.index] = preds
        return out

    def get_metrics(
            self: GroupwiseRegressor,
            summarise: bool = True
    ) -> Union[dict, pd.DataFrame]:
        """
        Get the metrics for each group.

        Parameters
        ----------
        summarise : bool, default=True
            If True, return a summary DataFrame; otherwise return raw metrics
            dict.

        Returns
        -------
        dict or pd.DataFrame
            If summarise=True, returns a DataFrame with group metrics.
            If False, returns the raw metrics dictionary.

        Raises
        ------
        RuntimeError
            If track_metrics was not set to True during initialization.
        """
        if not self.track_metrics:
            raise RuntimeError("Metrics tracking is disabled. "
                               "Set track_metrics=True to enable.")
        if summarise:
            df = pd.DataFrame.from_dict(self.group_metrics_,
                                        orient="index")
            df.index.name = self.group_col
            df.reset_index(inplace=True)
            return df
        return self.group_metrics_
