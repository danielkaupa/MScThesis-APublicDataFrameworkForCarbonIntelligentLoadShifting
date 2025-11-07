# ─────────────────────────────────────────────────────────────────────────────
# FILE: step4_marginal_emissions_feature_engineering.py
#
# PURPOSE: Feature engineering and binning transformers used in the marginal
#   emissions pipeline. These classes provide reusable transformations
#   for extracting temporal, quantitative, and share-based features,
#   as well as discretization utilities for grouping observations.
#
#
# SECTIONS IN THIS FILE:
#   - Feature Engineering Transformers:
#       Classes to add new temporal, quantitative, and share features.
#   - Binning Transformers:
#       Classes to quantile- or median-bin variables into discrete groups.
#
# RUN REQUIREMENTS:
#   - Python 3.10+ recommended
#   - Dependencies:
#       pandas, numpy, polars, scikit-learn, statsmodels
#   - Optional:
#       mpi4py (for distributed use in parallel contexts)
#
# USAGE:
#   Import these transformers into model preparation or pipeline scripts:
#       "from filename import AnalysisFeatureAdder"
#
# RUN REQUIREMENTS:
# - Python 3.10+
# - pandas, numpy, matplotlib, seaborn, scikit-learn
# - Designed for integration with the wider marginal emissions pipeline.
#
# CLASS LIST:
# * AnalysisFeatureAdder
# * DateTimeFeatureAdder
# * GenerationShareAdder
# * MultiQuantileBinner
# * MultiMedianBinner
#
# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List


# ────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING TRANSFORMERS
# ────────────────────────────────────────────────────────────────────────────
# Transformers that extract and generate features from raw data.
#   Includes:
#     - Temporal encodings from timestamps
#     - Polynomial/log-transformed demand & emissions features
#     - Relative generation shares
#
# CLASSES:
#   - AnalysisFeatureAdder: add core temporal/quantitative features
#   - DateTimeFeatureAdder: extract rich datetime-based features
#   - GenerationShareAdder: compute generation percentages of total
#
# NOTES:
#   - All transformers follow scikit-learn’s fit/transform API.
#   - Compatible with sklearn Pipelines.
# ────────────────────────────────────────────────────────────────────────────

class AnalysisFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Add core temporal and quantitative features used in the original analysis.

    Adds:
      - time_id:              HH-MM string from `timestamp_col`
      - <Q>_sqrd:             square of `demand_met_col`
      - log_<Q>:              log(demand_met + ε)
      - log_<Q>_sqrd:         (log_<Q>)^2
      - log_<CO2>:            log(tons_co2 + ε) (only if `co2_col` present)
    """

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        demand_met_col: str = "demand_met",
        co2_col: str = "tons_co2",
        epsilon: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        timestamp_col : str
            Name of the datetime column (parseable by pandas).
        demand_met_col : str
            Name of the demand column.
        co2_col : str
            Name of the CO2 column (optional at transform time).
        epsilon : float, default 1e-6
            Small constant to avoid log(0).
        """
        if not isinstance(timestamp_col, str):
            raise ValueError("timestamp_col must be a string")
        if not isinstance(demand_met_col, str):
            raise ValueError("demand_met_col must be a string")
        if not isinstance(co2_col, str):
            raise ValueError("co2_col must be a string")
        if not isinstance(epsilon, (float, int)):
            raise ValueError("epsilon must be a float or int")

        self.timestamp_col = timestamp_col
        self.demand_met_col = demand_met_col
        self.co2_col = co2_col
        self.epsilon = float(epsilon)

    def fit(
            self,
            X,
            y=None
    ) -> AnalysisFeatureAdder:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        for col in [self.timestamp_col, self.demand_met_col]:
            if col not in X.columns:
                raise ValueError(f"Missing required column '{col}' "
                                 f"in input DataFrame")
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(
            self,
            X: pd.DataFrame,
            y: pd.Series | None = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Must contain `timestamp_col` and `demand_met_col`.

        Returns
        -------
        pd.DataFrame
            Copy of X with additional feature columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        df = X.copy()

        for col in [self.timestamp_col, self.demand_met_col]:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}'")

        df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col],
                                                errors="coerce")
        if df[self.timestamp_col].isna().any():
            raise ValueError(f"Column '{self.timestamp_col}' "
                             f"contains non-parseable datetimes")

        # temporal
        df["time_id"] = df[self.timestamp_col].dt.strftime("%H-%M"
                                                           ).astype("string")

        # quantitative
        q = self.demand_met_col
        df[f"{q}_sqrd"] = df[q] ** 2
        df[f"log_{q}"] = np.log(df[q] + self.epsilon)
        df[f"log_{q}_sqrd"] = df[f"log_{q}"] ** 2

        if self.co2_col in df.columns:
            df[f"log_{self.co2_col}"] = np.log(df[self.co2_col] + self.epsilon)

        return df

    def get_feature_names_out(
            self,
            input_features=None,
    ) -> np.ndarray:
        base = [
            "time_id",
            f"{self.demand_met_col}_sqrd",
            f"log_{self.demand_met_col}",
            f"log_{self.demand_met_col}_sqrd",
        ]
        # Only include log_CO2 if CO2 will actually be present
        if input_features is None or (self.co2_col in set(input_features)):
            base.append(f"log_{self.co2_col}")
        if input_features is not None:
            return np.array(list(input_features) + base)
        return np.array(base)


class DateTimeFeatureAdder(BaseEstimator, TransformerMixin):
    """
    Add datetime-based features from a timestamp column.

    New columns:
      - year (int)
      - month (int)
      - week_of_year (ISO week, int)
      - day (int)
      - hour (int)
      - half_hour (0..47, int)
      - day_of_week (1=Mon..7=Sun, int)
      - is_weekend (0/1, int)


    Parameters
    ----------
    timestamp_col : str, default="timestamp"
        Name of the column containing datetime strings or pd.Timestamp.
    drop_original : bool, default=True
        Whether to drop the original timestamp column after extraction.

    Raises
    ------
    TypeError
        If `timestamp_col` is not found in the DataFrame.
    KeyError
        If `timestamp_col` is not present in X.
\
    """
    def __init__(
        self,
        timestamp_col: str = "timestamp",
        drop_original: bool = False,
    ) -> None:
        """
        Initialize the feature adder.

        Parameters
        ----------
        timestamp_col : str
            Column name to parse as datetime.
        drop_original : bool
            Whether to drop the original timestamp column after extraction.

        Returns
        -------
        None
        """
        if not isinstance(timestamp_col, str):
            raise TypeError("timestamp_col must be a string.")
        if not isinstance(drop_original, bool):
            raise TypeError("drop_original must be a bool.")
        self.timestamp_col = timestamp_col
        self.drop_original = drop_original

    def fit(
            self,
            X,
            y=None
    ) -> DateTimeFeatureAdder:
        """
        No-op fit. Exists for sklearn compatibility.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : pd.Series, optional
            Target variable (not used).

        Returns
        -------
        self : DateTimeFeatureAdder
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if self.timestamp_col not in X.columns:
            raise KeyError(f"Column '{self.timestamp_col}' not found in "
                           f"DataFrame.")

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def transform(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform X by adding:

        - year (int)
        - month (int)
        - week_of_year (int)
        - day (int)
        - hour (int)
        - half_hour (int, 0-47)
        - day_of_week (int, 1=Mon)
        - is_weekend (0/1)

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame with a column named `self.timestamp_col`.

        Returns
        -------
        X_out : pd.DataFrame
            Copy of X with the above new columns appended.

        Raises
        ------
        KeyError
            If `self.timestamp_col` is not present in X.
        """
        df = X.copy()
        # Attempt to convert the timestamp column to datetime (if not already)
        try:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col],
                                                    errors='raise')
        except Exception as e:
            raise TypeError(f"Column '{self.timestamp_col}' could not be "
                            f"converted to datetime: {e}")

        dt = df[self.timestamp_col]
        df["year"] = dt.dt.year.astype('int32')
        df["month"] = dt.dt.month.astype('int32')
        df["week_of_year"] = dt.dt.isocalendar().week.astype('int32')
        df["day"] = dt.dt.day.astype('int32')
        df["hour"] = dt.dt.hour.astype('int32')
        df["half_hour"] = (dt.dt.hour * 2 + (dt.dt.minute // 30)
                           ).astype("int32")
        df["day_of_week"] = (dt.dt.dayofweek).astype('int32') + 1  # Monday=1
        df["is_weekend"] = (df["day_of_week"] >= 6).astype('int32')

        if self.drop_original:
            df = df.drop(columns=[self.timestamp_col])

        return df

    def get_feature_names_out(
            self,
            input_features=None
    ) -> np.ndarray:
        """
        Get the names of the output features.

        Parameters
        ----------
        input_features : array-like, optional
            The input feature names. If None, the original feature names
            are used.

        Returns
        -------
        np.ndarray
            The output feature names.
        """
        added = ["year", "month", "week_of_year", "day", "hour", "half_hour",
                 "day_of_week", "is_weekend"]
        if self.drop_original or input_features is None:
            base = [] if input_features is None else (
                    [c for c in input_features if c != self.timestamp_col])
        else:
            base = list(input_features)
        return np.array(base + added, dtype=object)


class GenerationShareAdder(BaseEstimator, TransformerMixin):
    """
    Add percentage‐share features for specified generation columns relative to
    a total.

    Parameters
    ----------
    generation_cols : List[str]
        Columns whose shares of `total_col` are computed.
    total_col : str, default="total_generation"
        Denominator column.
    suffix : str, default="_share"
        Suffix appended to new share columns.
    as_percent : bool, default=True
        If True, multiply shares by 100; otherwise keep as 0..1 fraction.
    clip_0_100 : bool, default=False
        If True and `as_percent=True`, clip results into [0, 100].
        If True and `as_percent=False`, clip into [0, 1].

    Raises
    ------
    TypeError
        Bad argument types.
    KeyError
        Missing `generation_cols` or `total_col`.
    """

    def __init__(
            self,
            generation_cols: List[str],
            total_col: str = "total_generation",
            suffix: str = "_share",
            as_percent: bool = True,
            clip_0_100: bool = False,
    ) -> None:
        """
        Initialize the share adder.

        Parameters
        ----------
        generation_cols : List[str]
            Columns to convert into percentage shares.
        total_col : str
            Column used as the denominator in share calculation.
        suffix : str
            Suffix for the new share columns.

        Raises
        ------
        TypeError
            If `generation_cols` is not a list of strings, or if `total_col`
            or `suffix` are not strings.
        """
        if not isinstance(generation_cols,
                          list) or not all(isinstance(col,
                                                      str)
                                           for col in generation_cols):
            raise TypeError("generation_cols must be a list of strings.")
        if not isinstance(total_col, str):
            raise TypeError("total_col must be a string.")
        if not isinstance(suffix, str):
            raise TypeError("suffix must be a string.")
        if not isinstance(as_percent, bool):
            raise TypeError("as_percent must be a bool.")
        if not isinstance(clip_0_100, bool):
            raise TypeError("clip_0_100 must be a bool.")

        self.generation_cols = generation_cols
        self.total_col = total_col
        self.suffix = suffix
        self.as_percent = as_percent
        self.clip_0_100 = clip_0_100

    def fit(
            self,
            X,
            y=None
    ) -> GenerationShareAdder:
        """
        No‐op fit for compatibility with sklearn’s transformer API.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame.
        y : Ignored

        Returns
        -------
        self : GenerationShareAdder

        Raises
        ------
        TypeError
            If `X` is not a pandas DataFrame.
        KeyError
            If any of the specified `generation_cols` or `total_col` is not
            present in the DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        missing_cols = (
            [col for col in self.generation_cols if col not in X.columns])
        if missing_cols:
            raise KeyError(f"Generation columns {missing_cols} not found "
                           f"in input DataFrame.")
        if self.total_col not in X.columns:
            raise KeyError(f"Total column '{self.total_col}' not found in"
                           f" input DataFrame.")
        return self

    def transform(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute and append share columns.

        For each `col` in `generation_cols`, creates a new column
        `col + suffix` = 100 * (X[col] / X[total_col]). Zeros in `total_col`
        are treated as NaN to avoid division‐by‐zero.

        Parameters
        ----------
        X : pd.DataFrame
            Input DataFrame containing `generation_cols` and `total_col`.

        Returns
        -------
        X_out : pd.DataFrame
            Copy of X with additional `<col><suffix>` columns.

        """
        df = X.copy()
        # avoid integer division & div-by-zero
        total = df[self.total_col].astype("float64").replace({0.0: np.nan})
        scale = 100.0 if self.as_percent else 1.0

        for col in self.generation_cols:
            share_col = f"{col}{self.suffix}"
            df[share_col] = (df[col].astype("float64") / total) * scale
            if self.clip_0_100:
                lo, hi = (0.0, 100.0) if self.as_percent else (0.0, 1.0)
                df[share_col] = df[share_col].clip(lower=lo, upper=hi)

        return df

    def get_feature_names_out(
            self,
            input_features=None
    ) -> np.ndarray:
        """
        Get the feature names of the output DataFrame.

        Parameters
        ----------
        input_features : array-like, shape (n_features,), default None
            Input feature names.

        Returns
        -------
        feature_names : np.ndarray
            Output feature names.
        """
        added = [f"{c}{self.suffix}" for c in self.generation_cols]
        base = [] if input_features is None else list(input_features)
        return np.array(base + added, dtype=object)


# ────────────────────────────────────────────────────────────────────────────
# BINNING TRANSFORMERS
# ────────────────────────────────────────────────────────────────────────────
# Discretize continuous variables into quantile- or median-based bins.
#   Provides compact group IDs for grouped regression or stratified analysis.
#
# CLASSES:
#   - MultiQuantileBinner: quantile-based binning with mixed-radix encoding
#   - MultiMedianBinner: median-split binning with binary encoding
#
# NOTES:
#   - Useful for creating categorical groups for regression models.
#   - OOB (out-of-bounds) handling policies are configurable.
# ────────────────────────────────────────────────────────────────────────────

class MultiQuantileBinner(BaseEstimator, TransformerMixin):
    """
    Quantile bin multiple variables, then combine their per-variable bin IDs
    intoa single mixed-radix group ID (1-based).

    Example: with bin_specs={'v1':5, 'v2':4}:
      - Fit stores quantile edges for each var.
      - Transform assigns v1_group∈{1..5}, v2_group∈{1..4},
        then builds group_col_name = 1 + (v1_group-1)*4 + (v2_group-1)*1.
    """

    def __init__(
        self,
        bin_specs: dict[str, int],
        group_col_name: str = "quantile_group_id",
        retain_flags: bool = True,
        oob_policy: str = "clip",
        max_oob_rate: float | None = None,
    ):
        """
        Parameters
        ----------
        bin_specs : dict[str, int]
            Mapping of variable -> # of quantile bins (positive integers).
        group_col_name : str, default "quantile_group_id"
            Output column for the combined mixed-radix group ID (1-based).
        retain_flags : bool, default True
            If True, keep per-variable `<var>_group` columns.
        oob_policy : {"clip","edge","error"}, default "clip"
            Handling for values falling outside learned edges at transform
            time:
              - "clip": send to nearest bin (1 or max)
              - "edge": send to the first bin
              - "error": raise ValueError
        max_oob_rate : float or None, default None
            If set, raise an error when an individual variable sees
            OOB rate > max_oob_rate during transform.
        """
        if not isinstance(bin_specs, dict) or not bin_specs:
            raise ValueError("bin_specs must be a non-empty dict")
        if oob_policy not in {"clip", "edge", "error"}:
            raise ValueError("oob_policy must be one of "
                             "{'clip','edge','error'}")

        self.bin_specs = self.validate_and_convert_bins(bin_specs)
        self.group_col_name = str(group_col_name)
        self.retain_flags = bool(retain_flags)
        self.oob_policy = oob_policy
        self.max_oob_rate = max_oob_rate

        self.variables_: list[str] | None = None
        self.quantile_edges_: dict[str, list[float]] = {}
        self.bin_sizes_: dict[str, int] = {}
        self.multipliers_: list[int] | None = None
        self.oob_counts_: dict[str, int] = {}

    def fit(
            self,
            X: pd.DataFrame,
            y=None
    ) -> MultiQuantileBinner:
        """
        Learn quantile edges for each variable.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all variables in `bin_specs`.
        y : pd.Series
            Target variable (not used).

        Returns
        -------
        self : MultiQuantileBinner
            Fitted transformer.
        """
        self.variables_ = list(self.bin_specs.keys())
        self.quantile_edges_.clear()
        self.bin_sizes_.clear()
        self.oob_counts_.clear()

        eps = 1e-4
        for var in self.variables_:
            n_bins = self.bin_specs[var]
            if var not in X.columns:
                raise ValueError(f"Column '{var}' not found in X")
            qs = np.linspace(0, 1, n_bins + 1)
            raw = X[var].quantile(qs, interpolation="midpoint").values
            vmin, vmax = X[var].min(), X[var].max()
            edges = np.unique(np.concatenate([[vmin - eps],
                                              raw, [vmax + eps]]))
            edges.sort()
            self.quantile_edges_[var] = edges.tolist()
            self.bin_sizes_[var] = len(edges) - 1

        bases = [self.bin_sizes_[v] for v in self.variables_]
        m = [1]
        for b in reversed(bases[1:]):
            m.insert(0, m[0] * b)
        self.multipliers_ = m
        return self

    def transform(
            self,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Assign per-variable quantile bins and the combined group ID.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all variables in `bin_specs`.

        Returns
        -------
        pd.DataFrame
            X plus `<var>_group` (optional) and `group_col_name`.
        """
        if not self.quantile_edges_:
            raise RuntimeError("Must fit binner before transform()")
        df = X.copy()
        self.oob_counts_ = {var: 0 for var in self.variables_}

        for var in self.variables_:
            edges = self.quantile_edges_[var]
            n = len(edges) - 1
            s = pd.cut(df[var],
                       bins=edges,
                       labels=range(1, n + 1),
                       include_lowest=True,
                       right=True)
            nan_mask = df[var].isna()
            oob_mask = s.isna() & ~nan_mask
            if oob_mask.any():
                self.oob_counts_[var] += int(oob_mask.sum())
                if self.oob_policy == "error":
                    bad = df.loc[oob_mask, var].unique()
                    raise ValueError(f"OOB values for '{var}': {bad[:10]} ...")
                elif self.oob_policy == "clip":
                    below = df[var] < edges[1]
                    s = s.astype("Float64")
                    s.loc[oob_mask & below] = 1
                    s.loc[oob_mask & ~below] = n
                    s = s.astype("Int64")
                else:  # "edge"
                    s = s.fillna(1)

            # Keep true NaNs as NaN (or decide a policy)
            s = s.astype("Int64")

            df[f"{var}_group"] = s.astype(int)

        total = len(df)
        if self.max_oob_rate is not None and total > 0:
            for var, cnt in self.oob_counts_.items():
                rate = cnt / total
                if rate > self.max_oob_rate:
                    raise ValueError(
                        f"OOB rate {rate:.2%} exceeds max_oob_rate="
                        f"{self.max_oob_rate:.2%} for '{var}'"
                    )

        df[self.group_col_name] = 1
        for v, m in zip(self.variables_, self.multipliers_):
            df[self.group_col_name] += (df[f"{v}_group"] - 1) * m

        if not self.retain_flags:
            df.drop(columns=[f"{v}_group" for v in self.variables_],
                    inplace=True)

        return df

    @staticmethod
    def validate_and_convert_bins(bin_specs: dict) -> dict[str, int]:
        """
        Validate and convert bin specifications to positive integers.

        Parameters
        ----------
        bin_specs : dict
            Mapping of variable names to bin specifications.

        Returns
        -------
        dict[str, int]
            Mapping of variable names to validated bin specifications.

        """
        converted: dict[str, int] = {}
        for k, v in bin_specs.items():
            try:
                v_int = int(float(v))
                if v_int != float(v) or v_int <= 0:
                    raise ValueError
                converted[str(k)] = v_int
            except (ValueError, TypeError) as e:
                raise TypeError(f"Bin spec '{k}' value '{v}' must be a "
                                f"positive integer") from e
        return converted

    def get_feature_names_out(
            self,
            input_features=None
    ) -> np.ndarray:
        """
        Get feature names after transformation.

        Parameters
        ----------
        input_features : array-like, optional
            Input feature names.

        Returns
        -------
        np.ndarray
            Array of feature names after transformation.
        """
        names = []
        if self.retain_flags and self.variables_:
            names += [f"{v}_group" for v in self.variables_]
        names.append(self.group_col_name)
        if input_features is not None:
            return np.array(list(input_features) + names)
        return np.array(names)


class MultiMedianBinner(BaseEstimator, TransformerMixin):
    """
    Median-split each variable and combine flags into a 1-based group ID.
    """

    def __init__(
            self: MultiMedianBinner,
            variables: list[str],
            group_col_name: str = "median_group_id",
            retain_flags: bool = True
    ) -> None:
        """
        Initialize the MultiMedianBinner.

        Parameters
        ----------
        variables : list[str]
            List of variable names to be binned.
        group_col_name : str
            Name of the output column for group IDs.
        retain_flags : bool
            Whether to retain individual group flags.

        Returns
        -------
        None
        """
        if not isinstance(variables, list) or len(variables) == 0:
            raise ValueError("`variables` must be a non-empty list of "
                             "column names.")
        if any(not isinstance(v, str) for v in variables):
            raise TypeError("All entries in `variables` must be strings.")
        if not isinstance(group_col_name, str) or not group_col_name:
            raise TypeError("`group_col_name` must be a non-empty string.")
        if not isinstance(retain_flags, bool):
            raise TypeError("`retain_flags` must be a boolean value.")

        self.variables = variables
        self.group_col_name = group_col_name
        self.retain_flags = retain_flags
        self.medians_: dict[str, float] = {}

    def fit(
            self,
            X,
            y=None
    ) -> MultiMedianBinner:
        """
        Fit the binner by computing medians for each variable.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : pd.Series, optional
            Target variable (not used).

        Returns
        -------
        MultiMedianBinner
            Fitted instance.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        missing = [v for v in self.variables if v not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in input DataFrame:"
                             f"{missing}")
        self.medians_ = X[self.variables].median(skipna=True).to_dict()
        return self

    def transform(
            self: MultiMedianBinner,
            X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform the input DataFrame by binning the specified variables.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Copy of X with optional `<var>_group` flags (0/1) and
            `group_col_name`.
        """
        if not self.medians_:
            raise RuntimeError("Must call fit() before transform().")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        missing = [v for v in self.variables if v not in X.columns]
        if missing:
            raise ValueError(f"Columns missing at transform time: {missing}")

        df = X.copy()
        # compare each column to its scalar median (aligned by column name)
        flags = (df[self.variables] > pd.Series(self.medians_)).astype(int)

        multipliers = 2 ** np.arange(len(self.variables))[::-1]
        df[self.group_col_name] = flags.values.dot(multipliers) + 1

        if self.retain_flags:
            for var in self.variables:
                df[f"{var}_group"] = flags[var]

        return df

    def get_feature_names_out(
            self: MultiMedianBinner,
            input_features=None
    ) -> np.ndarray:
        """
        Get the names of the output features after transformation.

        Parameters
        ----------
        input_features : array-like, optional
            Input feature names.

        Returns
        -------
        np.ndarray
            Output feature names.
        """
        names = []
        if self.retain_flags:
            names += [f"{v}_group" for v in self.variables]
        names.append(self.group_col_name)
        if input_features is not None:
            return np.array(list(input_features) + names)
        return np.array(names)
