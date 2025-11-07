# ─────────────────────────────────────────────────────────────────────────────
# FILE: step5_joining_customer_meter_emissions_full.py
#
# PURPOSE:
# - Build a clean, join-ready emissions layer to pair with smart-meter data.
# - Standardise units and timezones for:
#     * average grid emissions (gCO2/kWh),
#     * marginal emissions (gCO2/kWh) and demand (kWh).
# - Create a *stable spatial key* by deduplicating emissions locations and
#   assigning each a unique `emission_factor_location_id`.
# - Map each customer to the nearest emissions location (geometry-only join),
#   then join time-series emissions to meter readings by
#   (`timestamp`/`date`, `emission_factor_location_id`).
# - Provide sensible city-level fallbacks to fill any remaining nulls.
# - Emit a trimmed test slice (fixed 2022-05-04 → 2022-05-18 IST) for quick QA.
#
# USAGE:
# - Run after generating the upstream Parquet inputs:
#     1) customers parquet with a `location` column (hex WKB),
#     2) half-hourly meter readings parquet,
#     3) marginal emissions time series parquet,
#     4) average emissions time series parquet.
# - Execute as a standalone script to materialize all downstream Parquet
#   outputs. The script resolves paths relative to the repo root
#   (one level above /scripts).
#
# RUN REQUIREMENTS:
# - Python 3.9+ (uses `zoneinfo` and type hints).
# - Libraries: polars, geopandas, shapely, pandas (for GeoPandas interop).
#   * GeoPandas needs a spatial index backend (Shapely ≥2.0/pygeos) for
#     `sjoin_nearest`; `rtree` is optional but can improve performance.
# - System deps: GEOS/PROJ (standard for GeoPandas/Shapely).
# - Disk access to the `data/` directory structure described below.
#
# INPUTS (expected on disk):
# - customers_20250714_1401.parquet
# - meter_readings_all_years_20250714_formatted.parquet
# - original_quantile_bins_marginal_emissions_timeseries.parquet
# - grid_readings_20250714_1401_processed_half_hourly.parquet
#
# MAIN STEPS (high level):
# 1) Average emissions:
#    - Keep `timestamp`, rename `g_co2_per_kwh`→`
#       average_emissions_grams_co2_per_kWh`,
#      localise to IST (Asia/Kolkata), and write a compact Parquet.
#
# 2) Marginal emissions:
#    - Add:
#        * `marginal_emissions_tons_co2_per_MW` (original ME),
#        * `marginal_emissions_grams_co2_per_kWh` = (ME / 0.5 h) * 1000,
#        * `demand_met_MW` (clarify),
#        * `demand_met_kWh` = demand_met_MW * 0.5 h * 1000.
#    - Write Parquet.
#
# 3) City-level fallbacks:
#    - Compute per-city, per-timestamp averages of
#      `demand_met_kWh` and `marginal_emissions_grams_co2_per_kWh`;
#      left-join average emissions by timestamp for a unified fallback layer.
#
# 4) Join marginal↔average emissions:
#    - Left-join on `timestamp`; keep only needed columns.
#    - Write `marginal_and_average_emissions_*.parquet`.
#
# 5) Build emissions locations dictionary:
#    - From the joined emissions file, select unique
#      (`land_longitude`,`land_latitude`), assign
#      `emission_factor_location_id`, and persist as a lookup table.
#
# 6) Decode customer coordinates and map to nearest emissions location:
#    - Convert customers’ `location` (hex WKB) → (`customer_longitude`,
#      `customer_latitude`).
#    - Geo join (EPSG:3857) with emissions locations via `sjoin_nearest`
#      to assign `emission_factor_location_id` to each customer.
#    - Save `customers_to_emission_factor_location_mapping.parquet`.
#
# 7) Attach emissions location ids to meter readings:
#    - Left-join meter readings with the customer-location mapping by
#      `ca_id`→`id`; keep city, coordinates, and `emission_factor_location_id`.
#
# 8) Attach time-series emissions to meter readings:
#    - Add `emission_factor_location_id` to the full emissions join,
#      then join readings by (`date`==`timestamp`,
#      `emission_factor_location_id`)
#      to get both marginal and average emissions per reading.
#
# 9) Fill gaps using city fallbacks:
#    - For any reading still missing emissions, left-join the city-level table
#      by (`date`,`city`) and fill:
#        * `demand_met_kWh`,
#        * `marginal_emissions_grams_co2_per_kWh`,
#        * `average_emissions_grams_co2_per_kWh`.
#
# 10) Emit a trimmed QA subset:
#    - Filter the filled dataset to a fixed two-week IST window and save.
#
# OUTPUTS (Parquet):
# - optimisation_development/processing_files/
#     * average_emissions_<start>_to_<end>.parquet
#     * marginal_emissions_<start>_to_<end>.parquet
#     * city_average_marginal_and_average_emissions_<start>_to_<end>.parquet
#     * marginal_and_average_emissions_<start>_to_<end>.parquet
#     * marginal_and_average_emissions_locations.parquet
#     * customers_to_emission_factor_location_mapping.parquet

#       (prefix = meter_readings_all_years_)
#     * prefix..._formatted_with_emission_factor_location.parquet
#     * marginal_and_average_emissions_with_locations_<start>_to_<end>.parquet
#     * prefix_..._formatted_with_emission_factors_raw.parquet
# - optimisation_development/
#     * prefix..._formatted_with_emission_factors_filled.parquet
#     * prefix..._formatted_with_emission_factors_filled_2022-05-04_
#       to_2022-05-18.parquet
#
# NOTES & ASSUMPTIONS:
# - Timezone: all timestamps are localised/converted to Asia/Kolkata (IST).
# - Spatial join is geometry-only: customers are mapped to a *location id*,
#   and time is joined later. This avoids duplicating time across geometries
#   and is robust for large time series.
# - `emission_factor_location_id` is created from the unique lon/lat pairs.
#   If reproducible ids across runs are required, replace the row-count id
#   with a deterministic hash of (`land_longitude`,`land_latitude`).
# - The script uses Polars lazy scans and `sink_parquet` to keep memory
#       use low.
# - Basic logging is via `print` (file sizes, column lists) for quick QA.
# ─────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# IMPORTING LIBRARIES
# ────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import binascii
import os
from zoneinfo import ZoneInfo
from datetime import datetime
import polars as pl
import geopandas as gpd
from shapely.wkb import loads

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# FUNCTIONS
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


def wkb_to_coords(hex_wkb: str):
    """
    Convert a hex WKB string to coordinates (x, y).
    Uses binascii to decode the hex string and shapely's load function to
    convert WKB to a Point object.

    Parameters:
    ----------
    hex_wkb : str
        Hexadecimal string representing the WKB (Well-Known Binary) format
        of a point.

    Returns:
    -------
    tuple
        A tuple containing the x and y coordinates of the point,
        or (None, None) if the conversion fails.
    """
    try:
        point = loads(binascii.unhexlify(hex_wkb))
        return point.x, point.y
    except Exception:
        return None, None

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# DEFINING FILEPATHS AND DIRECTORIES
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


# DIRECTORIES AND PATHS
# Resolve repo root → .../code_and_analysis (one level up from scripts/)
try:
    base_directory = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
except NameError:
    # Fallback for interactive runs
    base_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))


def _abs_join(root: str, maybe_rel: str) -> str:
    """Join to root if path is relative; return as-is if already absolute."""
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(
        root, maybe_rel)


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
optimisation_results_directory = os.path.join(
    optimisation_development_directory,
    "results")
optimisation_processing_directory = os.path.join(
    optimisation_development_directory,
    "processing_files")

# Defining Full files
full_customers_filename = "customers_20250714_1401"
full_meter_readings_filename = "meter_readings_all_years_20250714_formatted"
quantile_marginal_emissions_filename = (
    "original_quantile_bins_marginal_emissions_timeseries")
full_average_emissions_filename = (
    "grid_readings_20250714_1401_processed_half_hourly")
pygam_marginal_emissions_filename = (
    "pyGAM_marginal_emissions_timeseries")

# Defining Full Filepaths
full_customers_filepath = os.path.join(
    hitachi_data_directory,
    full_customers_filename + ".parquet")

full_meter_readings_filepath = os.path.join(
    meter_save_directory,
    full_meter_readings_filename + ".parquet")

quantile_marginal_emissions_filepath = os.path.join(
    marginal_emissions_results_directory,
    quantile_marginal_emissions_filename + ".parquet")

pygam_marginal_emissions_filepath = os.path.join(
    marginal_emissions_results_directory,
    pygam_marginal_emissions_filename + ".parquet")

full_average_emissions_filepath = os.path.join(
    hitachi_data_directory,
    full_average_emissions_filename + ".parquet")

# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
#
# LOADING AND PROCESSING DATA
#
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────


ROUND = 3

# Average Emission Factors
# ────────────────────────────────────────────────────────────────────────────

average_emissions_pldf = pl.read_parquet(full_average_emissions_filepath)

average_emissions_pldf = average_emissions_pldf.drop(
    ["thermal_generation", "gas_generation", "hydro_generation",
     "nuclear_generation", "renewable_generation", "total_generation",
     "demand_met", "non_renewable_generation", "tons_co2",
     "tons_co2_per_mwh"])

average_emissions_pldf = average_emissions_pldf.rename(
    {"g_co2_per_kwh": "average_emissions_grams_co2_per_kWh"}
)

average_emissions_pldf = average_emissions_pldf.with_columns(
    pl.col("timestamp").dt.replace_time_zone("Asia/Kolkata")
)

average_emissions_start_date = str(
    average_emissions_pldf["timestamp"].min())[:10]
average_emissions_end_date = str(
    average_emissions_pldf["timestamp"].max())[:10]


average_emissions_filename = (
    f"average_emissions_{average_emissions_start_date}_to_"
    f"{average_emissions_end_date}")
average_emissions_filepath = os.path.join(
    optimisation_processing_directory,
    average_emissions_filename + ".parquet")

try:
    average_emissions_pldf.write_parquet(
        file=average_emissions_filepath,
        compression="snappy",
        statistics=True)

    print(f"Successfully saved average emissions data to "
          f"{average_emissions_filepath}")
    fls = os.path.getsize(average_emissions_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    print(f"Columns present: {average_emissions_pldf.columns}")

except Exception as e:
    print(f"Error occurred while saving average emissions data: {e}")


# Marginal Emission Factors
# ────────────────────────────────────────────────────────────────────────────

quantile_marginal_emissions_pldf = pl.read_parquet(quantile_marginal_emissions_filepath)

quantile_marginal_emissions_pldf = quantile_marginal_emissions_pldf.with_columns(
    # clarifying units - ME currently: tons CO2 per MW per interval
    pl.col("ME").alias("quantile_marginal_emissions_tons_co2_per_MW"),
    # Convert to g CO2 per MWh: divide by interval_hours
    ((pl.col("ME")/0.5)*1000).alias("quantile_marginal_emissions_grams_co2_per_kWh"),
    # clarify units of demand_met
    pl.col("demand_met").alias("demand_met_MW"),
    # convert to kWh
    ((pl.col("demand_met") * 0.5 * 1000).alias("demand_met_kWh"))
)

pygam_marginal_emissions_pldf = pl.read_parquet(pygam_marginal_emissions_filepath)

pygam_marginal_emissions_pldf = pygam_marginal_emissions_pldf.with_columns(
    # clarifying units - ME currently: tons CO2 per MW per interval
    pl.col("ME").alias("pg_marginal_emissions_tons_co2_per_MW"),
    # Convert to g CO2 per MWh: divide by interval_hours
    ((pl.col("ME")/0.5)*1000).alias("pg_marginal_emissions_grams_co2_per_kWh"),
    # clarify units of demand_met
    pl.col("ME_cal").alias("pg_marginal_emissions_tons_co2_per_MW_calibrated"),
    (pl.col("ME_cal")/0.5*1000).alias("pg_marginal_emissions_grams_co2_per_kWh_calibrated")
)

pygam_marginal_emissions_pldf = pygam_marginal_emissions_pldf.drop(["y_true", "y_pred", "demand_met", "demand_minus_renewables", "ME_cal", "ME"])


# join the two marginal emissions dataframes
joined_marginal_emissions_pldf = quantile_marginal_emissions_pldf.join(
    pygam_marginal_emissions_pldf,
    on=["timestamp", "land_latitude", "land_longitude"],
    how="left",
    suffix="_pygam"
)

joined_marginal_emissions_pldf.drop([
    "original_quantile_group_id",
    "demand_met",
    "demand_met_MW",
])

marginal_emissions_start_date = str(
    joined_marginal_emissions_pldf["timestamp"].min())[:10]
marginal_emissions_end_date = str(
    joined_marginal_emissions_pldf["timestamp"].max())[:10]

marginal_emissions_filename = (
    f"marginal_emissions_{marginal_emissions_start_date}_to_"
    f"{marginal_emissions_end_date}")
marginal_emissions_filepath = os.path.join(
    optimisation_processing_directory,
    marginal_emissions_filename + ".parquet")

try:
    joined_marginal_emissions_pldf.write_parquet(
        marginal_emissions_filepath,
        compression="snappy",
        statistics=True
    )
    print(f"Successfully saved marginal emissions data to "
          f"{marginal_emissions_filepath}")
    fls = os.path.getsize(marginal_emissions_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    print(f"Columns present: {joined_marginal_emissions_pldf.columns}")

except Exception as e:
    print(f"Error occurred while saving marginal emissions data: {e}")


# City Average Marginal and Average Emission Factors For Gap Filling
# ────────────────────────────────────────────────────────────────────────────

# Per-city, per-timestamp averages (ignores rows where city is null)
city_average_marginal_emissions_pldf = (
    joined_marginal_emissions_pldf
    .filter(pl.col("city").is_not_null())
    .group_by(["city", "timestamp"])
    .agg([pl.col(c).mean().alias(c)
          for c in ["demand_met_kWh", "quantile_marginal_emissions_grams_co2_per_kWh", "pg_marginal_emissions_grams_co2_per_kWh", "pg_marginal_emissions_grams_co2_per_kWh_calibrated"]])
    .sort(["city", "timestamp"])
)

# Perform a left join to keep all rows from gapfill_marginal_emissions_pldf
city_average_marginal_and_average_emissions_pldf = (
    city_average_marginal_emissions_pldf.join(
        other=average_emissions_pldf,
        on="timestamp",
        how="left"
    )
)

city_average_start_date = str(
    city_average_marginal_emissions_pldf["timestamp"].min())[:10]
city_average_end_date = str(
    city_average_marginal_emissions_pldf["timestamp"].max())[:10]

city_average_marginal_and_average_emissions_filename = (
    f"city_average_marginal_and_average_emissions_"
    f"{city_average_start_date}_to_{city_average_end_date}")
city_average_marginal_and_average_emissions_filepath = os.path.join(
    optimisation_processing_directory,
    city_average_marginal_and_average_emissions_filename + ".parquet")

try:
    city_average_marginal_and_average_emissions_pldf.write_parquet(
        file=city_average_marginal_and_average_emissions_filepath,
        compression="snappy",
        statistics=True)
    print(f"Successfully saved city average marginal and average emissions "
          f"data to {city_average_marginal_and_average_emissions_filepath}")
    fls = os.path.getsize(city_average_marginal_and_average_emissions_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    print(f"Columns present: "
          f"{city_average_marginal_and_average_emissions_pldf.columns}")
except Exception as e:
    print(f"Error occurred while saving city average marginal and "
          f"average emissions data: {e}")


# Full Marginal and Average Emission Factors
# ────────────────────────────────────────────────────────────────────────────

joined_emission_factors_filename = (
    f"marginal_and_average_emissions_{str(marginal_emissions_start_date)}"
    f"_to_{str(marginal_emissions_end_date)}")
joined_emission_factors_filepath = os.path.join(
    optimisation_processing_directory,
    joined_emission_factors_filename + ".parquet")

average_emissions_lazy_df = pl.scan_parquet(average_emissions_filepath)
marginal_emissions_lazy_df = pl.scan_parquet(marginal_emissions_filepath)

joined_emission_factors_lazy_df = (
    marginal_emissions_lazy_df
    .join(
        average_emissions_lazy_df,
        left_on=["timestamp"],
        right_on=["timestamp"],
        how="left"  # Keep all marginal emissions
    )
    # Optional: Select only the columns you need
    .select([
        # Columns from avg_emissions
        "average_emissions_grams_co2_per_kWh",
        # Columns from customer data
        "timestamp", "city", "land_longitude",
        "land_latitude", "demand_met_kWh",
        "quantile_marginal_emissions_grams_co2_per_kWh",
        "pg_marginal_emissions_grams_co2_per_kWh",
        "pg_marginal_emissions_grams_co2_per_kWh_calibrated",
        "National MW Shift", "Pearson R Score", "Spearman Score",
        "Confidence Level"
    ])
)

# Write the result to a new Parquet file
try:
    (
        joined_emission_factors_lazy_df
        .sink_parquet(
            joined_emission_factors_filepath,
            # Optional optimization parameters:
            compression="snappy",  # Good balance of speed and compression
            statistics=True,    # Write statistics for better query performance
            row_group_size=100_000  # Adjust based on your data size
        )
    )
    print(f"Successfully saved [joined_emission_factors_lazy_df] to "
          f"{joined_emission_factors_filepath}")
    fls = os.path.getsize(joined_emission_factors_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    print(f"Columns present: "
          f"{joined_emission_factors_lazy_df.collect_schema().names()}")

except Exception as e:
    print(f"Error saving [joined_emission_factors_lazy_df] to "
          f"{joined_emission_factors_filepath}: {e}")


# Getting Emissions Locations
# ────────────────────────────────────────────────────────────────────────────

joined_emission_factors_locations_filename = (
    "marginal_and_average_emissions_locations")
joined_emission_factors_locations_filepath = os.path.join(
    optimisation_processing_directory,
    joined_emission_factors_locations_filename + ".parquet"
)

joined_emission_factors_locations_pldf = (
    pl.scan_parquet(
        joined_emission_factors_filepath
    ).select(["land_longitude", "land_latitude"])
     .unique()
     .with_row_count("emission_factor_location_id")
     .collect()
)

try:
    joined_emission_factors_locations_pldf.write_parquet(
        joined_emission_factors_locations_filepath,
        compression="snappy",
        statistics=True
        )
    print(f"Successfully saved [joined_emission_factors_locations_pldf] to"
          f" {joined_emission_factors_locations_filepath}")
    fls = os.path.getsize(joined_emission_factors_locations_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    print(f"Columns present: {joined_emission_factors_locations_pldf.columns}")

except Exception as e:
    print(f"Error saving [joined_emission_factors_locations_pldf] to"
          f"{joined_emission_factors_locations_filepath}: {e}")


# Generating Geolocations & Geodataframe for Customers
# ────────────────────────────────────────────────────────────────────────────

# Generating Longitude and Latitude from string location encoding
customers_coords_lzdf = (
    pl.scan_parquet(full_customers_filepath).with_columns(
            # convert location fields to latitude and longitude
            pl.col("location").map_elements(
                wkb_to_coords,
                return_dtype=pl.List(pl.Float64)
            ).alias("location_coords")
        ).drop(["location"]).with_columns([
            pl.col("location_coords").list.get(0).alias("customer_longitude"),
            pl.col("location_coords").list.get(1).alias("customer_latitude"),
        ]).drop(["location_coords"])
)

# collect dataframe into memory
customers_coords_pldf = customers_coords_lzdf.collect()

# Convert to GeoDataFrame for spatial operations
customers_pd = customers_coords_pldf.to_pandas()
customers_pd["city"] = customers_pd["city"].astype("category")
customers_data_gdf = gpd.GeoDataFrame(
    customers_pd,
    geometry=gpd.points_from_xy(
            customers_pd["customer_longitude"],
            customers_pd["customer_latitude"]
    ),
    crs="EPSG:4326",
).to_crs("EPSG:3857")


# Generating Geodataframe for Emission Factors
# ────────────────────────────────────────────────────────────────────────────

emission_factor_locations_pd = (
    joined_emission_factors_locations_pldf.to_pandas())
emission_factor_locations_gdf = gpd.GeoDataFrame(
    emission_factor_locations_pd,
    geometry=gpd.points_from_xy(
            emission_factor_locations_pd["land_longitude"],
            emission_factor_locations_pd["land_latitude"]
    ),
    crs="EPSG:4326",
).to_crs("EPSG:3857")


# Joining Customer Data <-> Emissions Locations
# ────────────────────────────────────────────────────────────────────────────

# Use sjoin to match nearest locations to each other
customer_and_emission_factor_locations_gdf = customers_data_gdf.sjoin_nearest(
    emission_factor_locations_gdf,
    how="left",
    lsuffix="_customers",
    rsuffix="_emission_factors",
    distance_col="distance_between_locations_meters",
)

# convert back to pandas, drop duplicates, keep only needed columns
customer_and_emission_factor_locations_pd = (
        customer_and_emission_factor_locations_gdf[
            ["id", "city", "customer_longitude",
             "customer_latitude", "emission_factor_location_id",
             "land_longitude", "land_latitude"]
            ].drop_duplicates().reset_index(drop=True))

# Save
customer_and_emission_factor_locations_filename = (
        "customers_to_emission_factor_location_mapping"
        )
customer_and_emission_factor_locations_filepath = os.path.join(
        optimisation_processing_directory,
        customer_and_emission_factor_locations_filename + ".parquet")

try:
    pl.from_pandas(customer_and_emission_factor_locations_pd).write_parquet(
        customer_and_emission_factor_locations_filepath,
        compression="snappy",
        statistics=True
        )

    print(f"Successfully saved [customer_and_emission_factor_locations_pd] to"
          f"{customer_and_emission_factor_locations_filepath}")
    fls = os.path.getsize(customer_and_emission_factor_locations_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    print(f"Columns present: "
          f"{customer_and_emission_factor_locations_pd.columns}")

except Exception as e:
    print(f"Error saving [customer_and_emission_factor_locations_pd] to "
          f"{customer_and_emission_factor_locations_filepath}: {e}")


# Joining Customer & Emission Locations to Meter Readings
# ────────────────────────────────────────────────────────────────────────────

# Load Lazy Dataframes
meter_readings_lazy_df = pl.scan_parquet(full_meter_readings_filepath)
customer_and_emission_factor_locations_lazy_df = (
    pl.scan_parquet(customer_and_emission_factor_locations_filepath))

# Peroform lazy join
meter_readings_with_emissions_locations_lazy_df = (
        meter_readings_lazy_df.join(
            customer_and_emission_factor_locations_lazy_df,
            left_on="ca_id",
            right_on="id",
            how="left"
        )
        .select([
            "ca_id", "date", "value", "city",
            "customer_longitude", "customer_latitude",
            "emission_factor_location_id", "land_longitude", "land_latitude"
        ])
    )


meter_readings_with_emissions_locations_lazy_df = (
    meter_readings_with_emissions_locations_lazy_df.with_columns(
        pl.col("date").dt.replace_time_zone("Asia/Kolkata")
    ))
# Filename and Path
meter_readings_with_emissions_locations_filename = (
    full_meter_readings_filename + "_with_emission_factor_location")
meter_readings_with_emissions_locations_filepath = os.path.join(
        optimisation_processing_directory,
        meter_readings_with_emissions_locations_filename +
        ".parquet"
        )

# Save
try:
    meter_readings_with_emissions_locations_lazy_df.sink_parquet(
        meter_readings_with_emissions_locations_filepath,
        compression="snappy",
        statistics=True,
        row_group_size=100_000,
    )
    print(f"Successfully saved "
          f"[meter_readings_with_emissions_locations_lazy_df] to "
          f"{meter_readings_with_emissions_locations_filepath}")
    fls = os.path.getsize(meter_readings_with_emissions_locations_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = meter_readings_with_emissions_locations_lazy_df.collect_schema(
        ).names()
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving [meter_readings_with_emissions_locations_lazy_df] to"
          f"{meter_readings_with_emissions_locations_filepath}: {e}")


# Add 'emission_factor_location_id' to full emissions data
# ────────────────────────────────────────────────────────────────────────────

# Lazy Operations
joined_emission_factors_with_locations_lazy_df = (
    pl.scan_parquet(joined_emission_factors_filepath)
    .join(
          pl.scan_parquet(joined_emission_factors_locations_filepath),
          on=["land_longitude", "land_latitude"],
          how="left"
      )
    # Keep only what's needed downstream
    .select([
          "timestamp", "emission_factor_location_id",
          "average_emissions_grams_co2_per_kWh",
          "demand_met_kWh",
          "quantile_marginal_emissions_grams_co2_per_kWh",
          "pg_marginal_emissions_grams_co2_per_kWh",
          "pg_marginal_emissions_grams_co2_per_kWh_calibrated",
          "National MW Shift",
          "Pearson R Score",
          "Spearman Score",
          "Confidence Level"
      ])
)

# Filename and Path
joined_emission_factors_with_locations_filename = (
    f"marginal_and_average_emissions_with_locations_"
    f"{str(marginal_emissions_start_date)}_to_"
    f"{str(marginal_emissions_end_date)}"
)

joined_emission_factors_with_locations_filepath = os.path.join(
    optimisation_processing_directory,
    joined_emission_factors_with_locations_filename +
    ".parquet"
)

# Save
try:
    joined_emission_factors_with_locations_lazy_df.sink_parquet(
        joined_emission_factors_with_locations_filepath,
        compression="snappy",
        statistics=True,
        row_group_size=100_000,
    )
    print(f"Successfully saved "
          f"[joined_emission_factors_with_locations_lazy_df] to "
          f"{joined_emission_factors_with_locations_filepath}")
    fls = os.path.getsize(joined_emission_factors_with_locations_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = joined_emission_factors_with_locations_lazy_df.collect_schema(
            ).names()
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving [joined_emission_factors_with_locations_lazy_df] to "
          f"{joined_emission_factors_with_locations_filepath}: {e}")


# Join Emission Factors to Meter Readings Data
# ────────────────────────────────────────────────────────────────────────────

meter_readings_with_emissions_locations_lazy_df = (
    pl.scan_parquet(meter_readings_with_emissions_locations_filepath)
    .with_columns(
        pl.col("date").dt.replace_time_zone("Asia/Kolkata")
    )
)

joined_emission_factors_with_locations_lazy_df = pl.scan_parquet(
    joined_emission_factors_with_locations_filepath)

meter_readings_with_emissions_lazy_df = (
    meter_readings_with_emissions_locations_lazy_df.join(
        joined_emission_factors_with_locations_lazy_df,
        left_on=["date", "emission_factor_location_id"],
        right_on=["timestamp", "emission_factor_location_id"],
        how="left"  # Keep all meter readings
        ).select([
            # Columns from meter readings
            "city", "ca_id", "date", "value",
            # Columns from customer data
            "customer_longitude", "customer_latitude",
            "demand_met_kWh",
            "average_emissions_grams_co2_per_kWh",
            "quantile_marginal_emissions_grams_co2_per_kWh",
            "pg_marginal_emissions_grams_co2_per_kWh",
            "pg_marginal_emissions_grams_co2_per_kWh_calibrated",
            "National MW Shift",
            "Pearson R Score",
            "Spearman Score",
            "Confidence Level"
        ])
    )

# Filename and Path
joined_meter_readings_marginal_emissions_filename = (
    full_meter_readings_filename + "_with_emission_factors_raw"
)
joined_meter_readings_marginal_emissions_filepath = os.path.join(
    optimisation_processing_directory,
    joined_meter_readings_marginal_emissions_filename + ".parquet"
)

# Save
try:
    (
        meter_readings_with_emissions_lazy_df
        .sink_parquet(
            joined_meter_readings_marginal_emissions_filepath,
            # Optional optimization parameters:
            compression="snappy",  # Good balance of speed and compression
            statistics=True,    # Write statistics for better query performance
            row_group_size=100_000  # Adjust based on your data size
        )
    )
    print(f"Successfully saved [meter_readings_with_emissions_lazy_df] to "
          f"{joined_meter_readings_marginal_emissions_filepath}")
    fls = os.path.getsize(joined_meter_readings_marginal_emissions_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = meter_readings_with_emissions_lazy_df.collect_schema().names()
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving [meter_readings_with_emissions_lazy_df] to "
          f"{joined_meter_readings_marginal_emissions_filepath}: {e}")


# Filling Any Locations without Emission Factors
# ────────────────────────────────────────────────────────────────────────────

meter_readings_with_emissions_lazy_df = pl.scan_parquet(
    joined_meter_readings_marginal_emissions_filepath)
city_average_marginal_and_average_emissions_lazy_df = pl.scan_parquet(
    city_average_marginal_and_average_emissions_filepath)

meter_readings_with_emissions_location_filled_lazy_df = (
    meter_readings_with_emissions_lazy_df
    .join(
        other=city_average_marginal_and_average_emissions_lazy_df,
        left_on=["date", "city"],
        right_on=["timestamp", "city"],
        how="left"
    ).with_columns([
        # Fill null demand_met_kWh with the averaged version
        pl.when(pl.col("demand_met_kWh").is_null())
        .then(pl.col("demand_met_kWh_right"))  # from joined table
        .otherwise(pl.col("demand_met_kWh"))
        .alias("demand_met_kWh"),

        # Fill null quantile marginal_emissions with the averaged version
        pl.when(pl.col("quantile_marginal_emissions_grams_co2_per_kWh").is_null())
        .then(pl.col("quantile_marginal_emissions_grams_co2_per_kWh_right"))
        .otherwise(pl.col("quantile_marginal_emissions_grams_co2_per_kWh"))
        .alias("quantile_marginal_emissions_grams_co2_per_kWh"),

        # Similarly for average emissions if needed
        pl.when(pl.col("average_emissions_grams_co2_per_kWh").is_null())
        .then(pl.col("average_emissions_grams_co2_per_kWh_right"))
        .otherwise(pl.col("average_emissions_grams_co2_per_kWh"))
        .alias("average_emissions_grams_co2_per_kWh"),

        # Fill null pygam marginal emissions with averaged version
        pl.when(pl.col("pg_marginal_emissions_grams_co2_per_kWh").is_null())
        .then(pl.col("pg_marginal_emissions_grams_co2_per_kWh_right"))
        .otherwise(pl.col("pg_marginal_emissions_grams_co2_per_kWh"))
        .alias("pg_marginal_emissions_grams_co2_per_kWh"),

        # Fill null pygam marginal emissions calibrated with averaged version
        pl.when(pl.col("pg_marginal_emissions_grams_co2_per_kWh_calibrated").is_null())
        .then(pl.col("pg_marginal_emissions_grams_co2_per_kWh_calibrated_right"))
        .otherwise(pl.col("pg_marginal_emissions_grams_co2_per_kWh_calibrated"))
        .alias("pg_marginal_emissions_grams_co2_per_kWh_calibrated"),

        ]
        # Drop the temporary columns from the join
        ).drop(["demand_met_kWh_right",
                "quantile_marginal_emissions_grams_co2_per_kWh_right",
                "average_emissions_grams_co2_per_kWh_right",
                "pg_marginal_emissions_grams_co2_per_kWh_right",
                "pg_marginal_emissions_grams_co2_per_kWh_calibrated_right"])
    )


# reorder columns for clarify
meter_readings_with_emissions_location_filled_lazy_df = (
    meter_readings_with_emissions_location_filled_lazy_df.select(
        ["city",
         "customer_longitude",
         "customer_latitude",
         "ca_id",
         "date",
         "value",
         "quantile_marginal_emissions_grams_co2_per_kWh",
         "average_emissions_grams_co2_per_kWh",
         "demand_met_kWh",
         "pg_marginal_emissions_grams_co2_per_kWh",
         "pg_marginal_emissions_grams_co2_per_kWh_calibrated",
         "National MW Shift",
         "Pearson R Score",
         "Spearman Score",
         "Confidence Level",
         ]
    )
)

joined_meter_readings_marginal_emissions_filled_filename = (
    full_meter_readings_filename + "_with_emission_factors_filled")
joined_meter_readings_marginal_emissions_filled_filepath = os.path.join(
    optimisation_development_directory,
    joined_meter_readings_marginal_emissions_filled_filename + ".parquet")

# Write the result to a new Parquet file
try:
    (
        meter_readings_with_emissions_location_filled_lazy_df.sink_parquet(
                joined_meter_readings_marginal_emissions_filled_filepath,
                # Optional optimization parameters:
                # Good balance of speed and compression
                compression="snappy",
                # Write statistics for better query performance
                statistics=True,
                row_group_size=100_000  # Adjust based on your data size
            )
    )
    print(f"Successfully saved"
          f"[meter_readings_with_emissions_location_filled_lazy_df] "
          f"to {joined_meter_readings_marginal_emissions_filled_filepath}")
    fls = os.path.getsize(
        joined_meter_readings_marginal_emissions_filled_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = (
        meter_readings_with_emissions_location_filled_lazy_df.collect_schema(
            ).names())
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving "
          f"[meter_readings_with_emissions_location_filled_lazy_df] to "
          f"{joined_meter_readings_marginal_emissions_filled_filepath}: {e}")


# ────────────────────────────────────────────────────────────────────────────


meter_readings_quantile_model_filename = (full_meter_readings_filename + "_with_quantile_emission_factors_filled")
meter_readings_quantile_model_filepath = os.path.join(
    optimisation_development_directory,
    meter_readings_quantile_model_filename + ".parquet")

quantile_meter_readings_with_emissions_location_filled_lazy_df = (
    meter_readings_with_emissions_location_filled_lazy_df.select(
        ["city",
         "customer_longitude",
         "customer_latitude",
         "ca_id",
         "date",
         "value",
         "quantile_marginal_emissions_grams_co2_per_kWh",
         "average_emissions_grams_co2_per_kWh",
         "demand_met_kWh",
         ]
    )
).rename({"quantile_marginal_emissions_grams_co2_per_kWh": "marginal_emissions_grams_co2_per_kWh"})

# Write the result to a new Parquet file
try:
    (
        quantile_meter_readings_with_emissions_location_filled_lazy_df.sink_parquet(
                meter_readings_quantile_model_filepath,
                # Optional optimization parameters:
                # Good balance of speed and compression
                compression="snappy",
                # Write statistics for better query performance
                statistics=True,
                row_group_size=100_000  # Adjust based on your data size
            )
    )
    print(f"Successfully saved"
          f"[quantile_meter_readings_with_emissions_location_filled_lazy_df] "
          f"to {meter_readings_quantile_model_filepath}")
    fls = os.path.getsize(
        meter_readings_quantile_model_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = (
        quantile_meter_readings_with_emissions_location_filled_lazy_df.collect_schema(
            ).names())
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving "
          f"[quantile_meter_readings_with_emissions_location_filled_lazy_df] to "
          f"{meter_readings_quantile_model_filepath}: {e}")

# ────────────────────────────────────────────────────────────────────────────

meter_readings_pygam_model_filename = (full_meter_readings_filename + "_with_pygam_emission_factors_filled")
meter_readings_pygam_model_filepath = os.path.join(
    optimisation_development_directory,
    meter_readings_pygam_model_filename + ".parquet")


# reorder columns for clarify
pygam_raw_meter_readings_with_emissions_location_filled_lazy_df = (
    meter_readings_with_emissions_location_filled_lazy_df.select(
        ["city",
         "customer_longitude",
         "customer_latitude",
         "ca_id",
         "date",
         "value",
         "pg_marginal_emissions_grams_co2_per_kWh",
         "average_emissions_grams_co2_per_kWh",
         "demand_met_kWh",
         "National MW Shift",
         "Pearson R Score",
         "Spearman Score",
         "Confidence Level",
         ]
    )
).rename({"pg_marginal_emissions_grams_co2_per_kWh": "marginal_emissions_grams_co2_per_kWh"})

# Write the result to a new Parquet file
try:
    (
        pygam_raw_meter_readings_with_emissions_location_filled_lazy_df.sink_parquet(
                meter_readings_pygam_model_filepath,
                # Optional optimization parameters:
                # Good balance of speed and compression
                compression="snappy",
                # Write statistics for better query performance
                statistics=True,
                row_group_size=100_000  # Adjust based on your data size
            )
    )
    print(f"Successfully saved"
          f"[pygam_raw_meter_readings_with_emissions_location_filled_lazy_df] "
          f"to {meter_readings_pygam_model_filepath}")
    fls = os.path.getsize(
        meter_readings_pygam_model_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = (
        pygam_raw_meter_readings_with_emissions_location_filled_lazy_df.collect_schema(
            ).names())
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving "
          f"[pygam_raw_meter_readings_with_emissions_location_filled_lazy_df] to "
          f"{meter_readings_pygam_model_filepath}: {e}")

# ────────────────────────────────────────────────────────────────────────────


meter_readings_pygam_cal_model_filename = (full_meter_readings_filename + "_with_pygam_cal_emission_factors_filled")
meter_readings_pygam_cal_model_filepath = os.path.join(
    optimisation_development_directory,
    meter_readings_pygam_cal_model_filename + ".parquet")



pygam_cal_meter_readings_with_emissions_location_filled_lazy_df = (
    meter_readings_with_emissions_location_filled_lazy_df.select(
        ["city",
         "customer_longitude",
         "customer_latitude",
         "ca_id",
         "date",
         "value",
         "pg_marginal_emissions_grams_co2_per_kWh_calibrated",
         "average_emissions_grams_co2_per_kWh",
         "demand_met_kWh",
         "National MW Shift",
         "Pearson R Score",
         "Spearman Score",
         "Confidence Level",
         ]
    )
).rename({"pg_marginal_emissions_grams_co2_per_kWh_calibrated": "marginal_emissions_grams_co2_per_kWh"})

# Write the result to a new Parquet file
try:
    (
        pygam_cal_meter_readings_with_emissions_location_filled_lazy_df.sink_parquet(
                meter_readings_pygam_cal_model_filepath,
                # Optional optimization parameters:
                # Good balance of speed and compression
                compression="snappy",
                # Write statistics for better query performance
                statistics=True,
                row_group_size=100_000  # Adjust based on your data size
            )
    )
    print(f"Successfully saved"
          f"[pygam_cal_meter_readings_with_emissions_location_filled_lazy_df] "
          f"to {meter_readings_pygam_cal_model_filepath}")
    fls = os.path.getsize(
        meter_readings_pygam_cal_model_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = (
        pygam_cal_meter_readings_with_emissions_location_filled_lazy_df.collect_schema(
            ).names())
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving "
          f"[pygam_cal_meter_readings_with_emissions_location_filled_lazy_df] to "
          f"{meter_readings_pygam_cal_model_filepath}: {e}")


# Generating a Smaller Test Set
# ────────────────────────────────────────────────────────────────────────────

test_date_range_start = datetime(2022, 5, 4, tzinfo=ZoneInfo("Asia/Kolkata"))
test_date_range_end = datetime(2022, 5, 18, tzinfo=ZoneInfo("Asia/Kolkata"))

# Trim the timeframe to fit the analysis that we are doing:
trimmed_meter_readings_with_emissions_location_filled_lazy_df = (
    meter_readings_with_emissions_location_filled_lazy_df.filter(
        (pl.col("date") >= test_date_range_start) &
        (pl.col("date") <= test_date_range_end)
    )
)

str(test_date_range_start)[:10]
str(test_date_range_end)[:10]

trimmed_joined_meter_readings_marginal_emissions_filled_filename = (
    joined_meter_readings_marginal_emissions_filled_filename +
    f"_{str(test_date_range_start)[:10]}_to_{str(test_date_range_end)[:10]}"
)
trimmed_joined_meter_readings_marginal_emissions_filled_filepath = (
    os.path.join(
        optimisation_development_directory,
        trimmed_joined_meter_readings_marginal_emissions_filled_filename
        + ".parquet"
    )
)

# Write the result to a new Parquet file
try:
    (
        trimmed_meter_readings_with_emissions_location_filled_lazy_df
        .sink_parquet(
            trimmed_joined_meter_readings_marginal_emissions_filled_filepath,
            # Optional optimization parameters:
            # Good balance of speed and compression
            compression="snappy",
            # Write statistics for better query performance
            statistics=True,
            row_group_size=100_000  # Adjust based on your data size
        )
    )
    print(f"Successfully saved [trimmed_meter_readings_with_emissions_"
          f"location_filled_lazy_df] to "
          f"{trimmed_joined_meter_readings_marginal_emissions_filled_filepath}"
          )
    fls = os.path.getsize(
        trimmed_joined_meter_readings_marginal_emissions_filled_filepath)
    print(f"File size: {fls / (1024 * 1024):.2f} MB")
    cols = (
        trimmed_meter_readings_with_emissions_location_filled_lazy_df
        .collect_schema().names())
    print(f"Columns present: {cols}")

except Exception as e:
    print(f"Error saving [trimmed_meter_readings_with_emissions_"
          f"location_filled_lazy_df] to "
          f"{trimmed_joined_meter_readings_marginal_emissions_filled_filepath}"
          f": {e}")
