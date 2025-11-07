#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------

import os
import io
import subprocess
import tempfile
from datetime import datetime
import re

import polars as pl

#---------------------------------------------------------------------------------------------------------
# Setting up Directories, Defining File Names and Paths
#---------------------------------------------------------------------------------------------------------

# directories
base_directory = os.path.join('..', '..')
data_save_directory = os.path.join(base_directory,'data', 'hitachi_copy')
meter_readings_directory = os.path.join(data_save_directory, 'meter_primary_files')

# file names
meter_readings_2021_filename = "meter_readings_2021_20250704_1311"
meter_readings_2022_filename = "meter_readings_2022_20250704_1150"
meter_readings_2023_filename = "meter_readings_2023_20250704_1029"

# file paths
meter_readings_2021_path = os.path.join(meter_readings_directory, f"{meter_readings_2021_filename}.parquet")
meter_readings_2022_path = os.path.join(meter_readings_directory, f"{meter_readings_2022_filename}.parquet")
meter_readings_2023_path = os.path.join(meter_readings_directory, f"{meter_readings_2023_filename}.parquet")

#---------------------------------------------------------------------------------------------------------
# Reading Data and Merging Meter Readings from Different Years
#---------------------------------------------------------------------------------------------------------

# Read the Parquet files for each year
meter_readings_2021_pldf = pl.read_parquet(meter_readings_2021_path)
meter_readings_2022_pldf = pl.read_parquet(meter_readings_2022_path)
meter_readings_2023_pldf = pl.read_parquet(meter_readings_2023_path)

# Concatenating DataFrames for all years
meter_readings_all_years_pldf = pl.concat([meter_readings_2021_pldf, meter_readings_2022_pldf, meter_readings_2023_pldf])

# Changing the data types of specific columns
meter_readings_all_years_pldf = meter_readings_all_years_pldf.with_columns(
    pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
                .cast(pl.Datetime("us")),
    pl.col("ca_id").cast(pl.Utf8),  # Convert 'ca_id' to string
    pl.col("value").cast(pl.Float32),  # Downcast 'value' to float32
    pl.col("city").cast(pl.Categorical),  # ~90% reduction vs strings
)
#---------------------------------------------------------------------------------------------------------
# Saving the Merged Meter Readings Data
#---------------------------------------------------------------------------------------------------------

meter_readings_filename = os.path.basename(meter_readings_2021_path)
temp = re.search(r"_(\d{8})_\d+\.parquet$", meter_readings_filename)

if not temp:
    raise ValueError(f"No date pattern found in {meter_readings_filename}")
retrieval_date = temp.group(1)

meter_readings_all_years_filename = f"meter_readings_all_years_{retrieval_date}.parquet"
meter_readings_all_years_path = os.path.join(
    meter_readings_directory, meter_readings_all_years_filename
)

# Save the hourly_meter_data_with_weather_locations_pldf to a Parquet file
meter_readings_all_years_pldf.write_parquet(
    meter_readings_all_years_path, compression="snappy"
)

print(f"\n--> Saved the hourly_meter_data_with_weather_locations_pldf to: '{meter_readings_all_years_filename}'")
