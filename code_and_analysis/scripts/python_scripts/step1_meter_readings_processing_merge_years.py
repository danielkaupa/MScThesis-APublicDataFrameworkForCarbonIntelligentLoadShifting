# ────────────────────────────────────────────────────────────────────────────
# FILE: step1_meter_readings_processing_merge_years.py
#
# PURPOSE: This script processes meter readings data that has been split by
# year and merges them into a single file.
#
# USAGE:
# - This script is designed to run on the HPC cluster at Imperial College
#   London.
# - Before running this script, ensure that the directory and filename
#   variables are updated to include the correct file names for your
#   local setup.
#
# OUTPUTS:
# - One Parquet file containing meter reading data for each year, saved
#   in the specified directory.
# - The file should be named in the format
#   'meter_readings_all_years_YYYYMMDD_formatted.parquet'.
# - In this case, we expect data for the years 2021, 2022, and 2023 to be
#   processed.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import polars as pl


# ─────────────────────────────────────────────────────────────────────────────
# Setting up Directories, Defining File Names and Paths
# ─────────────────────────────────────────────────────────────────────────────


# Directories
base_directory = os.path.join('..', '..')
data_directory = os.path.join(base_directory, 'data')
hitachi_data_directory = os.path.join(data_directory, 'hitachi')
meter_readings_directory = os.path.join(hitachi_data_directory,
                                        'meter_primary_files')

# File names
file_names = [
    "meter_readings_2021_20250714_2015_formatted",
    "meter_readings_2022_20250714_2324_formatted",
    "meter_readings_2023_20250714_2039_formatted",
]

# File paths
file_paths = [os.path.join(meter_readings_directory,
              f"{name}.parquet") for name in file_names]


# ─────────────────────────────────────────────────────────────────────────────
# Data Processing
# ─────────────────────────────────────────────────────────────────────────────


# Reading Data and Merging Meter Readings using LazyFrames
# ──────────────────────────────

# wrap operation with string cache to avoid repeated string parsing
with pl.StringCache():

    # Create a list to hold the lazy frames
    lazy_frames = []

    # Read each file as a LazyFrame and apply necessary transformations
    for path in file_paths:
        lf = pl.read_parquet(path, use_pyarrow=True).lazy()
        lazy_frames.append(lf)

    # Concatenate all lazy frames into a single LazyFrame
    # ──────────────────────────────
    meter_readings_df = pl.concat(lazy_frames).with_columns([
        pl.col("city").cast(pl.Categorical),
        pl.col("ca_id").cast(pl.String),
        pl.col("value").cast(pl.Float64),
        pl.col("date").cast(pl.Datetime)
    ]).collect()
# We do the city casting here to ensure that any different category mappings
# across years are handled correctly, as the city column is expected to be
# consistent across all years.
# The rest of the casting is to ensure consistency in types and guard against
# any potential error. These types should already be set from the formatted
# files.


# Retrieve File naming convention from the first file
# ──────────────────────────────
match = re.search(r"_(\d{8})_\d+_formatted$", file_names[0])
if not match:
    raise ValueError(f"No date pattern found in {file_names[0]}")
retrieval_date = match.group(1)

# Define the output file name
output_filepath = os.path.join(
    meter_readings_directory,
    f"meter_readings_all_years_{retrieval_date}_formatted.parquet"
)


# Save the merged LazyFrame to a Parquet file
# ──────────────────────────────
try:
    meter_readings_df.write_parquet(
        output_filepath,
        compression="snappy",
        statistics=True
    )
    print((f"\n--> Saved the merged meter readings data "
           f"to: '{output_filepath}'"))
except Exception as e:
    print(f"ERROR saving merged meter readings data: {e}")
