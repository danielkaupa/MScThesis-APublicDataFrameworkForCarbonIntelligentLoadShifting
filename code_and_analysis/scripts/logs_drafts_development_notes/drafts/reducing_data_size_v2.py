# ────────────────────────────────────────────────────────────────────────────
# author: Daniel Kaupa
# date: 2025-07-14
# summary: This script is designed to unify and optimise the data used in the analysis.
#          This is primarily achieved by changing data types of columns.
# instructions: Change the names of the files in the `data_save_directory` variable to match your local setup
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

import os
import io
import subprocess
import tempfile
import binascii
import re
from datetime import datetime
from mpi4py import MPI

# For general data manipulation and analysis
import polars as pl

# For geospatial data handling
from shapely.wkb import loads

# ────────────────────────────────────────────────────────────────────────────
# Initialise MPI
# ────────────────────────────────────────────────────────────────────────────

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ────────────────────────────────────────────────────────────────────────────
# Setting up Directories, Defining File Names and Paths
# ────────────────────────────────────────────────────────────────────────────

base_directory = os.path.join('..', '..')
data_save_directory = os.path.join(base_directory,'data', 'hitachi_copy')
meter_readings_directory = os.path.join(data_save_directory, 'meter_primary_files')

# ****** NOTE ****** CHANGE THE FOLLOWING PATHS TO MATCH YOUR LOCAL SETUP
customers_path = os.path.join(
    data_save_directory, "customers_20250714_1401.parquet"
)
grid_readings_path = os.path.join(
    data_save_directory, "grid_readings_20250714_1401.parquet"
)
weather_path = os.path.join(
    data_save_directory, "weather_20250714_1401.parquet"
)

# # listed in relative order of size, smallest first
meter_readings_filenames = [
   "meter_readings_2021_20250714_2015.parquet",
   "meter_readings_2022_20250714_2324.parquet",
   "meter_readings_2023_20250714_2039.parquet",
#    "meter_readings_delhi_2022_Q3and4_20250704.parquet",
#    "meter_readings_mumbai_all_years_20250704.parquet",
#     "meter_readings_delhi_all_years_20250704.parquet",
#     "meter_readings_all_years_20250704.parquet",
]

# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────
def wkb_to_coords(hex_wkb: str):
    """
    Convert a hex WKB string to coordinates (x, y).
    Uses binascii to decode the hex string and shapely's load function to convert WKB to a Point object.

    Parameters:
    ----------
    hex_wkb : str
        Hexadecimal string representing the WKB (Well-Known Binary) format of a point.
    
    Returns:
    -------
    tuple
        A tuple containing the x and y coordinates of the point, or (None, None)
        if the conversion fails.
    """
    try:
        point = loads(binascii.unhexlify(hex_wkb))
        return point.x, point.y
    except Exception:
        return None, None


# ────────────────────────────────────────────────────────────────────────────
# Data Processing
# ────────────────────────────────────────────────────────────────────────────

# ********* Weather Data Processing ********

if rank ==0:
    try:
        weather_lazy_operations = (
            pl.scan_parquet(weather_path)
            .with_columns([
                # converting to two separate columns for longitude and latitude
                pl.col("location")
                  .map_elements(lambda h: wkb_to_coords(h)[0], return_dtype=pl.Float64)
                  .alias("weather_longitude"),

                pl.col("location")
                  .map_elements(lambda h: wkb_to_coords(h)[1], return_dtype=pl.Float64)
                  .alias("weather_latitude"),

                # converting 'city' column to categorical data type
                pl.col("city").cast(pl.Categorical),

                # converting wind direction & speed to float32
                pl.col("wind_direction").cast(pl.Float32),
                pl.col("wind_speed").cast(pl.Float32),

                # converting temperature to float32
                pl.col("temperature").cast(pl.Float32),

                # converting precipitation to float 32, then to mm
                (pl.col("precipitation").cast(pl.Float32) * 1000).alias("precipitation_mm"), # precimpitation is in m, converting to mm

                # converting surface net solar radiation to Float64, then kilo-watt-hours
                ((pl.col("surface_net_solar_radiation").cast(pl.Float64) / (60 * 60))/1000).alias("surface_net_solar_radiation_kwh"),  # converting from J/m^2 to kWh/m^2
                # converting surface solar radiation downwards to Float64, then kilo-watt-hours
                ((pl.col("surface_solar_radiation_downwards").cast(pl.Float64) / (60 * 60))/1000).alias("surface_solar_radiation_downwards_kwh"),  # converting from J/m^2 to kWh/m^2
            ])
            .drop("location", "precipitation", "surface_net_solar_radiation", "surface_solar_radiation_downwards")  # drop the original versions of changed columns
        )
        weather_table_optimised_path = weather_path.replace(".parquet", "_optimised.parquet")
        weather_lazy_operations.sink_parquet(
            weather_table_optimised_path,
            compression="snappy",  # Using snappy compression for better performance
            statistics=True,  # Enable statistics for better query performance (doesn't make a noticeable change on disk storage)
        )
        print(f"Rank {rank} SUCCESS: Saved optimised data to: {weather_table_optimised_path}")
    except Exception as e:
        print(f"Rank {rank} ERROR: Failed to process weather data: {e}")    

# ********* Customers Data Processing ********

if rank == 0:
    try:
        customers_lazy_operations = (
            pl.scan_parquet(customers_path)
            .with_columns([
                # converting to two separate columns for longitude and latitude
                pl.col("location")
                  .map_elements(lambda h: wkb_to_coords(h)[0], return_dtype=pl.Float64)
                  .alias("customer_longitude"),

                pl.col("location")
                  .map_elements(lambda h: wkb_to_coords(h)[1], return_dtype=pl.Float64)
                  .alias("customer_latitude"),

                # converting 'city' column to categorical data type
                pl.col("city").cast(pl.Categorical),
            ])
            .drop("location")  # drop the original 'location' column
        )
        customers_table_optimised_path = customers_path.replace(".parquet", "_optimised.parquet")
        customers_lazy_operations.sink_parquet(
            customers_table_optimised_path,
            compression="snappy",  # Using snappy compression for better performance
            statistics=True,  # Enable statistics for better query performance (doesn't make a noticeable change on disk storage)
        )
        print(f"Rank {rank} SUCCESS: Saved optimised data to: {customers_table_optimised_path}")
    except Exception as e:
        print(f"Rank {rank} ERROR: Failed to process customers data: {e}")


# ********* Meter Readings Data Processing ********

comm.Barrier()  # Ensure all ranks have completed the previous tasks before proceeding

# Broadcast filenames to all ranks (MPI)
if rank == 0:
    file_list = meter_readings_filenames
else:
    file_list = None
file_list = comm.bcast(file_list, root=0)

# Slice list among ranks
tasks = file_list[rank::size] if file_list else []

for filename in tasks:
    try:
        # Ensure the filename is valid and exists in the directory
        if not filename.endswith('.parquet'):
            raise ValueError(f"Invalid file format: {filename}. Expected a .parquet file.")
        
        input_path = os.path.join(meter_readings_directory, filename)
        output_path = input_path.replace(".parquet", "_optimised.parquet")

        # Check if the file exists
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"File not found: {filename} in {input_path}")
        
        (pl.scan_parquet(input_path)
            .with_columns([
                # converting 'city' column to categorical data type
                pl.col("city").cast(pl.Categorical),
                # converting the 'date' column to datetime type
                pl.col("date").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S"),
                # converting 'value' column to float64
                pl.col("value").cast(pl.Float64),
                pl.col("ca_id").cast(pl.String),  # converting 'ca_id' to string (if not already)
            ])
            .sink_parquet(
                output_path,
                compression="snappy",  # Using snappy compression for better performance
                statistics=True,  # Enable statistics for better query performance (doesn't make a noticeable change on disk storage)
            )
        )
        print(f"Rank {rank} SUCCESS: Saved optimised data to: {output_path}")
    except Exception as e:
        print(f"Rank {rank} ERROR: Failed to process '{filename}': {e}")


comm.Barrier()  # Ensure all ranks have completed the processing before finishing
if rank == 0:
    print(f"=== MPI Job Completed on {size} ranks ===")

MPI.Finalize()  # Finalize MPI environment
