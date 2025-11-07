# ────────────────────────────────────────────────────────────────────────────
# FILE: meter_readings_standardisation.py

# PURPOSE: This script is designed to unify the datatypes of the meter
# readings data to ensure consistency and ease of analysis at later stages

# USAGE: This script is designed to run on the HPC cluster at Imperial College
# London. It uses MPI to distribute the workload across multiple processes.
# Before running this script, ensure that the directory and filename variables
# are updated to include the correct file names for your local setup.

# OUTPUTS: Parquet files containing meter reading data for each year, saved
# in the specified directory. The files should be named in the format
# 'meter_readings_YEAR_TIMESTAMP_formatted.parquet'. In this case, we expect
# data for the years 2021, 2022, and 2023 to be processed.
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

import os
import binascii
from mpi4py import MPI

# For general data manipulation and analysis
import polars as pl

# For geospatial data handling
from shapely.wkb import loads

# ────────────────────────────────────────────────────────────────────────────
# Initialise MPI
# ────────────────────────────────────────────────────────────────────────────

# initialises mpi communicator to handle communications between processes
comm = MPI.COMM_WORLD
# retrieves the rank (unique id) of the current process
rank = comm.Get_rank()
# retrieves the total number of processes in the communicator
size = comm.Get_size()

# ────────────────────────────────────────────────────────────────────────────
# Setting up Directories, Defining File Names and Paths
# ────────────────────────────────────────────────────────────────────────────

# Directories
base_directory = os.path.join('..', '..')
hitachi_data_directory = os.path.join(base_directory, 'data', 'hitachi_copy')
meter_readings_directory = os.path.join(hitachi_data_directory,
                                        'meter_primary_files')

# ****** NOTE ****** CHANGE THE FOLLOWING PATHS TO MATCH YOUR LOCAL SETUP
# listed in relative order of size, smallest first
meter_readings_filenames = [
   "meter_readings_2021_20250714_2015.parquet",
   "meter_readings_2022_20250714_2324.parquet",
   "meter_readings_2023_20250714_2039.parquet",
]

file_suffix = "formatted"

# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


def wkb_to_coords(hex_wkb: str):
    """
    Convert a hex WKB string to coordinates (x, y). Uses binascii to decode
    the hex string and shapely's load function to convert WKB to a Point
    object.

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
# Data Processing
# ────────────────────────────────────────────────────────────────────────────


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
            raise ValueError((f"Invalid file format: {filename}. "
                             f"Expected a .parquet file."))

        input_path = os.path.join(meter_readings_directory, filename)
        output_path = input_path.replace(".parquet", f"_{file_suffix}.parquet")

        # Check if the file exists
        if not os.path.isfile(input_path):
            raise FileNotFoundError((f"File not found: {filename} "
                                    f"in {input_path}"))

        (pl.scan_parquet(input_path)
            .with_columns([
                # converting 'city' column to categorical data type
                pl.col("city").cast(pl.Categorical),
                # converting the 'date' column to datetime type
                pl.col("date").str.strptime(pl.Datetime,
                                            format="%Y-%m-%d %H:%M:%S"),
                # converting 'value' column to float64
                pl.col("value").cast(pl.Float64),
                # converting 'ca_id' to string (if not already)

                pl.col("ca_id").cast(pl.String),
            ])
            .sink_parquet(
                output_path,
                # Using snappy compression for better performance
                compression="snappy",
                # Enable statistics for better query performance
                # (doesn't make a noticeable change on disk storage)
                statistics=True,
            )
         )
        print(f"Rank {rank} SUCCESS: Saved optimised data to: {output_path}")
    except Exception as e:
        print(f"Rank {rank} ERROR: Failed to process '{filename}': {e}")

# Ensure all ranks have completed the processing before finishing
comm.Barrier()
if rank == 0:
    print(f"=== MPI Job Completed on {size} ranks ===")

MPI.Finalize()  # Finalize MPI environment
