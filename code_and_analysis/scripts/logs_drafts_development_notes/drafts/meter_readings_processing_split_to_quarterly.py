# ────────────────────────────────────────────────────────────────────────────
# FILE: meter_readings_processing_split_to_quarterly.py

# PURPOSE: This script processes meter readings data that has been split by
# year and city, and further splits it into quarterly files.

# USAGE: This script is designed to run on the HPC cluster at Imperial College
# London. It uses MPI to distribute the workload across multiple processes.
# Before running this script, ensure that the directory and filename variables
# are updated to include the correct file names for your local setup.

# OUTPUTS: Parquet files containing meter reading data for each year, saved
# in the specified directory. The files should be named in the format
# 'meter_readings_YEAR_TIMESTAMP.parquet'. In this case, we expect data for
# the years 2021, 2022, and 2023 to be processed.
# ─────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

import os
import polars as pl
from mpi4py import MPI
import re

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

# # listed in relative order of size, smallest first
meter_readings_filenames = [
   "meter_readings_2021_20250714_2015_formatted.parquet",
   "meter_readings_2022_20250714_2324_formatted.parquet",
   "meter_readings_2023_20250714_2039_formatted.parquet",
   "meter_readings_delhi_2021_20250714_2015_formatted.parquet",
   "meter_readings_delhi_2022_20250714_2324_formatted.parquet",
   "meter_readings_mumbai_2022_20250714_2324_formatted.parquet",
   "meter_readings_mumbai_2023_20250714_2039_formatted.parquet",
]

# ────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────────────────────


def extract_quarter() -> pl.Expr:
    """
    Creates a Polars expression that maps a Datetime column to a
    quarter string.

    Parameters:
    ------------
    None

    Returns:
    --------
    pl.Expr:
        Polars expression that extracts the quarter from a date column.
    """
    return (
        pl.when(pl.col(name="date").is_not_null())
        .then(
            pl.format("Q{}", (pl.col("date").dt.month() - 1) // 3 + 1)
        )
        .otherwise(statement=None)
        .alias(name="quarter")
    )


def split_and_save_by_quarter(file_path: str, output_dir: str):
    """
    Splits a meter readings Parquet file into quarterly files based on
    the 'date' column.

    Parameters:
    -----------
    file_path: str
        Full path to the Parquet file
    output_dir: str
        Directory where split files will be saved

    Returns:
    --------
    None
        Saves split files to the specified output directory
    """
    try:
        lf = pl.scan_parquet(source=file_path)

        # Add quarter column
        lf = lf.with_columns([
            pl.col(name="date").cast(dtype=pl.Datetime),
            extract_quarter()
        ])

        # Collect unique quarters
        quarters = (
            lf.select("quarter").unique().collect()
            .get_column(name="quarter").to_list()
        )

        base_name = os.path.basename(file_path)

        # match timestamp pattern from right to left
        pattern = r"^(.+?)_(\d{4})_(\d{8})_(\d{4})\_formatted\.parquet$"
        match = re.search(pattern=pattern, string=base_name)

        if not match:
            raise ValueError(f"Could not find timestamp pattern in"
                             f" filename: {base_name}")

        prefix, year, datestamp, timestamp = match.groups()

        for q in quarters:
            output_filename = (f"{prefix}_{year}_{q}_{datestamp}_"
                               f"{timestamp}_formatted.parquet")
            output_path = os.path.join(output_dir, output_filename)

            (
                lf.filter(pl.col(name="quarter") == q)
                .drop("quarter")
                .sink_parquet(
                    path=output_path,
                    compression="snappy",
                    statistics=True,
                )
            )
            print(f"Rank {rank} SUCCESS: Saved {output_path}")

    except Exception as e:
        print(f"Rank {rank} ERROR splitting file {file_path}: {e}")

# ────────────────────────────────────────────────────────────────────────────
# Processing Meter Readings Files
# ────────────────────────────────────────────────────────────────────────────


file_list = comm.bcast(meter_readings_filenames if rank == 0 else None,
                           root=0)
tasks = file_list[rank::size]

for filename in tasks:
    input_path = os.path.join(meter_readings_directory, filename)

    if not os.path.isfile(input_path):
        print(f"Rank {rank} WARNING: File not found: {input_path}")
        continue
    split_and_save_by_quarter(file_path=input_path,
                              output_dir=meter_readings_directory)

comm.Barrier()
if rank == 0:
    print(f"=== Quarter-splitting complete on {size} ranks ===")

MPI.Finalize()
