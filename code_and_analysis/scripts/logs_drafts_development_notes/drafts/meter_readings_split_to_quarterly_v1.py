# ────────────────────────────────────────────────────────────────────────────
# author: Daniel Kaupa
# date: 2025-07-15
# summary: This script is designed to take meter readings data divided into individual years, and or by city
#          and split them further by quarters (Q1,Q2,Q3,Q4).
# instructions: Change the names of the files in the `data_save_directory` variable to match your local setup
# ────────────────────────────────────────────────────────────────────────────

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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ────────────────────────────────────────────────────────────────────────────
# Setting up Directories, Defining File Names and Paths
# ────────────────────────────────────────────────────────────────────────────

# directories
base_directory = os.path.join('..', '..')
data_save_directory = os.path.join(base_directory,'data', 'hitachi_copy')
meter_readings_directory = os.path.join(data_save_directory, 'meter_primary_files')

# ****** NOTE ****** CHANGE THE FOLLOWING PATHS TO MATCH YOUR LOCAL SETUP

# # listed in relative order of size, smallest first
meter_readings_filenames = [
   "meter_readings_2021_20250714_2015_optimised.parquet",
   "meter_readings_2022_20250714_2324_optimised.parquet",
   "meter_readings_2023_20250714_2039_optimised.parquet",
   "meter_readings_delhi_2021_20250714_2015_optimised.parquet",
   "meter_readings_delhi_2022_20250714_2324_optimised.parquet",
   "meter_readings_mumbai_2022_20250714_2324_optimised.parquet",
   "meter_readings_mumbai_2023_20250714_2039_optimised.parquet",
]

#────────────────────────────────────────────────────────────────────────────
# Helper Functions
#────────────────────────────────────────────────────────────────────────────

def extract_quarter() -> pl.Expr:
    """
    Creates a Polars expression that maps a Datetime column to a quarter string.

    Parameters:
    ------------
    None

    Returns:
    --------
    pl.Expr:
        Polars expression that extracts the quarter from a date column.
    """
    return (
        pl.when(pl.col("date").is_not_null())
        .then(
            pl.format("Q{}", (pl.col("date").dt.month() - 1) // 3 + 1)
        )
        .otherwise(None)
        .alias("quarter")
    )

def split_and_save_by_quarter(file_path: str, output_dir: str):
    """
    Splits a meter readings Parquet file into quarterly files based on the 'date' column.

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
        lf = pl.scan_parquet(file_path)

        # Add quarter column
        lf = lf.with_columns([
            pl.col("date").cast(pl.Datetime),
            extract_quarter()
        ])

        # Collect unique quarters
        quarters = (
            lf.select("quarter").unique().collect().get_column("quarter").to_list()
        )

        base_name = os.path.basename(file_path)

        # match timestamp pattern from right to left
        pattern = r"^(.+?)_(\d{4})_(\d{8})_(\d{4})_optimised\.parquet$"
        match = re.search(pattern, base_name)

        if not match:
            raise ValueError(f"Could not find timestamp pattern in filename: {base_name}")
        
        prefix, year, datestamp, timestamp = match.groups()

        for q in quarters:
            output_filename = f"{prefix}_{year}_{q}_{datestamp}_{timestamp}_optimised.parquet"
            output_path = os.path.join(output_dir, output_filename)
    
            (
                lf.filter(pl.col("quarter") == q)
                .drop("quarter")  
                .sink_parquet(
                    output_path,
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

file_list = comm.bcast(meter_readings_filenames if rank == 0 else None, root=0)
tasks = file_list[rank::size]

for filename in tasks:
    input_path = os.path.join(meter_readings_directory, filename)

    if not os.path.isfile(input_path):
        print(f"Rank {rank} WARNING: File not found: {input_path}")
        continue

    split_and_save_by_quarter(input_path, meter_readings_directory)

comm.Barrier()
if rank == 0:
    print(f"=== Quarter-splitting complete on {size} ranks ===")

MPI.Finalize()
