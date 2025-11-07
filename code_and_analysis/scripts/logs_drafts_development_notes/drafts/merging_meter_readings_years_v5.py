
# ────────────────────────────────────────────────────────────────────────────
# author: Daniel Kaupa
# date: 2025-07-15
# summary: This script is designed to take meter readings data divided into individual years, months, etc,
#          and join them into a single file
# instructions: Change the names of the files in the `data_save_directory` variable to match your local setup
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

import os
import polars as pl
from mpi4py import MPI
import re
import pickle
import shutil

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
temp_directory = os.path.join(data_save_directory, 'temp_merge_outputs')
os.makedirs(temp_directory, exist_ok=True)


# ****** NOTE ****** CHANGE THE FOLLOWING PATHS TO MATCH YOUR LOCAL SETUP

# # listed in relative order of size, smallest first
meter_readings_filenames = [
   "meter_readings_2021_20250714_2015_optimised.parquet",
   "meter_readings_2022_20250714_2324_optimised.parquet",
   "meter_readings_2023_20250714_2039_optimised.parquet",
]

# ────────────────────────────────────────────────────────────────────────────
# Processing Meter Readings Files
# ────────────────────────────────────────────────────────────────────────────

# broadcast the list of filenames to all ranks (processes)
file_list = comm.bcast(meter_readings_filenames if rank ==0 else None, root=0)

# assign the files across ranks (processes)
tasks = file_list[rank::size] if file_list else []


# Process & write to temporary files
local_frames = []
for filename in tasks:
    file_path = os.path.join(meter_readings_directory, filename)
    if os.path.exists(file_path):
        lazy_df = pl.scan_parquet(file_path)
        local_frames.append(lazy_df)

if local_frames:
    result_df = pl.concat(local_frames).collect(engine='streaming')
    print(f"Rank {rank}: Rows collected = {result_df.height}")
    temp_output = os.path.join(temp_directory, f"merged_temp_rank{rank}.parquet")
    result_df.write_parquet(temp_output, compression="snappy", statistics=True)
    print(f"Rank {rank}: wrote {temp_output}")
else:
    print(f"Rank {rank}: No input files found.")

comm.Barrier()

# Merge all temp files on rank 0
if rank == 0:
    all_temp_files = sorted(
        os.path.join(temp_directory, f)
        for f in os.listdir(temp_directory)
        if f.startswith("merged_temp_rank") and f.endswith(".parquet")
    )
    merged_lfs = [pl.scan_parquet(f) for f in all_temp_files]
    final_df = pl.concat(merged_lfs).collect(engine='streaming')

    if not all_temp_files:
        print("Rank 0: No temporary files to merge.")

    else:
        merged_lfs = [pl.scan_parquet(f) for f in all_temp_files]
        final_df = pl.concat(merged_lfs).collect(engine='streaming')
        print(f"Rank 0: Total merged rows = {final_df.height}")

        # Use one of the original filenames as template
        sample_file = os.path.basename(file_list[0])
        pattern = r"^(.+?)_(\d{4})_(\d{8})_(\d{4})_optimised\.parquet$"
        match = re.search(pattern, sample_file)
        
        if match:
            prefix, year, datestamp, timestamp = match.groups()
            output_path = os.path.join(meter_readings_directory,
                                       f"{prefix}_all_years_{datestamp}_{timestamp}_optimised.parquet")
            final_df.write_parquet(output_path, compression="snappy", statistics=True)
            print(f"Rank 0: Final merged file written to: {output_path}")
        else:
            print("Rank 0: Could not parse output filename pattern.")
        
        shutil.rmtree(temp_directory)
        print(f"Rank 0: Removed temp directory: {temp_directory}")

MPI.Finalize()
