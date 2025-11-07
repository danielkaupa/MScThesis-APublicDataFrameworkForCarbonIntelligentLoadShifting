
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
]

# ────────────────────────────────────────────────────────────────────────────
# Processing Meter Readings Files
# ────────────────────────────────────────────────────────────────────────────

# broadcast the list of filenames to all ranks (processes)
file_list = comm.bcast(meter_readings_filenames if rank ==0 else None, root=0)

# assign the files across ranks (processes)
tasks = file_list[rank::size] if file_list else []

lazy_frames = []
for filename in tasks:
    file_path = os.path.join(meter_readings_directory, filename)
    if os.path.exists(file_path):
        lazy_frames.append(pl.scan_parquet(file_path))


if lazy_frames:
    local_df = pl.concat(lazy_frames).collect(engine='streaming')
    send_data = pickle.dumps(local_df)

else:
    send_data = b""


# Gather to rank 0 for final merge
if rank == 0:
    # Start with rank 0's data
    final_df = None
    if send_data:
        final_df = pickle.loads(send_data)
    
  # Receive and concatenate data from other ranks
    for sender in range(1, size):
        received_data = comm.recv(source=sender, tag=1)
        if received_data:
            received_df = pickle.loads(received_data)
            if final_df is None:
                # Concatenate the received data to the final DataFrame
                final_df = received_df
            else:
                final_df = final_df.vstack(received_df)

    if final_df is not None:
        # Generate output filename
        sample_file = os.path.basename(file_list[0])
        pattern = r"^(.+?)_(\d{4})_(\d{8})_(\d{4})_optimised\.parquet$"
        match = re.search(pattern, sample_file)

        if match:
            prefix, year, datestamp, timestamp = match.groups()
            output_path = os.path.join(meter_readings_directory, 
                                       f"{prefix}_all_years_{datestamp}_{timestamp}_optimised.parquet")
            final_df.write_parquet(
                output_path,
                compression="snappy",
                statistics=True
            )
            print(f"Rank {rank}: Merged data saved to {output_path}")
else:
    # Send processed data to rank 0
    comm.send(send_data, dest=0, tag=1)

MPI.Finalize()
