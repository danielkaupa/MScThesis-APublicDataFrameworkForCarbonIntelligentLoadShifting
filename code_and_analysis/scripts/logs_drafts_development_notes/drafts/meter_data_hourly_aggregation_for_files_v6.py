
#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------
from mpi4py import MPI
import os
from datetime import datetime
import re
import gc

import polars as pl


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#---------------------------------------------------------------------------------------------------------
# Script Purpose
#---------------------------------------------------------------------------------------------------------
# The purpose of this script is to read through provided meter_readings data files,
# aggregate the data into hourly intervals, and save the results into a new Parquet file
# with the same name as the original file but with an 'hourly_' suffix just before the year_date timestamp
# user must provide the names of the files to be processed

#---------------------------------------------------------------------------------------------------------
# Setting up Directories, Defining File Names and Paths
#---------------------------------------------------------------------------------------------------------

# directories
base_directory = os.path.join('..', '..')
data_save_directory = os.path.join(base_directory,'data', 'hitachi_copy')
meter_readings_directory = os.path.join(data_save_directory, 'meter_primary_files')

# file names
# listed in relative order of size, smallest first
meter_readings_filenames = [
#    "meter_readings_mumbai_2022_Q1and2_20250704.parquet",
#    "meter_readings_mumbai_2022_Q3and4_20250704.parquet",
#    "meter_readings_delhi_2022_Q1and2_20250704.parquet",
#    "meter_readings_delhi_2022_Q3and4_20250704.parquet",
#    "meter_readings_mumbai_all_years_20250704.parquet",
    "meter_readings_delhi_all_years_20250704.parquet",
    "meter_readings_all_years_20250704.parquet",
]

# Broadcast filenames to all ranks
if rank == 0:
    file_list = meter_readings_filenames
else:
    file_list = None
file_list = comm.bcast(file_list, root=0)

# Slice list among ranks
tasks = file_list[rank::size]


#---------------------------------------------------------------------------------------------------------
# Processing Files
#---------------------------------------------------------------------------------------------------------
for filename in tasks:
    input_path = os.path.join(meter_readings_directory, filename)
    if not os.path.exists(input_path):
        if rank == 0:
            print(f"ERROR: File not found: {input_path}")
        continue

    match = re.search(r"(.*?)(\d{8})\.parquet$", filename)
    if not match:
        if rank == 0:
            print(f"ERROR: Could not parse filename: {filename}")
        continue

    base_name, date_part = match.groups()
    output_path = os.path.join(
        meter_readings_directory,
        f"{base_name}hourly_{date_part}.parquet"
    )

    try:
        lazy_hourly_df = (
            pl.scan_parquet(input_path)
              .sort("date")
              .group_by_dynamic(
                  index_column="date",
                  every="1h",
                  group_by=["ca_id", "city"]
              )
              .agg([
                  pl.col("value").sum().alias("hourly_kWh_sum"),
                  pl.col("value").mean().alias("hourly_kWh_mean"),
              ])
        )
        lazy_hourly_df.sink_parquet(output_path, compression="snappy")
        print(f"Rank {rank} SUCCESS: Saved hourly data to: {output_path}")
    except Exception as e:
        print(f"Rank {rank} ERROR: Failed to process '{filename}': {e}")

# Optional: gather logs or results (if needed) or barrier to synchronize
comm.Barrier()
if rank == 0:
    print(f"=== MPI Job Completed on {size} ranks ===")
