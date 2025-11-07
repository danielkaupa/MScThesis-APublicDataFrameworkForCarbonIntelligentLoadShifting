
#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------

import os
from datetime import datetime
import re
import gc

import polars as pl

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
meter_readings_filenames = [
#    "meter_readings_all_years_20250704.parquet",
#    "meter_readings_delhi_all_years_20250704.parquet",
    "meter_readings_delhi_2022_Q1and2_20250704.parquet",
    "meter_readings_delhi_2022_Q3and4_20250704.parquet",
#    "meter_readings_mumbai_all_years_20250704.parquet",
    "meter_readings_mumbai_2022_Q1and2_20250704.parquet",
    "meter_readings_mumbai_2022_Q3and4_20250704.parquet",
]

#---------------------------------------------------------------------------------------------------------
# Processing Files
#---------------------------------------------------------------------------------------------------------
pl.Config.set_io_thread_count(24)  # Default is usually 2

for filename in meter_readings_filenames:
    input_path = os.path.join(meter_readings_directory, filename)
    
    # Skip if file doesn't exist
    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        continue
    
    # Extract date pattern and create new filename
    match = re.search(r"(.*?)(\d{8})\.parquet$", filename)
    if not match:
        print(f"ERROR: Could not parse filename: {filename}")
        continue
    
    base_name, date_part = match.groups()
    output_path = os.path.join(meter_readings_directory, f"{base_name}hourly_{date_part}")
    
    # Process file
    try:
        df = pl.read_parquet(input_path)
        
        hourly_df = df.group_by_dynamic(
            index_column="date",
            every="1h",
            group_by=["ca_id", "city"]
        ).agg([
            pl.col("value").sum().alias("hourly_kWh_sum"),
            pl.col("value").mean().alias("hourly_kWh_mean"),
        ])
        
        hourly_df.write_parquet(output_path, compression="snappy")
        print(f"SUCCESS: Saved hourly data to: {output_path}")

        # 4. EXPLICIT MEMORY CLEANUP
        del df, hourly_df  # Remove references
        gc.collect()  # Force garbage collection
    
    except Exception as e:
        print(f"ERROR: Failed to process '{filename}': {str(e)}")
