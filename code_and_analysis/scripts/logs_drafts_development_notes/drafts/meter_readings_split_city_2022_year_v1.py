#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------
import os
from datetime import datetime

import polars as pl

#---------------------------------------------------------------------------------------------------------
# Setting up Directories, Defining File Name and Path
#---------------------------------------------------------------------------------------------------------

# directories
base_directory = os.path.join('..', '..')
data_save_directory = os.path.join(base_directory,'data', 'hitachi_copy')
meter_readings_directory = os.path.join(data_save_directory, 'meter_primary_files')

# file names
meter_readings_all_years_filename = "meter_readings_all_years_20250704"

# file paths
meter_readings_all_years_path = os.path.join(meter_readings_directory, f"{meter_readings_all_years_filename}.parquet")

#---------------------------------------------------------------------------------------------------------
# Reading Data and Merging Meter Readings from Different Years
#---------------------------------------------------------------------------------------------------------

# Read the Parquet file
meter_readings_all_years_pldf = pl.read_parquet(meter_readings_all_years_path)

# Filter to delhi and mumbai specific data
meter_readings_delhi = meter_readings_all_years_pldf.filter(
    (pl.col("city") == "delhi")
)
meter_readings_mumbai = meter_readings_all_years_pldf.filter(
    (pl.col("city") == "mumbai")
)

# Filter to first and second half of 2022 for each city
meter_readings_delhi_first_half_2022 = meter_readings_delhi.filter(
    (pl.col("date") >= "2022-01-01") & (pl.col("date") < "2022-07-01")
)
meter_readings_delhi_second_half_2022 = meter_readings_delhi.filter(
    (pl.col("date") >= "2022-07-01") & (pl.col("date") < "2023-01-01")
)
meter_readings_mumbai_first_half_2022 = meter_readings_mumbai.filter(
    (pl.col("date") >= "2022-01-01") & (pl.col("date") < "2022-07-01")
)
meter_readings_mumbai_second_half_2022 = meter_readings_mumbai.filter(
    (pl.col("date") >= "2022-07-01") & (pl.col("date") < "2023-01-01")
)

#---------------------------------------------------------------------------------------------------------
# Filenames
#---------------------------------------------------------------------------------------------------------

# Get date for filename
meter_readings_all_years_filename = os.path.basename(meter_readings_all_years_filename)
retrieval_date = meter_readings_all_years_filename.split("_")[-1].split(".")[0] # Extracting date from filename

# Delhi Data Filenames and Paths
meter_readings_delhi_all_years_filename = f"meter_readings_delhi_all_years_{retrieval_date}.parquet"
meter_readings_delhi_all_years_path = os.path.join(
    meter_readings_directory, meter_readings_delhi_all_years_filename
)
meter_readings_delhi_2022_Q1and2_filename = f"meter_readings_delhi_2022_Q1and2_{retrieval_date}.parquet"
meter_readings_delhi_2022_Q1and2_path = os.path.join(
    meter_readings_directory, meter_readings_delhi_2022_Q1and2_filename
) 
meter_readings_delhi_2022_Q3and4_filename = f"meter_readings_delhi_2022_Q3and4_{retrieval_date}.parquet"
meter_readings_delhi_2022_Q3and4_path = os.path.join(
    meter_readings_directory, meter_readings_delhi_2022_Q3and4_filename
) 

# Mumbai Data Filenames and Paths
meter_readings_mumbai_all_years_filename = f"meter_readings_mumbai_all_years_{retrieval_date}.parquet"
meter_readings_mumbai_all_years_path = os.path.join(
    meter_readings_directory, meter_readings_mumbai_all_years_filename
)
meter_readings_mumbai_2022_Q1and2_filename = f"meter_readings_mumbai_2022_Q1and2_{retrieval_date}.parquet"
meter_readings_mumbai_2022_Q1and2_path = os.path.join(
    meter_readings_directory, meter_readings_mumbai_2022_Q1and2_filename
)
meter_readings_mumbai_2022_Q3and4_filename = f"meter_readings_mumbai_2022_Q3and4_{retrieval_date}.parquet"
meter_readings_mumbai_2022_Q3and4_path = os.path.join(
    meter_readings_directory, meter_readings_mumbai_2022_Q3and4_filename
) 

#---------------------------------------------------------------------------------------------------------
# Saving Filtered Data to Parquet Files
#---------------------------------------------------------------------------------------------------------

# Save the delhi data for all years
meter_readings_delhi.write_parquet(
    meter_readings_delhi_all_years_path, compression="snappy"
)
print(f"\n--> Saved the delhi data for all years to: '{meter_readings_delhi_all_years_path}'")  
# Save the delhi data for first and second half of 2022
meter_readings_delhi_first_half_2022.write_parquet(
    meter_readings_delhi_2022_Q1and2_path, compression="snappy"
)
print(f"\n--> Saved the delhi data for first half of 2022 to: '{meter_readings_delhi_2022_Q1and2_path}'")
meter_readings_delhi_second_half_2022.write_parquet(
    meter_readings_delhi_2022_Q3and4_path, compression="snappy"
)
print(f"\n--> Saved the delhi data for second half of 2022 to: '{meter_readings_delhi_2022_Q3and4_path}'")   

# Save the mumbai data for all years
meter_readings_mumbai.write_parquet(
    meter_readings_mumbai_all_years_path, compression="snappy"
)
print(f"\n--> Saved the mumbai data for all years to: '{meter_readings_mumbai_all_years_path}'")  
# Save the mumbai data for first and second half of 2022
meter_readings_mumbai_first_half_2022.write_parquet(
    meter_readings_mumbai_2022_Q1and2_path, compression="snappy"
)
print(f"\n--> Saved the mumbai data for first half of 2022 to: '{meter_readings_mumbai_2022_Q1and2_path}'")
meter_readings_mumbai_second_half_2022.write_parquet(
    meter_readings_mumbai_2022_Q3and4_path, compression="snappy"
)
print(f"\n--> Saved the mumbai data for second half of 2022 to: '{meter_readings_mumbai_2022_Q3and4_path}'")            

print("\n--> Successfully saved all filtered data to Parquet files.")