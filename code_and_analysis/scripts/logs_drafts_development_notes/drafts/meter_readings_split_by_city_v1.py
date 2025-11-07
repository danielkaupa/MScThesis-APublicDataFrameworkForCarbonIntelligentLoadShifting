# ────────────────────────────────────────────────────────────────────────────
# author: Daniel Kaupa
# date: 2025-07-15
# summary: This script is designed to take meter readings data divided into individual years, months, etc,
#          and split them further by city.
# instructions: Change the names of the files in the `data_save_directory` variable to match your local setup
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ────────────────────────────────────────────────────────────────────────────

import os
import polars as pl
from mpi4py import MPI

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

#────────────────────────────────────────────────────────────────────────────
# Helper Functions
#────────────────────────────────────────────────────────────────────────────

def get_unique_cities(file_path: str) -> list:
    """
    Function to retrieve the unique cities present in a meter readings file.

    Parameters:
    -----------
    file_path: str
        The path to the Parquet file containing meter readings.

    Returns:
    --------
    list
        A list of unique city names found in the file.

    """
    try:
        return(
            pl.scan_parquet(file_path)
            .select(pl.col("city").unique())
            .collect()
            .get_column("city")
            .to_list()
        )
    except Exception as e:
        print(f"RANK {rank} ERROR processing unique cities in {file_path}: {e}")
        return []

def filter_and_save_unique_city_file(file_path: str, city: str, output_dir: str) -> None:
    """
    Fuction to save the meter readings for a specific city to a new Parquet file.

    Parameters:
    -----------
    file_path: str
        The path to the Parquet file containing meter readings.
    city: str
        The name of the city for which the data should be filtered.
    output_dir: str
        The directory where the filtered data should be saved.

    Returns:
    --------
    None
        The function saves the filtered data to a new Parquet file in the specified directory.
    """
    try:
        # sanitize the city name for use in file names
        city_sanitized = city.lower().replace(" ", "_")
        # grab original file name
        original_name = os.path.basename(file_path)
        # add city name to the file name
        output_filename = original_name.replace("meter_readings_", f"meter_readings_{city_sanitized}_")
        # create output path
        output_path = os.path.join(output_dir, output_filename)

        # read the file, filter by city, and save to new file
        (
            pl.scan_parquet(file_path)
            .filter(pl.col("city") == city)
            .sink_parquet(
                output_path,
                compression="snappy",
                statistics=True,
            )
        )
        print(f"Rank {rank} SUCCESS: Saved {output_path}")
    except Exception as e:
        print(f"Rank {rank} ERROR saving city {city} from {file_path}: {e}")


# ────────────────────────────────────────────────────────────────────────────
# Processing Meter Readings Files
# ────────────────────────────────────────────────────────────────────────────

# broadcast the list of filenames to all ranks (processes)
file_list = comm.bcast(meter_readings_filenames if rank ==0 else None, root=0)

# assign the files across ranks (processes)
tasks = file_list[rank::size] if file_list else []

for filename in tasks:
    input_path = os.path.join(meter_readings_directory, filename)
    try:
        # Ensure the file exists before processing
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"File not found: {input_path}")

        # Get unique cities from the file
        unique_cities = get_unique_cities(input_path)
        
        # Filter and save data for each unique city
        for city in unique_cities:
            filter_and_save_unique_city_file(input_path, city, meter_readings_directory)

    except FileNotFoundError as e:
        print(f"Rank {rank} FILE NOT FOUND: {e}")
    except pl.exceptions.PolarsError as e:
        print(f"Rank {rank} POLARS ERROR in {filename}: {e}")
    except Exception as e:
        print(f"Rank {rank} UNKNOWN ERROR in {filename}: {e}")

comm.Barrier()  # Ensure all ranks have completed the processing before finishing
if rank == 0:
    print(f"=== Split complete on {size} ranks ===")

MPI.Finalize()  # Finalize MPI to clean up resources