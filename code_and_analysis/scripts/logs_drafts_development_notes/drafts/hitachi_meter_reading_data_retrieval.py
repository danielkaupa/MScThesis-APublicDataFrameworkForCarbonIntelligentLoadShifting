# ─────────────────────────────────────────────────────────────────────────────
# FILE: hitachi_meter_reading_data_retrieval.py

# PURPOSE: Retrieve meter reading data for each year from the 'hitachi'
# database and save it as Parquet files.

# USAGE: This script is designed to run on the HPC cluster at
# Imperial College London

# RUN REQUIREMENTS: Requires access to the 'hitachi' database, which is hosted
# on a PostgreSQL server. Requires installation and setup of dependent
# libraries, directories, and environment variables.
# This script specifically requires a sh file to run it on the HPC cluster
# 'hitachi_meter_reading_data_retrieval.sh'

# OUTPUTS: Parquet files containing meter reading data for each year, saved
# in the specified directory. The files should be named in the format
# 'meter_readings_YEAR_TIMESTAMP.parquet'. In this case, we expect data for
# the years 2021, 2022, and 2023 to be processed.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Importing Libraries
# ─────────────────────────────────────────────────────────────────────────────
import os
import io
import logging
from datetime import datetime
import time

from contextlib import closing
import polars as pl
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# Setting up Directories
# ─────────────────────────────────────────────────────────────────────────────
base_directory = os.path.join('..', '..')
hitachi_data_directory = os.path.join(base_directory, 'data', 'hitachi_copy')
meter_readings_directory = os.path.join(hitachi_data_directory,
                                        'meter_primary_files')

# ─────────────────────────────────────────────────────────────────────────────
# Setting up Constants
# ─────────────────────────────────────────────────────────────────────────────

max_workers = 12  # Number of threads to use for concurrent execution
# This database allows a maximum of 100 connections,
# so we set the pool size accordingly
pool_size = 10  # Size of the connection pool for SQLAlchemy
max_retries = 3  # Number of retries for failed queries
retry_delay = 5  # Delay in seconds before retrying a failed query
parquet_compression = "snappy"  # Compression method for Parquet files

# ─────────────────────────────────────────────────────────────────────────────
# Credentials and Database Connection
# ─────────────────────────────────────────────────────────────────────────────

# IP address of the database server to be used while off the campus network
database_IP_off_campus = '146-169-11-239.dsi.ic.ac.uk'
database_name = 'hitachi'  # Name of the database to connect to
database_port = '5432'  # Default port for PostgreSQL databases

database_user = 'daniel'
database_password = 'Iamdaniel00!'
database_IP = database_IP_off_campus

# ─────────────────────────────────────────────────────────────────────────────
# # Setting up SQLAlchemy Engine
# ─────────────────────────────────────────────────────────────────────────────

# enable logging for SQLAlchemy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name=__name__)
logging.getLogger(name='sqlalchemy.engine').setLevel(level=logging.WARNING)

# create engine with connection pooling and retry logic
engine = create_engine(
    # Define the connection string for SQLAlchemy
    url=(
        f"postgresql://{database_user}:{database_password}"
        f"@{database_IP}:{database_port}/{database_name}"
    ),
    pool_size=pool_size,
    max_overflow=0,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args={
        # Additional connection arguments for PostgreSQL
        # Make sure that the connection does not hang indefinitely
        "connect_timeout": 60,
        # Enable TCP keepalive to detect dead connections
        "keepalives": 1,
        # Seconds of inactivity before sending first keepalive probe
        "keepalives_idle": 30,
        # Seconds between keepalive probes once the initial idle period passes
        "keepalives_interval": 10,
        # Number of failed probes before considering the connection dead
        "keepalives_count": 5,
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Supporting Functions
# ─────────────────────────────────────────────────────────────────────────────


def query_to_polars(sql: str, engine, schema_infer_range: int = 100):
    """
    Executes a SQL query and returns the results as a Polars DataFrame.

    Parameters:
    ------------
    sql: str
        The SQL query to execute.
    engine: sqlalchemy.engine.Engine
        The SQLAlchemy engine to use for the connection.
    schema_infer_range: int, optional
        The number of rows to use for inferring the schema of the DataFrame.
        Default is 100. This value is then passed to the 'infer_schema_length'
        parameter of the Polars DataFrame constructor.

    Returns:
    ------------
    pl.DataFrame
        The results of the query as a Polars DataFrame.
    """
    # Establish a connection to the database
    with engine.connect() as conn:
        # Execute the SQL query
        # (text() function is used to safely handle SQL queries)
        result = conn.execute(text(text=sql))
        # Fetch all the results from the executed query
        rows = result.fetchall()
        # Get the column names from the result set
        columns = result.keys()

    # Create a Polars DataFrame from the fetched rows and column names
    return pl.DataFrame(data=rows,
                        schema=columns,
                        infer_schema_length=schema_infer_range)


def save_to_parquet(df: pl.DataFrame, year: int, save_dir: str,
                    compression_method="snappy") -> str:
    """
    Save DataFrame to parquet file with timestamp.

    Parameters:
    ----------
    df : pl.DataFrame
        The Polars DataFrame to save.
    year : int
        The year associated with the data, used in the filename.
    save_dir : str
        The directory where the parquet file will be saved.
    compression_method : str, optional
        The compression method to use for the parquet file.
        Default is "snappy".

    Returns:
    -------
    filepath : str
        The file path where the DataFrame was saved.
    """
    # Retrieve current timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Create the filename using the year and timestamp
    filename = f"meter_readings_{year}_{timestamp}.parquet"
    # Ensure the save directory exists
    os.makedirs(name=save_dir, exist_ok=True)
    # Construct the full file path
    filepath = os.path.join(save_dir, filename)

    # Save the DataFrame to a parquet file with the specified compression
    df.write_parquet(
        file=filepath,
        compression=compression_method,
        statistics=True,
    )
    return filepath


def retrieve_years_data_copy(year: int) -> pl.DataFrame:
    """
    This function constructs a SQL query to select meter readings for the
    specified year, executes it using the COPY command to export the data in
    CSV format, reads it into a Polars DataFrame, and returns the DataFrame.

    Parameters:
    ----------
    year : int
        The year for which to retrieve the meter reading data.

    Returns:
    -------
    pl.DataFrame
        A Polars DataFrame containing the meter reading data for
        the specified year.
    """
    # Construct the SQL query to retrieve meter readings for the specified year
    sql = f"""
        COPY (
            SELECT ca_id, date, value, city
            FROM meter_readings
            WHERE date >= '{year}-01-01'
              AND date < '{year+1}-01-01'
            ORDER BY date
        ) TO STDOUT WITH (FORMAT CSV, HEADER)
    """
    # Use a raw connection to execute the COPY command
    with closing(thing=engine.raw_connection()) as raw_conn:
        # Use a context manager to create a cursor and execute the COPY command
        with raw_conn.cursor() as cursor:
            # Execute the COPY command to export data to a CSV format
            with io.StringIO() as buffer:
                # Use the COPY command to write the data to the buffer
                cursor.copy_expert(sql, buffer)
                buffer.seek(0)
                return pl.read_csv(source=buffer, infer_schema_length=10_000)


def process_year_with_retry(year: int, save_dir: str,
                            compression_method: str = "snappy",
                            attempt: int = 1,
                            max_retries: int = max_retries,
                            retry_delay: int = retry_delay
                            ) -> tuple[int, int, str]:
    """
    Process a year of meter reading data with retry logic.

    Parameters:
    ----------
    year : int
        The year to process.
    save_dir : str
        The directory where the processed data will be saved.
    compression_method : str, optional
        The compression method to use for saving the data. Default is "snappy".
    attempt : int, optional
        The current attempt number for retrying the operation. Default is 1.

    Returns:
    -------
    tuple(int,str,str,int,int,int)
        A tuple containing the year, save_dir, compression_method, attempt
        number, max retries, and retry delay.

    """
    try:
        df = retrieve_years_data_copy(year=year)
        filepath = save_to_parquet(df=df, year=year, save_dir=save_dir,
                                   compression_method=compression_method)
        return (year, df.shape[0], f"Saved to {filepath}")
    except Exception as e:
        if attempt < max_retries:
            logger.warning(msg=(f"Retry {attempt} for year {year} "
                                f"after error: {str(object=e)}"))
            time.sleep(retry_delay * attempt)
            return process_year_with_retry(
                    year=year,
                    save_dir=save_dir,
                    compression_method=compression_method,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    retry_delay=retry_delay
                )
        return (year, 0,
                f"Failed after {max_retries} attempts: {str(object=e)}")

# ─────────────────────────────────────────────────────────────────────────────
# Queries
# ─────────────────────────────────────────────────────────────────────────────


try:
    with engine.connect() as conn:
        conn.execute(statement=text(text="SELECT 1"))
    logger.info(msg="Database connection successful")
except Exception as e:
    logger.error(msg=f"Database connection failed: {str(object=e)}")
    raise

# get number of available years in the meter readings table
# Check only 10000 random records to find years
years_df = query_to_polars(sql="""
    SELECT DISTINCT EXTRACT(YEAR FROM date)::INT AS year
    FROM meter_readings
    TABLESAMPLE SYSTEM(0.1)  -- Adjust sample percentage
    ORDER BY year
    LIMIT 10000
""", engine=engine)

years = years_df["year"].to_list()
year_count = len(years)
print(f"Available years (sampled): {years}")
logger.info(msg=f"[{year_count}] years of data available spanning\n\t {years}")

logger.info(msg="Processing...")
completed = 0
total_rows = 0
failed_years = []

# creates threads to process years concurrently
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # creates a "future" for each year to be processed (dictionary mapping and
    # placeholder for result to be returned)
    futures = {
        executor.submit(process_year_with_retry,
                        year,
                        meter_readings_directory,
                        parquet_compression
                        ): year
        for year in years
    }

    # as completed enables results to be returned as they are ready
    for future in as_completed(fs=futures):
        # get the year for the current future
        year = futures[future]
        # runs function
        try:
            year_result, rows, status = future.result()

            if rows > 0:
                completed += 1
                total_rows += rows
                logger.info(msg=f"-> Processed year [{year}] with [{rows}]"
                            f" rows and status | [{status}]")
            else:
                failed_years.append(year)
                logger.error(msg=(f"-> Failed to process year"
                             f" [{year}] with status | [{status}]"))

            logger.info(msg=f"\t[{completed}/{year_count}] years completed")

        except Exception as e:
            failed_years.append(year)
            logger.error(msg=(f"-> Exception processing year [{year}]: "
                         f"{str(object=e)}"))

logger.info(msg="\nProcessing complete")
logger.info(msg=(f"Successfully processed {total_rows:,} rows from"
            f" {completed} years"))

if failed_years:
    logger.warning(msg=(f"Failed to process {len(failed_years)} years:"
                   f"{failed_years}"))

engine.dispose()
