#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------
import os
import io
import logging
from datetime import datetime
import time
import tempfile

from contextlib import closing
import polars as pl
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

#---------------------------------------------------------------------------------------------------------
# Setting up Directories
#---------------------------------------------------------------------------------------------------------
base_directory = os.path.join('..', '..')
data_save_directory = os.path.join(base_directory,'data', 'hitachi_copy')
meter_readings_directory = os.path.join(data_save_directory, 'meter_primary_files')

#---------------------------------------------------------------------------------------------------------
# Setting up Constants
#---------------------------------------------------------------------------------------------------------

max_workers = 12  # Number of threads to use for concurrent execution
# This database allows a maximum of 100 connections, so we set the pool size accordingly
pool_size = 10  # Size of the connection pool for SQLAlchemy
max_retries = 3  # Number of retries for failed queries
retry_delay = 5  # Delay in seconds before retrying a failed query
parquet_compression = "snappy"  # Compression method for Parquet files

#---------------------------------------------------------------------------------------------------------
# Credentials and Database Connection
#---------------------------------------------------------------------------------------------------------

# database_IP_on_campus = '146.169.11.239'    # IP address of the database server to be used while on the campus network
database_IP_off_campus = '146-169-11-239.dsi.ic.ac.uk'  # IP address of the database server to be used while off the campus network
database_name = 'hitachi'  # Name of the database to connect to
database_port = '5432'  # Default port for PostgreSQL databases

database_user = 'daniel'
database_password = 'Iamdaniel00!'
database_IP = database_IP_off_campus

#---------------------------------------------------------------------------------------------------------
# Setting up SQLAlchemy Engine
#---------------------------------------------------------------------------------------------------------

# enable logging for SQLAlchemy
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

# create engine with connection pooling and retry logic
engine = create_engine(f"postgresql://{database_user}:{database_password}@{database_IP}:{database_port}/{database_name}",
                       pool_size=pool_size,
                       max_overflow=0,
                       pool_pre_ping=True,
                       pool_recycle=3600,
                       connect_args={"connect_timeout": 60,
                                        "keepalives": 1,
                                        "keepalives_idle": 30,
                                        "keepalives_interval": 10,
                                        "keepalives_count": 5,
                                    })

#---------------------------------------------------------------------------------------------------------
# Supporting Functions
#---------------------------------------------------------------------------------------------------------
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
        The number of rows to use for inferring the schema of the DataFrame. Default is 100.
        This value is then passed to the 'infer_schema_length' parameter of the Polars DataFrame constructor.
    
    Returns:
    ------------
    pl.DataFrame
        The results of the query as a Polars DataFrame.
    """
    with engine.connect() as conn:          # Establish a connection to the database
        result = conn.execute(text(sql))    # Execute the SQL query (text() function is used to safely handle SQL queries)
        rows = result.fetchall()            # Fetch all the results from the executed query
        columns = result.keys()             # Get the column names from the result set
    return pl.DataFrame(rows, schema=columns, infer_schema_length=schema_infer_range)  # Create a Polars DataFrame from the fetched rows and column names

def save_to_parquet(df: pl.DataFrame, year: int, save_dir:str, compression_method="snappy") -> str:
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
        The compression method to use for the parquet file. Default is "snappy".
                
    Returns:
    -------
    filepath : str
        The file path where the DataFrame was saved.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"meter_readings_{year}_{timestamp}.parquet"
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    
    df.write_parquet(
        filepath,
        compression=parquet_compression,
        statistics=True,
    )
    return filepath

def retrieve_years_data_copy(year: int) -> pl.DataFrame:
    """
    Retrieve a year's worth of meter reading data using the COPY command for efficiency.

    Parameters:
    ----------
    year : int
        The year for which to retrieve the meter reading data.
    Returns:
    -------
    pl.DataFrame
        A Polars DataFrame containing the meter reading data for the specified year.
    This function constructs a SQL query to select meter readings for the specified year,
    executes it using the COPY command to export the data in CSV format, and reads it into a Polars DataFrame.
    """
    sql = f"""
        COPY (
            SELECT ca_id, date, value, city
            FROM meter_readings
            WHERE date >= '{year}-01-01'
              AND date < '{year+1}-01-01'
            ORDER BY date
        ) TO STDOUT WITH (FORMAT CSV, HEADER)
    """
    with closing(engine.raw_connection()) as raw_conn:
        with raw_conn.cursor() as cursor:
            with io.StringIO() as buffer:
                cursor.copy_expert(sql, buffer)
                buffer.seek(0)
                return pl.read_csv(buffer, infer_schema_length=10_000)
        
def process_year_with_retry(year: int, save_dir: str, compression_method: str="snappy", attempt: int = 1, max_retries: int=max_retries, retry_delay: int=retry_delay) -> tuple[int, int, str]:
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
        A tuple containing the year, save_dir, compression_method, attempt number, max retries, and retry delay.
    
    """
    try:
        df = retrieve_years_data_copy(year)
        filepath = save_to_parquet(df, year, save_dir, compression_method)
        return (year, df.shape[0], f"Saved to {filepath}")
    except Exception as e:
        if attempt < max_retries:
            logger.warning(f"Retry {attempt} for year {year} after error: {str(e)}")
            time.sleep(retry_delay * attempt)
            return process_year_with_retry(year,
                                            save_dir,
                                            compression_method,
                                            attempt + 1,
                                            max_retries,
                                            retry_delay)
        return (year, 0, f"Failed after {max_retries} attempts: {str(e)}")

#-------------------------------------------------------------------------------------------------------
# Queries
#---------------------------------------------------------------------------------------------------------

try:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection successful")
except Exception as e:
    logger.error(f"Database connection failed: {str(e)}")
    raise

# get number of available years in the meter readings table
# Check only 1000 random records to find years
years_df = query_to_polars("""
    SELECT DISTINCT EXTRACT(YEAR FROM date)::INT AS year
    FROM meter_readings
    TABLESAMPLE SYSTEM(0.1)  -- Adjust sample percentage
    ORDER BY year
    LIMIT 1000
""", engine)
years = years_df["year"].to_list()
year_count = len(years)
print(f"Available years (sampled): {years}")
# Main processing loop

logger.info(f"[{year_count}] years of data available spanning \n\t {years}")
logger.info(f"Processing...")

completed = 0
total_rows = 0
failed_years = []

# createds threads to process years concurrently
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # creates a "future" for each year to be processed (dictionary mapping and placeholder for result to be returned) )
    futures = {
        executor.submit(process_year_with_retry, year, meter_readings_directory,parquet_compression): year 
        for year in years
    }

    # as completed enables results to be returned as they are ready
    for future in as_completed(futures):
        # get the year for the current future
        year = futures[future]
        # runs function 
        try:
            year_result, rows, status = future.result()

            if rows > 0:
                completed += 1
                total_rows += rows
                logger.info(f"-> Processed year [{year}] with [{rows}] rows and status | [{status}]")
            else:
                failed_years.append(year)
                logger.error(f"-> Failed to process year [{year}] with status | [{status}]")
            
            logger.info(f"\t[{completed}/{year_count}] years completed")

        except Exception as e:
            failed_years.append(year)
            logger.error(f"-> Exception processing year [{year}]: {str(e)}")
        
logger.info(f"\nProcessing complete")
logger.info(f"Successfully processed {total_rows:,} rows from {completed} years")
if failed_years:
    logger.warning(f"Failed to process {len(failed_years)} years: {failed_years}")

engine.dispose()
