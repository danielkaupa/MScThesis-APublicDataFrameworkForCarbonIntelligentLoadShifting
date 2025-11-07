#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------
import sqlalchemy
from sqlalchemy import create_engine       
from sqlalchemy import inspect             
from sqlalchemy import text                
# from sqlalchemy.orm import  sessionmaker, Session
# from geoalchemy2 import Geometry       
from sqlalchemy.exc import SAWarning  
from graphviz import Digraph            
import psycopg2                         
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import os
import warnings
import logging
from datetime import datetime

#---------------------------------------------------------------------------------------------------------
# Setting up Directories
#---------------------------------------------------------------------------------------------------------
os.makedirs('data', exist_ok=True)  # Create a directory to save the data files if it doesn't exist
os.makedirs('data/hitachi_copy', exist_ok=True)  # Create a subdirectory for this copy of the hitachi database
os.makedirs('images', exist_ok=True)  # Create a directory to save the images if it doesn't exist
os.makedirs('images/hitachi', exist_ok=True)  # Create a subdirectory for images related to the hitachi data and database
os.makedirs('images/hitachi/diagrams', exist_ok=True)  # Create a subdirectory for diagrams related to the hitachi data and database

base_data_directory = 'data'  # Base directory where the dataframes will be saved
data_save_directory = 'data/hitachi_copy'  # Directory where the dataframes will be saved
img_save_directory = 'images/hitachi/diagrams'  # Directory where the images will be saved

#---------------------------------------------------------------------------------------------------------
# Setting up SQLAlchemy Engine
#---------------------------------------------------------------------------------------------------------

# database_IP_on_campus = '146.169.11.239'    # IP address of the database server to be used while on the campus network
database_IP_off_campus = '146-169-11-239.dsi.ic.ac.uk'  # IP address of the database server to be used while off the campus network
database_name = 'hitachi'  # Name of the database to connect to
database_port = '5432'  # Default port for PostgreSQL databases

database_user = 'daniel'
database_password = 'Iamdaniel00!'
database_IP = database_IP_off_campus

engine = create_engine(f"postgresql://{database_user}:{database_password}@{database_IP}:{database_port}/{database_name}")
inspector = inspect(engine)  # Create an inspector object to inspect the database schema

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

#---------------------------------------------------------------------------------------------------------
# Suppressing Warnings
#---------------------------------------------------------------------------------------------------------
# Set logging level to WARNING to reduce verbosity
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

# Suppress SQLAlchemy warnings - uncomment this line if you want to ignore SQLAlchemy warnings
# warnings.filterwarnings("ignore", category=SAWarning)

#---------------------------------------------------------------------------------------------------------
# Queries
#---------------------------------------------------------------------------------------------------------

# Query to get meter readings for the year 2021
meter_readings_table_2021_pldf = query_to_polars("""
    SELECT ca_id, date, value FROM meter_readings
    WHERE date >= '2021-01-01' AND date < '2022-01-01'
""", engine)

# Save the 2021 data
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Current timestamp for file naming (YYYYMMDD_HHMM)
meter_readings_table_2021_pldf.write_parquet(os.path.join(data_save_directory, f'meter_readings_2021_{timestamp}.parquet'))

# Query to get meter readings for the year 2022
meter_readings_table_2022_pldf = query_to_polars("""
    SELECT ca_id, date, value FROM meter_readings
    WHERE date >= '2022-01-01' AND date < '2023-01-01'
""", engine)    

# Save the 2022 data
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Current timestamp for file naming (YYYYMMDD_HHMM)
meter_readings_table_2022_pldf.write_parquet(os.path.join(data_save_directory, f'meter_readings_2022_{timestamp}.parquet'))

# Query to get meter readings for the year 2023
meter_readings_table_2023_pldf = query_to_polars("""
    SELECT ca_id, date, value FROM meter_readings
    WHERE date >= '2023-01-01' AND date < '2024-01-01'
""", engine)  

# Save the 2023 data
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Current timestamp for file naming (YYYYMMDD_HHMM)
meter_readings_table_2023_pldf.write_parquet(os.path.join(data_save_directory, f'meter_readings_2023_{timestamp}.parquet'))

# Query to get meter readings for all years
meter_readings_table_full_pldf = query_to_polars("""
    SELECT ca_id, date, value FROM meter_readings
""", engine)

# Save the full meter readings data
timestamp = datetime.now().strftime("%Y%m%d_%H%M")  # Current timestamp for file naming (YYYYMMDD_HHMM)
meter_readings_table_full_pldf.write_parquet(os.path.join(data_save_directory, f'meter_readings_table_full_{timestamp}.parquet'))

# dispose of the engeine to close the connection
engine.dispose()