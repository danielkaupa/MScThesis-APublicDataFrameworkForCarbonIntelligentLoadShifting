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
import io
import subprocess
import tempfile


#---------------------------------------------------------------------------------------------------------
# Credentials and Connection Information
#---------------------------------------------------------------------------------------------------------

database_IP_on_campus = '146.169.11.239'    # IP address of the database server to be used while on the campus network
# database_IP_off_campus = '146-169-11-239.dsi.ic.ac.uk'  # IP address of the database server to be used while off the campus network
database_name = 'hitachi'  # Name of the database to connect to
database_port = '5432'  # Default port for PostgreSQL databases

database_user = 'daniel'
database_IP = database_IP_on_campus

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
# Running Query
#---------------------------------------------------------------------------------------------------------

for year in [2021, 2022, 2023]:

    # Create the base SQL query to fetch data for a specific year
    start = f"'{year}-01-01'"
    end   = f"'{year+1}-01-01'"
    copy_sql = f"""
    COPY (
    SELECT ca_id, date, value
        FROM meter_readings
    WHERE date >= {start}
        AND date <  {end}
    ) TO STDOUT WITH (FORMAT CSV, HEADER TRUE)
    """

    # Create the psql command to execute the COPY command
    psql_cmd = [
        "psql",
        f"--host={database_IP}",
        f"--port={database_port}",
        f"--username={database_user}",
        f"--dbname={database_name}",
        "--no-align",          # plain CSV
        "--field-separator=,", # comma sep
        "--quiet",             # suppress diagnostics
        "-c", copy_sql
    ]

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_csv:
        proc = subprocess.Popen(psql_cmd, stdout=tmp_csv, stderr=subprocess.PIPE, text=True)
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read()
            # cleanup then error
            tmp_csv_path = tmp_csv.name
            os.remove(tmp_csv_path)
            raise RuntimeError(f"psql COPY failed (exit {rc}):\n{err}")
        tmp_csv_path = tmp_csv.name

    df = pl.read_csv(tmp_csv_path, infer_schema_length=50_000)

    os.remove(tmp_csv_path)

    fn = f"meter_readings_{year}_{datetime.now():%Y%m%d_%H%M}.parquet"
    path = os.path.join(data_save_directory, fn)
    df.write_parquet(path)
