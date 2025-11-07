#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------
import os
import io
import subprocess
import tempfile
from datetime import datetime

import polars as pl

#---------------------------------------------------------------------------------------------------------
# Setting up Directories
#---------------------------------------------------------------------------------------------------------
base_directory = os.path.join('..', '..')
data_save_directory = os.path.join(base_directory,'data', 'hitachi_copy')
meter_readings_directory = os.path.join(data_save_directory, 'meter_primary_files')


#---------------------------------------------------------------------------------------------------------
# Credentials and Connection Information
#---------------------------------------------------------------------------------------------------------

# database_IP_on_campus = '146.169.11.239'    # IP address of the database server to be used while on the campus network
database_IP_off_campus = '146-169-11-239.dsi.ic.ac.uk'  # IP address of the database server to be used while off the campus network
database_name = 'hitachi'  # Name of the database to connect to
database_port = '5432'  # Default port for PostgreSQL databases

database_user = 'daniel'
database_IP = database_IP_off_campus

#---------------------------------------------------------------------------------------------------------
# Queries
#---------------------------------------------------------------------------------------------------------

target_year = 2022

for month in range(1, 13):
    # Create the base SQL query to fetch data for a specific month
    start = f"{target_year}-{month:02d}-01"
    end = f"{target_year}-{month+1:02d}-01" if month < 12 else f"{target_year+1}-01-01"
        
    copy_sql = f"""
    COPY (
        SELECT ca_id, date, value, city
        FROM meter_readings
        WHERE date >= '{start}'
        AND date < '{end}'
    ) TO STDOUT WITH (FORMAT CSV, HEADER)
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
        proc = subprocess.Popen(psql_cmd, stdout=tmp_csv, stderr=subprocess.PIPE)
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

    fn = f"meter_readings_{target_year}_{month}_{datetime.now():%Y%m%d_%H%M}.parquet"
    path = os.path.join(meter_readings_directory, fn)
    print(f"Successfully retrieved data for {target_year}-{month:02d} and saved to {path}")
    df.write_parquet(path)
