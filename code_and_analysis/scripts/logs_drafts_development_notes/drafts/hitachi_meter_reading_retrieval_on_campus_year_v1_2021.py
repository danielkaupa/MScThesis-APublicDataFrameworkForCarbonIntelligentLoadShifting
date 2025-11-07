#---------------------------------------------------------------------------------------------------------
# Importing Libraries
#---------------------------------------------------------------------------------------------------------
import os
import io
import logging
from datetime import datetime

import polars as pl
from sqlalchemy import create_engine


#---------------------------------------------------------------------------------------------------------
# Setting up Directories
#---------------------------------------------------------------------------------------------------------
base_data_directory = os.path.join('..', 'data')
data_save_directory = os.path.join('..', 'data', 'hitachi_copy')

#---------------------------------------------------------------------------------------------------------
# Setting up SQLAlchemy Engine
#---------------------------------------------------------------------------------------------------------

database_IP_on_campus = '146.169.11.239'    # IP address of the database server to be used while on the campus network
# database_IP_off_campus = '146-169-11-239.dsi.ic.ac.uk'  # IP address of the database server to be used while off the campus network
database_name = 'hitachi'  # Name of the database to connect to
database_port = '5432'  # Default port for PostgreSQL databases

database_user = 'daniel'
database_password = 'Iamdaniel00!'
database_IP = database_IP_on_campus

engine = create_engine(f"postgresql://{database_user}:{database_password}@{database_IP}:{database_port}/{database_name}",
                       connect_args={"connect_timeout": 60,
                                        "keepalives": 1,
                                        "keepalives_idle": 30,
                                        "keepalives_interval": 10,
                                        "keepalives_count": 5,
                                    })

#---------------------------------------------------------------------------------------------------------
# Supporting Functions
#---------------------------------------------------------------------------------------------------------
def query_to_polars_copy(sql: str) -> pl.DataFrame:
    """
    Execute a SQL query via COPY TO STDOUT, stream into memory as CSV, then read into Polars.
    """
    # 1) grab a raw DBAPI connection
    raw = engine.raw_connection()
    buf = io.StringIO()

    # 2) wrap the query in a COPY command
    copy_sql = f"""
        COPY (
            {sql.strip().rstrip(';')}
        ) TO STDOUT WITH CSV HEADER
    """

    # 3) push into our buffer
    with raw.cursor() as cur:
        cur.copy_expert(copy_sql, buf)

    # 4) rewind and read into Polars
    buf.seek(0)
    return pl.read_csv(buf, infer_schema_length=1000)

#---------------------------------------------------------------------------------------------------------
# Suppressing Warnings
#---------------------------------------------------------------------------------------------------------

# Set logging level to WARNING to reduce verbosity
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

#---------------------------------------------------------------------------------------------------------
# Queries
#---------------------------------------------------------------------------------------------------------

target_year = 2021

start = f"'{target_year}-01-01'"
end   = f"'{target_year+1}-01-01'"
sql   = f"""
        SELECT ca_id, date, value
        FROM meter_readings
        WHERE date >= {start} AND date < {end}
    """
print(f"Fetching {target_year} data…")
df = query_to_polars_copy(sql)
fn = f"meter_readings_{target_year}_{datetime.now():%Y%m%d_%H%M}.parquet"
path = os.path.join(data_save_directory, fn)
df.write_parquet(path)
print(f"Successfully saved {df.shape[0]} rows to {fn}")

engine.dispose()