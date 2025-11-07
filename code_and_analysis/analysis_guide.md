# Analysis Guide

## Run Order

---

### [STEP 1] : Data Retrieval

---

This section details the processes and code used to retrieve data and perform initial processing for analysis.

The notebooks and scripts are listed in the order in which they should be run as some operations are dependent on previous outputs.

**Primary Outputs:**
- customers_20250714_1401.parquet
- grid_readings_20250714_1401.parquet
- meter_readings_2021_20250714_2015_formatted.parquet
- meter_readings_2022_20250714_2324_formatted.parquet
- meter_readings_2023_20250714_2039_formatted.parquet
- weather_20250714_1401.parquet


**Secondary Outputs:**
- meter_readings_all_years_20250714_2015_formatted.parquet
- meter_readings_2021_Q4_20250714_2015_formatted.parquet
- meter_readings_2022_Q1_20250714_2324_formatted.parquet
- meter_readings_2022_Q2_20250714_2324_formatted.parquet
- meter_readings_2022_Q3_20250714_2324_formatted.parquet
- meter_readings_2022_Q4_20250714_2324_formatted.parquet
- meter_readings_2023_Q1_20250714_2039_formatted.parquet
- meter_readings_delhi_2021_Q4_20250714_2015_formatted.parquet
- meter_readings_delhi_2022_Q1_20250714_2324_formatted.parquet
- meter_readings_delhi_2022_Q2_20250714_2324_formatted.parquet
- meter_readings_delhi_2022_Q3_20250714_2324_formatted.parquet
- meter_readings_delhi_2022_Q4_20250714_2324_formatted.parquet
- meter_readings_mumbai_2022_Q3_20250714_2324_formatted.parquet
- meter_readings_mumbai_2022_Q4_20250714_2324_formatted.parquet
- meter_readings_mumbai_2023_Q1_20250714_2039_formatted.parquet

<br>

<details>
<summary><strong>step1_hitachi_data_retrieval.ipynb</strong></summary>
<br>

- **Purpose:**
    - Investigates structure and contents of the 'hitachi' database.
    - Retrieves the ERA5-Land, Carbontracker.in, and customer id data from the Data Science Institute's Postgres 'hitachi' database.
- **Inputs:**
    - N/A
- **Outputs:**
    - weather_20250714_1401.parquet: data from the weather table, originally sourced from the ERA5-Land dataset.
    - grid_readings_20250714_1401.parquet: data from the grid_readings table, originally sourced from [carbontracker.in](https://carbontracker.in/).
    - customers_20250714_1401.parquet: data from the customers table containing customer ids and locations.
- **Notes:**
    - The meter readings data is **not** retrieved using this file due to the size of the data.  See the following hpc scripts for retrieval.
- [notebook](step1_hitachi_data_retrieval.ipynb)

<br>
</details>

<details>
<summary><strong>step1_hitachi_meter_reading_data_retrieval.py</strong></summary>
<br>

- **Purpose:**
    - Retrieve meter reading data for each year (2021-2023) from the 'hitachi' database and save as Parquet files.
- **Inputs:**
    - N/A
- **Outputs:**
    - meter_readings_2021_20250714_2015.parquet
    - meter_readings_2022_20250714_2324.parquet
    - meter_readings_2023_20250714_2039.parquet
- [script](scripts/step1_hitachi_meter_reading_data_retrieval.py)

<br>
</details>

<details>
<summary><strong>step1_hitachi_meter_readings_standardisation.py</strong></summary>
<br>

- **Purpose:**
    - Unify the datatypes of the meter readings data to ensure consistency and ease of analysis in later stages.
- **Inputs:**
    - meter_readings_2021_20250714_2015.parquet
    - meter_readings_2022_20250714_2324.parquet
    - meter_readings_2023_20250714_2039.parquet
- **Outputs:**
    - meter_readings_2021_20250714_2015_formatted.parquet
    - meter_readings_2022_20250714_2324_formatted.parquet
    - meter_readings_2023_20250714_2039_formatted.parquet
- [script](scripts/step1_hitachi_meter_readings_standardisation.py)

<br>
</details>

<details>
<summary><strong>step1_hitachi_meter_readings_processing_merge_years.py</strong></summary>
<br>

- **Purpose:**
    - Process yearly meter readings data and merge into a single file.
- **Inputs:**
    - meter_readings_2021_20250714_2015_formatted.parquet
    - meter_readings_2022_20250714_2324_formatted.parquet
    - meter_readings_2023_20250714_2039_formatted.parquet
- **Outputs:**
    - meter_readings_all_years_20250714_2015_formatted.parquet
- [script](scripts/step1_hitachi_meter_readings_processing_merge_years.py)

<br>
</details>

<details>
<summary><strong>step1_hitachi_meter_readings_processing_split_by_city.py</strong></summary>
<br>

- **Purpose:**
    - Process yearly meter readings data and split into different files based on city.
- **Inputs:**
    - meter_readings_2021_20250714_2015_formatted.parquet
    - meter_readings_2022_20250714_2324_formatted.parquet
    - meter_readings_2023_20250714_2039_formatted.parquet
- **Outputs:**
    - meter_readings_delhi_2021_20250714_2015_formatted.parquet
    - meter_readings_delhi_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2023_20250714_2039_formatted.parquet
- [script](scripts/step1_hitachi_meter_readings_processing_split_by_city.py)

<br>
</details>

<details>
<summary><strong>step1_meter_readings_processing_split_to_quarterly.py</strong></summary>
<br>

- **Purpose:**
    - Process meter readings data split by year and city into quarterly files.
- **Inputs:**
    - meter_readings_2021_20250714_2015_formatted.parquet
    - meter_readings_2022_20250714_2324_formatted.parquet
    - meter_readings_2023_20250714_2039_formatted.parquet
    - meter_readings_delhi_2021_20250714_2015_formatted.parquet
    - meter_readings_delhi_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2023_20250714_2039_formatted.parquet
- **Outputs:**
    - meter_readings_2021_Q4_20250714_2015_formatted.parquet
    - meter_readings_2022_Q1_20250714_2324_formatted.parquet
    - meter_readings_2022_Q2_20250714_2324_formatted.parquet
    - meter_readings_2022_Q3_20250714_2324_formatted.parquet
    - meter_readings_2022_Q4_20250714_2324_formatted.parquet
    - meter_readings_2023_Q1_20250714_2039_formatted.parquet
    - meter_readings_delhi_2021_Q4_20250714_2015_formatted.parquet
    - meter_readings_delhi_2022_Q1_20250714_2324_formatted.parquet
    - meter_readings_delhi_2022_Q2_20250714_2324_formatted.parquet
    - meter_readings_delhi_2022_Q3_20250714_2324_formatted.parquet
    - meter_readings_delhi_2022_Q4_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2022_Q3_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2022_Q4_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2023_Q1_20250714_2039_formatted.parquet
- [script](scripts/step1_meter_readings_processing_split_to_quarterly.py)

<br>
</details>

<details>
<summary><strong>ERA5 hourly data on single levels from 1940 to present</strong></summary>
<br>

- **Link:** [Copernicus ERA5 Dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview)
- **Download Instructions:**
    - Variables:
        - 10m u-component of wind
        - 10m v-component of wind
        - 2m temperature
        - Total precipitation
        - Surface net solar radiation
        - Surface solar radiation downwards
        - High cloud cover
        - Medium cloud cover
        - Low cloud cover
        - Total cloud cover
    - Year: Individual requests for 2020-2025
    - Month, Day, Time: All available for the selected year
    - Geographical area:
        - Delhi: Latitude [26.000] to [30.000], Longitude [75.000] to [79.000]
        - Mumbai: Latitude [17.000] to [21.000], Longitude [70.000] to [74.000]
    - Data format: 'grib'
- **Outputs:**
    - Weather data for regions including Delhi and Mumbai spanning 2020-2025
        - 125ae282169904325e8bc153160be150.grib
        - 289f2aac241f8a158ff074a66682452e.grib
        - 554832a6209258041784298e5401a7ab.grib
        - 5aee58993569287064988fbc8ad385dd.grib
        - 5bcc58c42bdde8ce6b147b00099404bc.grib
        - ad36c26a5d6daae43c9aeab1747e078c.grib
        - b4eac1bff8a020500806be638e9d4ab9.grib
        - bc20f736fa82ab5167820d9116ab4859.grib
        - c8a985ffc4908e6597c4498ff596cbad.grib
        - d1313a3f750d6e7bd89dff34b112d8a8.grib
        - de87f0d77e8aeed868c68ac0daae3dc9.grib
        - e23fa435dfdf294eba51378e96410b31.grib

<br>
</details>

---

### [STEP 2] : Initial Data Analysis

---

This section details the processes and code used to perform initial data analysis on the retrieved data.

<details>
<summary><strong>step2_initial_data_analysis.ipynb</strong></summary>
<br>

- **Purpose:**
    - Perform initial data analysis on the customer, weather, grid and meter readings data pulled from the 'hitachi' database.
- **Inputs:**
    - customers_20250714_1401.parquet
    - grid_readings_20250714_1401.parquet
    - meter_readings_20250701_1318.parquet
    - meter_readings_2021_20250714_2015.parquet
    - meter_readings_2022_20250714_2324.parquet
    - meter_readings_2023_20250714_2039.parquet
    - meter_readings_2021_20250714_2015_formatted.parquet
    - meter_readings_2022_20250714_2324_formatted.parquet
    - meter_readings_2023_20250714_2039_formatted.parquet
    - meter_readings_delhi_2021_20250714_2015_formatted.parquet
    - meter_readings_delhi_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2023_20250714_2039_formatted.parquet
    - weather_20250714_1401.parquet
- **Outputs:**
    - [images/delhi_customers_map.png](code_and_analysis/images/delhi_customers_map.png)
    - [images/mumbai_customers_map.png](code_and_analysis/images/mumbai_customers_map.png)
- [notebook](step2_initial_data_analysis.ipynb)

<br>
</details>

<details>
<summary><strong>step2_investigating_data_size_and_types.ipynb</strong></summary>
<br>

- **Purpose:**
    - Explore methods for reducing the overall size of data used in this project, as well as more efficient methods of operating on the data.
- **Inputs:**
    - customers_20250714_1401.parquet
    - grid_readings_20250714_1401.parquet
    - meter_readings_2021_20250714_2015.parquet
    - meter_readings_2022_20250714_2324.parquet
    - meter_readings_2023_20250714_2039.parquet
    - weather_20250714_1401.parquet
- **Outputs:**
    - N/A
- [notebook](step2_investigating_data_size_and_types.ipynb)

<br>
</details>

<details>
<summary><strong>step2_investigating_era5_world_data.ipynb</strong></summary>
<br>

- **Purpose:**
    - Investigate structure and contents of data retrieved from the ERA5 World dataset.
- **Inputs:**
    - 125ae282169904325e8bc153160be150.grib
    - 289f2aac241f8a158ff074a66682452e.grib
    - 554832a6209258041784298e5401a7ab.grib
    - 5aee58993569287064988fbc8ad385dd.grib
    - 5bcc58c42bdde8ce6b147b00099404bc.grib
    - ad36c26a5d6daae43c9aeab1747e078c.grib
    - b4eac1bff8a020500806be638e9d4ab9.grib
    - bc20f736fa82ab5167820d9116ab4859.grib
    - c8a985ffc4908e6597c4498ff596cbad.grib
    - d1313a3f750d6e7bd89dff34b112d8a8.grib
    - de87f0d77e8aeed868c68ac0daae3dc9.grib
    - e23fa435dfdf294eba51378e96410b31.grib
- **Outputs:**
    - N/A
- [notebook](step2_investigating_era5_world_data.ipynb)

<br>
</details>

<details>
<summary><strong>step2_meter_readings_analysis.py</strong></summary>
<br>

- **Purpose:**
    - Analyze meter readings data to extract insights and identify patterns.
- **Inputs:**
    - meter_readings_2021_20250714_2015_formatted.parquet
    - meter_readings_2022_20250714_2324_formatted.parquet
    - meter_readings_2023_20250714_2039_formatted.parquet
    - meter_readings_delhi_2021_20250714_2015_formatted.parquet
    - meter_readings_delhi_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2022_20250714_2324_formatted.parquet
    - meter_readings_mumbai_2023_20250714_2039_formatted.parquet
    - meter_readings_all_years_20250714_formatted.parquet
    - customers_20250714_1401.parquet
- **Outputs:** code_and_analysis/outputs
    1. Individual Files Stats CSV (one row per input file) with customer counts, half-hour totals (and hours/days equivalents), non-zero counts, and per-customer averages (overall and per city).
    2. A Bigfile Usage Stats CSV covering totals and per-customer usage for hour/week/month/year across scopes (all, Delhi, Mumbai, each year, city×year), using coverage-aware day fill to extrapolate missing days, and also reporting an annualized figure (weekly mean ×52).
    3. Per-customer distributions saved as CSVs plus histograms and boxplots for each scope/frequency.
    4. Monthly totals by city: a CSV per year and double-bar charts (Delhi vs Mumbai) for each month with data.
    5. Visuals/metrics: weekday vs weekend per-customer hour curves (line plot + CSV), zero-usage share, top-10% concentration of consumption, and city-scope peak-hour distribution (bar chart).
    6. A coverage table (half-hours present vs expected) per city-year,
    7. Distance analytics per city (nearest-neighbor and sampled pairwise stats) with two histograms and a distance stats CSV.
- [script](scripts/step2_meter_readings_analysis.py)

<br>
 </details>

---

### [STEP 3] : Data Processing

---

#### Individual Dataset Processing


<details>
<summary><strong>step3_processing_era5_land_data.ipynb</strong></summary>
<br>

- **Purpose:**
    - Process the era5 land data downloaded from the hitachi database, including dropping null locations, converting units, adjusting wind direction, adding longitude and latitude

- **Input:**
    - weather_20250714_1401.parquet
- **Output:**
    - weather_20250714_1401_processed.parquet
- [notebook](step3_processing_era5_land_data.ipynb)

<br>
</details>

<details>
<summary><strong>step3_processing_era5_world_data.ipynb</strong></summary>
<br>

- **Purpose:**
    - Process ERA5 World data for analysis and integration with other datasets. Includes converting grib to parquet files, aligning inconsistent time axes across variables, regridding to match the resolution of the ERA5 Land dataset (involves various interpolation methods), converting units.

- **Input:**
    - grib files in 'data/era5/grib_downloads' directory
        - 125ae282169904325e8bc153160be150.grib
        - 289f2aac241f8a158ff074a66682452e.grib
        - 554832a6209258041784298e5401a7ab.grib
        - 5aee58993569287064988fbc8ad385dd.grib
        - 5bcc58c42bdde8ce6b147b00099404bc.grib
        - ad36c26a5d6daae43c9aeab1747e078c.grib
        - b4eac1bff8a020500806be638e9d4ab9.grib
        - bc20f736fa82ab5167820d9116ab4859.grib
        - c8a985ffc4908e6597c4498ff596cbad.grib
        - d1313a3f750d6e7bd89dff34b112d8a8.grib
        - de87f0d77e8aeed868c68ac0daae3dc9.grib
        - e23fa435dfdf294eba51378e96410b31.grib

- **Output:**
    - era5_world_reanalysis_data_2020-2025.parquet

- [notebook](step3_processing_era5_world_data.ipynb)

<br>
</details>


<details>
<summary><strong>step3_processing_grid_readings_data.ipynb</strong></summary>
<br>

- **Purpose:**
    - Process grid readings data for analysis and integration with other datasets. Explores methods for filling gaps in the grid readings dataset. Uses linear interpolation for short gaps, average gradient of neighboring day data for longer gaps.

- **Input:**
    - grid_readings_20250714_1401.parquet

- **Output:**
    - grid_readings_20250714_1401_processed.parquet

- [notebook](step3_processing_grid_readings_data.ipynb)

<br>
 </details>



#### Combining Datasets

<details>
<summary><strong>step3_combining_era5_datasets.ipynb</strong></summary>
<br>

- **Purpose:**
    - Clean and Combine weather datasets
        - fill midnight gaps in ERA5 land with ERA5 World data
        - Fill consecutive gaps for instantaneous measurement variables outright
        - Deaccumulate the cumulative measurements from the land dataset, while avoiding gaps
        - Fill gaps present in now deaccumulated dataset with ERA5 World data
        - Add total cloud cover from ERA5 World data to the combined dataset

- **Input:**
    - weather_20250714_1401_processed.parquet
    - era5_world_reanalysis_data_2020-2025.parquet

- **Output:**
    - weather_data_combined_20250714_1401.parquet
- [notebook](step3_combining_weather_and_grid_data.ipynb)

<br>
 </details>

<details>
<summary><strong>step3_combining_grid_and_weather_data.ipynb</strong></summary>

- **Purpose:**
    - Clean and Combine weather and grid readings datasets so they can be used for generating marginal emission factors
        - Grid data processing includes
            - Aligning time scales so the time is the end of the interval for which data is recorded (mimic electricity data standards)
            - Aggregating to half-hourly intervals (to align with meter readings dataset)
                - Rates are averages, tons is summed
        - Weather data processing includes:
            - interpolating to 30 minutes intervals
                - midpoint average for instantaneous values (temperature, wind speed, cloud cover)
                - speed weighted circular midpoint average for wind direction
                - split into halves to conserve mass for precipitation
                - rate based split for solar to conserve energy, but capture quadratic trends

- **Inputs:**
    - weather_data_combined_20250714_1401.parquet
    - grid_readings_20250714_1401_processed.parquet

- **Output:**
    - weather_and_grid_data_half-hourly_20250714_1401.parquet

- [notebook](code_and_analysis/step3_combining_grid_and_weather_data.ipynb)

<br>
 </details>

---

### [STEP 4] : Marginal Emission Factors Estimation

---

<details>
<summary><strong>[step4_marginal_emissions_eda.ipynb](code_and_analysis/step4_marginal_emissions_eda.ipynb)</strong></summary>
<br>

- **Purpose:**

- **Inputs:**
    - weather_and_grid_data_half-hourly_20250714_1401.parquet

- **Outputs:**
    - marginal_emission_factors_20250714_1401.parquet

<br>
</details>

<details>
<summary><strong>[step3_processing_era5_land_data.ipynb](code_and_analysis/step3_processing_era5_land_data.ipynb)</strong></summary>
<br>

- **Purpose:**

- **Inputs:**

- **Outputs:**


<br>
 </details>

<details>
<summary><strong>[step3_combining_era5_datasets.ipynb](code_and_analysis/step3_combining_era5_datasets.ipynb)</strong></summary>
<br>

- **Purpose:**

- **Inputs:**

- **Outputs:**

<br>
</details>

<details>
<summary><strong>[step3_combining_grid_and_weather_data.ipynb](code_and_analysis/step3_combining_era5_datasets.ipynb)</strong></summary>
<br>

- **Purpose:**
    - Combine the Grid Readings with Cleaned Weather Data
    - Aggregate to half-hourly intervals

- **Inputs:**
    - weather_and_grid_data_half-hourly_20250714_1401.parquet
    - grid_readings_20250714_1401.parquet

- **Outputs:**
    - Primary output:
        - weather_and_grid_data_half-hourly_20250714_1401.parquet
    - Intermediate outputs:
        - grid_readings_20250714_1401_half_hourly.parquet
        - weather_20250714_1401_processed_clean_combined_half_hourly.parquet
        - weather_20250714_1401_processed_clean_combined_half_hourly_step2.parquet
        - weather_20250714_1401_processed_clean_combined_half_hourly_step2_trimmed.parquet
        - grid_readings_20250714_1401_trimmed.parquet

<br>
</details>

**[STEP 4] : Marginal Emissions Calculations**


* What does this notebook do?



* Useful Outputs:



**[STEP 4] : initial_data_analysis.ipynb**
* What does this notebook do?



* Useful Outputs:



**[STEP 6] : initial_data_analysis.ipynb**

* What does this notebook do?



* Useful Outputs:



So first we process the ERA5 data - then we write scripts toproces the weather data - both the world data and the land data

the land data should contain a n analysis on it's missing data


then there is a joining of the data to fill in the gaps..


i should also fix the calculations that were dont by brython on the weathere data..



Then the carbontrackerdata:
* Column Types
* SEt Timezone
* Aggregate using appropriate methodology



Weather Data: - ERA5 Land
* column types
* Set timezone
* de-aggregate fields as necessary
* Change location column to two separate columns for longitude and latitude
* Change wind_direction, wind_speed, and temperature to float32

* Change precipitation to millimeters and float32
* Add solar radiation fields of kWh/m^2 and change measurements tofloat32
* MaKE ADJUSTMENTS TO THE WIND DIRECTION DATA
* Convert wind_speed_mps = wind_speed_mph / 2.23694



Weather Data: - ERA5 World
* column types
* Set timezone


Processing Step 2 -ERA5Land Data
- Do the gap filling
- Add the additional fiels frome ERA5 World
