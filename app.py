import streamlit as st
import os, calendar, copy, warnings, json, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure Streamlit page
st.set_page_config(
    page_title="Iowa Nitrate - Matched Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# constants
RANDOM_STATE = 5160
YEAR_MIN, YEAR_MAX = 2016, 2025
EXPECTED_SAMPLE_INTERVAL_MINUTES = 15
MIN_ANNUAL_COVERAGE = 0.80
ANALYSIS_WINDOW = 'march_november'
DATA_DIR = Path('.')
PROJECTED_CRS = 'EPSG:5070'
TARGET = 'period_fw_nitrate_mg_L_as_N'
BASE_SPATIAL = ['cropped_pct', 'som_mean']

NEON = {
    'cyan': '#00E5FF', 'green': '#39FF14', 'pink': '#FF2BD6', 'yellow': '#FFE600',
    'purple': '#AF7BFF', 'white': '#F5F7FA', 'bg': '#050509', 'grid': '#525A66'
}

# 1. Caching data loading, GIS elements, and observations
@st.cache_data
def load_data_inputs():
    PREDICTORS = DATA_DIR / 'Iowa_HUC8_predictors_clean.csv'
    MAPPING = DATA_DIR / 'station_huc8_mapping.csv'
    STATION_SUMMARY = DATA_DIR / 'station_summary.csv'
    MONITORING = DATA_DIR / 'monitoring_paired.csv.gz'
    
    for p in [PREDICTORS, MAPPING, STATION_SUMMARY, MONITORING]:
        if not p.is_file():
            return None, f"Missing file: {p.name}"
            
    huc = pd.read_csv(PREDICTORS, dtype={'huc8_id': 'string'})
    station_map = pd.read_csv(MAPPING, dtype={'station_id': 'string', 'huc8_id': 'string'})
    stations = pd.read_csv(STATION_SUMMARY, dtype={'station_id': 'string'})
    return huc, station_map, stations, None

@st.cache_data
def load_monitoring_table():
    MONITORING = DATA_DIR / 'monitoring_paired.csv.gz'
    required_monitoring = ['station_id', 'datetime', 'nitrate_mg_L_as_N', 'discharge_cms']
    mon = pd.read_csv(MONITORING, usecols=required_monitoring, dtype={'station_id': 'string'}, low_memory=False)
    mon['station_id'] = mon.station_id.str.strip()
    mon['datetime_utc'] = pd.to_datetime(mon['datetime'], errors='coerce', utc=True, format='mixed')
    for col in ['nitrate_mg_L_as_N', 'discharge_cms']:
        mon[col] = pd.to_numeric(mon[col], errors='coerce')
    mon = mon.dropna(subset=['station_id', 'datetime_utc', 'nitrate_mg_L_as_N', 'discharge_cms'])
    mon = mon.loc[mon.discharge_cms.gt(0) & mon.nitrate_mg_L_as_N.ge(0)].copy()
    mon = mon.loc[mon.datetime_utc.dt.year.between(YEAR_MIN, YEAR_MAX)].copy()
    mon = mon.drop_duplicates(subset=['station_id', 'datetime_utc'], keep='first')
    mon['year'] = mon.datetime_utc.dt.year.astype('int16')
    mon['month'] = mon.datetime_utc.dt.month.astype('int8')
    if ANALYSIS_WINDOW == 'march_november':
        mon = mon.loc[mon.month.between(3, 11)].copy()
    mon['cq'] = mon.nitrate_mg_L_as_N * mon.discharge_cms
    mon['q2'] = mon.discharge_cms ** 2
    return mon

@st.cache_data
def load_gis_data():
    members = pd.read_csv(DATA_DIR / 'iowa_upstream_huc8_memberships.csv', dtype={'station_id': 'string', 'huc8_id': 'string'})
    station_upstream = pd.read_csv(DATA_DIR / 'iowa_upstream_station_predictors.csv', dtype={'station_id': 'string', 'local_huc8_id': 'string'})
    iowa_basins = gpd.read_file(DATA_DIR / 'iowa_upstream_basins.geojson').to_crs(PROJECTED_CRS)
    return members, station_upstream, iowa_basins

# Helper to style plot grids
def neon_axes(ax, grid='y'):
    ax.set_facecolor(NEON['bg'])
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color('#AAB3C1')
    ax.tick_params(colors=NEON['white'], direction='in')
    ax.xaxis.label.set_color(NEON['white'])
    ax.yaxis.label.set_color(NEON['white'])
    ax.title.set_color(NEON['white'])
    if grid:
        ax.grid(axis=grid, color=NEON['grid'], alpha=.27, lw=.65)

# Render application
st.title("Iowa Nitrate Monitoring & Predictive Dashboard")
st.markdown("Explore **Option B** (same qualifying station-years) and **Option C** (balanced station/year) designs across four temporal resolutions.")

huc, station_map, stations, err = load_data_inputs()
if err:
    st.error(f"❌ {err}. Please make sure you have the expected data files uploaded.")
    st.stop()

# Run cached steps
with st.spinner("Initializing analytical databases & geo-caches..."):
    mon = load_monitoring_table()
    members, station_upstream, iowa_basins = load_gis_data()

# Sidebar selections
st.sidebar.header("Configuration Panels")
design_opt = st.sidebar.selectbox("Select Design option:", 
    ["B_matched_station_years", "C_balanced_7stations_7years"])
grain_opt = st.sidebar.selectbox("Select Timescale:", 
    ["season_year", "monthly", "weekly", "daily"])

# Quick stat metrics
col1, col2, col3 = st.columns(3)
col1.metric("Active Design", design_opt.split('_')[0])
col2.metric("Selected Resolution", grain_opt.upper())
col3.metric("Total Available Raw Records", f"{len(mon):,}")

# Display dynamic visualizations
st.subheader(f"Visual Analytics: {design_opt} / {grain_opt.upper()}")

# Attempt to read existing figures/results if run has completed in notebook
base_results_dir = DATA_DIR / 'Iowa_Nitrate_Matched_B_and_C_Results' / design_opt / grain_opt

if base_results_dir.exists():
    col1_plot, col2_plot = st.columns(2)
    fig1_path = base_results_dir / 'Figure_01_Neon_Manure_RMSE.png'
    fig2_path = base_results_dir / 'Figure_02_Neon_Station_Bias.png'
    
    if fig1_path.exists():
        col1_plot.image(str(fig1_path), caption="Manure Added Value (RMSE comparisons)", use_container_width=True)
    if fig2_path.exists():
        col2_plot.image(str(fig2_path), caption="Random Forest Station Bias", use_container_width=True)

    # Table values
    metric_path = base_results_dir / 'spatial_cv_metrics.csv'
    if metric_path.exists():
        st.write("### Spatial Cross Validation Metrics")
        metrics_df = pd.read_csv(metric_path)
        st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("Results not found for the selected configuration. Run the full model comparisons in your notebook to export diagnostic outputs.")
