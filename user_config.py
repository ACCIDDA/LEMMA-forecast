from datetime import datetime
import numpy as np

# ============================
# User-Facing Configuration
# ============================

# Time semantics for your source data and forecast bins.
base_time_step_unit = "day"
forecast_bin_unit = "week"

# Number of base timesteps in one forecast bin (e.g., 7 days per week bin).
timesteps_per_bin = 7

# Number of bins to predict ahead.
bins_ahead = 4

# Smoothing applied to base-time-step signal before model processing.
smoothing_window_timesteps = 7

# Reference dates used for indexing.
reference_date = datetime(2021, 9, 1)

# Season start date if applicable (used in SIKJalpha-style approaches to initialize mechanisitic models, ignores prior time-series).
season_start_date = datetime(2023, 9, 30)

# Forecasting approach and ensemble aggregation.
# Recommended settings for simple fast and good forecasts: predictor_approach = "Flatline", ensemble_method = "Random Forest"
predictor_approach = "Flatline"
ensemble_method = "Random Forest"

# Forecast quantiles (must be in [0, 1]).
quantiles = np.array([0.0, 0.5, 1.0])

# Input data paths.
target_data_path = "data/ts_dat.csv"
location_metadata_path = "data/location_dat.csv"

# Optional Hubverse observed target-data CSV input. If set, this overrides
# target_data_path/location metadata-derived matrix/date settings.
# Keep as None to use the legacy matrix input at target_data_path.
hubverse_input_path = "data/hubverse_target_data.csv"

# Optional compatibility alias (kept for older references/typos).
hubvsereInput = "data/hubverse_target_data.csv"

# Optional target name to filter when Hubverse file has a 'target' column.
# If None and 'target' exists, each (location, target) pair is treated as one
# location row in the internal 2D matrix.
hubverse_target = "wk inc flu hosp"

# Training/forecast window dates.
# Set to None to use automatic defaults based on available data:
# - training start: 10 bins before last trainable bin (or first date)
# - training end: last trainable bin given bins_ahead
# - forecast start/end: last available origin date
training_window_start_date = None
training_window_end_date = None
forecast_window_start_date = None
forecast_window_end_date = None

# CLI export settings.
forecast_output_path = "outputs/predictions.csv"
forecast_output_format = "csv"

# -------- Approach-specific hyperparameters --------
# ARIMA(p,d,0)
arima_autoregressive_orders = np.array([7.0, 14.0])
arima_differencing_orders = np.array([0.0, 1.0, 2.0])

# Flatline approach
flatline_lag_timesteps = np.array([0.0, 7.0, 14.0])

# SIKJalpha-style settings
rlags = np.array([0])
retro_lag_bins = np.array([1.0])
un_list = np.array([50.0])
halpha_list = np.arange(0.98, 0.92, -0.02)
S = np.array([0.0])
hk = 2
hjp = 7
