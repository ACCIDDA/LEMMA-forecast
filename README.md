# LEMMA: A Lightweight Epidemic Modeling and Analytics tool for Forecasting


LEMMA is a lightweight forecasting tool for infectious disease time series. It supports:
- GUI-based interaction through Streamlit
- CLI-based forecasting for scripted/batch runs
- Hubverse-style quantile outputs

The app can read either:
- Matrix + location/population inputs
- Hubverse observed target-data CSV inputs

## Requirements

- Python 3.12+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## GUI Interaction (Streamlit)

Run:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in your terminal.

### What you can do in the GUI

1. Choose an observed data input mode in **Input Files (Source Files)**:
- **Hubverse Target Data**
- **Matrix + Location/Population**

2. Upload source files:
- Hubverse mode:
  - Required: Hubverse target-data CSV with `location`, `target_end_date`, and either `weekly_rate` or `observation`
  - Optional: location/population CSV (`location_name`, `population`)
  - Optional: set `hubverse_target` filter text
- Matrix mode:
  - Target matrix CSV (no header, one row per location)
  - Location/population CSV with `location_name` and `population`

3. Configure forecast target parameters:
- Training Start Date / End Date
- Forecast Start Date
- Forecast mode:
  - **Single origin date**
  - **Date range**
- `bins_ahead`
- `timesteps_per_bin`
- `quantiles`

4. Configure modeling options:
- Forecast approach (sidebar): `Flatline`, `ARIMA`, or `SIKJalpha Basic`
- Approach hyperparameters (sidebar)
- Ensemble method in results panel:
  - `Random Forest`
  - `Basic`

5. Run forecasts:
- Click **Save & Run Forecasts** to save config and regenerate predictors + ensemble forecasts
- If predictors already exist, click **Rerun Ensemble** to rerun only ensemble prediction from cached predictors

6. Inspect and export:
- Select state/location
- Select forecast origin date
- Plot overlays observed data and forecast quantile trajectories
- Download all forecasts with **Download All Forecasts (CSV)**

### Notes on GUI persistence

The GUI writes user changes into `user_config.py` and reloads runtime config. Uploaded files are saved into `data/`.

## CLI Interaction

Run:

```bash
python main.py
```

The CLI pipeline does:
1. Validate that target rows, population rows, and location names have matching counts
2. Generate predictors
3. Build ensemble predictions
4. Save output to the configured file

Progress and output path are printed to the console.

## CLI Configuration

Edit `user_config.py` to control CLI behavior.

### Key settings

- Data input:
  - `target_data_path`
  - `location_metadata_path`
  - `hubverse_input_path` (if set, overrides matrix input)
  - `hubverse_target` (optional filter when `target` column exists)

- Forecasting:
  - `predictor_approach`
  - `ensemble_method`
  - `timesteps_per_bin`
  - `bins_ahead`
  - `quantiles`

- Windowing:
  - `training_window_start_date`
  - `training_window_end_date`
  - `forecast_window_start_date`
  - `forecast_window_end_date`

- Export:
  - `forecast_output_path`
  - `forecast_output_format` (`csv` or `json`)

If date windows are set to `None`, defaults are computed from available data and horizon.

## Output Schema

Forecast output is written in long-form Hubverse-style quantile rows:

- `origin_date`
- `horizon`
- `location`
- `output_type` (always `quantile`)
- `output_type_id` (quantile level)
- `value`

## Core Files

- `app.py`: Streamlit GUI entry point
- `main.py`: CLI entry point
- `user_config.py`: user-facing configuration
- `config_param.py`: runtime config mapping/derived values and validation
- `config_model.py`: ensemble/model factory
- `gen_predictions.py`: predictor generation, ensemble prediction, and export helpers
- `predictors.py`: predictor generation pipeline
- `process_scenario.py`: scenario-level forecast generation
- `approaches/`: pluggable approach implementations

## Add a New Forecasting Approach

See:
- `approaches/README.md`

That guide documents required module symbols/functions and registration in `approaches/__init__.py`.
