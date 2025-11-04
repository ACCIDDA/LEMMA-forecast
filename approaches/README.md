# Approaches: How to add a new forecasting approach

This folder contains pluggable predictor-generation approaches (e.g., SIKJalpha Basic, ARIMA, Flatline). Each approach module follows a simple interface so the app can build scenarios, run them in parallel, and aggregate predictions consistently.

Use this guide to add a new approach.

## What an approach module must provide

Required symbols (all caps for metadata):
- NAME: short identifier string used for internal reference (e.g., "ARIMA")
- HYPERPARAM_NAMES: list of hyperparameter names exposed in the UI (strings)
- REQUIRED_CONFIG_KEYS: list of runtime config keys your approach requires (strings)
- HYPERPARAMS_DOC: a human-readable Markdown string explaining the hyperparameters

Required functions:
- build_scenarios(config_param) -> np.ndarray
  - Returns an array of shape (n_scenarios, k) describing the hyperparameter combinations to simulate.
  - Each row is a single scenario. Types should be numeric and castable to float.
- make_config(config_param) -> dict
  - Returns a lightweight dict with only the runtime configuration your process_scenario needs (e.g., horizon, precomputed constants). Avoid adding large arrays here to reduce pickling overhead.
- process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp) -> np.ndarray
  - args is a tuple (simnum, scenario_details) where scenario_details is a row from build_scenarios.
  - Must return a numpy array temp_res with shape (num_samples, n_locations, horizon). If your approach is deterministic, set num_samples = 1.
  - The returned values must be cumulative predictions relative to base_hosp (i.e., pred_cumu - base_hosp[:, None]). The predictors pipeline will take daily diffs.

## Data contracts and shapes

- hosp_cumu_s: np.ndarray (n_locations, T_cut)
  - Smoothed cumulative target series up to the current lookback cut.
- hosp_cumu: np.ndarray (n_locations, T_cut)
  - Raw cumulative target series up to the current lookback cut.
- popu: np.ndarray (n_locations,)
  - Population per location.
- base_hosp: np.ndarray (n_locations,)
  - Baseline cumulative value at the cut (used to convert increments back to cumulative predictions).
- horizon (in config_params): int
  - Number of binned steps to forecast ahead. Predictors will compute daily increments and then bin by config_param.bin_size later.

Output from process_scenario:
- temp_res: np.ndarray (num_samples, n_locations, horizon)
  - Cumulative predictions relative to base_hosp (not absolute; absolute = base_hosp[:, None] + temp_res).

## UI wiring and registration

- Add your module file here, e.g., approaches/my_new_method.py.
- Register it in approaches/__init__.py by adding an entry to _REGISTRY mapping a friendly label (as used in the UI) to your module import path. Example:
  ```python
  _REGISTRY = {
      "SIKJalpha Basic": "approaches.sikjalpha_basic",
      "ARIMA": "approaches.arima_basic",
      "Flatline": "approaches.flatline",
      "My New Method": "approaches.my_new_method",
  }
  ```
- Update the hyperparameter list in approaches/__init__.py so the sidebar renders controls for your approach:
  ```python
  approaches = {
      "SIKJalpha Basic": ["halpha_list", "rlag_list", "un_list", "S"],
      "ARIMA": ["ar_p_list", "d_list"],
      "Flatline": ["flat_k_list"],
      "My New Method": ["my_param1", "my_param2"],
  }
  ```
  Note: The UI currently reads this mapping for hyperparameter names. The help text comes from HYPERPARAMS_DOC in your module.

## Example skeleton

```python
import numpy as np

NAME = "MyNewMethod"
HYPERPARAM_NAMES = ["my_param1_list", "my_param2_list"]
REQUIRED_CONFIG_KEYS = ["horizon"]

HYPERPARAMS_DOC = """
My New Method — hyperparameters

- my_param1_list: Describe what this controls.
- my_param2_list: Describe what this controls.

Notes
- horizon (config): Forecast length in binned steps.
"""


def build_scenarios(config_param):
    p1 = np.array(getattr(config_param, "my_param1_list", np.array([1])))
    p2 = np.array(getattr(config_param, "my_param2_list", np.array([0])))
    scenarios = []
    for a in p1:
        for b in p2:
            scenarios.append([float(a), float(b)])
    return np.array(scenarios, dtype=float)


def make_config(config_param):
    return {
        "horizon": int(getattr(config_param, "horizon")),
    }


def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp):
    simnum, scenario = args
    horizon = int(config_params["horizon"])
    n_loc = hosp_cumu_s.shape[0]
    num_samples = 1  # set >1 if you sample uncertainty internally

    temp_res = np.zeros((num_samples, n_loc, horizon))

    # Example: copy last daily increment forward deterministically
    inc = np.diff(hosp_cumu_s, axis=1)
    last_inc = inc[:, -1]
    pred_inc = np.tile(last_inc[:, None], (1, horizon))
    pred_cumu = base_hosp[:, None] + np.cumsum(pred_inc, axis=1)

    temp_res[0] = pred_cumu - base_hosp[:, None]
    return temp_res
```

## Best practices

- Keep make_config small and serializable (numbers, small arrays). Large arrays should be derived in the worker to avoid pickling overhead.
- Ensure process_scenario is pure (no global state mutations) and safe for multi-process execution.
- If you sample multiple trajectories internally, return them in temp_res and let the predictors pipeline aggregate via mean.
- Match units carefully: temp_res should be cumulative relative to base_hosp; predictors will take np.diff along time and handle weekly binning later.
- Validate shapes with small inputs first; nans will be filtered later but avoiding them improves training stability.

## Testing your approach

1. Add your module and register it in approaches/__init__.py.
2. Add your hyperparameters to the approaches dict in approaches/__init__.py.
3. Run the app and select your approach from the sidebar.
4. Use the "Save & Run Forecasts" button to generate predictors and ensemble outputs.
5. Verify lines appear on the plot and check logs for any scenario failures.
