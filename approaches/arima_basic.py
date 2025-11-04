import numpy as np

NAME = "ARIMA"
# Hyperparameters as lists (UI will pass arrays)
# We will implement AR(p) with optional differencing d. MA(q)=0 for now.
HYPERPARAM_NAMES = ["ar_p_list", "d_list"]
REQUIRED_CONFIG_KEYS = ["horizon"]

# Human-readable help text for UI; shown in the sidebar when this approach is selected.
HYPERPARAMS_DOC = (
    """
    ARIMA — hyperparameters

    - ar_p_list: Autoregressive order p for AR(p). Controls how many past increments are used in the linear prediction.
    - d_list: Differencing order d. Use d=0 for stationary increments; higher d differences the increment series before fitting and inverts it after forecasting.

    Notes
    - This implementation uses AR(p,d,0) (no MA component) and returns a single deterministic trajectory per (p,d) setting.
    - horizon (config): Forecast length in weeks (binned steps).
    """
)


def build_scenarios(config_param):
    p_list = np.array(getattr(config_param, "ar_p_list", np.array([1])))
    d_list = np.array(getattr(config_param, "d_list", np.array([0])))
    scenarios = []
    for p in p_list:
        for d in d_list:
            scenarios.append([int(p), int(d)])
    return np.array(scenarios, dtype=float)


def make_config(config_param):
    return {
        'horizon': getattr(config_param, 'horizon'),
    }


def _difference(series: np.ndarray, d: int) -> np.ndarray:
    x = series.copy()
    for _ in range(max(0, d)):
        x = np.diff(x, n=1)
    return x


def _invert_difference(last_values: np.ndarray, diffs: np.ndarray, d: int) -> np.ndarray:
    """Invert d-th order differencing for a 1D forecast sequence.
    last_values: shape (d,), the last d values of the original increment series (z^(0)).
    diffs: shape (H,), the forecast sequence in the d-th differenced space (z^(d)).
    Returns: shape (H,), the forecast in the original increment space.
    """
    if d <= 0:
        return diffs
    # Ensure 1D inputs
    last_values = np.asarray(last_values).ravel()
    diffs = np.asarray(diffs).ravel()

    # Compute the last observed values for all lower-order differences at time t
    # levels[k] holds z^(k)_t for k = 0..d-1
    levels = []
    tmp = last_values.copy()
    for _k in range(d):
        if tmp.size == 0:
            levels.append(0.0)
            tmp = np.array([])
        else:
            levels.append(float(tmp[-1]))
            tmp = np.diff(tmp)

    # Iteratively integrate from order d to 0
    out = diffs.copy()
    for j in range(d - 1, -1, -1):
        init = levels[j]
        out = np.cumsum(out) + init
    return out


def _fit_ar_least_squares(x: np.ndarray, p: int):
    # x shape: (n_time,), returns coeffs shape: (p,)
    if p <= 0 or len(x) <= p:
        return np.zeros((p,))
    # Build design matrix
    T = len(x)
    X = np.column_stack([x[t:T - (p - t)] for t in range(p)])
    y = x[p:]
    # Solve least squares
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp):
    """Simple ARIMA(p,d,0) on daily increments per location.
    Returns temp_res with cumulative predictions relative to base_hosp.
    """
    horizon = config_params['horizon']
    num_samples = 1  # ARIMA returns a single deterministic trajectory here
    temp_res = np.zeros((num_samples, hosp_cumu_s.shape[0], horizon))

    simnum, scenario_details = args
    p, d = [int(v) for v in scenario_details]

    # Daily increments up to the current cut (use hosp_cumu_s for consistent window)
    # cumu: (n_loc, T_cut)
    inc = np.diff(hosp_cumu_s, axis=1)
    n_loc, T = inc.shape

    # Forecast daily increments for 'horizon' steps using AR(p) on differenced data d times
    preds_inc = np.zeros((n_loc, horizon))
    for cid in range(n_loc):
        series = inc[cid]
        x = _difference(series, d)
        # Initialize with last p values for recursion
        hist = x[-max(p, 1):].tolist()
        coef = _fit_ar_least_squares(x, p)
        # Generate forecasts in differenced space
        fx = []
        for _ in range(horizon):
            if p > 0 and len(hist) >= p:
                pred = float(np.dot(coef, hist[-p:][::-1]))
            else:
                pred = hist[-1] if hist else 0.0
            fx.append(pred)
            hist.append(pred)
        fx = np.array(fx)
        # Invert differencing back to original increment space
        if d > 0:
            last_vals = series[-d:]
            inc_pred = _invert_difference(last_vals, fx, d=d)
        else:
            inc_pred = fx
        preds_inc[cid] = inc_pred

    # Convert to cumulative predictions relative to base_hosp
    pred_cumu = base_hosp[:, None] + np.cumsum(preds_inc, axis=1)
    temp_res[:] = pred_cumu - base_hosp[:, None]
    return temp_res
