import numpy as np

NAME = "Flatline"
HYPERPARAM_NAMES = ["flat_k_list"]  # k is how many steps back to use for flatline base
REQUIRED_CONFIG_KEYS = ["horizon"]

# Human-readable help text for UI; shown in the sidebar when this approach is selected.
HYPERPARAMS_DOC = (
    """
    Flatline — hyperparameters

    - flat_k_list: How many steps back (k) to take the last observed daily increment from; the forecast then repeats that increment for all future steps.

    Notes
    - This approach returns a single deterministic trajectory per k.
    - horizon (config): Forecast length in weeks (binned steps).
    """
)


def build_scenarios(config_param):
    k_list = np.array(getattr(config_param, "flat_k_list", np.array([0])))
    return k_list.reshape(-1, 1).astype(float)


def make_config(config_param):
    return {
        'horizon': getattr(config_param, 'horizon'),
    }


def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp):
    """Flatline daily increments using the value at t-k, constant over the horizon.
    temp_res returns cumulative predictions relative to base_hosp.
    """
    horizon = config_params['horizon']
    num_samples = 1  # Flatline does not sample additional trajectories
    temp_res = np.zeros((num_samples, hosp_cumu_s.shape[0], horizon))

    simnum, scenario_details = args
    k = int(scenario_details[0])

    # Use hosp_cumu consistent with main windowing
    inc = np.diff(hosp_cumu_s, axis=1)  # (n_loc, T)
    n_loc, T = inc.shape
    idx1 = max(1, T - 1 - k)
    idx0 = idx1 - 1
    last_inc = inc[:, idx1] if (idx1 < T) else inc[:, -1]

    # Build cumulative predictions
    flat = np.tile(last_inc[:, None], (1, horizon))  # constant daily inc
    pred_cumu = base_hosp[:, None] + np.cumsum(flat, axis=1)

    # Replicate for num_samples (no uncertainty sampling here)
    for rr in range(num_samples):
        temp_res[rr] = pred_cumu - base_hosp[:, None]

    return temp_res
