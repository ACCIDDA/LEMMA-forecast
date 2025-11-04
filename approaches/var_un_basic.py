import numpy as np
from itertools import product
from input_models.var_ind_beta_un import var_ind_beta_un
from input_models.var_simulate_pred_un import var_simulate_pred_un

# Public API expected by registry consumers
NAME = "var_un_basic"
HYPERPARAM_NAMES = ["halpha_list", "rlag_list", "un_list", "S"]


def build_scenarios(config_param):
    """Return scenario array where each row holds a tuple of hyperparameter choices.
    Shape: (n_scenarios, len(HYPERPARAM_NAMES))
    """
    halpha_list = np.array(getattr(config_param, "halpha_list"))
    rlag_list = np.array(getattr(config_param, "rlag_list"))
    un_list = np.array(getattr(config_param, "un_list"))
    S = np.array(getattr(config_param, "S"))
    scenarios = [list(s) for s in product(halpha_list, rlag_list, un_list, S)]
    return np.array(scenarios, dtype=float)


essential_params = ["num_dh_rates_sample", "horizon", "rlags", "hk", "hjp"]


def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp):
    """Mirror the existing scenario logic using VAR-UN estimation/simulation.
    args: (simnum, scenario_details)
    Returns: temp_res array of shape (num_dh_rates_sample, n_locations, horizon)
    """
    num_dh_rates_sample = config_params['num_dh_rates_sample']
    horizon = config_params['horizon']
    rlags = config_params['rlags']
    hk = config_params['hk']
    hjp = config_params['hjp']

    temp_res = np.zeros((num_dh_rates_sample, hosp_cumu_s.shape[0], horizon))

    simnum, scenario_details = args
    halpha, rlag, un, s = scenario_details
    rr = rlags[int(rlag) - 1]

    if rr == 0:
        sliced_array = hosp_cumu_s
    else:
        sliced_array = hosp_cumu_s[:, :rr]

    hosp_rate, fC, ci_h = var_ind_beta_un(sliced_array, 0, halpha, hk, un, popu, hjp, 0.95, s, None, 80)

    for rr in range(num_dh_rates_sample):
        this_rate = hosp_rate.copy()
        if rr != (num_dh_rates_sample + 1) // 2 - 1:
            for cid in range(hosp_cumu_s.shape[0]):
                this_rate[cid] = ci_h[cid][:, 0] + (ci_h[cid][:, 1] - ci_h[cid][:, 0]) * (rr) / (num_dh_rates_sample - 1)
        else:
            for cid in range(hosp_cumu_s.shape[0]):
                this_rate[cid] = (ci_h[cid][:, 1] + ci_h[cid][:, 0]) / 2

        pred_hosps = var_simulate_pred_un(hosp_cumu_s, 0, this_rate, popu, hk, horizon, hjp, un, base_hosp)
        h_start = 0
        temp_res[rr, :, :] = pred_hosps[:, h_start:h_start + horizon] - base_hosp.reshape(-1, 1)

    return temp_res
