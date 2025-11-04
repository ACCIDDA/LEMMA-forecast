import numpy as np
from itertools import product
# Inlined from input_models to avoid cross-package dependency
import scipy.optimize
import scipy.stats
from config_param import season_start_day


NAME = "SIKJalpha_basic"
HYPERPARAM_NAMES = ["halpha_list", "rlag_list", "un_list", "S"]
REQUIRED_CONFIG_KEYS = ["num_dh_rates_sample", "horizon", "rlags", "hk", "hjp"]

# Human-readable help text for UI; shown in the sidebar when this approach is selected.
HYPERPARAMS_DOC = (
    """
    SIKJalpha Basic — hyperparameters

    - halpha_list: Exponential weighing for past observations. Higher values discount older data more strongly.
    - rlag_list: Estimating the patameters discounting these many recent days.
    - un_list: Multiplier represeting inverse ascertainment rate.
    - S: Susceptibility at the beginning.

    Notes
    - num_dh_rates_sample (config): Number of samples for hospitalization delay-rate uncertainty. Not a hyperparameter grid, but used internally to propagate uncertainty.
    - horizon (config): Forecast length in weeks (binned steps).
    """
)


def var_ind_beta_un(data_4, passengerFlow, alpha_l, k_l, un_fact, popu, jp_l, ret_conf, S, compute_region=None, window_size=None, extra_imm=None):
    """Estimate infection-rate parameters per location with optional smoothing and uncertainty bands.
    Returns (beta_all_cell, fittedC, ci) where beta_all_cell[j] holds k+1 params for location j,
    and ci[j] holds corresponding lower/upper bounds.
    """
    maxt = np.size(data_4, 1)
    if compute_region is None:
        compute_region = np.ones((np.size(popu), 0))
    if window_size is None:
        window_size = maxt * np.ones((np.size(data_4, 0), 1))
    if extra_imm is None:
        extra_imm = np.zeros((np.size(data_4, 0), maxt)) + S

    F = passengerFlow
    beta_all_cell = [None] * np.size(popu)
    fittedC = [None] * np.size(popu)
    ci = [None] * np.size(popu)
    nn = np.size(popu)

    if np.isscalar(un_fact):
        un_fact = np.ones(len(popu)) * un_fact
    if np.isscalar(jp_l):
        jp_l = np.ones(len(popu)) * jp_l
    if np.isscalar(k_l):
        k_l = np.ones(len(popu)) * k_l
    if np.isscalar(alpha_l):
        alpha_l = np.ones(len(popu)) * alpha_l
    if np.isscalar(window_size):
        window_size = np.ones(nn) * window_size

    deldata = np.diff(data_4, 1)

    for j in range(1, np.size(popu) + 1):
        jp = int(jp_l[j - 1])
        k = int(k_l[j - 1])
        alpha = alpha_l[j - 1]
        jk = int(jp * k)

        beta_all_cell[j - 1] = np.zeros((k + 1, 1))
        ci[j - 1] = np.zeros((k + 1, 2))
        fittedC[j - 1] = np.zeros((1, 2))

        skip_days = int(maxt - window_size[j - 1])
        if season_start_day + 15 < maxt:
            skip_days = season_start_day
        if skip_days < 0:
            skip_days = 0

        ex = (np.arange(maxt - skip_days - jk - 1, 0, -1)).transpose()
        alphavec = np.power(alpha, ex)
        alphamat = np.tile(alphavec, np.concatenate(([k + 1], [1]))).transpose()

        tdim = maxt - jk - skip_days - 1
        if tdim < 0:
            tdim = 0
        y = np.zeros((tdim, 1))
        X = np.zeros((tdim, k + 1))
        Ikt = np.zeros((1, k))

        for t in range(skip_days + jk + 1, maxt):
            Ikt1 = deldata[j - 1, t - jk - 1:t - 1]
            Sfac = (1 - (extra_imm[j - 1, t - 1] + (un_fact[j - 1] * data_4[j - 1, t - 1])) / popu[j - 1])
            for kk in range(1, k + 1):
                m0 = Ikt1[(kk - 1) * jp: kk * jp]
                Ikt[0, kk - 1] = Sfac * sum(m0, 0)

            if F is None or (np.isscalar(F) or len(np.shape(F)) == 0) or np.size(F, 0) != np.size(popu):
                incoming_travel = np.zeros((1, 1))
            else:
                incoming_travel = np.transpose(F[:, j - 1] / popu) @ sum(Ikt1, 1)

            X[(t - 1) - jk - skip_days, :] = np.concatenate((Ikt, incoming_travel), 1)
            y[(t - 1) - jk - skip_days] = np.transpose(deldata[j - 1, t - 1])

        X = alphamat * X
        y = alphavec.transpose() * y.flatten()

        if np.size(X) != 0 and np.size(y) != 0:
            a = np.concatenate((np.ones(k), [np.inf]), 0)
            if ret_conf is None:
                beta_vec = scipy.optimize.lsq_linear(X, y, bounds=(np.zeros(k + 1), a)).x
                beta_CI = np.zeros((k + 1, 2))
                beta_CI[:, 0] = beta_vec
                beta_CI[:, 1] = beta_vec
            else:
                def sigmoid(Xm, *w):
                    w1 = np.array(w).flatten()
                    return Xm @ (1 / (1 + np.exp(-w1)))

                k1 = X.shape[1]
                w0 = np.zeros(k1)
                popt, pcov = scipy.optimize.curve_fit(sigmoid, X, y, p0=w0, method="dogbox")

                n = len(y)
                mse = np.sum((y - sigmoid(X, popt)) ** 2) / (n - k1)
                se = np.sqrt(np.diag(pcov) * mse)
                tval = scipy.stats.t.ppf((1 + ret_conf) / 2, n - k1)

                CI_lower = popt - tval * se
                CI_upper = popt + tval * se

                beta_vec = 1 / (1 + np.exp(-popt))
                beta_CI = 1 / (1 + np.exp(-np.vstack((CI_lower, CI_upper))))

            ci[j - 1] = beta_CI
            beta_all_cell[j - 1] = beta_vec
    return (beta_all_cell, fittedC, ci)


def var_simulate_pred_un(data_4, passengerFlowDarpa, beta_all_cell, popu, k_l, horizon, jp_l, un_fact, base_infec=None, vac=None, rate_change=None):
    """Simulate cumulative infections forward given fitted beta parameters.
    Returns array of shape (n_regions, horizon) with cumulative counts.
    """
    num_countries = np.size(data_4, 0)
    infec = np.zeros((num_countries, horizon))
    F = passengerFlowDarpa

    if rate_change is None:
        rate_change = np.ones((num_countries, horizon))
    if base_infec is None:
        base_infec = data_4[:, -1:]
    if vac is None:
        vac = np.zeros((num_countries, horizon))

    if np.size(rate_change) == 0:
        rate_change = np.ones((num_countries, horizon))
    if np.size(un_fact) == 1:
        un_fact = un_fact * np.ones((np.size(popu), 1))
    if np.size(jp_l) == 1:
        jp_l = np.ones((np.size(popu), 1)) * jp_l
    if np.size(k_l) == 1:
        k_l = np.ones((np.size(popu), 1)) * k_l

    for j in range(0, len(beta_all_cell)):
        this_beta = beta_all_cell[j]
        if np.size(this_beta) == k_l[j]:
            beta_all_cell[j] = np.append(this_beta, 0)

    data_4_s = data_4
    deltemp = np.diff(data_4_s, axis=1)

    # Optimized code when mobility not considered
    if isinstance(F, (int, float)) or F is None:
        for j in range(0, np.size(popu)):
            lastinfec = base_infec[j]
            if np.sum(beta_all_cell[j]) == 0:
                infec[j, :] = lastinfec
                continue

            jp = int(jp_l[j])
            k = int(k_l[j])
            jk = int(jp * k)
            Ikt1 = deltemp[j, -(jk):]
            Ikt = np.zeros((k, 1))

            for t in range(0, horizon):
                true_infec = un_fact[j] * lastinfec
                Sfac = 1 - true_infec / popu[j] - ((1 - true_infec / popu[j]) * vac[j, t]) / popu[j]
                for kk in range(1, k + 1):
                    Ikt[kk - 1] = np.sum(Ikt1[(kk - 1) * jp: kk * jp], 0)
                Xt = np.concatenate(((Sfac * Ikt), [[0]]), axis=0)
                yt = rate_change[j, t] * np.sum(np.transpose(beta_all_cell[j]) @ Xt, axis=0)
                yt = max(yt, 0)
                lastinfec = lastinfec + yt
                infec[j, t] = lastinfec
                Ikt1 = np.append(Ikt1, yt)
                Ikt1 = np.delete(Ikt1, 0)

    return infec



def build_scenarios(config_param):
    halpha_list = np.array(getattr(config_param, "halpha_list"))
    rlag_list = np.array(getattr(config_param, "rlag_list"))
    un_list = np.array(getattr(config_param, "un_list"))
    S = np.array(getattr(config_param, "S"))
    scenarios = [list(s) for s in product(halpha_list, rlag_list, un_list, S)]
    return np.array(scenarios, dtype=float)


def make_config(config_param):
    return {
        'num_dh_rates_sample': getattr(config_param, 'num_dh_rates_sample'),
        'horizon': getattr(config_param, 'horizon'),
        'rlags': getattr(config_param, 'rlags'),
        'hk': getattr(config_param, 'hk'),
        'hjp': getattr(config_param, 'hjp'),
    }


def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp):
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
