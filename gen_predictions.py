import pandas as pd
import numpy as np
import config_param
import config_model
from predictors import generate_predictors
import shared_utils
from shared_utils.utils import bin_array
import matplotlib.pyplot as plt


def generate_all_preds(progress_callback=None):
    # hosp_cumu_s_org= np.loadtxt('data/hosp_cumu_s.csv', delimiter=',')

    # hosp_dat = pd.read_csv('data/hosp_dat.csv')
    # popu = np.loadtxt('data/us_states_population_data.txt') #population of each state
    hosp_cumu_s_org = config_param.hosp_cumu_s_org
    hosp_dat = config_param.hosp_dat
    popu = config_param.popu
    retro_lookback = config_param.retro_lookback
    all_preds = generate_predictors(hosp_cumu_s_org, hosp_dat, popu, config_param, retro_lookback, progress_callback)    
    return all_preds, hosp_dat


def generate_preds(all_preds, hosp_dat, ensemble=None):
    """Generate predictions using either a supervised RF ensemble (default)
    or unsupervised aggregators (mean/median/quantile) across predictors.

    Returns a dict mapping (lookback, location) -> array (weeks_ahead, n_quantiles).
    """
    test_lookback_array = config_param.test_lookback
    quantiles = config_param.quantiles

    # If ensemble is provided and unsupervised, aggregate directly across predictors
    if ensemble is not None and not getattr(ensemble, "is_supervised", False):
        all_test_preds = {}
        for test_lookback in test_lookback_array:
            if test_lookback not in all_preds:
                continue
            preds_lb = all_preds[test_lookback]  # shape: (n_predictors, n_locations, weeks_ahead*bin_size)
            for test_location in range(hosp_dat.shape[0]):
                try:
                    # Slice predictors for this location -> (n_predictors, horizon_days)
                    p_loc = preds_lb[:, test_location, :]
                    # Bin days into weeks -> (n_predictors, weeks_ahead)
                    p3 = p_loc[np.newaxis, :, :]
                    p_binned = shared_utils.utils.bin_array(p3, 0, config_param.bin_size, 2)[0]
                    # Transpose to (weeks_ahead, n_predictors) for ensemble API
                    Xw = p_binned.T
                    qpred = ensemble.predict(Xw, quantiles)  # (weeks_ahead, n_quantiles)
                    # Basic ensemble operates on absolute predictor outputs already; do NOT rescale by population.
                    all_test_preds[(test_lookback, test_location)] = qpred
                except Exception as e:
                    print(f"Agg ensemble error loc {test_location} lb {test_lookback}: {e}")
                    continue
        return all_test_preds

    # Supervised path (default: RF quantile models)
    retro_lookback = config_param.retro_lookback
    X_train, Y_train, W_train, ID = shared_utils.utils.regression_prep(all_preds, hosp_dat, retro_lookback)
    for col in ID:
        print(f"Column: {col}, Unique values: {ID[col].unique()}")
    models = config_model.get_models(config_param.weeks_ahead)
    valid_train = ~ID['lookback'].isin(config_param.test_lookback)
    for i, model in enumerate(models):
        model.fit(X_train[valid_train, :], Y_train[valid_train, i], sample_weight=W_train[valid_train])

    all_test_preds = {}
    for test_lookback in test_lookback_array:
        for test_location in range(hosp_dat.shape[0]):
            try:
                rows = ID[(ID['lookback'] == test_lookback) & (ID['location'] == test_location)].index
                preds = np.zeros((len(models), len(quantiles)))
                for i, model in enumerate(models):
                    preds[i, :] = shared_utils.utils.predict_quantiles(model, X_train[rows], quantiles)
                all_test_preds[(test_lookback, test_location)] = preds * config_param.popu[test_location]
            except Exception as e:
                print(f"Error in test_location {test_location} for test_lookback {test_lookback}: {e}")
                continue

    return all_test_preds