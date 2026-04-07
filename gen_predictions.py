import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
import config_param
import config_model
from predictors import generate_predictors
import shared_utils
from shared_utils.utils import bin_array
import matplotlib.pyplot as plt


def generate_all_preds(progress_callback=None, use_threads=False):
    # hosp_cumu_s_org= np.loadtxt('data/hosp_cumu_s.csv', delimiter=',')

    # hosp_dat = pd.read_csv('data/hosp_dat.csv')
    # popu = np.loadtxt('data/us_states_population_data.txt') #population of each state
    hosp_cumu_s_org = config_param.hosp_cumu_s_org
    hosp_dat = config_param.hosp_dat
    popu = config_param.popu
    retro_lookback = config_param.retro_lookback
    all_preds = generate_predictors(
        hosp_cumu_s_org,
        hosp_dat,
        popu,
        config_param,
        retro_lookback,
        progress_callback,
        use_threads=use_threads,
    )
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
        skipped = 0
        for test_lookback in test_lookback_array:
            if test_lookback not in all_preds:
                skipped += hosp_dat.shape[0]
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
                except Exception:
                    skipped += 1
                    continue
        if skipped > 0:
            print(f"Note: skipped {skipped} forecast combinations due to processing issues.")
        return all_test_preds

    # Supervised path (default: RF quantile models)
    retro_lookback = config_param.retro_lookback
    X_train, Y_train, W_train, ID = shared_utils.utils.regression_prep(all_preds, hosp_dat, retro_lookback)
    models = config_model.get_models(config_param.weeks_ahead)
    valid_train = ~ID['lookback'].isin(config_param.test_lookback)
    for i, model in enumerate(models):
        valid_target = ~np.isnan(Y_train[:, i])
        fit_mask = valid_train.to_numpy() & valid_target
        if np.sum(fit_mask) == 0:
            raise ValueError(
                f"No valid training rows available for horizon {i + 1}. "
                "Adjust training/forecast windows or bins_ahead."
            )
        model.fit(X_train[fit_mask, :], Y_train[fit_mask, i], sample_weight=W_train[fit_mask])

    all_test_preds = {}
    skipped = 0
    for test_lookback in test_lookback_array:
        for test_location in range(hosp_dat.shape[0]):
            try:
                rows = ID[(ID['lookback'] == test_lookback) & (ID['location'] == test_location)].index
                preds = np.zeros((len(models), len(quantiles)))
                for i, model in enumerate(models):
                    preds[i, :] = shared_utils.utils.predict_quantiles(model, X_train[rows], quantiles)
                all_test_preds[(test_lookback, test_location)] = preds * config_param.popu[test_location]
            except Exception:
                skipped += 1
                continue

    if skipped > 0:
        print(f"Note: skipped {skipped} forecast combinations due to processing issues.")

    return all_test_preds


def _timedelta_from_timesteps(num_steps):
    unit = str(getattr(config_param, "base_time_step_unit", "day")).lower().strip()
    if unit in {"day", "days", "d"}:
        return timedelta(days=int(num_steps))
    if unit in {"week", "weeks", "w"}:
        return timedelta(weeks=int(num_steps))
    # Default to day-level increments for unknown units.
    return timedelta(days=int(num_steps))


def predictions_to_long_df(preds):
    """Convert prediction dict into Hubverse-style quantile output rows.

    Expected preds shape:
    {(lookback, location_index): np.ndarray(weeks_ahead, n_quantiles)}
    """
    expected_columns = [
        "origin_date",
        "horizon",
        "location",
        "output_type",
        "output_type_id",
        "value",
    ]
    quantiles = [float(q) for q in np.asarray(config_param.quantiles)]
    rows = []
    max_t = int(config_param.hosp_dat.shape[1])
    reference_date = pd.Timestamp(config_param.zero_date)

    for (lookback, location_idx), pred_matrix in sorted(preds.items()):
        pred_matrix = np.asarray(pred_matrix)
        if pred_matrix.ndim != 2:
            raise ValueError(
                f"Unexpected prediction shape for lookback={lookback}, location={location_idx}: {pred_matrix.shape}"
            )

        n_quantiles = pred_matrix.shape[1]
        if n_quantiles != len(quantiles):
            raise ValueError(
                f"Prediction quantiles mismatch for lookback={lookback}, location={location_idx}. "
                f"Expected {len(quantiles)} but got {n_quantiles}."
            )

        location_name = str(config_param.state_abbr[location_idx])
        origin_offset_steps = max_t - int(lookback) * int(config_param.bin_size)
        origin_date = (reference_date + _timedelta_from_timesteps(origin_offset_steps)).date().isoformat()

        for horizon_bin in range(pred_matrix.shape[0]):
            for q_idx, quantile in enumerate(quantiles):
                rows.append(
                    {
                        "origin_date": origin_date,
                        "horizon": int(horizon_bin + 1),
                        "location": location_name,
                        "output_type": "quantile",
                        "output_type_id": quantile,
                        "value": float(pred_matrix[horizon_bin, q_idx]),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=expected_columns)

    return pd.DataFrame(rows, columns=expected_columns)


def save_predictions_df(df, output_path, file_format="csv"):
    """Save predictions DataFrame to disk as CSV or JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(file_format).lower()

    if fmt == "csv":
        df.to_csv(output_path, index=False)
        return
    if fmt == "json":
        df.to_json(output_path, orient="records", indent=2)
        return

    raise ValueError("Unsupported output format. Use 'csv' or 'json'.")