import pandas as pd
import numpy as np
import config_param

def bin_array(data, start_idx, bin_size, axis):
    # Ensure the start index is within the bounds of the array
    if start_idx < 0 or start_idx >= data.shape[axis]:
        raise ValueError("start_idx is out of bounds")

    # Calculate the number of bins
    num_bins = (data.shape[axis] - start_idx) // bin_size

    # Initialize the binned array
    shape = list(data.shape)
    shape[axis] = num_bins
    binned_data = np.zeros(shape)

    # Sum the data along the specified axis for each bin
    for i in range(num_bins):
        idx_start = start_idx + i * bin_size
        idx_end = idx_start + bin_size
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(idx_start, idx_end)
        binned_data[(slice(None),) * axis + (i,)] = np.sum(data[tuple(slices)], axis=axis)

    return binned_data

def regression_prep(all_preds, hosp_dat, retro_lookback):
    try:
        GT = hosp_dat.to_numpy()
    except:
        GT = hosp_dat
        
    recent_lookback = min(retro_lookback)
    X_train_list = []
    Y_train_list = []
    W_train_list = []
    ID_list = []  # Use a list to collect rows for the DataFrame


    for lookback in retro_lookback:
        preds = all_preds[lookback] / config_param.popu[np.newaxis, :, np.newaxis]
        preds_binned = bin_array(preds, 0, config_param.bin_size, 2)
        T = hosp_dat.shape[1] - lookback * config_param.bin_size
        rows = []

        # Iterate over the first dimension
        for i in range(preds_binned.shape[1]):
            # Flatten the slice along the second dimension and add it to the list
            rows.append(preds_binned[:, i, :].flatten())
            ID_list.append({'location': i, 'lookback': lookback, 'train': 1})
            # Convert the list of rows to a 2D numpy array
        
        X = np.array(rows)
        y_start = T + 1
        if y_start >= GT.shape[1]:
            # No future observed bins available from this origin; keep as NaN targets
            # so test/lookback rows can still be used for prospective inference.
            Y = np.nan * np.ones((GT.shape[0], config_param.weeks_ahead))
        else:
            Y = bin_array(GT/config_param.popu[:, np.newaxis], y_start, config_param.bin_size, 1)
            if Y.shape[1] > config_param.weeks_ahead:
                Y = Y[:, 0:config_param.weeks_ahead]
            else:
                Y = np.concatenate([Y, np.nan*np.ones((Y.shape[0], config_param.weeks_ahead - Y.shape[1]))], axis=1)
        
        w = config_param.decay_factor**(lookback-recent_lookback)
        w = w*np.ones(Y.shape[0])
        
        X_train_list.append(X)
        Y_train_list.append(Y)
        W_train_list.append(w)

    # Concatenate all arrays in the lists into single 2D arrays
    X_train = np.vstack(X_train_list)
    Y_train = np.vstack(Y_train_list)
    W_train = np.concatenate(W_train_list)

    # Create the DataFrame from the list of dictionaries
    ID = pd.DataFrame(ID_list)

    is_test_lookback = ID['lookback'].isin(config_param.test_lookback).to_numpy()
    all_nan_target = np.isnan(Y_train).all(axis=1)
    # Keep NaN-target rows only for requested test lookbacks (prospective origins).
    nan_rows = np.isnan(X_train).any(axis=1) | (all_nan_target & ~is_test_lookback)
    X_train = X_train[~nan_rows]
    Y_train = Y_train[~nan_rows]
    W_train = W_train[~nan_rows]
    ID = ID[~nan_rows]

    return X_train, Y_train, W_train, ID

def predict_quantiles(model, X, quantiles):
    predictions = []
    for q in quantiles:
        model.q = q
        predictions.append(model.predict(X))
    return np.array(predictions).T  # Transpose to have shape (n_samples, n_quantiles)

