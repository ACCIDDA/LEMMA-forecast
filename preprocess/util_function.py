import numpy as np
from numpy import *
import scipy.optimize
import scipy.stats
from functools import partial
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import signal
import pandas as pd

import warnings

warnings.filterwarnings('ignore')

def hampel_filter_forloop(input_series, window_size=15, n_sigmas=4):
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution

    indices = []

    for i in range((window_size), (n - window_size)):
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (np.abs(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)

    return new_series, indices


def smooth_epidata(data_4, smooth_factor=14, week_correction=0, week_smoothing=0):
    if smooth_factor <= 0:
        return data_4

    df = pd.DataFrame(data_4)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.interpolate(method='linear', limit=2, limit_direction='both')

    data_4 =  df

    deldata = np.diff(data_4, axis=1)

    deldata = pd.DataFrame(deldata).T

    data_4_s = data_4

    maxt = data_4.shape[1]

    date_map = np.ceil((np.arange(1, maxt) - np.mod(maxt - 1, 7)) / 7).reshape(1, -1)

    cleandel = deldata

    if week_correction == 1:
        for cid in range(len(data_4)):
            week_dat = pd.DataFrame(data_4).loc[cid][list(range((maxt - 1) % 7, maxt, 7))].diff()[1:]

            

            clean_week_tmp, TF = hampel_filter_forloop(week_dat.to_numpy())

            week_dat_tmp = week_dat.to_list()

            for idx_out in TF:
                week_dat_tmp[idx_out] = np.nan

            clean_week = pd.DataFrame(week_dat_tmp).interpolate(method='linear')

            week_dat_1 = week_dat.to_list()
            week_dat_1.append(0)
            peak_idx = np.array([i for i in range(1, len(week_dat_1) - 1) if
                                 week_dat_1[i] > week_dat_1[i - 1] and week_dat_1[i] > week_dat_1[i + 1]])

            tf_vals = []
            for i in TF:
                if i in peak_idx:

                    tf_vals.append(i)

            for jj in tf_vals:
                get_idx = [w for w in range(len(date_map[0])) if date_map[0][w] == jj]
                for w in get_idx:
                    cleandel.iloc[w][cid] = clean_week.iloc[jj][0] / 7

        deldata = cleandel

    if week_smoothing == 1:
        temp = np.cumsum(deldata.T, axis=1)

        temp['new'] = 0  ### add a constant colum with value 0

        temp = temp[[list(temp.columns)[-1]] + list(temp.columns)[0:-1]]

        temp_array = np.array(temp)

        diff = np.diff(temp_array[:, (maxt - 1) % 7::7].T, axis=0)

        week_dat = pd.DataFrame(diff)

        xx = np.full((deldata.shape), np.nan)

        xx[int((maxt - 1 - 7 * week_dat.shape[0]) + 7) - 1:maxt - 1:7, :] = np.cumsum(week_dat, axis=0)

        xx = pd.DataFrame(xx).interpolate(method='linear', limit_direction='both')

        deldata.iloc[1:, :] = np.diff(xx, axis=0)

    deldata[deldata < 0] = 0

    for state_idx in range(deldata.shape[1]):
        deldata[state_idx] = deldata[state_idx].rolling(smooth_factor, min_periods=0).mean()

    data_4_s = np.concatenate((data_4.iloc[:, :1].to_numpy(), np.cumsum(deldata, axis=0).T), axis=1)

    return data_4_s



def opt_fun(X):
    def F(w):
        return X @ (1 / (1+np.exp(-w)))
    return F

