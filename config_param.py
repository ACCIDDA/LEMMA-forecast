import numpy as np
from datetime import datetime

import pandas as pd

############################
# offset = 11
zero_date = datetime(2021, 9, 1)
days_back = 0
bin_size = 7
#### model determining
weeks_ahead = 4
smooth_factor = 7
####
num_dh_rates_sample = 1
season_start = datetime(2023, 9, 30)
season_end = zero_date
season_start_day = (season_start - season_end).days

##############################
import config_model
import shared_utils.utils
import preprocess.util_function as pp

############ Model Hyperparams ##################
rlags = np.array([0])
rlag_list = np.arange(1, len(rlags) + 1)
un_list = np.array([50.0])
halpha_list = np.arange(0.98, 0.92, -0.02)
S = np.array([0.0])
hyperparams_lists = [halpha_list, rlag_list, un_list, S]
hk = 2
hjp = 7
########################

npredictors = (len(S) * len(halpha_list) * len(un_list) * len(rlag_list))*(weeks_ahead)
horizon = (weeks_ahead+1)*bin_size 


decay_factor = 0.99
wks_back = 1
# default_n_estimators = 100     #Moved to Config_model
# quantiles = [0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]
quantiles = np.array([0.025, 0.5, 0.975])


#Retro Lookback
# predictor_generation_lookbacks = 20:30
# predictor_predict = 27:30



# range = np.arange(start, end)
# split = int(0.8 * len(range))
# retro_train, retro_test  = range[:split], range[split:]
# retro_lookback = np.concatenate((retro_train, retro_test))

#hosp_cumu_s_org= np.loadtxt('data/hosp_cumu_s.csv', delimiter=',')
hosp_dat = pd.read_csv('data/ts_dat.csv', delimiter=',', header = None).to_numpy()

location_dat = pd.read_csv("data/location_dat.csv", delimiter=',')


alpha = 1
beta = 1


start_train = 28
end_train = 34
retro_lookback = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

start_test = 35
end_test = 38
test_lookback = np.array([103, 102, 101, 100])

predictor_progress = 0

ts_dat = "data/ts_dat.csv"
hosp_dat_cumu = pp.smooth_epidata(np.cumsum(hosp_dat, axis=1), 1)
hosp_dat = np.diff(hosp_dat_cumu, axis=1)
hosp_dat = np.concatenate((hosp_dat[:, 0:1], hosp_dat), axis=1)
hosp_cumu_s_org = pp.smooth_epidata(np.cumsum(hosp_dat, axis=1))

popu = location_dat['population'].to_numpy()
state_abbr = location_dat['location_name'].to_list()
