import pandas as pd
import numpy as np
import config_param
from approaches import get_approach
from process_scenario import process_scenario
from math import ceil
from preprocess.util_function import smooth_epidata
from itertools import product
from loky import ProcessPoolExecutor, as_completed


def generate_predictors(hosp_cumu_s_org, hosp_dat, popu, config_param, retro_lookback, progress_callback = None):

    maxt = hosp_dat.shape[1]
    days = maxt
    hosp_cumu_orig = np.nancumsum(np.nan_to_num(hosp_dat), axis=1)
    num_states = len(popu)

    # Prepare tasks across both retro_lookback and scenarios (one task per (lookback, scenario))
    all_preds = {}
    approach_name = getattr(config_param, "selected_approach", "SIKJalpha Basic")
    approach = get_approach(approach_name)

    # Metadata and tracking
    meta = {}
    results_by_x = {}
    remaining_by_x = {}
    finished_x = set()

    # Build tasks and per-lookback metadata
    tasks = []
    for x in retro_lookback:
        wks_back = x
        T_full = days - wks_back * config_param.bin_size
        ns = hosp_cumu_s_org.shape[0]

        if wks_back == 0:
            hosp_cumu_s_cut = hosp_cumu_s_org[:, :]
            hosp_cumu_cut = hosp_cumu_orig[:, :]
        else:
            hosp_cumu_s_cut = hosp_cumu_s_org[:, :-(wks_back * config_param.bin_size)]
            hosp_cumu_cut = hosp_cumu_orig[:, :-(wks_back * config_param.bin_size)]

        scen_list = approach.build_scenarios(config_param)
        config_params = approach.make_config(config_param)
        base_hosp = hosp_cumu_cut[:, T_full - 1]
        n_scen = scen_list.shape[0]

        meta[x] = {"ns": ns, "horizon": config_param.horizon, "n_scen": n_scen}
        results_by_x[x] = [None] * n_scen
        remaining_by_x[x] = n_scen

        for simnum in range(n_scen):
            scenario = scen_list[simnum]
            tasks.append(
                (
                    x,
                    simnum,
                    (simnum, scenario),
                    hosp_cumu_s_cut,
                    hosp_cumu_cut,
                    popu,
                    config_params,
                    base_hosp,
                )
            )

    # Submit all tasks in a single pool; the executor will handle distribution
    with ProcessPoolExecutor() as executor:
        future_map = {}
        total_tasks = len(tasks)
        tasks_done = 0
        for (x, simnum, simargs, hosp_s, hosp_c, popu_, cfg_params, base_h) in tasks:
            fut = executor.submit(
                process_scenario, simargs, hosp_s, hosp_c, popu_, cfg_params, base_h
            )
            future_map[fut] = (x, simnum)

        for future in as_completed(future_map):
            x, simnum = future_map[future]
            try:
                res = future.result()
            except Exception:
                res = None
            results_by_x[x][simnum] = res
            remaining_by_x[x] -= 1

            # Smooth, task-based progress update across all lookbacks and scenarios
            try:
                tasks_done += 1
                if total_tasks > 0 and progress_callback is not None:
                    progress_callback(tasks_done / total_tasks)
            except Exception:
                pass

            # When a lookback completes, aggregate and emit predictors for that x
            if remaining_by_x[x] == 0 and x not in finished_x:
                finished_x.add(x)
                ns = meta[x]["ns"]
                horizon = meta[x]["horizon"]
                n_scen = meta[x]["n_scen"]

                net_hosp_A = np.empty((n_scen, ns, horizon))
                net_hosp_A[:] = np.nan
                for s in range(n_scen):
                    if results_by_x[x][s] is not None:
                        net_hosp_A[s, :, :] = np.nanmean(results_by_x[x][s], axis=0)

                p = net_hosp_A
                predictors = np.diff(p, axis=2)
                lo = predictors[:, :, 0 : config_param.weeks_ahead * config_param.bin_size]
                all_preds[x] = lo

                print(f"Done for lookback {x}. {len(finished_x)}/{len(retro_lookback)} completed.")

    return all_preds