import numpy as np
import warnings
from approaches import get_approach
import config_param


def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp):
    """Delegates scenario processing to the selected approach module."""
    warnings.filterwarnings("ignore")
    approach_name = getattr(config_param, "selected_approach", "SIKJalpha Basic")
    approach = get_approach(approach_name)
    return approach.process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp)