import numpy as np
import warnings
from approaches import get_approach


def process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp, approach_name=None):
    """Delegates scenario processing to the selected approach module."""
    warnings.filterwarnings("ignore")
    selected = approach_name or "SIKJalpha Basic"
    approach = get_approach(selected)
    return approach.process_scenario(args, hosp_cumu_s, hosp_cumu, popu, config_params, base_hosp)