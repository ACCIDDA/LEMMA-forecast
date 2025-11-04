from sklearn_quantile import RandomForestQuantileRegressor
# import config_param

default_random_state_base = 1
default_n_estimators = 50  #Moved from Config_param. 

def get_models(num_models):
    models = [
        RandomForestQuantileRegressor(default_n_estimators, 
                                       random_state=default_random_state_base + i) 
        for i in range(num_models)
    ]
    return models

# --- Pluggable ensemble factory ---
from ensembles import BasicEnsemble, RFQuantileEnsemble

def create_ensemble(name: str | None, **kwargs):
    """
    Create a pluggable ensemble strategy.
    name: one of {"rf", "random forest", "basic"}
    """
    choice = (name or "rf").lower().replace("_", " ")
    if choice in ("basic",):
        return BasicEnsemble()
    if choice in ("rf", "random forest", "random_forest", "rfq"):
        n_estimators = kwargs.get("n_estimators", default_n_estimators)
        rs = kwargs.get("random_state_base", default_random_state_base)
        return RFQuantileEnsemble(n_estimators=n_estimators, random_state_base=rs)
    raise ValueError(f"Unknown ensemble '{name}'")
