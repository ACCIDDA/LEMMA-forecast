import streamlit as st
import time
import config_param
import importlib
import numpy as np
import gen_predictions
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sys
#sys.path.append('~/code/lemma-repo/LEMMA')
from shared_utils.utils import bin_array
from input_models import model_config
if "forecast_ready" not in st.session_state:
    st.session_state["forecast_ready"] = False

st.set_page_config(initial_sidebar_state= "collapsed", layout="wide", page_title="Forecast Generator", page_icon="💡")

CONFIG_FILE = "config_param.py"

EDITABLE_PARAMS = ["weeks_ahead", "bin_size", "start_train", "end_train", "start_test", "end_test", "quantiles"]

# EDITABLE_PARAMS = PARAMS + hyper_params
st.markdown("👈 **Open the sidebar to update predictors**")

st.markdown(
    """
    <h1 style="text-align: center;">Forecast Generation 💡</h1>
    <style>
        .block-container {
            padding-top: 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def get_parameters():
    return {attr: getattr(config_param, attr) for attr in dir(config_param) if not attr.startswith("__")}


# FOR ADDING PARAMETERS THAT DO NOT EXIST IN THE CONFIG FILE
def update_config_file(updated_keys):
    with open(CONFIG_FILE, "r") as f:
        lines = f.readlines()

    new_lines = []
    found_keys = set()

    for line in lines:
        if "=" not in line:
            new_lines.append(line)
            continue

        key = line.split("=")[0].strip()

        if key in updated_keys:
            value = updated_keys[key]
            found_keys.add(key)
        else:
            new_lines.append(line)
            continue

        # Format value
        if key == "hosp_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "location_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "popu":
            formatted_value = f"np.loadtxt('{value}')"
        elif key == "halpha_list":
            try:
                start = round(float(value[0]), 5)
                step = round(float(value[1] - value[0]), 5)
                stop = round(float(value[-1] + step), 5)
                formatted_value = f"np.arange({start}, {stop}, {step})"
            except Exception:
                formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, np.ndarray):
            formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, list):
            formatted_value = str(value)
        elif isinstance(value, str):
            formatted_value = f'"{value}"'
        else:
            formatted_value = value

        new_lines.append(f"{key} = {formatted_value}\n")

    # Add any missing (new) keys
    for key in updated_keys.keys() - found_keys:
        value = updated_keys[key]

        if key == "hosp_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "location_dat":
            formatted_value = f"pd.read_csv('{value}')"
        elif key == "popu":
            formatted_value = f"np.loadtxt('{value}')"
        elif key == "halpha_list":
            try:
                start = round(float(value[0]), 5)
                step = round(float(value[1] - value[0]), 5)
                stop = round(float(value[-1] + step), 5)
                formatted_value = f"np.arange({start}, {stop}, {step})"
            except Exception:
                formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, np.ndarray):
            formatted_value = f"np.array({value.tolist()})"
        elif isinstance(value, list):
            formatted_value = str(value)
        elif isinstance(value, str):
            formatted_value = f'"{value}"'
        else:
            formatted_value = value

        new_lines.append(f"{key} = {formatted_value}\n")

    try:
        with open(CONFIG_FILE, "w") as f:
            f.writelines(new_lines)

        importlib.reload(config_param)
    except Exception as e:
        st.error(f"❌ Error updating config file: {e}")
        return





params = get_parameters()
updated_params = {}

st.sidebar.header("🔬 Predictor Settings")
st.sidebar.markdown("### 🔧 Model Selection")

est_model = st.sidebar.selectbox("Estimation Model", list(model_config.estimation_models.keys()))
sim_model = st.sidebar.selectbox("Simulation Model", list(model_config.simulation_models.keys()))

st.session_state["selected_estimation_model"] = est_model
st.session_state["selected_simulation_model"] = sim_model
st.markdown("---")

st.sidebar.markdown("### 🔧 Model Hyperparameters")
        
model_hyperparams = list(set(
    model_config.estimation_models.get(est_model, []) +
    model_config.simulation_models.get(sim_model, [])
))

for param in model_hyperparams:
    default = None
    input_type = "text"
    param_key = f"hyperparam_{param}"  # Unique key per param

    if hasattr(config_param, param):
        default = getattr(config_param, param)
        if isinstance(default, int):
            input_type = "int"
        elif isinstance(default, float):
            input_type = "float"
        elif isinstance(default, (list, np.ndarray)):
            input_type = "array"

    # Render input field with unique key
    if input_type == "int":
        val = st.sidebar.number_input(
            param,
            value=default if default is not None else 0,
            step=1,
            key=param_key
        )
    elif input_type == "float":
        val = st.sidebar.number_input(
            param,
            value=default if default is not None else 0.0,
            format="%.5f",
            key=param_key
        )
    elif input_type == "array":
        val_str = st.sidebar.text_input(
            param,
            value=", ".join(map(str, default)) if default is not None else "",
            key=param_key
        )
        val = np.array([float(v.strip()) for v in val_str.split(",") if v.strip()])
    else:
        val = st.sidebar.text_input(
            param,
            value=str(default) if default is not None else "",
            key=param_key
        )
        try:
            val_eval = eval(val)
            val = val_eval
        except Exception:
            pass

    updated_params[param] = val
with st.sidebar.expander("ℹ️ How to configure hyperparameters"):
    st.markdown("""
    - **bin_size**: Size of the rolling window.
    - **rlags**: Smoothing factor for the forecast. Higher values = smoother.
    - **un_list**: Number of training iterations.
    - **S**: Choose a loss function that matches your objective.
    - **num_dh_rates_sample**: Number of samples drawn for hospitalization delay rates.
    """)


if "train_ranges" not in st.session_state:
    st.session_state.train_ranges = [(params["start_train"], params["end_train"])]
if "test_ranges" not in st.session_state:
    st.session_state.test_ranges = [(params["start_test"], params["end_test"])]

def render_range_inputs(label, ranges_key):
    st.markdown(f"### {label} Ranges")

    # Initialize add/remove trackers in session_state
    if f"{ranges_key}_add_clicked" not in st.session_state:
        st.session_state[f"{ranges_key}_add_clicked"] = False
    if f"{ranges_key}_remove_clicked" not in st.session_state:
        st.session_state[f"{ranges_key}_remove_clicked"] = None

    # Handle add clicked
    if st.session_state[f"{ranges_key}_add_clicked"]:
        st.session_state[ranges_key].append((0, 0))
        st.session_state[f"{ranges_key}_add_clicked"] = False  # reset

    # Handle remove clicked
    remove_idx = st.session_state[f"{ranges_key}_remove_clicked"]
    if remove_idx is not None and 0 <= remove_idx < len(st.session_state[ranges_key]):
        st.session_state[ranges_key].pop(remove_idx)
        st.session_state[f"{ranges_key}_remove_clicked"] = None  # reset

    new_ranges = []

    for i, (start, end) in enumerate(st.session_state[ranges_key]):
        cols = st.columns([3, 3, 1])
        with cols[0]:
            start_val = st.number_input(f"{label} Start {i+1}", value=start, key=f"{ranges_key}_start_{i}")
        with cols[1]:
            end_val = st.number_input(f"{label} End {i+1}", value=end, key=f"{ranges_key}_end_{i}", min_value=start_val)
        with cols[2]:
            if st.button("➖", key=f"{ranges_key}_remove_{i}"):
                st.session_state[f"{ranges_key}_remove_clicked"] = i

        new_ranges.append((start_val, end_val))

    st.session_state[ranges_key] = new_ranges

    # Add range button
    if st.button(f"➕ Add {label} Range", key=f"add_{ranges_key}"):
        st.session_state[f"{ranges_key}_add_clicked"] = True

    return new_ranges



left_col, right_col = st.columns(2)  # Adjust width

with left_col:
    st.markdown("### 📂 Input Files (Source Files)")
    hosp_dat_file = st.file_uploader("Upload Target Data (CSV)", type=["csv"])
    if hosp_dat_file:
        hosp_dat = pd.read_csv(hosp_dat_file, header=None)
        try:
            hosp_dat.to_csv("data/ts_dat.csv", index=False, header=False)  
            updated_params["ts_dat"] = "data/ts_dat.csv"  
            st.success("✅ Input data updated!")
            update_config_file(updated_params)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    location_file = st.file_uploader("Upload Location, Population Data (CSV)", type=["csv"])
    if location_file:
        try:
            location_dat = pd.read_csv(location_file, delimiter=',')
            location_dat.to_csv("data/location_dat.csv", index=False)  
            
            # Force reload the config to pick up the new data
            importlib.reload(config_param)
            
            st.success("✅ Location data updated!")
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
         

    st.markdown("---")
    st.markdown("### 🎯 Target Parameters")
    st.subheader("Forecast Target Configurations 🔧")

    train_ranges = render_range_inputs("Training Steps", "train_ranges")
    test_ranges = render_range_inputs("Forecasting Steps", "test_ranges")

    maxTbinned = len(bin_array(config_param.hosp_dat[0, :], 0, config_param.bin_size, 0))

    # Merge lookbacks
    train_days = sorted(set(np.concatenate([np.arange(s, e+2) for s, e in train_ranges])))
    train_lookback = [maxTbinned - i for i in train_days]
    test_days = sorted(set(np.concatenate([np.arange(s, e+1) for s, e in test_ranges])))
    test_lookback = [maxTbinned - i for i in test_days]
    retro_lookback = sorted(set(train_lookback + test_lookback))

    #print(retro_lookback)

    updated_params["retro_lookback"] = np.array(retro_lookback)
    updated_params["test_lookback"] = np.array(test_lookback)

    # st.markdown("#### Preview")
    # st.code(f"Run the predictors for:  {retro_lookback}")
    # st.code(f"Generate forecasts for: {test_lookback}")




    skip_keys = {"start_train", "end_train", "start_test", "end_test"}
    for idx, key in enumerate(EDITABLE_PARAMS):
        if key in skip_keys:
            continue
        
        if key not in params:
            continue

        value = params[key]
        new_value = value  # Default


        st.write(f"**{key}**")

        if isinstance(value, int):
            new_value = st.number_input(f"{key}", value=value, step=1, label_visibility="collapsed", min_value=0)
        elif isinstance(value, float):
            new_value = st.number_input(f"{key}", value=value, step=0.01, label_visibility="collapsed", min_value=0.0)
        elif isinstance(value, str):
            new_value = st.text_input(f"{key}", value=value, label_visibility="collapsed")
        elif isinstance(value, list):
            new_value = st.text_area(f"{key} (Comma-separated)", value=", ".join(map(str, value)))
            new_value = [v.strip() for v in new_value.split(",") if v.strip()]
            try:
                new_value = [float(v) for v in new_value]
                if any(v < 0 for v in new_value):
                    st.error(f"❌ {key} cannot contain negative values.")
                    new_value = value  # Revert to old value
            except ValueError:
                st.error(f"❌ Invalid input for {key}. Please enter numbers only.")
                new_value = value  # Revert to old value

        elif isinstance(value, np.ndarray):
            new_value = st.text_area(f"{key} (Comma-separated NumPy Array)", value=", ".join(map(str, value.tolist())))
            try:
                new_value = np.array([float(v.strip()) for v in new_value.split(",") if v.strip()])
                if (new_value < 0).any():
                    st.error(f"❌ {key} cannot contain negative values.")
                    new_value = value  # Revert to old value
            except ValueError:
                st.error(f"❌ Invalid input for {key}. Please enter numbers only.")
                new_value = value

            # Store updates only if changed
        if isinstance(value, np.ndarray) and not np.array_equal(new_value, value):
            updated_params[key] = new_value
        elif isinstance(value, list) and new_value != value:
            updated_params[key] = new_value
        elif not isinstance(value, (list, np.ndarray)) and new_value != value:
            updated_params[key] = new_value


with right_col:
   

    st.subheader("📊 Forecast Results")
    if st.button("💾 Save & Run Forecasts", use_container_width=True):
        if updated_params:
            update_config_file(updated_params)
            st.toast("✅ Configuration Updated!")
        
            progress_bar = st.progress(0.0)
            progress_text = st.empty()  # Create an empty container for text

            def update_progress(progress):
                progress_bar.progress(progress)
                progress_text.info(f"Progress: {int(progress * 100)}%")

            st.session_state["all_preds"], st.session_state["hosp_dat"] = gen_predictions.generate_all_preds(update_progress)

        st.toast("✅ Predictors generated successfully!")

        with st.spinner('Generating "ensemble" forecasts...'):
            st.session_state["preds"] = gen_predictions.generate_preds(
                st.session_state["all_preds"], st.session_state["hosp_dat"]
            )
        st.toast("✅ Predictions generated successfully!")
        st.session_state["forecast_ready"] = True
  
    if st.session_state.get("forecast_ready"):    
        if st.button("🔁 Run Predictions", use_container_width=True):
            if updated_params:
                update_config_file(updated_params)
                st.toast("✅ Predictors Configuration Updated!")
                
            if "all_preds" in st.session_state and "hosp_dat" in st.session_state:
                with st.spinner("Re-generating predictions..."):
                    st.session_state["preds"] = gen_predictions.generate_preds(
                        st.session_state["all_preds"], st.session_state["hosp_dat"]
                    )
                st.toast("✅ Predictions re-generated successfully!")
                st.session_state["forecast_ready"] = True
            else:
                st.error("❌ Please run the initial forecast generation first.")

    state_abbr = config_param.state_abbr
    
    state_select = st.selectbox("Select State", state_abbr)
    cid = state_abbr.index(state_select)    
    hosp_dat = config_param.hosp_dat
    maxt = hosp_dat.shape[1]
    observed_data = bin_array((config_param.hosp_dat[cid, :]), 0, config_param.bin_size, 0)



# Create the Plotly figure
    fig = go.Figure()

    # Add observed data trace
    fig.add_trace(go.Scatter(
        x=list(range(0, len(observed_data))),
        y=observed_data,
        mode='lines',
        name='Observed'
    ))

    fig.update_layout(
        title=f"📊 Ground Truth for {state_select}",
        xaxis_title="Day",
        yaxis_title="Values",
        template="plotly_white"
    )

    if st.session_state.get("forecast_ready"):
        all_preds = st.session_state["all_preds"]
        preds = st.session_state["preds"]

        #cid represents the state
        
        #x represents the lookback for which we are going to generate the predictions
        dd = st.selectbox(
            "Select Forecast Day",
            options=[maxTbinned-i for i in list(config_param.test_lookback)],
            index=len(config_param.test_lookback) - 1
        )
        x = maxTbinned - dd
        
        ## Specify a range of final predictions
        test_lookback = x
        pred_start = (maxt - test_lookback * config_param.bin_size)
        tt = np.arange(2 + pred_start // config_param.bin_size, pred_start // config_param.bin_size + config_param.weeks_ahead + 2)


        if (test_lookback, cid) in preds:
            prediction_data = preds[(test_lookback, cid)]  # Assuming this is a 2D array
            num_lines = prediction_data.shape[1]  # Number of columns (lines)

            for i in range(num_lines):
                fig.add_trace(go.Scatter(
                x=tt,
                y=prediction_data[:, i],  # Add each column as a separate line
                mode='lines',
                name=f'Predicted ({config_param.quantiles[i]})'  # Label each line uniquely
                ))
                fig.update_layout(
                    title=f"📊 Ground Truth and Forecasts for {state_select}",
                )
        else:
            st.warning("Prediction not available for this selection.")

    # Render the Plotly chart
    st.plotly_chart(fig, use_container_width=True)