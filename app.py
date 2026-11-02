import importlib
import streamlit as st
from src.artifacts import load_all_artifacts
import src.architecture_inspector
import src.architecture_diagram
import src.playground_canvas
import src.architecture_explorer
import src.gate_memory_inspector
import src.gate_memory_explorer
import src.model_comparison_inspector
import src.model_comparison_explorer

importlib.reload(src.architecture_inspector)
importlib.reload(src.architecture_diagram)
importlib.reload(src.playground_canvas)
importlib.reload(src.architecture_explorer)
importlib.reload(src.gate_memory_inspector)
importlib.reload(src.gate_memory_explorer)
importlib.reload(src.model_comparison_inspector)
importlib.reload(src.model_comparison_explorer)

from src.architecture_explorer import render_architecture_explorer
from src.gate_memory_explorer import render_gate_memory_explorer
from src.model_comparison_explorer import render_model_comparison_explorer

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LSTM Playground",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. LOAD TRAINED MODELS AND ARTIFACTS
# ==========================================

@st.cache_resource
def load_project_artifacts():
    return load_all_artifacts()


try:
    artifacts = load_project_artifacts()
    models = artifacts["models"]

except Exception as e:
    st.error(f"Could not load project artifacts: {e}")
    st.stop()

# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("🌦️ LSTM Playground")

st.markdown(
    """
    ### Understanding LSTM Networks Through Weather Forecasting

    Explore how an LSTM processes weather observations,
    updates its internal memory, and predicts next-day
    maximum temperature.
    """
)

st.divider()

# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose an explorer",
    [
        "Home",
        "Architecture Explorer",
        "Gate & Memory Explorer",
        "Model Comparison"
    ]
)

st.sidebar.success("All models loaded successfully")
st.sidebar.caption(
    f"Execution device: {artifacts['device']}"
)

# ==========================================
# 5. PAGE CONTENT
# ==========================================

if page == "Home":

    st.subheader("Welcome to LSTM Playground")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            """
            **Architecture Explorer**

            Understand the structure of our trained
            LSTM models, their layers, tensor dimensions,
            and information flow.
            """
        )

    with col2:
        st.info(
            """
            **Gate & Memory Explorer**

            Explore actual input, forget, candidate,
            and output gate activations across
            a 30-day weather sequence.
            """
        )

    with col3:
        st.info(
            """
            **Model Comparison**

            Compare Baseline V1, Model A, and Model B
            head-to-head across 50 Indian cities and
            18,750 held-out test forecasts.
            """
        )

    st.subheader("Our Learning Journey")

    st.markdown(
        """
        1. Understand the input weather sequence.
        2. Explore the LSTM architecture.
        3. Inspect the four LSTM gates.
        4. Understand cell-state and hidden-state updates.
        5. Observe the next-day temperature prediction.
        """
    )

    # --------------------------------------
    # AVAILABLE TRAINED MODELS
    # --------------------------------------

    st.divider()

    st.subheader("Available Trained Models")

    model_info = {
        "Baseline V1": {
            "input_features": 1,
            "hidden_units": 64,
            "geo_features": 0
        },
        "Model A": {
            "input_features": 8,
            "hidden_units": 64,
            "geo_features": 3
        },
        "Model B": {
            "input_features": 8,
            "hidden_units": 64,
            "geo_features": 4
        }
    }

    col1, col2, col3 = st.columns(3)

    for column, (model_name, details) in zip(
        [col1, col2, col3],
        model_info.items()
    ):

        with column:

            st.metric(
                "Hidden Units",
                details["hidden_units"]
            )

            st.markdown(f"**{model_name}**")

            st.write(
                "Input features:",
                details["input_features"]
            )

            st.write(
                "Geography features:",
                details["geo_features"]
            )

    st.divider()

    st.write(
        "Execution device:",
        artifacts["device"]
    )


# ==========================================
# ARCHITECTURE EXPLORER
# ==========================================

elif page == "Architecture Explorer":

    render_architecture_explorer(artifacts)


# ==========================================
# GATE AND MEMORY EXPLORER
# ==========================================

elif page == "Gate & Memory Explorer":

    render_gate_memory_explorer(artifacts)


# ==========================================
# MODEL COMPARISON
# ==========================================

elif page == "Model Comparison":

    render_model_comparison_explorer(artifacts)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "LSTM Playground | Educational ML Project"
)
