import numpy as np
import plotly.graph_objects as go
import torch
from src.artifacts import load_all_artifacts
from src.lstm_engine import LSTMGateEngine
from src.visualization import prepare_gate_data


# 1. Load Model B
artifacts = load_all_artifacts()
model = artifacts["models"]["model_B"].cpu()

# 2. Initialize gate engine
engine = LSTMGateEngine(model)
engine.split_gate_weights(print_shapes=False)

# 3. Load actual weather sequence
X_test = np.load(
    "data/processed/artifacts/v2/test/X_test.npy"
)

sequence = torch.tensor(
    X_test[0],
    dtype=torch.float32
)

# 4. Reconstruct LSTM history
history, _, _ = engine.process_sequence(sequence)

# 5. Prepare visualization data
gate_data = prepare_gate_data(history)


# 6. Prepare all four gate matrices
gate_names = {
    "Input Gate": "input_gate",
    "Forget Gate": "forget_gate",
    "Candidate Gate": "candidate_gate",
    "Output Gate": "output_gate"
}

# 7. Create initial heatmap
initial_gate = "input_gate"

fig = go.Figure(
    data=go.Heatmap(
        z=gate_data[initial_gate].T,
        x=list(range(1, 31)),
        y=list(range(1, 65)),
        colorscale="Plasma",
        zmin=0,
        zmax=1,
        colorbar=dict(title="Activation"),
        hovertemplate=(
            "Day: %{x}<br>"
            "Neuron: %{y}<br>"
            "Activation: %{z:.5f}"
            "<extra></extra>"
        )
    )
)

# 8. Create dropdown options
buttons = []

for display_name, data_name in gate_names.items():

    is_candidate = data_name == "candidate_gate"

    buttons.append(
        dict(
            label=display_name,
            method="update",
            args=[
                {
                    "z": [gate_data[data_name].T],
                    "zmin": [-1 if is_candidate else 0],
                    "zmax": [1]
                },
                {
                    "title": f"{display_name} Activation"
                }
            ]
        )
    )

# 9. Apply dashboard styling
fig.update_layout(
    title=dict(
        text="Input Gate Activation",
        x=0.02,
        font=dict(size=22)
    ),
    xaxis=dict(
        title="Time Step (Day)",
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title="Hidden Neuron",
        showgrid=False,
        zeroline=False,
        autorange="reversed"
    ),
    updatemenus=[
        dict(
            buttons=buttons,
            direction="down",
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.15,
            yanchor="top"
        )
    ],
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(
        family="Arial",
        color="#E5E7EB",
        size=12
    ),
    height=720,
    margin=dict(l=75, r=40, t=110, b=65)
)

fig.show()


# ==========================================
# INDIVIDUAL NEURON EXPLORER
# ==========================================

import plotly.graph_objects as go

days = list(range(1, 31))

gate_colors = {
    "Input Gate": "#2DD4BF",
    "Forget Gate": "#60A5FA",
    "Candidate Gate": "#F59E0B",
    "Output Gate": "#A78BFA"
}

# Initial neuron selection
selected_neuron = 25
neuron_id = selected_neuron - 1

# Create figure
neuron_fig = go.Figure()

# Add four gate traces
for display_name, data_name in gate_names.items():

    values = gate_data[data_name][:, neuron_id]

    neuron_fig.add_trace(
        go.Scatter(
            x=days,
            y=values,
            mode="lines",
            name=display_name,
            line=dict(
                color=gate_colors[display_name],
                width=2.5
            )
        )
    )

# Create dropdown options for all 64 neurons
neuron_buttons = []

for neuron in range(64):

    neuron_number = neuron + 1

    new_y_values = [
        gate_data[data_name][:, neuron]
        for data_name in gate_names.values()
    ]

    neuron_buttons.append(
        dict(
            label=f"Neuron {neuron_number}",
            method="update",
            args=[
                {"y": new_y_values},
                {
                    "title": (
                        f"Neuron {neuron_number} "
                        "- Gate Activation Over Time"
                    )
                }
            ]
        )
    )

# Dark dashboard layout
neuron_fig.update_layout(
    title=dict(
        text=f"Neuron {selected_neuron} - Gate Activation Over Time",
        x=0.02,
        font=dict(size=22)
    ),

    xaxis=dict(
        title="Weather Day",
        showgrid=True,
        gridcolor="#263449",
        zeroline=False
    ),

    yaxis=dict(
        title="Activation",
        showgrid=True,
        gridcolor="#263449",
        zeroline=False
    ),

    updatemenus=[
        dict(
            buttons=neuron_buttons,
            direction="down",
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.17,
            yanchor="top",
            bgcolor="#263449",
            bordercolor="#475569",
            font=dict(color="#FFFFFF")
        )
    ],

    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",

    font=dict(
        family="Arial",
        color="#E5E7EB",
        size=12
    ),

    height=600,

    hovermode="x unified",

    margin=dict(
        l=65,
        r=35,
        t=110,
        b=90
    ),

    legend=dict(
        orientation="h",
        y=-0.18
    )
)

neuron_fig.show()


# ==========================================
# MEMORY EXPLORER
# ==========================================

# Get actual cell and hidden states
cell_values = gate_data["cell_state"][:, neuron_id]
hidden_values = gate_data["hidden_state"][:, neuron_id]

# Cell state chart

# ==========================================
# INTERACTIVE MEMORY EXPLORER
# ==========================================

# Initial neuron
initial_neuron = 25
initial_id = initial_neuron - 1

# ------------------------------------------
# 1. CELL STATE EXPLORER
# ------------------------------------------

memory_fig = go.Figure()

memory_fig.add_trace(
    go.Scatter(
        x=days,
        y=gate_data["cell_state"][:, initial_id],
        mode="lines",
        name="Cell State",
        line=dict(color="#2DD4BF", width=3),
        fill="tozeroy",
        fillcolor="rgba(45, 212, 191, 0.10)"
    )
)

cell_buttons = []

for neuron in range(64):

    cell_buttons.append(
        dict(
            label=f"Neuron {neuron + 1}",
            method="update",
            args=[
                {
                    "y": [
                        gate_data["cell_state"][:, neuron]
                    ]
                },
                {
                    "title": (
                        f"Neuron {neuron + 1} "
                        "- Cell State Evolution"
                    )
                }
            ]
        )
    )

memory_fig.update_layout(
    title=f"Neuron {initial_neuron} - Cell State Evolution",
    xaxis_title="Weather Day",
    yaxis_title="Cell State",
    updatemenus=[
        dict(
            buttons=cell_buttons,
            direction="down",
            showactive=True,
            x=1,
            xanchor="right",
            y=1.17,
            yanchor="top"
        )
    ],
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    height=500,
    hovermode="x unified"
)

memory_fig.show()


# ------------------------------------------
# 2. HIDDEN STATE EXPLORER
# ------------------------------------------

hidden_fig = go.Figure()

hidden_fig.add_trace(
    go.Scatter(
        x=days,
        y=gate_data["hidden_state"][:, initial_id],
        mode="lines",
        name="Hidden State",
        line=dict(color="#A78BFA", width=3),
        fill="tozeroy",
        fillcolor="rgba(167, 139, 250, 0.10)"
    )
)

hidden_buttons = []

for neuron in range(64):

    hidden_buttons.append(
        dict(
            label=f"Neuron {neuron + 1}",
            method="update",
            args=[
                {
                    "y": [
                        gate_data["hidden_state"][:, neuron]
                    ]
                },
                {
                    "title": (
                        f"Neuron {neuron + 1} "
                        "- Hidden State Evolution"
                    )
                }
            ]
        )
    )

hidden_fig.update_layout(
    title=f"Neuron {initial_neuron} - Hidden State Evolution",
    xaxis_title="Weather Day",
    yaxis_title="Hidden State",
    updatemenus=[
        dict(
            buttons=hidden_buttons,
            direction="down",
            showactive=True,
            x=1,
            xanchor="right",
            y=1.17,
            yanchor="top"
        )
    ],
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    height=500,
    hovermode="x unified"
)

hidden_fig.show()
