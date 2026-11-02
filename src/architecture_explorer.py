from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st

from src.architecture_inspector import (
    MODEL_DISPLAY_NAMES,
    inspect_model_architecture,
    calculate_tensor_flow,
    get_component_inspection_details,
    extract_parameter_statistics,
    get_lstm_gate_weight_statistics,
    slice_parameter_matrix,
    get_city_date_bounds,
    compute_real_gate_history,
    extract_single_neuron_series,
    run_prediction_walkthrough,
)
from src.architecture_diagram import (
    get_architecture_diagram_data,
    build_architecture_diagram,
    build_weight_heatmap,
    build_neuron_six_timeseries_grid,
    build_neuron_combined_gates_chart,
    build_gate_activation_heatmap,
    build_walkthrough_temperature_chart,
    build_concatenated_vector_chart,
)
from src.playground_canvas import render_interactive_neural_playground


def _init_explorer_session_state() -> None:
    """Initializes persistent Streamlit session state keys for the Architecture Explorer."""
    defaults = {
        "arch_model_key": "model_B",
        "arch_mode": "Beginner",
        "arch_batch_size": 1,
        "arch_selected_component": "lstm",
        "arch_selected_city": "Agartala",
        "arch_window_preset": "Test Split Benchmark (Ends 2025-10-19 → Predicts 2025-10-20)",
        "arch_custom_end_date": "2025-10-19",
        "arch_selected_neuron": 25,
        "arch_selected_param": "lstm.weight_hh_l0",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _inject_playground_styles() -> None:
    """Injects clean CSS styling for compact cards, non-truncating metric pills, and badges."""
    st.markdown(
        """
        <style>
        .arch-card {
            background-color: #111827;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }
        .arch-badge {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 600;
            background-color: #1E293B;
            color: #38BDF8;
            border: 1px solid #38BDF8;
            margin-right: 6px;
        }
        .arch-shape-pill {
            font-family: Consolas, monospace;
            background-color: #1E1B4B;
            color: #C7D2FE;
            padding: 2px 8px;
            border-radius: 6px;
            border: 1px solid #6366F1;
            font-size: 13px;
        }
        .arch-metric-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
            margin-bottom: 12px;
        }
        .arch-metric-box {
            background: linear-gradient(180deg, #162033 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 8px 12px;
        }
        .arch-metric-lbl {
            font-size: 11px;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 3px;
        }
        .arch-metric-val {
            font-size: 15px;
            font-weight: 700;
            color: #F8FAFC;
            font-family: Consolas, monospace;
            white-space: nowrap;
        }
        .comp-dim-grid {
            display: grid;
            grid-template-columns: 1fr 1.45fr 0.85fr;
            gap: 12px;
            margin-top: 4px;
            margin-bottom: 14px;
        }
        .comp-dim-card {
            background: linear-gradient(180deg, #162033 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 12px 14px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .comp-dim-lbl {
            font-size: 12px;
            font-weight: 600;
            color: #94A3B8;
            margin-bottom: 6px;
        }
        .comp-dim-val {
            font-size: 20px;
            font-weight: 700;
            color: #F8FAFC;
            font-family: Consolas, monospace;
            line-height: 1.35;
            word-break: break-word;
        }
        .comp-dim-line {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            background: rgba(30, 41, 59, 0.75);
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 4px 9px;
            margin-bottom: 4px;
            font-family: Consolas, monospace;
            font-size: 13.5px;
        }
        .comp-dim-line:last-child {
            margin-bottom: 0;
        }
        .comp-dim-line-key {
            color: #94A3B8;
            font-weight: 600;
        }
        .comp-dim-line-shape {
            color: #38BDF8;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_plotly_with_selection(fig, key: str):
    """
    Renders a Plotly chart with point-click selection support on Streamlit >= 1.35,
    falling back gracefully to standard rendering on older versions.
    """
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
            on_select="rerun",
            selection_mode="points",
            theme=None,
        )
        if event and hasattr(event, "selection"):
            points = event.selection.get("points", [])
            if points:
                custom = points[0].get("customdata")
                if isinstance(custom, list) and custom:
                    custom = custom[0]
                if isinstance(custom, str) and custom:
                    if st.session_state.get("arch_selected_component") != custom:
                        st.session_state["arch_selected_component"] = custom
                        st.rerun()
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key, theme=None)


def _format_dimension_card_content(raw_dim_str: str) -> str:
    """Formats single or multi-part tensor dimension strings into non-truncating HTML."""
    text = str(raw_dim_str)
    if " | " in text:
        parts = [p.strip() for p in text.split(" | ") if p.strip()]
        rows_html = []
        for part in parts:
            if ":" in part:
                k, v = part.split(":", 1)
                rows_html.append(
                    f'<div class="comp-dim-line">'
                    f'<span class="comp-dim-line-key">{k.strip()}:</span>'
                    f'<span class="comp-dim-line-shape">{v.strip()}</span>'
                    f'</div>'
                )
            else:
                rows_html.append(
                    f'<div class="comp-dim-line"><span class="comp-dim-line-shape">{part}</span></div>'
                )
        return "".join(rows_html)

    if " + " in text:
        parts = [p.strip() for p in text.split(" + ") if p.strip()]
        rows_html = []
        for part in parts:
            rows_html.append(
                f'<div class="comp-dim-line"><span class="comp-dim-line-shape">{part}</span></div>'
            )
        return "".join(rows_html)

    return f'<div class="comp-dim-val">{text}</div>'


def _render_component_and_tensor_tab(
    summary: Dict[str, Any],
    batch_size: int,
    mode: str,
    selected_component: str,
) -> None:
    """Renders Tab 1: Component & Layer Inspector + Stage-by-Stage Tensor Flow."""
    diagram_data = get_architecture_diagram_data(
        summary=summary,
        batch_size=batch_size,
        mode=mode,
        selected_component=selected_component,
    )
    node_list = diagram_data["nodes"]
    tab_btn_cols = st.columns(len(node_list))
    for idx, node in enumerate(node_list):
        is_active = node["id"] == selected_component
        btn_label = f"{'🟡 ' if is_active else ''}{node['title']}"
        if tab_btn_cols[idx].button(
            btn_label,
            key=f"tab1_comp_btn_{node['id']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state["arch_selected_component"] = node["id"]
            st.rerun()

    details = get_component_inspection_details(
        summary=summary,
        component_id=selected_component,
        batch_size=batch_size,
    )
    flow = calculate_tensor_flow(summary=summary, batch_size=batch_size)

    col_left, col_right = st.columns([1.15, 0.85], gap="large")

    with col_left:
        st.markdown(
            f"""
            <div class="arch-card">
                <span class="arch-badge">{details['category']}</span>
                <span class="arch-shape-pill">Params: {details['parameter_count']:,}</span>
                <h3 style="margin-top:8px;margin-bottom:4px;">{details['component_name']}</h3>
                <p style="color:#CBD5E1;margin-bottom:8px;">{details['purpose']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        in_dim_html = _format_dimension_card_content(details["input_dimensions"])
        out_dim_html = _format_dimension_card_content(details["output_dimensions"])

        st.markdown(
            f"""
            <div class="comp-dim-grid">
                <div class="comp-dim-card">
                    <div class="comp-dim-lbl">Input Dimensions</div>
                    {in_dim_html}
                </div>
                <div class="comp-dim-card">
                    <div class="comp-dim-lbl">Output Dimensions</div>
                    {out_dim_html}
                </div>
                <div class="comp-dim-card">
                    <div class="comp-dim-lbl">Trainable Parameters</div>
                    <div class="comp-dim-val" style="color:#C084FC;">{details['parameter_count']:,}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if mode == "Beginner":
            st.info(details["beginner_explanation"])
            with st.expander("View Mathematical Operation & Technical Details"):
                st.latex(details["math_operation"])
                st.markdown(details["technical_explanation"])
        else:
            st.markdown("#### Forward-Pass Operation")
            st.latex(details["math_operation"])
            st.markdown(details["technical_explanation"])
            with st.expander("View Beginner Analogy"):
                st.markdown(details["beginner_explanation"])

        if details["weight_and_bias_shapes"]:
            st.markdown("#### Actual Weight & Bias Tensor Shapes")
            wb_rows = [
                {"Parameter": k, "Tensor Shape": str(v), "Elements": int(np.prod(v))}
                for k, v in details["weight_and_bias_shapes"].items()
            ]
            st.dataframe(
                pd.DataFrame(wb_rows),
                use_container_width=True,
                hide_index=True,
            )

        # Special detailed gate breakdown when the LSTM block is selected
        if details["component_id"] == "lstm" and "packed_matrix_info" in details:
            packed = details["packed_matrix_info"]
            st.markdown("#### Packed PyTorch LSTM Matrix Layout (`256 = 4 × 64`)")
            st.markdown(packed["packing_explanation"])

            st.markdown("#### Inside the LSTM Block: The Four Gates & Two Memory States")
            gate_cols = st.columns(2)
            for idx, g_info in enumerate(details["gates_and_states"]):
                with gate_cols[idx % 2]:
                    with st.container(border=True):
                        st.markdown(
                            f"**{g_info['name']}** · `{g_info['activation']}` · `{g_info['slice_rows']}`"
                        )
                        if mode == "Beginner":
                            st.markdown(g_info["beginner"])
                        else:
                            st.markdown(g_info["technical"])
                            st.caption(
                                f"Unpacked shapes: `W_ih`: `{g_info['w_ih_slice_shape']}`, "
                                f"`W_hh`: `{g_info['w_hh_slice_shape']}`"
                            )

    with col_right:
        st.markdown("#### Stage-by-Stage Tensor Dimension Flow")
        st.caption(
            f"Showing exact tensor shapes for **{summary['model_name']}** at **Batch Size = {batch_size}**. "
            "Notice the distinction between the 30-day **LSTM sequence output** and the Day-30 **Final hidden state**."
        )

        for idx, stage in enumerate(flow["stages"], start=1):
            with st.container(border=True):
                top_a, top_b = st.columns([0.62, 0.38])
                top_a.markdown(f"**{idx}. {stage['label']}**")
                top_b.markdown(
                    f"<div style='text-align:right'><span class='arch-shape-pill'>{stage['shape']}</span></div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Axes: `{stage['axis_meaning']}`")
                if mode == "Beginner":
                    st.write(stage["beginner_explanation"])
                else:
                    st.write(stage["technical_explanation"])


def _render_weights_inspector_tab(
    model,
    summary: Dict[str, Any],
    city_to_id: Dict[str, int],
) -> None:
    """Renders Tab 2: Actual Trained Weights Inspector with statistics and controlled heatmaps."""
    st.markdown(
        f"### Actual Trained Weights Inspector — {summary['model_name']}"
    )
    st.caption(
        "All values below are extracted directly from the saved `.pth` checkpoint in read-only mode."
    )

    param_stats = extract_parameter_statistics(model)
    stats_df = pd.DataFrame(param_stats)
    stats_df["shape"] = stats_df["shape"].astype(str)
    stats_df = stats_df.rename(
        columns={
            "parameter_name": "Parameter Name",
            "component": "Component",
            "shape": "Tensor Shape",
            "num_values": "Number of Values",
            "min": "Min",
            "max": "Max",
            "mean": "Mean",
            "std": "Std Dev",
        }
    )

    st.dataframe(
        stats_df.style.format(
            {
                "Number of Values": "{:,}",
                "Min": "{:+.5f}",
                "Max": "{:+.5f}",
                "Mean": "{:+.5f}",
                "Std Dev": "{:.5f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View Unpacked LSTM Gate Weight Statistics (4 Gates × 4 Tensors = 16 Slices)"):
        gate_stats = get_lstm_gate_weight_statistics(model)
        gate_df = pd.DataFrame(gate_stats)
        gate_df["shape"] = gate_df["shape"].astype(str)
        st.dataframe(
            gate_df.style.format(
                {
                    "num_values": "{:,}",
                    "min": "{:+.5f}",
                    "max": "{:+.5f}",
                    "mean": "{:+.5f}",
                    "std": "{:.5f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.markdown("#### Interactive Parameter Matrix Slice & Heatmap")
    st.caption(
        "Large matrices (such as `lstm.weight_hh_l0` with 16,384 values) are sliced into controlled views "
        "so you can inspect specific gates, neurons, or cities clearly."
    )

    param_names = list(summary["parameter_shapes"].keys())
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.2, 1.0])

    with ctrl1:
        selected_param = st.selectbox(
            "Select Trained Parameter Tensor",
            options=param_names,
            index=(
                param_names.index(st.session_state["arch_selected_param"])
                if st.session_state["arch_selected_param"] in param_names
                else 0
            ),
            key="arch_param_selector_widget",
        )
        st.session_state["arch_selected_param"] = selected_param

    raw_shape = summary["parameter_shapes"][selected_param]
    total_rows = 1 if len(raw_shape) == 1 else raw_shape[0]
    total_cols = raw_shape[0] if len(raw_shape) == 1 else raw_shape[1]

    r_start, r_end = 0, min(16, total_rows)
    c_start, c_end = 0, min(16, total_cols)

    with ctrl2:
        if total_rows == 256:
            gate_slice_preset = st.selectbox(
                "LSTM Packed Row Preset (256 = 4 × 64)",
                options=[
                    "Input Gate (i) — First 16 Units [Rows 0..15]",
                    "Forget Gate (f) — First 16 Units [Rows 64..79]",
                    "Candidate Gate (g) — First 16 Units [Rows 128..143]",
                    "Output Gate (o) — First 16 Units [Rows 192..207]",
                    "Full Gate Slice (64 Rows) — Input Gate [0..63]",
                    "Custom Row/Column Range",
                ],
            )
            if gate_slice_preset.startswith("Input Gate"):
                r_start, r_end = 0, 16
            elif gate_slice_preset.startswith("Forget Gate"):
                r_start, r_end = 64, 80
            elif gate_slice_preset.startswith("Candidate Gate"):
                r_start, r_end = 128, 144
            elif gate_slice_preset.startswith("Output Gate"):
                r_start, r_end = 192, 208
            elif gate_slice_preset.startswith("Full Gate"):
                r_start, r_end = 0, 64
            else:
                r_start, r_end = st.slider(
                    "Row Range",
                    min_value=0,
                    max_value=total_rows,
                    value=(0, min(24, total_rows)),
                    step=1,
                )
        elif total_rows > 1:
            r_start, r_end = st.slider(
                "Row Range",
                min_value=0,
                max_value=total_rows,
                value=(0, min(20, total_rows)),
                step=1,
            )
        else:
            st.info(f"Single-row parameter (`total_rows = 1`, `total_cols = {total_cols}`).")

    with ctrl3:
        if total_cols > 1:
            c_start, c_end = st.slider(
                "Column Range",
                min_value=0,
                max_value=total_cols,
                value=(0, min(20, total_cols)),
                step=1,
            )
        else:
            c_start, c_end = 0, 1
            st.info("Single-column parameter (`total_cols = 1`).")

    slice_data = slice_parameter_matrix(
        model=model,
        param_name=selected_param,
        row_start=r_start,
        row_end=r_end,
        col_start=c_start,
        col_end=c_end,
        summary=summary,
        city_to_id=city_to_id,
    )

    heatmap_fig = build_weight_heatmap(slice_data, summary["model_name"])
    st.plotly_chart(heatmap_fig, use_container_width=True, theme=None)

    with st.expander("Inspect Numeric Values of Selected Matrix Slice"):
        st.dataframe(
            slice_data["dataframe"].style.format("{:+.5f}"),
            use_container_width=True,
        )


def _render_neuron_inspector_tab(
    model_key: str,
    summary: Dict[str, Any],
    gate_result: Dict[str, Any],
    walkthrough: Dict[str, Any],
    mode: str,
) -> None:
    """
    Renders Tab 3: LSTM Neuron & Gate Inspector (64 Units) reusing the exact
    30-day scaled `weather_tensor` and `gate_result` from the active city/window.
    """
    st.markdown(
        f"### LSTM Neuron & Gate Inspector — {summary['model_name']}"
    )
    st.caption(
        f"Driven by the real 30-day scaled input tensor `{walkthrough['stage3_scaled_input']['shape']}` "
        f"for **{walkthrough['city']}** (`{walkthrough['window_start_date']}` to `{walkthrough['last_observation_date']}`). "
        "Reuses `LSTMGateEngine` (`src/lstm_engine.py`) and `prepare_gate_data` (`src/visualization.py`)."
    )

    gate_data = gate_result["gate_data"]

    top_n1, top_n2, top_n3 = st.columns([1.3, 1.0, 1.0])
    with top_n1:
        selected_neuron = st.slider(
            "Select Hidden Neuron (1 to 64)",
            min_value=1,
            max_value=64,
            value=int(st.session_state.get("arch_selected_neuron", 25)),
            step=1,
            key="arch_neuron_slider",
        )
        st.session_state["arch_selected_neuron"] = selected_neuron

    neuron_info = extract_single_neuron_series(
        gate_data=gate_data,
        neuron_number=selected_neuron,
        dates=walkthrough["dates"],
    )

    with top_n2:
        st.metric(
            f"Neuron {selected_neuron} Final Hidden (h_30)",
            f"{neuron_info['series']['hidden_state'][-1]:+.5f}",
            delta=f"Day 1: {neuron_info['series']['hidden_state'][0]:+.5f}",
        )
    with top_n3:
        st.metric(
            f"Neuron {selected_neuron} Final Cell (c_30)",
            f"{neuron_info['series']['cell_state'][-1]:+.5f}",
            delta=f"PyTorch Parity Δ: {gate_result['max_hidden_diff_vs_pytorch']:.2e}",
        )

    if mode == "Beginner":
        st.info(
            f"**How to read Neuron {selected_neuron}'s six charts:**\n"
            "- **Input Gate ($i_t$, 0 to 1):** How much of each day's new weather reading this neuron chooses to write in.\n"
            "- **Forget Gate ($f_t$, 0 to 1):** How much of yesterday's memory this neuron retains (`1.0` = remember everything).\n"
            "- **Candidate Gate ($g_t$, -1 to +1):** The new weather note proposed on each day.\n"
            "- **Output Gate ($o_t$, 0 to 1):** How strongly this neuron shares its internal memory each day.\n"
            "- **Cell State ($c_t$):** The neuron's accumulating long-term memory across the 30 days.\n"
            "- **Hidden State ($h_t$):** The neuron's daily output summary. Day 30's value joins the final prediction!"
        )

    grid_fig = build_neuron_six_timeseries_grid(
        gate_data=gate_data,
        neuron_number=selected_neuron,
        dates=walkthrough["dates"],
        model_name=summary["model_name"],
        city=walkthrough["city"],
    )
    st.plotly_chart(grid_fig, use_container_width=True, theme=None)

    sub_c1, sub_c2 = st.columns([1.15, 0.85], gap="large")
    with sub_c1:
        combined_fig = build_neuron_combined_gates_chart(
            gate_data=gate_data,
            neuron_number=selected_neuron,
            dates=walkthrough["dates"],
            model_name=summary["model_name"],
        )
        st.plotly_chart(combined_fig, use_container_width=True, theme=None)

    with sub_c2:
        st.markdown(f"#### Neuron {selected_neuron} — 30-Day Signal Summary")
        n_stats_df = pd.DataFrame(neuron_info["statistics"]).drop(columns=["key"])
        n_stats_df = n_stats_df.rename(
            columns={
                "signal": "Signal",
                "day_1": "Day 1",
                "day_30": "Day 30",
                "min": "Min",
                "max": "Max",
                "mean": "Mean",
                "std": "Std",
            }
        )
        st.dataframe(
            n_stats_df.style.format(
                {
                    "Day 1": "{:+.4f}",
                    "Day 30": "{:+.4f}",
                    "Min": "{:+.4f}",
                    "Max": "{:+.4f}",
                    "Mean": "{:+.4f}",
                    "Std": "{:.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Explore All 64 Neurons × 30 Days Activation Heatmap"):
        hm_gate = st.selectbox(
            "Select Signal for 64-Neuron Heatmap",
            options=[
                ("input_gate", "Input Gate (i_t) — Shape (30, 64)"),
                ("forget_gate", "Forget Gate (f_t) — Shape (30, 64)"),
                ("candidate_gate", "Candidate Gate (g_t) — Shape (30, 64)"),
                ("output_gate", "Output Gate (o_t) — Shape (30, 64)"),
                ("cell_state", "Cell State (c_t) — Shape (30, 64)"),
                ("hidden_state", "Hidden State (h_t) — Shape (30, 64)"),
            ],
            format_func=lambda x: x[1],
        )[0]
        hm_fig = build_gate_activation_heatmap(
            gate_data=gate_data,
            gate_key=hm_gate,
            dates=walkthrough["dates"],
            model_name=summary["model_name"],
        )
        st.plotly_chart(hm_fig, use_container_width=True, theme=None)


def _render_prediction_walkthrough_tab(
    summary: Dict[str, Any],
    walkthrough: Dict[str, Any],
    mode: str,
) -> None:
    """Renders Tab 4: 10-Stage Step-by-Step Prediction Walkthrough."""
    st.markdown(
        f"### Step-by-Step Prediction Walkthrough — {summary['model_name']}"
    )
    st.caption(
        f"Tracing **1 real prediction** for **{walkthrough['city']}** (`city_id = {walkthrough['city_id']}`) "
        f"using the 30-day historical window `{walkthrough['window_start_date']}` → `{walkthrough['last_observation_date']}` "
        f"to predict Next-Day Maximum Temperature on **`{walkthrough['prediction_date']}`**."
    )

    scaled_pred = walkthrough["stage9_dense_output"]["scaled_prediction"]
    celsius_pred = walkthrough["stage10_celsius_prediction"][
        "predicted_max_temperature_celsius"
    ]
    actual_temp = walkthrough["stage10_celsius_prediction"][
        "actual_max_temperature_celsius"
    ]
    abs_err = walkthrough["stage10_celsius_prediction"]["absolute_error_celsius"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Target Forecast Date",
        walkthrough["prediction_date"],
        delta=f"After 30d ending {walkthrough['last_observation_date']}",
        delta_color="off",
    )
    m2.metric(
        "Internal Scaled Output (Stage 9)",
        f"{scaled_pred:.5f}",
        delta="Normalized [0, 1] scale",
        delta_color="off",
    )
    m3.metric(
        "Predicted Max Temp (Stage 10)",
        f"{celsius_pred:.2f} °C",
        delta="Inverse-scaled to Celsius",
        delta_color="off",
    )
    if actual_temp is not None and abs_err is not None:
        m4.metric(
            "Observed Max Temp (Ground Truth)",
            f"{actual_temp:.2f} °C",
            delta=f"Error: {celsius_pred - actual_temp:+.2f} °C (|e|={abs_err:.2f} °C)",
            delta_color="inverse",
        )
    else:
        m4.metric("Observed Max Temp", "N/A", delta="Outside historical table")

    st.plotly_chart(
        build_walkthrough_temperature_chart(walkthrough),
        use_container_width=True,
        theme=None,
    )

    st.markdown("#### Trace All 10 Forward-Pass Stages")

    with st.expander(
        f"Stages 1–3: Raw Weather Observations → Preprocessed Features → Scaled Input Tensor {walkthrough['stage3_scaled_input']['shape']}",
        expanded=True,
    ):
        t1, t2, t3 = st.tabs(
            [
                "1. Original Weather Observations (30 Days)",
                f"2. Preprocessed Temporal Features ({len(summary['temporal_features'])} Features)",
                f"3. Scaled Input Tensor {walkthrough['stage3_scaled_input']['shape']}",
            ]
        )
        with t1:
            st.caption(
                "Raw daily observations validated and sliced to the latest 30 days by `WeatherInference.prepare_history()`."
            )
            st.dataframe(
                walkthrough["stage1_raw_observations"],
                use_container_width=True,
                hide_index=True,
                height=250,
            )
        with t2:
            st.caption(
                f"Feature order enforced from `config.json`: `{summary['temporal_features']}`."
            )
            st.dataframe(
                walkthrough["stage2_preprocessed_features"],
                use_container_width=True,
                hide_index=True,
                height=250,
            )
        with t3:
            st.caption(
                f"Normalized via saved scaler into PyTorch tensor of shape `{walkthrough['stage3_scaled_input']['shape']}`."
            )
            st.dataframe(
                walkthrough["stage3_scaled_input"]["dataframe"],
                use_container_width=True,
                hide_index=True,
                height=250,
            )

    with st.expander(
        f"Stages 4–5: Auxiliary Bypass Inputs — Geography ({walkthrough['stage4_geography']['shape']}) & City Embedding {walkthrough['stage5_city_embedding']['shape']}",
        expanded=True,
    ):
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("##### 4. Geography Tensor (Bypasses LSTM)")
            if walkthrough["stage4_geography"]["has_geography"]:
                geo_rows = [
                    {
                        "Feature": feat,
                        "Raw Physical Value": walkthrough["stage4_geography"]["raw_values"][feat],
                        "Standardized Tensor Value": walkthrough["stage4_geography"]["scaled_values"][feat],
                    }
                    for feat in walkthrough["stage4_geography"]["feature_names"]
                ]
                st.caption(
                    f"Tensor shape: `{walkthrough['stage4_geography']['shape']}` (standardized via `geo_scaler`)."
                )
                st.dataframe(
                    pd.DataFrame(geo_rows).style.format(
                        {
                            "Raw Physical Value": "{:.4f}",
                            "Standardized Tensor Value": "{:+.5f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "**Baseline V1** does not use geographical features (`geography = None`)."
                )

        with b_col2:
            st.markdown("##### 5. City Embedding Vector (Bypasses LSTM)")
            emb_vec = walkthrough["stage5_city_embedding"]["vector"]
            st.caption(
                f"City: **{walkthrough['city']}** (`city_id = {walkthrough['city_id']}`) → "
                f"Row `{walkthrough['city_id']}` of `city_embedding.weight (50, 8)` → Shape `{walkthrough['stage5_city_embedding']['shape']}`."
            )
            emb_df = pd.DataFrame(
                [emb_vec],
                columns=[f"Emb[{i}]" for i in range(len(emb_vec))],
            )
            st.dataframe(
                emb_df.style.format("{:+.5f}"),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander(
        f"Stages 6–8: LSTM Sequence Output {walkthrough['stage6_lstm_output']['shape']} → Final Hidden State {walkthrough['stage7_final_states']['hidden_shape']} → Concatenated Vector {walkthrough['stage8_concatenated_vector']['shape']}",
        expanded=True,
    ):
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown(
                f"##### 6. LSTM Sequence Output `{walkthrough['stage6_lstm_output']['shape']}`"
            )
            st.write(
                "Contains all 30 daily 64-unit hidden states (`Day 1` through `Day 30`). "
                "First 8 units of Day 1 vs. Day 30:"
            )
            seq_preview_df = pd.DataFrame(
                {
                    "Unit": [f"Unit {i + 1}" for i in range(8)],
                    "Day 1 (h_1)": walkthrough["stage6_lstm_output"]["day1_first8"],
                    "Day 30 (h_30)": walkthrough["stage6_lstm_output"]["day30_first8"],
                }
            )
            st.dataframe(
                seq_preview_df.style.format(
                    {"Day 1 (h_1)": "{:+.5f}", "Day 30 (h_30)": "{:+.5f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with s_col2:
            st.markdown(
                f"##### 7. Final Hidden State `{walkthrough['stage7_final_states']['hidden_shape']}`"
            )
            st.write(
                "Extracted from the final time step (`hidden_state[-1]`, identical to Day 30 of the sequence output). "
                "This 64-number vector represents the entire 30-day weather history."
            )
            fh_vec = walkthrough["stage7_final_states"]["final_hidden"]
            st.caption(
                f"Min: `{np.min(fh_vec):+.4f}` | Max: `{np.max(fh_vec):+.4f}` | "
                f"Mean: `{np.mean(fh_vec):+.4f}` | Std: `{np.std(fh_vec):.4f}`"
            )
            st.dataframe(
                pd.DataFrame(
                    [fh_vec[:12]],
                    columns=[f"h_30[{i}]" for i in range(12)],
                ).style.format("{:+.5f}"),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            f"##### 8. Concatenated Feature Vector `{walkthrough['stage8_concatenated_vector']['shape']}`"
        )
        st.plotly_chart(
            build_concatenated_vector_chart(walkthrough),
            use_container_width=True,
            theme=None,
        )

    with st.expander(
        f"Stages 9–10: Dense Layer Scaled Output ({scaled_pred:.5f}) → Inverse-Scaled Prediction ({celsius_pred:.2f} °C)",
        expanded=True,
    ):
        d_col1, d_col2 = st.columns(2)
        contribs = walkthrough["stage9_dense_output"]["contributions"]
        with d_col1:
            st.markdown(
                f"##### 9. Dense Layer Output `{walkthrough['stage9_dense_output']['shape']}` (Scaled Space)"
            )
            st.write(
                "Exact additive breakdown of `y_scaled = combined @ W_fc.T + b_fc`:"
            )
            contrib_rows = [
                {
                    "Component Slice": "LSTM Final Hidden State (64 weights [0:64])",
                    "Contribution to Scaled Output": contribs["lstm_hidden_64_sum"],
                },
            ]
            if summary["has_geography"]:
                contrib_rows.append(
                    {
                        "Component Slice": f"Geography Features ({summary['geo_dim']} weights [64:{64 + summary['geo_dim']}])",
                        "Contribution to Scaled Output": contribs["geography_sum"],
                    }
                )
            contrib_rows.extend(
                [
                    {
                        "Component Slice": "City Embedding (8 weights)",
                        "Contribution to Scaled Output": contribs["city_embedding_8_sum"],
                    },
                    {
                        "Component Slice": "Dense Layer Bias (fc.bias)",
                        "Contribution to Scaled Output": contribs["fc_bias"],
                    },
                    {
                        "Component Slice": "Total Scaled Prediction (Stage 9 Output)",
                        "Contribution to Scaled Output": contribs["total_scaled_sum"],
                    },
                ]
            )
            st.dataframe(
                pd.DataFrame(contrib_rows).style.format(
                    {"Contribution to Scaled Output": "{:+.6f}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

        with d_col2:
            st.markdown("##### 10. Inverse-Scaled Prediction in Celsius (°C)")
            st.success(
                f"**Internal Scaled Output (`Stage 9`):** `{scaled_pred:.6f}` (normalized `[0, 1]` scale)\n\n"
                f"**Final Physical Forecast (`Stage 10`):** **`{celsius_pred:.2f} °C`** on `{walkthrough['prediction_date']}`\n\n"
                + (
                    f"**Ground-Truth Observed Max Temp:** `{actual_temp:.2f} °C` "
                    f"(Absolute Error: `{abs_err:.2f} °C`)"
                    if actual_temp is not None
                    else ""
                )
            )
            st.caption(
                "Computed via `WeatherInference.inverse_transform_prediction()` using the saved target `MinMaxScaler`. "
                "Verified identical to `WeatherInference.predict_next_day()`."
            )


def render_architecture_explorer(artifacts: Dict[str, Any]) -> None:
    """
    Main entry point for the Trained Architecture Explorer page in Streamlit.
    Reuses the preloaded `artifacts` dictionary from `app.py` without loading
    duplicate model instances.
    """
    _init_explorer_session_state()
    _inject_playground_styles()

    st.subheader("🧠 Interactive LSTM Neural Network Playground")
    st.caption(
        "Hands-on, real-time neural network playground powered by our three trained PyTorch weather models. "
        "**Play or scrub Day 1 → Day 30**, **click any of the 64 neurons or 4 gates**, **double-click neurons to mute/ablate them**, "
        "and **drag live What-If weather sliders** to watch real-time gate activations and next-day °C forecasts respond!"
    )

    # ==========================================
    # 1. TOP CONTROL BAR
    # ==========================================
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns(
        [1.15, 1.0, 0.75, 1.1, 1.4]
    )

    model_options = ["baseline", "model_A", "model_B"]
    with ctrl_c1:
        selected_model_key = st.selectbox(
            "1. Trained Model",
            options=model_options,
            index=model_options.index(st.session_state["arch_model_key"]),
            format_func=lambda k: MODEL_DISPLAY_NAMES[k],
        )
        st.session_state["arch_model_key"] = selected_model_key

    with ctrl_c2:
        selected_mode = st.radio(
            "2. Explanation Mode",
            options=["Beginner", "Technical"],
            index=0 if st.session_state["arch_mode"] == "Beginner" else 1,
            horizontal=True,
        )
        st.session_state["arch_mode"] = selected_mode

    with ctrl_c3:
        batch_options = [1, 2, 4, 8, 16, 32, 64, 128]
        selected_batch_size = st.selectbox(
            "3. Batch Size (B)",
            options=batch_options,
            index=batch_options.index(st.session_state["arch_batch_size"])
            if st.session_state["arch_batch_size"] in batch_options
            else 0,
        )
        st.session_state["arch_batch_size"] = selected_batch_size

    cities_sorted = sorted(list(artifacts["city_to_id"].keys()))
    with ctrl_c4:
        selected_city = st.selectbox(
            "4. City (Real 30-Day Input)",
            options=cities_sorted,
            index=(
                cities_sorted.index(st.session_state["arch_selected_city"])
                if st.session_state["arch_selected_city"] in cities_sorted
                else 0
            ),
        )
        st.session_state["arch_selected_city"] = selected_city

    bounds = get_city_date_bounds(selected_city)
    preset_labels = list(bounds["presets"].keys()) + ["Custom End Date"]
    with ctrl_c5:
        selected_preset = st.selectbox(
            "5. Historical 30-Day Window",
            options=preset_labels,
            index=(
                preset_labels.index(st.session_state["arch_window_preset"])
                if st.session_state["arch_window_preset"] in preset_labels
                else 0
            ),
        )
        st.session_state["arch_window_preset"] = selected_preset

    if selected_preset == "Custom End Date":
        custom_dt = st.date_input(
            "Select 30-Day Window Final Observation Date",
            value=pd.to_datetime(st.session_state["arch_custom_end_date"]).date(),
            min_value=pd.to_datetime(bounds["min_end_date"]).date(),
            max_value=pd.to_datetime(bounds["max_end_date"]).date(),
        )
        active_end_date = custom_dt.strftime("%Y-%m-%d")
        st.session_state["arch_custom_end_date"] = active_end_date
    else:
        active_end_date = bounds["presets"][selected_preset]

    # Extract live architecture summary from the loaded model
    summary = inspect_model_architecture(selected_model_key, artifacts)

    if (
        st.session_state["arch_selected_component"] == "geography"
        and not summary["has_geography"]
    ):
        st.session_state["arch_selected_component"] = "lstm"

    # ==========================================
    # 2. NON-TRUNCATING MODEL METRIC STRIP
    # ==========================================
    bypass_str = (
        f"Geo ({summary['geo_dim']}) + City ({summary['city_embedding_dim']})"
        if summary["has_geography"]
        else f"City ({summary['city_embedding_dim']}) only"
    )
    st.markdown(
        f"""
        <div class="arch-metric-grid">
            <div class="arch-metric-box">
                <div class="arch-metric-lbl">PyTorch Architecture</div>
                <div class="arch-metric-val" style="color:#38BDF8;">{summary['architecture_class']}</div>
            </div>
            <div class="arch-metric-box">
                <div class="arch-metric-lbl">Temporal Sequence Input</div>
                <div class="arch-metric-val">{summary['sequence_length']} days × {summary['lstm_input_size']} features</div>
            </div>
            <div class="arch-metric-box">
                <div class="arch-metric-lbl">Recurrent LSTM Core</div>
                <div class="arch-metric-val" style="color:#C084FC;">{summary['lstm_hidden_size']} units ({summary['component_parameters']['lstm']:,} params)</div>
            </div>
            <div class="arch-metric-box">
                <div class="arch-metric-lbl">Auxiliary Bypass Track</div>
                <div class="arch-metric-val" style="color:#34D399;">{bypass_str}</div>
            </div>
            <div class="arch-metric-box">
                <div class="arch-metric-lbl">Dense Readout & Total Params</div>
                <div class="arch-metric-val" style="color:#FBBF24;">{summary['fc_input_size']} → {summary['fc_output_size']} ({summary['trainable_parameters']:,} total)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute the real 30-day walkthrough and 64-unit gate histories once for both
    # the Interactive Neural Network Playground and the 4 deep-dive tabs below
    walkthrough = run_prediction_walkthrough(
        model_key=selected_model_key,
        city=selected_city,
        artifacts=artifacts,
        end_date=active_end_date,
    )
    gate_result = compute_real_gate_history(
        model_key=selected_model_key,
        artifacts=artifacts,
        city=selected_city,
        weather_tensor=walkthrough["weather_tensor"],
    )
    gate_result["dates"] = walkthrough["dates"]

    # ==========================================
    # 3. LIVE INTERACTIVE NEURAL NETWORK PLAYGROUND CANVAS
    # ==========================================
    render_interactive_neural_playground(
        model_key=selected_model_key,
        summary=summary,
        artifacts=artifacts,
        walkthrough=walkthrough,
        gate_result=gate_result,
        batch_size=selected_batch_size,
        mode=selected_mode,
        selected_neuron=int(st.session_state.get("arch_selected_neuron", 25)),
    )

    # Optional High-Level Block Architecture Diagram & Synchronized Selector
    with st.expander(
        "📐 View High-Level Block Architecture Diagram & Component Selector",
        expanded=False,
    ):
        diagram_fig = build_architecture_diagram(
            summary=summary,
            batch_size=selected_batch_size,
            mode=selected_mode,
            selected_component=st.session_state["arch_selected_component"],
        )
        _render_plotly_with_selection(
            diagram_fig,
            key=f"arch_canvas_{selected_model_key}_{selected_batch_size}_{selected_mode}",
        )

        diagram_data = get_architecture_diagram_data(
            summary=summary,
            batch_size=selected_batch_size,
            mode=selected_mode,
            selected_component=st.session_state["arch_selected_component"],
        )
        node_list = diagram_data["nodes"]
        btn_cols = st.columns(len(node_list))
        for idx, node in enumerate(node_list):
            is_active = node["id"] == st.session_state["arch_selected_component"]
            btn_label = f"{'🟡 ' if is_active else ''}{node['title']}"
            if btn_cols[idx].button(
                btn_label,
                key=f"comp_btn_{node['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["arch_selected_component"] = node["id"]
                st.rerun()

    st.divider()
    st.markdown("### 📊 Deep-Dive Analytical Workspaces (Tables, Weight Slicer & Full 10-Stage DataFrames)")

    # ==========================================
    # 4. FOUR DEEP-DIVE WORKSPACE TABS
    # ==========================================
    tab_layer, tab_weights, tab_neurons, tab_walkthrough = st.tabs(
        [
            "🔍 1. Component & Tensor Flow Inspector",
            "⚖️ 2. Actual Trained Weights Inspector",
            "🧠 3. LSTM Neuron & Gate Inspector (64 Units)",
            "🌡️ 4. Prediction Walkthrough (10 Stages)",
        ]
    )

    with tab_layer:
        _render_component_and_tensor_tab(
            summary=summary,
            batch_size=selected_batch_size,
            mode=selected_mode,
            selected_component=st.session_state["arch_selected_component"],
        )

    with tab_weights:
        _render_weights_inspector_tab(
            model=artifacts["models"][selected_model_key],
            summary=summary,
            city_to_id=artifacts["city_to_id"],
        )

    with tab_neurons:
        _render_neuron_inspector_tab(
            model_key=selected_model_key,
            summary=summary,
            gate_result=gate_result,
            walkthrough=walkthrough,
            mode=selected_mode,
        )

    with tab_walkthrough:
        _render_prediction_walkthrough_tab(
            summary=summary,
            walkthrough=walkthrough,
            mode=selected_mode,
        )
