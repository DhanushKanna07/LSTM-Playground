from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.architecture_inspector import calculate_tensor_flow


COMPONENT_STYLES = {
    "temporal": {
        "category_label": "Temporal Input",
        "fill": "rgba(6, 182, 212, 0.18)",
        "border": "#22D3EE",
        "accent": "#06B6D4",
        "text": "#ECFEFF",
    },
    "recurrent": {
        "category_label": "Recurrent Memory Core",
        "fill": "rgba(99, 102, 241, 0.22)",
        "border": "#818CF8",
        "accent": "#6366F1",
        "text": "#EEF2FF",
    },
    "auxiliary_geo": {
        "category_label": "Static Auxiliary Input (Bypass)",
        "fill": "rgba(16, 185, 129, 0.18)",
        "border": "#34D399",
        "accent": "#10B981",
        "text": "#ECFDF5",
    },
    "auxiliary_city": {
        "category_label": "Learned Auxiliary Embedding (Bypass)",
        "fill": "rgba(245, 158, 11, 0.18)",
        "border": "#FBBF24",
        "accent": "#F59E0B",
        "text": "#FFFBEB",
    },
    "fusion": {
        "category_label": "Feature Fusion",
        "fill": "rgba(56, 189, 248, 0.18)",
        "border": "#38BDF8",
        "accent": "#0284C7",
        "text": "#F0F9FF",
    },
    "dense": {
        "category_label": "Dense Projection Layer",
        "fill": "rgba(168, 85, 247, 0.20)",
        "border": "#C084FC",
        "accent": "#A855F7",
        "text": "#FAF5FF",
    },
    "output": {
        "category_label": "Prediction Output",
        "fill": "rgba(244, 63, 94, 0.20)",
        "border": "#FB7185",
        "accent": "#F43F5E",
        "text": "#FFF1F2",
    },
}

NEURON_SIGNAL_CONFIG = [
    {
        "key": "input_gate",
        "title": "Input Gate (i_t)",
        "color": "#2DD4BF",
        "fill": "rgba(45, 212, 191, 0.12)",
        "yrange": [-0.05, 1.05],
    },
    {
        "key": "forget_gate",
        "title": "Forget Gate (f_t)",
        "color": "#60A5FA",
        "fill": "rgba(96, 165, 250, 0.12)",
        "yrange": [-0.05, 1.05],
    },
    {
        "key": "candidate_gate",
        "title": "Candidate Gate (g_t)",
        "color": "#F59E0B",
        "fill": "rgba(245, 158, 11, 0.12)",
        "yrange": [-1.08, 1.08],
    },
    {
        "key": "output_gate",
        "title": "Output Gate (o_t)",
        "color": "#A78BFA",
        "fill": "rgba(167, 139, 250, 0.12)",
        "yrange": [-0.05, 1.05],
    },
    {
        "key": "cell_state",
        "title": "Cell State (c_t)",
        "color": "#34D399",
        "fill": "rgba(52, 211, 153, 0.15)",
        "yrange": None,
    },
    {
        "key": "hidden_state",
        "title": "Hidden State (h_t)",
        "color": "#F472B6",
        "fill": "rgba(244, 114, 182, 0.15)",
        "yrange": [-1.08, 1.08],
    },
]


def get_architecture_diagram_data(
    summary: Dict[str, Any],
    batch_size: int = 1,
    mode: str = "Beginner",
    selected_component: Optional[str] = "lstm",
) -> Dict[str, Any]:
    """
    Builds the structured node and directional edge specifications for the
    selected model's architecture diagram.

    Ensures geography (if present) and city embedding are placed on parallel
    bypass tracks that skip the LSTM and join `final_hidden_state` at
    the concatenation node.
    """
    flow = calculate_tensor_flow(summary, batch_size=batch_size)
    shapes = flow["shapes"]

    has_geo = bool(summary["has_geography"])
    seq_len = int(summary["sequence_length"])
    in_features = int(summary["lstm_input_size"])
    hidden_size = int(summary["lstm_hidden_size"])
    geo_dim = int(summary["geo_dim"])
    vocab_size = int(summary["city_vocab_size"])
    emb_dim = int(summary["city_embedding_dim"])
    fc_in = int(summary["fc_input_size"])
    fc_out = int(summary["fc_output_size"])
    comp_params = summary["component_parameters"]

    top_y = 0.76 if has_geo else 0.70
    geo_y = 0.44
    city_y = 0.14 if has_geo else 0.26
    merge_y = 0.48 if has_geo else 0.50

    nodes: List[Dict[str, Any]] = [
        {
            "id": "weather_input",
            "title": "Weather Input",
            "subtitle": f"{seq_len}d × {in_features} feat",
            "tensor_shape": shapes["weather_input"],
            "input_shape": shapes["weather_input"],
            "output_shape": shapes["weather_input"],
            "params": 0,
            "style_key": "temporal",
            "x": 0.09,
            "y": top_y,
            "width": 0.14,
            "height": 0.18,
            "bypasses_lstm": False,
            "hover_summary": (
                f"{seq_len}-day historical weather window with {in_features} scaled daily feature(s)."
                if mode == "Beginner"
                else f"Input tensor X ∈ R^{shapes['weather_input']} ({', '.join(summary['temporal_features'])})."
            ),
        },
        {
            "id": "lstm",
            "title": f"LSTM Layer ({hidden_size} Units)",
            "subtitle": (
                f"Seq: {shapes['lstm_sequence_output']} → h_30: {shapes['final_hidden_state']}"
            ),
            "tensor_shape": shapes["final_hidden_state"],
            "sequence_output_shape": shapes["lstm_sequence_output"],
            "cell_state_shape": shapes["final_cell_state"],
            "input_shape": shapes["weather_input"],
            "output_shape": shapes["final_hidden_state"],
            "params": comp_params["lstm"],
            "style_key": "recurrent",
            "x": 0.35,
            "y": top_y,
            "width": 0.18,
            "height": 0.21,
            "bypasses_lstm": False,
            "hover_summary": (
                f"Compact block of {hidden_size} memory units (4 gates: i, f, g, o). "
                f"Processes all {seq_len} days and outputs Final Hidden State {shapes['final_hidden_state']}."
                if mode == "Beginner"
                else (
                    f"nn.LSTM(input_size={in_features}, hidden_size={hidden_size}, num_layers=1). "
                    f"Seq output {shapes['lstm_sequence_output']}; slices final step h_30 {shapes['final_hidden_state']}."
                )
            ),
        },
    ]

    if has_geo:
        nodes.append(
            {
                "id": "geography",
                "title": "Geography Input",
                "subtitle": f"{geo_dim} static features (Bypass)",
                "tensor_shape": shapes["geography"],
                "input_shape": shapes["geography"],
                "output_shape": shapes["geography"],
                "params": 0,
                "style_key": "auxiliary_geo",
                "x": 0.35,
                "y": geo_y,
                "width": 0.18,
                "height": 0.16,
                "bypasses_lstm": True,
                "hover_summary": (
                    f"Static city coordinates & elevation ({', '.join(summary['geo_features'])}). "
                    f"Bypasses the LSTM and goes straight to Concatenation."
                    if mode == "Beginner"
                    else (
                        f"Auxiliary tensor G ∈ R^{shapes['geography']} ({', '.join(summary['geo_features'])}). "
                        f"Bypasses recurrent gates."
                    )
                ),
            }
        )

    nodes.extend(
        [
            {
                "id": "city_embedding",
                "title": "City Embedding",
                "subtitle": f"Lookup ({vocab_size}, {emb_dim}) (Bypass)",
                "tensor_shape": shapes["city_embedding"],
                "input_shape": (batch_size,),
                "output_shape": shapes["city_embedding"],
                "params": comp_params["city_embedding"],
                "style_key": "auxiliary_city",
                "x": 0.35,
                "y": city_y,
                "width": 0.18,
                "height": 0.16,
                "bypasses_lstm": True,
                "hover_summary": (
                    f"Learned {emb_dim}-number profile for each of the {vocab_size} cities. "
                    f"Bypasses the LSTM and joins at Concatenation."
                    if mode == "Beginner"
                    else (
                        f"nn.Embedding({vocab_size}, {emb_dim}) -> E ∈ R^{shapes['city_embedding']} "
                        f"({comp_params['city_embedding']:,} trainable params). Bypasses LSTM."
                    )
                ),
            },
            {
                "id": "concatenation",
                "title": "Concatenation",
                "subtitle": (
                    f"64 + {geo_dim} + {emb_dim} = {fc_in}"
                    if has_geo
                    else f"64 + {emb_dim} = {fc_in}"
                ),
                "tensor_shape": shapes["concatenated_vector"],
                "input_shape": (
                    f"{shapes['final_hidden_state']} + {shapes['geography']} + {shapes['city_embedding']}"
                    if has_geo
                    else f"{shapes['final_hidden_state']} + {shapes['city_embedding']}"
                ),
                "output_shape": shapes["concatenated_vector"],
                "params": 0,
                "style_key": "fusion",
                "x": 0.62,
                "y": merge_y,
                "width": 0.15,
                "height": 0.19,
                "bypasses_lstm": False,
                "hover_summary": (
                    f"Joins the LSTM's 64-number summary with "
                    + (
                        f"{geo_dim} geography numbers and {emb_dim} city embedding numbers -> {fc_in} numbers."
                        if has_geo
                        else f"{emb_dim} city embedding numbers -> {fc_in} numbers."
                    )
                    if mode == "Beginner"
                    else f"torch.cat(..., dim=1) -> combined vector ∈ R^{shapes['concatenated_vector']}."
                ),
            },
            {
                "id": "fc",
                "title": "Fully Connected",
                "subtitle": f"Linear({fc_in} → {fc_out})",
                "tensor_shape": shapes["dense_output"],
                "input_shape": shapes["concatenated_vector"],
                "output_shape": shapes["dense_output"],
                "params": comp_params["fc"],
                "style_key": "dense",
                "x": 0.80,
                "y": merge_y,
                "width": 0.13,
                "height": 0.18,
                "bypasses_lstm": False,
                "hover_summary": (
                    f"Weighs all {fc_in} combined clues ({comp_params['fc']} learned weights/bias) "
                    f"to compute 1 scaled temperature prediction."
                    if mode == "Beginner"
                    else (
                        f"nn.Linear(in_features={fc_in}, out_features={fc_out}) with W ∈ R^(1,{fc_in}), "
                        f"b ∈ R^(1,) ({comp_params['fc']} params)."
                    )
                ),
            },
            {
                "id": "output",
                "title": "Next-Day Max Temp",
                "subtitle": f"Output {shapes['dense_output']} → °C",
                "tensor_shape": shapes["dense_output"],
                "input_shape": shapes["dense_output"],
                "output_shape": shapes["dense_output"],
                "params": 0,
                "style_key": "output",
                "x": 0.94,
                "y": merge_y,
                "width": 0.11,
                "height": 0.18,
                "bypasses_lstm": False,
                "hover_summary": (
                    "Final predicted next-day maximum temperature (inverse-scaled to °C)."
                    if mode == "Beginner"
                    else f"Scalar prediction ŷ ∈ R^{shapes['dense_output']} inverse-transformed via target_scaler to °C."
                ),
            },
        ]
    )

    node_map = {n["id"]: n for n in nodes}

    edges: List[Dict[str, Any]] = [
        {
            "source": "weather_input",
            "target": "lstm",
            "tensor_shape": shapes["weather_input"],
            "label": f"{shapes['weather_input']}",
            "detail": f"30-day sequence {shapes['weather_input']}",
            "is_bypass": False,
        },
        {
            "source": "lstm",
            "target": "concatenation",
            "tensor_shape": shapes["final_hidden_state"],
            "label": f"h_30: {shapes['final_hidden_state']}",
            "detail": (
                f"Final hidden state {shapes['final_hidden_state']} "
                f"(from seq {shapes['lstm_sequence_output']})"
            ),
            "is_bypass": False,
        },
    ]

    if has_geo:
        edges.append(
            {
                "source": "geography",
                "target": "concatenation",
                "tensor_shape": shapes["geography"],
                "label": f"Geo: {shapes['geography']}",
                "detail": f"Geography tensor {shapes['geography']} (bypasses LSTM)",
                "is_bypass": True,
            }
        )

    edges.extend(
        [
            {
                "source": "city_embedding",
                "target": "concatenation",
                "tensor_shape": shapes["city_embedding"],
                "label": f"City: {shapes['city_embedding']}",
                "detail": f"City embedding {shapes['city_embedding']} (bypasses LSTM)",
                "is_bypass": True,
            },
            {
                "source": "concatenation",
                "target": "fc",
                "tensor_shape": shapes["concatenated_vector"],
                "label": f"{shapes['concatenated_vector']}",
                "detail": f"Concatenated vector {shapes['concatenated_vector']}",
                "is_bypass": False,
            },
            {
                "source": "fc",
                "target": "output",
                "tensor_shape": shapes["dense_output"],
                "label": f"{shapes['dense_output']}",
                "detail": f"Dense output {shapes['dense_output']} -> °C",
                "is_bypass": False,
            },
        ]
    )

    return {
        "model_key": summary["model_key"],
        "model_name": summary["model_name"],
        "batch_size": batch_size,
        "mode": mode,
        "selected_component": selected_component,
        "nodes": nodes,
        "node_map": node_map,
        "edges": edges,
        "flow": flow,
    }


def _bezier_curve(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    num_points: int = 36,
) -> Tuple[List[float], List[float]]:
    """
    Generates a smooth horizontal S-curve (cubic Bezier) between two component ports.
    """
    cx0 = x0 + 0.55 * (x1 - x0)
    cy0 = y0
    cx1 = x0 + 0.45 * (x1 - x0)
    cy1 = y1

    t = np.linspace(0.0, 1.0, num_points)
    bx = (
        ((1 - t) ** 3) * x0
        + 3 * ((1 - t) ** 2) * t * cx0
        + 3 * (1 - t) * (t**2) * cx1
        + (t**3) * x1
    )
    by = (
        ((1 - t) ** 3) * y0
        + 3 * ((1 - t) ** 2) * t * cy0
        + 3 * (1 - t) * (t**2) * cy1
        + (t**3) * y1
    )
    return bx.tolist(), by.tolist()


def build_architecture_diagram(
    summary: Dict[str, Any],
    batch_size: int = 1,
    mode: str = "Beginner",
    selected_component: Optional[str] = "lstm",
) -> go.Figure:
    """
    Constructs an interactive Plotly architecture diagram inspired by
    TensorFlow Neural Network Playground.
    """
    diagram_data = get_architecture_diagram_data(
        summary=summary,
        batch_size=batch_size,
        mode=mode,
        selected_component=selected_component,
    )
    nodes = diagram_data["nodes"]
    node_map = diagram_data["node_map"]
    edges = diagram_data["edges"]

    fig = go.Figure()

    for edge in edges:
        src = node_map[edge["source"]]
        dst = node_map[edge["target"]]

        x0 = src["x"] + src["width"] / 2.0
        y0 = src["y"]
        x1 = dst["x"] - dst["width"] / 2.0
        y1 = dst["y"]

        if edge["target"] == "concatenation":
            if edge["source"] == "lstm":
                y1 = dst["y"] + dst["height"] * 0.28
            elif edge["source"] == "city_embedding":
                y1 = dst["y"] - dst["height"] * 0.28

        is_highlighted = selected_component in (edge["source"], edge["target"])
        src_style = COMPONENT_STYLES[src["style_key"]]

        line_color = (
            src_style["border"]
            if is_highlighted
            else ("rgba(148, 163, 184, 0.72)" if not edge["is_bypass"] else "rgba(251, 191, 36, 0.65)")
        )
        line_width = 3.6 if is_highlighted else 2.4
        line_dash = "solid" if not edge["is_bypass"] else "dot"

        bx, by = _bezier_curve(x0, y0, x1, y1)

        fig.add_trace(
            go.Scatter(
                x=bx,
                y=by,
                mode="lines",
                line=dict(color=line_color, width=line_width, dash=line_dash),
                hoverinfo="text",
                hovertext=(
                    f"<b>Connection:</b> {src['title']} → {dst['title']}<br>"
                    f"<b>Tensor Shape:</b> {edge['tensor_shape']}<br>"
                    f"<b>Flow:</b> {edge['detail']}"
                ),
                showlegend=False,
            )
        )

        fig.add_annotation(
            x=x1,
            y=y1,
            ax=bx[-4],
            ay=by[-4],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=2.0,
            arrowcolor=line_color,
            text="",
        )

        mid_idx = len(bx) // 2
        mx, my = bx[mid_idx], by[mid_idx]
        if abs(y0 - y1) < 0.02:
            my += 0.048

        fig.add_annotation(
            x=mx,
            y=my,
            xref="x",
            yref="y",
            text=f"<b>{edge['label']}</b>",
            showarrow=False,
            font=dict(size=11, color="#F8FAFC", family="Consolas, monospace"),
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor=line_color,
            borderwidth=1,
            borderpad=3,
        )

    lstm_node = node_map["lstm"]
    shapes_dict = diagram_data["flow"]["shapes"]
    fig.add_annotation(
        x=lstm_node["x"],
        y=lstm_node["y"] + lstm_node["height"] / 2.0 + 0.055,
        xref="x",
        yref="y",
        text=(
            f"All 30 steps: <b>{shapes_dict['lstm_sequence_output']}</b> | "
            f"Cell state c_30: <b>{shapes_dict['final_cell_state']}</b>"
        ),
        showarrow=False,
        font=dict(size=10.5, color="#C7D2FE", family="Consolas, monospace"),
        bgcolor="rgba(30, 27, 75, 0.85)",
        bordercolor="#6366F1",
        borderwidth=1,
        borderpad=4,
    )

    for node in nodes:
        style = COMPONENT_STYLES[node["style_key"]]
        is_selected = node["id"] == selected_component

        x0 = node["x"] - node["width"] / 2.0
        x1 = node["x"] + node["width"] / 2.0
        y0 = node["y"] - node["height"] / 2.0
        y1 = node["y"] + node["height"] / 2.0

        if is_selected:
            pad = 0.010
            fig.add_shape(
                type="rect",
                x0=x0 - pad,
                y0=y0 - pad,
                x1=x1 + pad,
                y1=y1 + pad,
                line=dict(color="#FACC15", width=2.5, dash="solid"),
                fillcolor="rgba(250, 204, 21, 0.07)",
                layer="below",
            )

        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(
                color="#FACC15" if is_selected else style["border"],
                width=3.0 if is_selected else 1.8,
            ),
            fillcolor=style["fill"],
            layer="below",
        )

        param_badge = (
            f"{node['params']:,} params"
            if node["params"] > 0
            else ("0 params (static)" if node["bypasses_lstm"] else "0 params")
        )
        card_html = (
            f"<b>{node['title']}</b><br>"
            f"<span style='font-size:11px;color:#CBD5E1'>{node['subtitle']}</span><br>"
            f"<span style='font-size:10px;color:{style['border']}'>[{param_badge}]</span>"
        )

        fig.add_annotation(
            x=node["x"],
            y=node["y"],
            xref="x",
            yref="y",
            text=card_html,
            showarrow=False,
            font=dict(size=12.5, color=style["text"], family="Arial, sans-serif"),
            align="center",
        )

    click_x: List[float] = []
    click_y: List[float] = []
    click_customdata: List[str] = []
    click_hover: List[str] = []
    click_colors: List[str] = []

    for node in nodes:
        style = COMPONENT_STYLES[node["style_key"]]
        hover_card = (
            f"<b>{node['title']}</b> ({style['category_label']})<br>"
            f"<b>Input Shape:</b> {node['input_shape']}<br>"
            f"<b>Output Shape:</b> {node['output_shape']}<br>"
            f"<b>Trainable Parameters:</b> {node['params']:,}<br>"
            f"<b>Role:</b> {node['hover_summary']}<br>"
            f"<i>Click to inspect this component below</i>"
        )
        for dx in (-0.30, 0.0, 0.30):
            for dy in (-0.25, 0.0, 0.25):
                click_x.append(node["x"] + dx * node["width"])
                click_y.append(node["y"] + dy * node["height"])
                click_customdata.append(node["id"])
                click_hover.append(hover_card)
                click_colors.append(style["border"])

    fig.add_trace(
        go.Scatter(
            x=click_x,
            y=click_y,
            mode="markers",
            marker=dict(
                size=28,
                color=click_colors,
                opacity=0.01,
            ),
            customdata=click_customdata,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=click_hover,
            showlegend=False,
            name="Architecture Components",
        )
    )

    bypass_note = (
        "<b>Auxiliary Bypass Track:</b> Geography & City Embedding skip the 30-day LSTM gates "
        "and concatenate directly with the Day-30 Final Hidden State."
        if summary["has_geography"]
        else (
            "<b>Auxiliary Bypass Track:</b> City Embedding skips the 30-day LSTM gates "
            "and concatenates directly with the Day-30 Final Hidden State."
        )
    )
    fig.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        text=bypass_note,
        showarrow=False,
        font=dict(size=11, color="#94A3B8"),
        bgcolor="rgba(15, 23, 42, 0.85)",
        bordercolor="#334155",
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{summary['model_name']}</b> — Trained Architecture & Tensor Flow "
                f"<span style='font-size:13px;color:#94A3B8'>"
                f"(Batch Size = {batch_size} | Total Params = {summary['trainable_parameters']:,} | {mode} Mode)"
                f"</span>"
            ),
            x=0.01,
            y=0.96,
            font=dict(size=18, color="#F8FAFC"),
        ),
        xaxis=dict(
            range=[0.0, 1.02],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[0.0, 0.98],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=470 if summary["has_geography"] else 420,
        margin=dict(l=15, r=15, t=55, b=15),
        dragmode="pan",
        clickmode="event+select",
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="#475569",
            font=dict(color="#F8FAFC", size=12),
        ),
    )

    return fig


def build_weight_heatmap(
    slice_data: Dict[str, Any],
    model_name: str,
) -> go.Figure:
    """
    Creates an interactive diverging Plotly heatmap for a controlled 2D slice
    of an actual trained parameter tensor from the loaded model.
    """
    matrix = slice_data["matrix"]
    row_labels = slice_data["row_labels"]
    col_labels = slice_data["col_labels"]
    param_name = slice_data["parameter_name"]
    r0, r1 = slice_data["row_range"]
    c0, c1 = slice_data["col_range"]

    max_abs = float(np.max(np.abs(matrix))) if matrix.size > 0 else 1.0
    if max_abs == 0.0:
        max_abs = 1.0

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=col_labels,
            y=row_labels,
            colorscale="RdBu_r",
            zmin=-max_abs,
            zmax=max_abs,
            colorbar=dict(title="Weight"),
            hovertemplate=(
                "<b>Parameter:</b> " + param_name + "<br>"
                "<b>Row:</b> %{y}<br>"
                "<b>Column:</b> %{x}<br>"
                "<b>Trained Value:</b> %{z:+.6f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{model_name}</b> — `{param_name}` "
                f"<span style='font-size:13px;color:#94A3B8'>"
                f"(Rows {r0}..{r1 - 1}, Cols {c0}..{c1 - 1} | Slice shape {slice_data['slice_shape']})"
                f"</span>"
            ),
            x=0.01,
            font=dict(size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            title=slice_data["xaxis_title"],
            tickangle=-30 if len(col_labels) > 8 else 0,
        ),
        yaxis=dict(
            title=slice_data["yaxis_title"],
            autorange="reversed",
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=max(340, min(620, 160 + 22 * len(row_labels))),
        margin=dict(l=60, r=30, t=55, b=65),
    )

    return fig


# ============================================================
# NEURON INSPECTOR CHARTS (CHECKPOINT 7)
# ============================================================

def build_neuron_six_timeseries_grid(
    gate_data: Dict[str, np.ndarray],
    neuron_number: int = 1,
    dates: Optional[List[str]] = None,
    model_name: str = "Model B",
    city: str = "Agartala",
) -> go.Figure:
    """
    Displays the six time-series plots (Input gate, Forget gate, Candidate gate,
    Output gate, Cell state, Hidden state) across 30 days for a selected neuron
    (1..64), reusing the color palette and dark dashboard styling from
    `test_visualization.py`.
    """
    if not (1 <= int(neuron_number) <= 64):
        raise ValueError(
            f"neuron_number must be in 1..64, received {neuron_number}."
        )

    neuron_idx = int(neuron_number) - 1
    days = list(range(1, 31))
    date_labels = dates if (dates and len(dates) == 30) else [f"Day {d}" for d in days]

    subplot_titles = [cfg["title"] for cfg in NEURON_SIGNAL_CONFIG]

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        vertical_spacing=0.11,
        horizontal_spacing=0.08,
    )

    for idx, cfg in enumerate(NEURON_SIGNAL_CONFIG):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        y_vals = gate_data[cfg["key"]][:, neuron_idx]

        fig.add_trace(
            go.Scatter(
                x=days,
                y=y_vals,
                customdata=date_labels,
                mode="lines+markers",
                name=cfg["title"],
                line=dict(color=cfg["color"], width=2.6),
                marker=dict(size=5, color=cfg["color"]),
                fill="tozeroy",
                fillcolor=cfg["fill"],
                hovertemplate=(
                    f"<b>{cfg['title']}</b> (Neuron {neuron_number})<br>"
                    "Step: Day %{x} (%{customdata})<br>"
                    "Value: %{y:+.5f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

        fig.update_xaxes(
            title_text="Weather Day (1..30)" if row == 3 else "",
            showgrid=True,
            gridcolor="#263449",
            zeroline=False,
            row=row,
            col=col,
        )
        fig.update_yaxes(
            range=cfg["yrange"],
            showgrid=True,
            gridcolor="#263449",
            zeroline=True,
            zerolinecolor="#475569",
            row=row,
            col=col,
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Neuron {neuron_number} / 64</b> — Six Internal Gate & Memory Signals Across 30 Days "
                f"<span style='font-size:13px;color:#94A3B8'>({model_name} | City: {city})</span>"
            ),
            x=0.01,
            font=dict(size=18, color="#E5E7EB"),
        ),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="Arial", color="#E5E7EB", size=12),
        height=760,
        showlegend=False,
        margin=dict(l=55, r=30, t=75, b=50),
    )

    return fig


def build_neuron_combined_gates_chart(
    gate_data: Dict[str, np.ndarray],
    neuron_number: int = 1,
    dates: Optional[List[str]] = None,
    model_name: str = "Model B",
) -> go.Figure:
    """
    Overlays the four gate activations (Input, Forget, Candidate, Output) for
    the selected neuron on a single unified timeline chart (reusing the exact
    design from `test_visualization.py`).
    """
    neuron_idx = int(neuron_number) - 1
    days = list(range(1, 31))
    date_labels = dates if (dates and len(dates) == 30) else [f"Day {d}" for d in days]

    fig = go.Figure()

    for cfg in NEURON_SIGNAL_CONFIG[:4]:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=gate_data[cfg["key"]][:, neuron_idx],
                customdata=date_labels,
                mode="lines+markers",
                name=cfg["title"],
                line=dict(color=cfg["color"], width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    f"<b>{cfg['title']}</b>: %{{y:+.5f}} (%{{customdata}})<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"<b>Neuron {neuron_number}</b> — Combined Four Gate Activations ({model_name})",
            x=0.01,
            font=dict(size=17, color="#E5E7EB"),
        ),
        xaxis=dict(
            title="Weather Day (1..30)",
            showgrid=True,
            gridcolor="#263449",
            zeroline=False,
        ),
        yaxis=dict(
            title="Gate Activation",
            showgrid=True,
            gridcolor="#263449",
            zeroline=True,
            zerolinecolor="#475569",
        ),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="Arial", color="#E5E7EB", size=12),
        height=420,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.20),
        margin=dict(l=55, r=30, t=60, b=75),
    )

    return fig


def build_gate_activation_heatmap(
    gate_data: Dict[str, np.ndarray],
    gate_key: str = "input_gate",
    dates: Optional[List[str]] = None,
    model_name: str = "Model B",
) -> go.Figure:
    """
    Builds the (64 neurons × 30 days) gate/state activation heatmap across
    all 64 hidden units (reusing the Plasma heatmap design from `test_visualization.py`).
    """
    title_map = {cfg["key"]: cfg["title"] for cfg in NEURON_SIGNAL_CONFIG}
    display_title = title_map.get(gate_key, gate_key)

    matrix = gate_data[gate_key].T  # shape (64, 30)
    days = list(range(1, 31))
    neurons = list(range(1, 65))

    is_signed = gate_key in ("candidate_gate", "cell_state", "hidden_state")
    zmin = -1.0 if gate_key in ("candidate_gate", "hidden_state") else (
        float(np.min(matrix)) if gate_key == "cell_state" else 0.0
    )
    zmax = 1.0 if gate_key != "cell_state" else float(np.max(matrix))

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=days,
            y=neurons,
            colorscale="RdBu_r" if is_signed else "Plasma",
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Activation"),
            hovertemplate=(
                "Day: %{x}<br>"
                "Neuron: %{y}<br>"
                "Activation: %{z:+.5f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>All 64 Neurons</b> — {display_title} Across 30 Days ({model_name})",
            x=0.01,
            font=dict(size=17, color="#E5E7EB"),
        ),
        xaxis=dict(title="Time Step (Day 1..30)", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Hidden Neuron (1..64)",
            showgrid=False,
            zeroline=False,
            autorange="reversed",
        ),
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(family="Arial", color="#E5E7EB", size=12),
        height=540,
        margin=dict(l=65, r=35, t=60, b=55),
    )

    return fig


# ============================================================
# PREDICTION WALKTHROUGH CHARTS (CHECKPOINT 8)
# ============================================================

def build_walkthrough_temperature_chart(
    walkthrough: Dict[str, Any],
) -> go.Figure:
    """
    Plots the 30-day historical maximum temperature trajectory (°C) together
    with the model's inverse-scaled Next-Day Maximum Temperature prediction
    (and ground-truth observation if available).
    """
    raw_df = walkthrough["stage1_raw_observations"]
    dates = walkthrough["dates"]
    pred_date = walkthrough["prediction_date"]
    pred_temp = walkthrough["stage10_celsius_prediction"][
        "predicted_max_temperature_celsius"
    ]
    actual_temp = walkthrough["stage10_celsius_prediction"][
        "actual_max_temperature_celsius"
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=raw_df["temperature_2m_max"],
            mode="lines+markers",
            name="Historical Max Temp (°C)",
            line=dict(color="#38BDF8", width=2.8),
            marker=dict(size=6, color="#38BDF8"),
            hovertemplate="Date: %{x}<br>Max Temp: %{y:.2f} °C<extra></extra>",
        )
    )

    if "temperature_2m_mean" in raw_df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=raw_df["temperature_2m_mean"],
                mode="lines",
                name="Historical Mean Temp (°C)",
                line=dict(color="#94A3B8", width=1.6, dash="dot"),
                hovertemplate="Date: %{x}<br>Mean Temp: %{y:.2f} °C<extra></extra>",
            )
        )

    # Dashed connector from Day 30 to Day 31 prediction
    fig.add_trace(
        go.Scatter(
            x=[dates[-1], pred_date],
            y=[float(raw_df["temperature_2m_max"].iloc[-1]), pred_temp],
            mode="lines",
            line=dict(color="#F43F5E", width=2.2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[pred_date],
            y=[pred_temp],
            mode="markers+text",
            name=f"Predicted Max Temp ({pred_temp:.2f} °C)",
            marker=dict(
                size=14,
                symbol="star",
                color="#F43F5E",
                line=dict(color="#FFF1F2", width=1.5),
            ),
            text=[f"Pred: {pred_temp:.2f} °C"],
            textposition="top center",
            hovertemplate=(
                f"<b>Prediction Date:</b> {pred_date}<br>"
                f"<b>Predicted Max Temp:</b> {pred_temp:.2f} °C<extra></extra>"
            ),
        )
    )

    if actual_temp is not None:
        fig.add_trace(
            go.Scatter(
                x=[pred_date],
                y=[actual_temp],
                mode="markers",
                name=f"Observed Max Temp ({actual_temp:.2f} °C)",
                marker=dict(
                    size=11,
                    symbol="diamond",
                    color="#10B981",
                    line=dict(color="#ECFDF5", width=1.5),
                ),
                hovertemplate=(
                    f"<b>Prediction Date:</b> {pred_date}<br>"
                    f"<b>Actual Observed:</b> {actual_temp:.2f} °C<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>{walkthrough['city']}</b> — 30-Day Historical Window & Next-Day Forecast "
                f"<span style='font-size:13px;color:#94A3B8'>({walkthrough['model_name']} | Target: {pred_date})</span>"
            ),
            x=0.01,
            font=dict(size=17, color="#F8FAFC"),
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="#1E293B",
            zeroline=False,
        ),
        yaxis=dict(
            title="Temperature (°C)",
            showgrid=True,
            gridcolor="#1E293B",
            zeroline=False,
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=390,
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=55, r=30, t=55, b=75),
    )

    return fig


def build_concatenated_vector_chart(
    walkthrough: Dict[str, Any],
) -> go.Figure:
    """
    Visualizes the concatenated feature vector (length 72, 75, or 76) colored
    by source segment (LSTM Final Hidden [0:64], Geography, City Embedding).
    """
    vec = walkthrough["stage8_concatenated_vector"]["vector"]
    segments = walkthrough["stage8_concatenated_vector"]["segments"]
    geo_slice = segments["geography_slice"]
    city_slice = segments["city_embedding_slice"]

    fig = go.Figure()

    # Segment 1: LSTM Final Hidden State (0..63)
    fig.add_trace(
        go.Bar(
            x=list(range(0, 64)),
            y=vec[0:64],
            name="LSTM Final Hidden State [0..63] (64 dims)",
            marker_color="#6366F1",
            hovertemplate="Index %{x} (LSTM h_30): %{y:+.5f}<extra></extra>",
        )
    )

    # Segment 2: Geography (if present)
    if geo_slice is not None:
        g0, g1 = geo_slice
        fig.add_trace(
            go.Bar(
                x=list(range(g0, g1)),
                y=vec[g0:g1],
                name=f"Geography [{g0}..{g1 - 1}] ({g1 - g0} dims)",
                marker_color="#10B981",
                hovertemplate="Index %{x} (Geography): %{y:+.5f}<extra></extra>",
            )
        )

    # Segment 3: City Embedding
    c0, c1 = city_slice
    fig.add_trace(
        go.Bar(
            x=list(range(c0, c1)),
            y=vec[c0:c1],
            name=f"City Embedding [{c0}..{c1 - 1}] (8 dims)",
            marker_color="#F59E0B",
            hovertemplate="Index %{x} (City Embedding): %{y:+.5f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Concatenated Feature Vector</b> — Shape {walkthrough['stage8_concatenated_vector']['shape']} "
                f"<span style='font-size:13px;color:#94A3B8'>(Input to Dense Linear Layer)</span>"
            ),
            x=0.01,
            font=dict(size=16, color="#F8FAFC"),
        ),
        xaxis=dict(
            title="Concatenated Feature Index",
            showgrid=False,
        ),
        yaxis=dict(
            title="Feature Value",
            showgrid=True,
            gridcolor="#1E293B",
            zeroline=True,
            zerolinecolor="#475569",
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0F172A",
        height=340,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=55, r=30, t=55, b=75),
    )

    return fig
