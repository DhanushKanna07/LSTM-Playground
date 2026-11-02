import json
from typing import Dict, Any
import numpy as np
import streamlit.components.v1 as components

from src.architecture_inspector import (
    calculate_tensor_flow,
    get_component_inspection_details,
)


def build_playground_payload(
    model_key: str,
    summary: Dict[str, Any],
    artifacts: Dict[str, Any],
    walkthrough: Dict[str, Any],
    gate_result: Dict[str, Any],
    batch_size: int = 1,
    mode: str = "Beginner",
    selected_neuron: int = 25,
) -> Dict[str, Any]:
    """
    Packages 100% real trained PyTorch weights, scaler coefficients, 30-day
    historical weather inputs, 64-unit gate/state histories, and walkthrough
    stages into a JSON-serializable payload for the interactive HTML5/SVG
    Neural Network Playground canvas.
    """
    model = artifacts["models"][model_key]
    flow = calculate_tensor_flow(summary=summary, batch_size=batch_size)

    component_ids = [
        "weather_input",
        "lstm",
        "geography",
        "city_embedding",
        "concatenation",
        "fc",
        "output",
    ]
    component_details = {}
    for cid in component_ids:
        if cid == "geography" and not summary["has_geography"]:
            continue
        det = get_component_inspection_details(
            summary=summary,
            component_id=cid,
            batch_size=batch_size,
        )
        component_details[cid] = {
            "component_id": det["component_id"],
            "component_name": det["component_name"],
            "category": det["category"],
            "purpose": det["purpose"],
            "input_dimensions": str(det["input_dimensions"]),
            "output_dimensions": str(det["output_dimensions"]),
            "parameter_count": int(det["parameter_count"]),
            "beginner_explanation": det["beginner_explanation"],
            "technical_explanation": det["technical_explanation"],
            "math_operation": det["math_operation"],
            "weight_and_bias_shapes": {
                k: list(v) for k, v in det["weight_and_bias_shapes"].items()
            },
        }

    # Extract actual trained weights from the PyTorch model in read-only mode
    w_ih = np.round(
        model.lstm.weight_ih_l0.detach().cpu().numpy().astype(np.float64), 6
    ).tolist()
    w_hh = np.round(
        model.lstm.weight_hh_l0.detach().cpu().numpy().astype(np.float64), 6
    ).tolist()
    b_ih = np.round(
        model.lstm.bias_ih_l0.detach().cpu().numpy().astype(np.float64), 6
    ).tolist()
    b_hh = np.round(
        model.lstm.bias_hh_l0.detach().cpu().numpy().astype(np.float64), 6
    ).tolist()
    fc_weight = np.round(
        model.fc.weight.detach().cpu().numpy()[0].astype(np.float64), 6
    ).tolist()
    fc_bias = float(model.fc.bias.detach().cpu().numpy()[0])

    # Extract exact scaler parameters for live What-If perturbations & Celsius conversion
    if model_key == "baseline":
        t_scaler = artifacts["v1"]["target_scaler"]
        y_scaler = artifacts["v1"]["target_scaler"]
    else:
        t_scaler = artifacts["v2"]["temporal_scaler"]
        y_scaler = artifacts["v2"]["target_scaler"]

    temporal_scale = np.round(t_scaler.scale_.astype(np.float64), 8).tolist()
    temporal_min = np.round(t_scaler.min_.astype(np.float64), 8).tolist()
    temporal_data_min = np.round(t_scaler.data_min_.astype(np.float64), 4).tolist()
    temporal_data_max = np.round(t_scaler.data_max_.astype(np.float64), 4).tolist()
    target_scale = float(y_scaler.scale_[0])
    target_min = float(y_scaler.min_[0])

    # Extract 30-day preprocessed (physical) and scaled features
    feature_names = list(summary["temporal_features"])
    pre_df = walkthrough["stage2_preprocessed_features"]
    raw_features_30d = np.round(
        pre_df[feature_names].to_numpy(dtype=np.float64), 4
    ).tolist()
    scaled_features_30d = np.round(
        walkthrough["stage3_scaled_input"]["array"][0].astype(np.float64), 6
    ).tolist()

    # Extract 30x64 gate and state matrices
    gate_data = gate_result["gate_data"]
    gates_30x64 = {
        k: np.round(gate_data[k].astype(np.float64), 5).tolist()
        for k in [
            "input_gate",
            "forget_gate",
            "candidate_gate",
            "output_gate",
            "cell_state",
            "hidden_state",
        ]
    }

    # Geography and City Embedding vectors
    has_geo = bool(summary["has_geography"])
    geo_features = list(summary["geo_features"])
    if has_geo:
        geo_raw = [
            round(float(walkthrough["stage4_geography"]["raw_values"][f]), 4)
            for f in geo_features
        ]
        geo_scaled = [
            round(float(walkthrough["stage4_geography"]["scaled_values"][f]), 6)
            for f in geo_features
        ]
    else:
        geo_raw = []
        geo_scaled = []

    city_vec = np.round(
        walkthrough["stage5_city_embedding"]["vector"].astype(np.float64), 6
    ).tolist()

    stages_serializable = [
        {
            "stage_id": s.get("key", s["label"]),
            "label": s["label"],
            "shape": str(s["shape"]),
            "axis_meaning": s["axis_meaning"],
            "beginner_explanation": s["beginner_explanation"],
            "technical_explanation": s["technical_explanation"],
        }
        for s in flow["stages"]
    ]

    return {
        "model_key": model_key,
        "model_name": summary["model_name"],
        "architecture_class": summary["architecture_class"],
        "batch_size": int(batch_size),
        "mode": mode,
        "initial_neuron": int(selected_neuron),
        "city": walkthrough["city"],
        "city_id": int(walkthrough["city_id"]),
        "window_start_date": walkthrough["window_start_date"],
        "last_observation_date": walkthrough["last_observation_date"],
        "prediction_date": walkthrough["prediction_date"],
        "dates": list(walkthrough["dates"]),
        "lstm_input_size": int(summary["lstm_input_size"]),
        "lstm_hidden_size": int(summary["lstm_hidden_size"]),
        "sequence_length": int(summary["sequence_length"]),
        "has_geography": has_geo,
        "geo_dim": int(summary["geo_dim"]),
        "geo_features": geo_features,
        "geo_raw": geo_raw,
        "geo_scaled": geo_scaled,
        "city_vocab_size": int(summary["city_vocab_size"]),
        "city_embedding_dim": int(summary["city_embedding_dim"]),
        "city_vector": city_vec,
        "fc_input_size": int(summary["fc_input_size"]),
        "fc_output_size": int(summary["fc_output_size"]),
        "total_parameters": int(summary["trainable_parameters"]),
        "component_parameters": {
            k: int(v) for k, v in summary["component_parameters"].items()
        },
        "temporal_features": feature_names,
        "raw_features_30d": raw_features_30d,
        "scaled_features_30d": scaled_features_30d,
        "temporal_scale": temporal_scale,
        "temporal_min": temporal_min,
        "temporal_data_min": temporal_data_min,
        "temporal_data_max": temporal_data_max,
        "target_scale": target_scale,
        "target_min": target_min,
        "weights": {
            "W_ih": w_ih,
            "W_hh": w_hh,
            "b_ih": b_ih,
            "b_hh": b_hh,
            "fc_weight": fc_weight,
            "fc_bias": fc_bias,
        },
        "gates_30x64": gates_30x64,
        "baseline_scaled_prediction": float(
            walkthrough["stage9_dense_output"]["scaled_prediction"]
        ),
        "baseline_celsius_prediction": float(
            walkthrough["stage10_celsius_prediction"][
                "predicted_max_temperature_celsius"
            ]
        ),
        "actual_celsius": walkthrough["stage10_celsius_prediction"][
            "actual_max_temperature_celsius"
        ],
        "tensor_flow_stages": stages_serializable,
        "component_details": component_details,
    }


def build_interactive_playground_html(payload: Dict[str, Any]) -> str:
    """
    Generates a self-contained HTML5 / SVG / JS Interactive Neural Network Playground
    inspired by TensorFlow Playground and CNN/Transformer Explainer, driven by the
    actual trained PyTorch weights and 30-day weather sequence.
    """
    payload_json = json.dumps(payload)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
  :root {{
    --bg-main: #0B0F19;
    --bg-panel: #111827;
    --bg-card: #1E293B;
    --bg-card-hover: #26354D;
    --border-subtle: #334155;
    --border-active: #F59E0B;
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
    --accent-blue: #38BDF8;
    --accent-indigo: #818CF8;
    --accent-emerald: #34D399;
    --accent-amber: #FBBF24;
    --accent-rose: #FB7185;
    --accent-purple: #C084FC;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }}
  body {{
    background: var(--bg-main);
    color: var(--text-main);
    overflow-x: hidden;
    padding: 10px 12px;
    user-select: none;
  }}
  /* Top Transport & Control Bar */
  .transport-bar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    background: linear-gradient(180deg, #162033 0%, #0F172A 100%);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.45);
  }}
  .transport-left, .transport-center, .transport-right {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .btn {{
    background: #1E293B;
    color: #F8FAFC;
    border: 1px solid #475569;
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }}
  .btn:hover {{
    background: #334155;
    border-color: #94A3B8;
  }}
  .btn-play {{
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    border-color: #60A5FA;
    padding: 6px 14px;
    font-size: 13px;
  }}
  .btn-play:hover {{
    background: linear-gradient(135deg, #1D4ED8 0%, #4338CA 100%);
  }}
  .btn-play.playing {{
    background: linear-gradient(135deg, #D97706 0%, #B45309 100%);
    border-color: #FBBF24;
  }}
  .btn-reset {{
    background: #3F1D2E;
    border-color: #F43F5E;
    color: #FDA4AF;
    display: none;
  }}
  .btn-reset.visible {{
    display: inline-flex;
  }}
  .select-pill {{
    background: #0F172A;
    color: #E2E8F0;
    border: 1px solid #475569;
    border-radius: 7px;
    padding: 5px 8px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
  }}
  .day-badge {{
    background: #1E1B4B;
    border: 1px solid #6366F1;
    color: #E0E7FF;
    font-family: Consolas, monospace;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 10px;
    border-radius: 8px;
    min-width: 195px;
    text-align: center;
  }}
  /* 30-Day Interactive Timeline Strip */
  .timeline-strip-wrap {{
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 6px 12px 8px 12px;
    margin-bottom: 10px;
  }}
  .timeline-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 4px;
  }}
  .timeline-bars {{
    display: grid;
    grid-template-columns: repeat(30, 1fr);
    gap: 3px;
    height: 30px;
    align-items: end;
    cursor: pointer;
  }}
  .t-bar-col {{
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    position: relative;
    border-radius: 3px;
    padding: 1px;
    transition: background 0.12s;
  }}
  .t-bar-col:hover {{
    background: rgba(255,255,255,0.08);
  }}
  .t-bar-col.active {{
    background: rgba(245, 158, 11, 0.22);
    outline: 1px solid #F59E0B;
  }}
  .t-bar-fill {{
    width: 100%;
    border-radius: 2px;
    transition: height 0.15s ease;
  }}
  .t-bar-lbl {{
    font-size: 9px;
    color: #64748B;
    margin-top: 2px;
    font-family: Consolas, monospace;
  }}
  .t-bar-col.active .t-bar-lbl {{
    color: #FBBF24;
    font-weight: 700;
  }}

  /* Main Neural Network Playground Stage */
  .playground-stage {{
    position: relative;
    background: radial-gradient(circle at 50% 35%, #131C31 0%, #090D16 100%);
    border: 1px solid #26354D;
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 12px;
    min-height: 515px;
    overflow: hidden;
  }}
  #synapse-svg {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1;
  }}
  .synapse-line {{
    fill: none;
    stroke-linecap: round;
    transition: stroke-width 0.15s, stroke-opacity 0.15s;
  }}
  .synapse-pulse {{
    fill: none;
    stroke-dasharray: 6 10;
    animation: pulseFlow 1.1s linear infinite;
  }}
  @keyframes pulseFlow {{
    to {{
      stroke-dashoffset: -32;
    }}
  }}
  .network-columns {{
    position: relative;
    z-index: 2;
    display: grid;
    grid-template-columns: 1.22fr 1.15fr 1.38fr 1.15fr;
    gap: 18px;
    align-items: stretch;
  }}
  .net-col {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .col-header {{
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 10px;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .col-header:hover {{
    border-color: #60A5FA;
    background: #1E293B;
  }}
  .col-header.selected-comp {{
    border-color: #F59E0B;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.35);
    background: #1E1B4B;
  }}
  .col-title {{
    font-size: 12px;
    font-weight: 700;
    color: #F8FAFC;
  }}
  .col-shape {{
    font-family: Consolas, monospace;
    font-size: 10px;
    color: #93C5FD;
    background: rgba(30, 58, 138, 0.45);
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid #3B82F6;
  }}

  /* Column 1: Input Feature Neuron Cards */
  .input-list {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .input-node {{
    background: rgba(17, 24, 39, 0.92);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 8px;
    cursor: pointer;
    transition: all 0.14s ease;
    position: relative;
  }}
  .input-node:hover, .input-node.active-feature {{
    border-color: #38BDF8;
    background: rgba(30, 41, 59, 0.95);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.22);
  }}
  .input-node-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 3px;
  }}
  .feat-name {{
    font-size: 11px;
    font-weight: 700;
    color: #E2E8F0;
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .neuron-dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
    display: inline-block;
    border: 1.5px solid #fff;
  }}
  .feat-vals {{
    font-family: Consolas, monospace;
    font-size: 10.5px;
    color: #38BDF8;
  }}
  .input-slider-row {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .feat-slider {{
    width: 100%;
    height: 4px;
    accent-color: #38BDF8;
    cursor: ew-resize;
  }}
  .scaled-pill {{
    font-family: Consolas, monospace;
    font-size: 9.5px;
    color: #94A3B8;
    min-width: 42px;
    text-align: right;
  }}

  /* Column 2: 4 LSTM Gate Controllers & Cell Highway */
  .gate-stack {{
    display: flex;
    flex-direction: column;
    gap: 7px;
    flex: 1;
    justify-content: space-between;
  }}
  .gate-card {{
    background: rgba(17, 24, 39, 0.93);
    border: 1px solid #334155;
    border-radius: 9px;
    padding: 7px 10px;
    cursor: pointer;
    transition: all 0.15s ease;
    position: relative;
  }}
  .gate-card:hover, .gate-card.active-gate {{
    border-color: #A855F7;
    background: #1E1B4B;
    box-shadow: 0 0 12px rgba(168, 85, 247, 0.28);
  }}
  .gate-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }}
  .gate-title {{
    font-size: 11.5px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .gate-sym {{
    font-family: Consolas, monospace;
    font-size: 11px;
    padding: 1px 6px;
    border-radius: 4px;
    font-weight: 700;
  }}
  .gate-val-readout {{
    font-family: Consolas, monospace;
    font-size: 11.5px;
    font-weight: 700;
  }}
  .gauge-track {{
    width: 100%;
    height: 7px;
    background: #090D16;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid #1E293B;
    margin-bottom: 3px;
  }}
  .gauge-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.14s ease, background-color 0.14s ease;
  }}
  .gate-sub {{
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
    font-family: Consolas, monospace;
  }}

  /* Column 3: 64-Neuron 8x8 Interactive Grid + Bypass Track */
  .lattice-card {{
    background: rgba(15, 23, 42, 0.92);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 8px 10px;
  }}
  .lattice-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }}
  .lattice-sub {{
    font-size: 10.5px;
    color: #CBD5E1;
  }}
  .neuron-grid-64 {{
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    gap: 4px;
  }}
  .n-cell {{
    aspect-ratio: 1;
    border-radius: 5px;
    border: 1px solid rgba(255,255,255,0.14);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    font-family: Consolas, monospace;
    font-size: 9.5px;
    font-weight: 700;
    color: #F8FAFC;
    cursor: pointer;
    transition: transform 0.1s ease, border-color 0.1s ease;
    position: relative;
  }}
  .n-cell:hover {{
    transform: scale(1.16);
    z-index: 5;
    border-color: #FFFFFF;
    box-shadow: 0 0 8px rgba(255,255,255,0.5);
  }}
  .n-cell.selected-neuron {{
    border: 2px solid #FBBF24;
    box-shadow: 0 0 10px rgba(251, 191, 36, 0.75);
    z-index: 4;
  }}
  .n-cell.muted-neuron {{
    opacity: 0.28;
    border-color: #EF4444;
  }}
  .n-cell.muted-neuron::after {{
    content: "×";
    position: absolute;
    color: #EF4444;
    font-size: 16px;
    font-weight: 900;
  }}
  .bypass-box {{
    background: rgba(15, 23, 42, 0.92);
    border: 1px dashed #475569;
    border-radius: 9px;
    padding: 7px 9px;
  }}
  .bypass-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
  }}
  .bypass-pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }}
  .b-pill {{
    background: #1E293B;
    border: 1px solid #475569;
    border-radius: 5px;
    padding: 3px 6px;
    font-family: Consolas, monospace;
    font-size: 10px;
    cursor: pointer;
    transition: all 0.12s;
  }}
  .b-pill:hover {{
    border-color: #FBBF24;
  }}
  .b-pill.muted-pill {{
    opacity: 0.3;
    text-decoration: line-through;
    border-color: #EF4444;
  }}

  /* Column 4: Concatenation, Dense Readout & Next-Day Celsius Output */
  .readout-card {{
    background: rgba(17, 24, 39, 0.94);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 9px 11px;
  }}
  .concat-strip {{
    display: flex;
    height: 16px;
    width: 100%;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #334155;
    margin: 5px 0;
  }}
  .contrib-row {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin: 3px 0;
    font-family: Consolas, monospace;
  }}
  .pred-hero {{
    background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
    border: 1.5px solid #F59E0B;
    border-radius: 11px;
    padding: 10px 12px;
    text-align: center;
    position: relative;
  }}
  .pred-temp {{
    font-size: 26px;
    font-weight: 800;
    color: #FBBF24;
    font-family: Consolas, monospace;
    line-height: 1.15;
  }}
  .pred-delta-badge {{
    display: inline-block;
    font-family: Consolas, monospace;
    font-size: 11px;
    padding: 2px 7px;
    border-radius: 999px;
    margin-top: 4px;
  }}

  /* Bottom 3-Panel Synchronized Live Telemetry Dock */
  .telemetry-dock {{
    display: grid;
    grid-template-columns: 1.05fr 1.35fr 0.95fr;
    gap: 12px;
  }}
  .dock-panel {{
    background: var(--bg-panel);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 11px 13px;
  }}
  .dock-title {{
    font-size: 12.5px;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 6px;
  }}
  .eq-box {{
    background: #090D16;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 7px 9px;
    font-family: Consolas, monospace;
    font-size: 11px;
    line-height: 1.48;
    color: #E2E8F0;
    margin-bottom: 6px;
  }}
  .eq-hi {{
    color: #38BDF8;
    font-weight: 700;
  }}
  .eq-amber {{
    color: #FBBF24;
    font-weight: 700;
  }}
  .spark-grid-6 {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 7px;
  }}
  .mini-chart-card {{
    background: #090D16;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 5px 7px;
    cursor: pointer;
  }}
  .mini-chart-card:hover {{
    border-color: #475569;
  }}
  .mini-chart-head {{
    display: flex;
    justify-content: space-between;
    font-size: 10.5px;
    font-family: Consolas, monospace;
    margin-bottom: 2px;
  }}
  .stage-flow-list {{
    display: flex;
    flex-direction: column;
    gap: 5px;
    max-height: 225px;
    overflow-y: auto;
  }}
  .stage-flow-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #090D16;
    border: 1px solid #1E293B;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 11px;
    cursor: pointer;
  }}
  .stage-flow-item.active-stage {{
    border-color: #F59E0B;
    background: #1E1B4B;
  }}
</style>
</head>
<body>

<!-- 1. TOP PLAYGROUND TRANSPORT & SANDBOX BAR -->
<div class="transport-bar">
  <div class="transport-left">
    <button id="btn-play" class="btn btn-play" onclick="togglePlay()">▶ Play 30-Day Pass</button>
    <button class="btn" onclick="stepDay(-1)" title="Previous Day">⏮</button>
    <button class="btn" onclick="stepDay(1)" title="Next Day">⏭</button>
    <div id="day-readout" class="day-badge">Day 30 / 30 · 2025-10-19</div>
    <label style="font-size:11px;color:#94A3B8;">Speed:</label>
    <select id="speed-select" class="select-pill" onchange="updateSpeed()">
      <option value="700">0.5×</option>
      <option value="380" selected>1×</option>
      <option value="190">2×</option>
      <option value="95">4×</option>
    </select>
  </div>

  <div class="transport-center">
    <label style="font-size:11px;color:#94A3B8;">64-Neuron Grid Signal:</label>
    <select id="grid-signal-select" class="select-pill" onchange="renderAll()">
      <option value="hidden_state" selected>Hidden State (h_t)</option>
      <option value="cell_state">Cell Memory (c_t)</option>
      <option value="forget_gate">Forget Gate (f_t)</option>
      <option value="input_gate">Input Gate (i_t)</option>
      <option value="candidate_gate">Candidate Gate (g_t)</option>
      <option value="output_gate">Output Gate (o_t)</option>
      <option value="dense_impact">Day-30 Dense Impact (h_30 × W_fc)</option>
    </select>
    <button class="btn" onclick="toggleMuteSelectedNeuron()" id="btn-mute-neuron" title="Mute/Unmute the selected LSTM neuron in the Dense readout">
      🔇 Mute Unit #25
    </button>
  </div>

  <div class="transport-right">
    <button id="btn-reset" class="btn btn-reset" onclick="resetSandbox()">
      ↺ Reset What-If & Muted Units
    </button>
    <span style="font-size:11px;color:#94A3B8;" id="status-pill">
      🟢 Live PyTorch Weights Active
    </span>
  </div>
</div>

<!-- 2. INTERACTIVE 30-DAY WEATHER TIMELINE SCRUBBER -->
<div class="timeline-strip-wrap">
  <div class="timeline-header">
    <span>📅 <strong>Click or drag any day (Day 1 → Day 30)</strong> to scrub the LSTM forward pass through the 30-day window:</span>
    <span id="timeline-hover-info" style="font-family:Consolas,monospace;color:#38BDF8;"></span>
  </div>
  <div id="timeline-bars" class="timeline-bars"></div>
</div>

<!-- 3. INTERACTIVE NEURAL NETWORK PLAYGROUND STAGE -->
<div class="playground-stage" id="playground-stage">
  <svg id="synapse-svg"></svg>

  <div class="network-columns">
    <!-- COLUMN 1: WEATHER INPUT NEURONS + LIVE WHAT-IF SLIDERS -->
    <div class="net-col">
      <div class="col-header" id="hdr-weather_input" onclick="selectComponent('weather_input')">
        <div>
          <div class="col-title">1. Weather Inputs (x_t)</div>
          <div style="font-size:10px;color:#94A3B8;">Click node or drag What-If slider</div>
        </div>
        <span class="col-shape" id="shape-weather_input"></span>
      </div>
      <div id="input-neurons-container" class="input-list"></div>
    </div>

    <!-- COLUMN 2: INSIDE THE LSTM CELL (4 INTERACTIVE GATES & MEMORY HIGHWAY) -->
    <div class="net-col">
      <div class="col-header selected-comp" id="hdr-lstm" onclick="selectComponent('lstm')">
        <div>
          <div class="col-title">2. LSTM Gate Controllers</div>
          <div style="font-size:10px;color:#94A3B8;">Click any gate to inspect W_ih & W_hh</div>
        </div>
        <span class="col-shape" id="shape-lstm"></span>
      </div>

      <div class="gate-stack" id="gate-stack">
        <!-- Forget Gate -->
        <div class="gate-card" id="gate-card-forget_gate" onclick="selectGate('forget_gate')">
          <div class="gate-head">
            <span class="gate-title">
              <span class="gate-sym" style="background:#3B0764;color:#E9D5FF;border:1px solid #A855F7;">f_t</span>
              Forget Gate (σ)
            </span>
            <span class="gate-val-readout" id="val-forget_gate" style="color:#C084FC;">0.000</span>
          </div>
          <div class="gauge-track"><div class="gauge-fill" id="bar-forget_gate" style="background:#A855F7;width:50%;"></div></div>
          <div class="gate-sub">
            <span id="sub-forget_gate">Unit #25</span>
            <span id="mean-forget_gate">64-Avg: 0.00</span>
          </div>
        </div>

        <!-- Input Gate -->
        <div class="gate-card" id="gate-card-input_gate" onclick="selectGate('input_gate')">
          <div class="gate-head">
            <span class="gate-title">
              <span class="gate-sym" style="background:#0C4A6E;color:#BAE6FD;border:1px solid #38BDF8;">i_t</span>
              Input Gate (σ)
            </span>
            <span class="gate-val-readout" id="val-input_gate" style="color:#38BDF8;">0.000</span>
          </div>
          <div class="gauge-track"><div class="gauge-fill" id="bar-input_gate" style="background:#38BDF8;width:50%;"></div></div>
          <div class="gate-sub">
            <span id="sub-input_gate">Unit #25</span>
            <span id="mean-input_gate">64-Avg: 0.00</span>
          </div>
        </div>

        <!-- Candidate Gate -->
        <div class="gate-card" id="gate-card-candidate_gate" onclick="selectGate('candidate_gate')">
          <div class="gate-head">
            <span class="gate-title">
              <span class="gate-sym" style="background:#064E3B;color:#A7F3D0;border:1px solid #34D399;">g_t</span>
              Candidate (tanh)
            </span>
            <span class="gate-val-readout" id="val-candidate_gate" style="color:#34D399;">+0.000</span>
          </div>
          <div class="gauge-track"><div class="gauge-fill" id="bar-candidate_gate" style="background:#34D399;width:50%;"></div></div>
          <div class="gate-sub">
            <span id="sub-candidate_gate">Unit #25</span>
            <span id="mean-candidate_gate">64-Avg: +0.00</span>
          </div>
        </div>

        <!-- Cell State Highway -->
        <div class="gate-card" id="gate-card-cell_state" onclick="selectGate('cell_state')" style="border-style:dashed;">
          <div class="gate-head">
            <span class="gate-title">
              <span class="gate-sym" style="background:#451A03;color:#FDE68A;border:1px solid #F59E0B;">c_t</span>
              Cell Memory Highway
            </span>
            <span class="gate-val-readout" id="val-cell_state" style="color:#FBBF24;">+0.000</span>
          </div>
          <div class="gauge-track"><div class="gauge-fill" id="bar-cell_state" style="background:#F59E0B;width:50%;"></div></div>
          <div class="gate-sub">
            <span id="sub-cell_state">c_t = f_t·c_{{t-1}} + i_t·g_t</span>
            <span id="mean-cell_state">64-Avg: +0.00</span>
          </div>
        </div>

        <!-- Output Gate -->
        <div class="gate-card" id="gate-card-output_gate" onclick="selectGate('output_gate')">
          <div class="gate-head">
            <span class="gate-title">
              <span class="gate-sym" style="background:#4C0519;color:#FECDD3;border:1px solid #FB7185;">o_t</span>
              Output Gate (σ)
            </span>
            <span class="gate-val-readout" id="val-output_gate" style="color:#FB7185;">0.000</span>
          </div>
          <div class="gauge-track"><div class="gauge-fill" id="bar-output_gate" style="background:#FB7185;width:50%;"></div></div>
          <div class="gate-sub">
            <span id="sub-output_gate">h_t = o_t · tanh(c_t)</span>
            <span id="mean-output_gate">64-Avg: 0.00</span>
          </div>
        </div>
      </div>
    </div>

    <!-- COLUMN 3: 64-NEURON INTERACTIVE LATTICE + PARALLEL BYPASS TRACK -->
    <div class="net-col">
      <div class="lattice-card">
        <div class="lattice-top">
          <div>
            <span style="font-size:12px;font-weight:700;color:#F8FAFC;">3. 64 LSTM Neurons (8×8 Lattice)</span>
            <div class="lattice-sub">Click neuron to inspect · Double-click to Mute/Ablate</div>
          </div>
          <span class="col-shape" id="active-neuron-pill">Unit #25</span>
        </div>
        <div id="neuron-grid-64" class="neuron-grid-64"></div>
      </div>

      <!-- Geography Bypass Box -->
      <div class="bypass-box" id="box-geography" onclick="selectComponent('geography')">
        <div class="bypass-head">
          <span style="font-size:11px;font-weight:700;color:#34D399;">🗺️ Geography Bypass (Skips LSTM)</span>
          <button class="btn" style="padding:2px 6px;font-size:10px;" onclick="event.stopPropagation();toggleMuteGroup('geo')" id="btn-mute-geo">Mute Geo</button>
        </div>
        <div id="geo-pills" class="bypass-pills"></div>
      </div>

      <!-- City Embedding Bypass Box -->
      <div class="bypass-box" id="box-city_embedding" onclick="selectComponent('city_embedding')">
        <div class="bypass-head">
          <span style="font-size:11px;font-weight:700;color:#FBBF24;">🏙️ City Embedding (50 × 8 Lookup)</span>
          <button class="btn" style="padding:2px 6px;font-size:10px;" onclick="event.stopPropagation();toggleMuteGroup('city')" id="btn-mute-city">Mute City</button>
        </div>
        <div id="city-pills" class="bypass-pills"></div>
      </div>
    </div>

    <!-- COLUMN 4: CONCATENATION, DENSE READOUT & NEXT-DAY CELSIUS PREDICTION -->
    <div class="net-col">
      <div class="col-header" id="hdr-concatenation" onclick="selectComponent('concatenation')">
        <div>
          <div class="col-title">4. Concat & Dense Head</div>
          <div style="font-size:10px;color:#94A3B8;">Combines h_30 + Geo + City</div>
        </div>
        <span class="col-shape" id="shape-concatenation"></span>
      </div>

      <div class="readout-card" id="card-fc" onclick="selectComponent('fc')" style="cursor:pointer;">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-size:11.5px;font-weight:700;color:#C084FC;">Concatenated Vector Barcode</span>
          <span style="font-family:Consolas,monospace;font-size:10px;color:#94A3B8;" id="concat-len-lbl"></span>
        </div>
        <div id="concat-strip" class="concat-strip" title="64 LSTM units + Geography + 8 City Embedding units"></div>

        <div style="font-size:11.5px;font-weight:700;color:#E2E8F0;margin-top:6px;margin-bottom:3px;">
          Dense Layer (y_scaled = W_fc · z + b_fc)
        </div>
        <div class="contrib-row">
          <span style="color:#93C5FD;">• LSTM 64 Units:</span>
          <span id="contrib-lstm">+0.0000</span>
        </div>
        <div class="contrib-row" id="contrib-geo-row">
          <span style="color:#6EE7B7;">• Geography Bypass:</span>
          <span id="contrib-geo">+0.0000</span>
        </div>
        <div class="contrib-row">
          <span style="color:#FDE68A;">• City Embedding (8):</span>
          <span id="contrib-city">+0.0000</span>
        </div>
        <div class="contrib-row">
          <span style="color:#CBD5E1;">• Dense Bias (b_fc):</span>
          <span id="contrib-bias">+0.0000</span>
        </div>
        <div class="contrib-row" style="border-top:1px solid #334155;padding-top:4px;font-weight:700;color:#F8FAFC;">
          <span>Stage 9 Scaled Output:</span>
          <span id="val-scaled-out" style="color:#38BDF8;">0.00000</span>
        </div>
      </div>

      <div class="pred-hero" id="card-output" onclick="selectComponent('output')" style="cursor:pointer;">
        <div style="font-size:11px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;">
          Stage 10 · Next-Day Max Temp Forecast
        </div>
        <div style="font-size:11px;color:#CBD5E1;margin-bottom:4px;" id="pred-date-lbl"></div>
        <div class="pred-temp" id="val-celsius-out">0.00 °C</div>
        <div id="pred-delta-badge" class="pred-delta-badge" style="display:none;"></div>
        <div style="font-size:11px;color:#94A3B8;margin-top:6px;font-family:Consolas,monospace;" id="ground-truth-lbl"></div>
      </div>
    </div>
  </div>
</div>

<!-- 4. BOTTOM 3-PANEL SYNCHRONIZED LIVE TELEMETRY DOCK -->
<div class="telemetry-dock">
  <!-- PANEL A: LIVE DAY-t MATHEMATICAL EQUATIONS WITH PLUGGED-IN NUMBERS -->
  <div class="dock-panel">
    <div class="dock-title">
      <span id="eq-panel-title">🧮 Live Day-30 Gate & State Math — Unit #25</span>
      <span style="font-size:10.5px;color:#38BDF8;font-family:Consolas,monospace;" id="eq-day-tag">t = 30</span>
    </div>
    <div id="live-equations-box"></div>
  </div>

  <!-- PANEL B: 6-SIGNAL INTERACTIVE TIME-SERIES SPARK-GRID FOR SELECTED NEURON -->
  <div class="dock-panel">
    <div class="dock-title">
      <span id="chart-panel-title">📈 Unit #25 — 30-Day Gate & Memory Trajectories (Click chart to scrub day)</span>
      <span style="font-size:10.5px;color:#FBBF24;font-family:Consolas,monospace;">Dashed = Baseline | Solid = Live</span>
    </div>
    <div class="spark-grid-6" id="spark-grid-6"></div>
  </div>

  <!-- PANEL C: SELECTED ARCHITECTURE BLOCK & STAGE-BY-STAGE TENSOR FLOW -->
  <div class="dock-panel">
    <div class="dock-title">
      <span id="comp-panel-title">🔍 Layer & Tensor Flow Inspector</span>
      <span style="font-size:10.5px;color:#A855F7;font-family:Consolas,monospace;" id="comp-param-tag"></span>
    </div>
    <div id="comp-inspector-box" style="font-size:11.5px;color:#CBD5E1;line-height:1.45;margin-bottom:8px;"></div>
    <div style="font-size:11px;font-weight:700;color:#94A3B8;margin-bottom:4px;">
      Stage-by-Stage Tensor Shapes (Batch Size B = <span id="batch-lbl">1</span>):
    </div>
    <div id="stage-flow-list" class="stage-flow-list"></div>
  </div>
</div>

<script>
const P = {payload_json};

// Live Interactive Playground State
let state = {{
  dayIdx: P.sequence_length - 1, // 0..29 (Day 1..30)
  selectedNeuron: (P.initial_neuron || 25) - 1, // 0..63
  selectedComponent: "lstm",
  selectedGate: "forget_gate",
  selectedFeatureIdx: 0,
  playing: false,
  timer: null,
  speedMs: 380,
  // Cloned modifiable scaled inputs for live What-If experiments
  rawInputs: P.raw_features_30d.map(row => [...row]),
  scaledInputs: P.scaled_features_30d.map(row => [...row]),
  mutedNeurons: new Set(),
  mutedGeo: new Set(),
  mutedCity: new Set(),
  // Live forward pass results
  liveGates: null,
  livePrediction: null,
}};

function sigmoid(x) {{
  if (x < -40) return 0.0;
  if (x > 40) return 1.0;
  return 1.0 / (1.0 + Math.exp(-x));
}}

function valueToColor(val, isZeroToOne) {{
  if (isZeroToOne) {{
    // 0 -> deep slate/blue, 0.5 -> purple, 1.0 -> bright amber/cyan
    const t = Math.max(0, Math.min(1, val));
    const r = Math.round(20 + t * 225);
    const g = Math.round(35 + t * 145);
    const b = Math.round(90 + (1 - Math.abs(t - 0.5) * 2) * 140);
    return `rgb(${{r}},${{g}},${{b}})`;
  }} else {{
    // Diverging [-1, +1]: negative -> cyan/blue, 0 -> dark slate, positive -> coral/amber
    const v = Math.max(-1, Math.min(1, val));
    if (v >= 0) {{
      const r = Math.round(30 + v * 215);
      const g = Math.round(41 + v * 125);
      const b = Math.round(59 - v * 35);
      return `rgb(${{r}},${{g}},${{b}})`;
    }} else {{
      const a = -v;
      const r = Math.round(30 - a * 18);
      const g = Math.round(41 + a * 145);
      const b = Math.round(59 + a * 185);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}
  }}
}}

// Exact 30-Day PyTorch LSTM + Dense Forward Pass in Browser
function runLiveForwardPass() {{
  const T = P.sequence_length;
  const H = P.lstm_hidden_size;
  const F = P.lstm_input_size;
  const W_ih = P.weights.W_ih;
  const W_hh = P.weights.W_hh;
  const b_ih = P.weights.b_ih;
  const b_hh = P.weights.b_hh;

  let h_prev = new Float64Array(H);
  let c_prev = new Float64Array(H);

  const hist = {{
    input_gate: [],
    forget_gate: [],
    candidate_gate: [],
    output_gate: [],
    cell_state: [],
    hidden_state: [],
    pre_i: [],
    pre_f: [],
    pre_g: [],
    pre_o: [],
  }};

  for (let t = 0; t < T; t++) {{
    const x_t = state.scaledInputs[t];
    const z = new Float64Array(4 * H);

    for (let r = 0; r < 4 * H; r++) {{
      let sum = b_ih[r] + b_hh[r];
      const w_in_row = W_ih[r];
      for (let f = 0; f < F; f++) {{
        sum += w_in_row[f] * x_t[f];
      }}
      const w_hh_row = W_hh[r];
      for (let j = 0; j < H; j++) {{
        sum += w_hh_row[j] * h_prev[j];
      }}
      z[r] = sum;
    }}

    const i_t = new Float64Array(H);
    const f_t = new Float64Array(H);
    const g_t = new Float64Array(H);
    const o_t = new Float64Array(H);
    const c_t = new Float64Array(H);
    const h_t = new Float64Array(H);

    for (let k = 0; k < H; k++) {{
      i_t[k] = sigmoid(z[k]);
      f_t[k] = sigmoid(z[H + k]);
      g_t[k] = Math.tanh(z[2 * H + k]);
      o_t[k] = sigmoid(z[3 * H + k]);
      c_t[k] = f_t[k] * c_prev[k] + i_t[k] * g_t[k];
      h_t[k] = o_t[k] * Math.tanh(c_t[k]);
    }}

    hist.input_gate.push(i_t);
    hist.forget_gate.push(f_t);
    hist.candidate_gate.push(g_t);
    hist.output_gate.push(o_t);
    hist.cell_state.push(c_t);
    hist.hidden_state.push(h_t);
    hist.pre_i.push(z.slice(0, H));
    hist.pre_f.push(z.slice(H, 2 * H));
    hist.pre_g.push(z.slice(2 * H, 3 * H));
    hist.pre_o.push(z.slice(3 * H, 4 * H));

    h_prev = h_t;
    c_prev = c_t;
  }}

  // Stage 8 & 9: Concatenation + Dense Layer with Neuron/Pathway Ablation support
  const final_h = hist.hidden_state[T - 1];
  const fc_w = P.weights.fc_weight;
  const fc_b = P.weights.fc_bias;

  let lstm_sum = 0.0;
  const perNeuronImpact = new Float64Array(H);
  for (let k = 0; k < H; k++) {{
    const activeVal = state.mutedNeurons.has(k) ? 0.0 : final_h[k];
    const prod = activeVal * fc_w[k];
    perNeuronImpact[k] = prod;
    lstm_sum += prod;
  }}

  let geo_sum = 0.0;
  const geoDim = P.geo_dim;
  for (let g = 0; g < geoDim; g++) {{
    const activeGeo = state.mutedGeo.has(g) ? 0.0 : P.geo_scaled[g];
    geo_sum += activeGeo * fc_w[H + g];
  }}

  let city_sum = 0.0;
  for (let c = 0; c < P.city_embedding_dim; c++) {{
    const activeCity = state.mutedCity.has(c) ? 0.0 : P.city_vector[c];
    city_sum += activeCity * fc_w[H + geoDim + c];
  }}

  const y_scaled = lstm_sum + geo_sum + city_sum + fc_b;
  const y_celsius = (y_scaled - P.target_min) / P.target_scale;

  state.liveGates = hist;
  state.livePrediction = {{
    lstm_sum,
    geo_sum,
    city_sum,
    fc_bias: fc_b,
    y_scaled,
    y_celsius,
    perNeuronImpact,
  }};
}}

function isModified() {{
  if (state.mutedNeurons.size > 0 || state.mutedGeo.size > 0 || state.mutedCity.size > 0) return true;
  for (let t = 0; t < P.sequence_length; t++) {{
    for (let f = 0; f < P.lstm_input_size; f++) {{
      if (Math.abs(state.scaledInputs[t][f] - P.scaled_features_30d[t][f]) > 1e-5) return true;
    }}
  }}
  return false;
}}

function resetSandbox() {{
  state.rawInputs = P.raw_features_30d.map(row => [...row]);
  state.scaledInputs = P.scaled_features_30d.map(row => [...row]);
  state.mutedNeurons.clear();
  state.mutedGeo.clear();
  state.mutedCity.clear();
  runLiveForwardPass();
  renderAll();
}}

function buildTimelineStrip() {{
  const container = document.getElementById("timeline-bars");
  container.innerHTML = "";
  // Use temperature_2m_max (index 0) for bar heights
  const temps = P.raw_features_30d.map(r => r[0]);
  const tMin = Math.min(...temps);
  const tMax = Math.max(...temps);
  const span = Math.max(1.0, tMax - tMin);

  for (let t = 0; t < P.sequence_length; t++) {{
    const col = document.createElement("div");
    col.className = "t-bar-col" + (t === state.dayIdx ? " active" : "");
    col.onclick = () => {{
      state.dayIdx = t;
      renderAll();
    }};
    col.onmouseenter = () => {{
      document.getElementById("timeline-hover-info").textContent =
        `Day ${{t + 1}} (${{P.dates[t]}}) · Max Temp: ${{temps[t].toFixed(1)}} °C`;
    }};

    const pct = Math.round(22 + ((temps[t] - tMin) / span) * 75);
    const fill = document.createElement("div");
    fill.className = "t-bar-fill";
    fill.style.height = `${{pct}}%`;
    fill.style.background = t === state.dayIdx
      ? "#F59E0B"
      : (t < state.dayIdx ? "#38BDF8" : "#334155");

    const lbl = document.createElement("div");
    lbl.className = "t-bar-lbl";
    lbl.textContent = (t + 1);

    col.appendChild(fill);
    col.appendChild(lbl);
    container.appendChild(col);
  }}
}}

function renderInputNeurons() {{
  const container = document.getElementById("input-neurons-container");
  container.innerHTML = "";
  const t = state.dayIdx;

  P.temporal_features.forEach((feat, fIdx) => {{
    const rawVal = state.rawInputs[t][fIdx];
    const scaledVal = state.scaledInputs[t][fIdx];
    const dMin = P.temporal_data_min[fIdx];
    const dMax = P.temporal_data_max[fIdx];
    const step = ((dMax - dMin) / 200) || 0.01;

    const card = document.createElement("div");
    card.className = "input-node" + (state.selectedFeatureIdx === fIdx ? " active-feature" : "");
    card.id = `input-node-${{fIdx}}`;
    card.onclick = () => {{
      state.selectedFeatureIdx = fIdx;
      state.selectedComponent = "weather_input";
      renderAll();
    }};

    const dotColor = valueToColor(scaledVal, true);
    const shortName = feat
      .replace("temperature_2m_", "temp_")
      .replace("relative_humidity_2m_mean", "humidity")
      .replace("wind_speed_10m_max", "wind_max")
      .replace("precipitation_log", "precip_log");

    card.innerHTML = `
      <div class="input-node-top">
        <span class="feat-name">
          <span class="neuron-dot" style="background:${{dotColor}};"></span>
          ${{shortName}}
        </span>
        <span class="feat-vals">${{rawVal.toFixed(2)}}</span>
      </div>
      <div class="input-slider-row" onclick="event.stopPropagation();">
        <input type="range" class="feat-slider"
          min="${{dMin}}" max="${{dMax}}" step="${{step}}" value="${{rawVal}}"
          oninput="onFeatureSliderChange(${{fIdx}}, parseFloat(this.value))"
          title="Drag to perturb ${{feat}} on Day ${{t + 1}}" />
        <span class="scaled-pill">${{scaledVal.toFixed(3)}}</span>
      </div>
    `;
    container.appendChild(card);
  }});
}}

function onFeatureSliderChange(fIdx, newRawVal) {{
  const t = state.dayIdx;
  state.rawInputs[t][fIdx] = newRawVal;
  const newScaled = newRawVal * P.temporal_scale[fIdx] + P.temporal_min[fIdx];
  state.scaledInputs[t][fIdx] = newScaled;
  runLiveForwardPass();
  renderAll();
}}

function renderGatesAndLattice() {{
  const t = state.dayIdx;
  const n = state.selectedNeuron;
  const g = state.liveGates;

  const gateSpecs = [
    {{ key: "forget_gate", zeroToOne: true }},
    {{ key: "input_gate", zeroToOne: true }},
    {{ key: "candidate_gate", zeroToOne: false }},
    {{ key: "cell_state", zeroToOne: false }},
    {{ key: "output_gate", zeroToOne: true }},
  ];

  gateSpecs.forEach(spec => {{
    const arr = g[spec.key][t];
    const val = arr[n];
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    const mean = sum / arr.length;

    const valEl = document.getElementById(`val-${{spec.key}}`);
    valEl.textContent = (spec.zeroToOne ? val.toFixed(4) : (val >= 0 ? "+" : "") + val.toFixed(4));

    const barEl = document.getElementById(`bar-${{spec.key}}`);
    const pct = spec.zeroToOne
      ? Math.max(0, Math.min(100, val * 100))
      : Math.max(0, Math.min(100, ((Math.tanh(val) + 1) / 2) * 100));
    barEl.style.width = `${{pct.toFixed(1)}}%`;

    document.getElementById(`sub-${{spec.key}}`).textContent = `Unit #${{n + 1}} (Day ${{t + 1}})`;
    document.getElementById(`mean-${{spec.key}}`).textContent =
      `64-Avg: ${{spec.zeroToOne ? mean.toFixed(3) : (mean >= 0 ? "+" : "") + mean.toFixed(3)}}`;

    const cardEl = document.getElementById(`gate-card-${{spec.key}}`);
    if (state.selectedGate === spec.key) cardEl.classList.add("active-gate");
    else cardEl.classList.remove("active-gate");
  }});

  // Render 64-Neuron 8x8 Grid
  const gridContainer = document.getElementById("neuron-grid-64");
  gridContainer.innerHTML = "";
  const signalMode = document.getElementById("grid-signal-select").value;
  const zeroToOneSignals = new Set(["forget_gate", "input_gate", "output_gate"]);
  const is01 = zeroToOneSignals.has(signalMode);

  for (let k = 0; k < P.lstm_hidden_size; k++) {{
    let v = 0;
    if (signalMode === "dense_impact") {{
      v = state.livePrediction.perNeuronImpact[k] * 8.0; // scaled for visual contrast
    }} else {{
      v = g[signalMode][t][k];
    }}

    const cell = document.createElement("div");
    cell.className = "n-cell"
      + (k === n ? " selected-neuron" : "")
      + (state.mutedNeurons.has(k) ? " muted-neuron" : "");
    cell.id = `neuron-cell-${{k}}`;
    cell.style.background = valueToColor(v, is01);
    cell.textContent = k + 1;
    cell.title = `Unit #${{k + 1}} | Day ${{t + 1}}\n`
      + `h_t: ${{g.hidden_state[t][k].toFixed(4)}} | c_t: ${{g.cell_state[t][k].toFixed(4)}}\n`
      + `f_t: ${{g.forget_gate[t][k].toFixed(3)}} | i_t: ${{g.input_gate[t][k].toFixed(3)}}\n`
      + `Click to inspect · Double-click to Mute/Unmute`;

    cell.onclick = () => {{
      state.selectedNeuron = k;
      renderAll();
    }};
    cell.ondblclick = (e) => {{
      e.stopPropagation();
      if (state.mutedNeurons.has(k)) state.mutedNeurons.delete(k);
      else state.mutedNeurons.add(k);
      runLiveForwardPass();
      renderAll();
    }};
    gridContainer.appendChild(cell);
  }}

  document.getElementById("active-neuron-pill").textContent =
    `Unit #${{n + 1}} ${{state.mutedNeurons.has(n) ? "(MUTED)" : ""}}`;
  document.getElementById("btn-mute-neuron").textContent =
    state.mutedNeurons.has(n) ? `🔊 Unmute Unit #${{n + 1}}` : `🔇 Mute Unit #${{n + 1}}`;
}}

function renderBypassAndReadout() {{
  // Geography Bypass Pills
  const geoBox = document.getElementById("box-geography");
  const geoPills = document.getElementById("geo-pills");
  geoPills.innerHTML = "";

  if (!P.has_geography) {{
    document.getElementById("btn-mute-geo").style.display = "none";
    geoPills.innerHTML = `<span style="font-size:10.5px;color:#64748B;">Not used in Baseline V1 (0 geo features)</span>`;
    document.getElementById("contrib-geo-row").style.display = "none";
  }} else {{
    document.getElementById("btn-mute-geo").style.display = "inline-flex";
    document.getElementById("contrib-geo-row").style.display = "flex";
    P.geo_features.forEach((gf, gIdx) => {{
      const pill = document.createElement("span");
      pill.className = "b-pill" + (state.mutedGeo.has(gIdx) ? " muted-pill" : "");
      pill.textContent = `${{gf}}: ${{P.geo_raw[gIdx]}} (${{P.geo_scaled[gIdx] >= 0 ? "+" : ""}}${{P.geo_scaled[gIdx].toFixed(2)}})`;
      pill.title = "Click to Mute/Unmute this geography feature in the Dense readout";
      pill.onclick = (e) => {{
        e.stopPropagation();
        if (state.mutedGeo.has(gIdx)) state.mutedGeo.delete(gIdx);
        else state.mutedGeo.add(gIdx);
        runLiveForwardPass();
        renderAll();
      }};
      geoPills.appendChild(pill);
    }});
  }}

  // City Embedding Pills
  const cityPills = document.getElementById("city-pills");
  cityPills.innerHTML = "";
  P.city_vector.forEach((cv, cIdx) => {{
    const pill = document.createElement("span");
    pill.className = "b-pill" + (state.mutedCity.has(cIdx) ? " muted-pill" : "");
    pill.textContent = `E[${{cIdx}}]: ${{cv >= 0 ? "+" : ""}}${{cv.toFixed(2)}}`;
    pill.title = `City Embedding dim ${{cIdx}} for ${{P.city}} (city_id=${{P.city_id}}). Click to Mute/Unmute.`;
    pill.onclick = (e) => {{
      e.stopPropagation();
      if (state.mutedCity.has(cIdx)) state.mutedCity.delete(cIdx);
      else state.mutedCity.add(cIdx);
      runLiveForwardPass();
      renderAll();
    }};
    cityPills.appendChild(pill);
  }});

  // Concatenated Barcode Strip
  const concatStrip = document.getElementById("concat-strip");
  concatStrip.innerHTML = "";
  const final_h = state.liveGates.hidden_state[P.sequence_length - 1];
  for (let k = 0; k < P.lstm_hidden_size; k++) {{
    const seg = document.createElement("div");
    seg.style.flex = "1";
    seg.style.background = state.mutedNeurons.has(k) ? "#111827" : valueToColor(final_h[k], false);
    concatStrip.appendChild(seg);
  }}
  for (let g = 0; g < P.geo_dim; g++) {{
    const seg = document.createElement("div");
    seg.style.flex = "1.2";
    seg.style.background = state.mutedGeo.has(g) ? "#111827" : "#10B981";
    concatStrip.appendChild(seg);
  }}
  for (let c = 0; c < P.city_embedding_dim; c++) {{
    const seg = document.createElement("div");
    seg.style.flex = "1.2";
    seg.style.background = state.mutedCity.has(c) ? "#111827" : "#F59E0B";
    concatStrip.appendChild(seg);
  }}

  // Readout numbers
  const pred = state.livePrediction;
  document.getElementById("concat-len-lbl").textContent = `${{P.fc_input_size}} dims`;
  document.getElementById("contrib-lstm").textContent = (pred.lstm_sum >= 0 ? "+" : "") + pred.lstm_sum.toFixed(5);
  document.getElementById("contrib-geo").textContent = (pred.geo_sum >= 0 ? "+" : "") + pred.geo_sum.toFixed(5);
  document.getElementById("contrib-city").textContent = (pred.city_sum >= 0 ? "+" : "") + pred.city_sum.toFixed(5);
  document.getElementById("contrib-bias").textContent = (pred.fc_bias >= 0 ? "+" : "") + pred.fc_bias.toFixed(5);
  document.getElementById("val-scaled-out").textContent = pred.y_scaled.toFixed(5);

  document.getElementById("pred-date-lbl").textContent =
    `${{P.city}} · Target Date: ${{P.prediction_date}}`;
  document.getElementById("val-celsius-out").textContent =
    `${{pred.y_celsius.toFixed(2)}} °C`;

  const deltaBadge = document.getElementById("pred-delta-badge");
  const modified = isModified();
  const resetBtn = document.getElementById("btn-reset");
  if (modified) {{
    resetBtn.classList.add("visible");
    const diff = pred.y_celsius - P.baseline_celsius_prediction;
    deltaBadge.style.display = "inline-block";
    deltaBadge.style.background = diff >= 0 ? "#451A03" : "#0C4A6E";
    deltaBadge.style.border = diff >= 0 ? "1px solid #F59E0B" : "1px solid #38BDF8";
    deltaBadge.style.color = diff >= 0 ? "#FDE68A" : "#BAE6FD";
    deltaBadge.textContent = `What-If Shift: ${{diff >= 0 ? "+" : ""}}${{diff.toFixed(2)}} °C (vs ${{P.baseline_celsius_prediction.toFixed(2)}} °C)`;
    document.getElementById("status-pill").textContent = "⚡ What-If / Ablation Sandbox Active";
    document.getElementById("status-pill").style.color = "#FBBF24";
  }} else {{
    resetBtn.classList.remove("visible");
    deltaBadge.style.display = "none";
    document.getElementById("status-pill").textContent = "🟢 Live PyTorch Weights Active";
    document.getElementById("status-pill").style.color = "#34D399";
  }}

  const gtLbl = document.getElementById("ground-truth-lbl");
  if (P.actual_celsius !== null && P.actual_celsius !== undefined) {{
    const err = pred.y_celsius - P.actual_celsius;
    gtLbl.textContent = `Observed: ${{P.actual_celsius.toFixed(2)}} °C (Error: ${{err >= 0 ? "+" : ""}}${{err.toFixed(2)}} °C)`;
  }} else {{
    gtLbl.textContent = `Observed Ground Truth: N/A`;
  }}
}}

function renderLiveEquationsAndCharts() {{
  const t = state.dayIdx;
  const n = state.selectedNeuron;
  const g = state.liveGates;

  const i_val = g.input_gate[t][n];
  const f_val = g.forget_gate[t][n];
  const g_val = g.candidate_gate[t][n];
  const o_val = g.output_gate[t][n];
  const c_val = g.cell_state[t][n];
  const h_val = g.hidden_state[t][n];
  const c_prev = t > 0 ? g.cell_state[t - 1][n] : 0.0;
  const fc_w = P.weights.fc_weight[n];
  const h_30 = g.hidden_state[P.sequence_length - 1][n];

  document.getElementById("eq-panel-title").textContent =
    `🧮 Live Day-${{t + 1}} Gate & Memory Math — Unit #${{n + 1}}`;
  document.getElementById("eq-day-tag").textContent =
    `Day ${{t + 1}} (${{P.dates[t]}})`;

  const eqBox = document.getElementById("live-equations-box");
  eqBox.innerHTML = `
    <div class="eq-box">
      <div><strong>1. Forget Gate (f_t):</strong> Retains <span class="eq-amber">${{(f_val * 100).toFixed(1)}}%</span> of yesterday's cell memory</div>
      <div>f_${{t+1}}[${{n+1}}] = σ(z_f = ${{(g.pre_f[t][n]).toFixed(3)}}) = <span class="eq-hi">${{f_val.toFixed(4)}}</span></div>
    </div>
    <div class="eq-box">
      <div><strong>2. Input & Candidate Gates (i_t × g_t):</strong> New weather write</div>
      <div>i_${{t+1}}[${{n+1}}] = σ(${{(g.pre_i[t][n]).toFixed(3)}}) = <span class="eq-hi">${{i_val.toFixed(4)}}</span> | g_${{t+1}}[${{n+1}}] = tanh(${{(g.pre_g[t][n]).toFixed(3)}}) = <span class="eq-hi">${{(g_val >= 0 ? "+" : "") + g_val.toFixed(4)}}</span></div>
    </div>
    <div class="eq-box">
      <div><strong>3. Cell State Update (c_t = f_t·c_{{t-1}} + i_t·g_t):</strong></div>
      <div>c_${{t+1}}[${{n+1}}] = (${{f_val.toFixed(3)}} × ${{(c_prev >= 0 ? "+" : "") + c_prev.toFixed(3)}}) + (${{i_val.toFixed(3)}} × ${{(g_val >= 0 ? "+" : "") + g_val.toFixed(3)}}) = <span class="eq-amber">${{(c_val >= 0 ? "+" : "") + c_val.toFixed(4)}}</span></div>
    </div>
    <div class="eq-box">
      <div><strong>4. Output Gate & Hidden State (h_t = o_t · tanh(c_t)):</strong></div>
      <div>h_${{t+1}}[${{n+1}}] = ${{o_val.toFixed(3)}} × tanh(${{c_val.toFixed(3)}}) = <span class="eq-amber">${{(h_val >= 0 ? "+" : "") + h_val.toFixed(4)}}</span></div>
      <div style="color:#94A3B8;margin-top:2px;">Day-30 Dense Readout Weight W_fc[${{n+1}}] = ${{fc_w >= 0 ? "+" : ""}}${{fc_w.toFixed(4)}} → Impact = ${{(h_30 * fc_w >= 0 ? "+" : "") + (h_30 * fc_w).toFixed(5)}}</div>
    </div>
  `;

  // Render 6-Signal Interactive Spark-Grid for Selected Neuron #n
  document.getElementById("chart-panel-title").textContent =
    `📈 Unit #${{n + 1}} — 30-Day Gate & Memory Curves (Click any point to jump day)`;
  const sparkGrid = document.getElementById("spark-grid-6");
  sparkGrid.innerHTML = "";

  const chartSpecs = [
    {{ key: "forget_gate", label: "Forget Gate (f_t)", color: "#C084FC", min: 0, max: 1 }},
    {{ key: "input_gate", label: "Input Gate (i_t)", color: "#38BDF8", min: 0, max: 1 }},
    {{ key: "candidate_gate", label: "Candidate (g_t)", color: "#34D399", min: -1, max: 1 }},
    {{ key: "output_gate", label: "Output Gate (o_t)", color: "#FB7185", min: 0, max: 1 }},
    {{ key: "cell_state", label: "Cell State (c_t)", color: "#FBBF24", min: null, max: null }},
    {{ key: "hidden_state", label: "Hidden State (h_t)", color: "#818CF8", min: -1, max: 1 }},
  ];

  chartSpecs.forEach(spec => {{
    const liveSeries = [];
    const baseSeries = [];
    for (let d = 0; d < P.sequence_length; d++) {{
      liveSeries.push(g[spec.key][d][n]);
      baseSeries.push(P.gates_30x64[spec.key][d][n]);
    }}

    let cMin = spec.min !== null ? spec.min : Math.min(...liveSeries, ...baseSeries, -0.5);
    let cMax = spec.max !== null ? spec.max : Math.max(...liveSeries, ...baseSeries, 0.5);
    if (Math.abs(cMax - cMin) < 1e-4) {{ cMax += 0.5; cMin -= 0.5; }}

    const W = 210, H = 52;
    const toX = (d) => 6 + (d / (P.sequence_length - 1)) * (W - 12);
    const toY = (v) => H - 6 - ((v - cMin) / (cMax - cMin)) * (H - 12);

    const basePts = baseSeries.map((v, d) => `${{toX(d).toFixed(1)}},${{toY(v).toFixed(1)}}`).join(" ");
    const livePts = liveSeries.map((v, d) => `${{toX(d).toFixed(1)}},${{toY(v).toFixed(1)}}`).join(" ");
    const curX = toX(t).toFixed(1);
    const curY = toY(liveSeries[t]).toFixed(1);
    const curVal = liveSeries[t];

    const card = document.createElement("div");
    card.className = "mini-chart-card";
    card.onclick = (e) => {{
      const rect = card.querySelector("svg").getBoundingClientRect();
      const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      state.dayIdx = Math.round(relX * (P.sequence_length - 1));
      renderAll();
    }};

    card.innerHTML = `
      <div class="mini-chart-head">
        <span style="color:${{spec.color}};font-weight:700;">${{spec.label}}</span>
        <span style="color:#F8FAFC;">${{curVal >= 0 && spec.min !== 0 ? "+" : ""}}${{curVal.toFixed(4)}}</span>
      </div>
      <svg viewBox="0 0 ${{W}} ${{H}}" width="100%" height="${{H}}">
        <line x1="${{curX}}" y1="2" x2="${{curX}}" y2="${{H - 2}}" stroke="#F59E0B" stroke-width="1.2" stroke-dasharray="2 2" />
        <polyline fill="none" stroke="#64748B" stroke-width="1.2" stroke-dasharray="3 2" points="${{basePts}}" />
        <polyline fill="none" stroke="${{spec.color}}" stroke-width="2" points="${{livePts}}" />
        <circle cx="${{curX}}" cy="${{curY}}" r="3.5" fill="#FBBF24" stroke="#090D16" stroke-width="1.2" />
      </svg>
    `;
    sparkGrid.appendChild(card);
  }});
}}

function renderComponentAndTensorDock() {{
  // Highlight selected component header
  ["weather_input", "lstm", "concatenation", "fc", "output"].forEach(cid => {{
    const el = document.getElementById(`hdr-${{cid}}`) || document.getElementById(`card-${{cid}}`);
    if (el) {{
      if (state.selectedComponent === cid) el.classList.add("selected-comp");
      else el.classList.remove("selected-comp");
    }}
  }});

  const det = P.component_details[state.selectedComponent] || P.component_details["lstm"];
  document.getElementById("comp-panel-title").textContent = `🔍 ${{det.component_name}}`;
  document.getElementById("comp-param-tag").textContent = `${{det.parameter_count.toLocaleString()}} params`;

  const expl = P.mode === "Beginner" ? det.beginner_explanation : det.technical_explanation;
  document.getElementById("comp-inspector-box").innerHTML = `
    <div style="color:#F8FAFC;font-weight:600;margin-bottom:3px;">${{det.purpose}}</div>
    <div style="color:#94A3B8;font-size:11px;">${{expl}}</div>
  `;

  document.getElementById("batch-lbl").textContent = P.batch_size;
  const flowList = document.getElementById("stage-flow-list");
  flowList.innerHTML = "";
  P.tensor_flow_stages.forEach(st => {{
    const row = document.createElement("div");
    row.className = "stage-flow-item";
    row.innerHTML = `
      <span style="color:#E2E8F0;font-weight:600;">${{st.label}}</span>
      <span class="col-shape">${{st.shape}}</span>
    `;
    row.title = P.mode === "Beginner" ? st.beginner_explanation : st.technical_explanation;
    flowList.appendChild(row);
  }});
}}

function drawSynapses() {{
  const svg = document.getElementById("synapse-svg");
  const stageRect = document.getElementById("playground-stage").getBoundingClientRect();
  svg.setAttribute("viewBox", `0 0 ${{stageRect.width}} ${{stageRect.height}}`);
  svg.innerHTML = "";

  const t = state.dayIdx;
  const n = state.selectedNeuron;

  function anchorRight(el) {{
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {{ x: r.right - stageRect.left, y: r.top + r.height / 2 - stageRect.top }};
  }}
  function anchorLeft(el) {{
    if (!el) return null;
    const r = el.getBoundingClientRect();
    return {{ x: r.left - stageRect.left, y: r.top + r.height / 2 - stageRect.top }};
  }}
  function makeCurve(p1, p2, color, width, opacity, animated) {{
    if (!p1 || !p2) return;
    const dx = Math.abs(p2.x - p1.x) * 0.48;
    const d = `M ${{p1.x.toFixed(1)}} ${{p1.y.toFixed(1)}} C ${{(p1.x + dx).toFixed(1)}} ${{p1.y.toFixed(1)}}, ${{(p2.x - dx).toFixed(1)}} ${{p2.y.toFixed(1)}}, ${{p2.x.toFixed(1)}} ${{p2.y.toFixed(1)}}`;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("class", "synapse-line" + (animated ? " synapse-pulse" : ""));
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", width.toFixed(2));
    path.setAttribute("stroke-opacity", opacity.toFixed(2));
    svg.appendChild(path);
  }}

  // 1. Synapses from Input Features -> 4 Gates (weighted by W_ih for selectedNeuron)
  const gateIds = [
    {{ id: "gate-card-input_gate", offset: 0 }},
    {{ id: "gate-card-forget_gate", offset: 64 }},
    {{ id: "gate-card-candidate_gate", offset: 128 }},
    {{ id: "gate-card-output_gate", offset: 192 }},
  ];
  for (let f = 0; f < P.lstm_input_size; f++) {{
    const src = anchorRight(document.getElementById(`input-node-${{f}}`));
    const xVal = state.scaledInputs[t][f];
    gateIds.forEach(gSpec => {{
      const dst = anchorLeft(document.getElementById(gSpec.id));
      const w = P.weights.W_ih[gSpec.offset + n][f];
      const signal = w * xVal;
      const col = signal >= 0 ? "#38BDF8" : "#F59E0B";
      const width = Math.min(3.5, 0.8 + Math.abs(w) * 2.2);
      const op = f === state.selectedFeatureIdx ? 0.78 : 0.25;
      makeCurve(src, dst, col, width, op, state.playing || f === state.selectedFeatureIdx);
    }});
  }}

  // 2. Synapses from 4 Gates -> Selected Neuron Cell in 64-Neuron Lattice
  const neuronEl = document.getElementById(`neuron-cell-${{n}}`);
  const nLeft = anchorLeft(neuronEl);
  ["gate-card-forget_gate", "gate-card-input_gate", "gate-card-candidate_gate", "gate-card-cell_state", "gate-card-output_gate"].forEach(gid => {{
    const src = anchorRight(document.getElementById(gid));
    makeCurve(src, nLeft, "#A855F7", 2.0, 0.65, true);
  }});

  // 3. Synapse from Selected Neuron + Bypass Boxes -> Concatenation / Dense Card
  const nRight = anchorRight(neuronEl);
  const fcLeft = anchorLeft(document.getElementById("card-fc"));
  const fcW = P.weights.fc_weight[n];
  makeCurve(
    nRight,
    fcLeft,
    fcW >= 0 ? "#FBBF24" : "#38BDF8",
    Math.min(4.0, 1.4 + Math.abs(fcW) * 6.0),
    state.mutedNeurons.has(n) ? 0.15 : 0.85,
    !state.mutedNeurons.has(n)
  );

  if (P.has_geography) {{
    const geoRight = anchorRight(document.getElementById("box-geography"));
    makeCurve(geoRight, fcLeft, "#34D399", 1.8, 0.55, true);
  }}
  const cityRight = anchorRight(document.getElementById("box-city_embedding"));
  makeCurve(cityRight, fcLeft, "#FBBF24", 1.8, 0.55, true);
}}

function selectComponent(cid) {{
  state.selectedComponent = cid;
  renderComponentAndTensorDock();
}}

function selectGate(gKey) {{
  state.selectedGate = gKey;
  state.selectedComponent = "lstm";
  renderAll();
}}

function toggleMuteSelectedNeuron() {{
  const n = state.selectedNeuron;
  if (state.mutedNeurons.has(n)) state.mutedNeurons.delete(n);
  else state.mutedNeurons.add(n);
  runLiveForwardPass();
  renderAll();
}}

function toggleMuteGroup(group) {{
  if (group === "geo") {{
    if (state.mutedGeo.size === P.geo_dim) state.mutedGeo.clear();
    else for (let i = 0; i < P.geo_dim; i++) state.mutedGeo.add(i);
  }} else if (group === "city") {{
    if (state.mutedCity.size === P.city_embedding_dim) state.mutedCity.clear();
    else for (let i = 0; i < P.city_embedding_dim; i++) state.mutedCity.add(i);
  }}
  runLiveForwardPass();
  renderAll();
}}

function stepDay(delta) {{
  state.dayIdx = (state.dayIdx + delta + P.sequence_length) % P.sequence_length;
  renderAll();
}}

function togglePlay() {{
  state.playing = !state.playing;
  const btn = document.getElementById("btn-play");
  if (state.playing) {{
    btn.textContent = "⏸ Pause Pass";
    btn.classList.add("playing");
    if (state.dayIdx >= P.sequence_length - 1) state.dayIdx = 0;
    state.timer = setInterval(() => {{
      if (state.dayIdx < P.sequence_length - 1) {{
        state.dayIdx++;
        renderAll();
      }} else {{
        togglePlay();
      }}
    }}, state.speedMs);
  }} else {{
    btn.textContent = "▶ Play 30-Day Pass";
    btn.classList.remove("playing");
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }}
  drawSynapses();
}}

function updateSpeed() {{
  state.speedMs = parseInt(document.getElementById("speed-select").value, 10);
  if (state.playing) {{
    clearInterval(state.timer);
    state.timer = setInterval(() => {{
      if (state.dayIdx < P.sequence_length - 1) {{
        state.dayIdx++;
        renderAll();
      }} else {{
        togglePlay();
      }}
    }}, state.speedMs);
  }}
}}

function renderAll() {{
  const t = state.dayIdx;
  document.getElementById("day-readout").textContent =
    `Day ${{t + 1}} / ${{P.sequence_length}} · ${{P.dates[t]}}`;

  document.getElementById("shape-weather_input").textContent =
    `(${{P.batch_size}}, 30, ${{P.lstm_input_size}})`;
  document.getElementById("shape-lstm").textContent =
    `(${{P.batch_size}}, 30, 64) → (${{P.batch_size}}, 64)`;
  document.getElementById("shape-concatenation").textContent =
    `(${{P.batch_size}}, ${{P.fc_input_size}}) → (${{P.batch_size}}, 1)`;

  buildTimelineStrip();
  renderInputNeurons();
  renderGatesAndLattice();
  renderBypassAndReadout();
  renderLiveEquationsAndCharts();
  renderComponentAndTensorDock();
  requestAnimationFrame(drawSynapses);
}}

window.addEventListener("resize", () => requestAnimationFrame(drawSynapses));

// Initialize on load
runLiveForwardPass();
renderAll();
</script>
</body>
</html>
"""
    return html


def render_interactive_neural_playground(
    model_key: str,
    summary: Dict[str, Any],
    artifacts: Dict[str, Any],
    walkthrough: Dict[str, Any],
    gate_result: Dict[str, Any],
    batch_size: int = 1,
    mode: str = "Beginner",
    selected_neuron: int = 25,
) -> None:
    """
    Renders the interactive HTML5/SVG Neural Network Playground in Streamlit.
    """
    payload = build_playground_payload(
        model_key=model_key,
        summary=summary,
        artifacts=artifacts,
        walkthrough=walkthrough,
        gate_result=gate_result,
        batch_size=batch_size,
        mode=mode,
        selected_neuron=selected_neuron,
    )
    html_str = build_interactive_playground_html(payload)
    components.html(html_str, height=960, scrolling=True)
