from typing import Dict, Any, List
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from src.architecture_inspector import get_city_date_bounds
from src.model_comparison_inspector import (
    evaluate_full_test_benchmark,
    run_head_to_head_comparison,
    build_architecture_comparison_table,
)


def _init_comparison_session_state() -> None:
    """Initializes persistent session state keys for the Model Comparison Explorer."""
    defaults = {
        "mc_city": "Chennai",
        "mc_window_preset": "Test Split Benchmark (Ends 2025-10-19 → Predicts 2025-10-20)",
        "mc_mode": "Beginner",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _inject_comparison_styles() -> None:
    """Injects clean, non-truncating card and header styles for the Model Comparison Explorer."""
    st.markdown(
        """
        <style>
        .mc-tier-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-top: 6px;
            margin-bottom: 14px;
        }
        .mc-tier-card {
            background: linear-gradient(180deg, #162033 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 11px 14px;
        }
        .mc-tier-card.winner {
            border: 1.5px solid #34D399;
            box-shadow: 0 0 16px rgba(52, 211, 153, 0.14);
        }
        .mc-tier-badge {
            display: inline-block;
            font-size: 10.5px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
            margin-bottom: 4px;
        }
        .mc-tier-title {
            font-size: 14.5px;
            font-weight: 700;
            color: #F8FAFC;
            margin-bottom: 3px;
        }
        .mc-tier-arch {
            font-size: 11.5px;
            color: #94A3B8;
            font-family: Consolas, monospace;
            margin-bottom: 8px;
        }
        .mc-tier-metrics {
            display: flex;
            justify-content: space-between;
            gap: 6px;
            border-top: 1px solid #1E293B;
            padding-top: 7px;
            font-family: Consolas, monospace;
            font-size: 12px;
        }
        .mc-panel-hdr {
            background: linear-gradient(180deg, #162033 0%, #111827 100%);
            border: 1px solid #334155;
            border-bottom: none;
            border-radius: 10px 10px 0 0;
            padding: 9px 14px;
            margin-top: 6px;
            margin-bottom: -8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
        }
        .mc-panel-title {
            font-size: 13.5px;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: 0.2px;
        }
        .mc-panel-sub {
            font-size: 11px;
            color: #94A3B8;
            font-family: Consolas, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _build_head_to_head_arena_payload(
    h2h: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extracts Day-29 states, Day-30 inputs, and the exact trained PyTorch LSTM +
    Dense parameters for all 3 models so the browser can run live 3-model
    What-If weather & geography perturbations at 60fps.
    """
    target_scaler = artifacts["v2"]["target_scaler"]
    temporal_scaler = artifacts["v2"]["temporal_scaler"]

    t_min = [float(x) for x in temporal_scaler.data_min_]
    t_max = [float(x) for x in temporal_scaler.data_max_]
    y_min = float(target_scaler.data_min_[0])
    y_max = float(target_scaler.data_max_[0])

    w_B = h2h["models"]["model_B"]["walkthrough"]
    raw_day30_row = w_B["stage2_preprocessed_features"].iloc[-1].to_dict()
    raw_day30 = {
        k: round(float(v), 4)
        for k, v in raw_day30_row.items()
        if k != "date" and isinstance(v, (int, float, np.number))
    }
    scaled_day30_8 = [
        float(x) for x in w_B["stage3_scaled_input"]["array"][0, -1, :]
    ]

    geo_raw = w_B["stage4_geography"]["raw_values"]
    geo_scaled_B = [
        float(w_B["stage4_geography"]["scaled_values"][f])
        for f in w_B["stage4_geography"]["feature_names"]
    ]
    w_A = h2h["models"]["model_A"]["walkthrough"]
    geo_scaled_A = [
        float(w_A["stage4_geography"]["scaled_values"][f])
        for f in w_A["stage4_geography"]["feature_names"]
    ]

    models_payload: Dict[str, Any] = {}
    for m_key in ["baseline", "model_A", "model_B"]:
        model = artifacts["models"][m_key]
        m_info = h2h["models"][m_key]
        gd = m_info["gate_data"]
        wt = m_info["walkthrough"]

        W_ih = model.lstm.weight_ih_l0.detach().cpu().numpy()
        W_hh = model.lstm.weight_hh_l0.detach().cpu().numpy()
        b_all = (
            model.lstm.bias_ih_l0.detach().cpu().numpy()
            + model.lstm.bias_hh_l0.detach().cpu().numpy()
        )
        W_fc = model.fc.weight.detach().cpu().numpy()[0]
        b_fc = float(model.fc.bias.detach().cpu().numpy()[0])

        # Day 29 state (index 28) is the recurrent input into Day 30 (index 29)
        h_29 = [round(float(x), 6) for x in gd["hidden_state"][28]]
        c_29 = [round(float(x), 6) for x in gd["cell_state"][28]]
        city_emb = [
            round(float(x), 6) for x in wt["stage5_city_embedding"]["vector"]
        ]

        models_payload[m_key] = {
            "model_key": m_key,
            "model_name": m_info["model_name"],
            "input_size": int(W_ih.shape[1]),
            "geo_dim": int(m_info["summary"]["geo_dim"]),
            "total_params": int(m_info["summary"]["trainable_parameters"]),
            "h_29": h_29,
            "c_29": c_29,
            "city_emb": city_emb,
            "W_ih": [[round(float(v), 5) for v in row] for row in W_ih],
            "W_hh": [[round(float(v), 5) for v in row] for row in W_hh],
            "b_lstm": [round(float(v), 5) for v in b_all],
            "W_fc": [round(float(v), 5) for v in W_fc],
            "b_fc": round(b_fc, 6),
        }

    return {
        "city": h2h["city"],
        "last_observation_date": h2h["last_observation_date"],
        "prediction_date": h2h["prediction_date"],
        "actual_celsius": h2h["actual_celsius"],
        "day30_max_temp": h2h["day30_max_temp_celsius"],
        "t_min": t_min,
        "t_max": t_max,
        "y_min": y_min,
        "y_max": y_max,
        "raw_day30": raw_day30,
        "scaled_day30_8": [round(x, 6) for x in scaled_day30_8],
        "geo_raw": {k: round(float(v), 2) for k, v in geo_raw.items()},
        "geo_scaled_A": [round(x, 6) for x in geo_scaled_A],
        "geo_scaled_B": [round(x, 6) for x in geo_scaled_B],
        "models": models_payload,
    }


def build_head_to_head_arena_html(
    h2h: Dict[str, Any],
    artifacts: Dict[str, Any],
) -> str:
    """
    Builds an interactive, compact (515px) HTML5/JS 3-Model Head-to-Head Arena
    where the user can perturb Day-30 weather features or mute geographical
    features (`coastal_distance_km`) and watch Baseline V1, Model A, and Model B
    recompute their 64 LSTM gates and next-day °C forecasts side by side.
    """
    payload = _build_head_to_head_arena_payload(h2h, artifacts)
    payload_json = json.dumps(payload)

    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; }
  body { background: transparent; color: #F8FAFC; overflow: hidden; }
  .arena-wrap {
    background: linear-gradient(180deg, #0F172A 0%, #090D16 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 10px 14px;
    height: 505px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 12px;
  }
  .top-title { font-size: 13px; font-weight: 700; color: #F8FAFC; }
  .top-sub { font-size: 11.5px; color: #94A3B8; font-family: Consolas, monospace; }
  .btn-reset {
    background: #334155; color: #F8FAFC; border: 1px solid #475569;
    padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer;
  }
  .btn-reset:hover { border-color: #38BDF8; color: #38BDF8; }
  .main-grid {
    display: grid;
    grid-template-columns: 285px 1fr;
    gap: 12px;
    flex: 1;
    min-height: 0;
  }
  .ctrl-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .sec-hdr {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    color: #38BDF8;
    margin-bottom: 6px;
  }
  .slider-row { margin-bottom: 7px; }
  .slider-lbl {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-bottom: 2px;
  }
  .slider-lbl span:first-child { color: #CBD5E1; font-weight: 600; }
  .slider-lbl span:last-child { color: #38BDF8; font-family: Consolas, monospace; font-weight: 700; }
  input[type=range] {
    width: 100%;
    accent-color: #38BDF8;
    height: 5px;
    cursor: pointer;
  }
  .tag-v2-only {
    font-size: 9.5px;
    background: rgba(192, 132, 252, 0.15);
    color: #C084FC;
    border: 1px solid rgba(192, 132, 252, 0.4);
    border-radius: 4px;
    padding: 0 4px;
    margin-left: 4px;
  }
  .geo-box {
    background: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 8px 10px;
    margin-top: 4px;
  }
  .geo-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 8px;
    font-size: 10.5px;
    font-family: Consolas, monospace;
    color: #CBD5E1;
    margin-bottom: 7px;
  }
  .toggle-row {
    display: flex;
    gap: 6px;
  }
  .btn-toggle {
    flex: 1;
    background: #1E293B;
    color: #CBD5E1;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 6px;
    font-size: 10.5px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
  }
  .btn-toggle.active {
    background: rgba(244, 63, 94, 0.18);
    color: #FDA4AF;
    border-color: #F43F5E;
  }
  .models-col {
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-height: 0;
  }
  .cards-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    flex: 1;
  }
  .m-card {
    background: #111827;
    border: 1.5px solid #1E293B;
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
  }
  .m-card.best {
    border-color: #34D399;
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, #111827 100%);
  }
  .win-pill {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid #34D399;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 7px;
    display: none;
  }
  .m-card.best .win-pill { display: inline-block; }
  .m-title { font-size: 13.5px; font-weight: 700; color: #F8FAFC; }
  .m-sub { font-size: 10.5px; color: #94A3B8; font-family: Consolas, monospace; margin-top: 2px; }
  .pred-box {
    background: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 7px 10px;
    text-align: center;
    margin: 6px 0;
  }
  .pred-temp {
    font-size: 21px;
    font-weight: 800;
    font-family: Consolas, monospace;
    color: #F8FAFC;
  }
  .pred-err {
    font-size: 11px;
    font-family: Consolas, monospace;
    margin-top: 2px;
  }
  .gate-mini-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 4px;
    margin-bottom: 6px;
  }
  .g-chip {
    background: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 5px;
    padding: 3px 4px;
    text-align: center;
    font-family: Consolas, monospace;
  }
  .g-chip-lbl { font-size: 9.5px; color: #94A3B8; }
  .g-chip-val { font-size: 11px; font-weight: 700; color: #E2E8F0; }
  .contrib-list {
    background: #0B0F19;
    border: 1px solid #1E293B;
    border-radius: 7px;
    padding: 6px 8px;
    font-size: 10.5px;
    font-family: Consolas, monospace;
  }
  .contrib-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
  }
  .contrib-row:last-child { margin-bottom: 0; border-top: 1px solid #1E293B; padding-top: 3px; font-weight: 700; }
  .insight-banner {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 11.5px;
    color: #E2E8F0;
    line-height: 1.4;
  }
</style>
</head>
<body>
<div class="arena-wrap">
  <div class="top-bar">
    <div>
      <span class="top-title" id="hdr-title">⚡ Live 3-Model Head-to-Head Forecast Arena</span>
      <span class="top-sub" id="hdr-sub" style="margin-left:10px;"></span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span class="top-sub" id="obs-pill" style="color:#34D399;font-weight:700;"></span>
      <button class="btn-reset" onclick="resetPerturbations()">↺ Reset to Real Historical Observation</button>
    </div>
  </div>

  <div class="main-grid">
    <!-- LEFT: LIVE WHAT-IF WEATHER & GEOGRAPHY PERTURBATION CONTROLS -->
    <div class="ctrl-card">
      <div>
        <div class="sec-hdr">1. Perturb Day-30 Weather Input (x_30)</div>
        <div class="slider-row">
          <div class="slider-lbl">
            <span>Max Temp (°C) [All 3 Models]</span>
            <span id="lbl-tmax"></span>
          </div>
          <input type="range" id="sl-tmax" min="-5" max="48" step="0.2" oninput="onInputChanged()">
        </div>

        <div class="slider-row">
          <div class="slider-lbl">
            <span>Humidity (%) <span class="tag-v2-only">A & B Only</span></span>
            <span id="lbl-hum"></span>
          </div>
          <input type="range" id="sl-hum" min="5" max="100" step="1" oninput="onInputChanged()">
        </div>

        <div class="slider-row">
          <div class="slider-lbl">
            <span>Rainfall log1p <span class="tag-v2-only">A & B Only</span></span>
            <span id="lbl-rain"></span>
          </div>
          <input type="range" id="sl-rain" min="0" max="5.2" step="0.1" oninput="onInputChanged()">
        </div>

        <div class="slider-row">
          <div class="slider-lbl">
            <span>Wind Max (km/h) <span class="tag-v2-only">A & B Only</span></span>
            <span id="lbl-wind"></span>
          </div>
          <input type="range" id="sl-wind" min="0" max="55" step="0.5" oninput="onInputChanged()">
        </div>
      </div>

      <div class="geo-box">
        <div class="sec-hdr" style="color:#34D399;margin-bottom:4px;">2. Geography Bypass Ablation</div>
        <div class="geo-grid" id="geo-info-grid"></div>
        <div class="toggle-row">
          <button class="btn-toggle" id="btn-mute-coast" onclick="toggleCoastMute()">Mute Coastal Dist (Model B)</button>
          <button class="btn-toggle" id="btn-mute-geo" onclick="toggleGeoMute()">Mute All Geo (A & B)</button>
        </div>
      </div>
    </div>

    <!-- RIGHT: 3 SIDE-BY-SIDE MODEL CARDS + INSIGHT BANNER -->
    <div class="models-col">
      <div class="cards-3">
        <!-- BASELINE V1 -->
        <div class="m-card" id="card-baseline">
          <span class="win-pill">🏆 Closest to Observed</span>
          <div>
            <div class="m-title" style="color:#38BDF8;">Baseline V1 (Univariate)</div>
            <div class="m-sub">Input: (1, 30, 1) · Geo: 0 · 17,625 params</div>
          </div>
          <div class="pred-box">
            <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;">Predicted Next-Day Max Temp</div>
            <div class="pred-temp" id="temp-baseline">--</div>
            <div class="pred-err" id="err-baseline">--</div>
          </div>
          <div>
            <div style="font-size:10px;color:#94A3B8;margin-bottom:3px;">Day-30 64-Neuron Mean Gate Activations:</div>
            <div class="gate-mini-grid" id="gates-baseline"></div>
          </div>
          <div class="contrib-list" id="contrib-baseline"></div>
        </div>

        <!-- MODEL A -->
        <div class="m-card" id="card-model_A">
          <span class="win-pill">🏆 Closest to Observed</span>
          <div>
            <div class="m-title" style="color:#C084FC;">Model A (Multivariate + 3 Geo)</div>
            <div class="m-sub">Input: (1, 30, 8) · Geo: 3 · 19,420 params</div>
          </div>
          <div class="pred-box">
            <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;">Predicted Next-Day Max Temp</div>
            <div class="pred-temp" id="temp-model_A">--</div>
            <div class="pred-err" id="err-model_A">--</div>
          </div>
          <div>
            <div style="font-size:10px;color:#94A3B8;margin-bottom:3px;">Day-30 64-Neuron Mean Gate Activations:</div>
            <div class="gate-mini-grid" id="gates-model_A"></div>
          </div>
          <div class="contrib-list" id="contrib-model_A"></div>
        </div>

        <!-- MODEL B -->
        <div class="m-card" id="card-model_B">
          <span class="win-pill">🏆 Closest to Observed</span>
          <div>
            <div class="m-title" style="color:#34D399;">Model B (Multivariate + 4 Geo)</div>
            <div class="m-sub">Input: (1, 30, 8) · Geo: 4 (+Coast) · 19,421 params</div>
          </div>
          <div class="pred-box">
            <div style="font-size:10px;color:#94A3B8;text-transform:uppercase;">Predicted Next-Day Max Temp</div>
            <div class="pred-temp" id="temp-model_B">--</div>
            <div class="pred-err" id="err-model_B">--</div>
          </div>
          <div>
            <div style="font-size:10px;color:#94A3B8;margin-bottom:3px;">Day-30 64-Neuron Mean Gate Activations:</div>
            <div class="gate-mini-grid" id="gates-model_B"></div>
          </div>
          <div class="contrib-list" id="contrib-model_B"></div>
        </div>
      </div>

      <div class="insight-banner" id="insight-banner"></div>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD_JSON__;

let muteCoast = false;
let muteAllGeo = false;
let origPreds = {};

function sigmoid(x) { return 1.0 / (1.0 + Math.exp(-x)); }

function scaleFeature(val, idx) {
  const mn = DATA.t_min[idx];
  const mx = DATA.t_max[idx];
  return (val - mn) / (mx - mn + 1e-12);
}

function inverseScaleTarget(s) {
  return DATA.y_min + s * (DATA.y_max - DATA.y_min);
}

function runModelStep30(mKey, x30_8, gA, gB) {
  const m = DATA.models[mKey];
  const xVec = (mKey === 'baseline') ? [x30_8[0]] : x30_8;
  const h29 = m.h_29;
  const c29 = m.c_29;
  const H = 64;

  const z = new Array(256).fill(0.0);
  for (let r = 0; r < 256; r++) {
    let s = m.b_lstm[r];
    const wihRow = m.W_ih[r];
    for (let j = 0; j < xVec.length; j++) s += wihRow[j] * xVec[j];
    const whhRow = m.W_hh[r];
    for (let j = 0; j < H; j++) s += whhRow[j] * h29[j];
    z[r] = s;
  }

  let fSum = 0, iSum = 0, gSum = 0, oSum = 0;
  const h30 = new Array(H);
  for (let u = 0; u < H; u++) {
    const i_u = sigmoid(z[u]);
    const f_u = sigmoid(z[64 + u]);
    const g_u = Math.tanh(z[128 + u]);
    const o_u = sigmoid(z[192 + u]);
    const c_u = f_u * c29[u] + i_u * g_u;
    const h_u = o_u * Math.tanh(c_u);
    h30[u] = h_u;
    fSum += f_u; iSum += i_u; gSum += g_u; oSum += o_u;
  }

  let lstmContrib = 0.0;
  for (let u = 0; u < H; u++) lstmContrib += m.W_fc[u] * h30[u];

  let geoContrib = 0.0;
  let coastContrib = 0.0;
  const geoVec = (mKey === 'model_A') ? gA : ((mKey === 'model_B') ? gB : []);
  for (let j = 0; j < geoVec.length; j++) {
    const term = m.W_fc[H + j] * geoVec[j];
    geoContrib += term;
    if (mKey === 'model_B' && j === 3) coastContrib = term;
  }

  let cityContrib = 0.0;
  const cityOffset = H + geoVec.length;
  for (let j = 0; j < 8; j++) {
    cityContrib += m.W_fc[cityOffset + j] * m.city_emb[j];
  }

  const scaledOut = lstmContrib + geoContrib + cityContrib + m.b_fc;
  const tempC = inverseScaleTarget(scaledOut);

  return {
    tempC: tempC,
    scaledOut: scaledOut,
    fMean: fSum / H,
    iMean: iSum / H,
    gMean: gSum / H,
    oMean: oSum / H,
    lstmContrib: lstmContrib,
    geoContrib: geoContrib,
    coastContrib: coastContrib,
    cityContrib: cityContrib,
    biasContrib: m.b_fc
  };
}

function initUI() {
  document.getElementById('hdr-sub').textContent =
    `${DATA.city} · Window Ending ${DATA.last_observation_date} → Target ${DATA.prediction_date}`;
  document.getElementById('obs-pill').textContent =
    DATA.actual_celsius !== null
      ? `Observed Ground Truth: ${DATA.actual_celsius.toFixed(2)} °C (Day 30: ${DATA.day30_max_temp.toFixed(1)} °C)`
      : `Day 30 Max Temp: ${DATA.day30_max_temp.toFixed(1)} °C`;

  const gr = DATA.geo_raw;
  document.getElementById('geo-info-grid').innerHTML = `
    <div>Lat: <b>${gr.latitude.toFixed(2)}°</b></div>
    <div>Lon: <b>${gr.longitude.toFixed(2)}°</b></div>
    <div>Elev: <b>${gr.elevation_m.toFixed(0)} m</b></div>
    <div style="color:#34D399;">Coast: <b>${gr.coastal_distance_km.toFixed(1)} km</b></div>
  `;

  resetPerturbations();
}

function resetPerturbations() {
  muteCoast = false;
  muteAllGeo = false;
  document.getElementById('btn-mute-coast').classList.remove('active');
  document.getElementById('btn-mute-geo').classList.remove('active');

  document.getElementById('sl-tmax').value = DATA.raw_day30.temperature_2m_max;
  document.getElementById('sl-hum').value = DATA.raw_day30.relative_humidity_2m_mean;
  document.getElementById('sl-rain').value = DATA.raw_day30.precipitation_log;
  document.getElementById('sl-wind').value = DATA.raw_day30.wind_speed_10m_max;

  onInputChanged(true);
}

function toggleCoastMute() {
  muteCoast = !muteCoast;
  document.getElementById('btn-mute-coast').classList.toggle('active', muteCoast);
  onInputChanged(false);
}

function toggleGeoMute() {
  muteAllGeo = !muteAllGeo;
  document.getElementById('btn-mute-geo').classList.toggle('active', muteAllGeo);
  onInputChanged(false);
}

function onInputChanged(isInit) {
  const tmax = parseFloat(document.getElementById('sl-tmax').value);
  const hum = parseFloat(document.getElementById('sl-hum').value);
  const rain = parseFloat(document.getElementById('sl-rain').value);
  const wind = parseFloat(document.getElementById('sl-wind').value);

  document.getElementById('lbl-tmax').textContent = `${tmax.toFixed(1)} °C`;
  document.getElementById('lbl-hum').textContent = `${hum.toFixed(0)} %`;
  document.getElementById('lbl-rain').textContent = `${rain.toFixed(2)}`;
  document.getElementById('lbl-wind').textContent = `${wind.toFixed(1)} km/h`;

  const x30 = DATA.scaled_day30_8.slice();
  x30[0] = scaleFeature(tmax, 0);
  x30[3] = scaleFeature(hum, 3);
  x30[4] = scaleFeature(rain, 4);
  x30[5] = scaleFeature(wind, 5);

  const gA = muteAllGeo ? [0, 0, 0] : DATA.geo_scaled_A.slice();
  const gB = muteAllGeo ? [0, 0, 0, 0] : DATA.geo_scaled_B.slice();
  if (muteCoast && !muteAllGeo) gB[3] = 0.0;

  const res = {
    baseline: runModelStep30('baseline', x30, gA, gB),
    model_A: runModelStep30('model_A', x30, gA, gB),
    model_B: runModelStep30('model_B', x30, gA, gB)
  };

  if (isInit) {
    origPreds = {
      baseline: res.baseline.tempC,
      model_A: res.model_A.tempC,
      model_B: res.model_B.tempC
    };
  }

  let bestKey = 'model_B';
  let minErr = 1e9;
  if (DATA.actual_celsius !== null) {
    ['baseline', 'model_A', 'model_B'].forEach(k => {
      const e = Math.abs(res[k].tempC - DATA.actual_celsius);
      if (e < minErr) { minErr = e; bestKey = k; }
    });
  }

  ['baseline', 'model_A', 'model_B'].forEach(k => {
    const r = res[k];
    const card = document.getElementById(`card-${k}`);
    card.classList.toggle('best', k === bestKey && DATA.actual_celsius !== null);

    document.getElementById(`temp-${k}`).textContent = `${r.tempC.toFixed(2)} °C`;

    const deltaVsOrig = r.tempC - origPreds[k];
    let errHtml = '';
    if (DATA.actual_celsius !== null) {
      const absE = Math.abs(r.tempC - DATA.actual_celsius);
      const col = (k === bestKey) ? '#34D399' : '#94A3B8';
      errHtml = `<span style="color:${col};">|Error vs Observed|: ${absE.toFixed(2)} °C</span>`;
    }
    if (Math.abs(deltaVsOrig) > 0.005) {
      const dCol = deltaVsOrig >= 0 ? '#FBBF24' : '#38BDF8';
      errHtml += ` · <span style="color:${dCol};">Δ What-If: ${deltaVsOrig >= 0 ? '+' : ''}${deltaVsOrig.toFixed(2)} °C</span>`;
    }
    document.getElementById(`err-${k}`).innerHTML = errHtml;

    document.getElementById(`gates-${k}`).innerHTML = `
      <div class="g-chip"><div class="g-chip-lbl">Forget f</div><div class="g-chip-val" style="color:#C084FC;">${r.fMean.toFixed(3)}</div></div>
      <div class="g-chip"><div class="g-chip-lbl">Input i</div><div class="g-chip-val" style="color:#38BDF8;">${r.iMean.toFixed(3)}</div></div>
      <div class="g-chip"><div class="g-chip-lbl">Cand g</div><div class="g-chip-val" style="color:#34D399;">${r.gMean >= 0 ? '+' : ''}${r.gMean.toFixed(3)}</div></div>
      <div class="g-chip"><div class="g-chip-lbl">Output o</div><div class="g-chip-val" style="color:#FB7185;">${r.oMean.toFixed(3)}</div></div>
    `;

    const geoLabel = (k === 'baseline')
      ? 'Geography (None)'
      : (k === 'model_A' ? 'Geography (3 dims)' : 'Geography (4 dims)');

    document.getElementById(`contrib-${k}`).innerHTML = `
      <div class="contrib-row"><span style="color:#94A3B8;">LSTM 64 Units (h_30):</span><span style="color:#E2E8F0;">${r.lstmContrib >= 0 ? '+' : ''}${r.lstmContrib.toFixed(4)}</span></div>
      <div class="contrib-row"><span style="color:#94A3B8;">${geoLabel}:</span><span style="color:#34D399;">${r.geoContrib >= 0 ? '+' : ''}${r.geoContrib.toFixed(4)}</span></div>
      <div class="contrib-row"><span style="color:#94A3B8;">City Embedding (8d):</span><span style="color:#FBBF24;">${r.cityContrib >= 0 ? '+' : ''}${r.cityContrib.toFixed(4)}</span></div>
      <div class="contrib-row"><span style="color:#94A3B8;">Scaled Output y_hat:</span><span style="color:#38BDF8;">${r.scaledOut.toFixed(4)}</span></div>
    `;
  });

  const nameMap = { baseline: 'Baseline V1', model_A: 'Model A', model_B: 'Model B' };
  const isPerturbed = (
    Math.abs(tmax - DATA.raw_day30.temperature_2m_max) > 0.05 ||
    Math.abs(hum - DATA.raw_day30.relative_humidity_2m_mean) > 0.5 ||
    Math.abs(rain - DATA.raw_day30.precipitation_log) > 0.05 ||
    Math.abs(wind - DATA.raw_day30.wind_speed_10m_max) > 0.2 ||
    muteCoast || muteAllGeo
  );

  if (isPerturbed) {
    document.getElementById('insight-banner').innerHTML =
      `🔬 <b>Live Perturbation Active:</b> Notice that changing <b>Humidity</b>, <b>Rainfall</b>, or <b>Wind Speed</b> shifts <b>Model A</b> (${(res.model_A.tempC - origPreds.model_A >= 0 ? '+' : '') + (res.model_A.tempC - origPreds.model_A).toFixed(2)} °C) and <b>Model B</b> (${(res.model_B.tempC - origPreds.model_B >= 0 ? '+' : '') + (res.model_B.tempC - origPreds.model_B).toFixed(2)} °C), whereas <b>Baseline V1</b> (${(res.baseline.tempC - origPreds.baseline >= 0 ? '+' : '') + (res.baseline.tempC - origPreds.baseline).toFixed(2)} °C) only sees <code>temp_max</code>!`;
  } else if (DATA.actual_celsius !== null) {
    document.getElementById('insight-banner').innerHTML =
      `🏆 <b>Head-to-Head Verdict on ${DATA.city} (${DATA.prediction_date}):</b> <b>${nameMap[bestKey]}</b> is closest to the observed ground truth (<b>${DATA.actual_celsius.toFixed(2)} °C</b>) with an absolute error of <b>${minErr.toFixed(2)} °C</b>. Drag the sliders on the left to test how each model responds to sudden humidity, rain, or wind changes!`;
  } else {
    document.getElementById('insight-banner').innerHTML =
      `💡 <b>Interactive What-If Mode:</b> Drag the weather sliders or toggle Geography Ablation on the left to compare how all 3 trained models react in real time.`;
  }
}

initUI();
</script>
</body>
</html>
"""
    return html.replace("__PAYLOAD_JSON__", payload_json)


def build_compact_city_trajectory_chart(
    bench: Dict[str, Any],
    city: str,
    focus_date: str,
) -> go.Figure:
    """
    Chart 1 (Compact 315px): Multi-day test split forecast trajectory for the
    selected city comparing Observed Max Temp (°C) against Baseline V1, Model A,
    and Model B around `focus_date`.
    """
    ts = bench["city_time_series"][city]
    dates = ts["dates"]
    n_total = len(dates)

    # Center a 45-day window around focus_date (or show last 45 test days)
    if focus_date in dates:
        center_idx = dates.index(focus_date)
    else:
        center_idx = n_total - 1

    start_idx = max(0, min(n_total - 45, center_idx - 22))
    end_idx = min(n_total, start_idx + 45)

    d_slice = dates[start_idx:end_idx]
    y_true = ts["y_true"][start_idx:end_idx]
    p_base = ts["baseline"][start_idx:end_idx]
    p_A = ts["model_A"][start_idx:end_idx]
    p_B = ts["model_B"][start_idx:end_idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d_slice,
            y=y_true,
            name="Observed (°C)",
            mode="lines+markers",
            line=dict(color="#F8FAFC", width=2.6),
            marker=dict(size=4.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_slice,
            y=p_base,
            name="Baseline V1",
            mode="lines",
            line=dict(color="#38BDF8", width=1.8, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_slice,
            y=p_A,
            name="Model A (3 Geo)",
            mode="lines",
            line=dict(color="#C084FC", width=1.9, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d_slice,
            y=p_B,
            name="Model B (4 Geo + Coast)",
            mode="lines",
            line=dict(color="#34D399", width=2.2),
        )
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=315,
        margin=dict(l=48, r=20, t=18, b=74),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(title="Test Split Date", gridcolor="#1E293B", nticks=8),
        yaxis=dict(title="Max Temp (°C)", gridcolor="#1E293B"),
    )
    return fig


def build_compact_error_comparison_chart(
    bench: Dict[str, Any],
    city: str,
) -> go.Figure:
    """
    Chart 2 (Compact 315px): Grouped bar chart comparing Global Test MAE & RMSE
    (18,750 samples) alongside the Selected City's Test MAE & RMSE (375 test days)
    across Naive Persistence, Baseline V1, Model A, and Model B.
    """
    gm = bench["global_metrics"]
    cm = bench["city_time_series"][city]["metrics"]

    keys = ["naive", "baseline", "model_A", "model_B"]
    labels = ["Naive (y_t=y_{t-1})", "Baseline V1", "Model A (3 Geo)", "Model B (4 Geo)"]

    global_mae = [gm[k]["MAE"] for k in keys]
    city_mae = [cm[k]["MAE"] for k in keys]
    city_rmse = [cm[k]["RMSE"] for k in keys]

    all_vals = global_mae + city_mae + city_rmse
    y_min = max(0.0, min(all_vals) - 0.25)
    y_max = max(all_vals) + 0.28

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=global_mae,
            name="All 50 Cities MAE (°C)",
            marker_color="#64748B",
            text=[f"{v:.3f}°" for v in global_mae],
            textposition="outside",
            textfont=dict(size=9.5),
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=city_mae,
            name=f"{city} Test MAE (°C)",
            marker_color="#34D399",
            text=[f"{v:.3f}°" for v in city_mae],
            textposition="outside",
            textfont=dict(size=9.5),
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=city_rmse,
            name=f"{city} Test RMSE (°C)",
            marker_color="#38BDF8",
            text=[f"{v:.3f}°" for v in city_rmse],
            textposition="outside",
            textfont=dict(size=9.5),
            cliponaxis=False,
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=315,
        margin=dict(l=48, r=20, t=22, b=74),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(
            title="Error (°C — Lower is Better)",
            gridcolor="#1E293B",
            range=[y_min, y_max],
        ),
    )
    return fig


def build_compact_coastal_impact_chart(
    bench: Dict[str, Any],
    selected_city: str,
) -> go.Figure:
    """
    Chart 3 (Compact 315px): Shows how Model B's MAE reduction over Baseline V1
    (`V1 MAE - Model B MAE`) and over Model A (`Model A MAE - Model B MAE`)
    behaves across all 50 cities grouped by Geographic Zone (Coastal < 60 km,
    Inland Plains, High Elevation ≥ 600 m).
    """
    df = bench["city_metrics_df"]
    zones = ["Coastal (< 60 km)", "Inland Plains (≥ 60 km)", "High Elevation (≥ 600 m)"]

    v1_zone_mae = []
    a_zone_mae = []
    b_zone_mae = []
    zone_labels = []

    for z in zones:
        sub = df[df["zone"] == z]
        if len(sub) > 0:
            zone_labels.append(f"{z} ({len(sub)} cities)")
            v1_zone_mae.append(float(sub["baseline_mae"].mean()))
            a_zone_mae.append(float(sub["model_A_mae"].mean()))
            b_zone_mae.append(float(sub["model_B_mae"].mean()))

    all_y = v1_zone_mae + a_zone_mae + b_zone_mae
    y_min = max(0.0, min(all_y) - 0.15)
    y_max = max(all_y) + 0.18

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=zone_labels,
            y=v1_zone_mae,
            name="Baseline V1 (0 Geo)",
            marker_color="#38BDF8",
            text=[f"{v:.3f}°C" for v in v1_zone_mae],
            textposition="outside",
            textfont=dict(size=9.5),
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=zone_labels,
            y=a_zone_mae,
            name="Model A (Lat, Lon, Elev)",
            marker_color="#C084FC",
            text=[f"{v:.3f}°C" for v in a_zone_mae],
            textposition="outside",
            textfont=dict(size=9.5),
            cliponaxis=False,
        )
    )
    fig.add_trace(
        go.Bar(
            x=zone_labels,
            y=b_zone_mae,
            name="Model B (+ Coastal Dist km)",
            marker_color="#34D399",
            text=[f"{v:.3f}°C" for v in b_zone_mae],
            textposition="outside",
            textfont=dict(size=9.5),
            cliponaxis=False,
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=315,
        margin=dict(l=48, r=20, t=22, b=74),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(
            title="Mean Test MAE (°C)",
            gridcolor="#1E293B",
            range=[y_min, y_max],
        ),
    )
    return fig


def build_compact_leaderboard_or_residual_chart(
    bench: Dict[str, Any],
    selected_city: str,
    mode_choice: str,
) -> go.Figure:
    """
    Chart 4 (Compact 295px): Either shows the Top 15 Cities where Model B
    achieves the largest MAE improvement over Baseline V1 (plus `selected_city`),
    OR the 18,750-sample Test Residual Distribution (`y_pred - y_true` in °C).
    """
    df = bench["city_metrics_df"].copy()

    if mode_choice.startswith("Top City"):
        df_sorted = df.sort_values("gain_B_vs_V1_mae", ascending=False).head(14)
        if selected_city not in df_sorted["city"].values:
            sel_row = df[df["city"] == selected_city]
            df_sorted = pd.concat([ sel_row, df_sorted.iloc[:13] ], ignore_index=True)

        cities = df_sorted["city"].tolist()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=cities,
                y=df_sorted["baseline_mae"],
                name="Baseline V1 MAE",
                marker_color="#38BDF8",
            )
        )
        fig.add_trace(
            go.Bar(
                x=cities,
                y=df_sorted["model_A_mae"],
                name="Model A MAE",
                marker_color="#C084FC",
            )
        )
        fig.add_trace(
            go.Bar(
                x=cities,
                y=df_sorted["model_B_mae"],
                name="Model B MAE",
                marker_color="#34D399",
            )
        )
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#0B0F19",
            height=290,
            margin=dict(l=48, r=20, t=18, b=78),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.28,
                xanchor="left",
                x=0.0,
                font=dict(size=10.5, color="#E2E8F0"),
            ),
            xaxis=dict(tickangle=-25, gridcolor="#1E293B"),
            yaxis=dict(title="Test MAE (°C)", gridcolor="#1E293B"),
        )
        return fig

    # Otherwise render Residual Error Histogram across all 18,750 test predictions
    res = bench["residuals"]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=res["baseline"],
            name="Baseline V1 Error",
            marker_color="#38BDF8",
            opacity=0.55,
            nbinsx=60,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=res["model_A"],
            name="Model A Error",
            marker_color="#C084FC",
            opacity=0.55,
            nbinsx=60,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=res["model_B"],
            name="Model B Error",
            marker_color="#34D399",
            opacity=0.65,
            nbinsx=60,
        )
    )
    fig.add_vline(x=0.0, line_width=1.5, line_dash="dash", line_color="#FBBF24")
    fig.update_layout(
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=290,
        margin=dict(l=48, r=20, t=18, b=74),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.26,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(title="Prediction Error y_pred - y_true (°C)", range=[-5.0, 5.0], gridcolor="#1E293B"),
        yaxis=dict(title="Test Samples", gridcolor="#1E293B"),
    )
    return fig


def render_model_comparison_explorer(artifacts: Dict[str, Any]) -> None:
    """
    Main Streamlit entry point for the Model Comparison Explorer page.
    Compares Naive Persistence, Baseline V1, Model A, and Model B across both
    the 18,750-sample test split and live interactive single-window What-If
    perturbations.
    """
    _init_comparison_session_state()
    _inject_comparison_styles()

    st.subheader("⚖️ Interactive Model Comparison Explorer (Baseline V1 vs. Model A vs. Model B)")
    st.caption(
        "Compare our three trained PyTorch weather LSTMs head to head across all **18,750 test predictions (50 Indian cities)**, "
        "perturb live weather & geographical inputs in the **3-Model Arena**, and see why adding multivariate weather + coastal distance "
        "makes **Model B** the most accurate model."
    )

    bench = evaluate_full_test_benchmark(artifacts)
    gm = bench["global_metrics"]
    wins = bench["city_win_counts"]
    city_df = bench["city_metrics_df"]

    # ==========================================
    # 1. TOP CONTROL BAR
    # ==========================================
    c1, c2, c3 = st.columns([1.35, 1.65, 1.0])

    cities_sorted = sorted(list(artifacts["city_to_id"].keys()))
    geo_map = city_df.set_index("city").to_dict(orient="index")

    with c1:
        selected_city = st.selectbox(
            "1. Select City for Head-to-Head Inspection (50 Cities)",
            options=cities_sorted,
            index=(
                cities_sorted.index(st.session_state["mc_city"])
                if st.session_state["mc_city"] in cities_sorted
                else 0
            ),
            format_func=lambda c: (
                f"{c} (Coast: {geo_map[c]['coastal_distance_km']:.0f} km, "
                f"Elev: {geo_map[c]['elevation_m']:.0f} m)"
            ),
            key="mc_city_select",
        )
        st.session_state["mc_city"] = selected_city

    bounds = get_city_date_bounds(selected_city)
    preset_labels = list(bounds["presets"].keys())
    with c2:
        selected_preset = st.selectbox(
            "2. 30-Day Historical Window Preset",
            options=preset_labels,
            index=(
                preset_labels.index(st.session_state["mc_window_preset"])
                if st.session_state["mc_window_preset"] in preset_labels
                else 0
            ),
            key="mc_preset_select",
        )
        st.session_state["mc_window_preset"] = selected_preset
    active_end_date = bounds["presets"][selected_preset]

    with c3:
        selected_mode = st.radio(
            "3. Explanation Depth",
            options=["Beginner", "Technical"],
            index=0 if st.session_state["mc_mode"] == "Beginner" else 1,
            horizontal=True,
            key="mc_mode_radio",
        )
        st.session_state["mc_mode"] = selected_mode

    # ==========================================
    # 2. NON-TRUNCATING 4-TIER BENCHMARK SCOREBOARD
    # ==========================================
    st.markdown(
        f"""
        <div class="mc-tier-grid">
            <div class="mc-tier-card">
                <span class="mc-tier-badge" style="background:#1E293B;color:#94A3B8;border:1px solid #475569;">Tier 0 · Reference</span>
                <div class="mc-tier-title">Naive Persistence</div>
                <div class="mc-tier-arch">Tomorrow = Day 30 Temp · 0 params</div>
                <div class="mc-tier-metrics">
                    <span>MAE: <b style="color:#E2E8F0;">{gm['naive']['MAE']:.4f}°C</b></span>
                    <span>RMSE: <b>{gm['naive']['RMSE']:.4f}°C</b></span>
                    <span>R²: <b>{gm['naive']['R2']:.4f}</b></span>
                </div>
            </div>
            <div class="mc-tier-card">
                <span class="mc-tier-badge" style="background:rgba(56,189,248,0.15);color:#38BDF8;border:1px solid #38BDF8;">Tier 1 · V1 Benchmark ({wins['Baseline V1']}/50 Cities)</span>
                <div class="mc-tier-title">Baseline V1 (WeatherLSTM)</div>
                <div class="mc-tier-arch">1 Temporal + City(8) · 17,625 params</div>
                <div class="mc-tier-metrics">
                    <span>MAE: <b style="color:#38BDF8;">{gm['baseline']['MAE']:.4f}°C</b></span>
                    <span>RMSE: <b>{gm['baseline']['RMSE']:.4f}°C</b></span>
                    <span>R²: <b>{gm['baseline']['R2']:.4f}</b></span>
                </div>
            </div>
            <div class="mc-tier-card">
                <span class="mc-tier-badge" style="background:rgba(192,132,252,0.15);color:#C084FC;border:1px solid #C084FC;">Tier 2 · V2 No Coast ({wins['Model A']}/50 Cities)</span>
                <div class="mc-tier-title">Model A (Multivariate + 3 Geo)</div>
                <div class="mc-tier-arch">8 Temp + 3 Geo + City(8) · 19,420 params</div>
                <div class="mc-tier-metrics">
                    <span>MAE: <b style="color:#C084FC;">{gm['model_A']['MAE']:.4f}°C</b></span>
                    <span>RMSE: <b>{gm['model_A']['RMSE']:.4f}°C</b></span>
                    <span>R²: <b>{gm['model_A']['R2']:.4f}</b></span>
                </div>
            </div>
            <div class="mc-tier-card winner">
                <span class="mc-tier-badge" style="background:rgba(52,211,153,0.18);color:#34D399;border:1px solid #34D399;">🏆 Tier 3 · Best Overall ({wins['Model B']}/50 Cities)</span>
                <div class="mc-tier-title">Model B (Multivariate + 4 Geo)</div>
                <div class="mc-tier-arch">8 Temp + 4 Geo (+Coast) + City(8) · 19,421 params</div>
                <div class="mc-tier-metrics">
                    <span>MAE: <b style="color:#34D399;">{gm['model_B']['MAE']:.4f}°C</b></span>
                    <span>RMSE: <b>{gm['model_B']['RMSE']:.4f}°C</b></span>
                    <span>R²: <b>{gm['model_B']['R2']:.4f}</b></span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================
    # 3. INTERACTIVE 3-MODEL HEAD-TO-HEAD ARENA
    # ==========================================
    h2h = run_head_to_head_comparison(
        city=selected_city,
        artifacts=artifacts,
        end_date=active_end_date,
    )
    arena_html = build_head_to_head_arena_html(h2h=h2h, artifacts=artifacts)
    components.html(arena_html, height=515, scrolling=False)

    # ==========================================
    # 4. NEAT, COMPACT 2×2 INTERACTIVE COMPARISON GRAPHS
    # ==========================================
    r1_left, r1_right = st.columns(2, gap="medium")
    with r1_left:
        st.markdown(
            f'<div class="mc-panel-hdr">'
            f'<span class="mc-panel-title">1. {selected_city} — Test Split Forecast Trajectory (45-Day Window)</span>'
            f'<span class="mc-panel-sub">Observed vs. V1, A, B</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_city_trajectory_chart(
                bench=bench,
                city=selected_city,
                focus_date=h2h["prediction_date"],
            ),
            use_container_width=True,
            theme=None,
        )

    with r1_right:
        st.markdown(
            f'<div class="mc-panel-hdr">'
            f'<span class="mc-panel-title">2. Global Benchmark (18,750 Samples) vs. {selected_city} Error</span>'
            f'<span class="mc-panel-sub">MAE & RMSE (°C)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_error_comparison_chart(
                bench=bench,
                city=selected_city,
            ),
            use_container_width=True,
            theme=None,
        )

    r2_left, r2_right = st.columns(2, gap="medium")
    with r2_left:
        st.markdown(
            f'<div class="mc-panel-hdr">'
            f'<span class="mc-panel-title">3. Why Coastal Distance Matters — MAE by Geographic Zone</span>'
            f'<span class="mc-panel-sub">All 50 Indian Cities</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_coastal_impact_chart(
                bench=bench,
                selected_city=selected_city,
            ),
            use_container_width=True,
            theme=None,
        )

    with r2_right:
        chart4_mode = st.radio(
            "Panel 4 Comparison View",
            options=[
                "Top City MAE Improvements (Model B vs. V1 & A)",
                "Test Error Distribution Histogram (18,750 Samples)",
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="mc_chart4_radio",
        )
        p4_title = (
            "4. Per-City Test MAE Leaderboard (Lowest Error Wins)"
            if chart4_mode.startswith("Top City")
            else "4. Test Residual Distribution (y_pred - y_true Across 18,750 Samples)"
        )
        st.markdown(
            f'<div class="mc-panel-hdr">'
            f'<span class="mc-panel-title">{p4_title}</span>'
            f'<span class="mc-panel-sub">50 Cities × 375 Days</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_leaderboard_or_residual_chart(
                bench=bench,
                selected_city=selected_city,
                mode_choice=chart4_mode,
            ),
            use_container_width=True,
            theme=None,
        )

    # ==========================================
    # 5. EXPANDABLE ARCHITECTURE & 50-CITY BENCHMARK TABLES
    # ==========================================
    with st.expander("📋 View Side-by-Side Architecture, Parameter & Benchmark Comparison Table", expanded=False):
        if selected_mode == "Beginner":
            st.info(
                "**Why Model B Beats Model A and Baseline V1:**\n"
                "1. **From Baseline V1 (`MAE 1.0373 °C`) to Model A (`MAE 1.0057 °C`):** Adding 7 daily weather signals "
                "(min/mean temp, humidity, log rainfall, wind speed, and seasonal $\\sin/\\cos$ day-of-year) plus 3 static coordinates "
                "(latitude, longitude, elevation) lets the LSTM detect rain cooling and seasonal transitions.\n"
                "2. **From Model A (`19,420 params`) to Model B (`19,421 params`, `MAE 0.9827 °C`):** Adding just **1 extra parameter** "
                "for `coastal_distance_km` informs the dense readout layer how strongly maritime sea-breeze buffering moderates "
                "daily maximum temperature swings!"
            )
        comp_df = build_architecture_comparison_table(artifacts)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with st.expander("🌏 Explore All 50 Indian Cities Test Set Leaderboard (375 Test Days per City)", expanded=False):
        display_df = city_df[
            [
                "city",
                "zone",
                "elevation_m",
                "coastal_distance_km",
                "baseline_mae",
                "model_A_mae",
                "model_B_mae",
                "gain_B_vs_V1_mae",
                "best_model",
            ]
        ].sort_values("model_B_mae", ascending=True)
        display_df = display_df.rename(
            columns={
                "city": "City",
                "zone": "Geographic Zone",
                "elevation_m": "Elevation (m)",
                "coastal_distance_km": "Coastal Dist (km)",
                "baseline_mae": "Baseline V1 MAE (°C)",
                "model_A_mae": "Model A MAE (°C)",
                "model_B_mae": "Model B MAE (°C)",
                "gain_B_vs_V1_mae": "MAE Improvement (V1 → B)",
                "best_model": "Winning Model",
            }
        )
        st.dataframe(
            display_df.style.format(
                {
                    "Elevation (m)": "{:.0f}",
                    "Coastal Dist (km)": "{:.1f}",
                    "Baseline V1 MAE (°C)": "{:.4f}",
                    "Model A MAE (°C)": "{:.4f}",
                    "Model B MAE (°C)": "{:.4f}",
                    "MAE Improvement (V1 → B)": "{:+.4f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
