import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from src.architecture_inspector import (
    MODEL_DISPLAY_NAMES,
    get_city_date_bounds,
)
from src.gate_memory_inspector import (
    analyze_gate_and_memory_flow,
    extract_neuron_gate_story,
)


def _init_gate_explorer_state() -> None:
    defaults = {
        "gm_model_key": "model_B",
        "gm_city": "Coimbatore",
        "gm_window_preset": "Test Split Benchmark (Ends 2025-10-19 → Predicts 2025-10-20)",
        "gm_custom_end_date": "2025-10-19",
        "gm_neuron": 25,
        "gm_day": 30,
        "gm_right_chart_mode": "Memory Survival Horizon (Past Days Still in c_t)",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_gate_memory_circuit_html(
    analysis: Dict[str, Any],
    initial_neuron: int = 25,
    initial_day: int = 30,
) -> str:
    """
    Builds a compact, self-contained HTML5/SVG Interactive LSTM Cell Circuit &
    Data-Flow Animator (~500px tall) that shows:
      1. How x_t (today's weather) and h_{t-1} (yesterday's summary) enter the cell
         and pass into all 4 gates (f_t, i_t, g_t, o_t).
      2. How each gate calculates z_t = W_ih*x_t + W_hh*h_{t-1} + b and applies
         sigmoid/tanh.
      3. How the Cell State Memory Conveyor Belt (c_{t-1} -> c_t) forgets a fraction
         via (× f_t) and adds new info via (+ i_t * g_t) as it moves forward day-by-day.
      4. 5 clickable internal sub-steps + Day 1->30 playback & scrubbing.
    """
    n_idx = initial_neuron - 1
    gate_data = analysis["gate_data"]
    mem = analysis["memory_decomposition"]
    pre = analysis["pre_activations"]
    feats = analysis["temporal_features"]

    # Package all 30 days for the selected neuron (plus all 64 neurons' summary)
    days_payload = []
    for t in range(30):
        gates_t = {}
        for g_key in ["forget_gate", "input_gate", "candidate_gate", "output_gate"]:
            f_contribs = [
                round(float(pre[g_key]["feature_contributions"][t, n_idx, f_i]), 4)
                for f_i in range(len(feats))
            ]
            gates_t[g_key] = {
                "in_drive": round(float(pre[g_key]["input_drive"][t, n_idx]), 4),
                "rec_drive": round(float(pre[g_key]["recurrent_drive"][t, n_idx]), 4),
                "bias_drive": round(float(pre[g_key]["bias_drive"][t, n_idx]), 4),
                "z": round(float(pre[g_key]["pre_activation"][t, n_idx]), 4),
                "val": round(float(gate_data[g_key][t, n_idx]), 4),
                "feat_contribs": f_contribs,
            }

        days_payload.append(
            {
                "day": t + 1,
                "date": analysis["dates"][t],
                "temp_max": round(float(analysis["raw_features_30d"][t, 0]), 2),
                "raw_x": [
                    round(float(v), 2) for v in analysis["raw_features_30d"][t]
                ],
                "scaled_x": [
                    round(float(v), 3) for v in analysis["scaled_features_30d"][t]
                ],
                "h_prev": (
                    0.0
                    if t == 0
                    else round(float(gate_data["hidden_state"][t - 1, n_idx]), 4)
                ),
                "c_prev": round(float(mem["c_prev"][t, n_idx]), 4),
                "retained": round(float(mem["retained_memory"][t, n_idx]), 4),
                "written": round(float(mem["new_memory_written"][t, n_idx]), 4),
                "c_t": round(float(mem["c_current"][t, n_idx]), 4),
                "tanh_c": round(float(mem["tanh_c"][t, n_idx]), 4),
                "h_t": round(float(mem["h_current"][t, n_idx]), 4),
                "gates": gates_t,
                "survival": [
                    round(float(v), 4)
                    for v in analysis["memory_survival_horizon"][t, : t + 1, n_idx]
                ],
            }
        )

    payload = {
        "model_name": analysis["model_name"],
        "city": analysis["city"],
        "neuron_number": initial_neuron,
        "initial_day": initial_day,
        "role": analysis["neuron_profiles"][n_idx]["role"],
        "features": [
            f.replace("temperature_2m_", "temp_")
            .replace("relative_humidity_2m_mean", "humidity")
            .replace("wind_speed_10m_max", "wind_max")
            .replace("precipitation_log", "precip_log")
            for f in feats
        ],
        "days": days_payload,
    }
    payload_json = json.dumps(payload)

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  body {
    background: #0B0F19;
    color: #F8FAFC;
    padding: 8px 10px;
    user-select: none;
    overflow: hidden;
  }
  /* Top Control & Sub-Step Bar */
  .top-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    background: #111827;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 7px 12px;
    margin-bottom: 8px;
  }
  .bar-group {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  .btn {
    background: #1E293B;
    color: #E2E8F0;
    border: 1px solid #475569;
    border-radius: 7px;
    padding: 5px 10px;
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.14s;
  }
  .btn:hover {
    background: #334155;
    border-color: #94A3B8;
  }
  .btn-play {
    background: linear-gradient(135deg, #2563EB 0%, #4F46E5 100%);
    border-color: #60A5FA;
    color: #FFF;
  }
  .btn-play.playing {
    background: linear-gradient(135deg, #D97706 0%, #B45309 100%);
    border-color: #FBBF24;
  }
  .step-pill {
    background: #0F172A;
    color: #94A3B8;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.14s;
  }
  .step-pill:hover {
    border-color: #60A5FA;
    color: #F8FAFC;
  }
  .step-pill.active-step {
    background: #451A03;
    border-color: #F59E0B;
    color: #FDE68A;
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.35);
  }
  .day-strip {
    display: grid;
    grid-template-columns: repeat(30, 1fr);
    gap: 2px;
    background: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 7px;
    padding: 4px 8px;
    margin-bottom: 8px;
  }
  .d-dot {
    text-align: center;
    font-family: Consolas, monospace;
    font-size: 10px;
    padding: 3px 0;
    border-radius: 4px;
    cursor: pointer;
    color: #94A3B8;
    background: #162033;
    transition: all 0.12s;
  }
  .d-dot:hover {
    background: #334155;
    color: #FFF;
  }
  .d-dot.active {
    background: #F59E0B;
    color: #090D16;
    font-weight: 800;
  }
  .d-dot.passed {
    border-bottom: 2px solid #38BDF8;
  }

  /* Main Side-by-Side Workspace (Circuit Left, Gate Calculation Right) */
  .main-grid {
    display: grid;
    grid-template-columns: 1.48fr 1fr;
    gap: 10px;
    height: 410px;
  }
  .circuit-card, .calc-card {
    background: radial-gradient(circle at 50% 35%, #151F34 0%, #0D1322 100%);
    border: 1px solid #26354D;
    border-radius: 12px;
    padding: 10px 12px;
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .story-banner {
    background: rgba(30, 27, 75, 0.75);
    border: 1px solid #6366F1;
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 11.5px;
    color: #E0E7FF;
    line-height: 1.38;
  }
  /* SVG Circuit Styling */
  .pipe {
    fill: none;
    stroke-linecap: round;
    transition: stroke 0.2s, stroke-width 0.2s, stroke-opacity 0.2s;
  }
  .pipe-flow {
    fill: none;
    stroke-dasharray: 6 8;
    animation: dashMove 0.9s linear infinite;
  }
  @keyframes dashMove {
    to { stroke-dashoffset: -28; }
  }
  .gate-node {
    cursor: pointer;
    transition: transform 0.15s;
  }
  .gate-node:hover {
    filter: brightness(1.18);
  }
  /* Right Side: Gate Calculation & Memory Strip */
  .calc-title {
    font-size: 12.5px;
    font-weight: 700;
    color: #F8FAFC;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 5px;
    margin-bottom: 6px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .drive-row {
    margin-bottom: 5px;
  }
  .drive-lbl {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-family: Consolas, monospace;
    margin-bottom: 2px;
  }
  .drive-bar-bg {
    height: 7px;
    background: #090D16;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid #1E293B;
    position: relative;
  }
  .drive-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.2s ease;
  }
  .formula-pill {
    background: #090D16;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px 8px;
    font-family: Consolas, monospace;
    font-size: 11px;
    line-height: 1.45;
    margin: 5px 0;
  }
  .mem-balance-box {
    background: #090D16;
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 7px 9px;
  }
  .horizon-mini-bars {
    display: flex;
    align-items: flex-end;
    gap: 2px;
    height: 44px;
    margin-top: 5px;
    padding-top: 4px;
    border-bottom: 1px solid #334155;
  }
  .h-bar {
    flex: 1;
    border-radius: 2px 2px 0 0;
    min-height: 2px;
    transition: height 0.15s;
  }
</style>
</head>
<body>

<!-- 1. TOP PLAYBACK & 5-STAGE SUB-STEP BAR -->
<div class="top-bar">
  <div class="bar-group">
    <button id="btn-play" class="btn btn-play" onclick="togglePlayDays()">▶ Play 30 Days</button>
    <button class="btn" onclick="stepDay(-1)">⏮</button>
    <button class="btn" onclick="stepDay(1)">⏭</button>
    <span id="day-badge" style="font-family:Consolas,monospace;font-size:12px;font-weight:700;color:#FBBF24;padding:0 6px;"></span>
  </div>

  <div class="bar-group">
    <span style="font-size:11px;color:#94A3B8;">Trace Data Inside Cell:</span>
    <span class="step-pill active-step" id="sp-0" onclick="setSubStep(0)">All Active</span>
    <span class="step-pill" id="sp-1" onclick="setSubStep(1)">1. Inputs Arrive</span>
    <span class="step-pill" id="sp-2" onclick="setSubStep(2)">2. Forget Gate (×)</span>
    <span class="step-pill" id="sp-3" onclick="setSubStep(3)">3. Input × Cand (+)</span>
    <span class="step-pill" id="sp-4" onclick="setSubStep(4)">4. Cell c_t Updates</span>
    <span class="step-pill" id="sp-5" onclick="setSubStep(5)">5. Output h_t Emits</span>
  </div>
</div>

<!-- 2. 30-DAY CLICKABLE TIMELINE STRIP -->
<div id="day-strip" class="day-strip"></div>

<!-- 3. SIDE-BY-SIDE INTERACTIVE CELL CIRCUIT + GATE CALCULATION INSPECTOR -->
<div class="main-grid">
  <!-- LEFT: ANIMATED SINGLE-CELL CIRCUIT DIAGRAM -->
  <div class="circuit-card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
      <span style="font-size:12.5px;font-weight:700;color:#F8FAFC;" id="circuit-header"></span>
      <span style="font-size:11px;color:#38BDF8;font-family:Consolas,monospace;" id="role-badge"></span>
    </div>

    <svg id="cell-svg" viewBox="0 0 680 310" width="100%" height="315"></svg>

    <div id="story-banner" class="story-banner"></div>
  </div>

  <!-- RIGHT: HOW THE SELECTED GATE CALCULATES ITS VALUE & HOW MEMORY MOVES FORWARD -->
  <div class="calc-card">
    <div>
      <div class="calc-title">
        <span id="gate-calc-title">🔬 How Forget Gate (f_t) Calculated Its Value</span>
        <span style="font-size:10.5px;color:#94A3B8;">Click any gate on circuit</span>
      </div>
      <div id="gate-calc-body"></div>
    </div>

    <div class="mem-balance-box">
      <div style="display:flex;justify-content:space-between;font-size:11.5px;font-weight:700;color:#FDE68A;margin-bottom:4px;">
        <span>🧠 How LSTM Remembers Moving Forward to Day <span id="mem-day-lbl">30</span></span>
        <span id="mem-c-lbl" style="font-family:Consolas,monospace;"></span>
      </div>
      <div id="mem-equation-line" style="font-family:Consolas,monospace;font-size:11px;color:#E2E8F0;margin-bottom:4px;"></div>
      <div style="font-size:10px;color:#94A3B8;">
        Surviving memory packets from Day 1 → Day <span id="hor-day-lbl">30</span> still alive inside today's Cell State (c_t):
      </div>
      <div id="horizon-mini-bars" class="horizon-mini-bars"></div>
      <div style="display:flex;justify-content:space-between;font-size:9.5px;color:#64748B;font-family:Consolas,monospace;margin-top:2px;">
        <span>Day 1</span>
        <span>Earlier days retained via f_t products</span>
        <span id="hor-end-lbl">Day 30 (Today)</span>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __PAYLOAD_JSON__;

let stState = {
  dayIdx: (DATA.initial_day || 30) - 1,
  subStep: 0, // 0=All, 1=Inputs, 2=Forget, 3=Write, 4=CellUpdate, 5=Output
  selectedGate: "forget_gate",
  playing: false,
  timer: null,
};

function fmtSigned(v, digits = 4) {
  return (v >= 0 ? "+" : "") + v.toFixed(digits);
}

function setSubStep(s) {
  stState.subStep = s;
  for (let i = 0; i <= 5; i++) {
    const el = document.getElementById(`sp-${i}`);
    if (i === s) el.classList.add("active-step");
    else el.classList.remove("active-step");
  }
  renderCircuit();
}

function stepDay(delta) {
  stState.dayIdx = (stState.dayIdx + delta + 30) % 30;
  renderCircuit();
}

function togglePlayDays() {
  stState.playing = !stState.playing;
  const btn = document.getElementById("btn-play");
  if (stState.playing) {
    btn.textContent = "⏸ Pause";
    btn.classList.add("playing");
    if (stState.dayIdx >= 29) stState.dayIdx = 0;
    stState.timer = setInterval(() => {
      if (stState.dayIdx < 29) {
        stState.dayIdx++;
        renderCircuit();
      } else {
        togglePlayDays();
      }
    }, 420);
  } else {
    btn.textContent = "▶ Play 30 Days";
    btn.classList.remove("playing");
    if (stState.timer) clearInterval(stState.timer);
    stState.timer = null;
  }
}

function selectGateBox(gKey) {
  stState.selectedGate = gKey;
  renderCircuit();
}

function renderDayStrip() {
  const strip = document.getElementById("day-strip");
  strip.innerHTML = "";
  for (let i = 0; i < 30; i++) {
    const d = document.createElement("div");
    d.className = "d-dot"
      + (i === stState.dayIdx ? " active" : "")
      + (i < stState.dayIdx ? " passed" : "");
    d.textContent = i + 1;
    d.title = `Day ${i + 1} (${DATA.days[i].date}) · Max Temp: ${DATA.days[i].temp_max} °C`;
    d.onclick = () => {
      stState.dayIdx = i;
      renderCircuit();
    };
    strip.appendChild(d);
  }
}

function renderCircuit() {
  renderDayStrip();
  const d = DATA.days[stState.dayIdx];
  const s = stState.subStep;

  document.getElementById("day-badge").textContent =
    `Day ${d.day}/30 (${d.date}) · ${d.temp_max.toFixed(1)} °C`;
  document.getElementById("circuit-header").textContent =
    `Inside LSTM Neuron #${DATA.neuron_number} on Day ${d.day} (${DATA.city} · ${DATA.model_name})`;
  document.getElementById("role-badge").textContent = `Role: ${DATA.role}`;

  // Determine which pathways are highlighted by subStep (0 = all active)
  const hiIn = (s === 0 || s === 1);
  const hiForget = (s === 0 || s === 2);
  const hiWrite = (s === 0 || s === 3);
  const hiCell = (s === 0 || s === 4);
  const hiOut = (s === 0 || s === 5);

  const op = (flag) => flag ? "0.95" : "0.22";
  const fl = (flag) => flag ? "pipe-flow" : "";

  const svg = document.getElementById("cell-svg");
  svg.innerHTML = `
    <!-- Background Cell Enclosure -->
    <rect x="95" y="14" width="490" height="282" rx="16" fill="#0F172A" stroke="#334155" stroke-width="1.8" />
    <text x="112" y="34" fill="#64748B" font-size="10.5" font-weight="700">LONG-TERM CELL STATE MEMORY CONVEYOR BELT (c_{t-1} → c_t)</text>

    <!-- TOP CELL MEMORY CONVEYOR PIPES -->
    <!-- c_{t-1} incoming pipe -->
    <line x1="18" y1="62" x2="188" y2="62" stroke="#F59E0B" stroke-width="4" stroke-opacity="${op(hiForget || hiCell)}" class="pipe ${fl(hiForget)}" />
    <!-- After Forget (×) to Adder (+) -->
    <line x1="224" y1="62" x2="352" y2="62" stroke="#F59E0B" stroke-width="4" stroke-opacity="${op(hiForget || hiCell)}" class="pipe ${fl(hiCell)}" />
    <!-- After Adder (+) to c_t output -->
    <line x1="388" y1="62" x2="662" y2="62" stroke="#FBBF24" stroke-width="4.5" stroke-opacity="${op(hiCell || hiOut)}" class="pipe ${fl(hiCell)}" />

    <!-- Branch from c_t down to tanh() for Output Gate -->
    <path d="M 495 62 L 495 112" stroke="#FBBF24" stroke-width="3" stroke-opacity="${op(hiOut)}" class="pipe ${fl(hiOut)}" />

    <!-- BOTTOM INPUT PIPES (h_{t-1} and x_t entering the 4 gates) -->
    <path d="M 18 245 L 520 245" stroke="#38BDF8" stroke-width="3.2" stroke-opacity="${op(hiIn)}" class="pipe ${fl(hiIn)}" />
    <!-- Vertical feeders into the 4 gates -->
    <line x1="206" y1="245" x2="206" y2="210" stroke="#C084FC" stroke-width="2.8" stroke-opacity="${op(hiIn || hiForget)}" class="pipe ${fl(hiForget)}" />
    <line x1="312" y1="245" x2="312" y2="210" stroke="#38BDF8" stroke-width="2.8" stroke-opacity="${op(hiIn || hiWrite)}" class="pipe ${fl(hiWrite)}" />
    <line x1="418" y1="245" x2="418" y2="210" stroke="#34D399" stroke-width="2.8" stroke-opacity="${op(hiIn || hiWrite)}" class="pipe ${fl(hiWrite)}" />
    <line x1="520" y1="245" x2="520" y2="210" stroke="#FB7185" stroke-width="2.8" stroke-opacity="${op(hiIn || hiOut)}" class="pipe ${fl(hiOut)}" />

    <!-- Gate Output Pipes up to Valves -->
    <!-- Forget gate f_t up to (×) valve -->
    <line x1="206" y1="152" x2="206" y2="80" stroke="#C084FC" stroke-width="3" stroke-opacity="${op(hiForget)}" class="pipe ${fl(hiForget)}" />
    <!-- Input gate i_t & Candidate gate g_t into Write (×) valve at (370, 118) -->
    <path d="M 312 152 L 312 118 L 352 118" stroke="#38BDF8" stroke-width="2.8" stroke-opacity="${op(hiWrite)}" class="pipe ${fl(hiWrite)}" />
    <path d="M 418 152 L 418 118 L 388 118" stroke="#34D399" stroke-width="2.8" stroke-opacity="${op(hiWrite)}" class="pipe ${fl(hiWrite)}" />
    <!-- Write (×) valve up to Cell Adder (+) at (370, 62) -->
    <line x1="370" y1="100" x2="370" y2="80" stroke="#34D399" stroke-width="3.2" stroke-opacity="${op(hiWrite || hiCell)}" class="pipe ${fl(hiWrite || hiCell)}" />

    <!-- Output gate o_t and tanh(c_t) into Readout (×) valve at (520, 118) -->
    <line x1="520" y1="152" x2="520" y2="136" stroke="#FB7185" stroke-width="2.8" stroke-opacity="${op(hiOut)}" class="pipe ${fl(hiOut)}" />
    <!-- Readout (×) valve out to h_t at right -->
    <line x1="538" y1="118" x2="662" y2="118" stroke="#818CF8" stroke-width="4" stroke-opacity="${op(hiOut)}" class="pipe ${fl(hiOut)}" />

    <!-- LEFT INPUT BADGES: c_{t-1}, h_{t-1}, x_t -->
    <g>
      <rect x="6" y="42" width="84" height="38" rx="7" fill="#1E1B4B" stroke="#F59E0B" stroke-width="1.5" />
      <text x="48" y="57" text-anchor="middle" fill="#FDE68A" font-size="10" font-weight="700">Old Cell c_{t-1}</text>
      <text x="48" y="72" text-anchor="middle" fill="#FFF" font-family="Consolas" font-size="11.5" font-weight="700">${fmtSigned(d.c_prev)}</text>
    </g>
    <g>
      <rect x="6" y="210" width="84" height="72" rx="7" fill="#162033" stroke="#38BDF8" stroke-width="1.5" />
      <text x="48" y="226" text-anchor="middle" fill="#93C5FD" font-size="9.5" font-weight="700">Inputs (Day ${d.day})</text>
      <text x="48" y="242" text-anchor="middle" fill="#E2E8F0" font-family="Consolas" font-size="10">h_{t-1}: ${fmtSigned(d.h_prev, 3)}</text>
      <text x="48" y="257" text-anchor="middle" fill="#38BDF8" font-family="Consolas" font-size="10">Temp: ${d.temp_max}°C</text>
      <text x="48" y="272" text-anchor="middle" fill="#94A3B8" font-family="Consolas" font-size="9.5">x_t[0]: ${d.scaled_x[0].toFixed(2)}</text>
    </g>

    <!-- VALVE 1: FORGET MULTIPLIER (×) ON CONVEYOR -->
    <g>
      <circle cx="206" cy="62" r="17" fill="#3B0764" stroke="#C084FC" stroke-width="2" />
      <text x="206" y="67" text-anchor="middle" fill="#FFF" font-size="15" font-weight="800">×</text>
      <rect x="232" y="38" width="106" height="20" rx="4" fill="#1E1B4B" stroke="#475569" />
      <text x="285" y="52" text-anchor="middle" fill="#E9D5FF" font-family="Consolas" font-size="10.5" font-weight="700">Kept: ${fmtSigned(d.retained)}</text>
    </g>

    <!-- VALVE 2: CANDIDATE WRITE MULTIPLIER (i_t × g_t) -->
    <g>
      <circle cx="370" cy="118" r="16" fill="#064E3B" stroke="#34D399" stroke-width="2" />
      <text x="370" y="123" text-anchor="middle" fill="#FFF" font-size="14" font-weight="800">×</text>
      <rect x="262" y="84" width="96" height="19" rx="4" fill="#064E3B" stroke="#34D399" />
      <text x="310" y="97" text-anchor="middle" fill="#A7F3D0" font-family="Consolas" font-size="10" font-weight="700">Write: ${fmtSigned(d.written)}</text>
    </g>

    <!-- VALVE 3: CELL MEMORY ADDER (+) ON CONVEYOR -->
    <g>
      <circle cx="370" cy="62" r="17" fill="#451A03" stroke="#FBBF24" stroke-width="2.2" />
      <text x="370" y="67" text-anchor="middle" fill="#FFF" font-size="16" font-weight="800">+</text>
    </g>

    <!-- VALVE 4: TANH(c_t) & OUTPUT MULTIPLIER (×) -->
    <g>
      <rect x="466" y="90" width="44" height="20" rx="5" fill="#451A03" stroke="#F59E0B" />
      <text x="488" y="104" text-anchor="middle" fill="#FDE68A" font-family="Consolas" font-size="10" font-weight="700">tanh</text>
      <circle cx="520" cy="118" r="16" fill="#31102F" stroke="#FB7185" stroke-width="2" />
      <text x="520" y="123" text-anchor="middle" fill="#FFF" font-size="14" font-weight="800">×</text>
    </g>

    <!-- THE 4 CLICKABLE GATE BOXES -->
    ${renderSvgGateBox("forget_gate", 160, 152, "f_t · Forget", "σ", d.gates.forget_gate.val, "#C084FC", "#3B0764", true)}
    ${renderSvgGateBox("input_gate", 266, 152, "i_t · Input", "σ", d.gates.input_gate.val, "#38BDF8", "#0C4A6E", true)}
    ${renderSvgGateBox("candidate_gate", 372, 152, "g_t · Cand", "tanh", d.gates.candidate_gate.val, "#34D399", "#064E3B", false)}
    ${renderSvgGateBox("output_gate", 476, 152, "o_t · Output", "σ", d.gates.output_gate.val, "#FB7185", "#4C0519", true)}

    <!-- RIGHT OUTPUT BADGES: New c_t and New h_t moving forward to Day t+1 -->
    <g>
      <rect x="590" y="42" width="84" height="38" rx="7" fill="#451A03" stroke="#FBBF24" stroke-width="1.8" />
      <text x="632" y="57" text-anchor="middle" fill="#FDE68A" font-size="10" font-weight="700">New Cell c_t</text>
      <text x="632" y="72" text-anchor="middle" fill="#FFF" font-family="Consolas" font-size="11.5" font-weight="700">${fmtSigned(d.c_t)}</text>
    </g>
    <g>
      <rect x="590" y="99" width="84" height="38" rx="7" fill="#1E1B4B" stroke="#818CF8" stroke-width="1.8" />
      <text x="632" y="114" text-anchor="middle" fill="#C7D2FE" font-size="10" font-weight="700">New Hidden h_t</text>
      <text x="632" y="129" text-anchor="middle" fill="#FFF" font-family="Consolas" font-size="11.5" font-weight="700">${fmtSigned(d.h_t)}</text>
    </g>
  `;

  // Update plain-English story banner at bottom of circuit card
  const banner = document.getElementById("story-banner");
  const stories = [
    `<strong>Full Day-${d.day} Summary:</strong> Old memory <code>c_${d.day-1} = ${fmtSigned(d.c_prev)}</code> is multiplied by <strong>Forget Gate (${(d.gates.forget_gate.val*100).toFixed(1)}%)</strong> → <code>${fmtSigned(d.retained)}</code> kept, then added to <strong>Today's Write (${fmtSigned(d.written)})</strong> → <strong>New Cell Memory <code>c_${d.day} = ${fmtSigned(d.c_t)}</code></strong> & <strong>Hidden <code>h_${d.day} = ${fmtSigned(d.h_t)}</code></strong>.`,
    `<strong>Step 1 — Inputs Arrive:</strong> Day ${d.day}'s weather features (Max Temp <code>${d.temp_max}°C</code>, scaled <code>${d.scaled_x[0]}</code>) and yesterday's working summary <code>h_${d.day-1} = ${fmtSigned(d.h_prev)}</code> flow into all 4 gates simultaneously.`,
    `<strong>Step 2 — Forget Gate Filters Old Memory:</strong> Forget Gate computes <code>f_${d.day} = σ(${fmtSigned(d.gates.forget_gate.z, 3)}) = ${d.gates.forget_gate.val.toFixed(4)}</code> and multiplies yesterday's cell state: <code>${d.gates.forget_gate.val.toFixed(3)} × ${fmtSigned(d.c_prev, 3)} = ${fmtSigned(d.retained)}</code>.`,
    `<strong>Step 3 — Input & Candidate Gates Create New Memory:</strong> Input Gate (<code>${d.gates.input_gate.val.toFixed(3)}</code>) × Candidate Note (<code>${fmtSigned(d.gates.candidate_gate.val, 3)}</code>) produces today's new memory packet: <code>${fmtSigned(d.written)}</code>.`,
    `<strong>Step 4 — Cell Conveyor Updates & Carries Forward:</strong> The conveyor adds Kept Memory + New Write: <code>${fmtSigned(d.retained)} + (${fmtSigned(d.written)}) = ${fmtSigned(d.c_t)}</code>. This updated memory <code>c_${d.day}</code> travels straight to Day ${d.day + 1}!`,
    `<strong>Step 5 — Output Gate Emits Hidden State:</strong> Cell memory is squashed via <code>tanh(${fmtSigned(d.c_t, 3)}) = ${fmtSigned(d.tanh_c, 3)}</code> and filtered by Output Gate (<code>${d.gates.output_gate.val.toFixed(3)}</code>) to emit <code>h_${d.day} = ${fmtSigned(d.h_t)}</code>.`
  ];
  banner.innerHTML = stories[s];

  renderRightInspector(d);
}

function renderSvgGateBox(gKey, x, y, title, fnName, val, strokeCol, fillCol, is01) {
  const isSel = stState.selectedGate === gKey;
  const pct = is01
    ? Math.max(0, Math.min(1, val))
    : Math.max(0, Math.min(1, (val + 1) / 2));
  const barW = Math.round(pct * 76);
  const valStr = is01 ? val.toFixed(4) : fmtSigned(val, 4);

  return `
    <g class="gate-node" onclick="selectGateBox('${gKey}')">
      <rect x="${x}" y="${y}" width="92" height="58" rx="8"
        fill="${fillCol}" stroke="${isSel ? '#FBBF24' : strokeCol}" stroke-width="${isSel ? '2.8' : '1.6'}" />
      <text x="${x + 46}" y="${y + 16}" text-anchor="middle" fill="#F8FAFC" font-size="10.5" font-weight="700">${title} (${fnName})</text>
      <rect x="${x + 8}" y="${y + 23}" width="76" height="7" rx="3.5" fill="#090D16" />
      <rect x="${x + 8}" y="${y + 23}" width="${barW}" height="7" rx="3.5" fill="${strokeCol}" />
      <text x="${x + 46}" y="${y + 47}" text-anchor="middle" fill="#FFF" font-family="Consolas" font-size="12" font-weight="800">${valStr}</text>
    </g>
  `;
}

function renderRightInspector(d) {
  const gKey = stState.selectedGate;
  const gInfo = d.gates[gKey];
  const metaMap = {
    forget_gate: { name: "Forget Gate (f_t)", fn: "σ", col: "#C084FC" },
    input_gate: { name: "Input Gate (i_t)", fn: "σ", col: "#38BDF8" },
    candidate_gate: { name: "Candidate Gate (g_t)", fn: "tanh", col: "#34D399" },
    output_gate: { name: "Output Gate (o_t)", fn: "σ", col: "#FB7185" },
  };
  const m = metaMap[gKey];
  document.getElementById("gate-calc-title").innerHTML =
    `<span style="color:${m.col};">🔬 How ${m.name} Calculated Its Value (Day ${d.day})</span>`;

  const maxAbs = Math.max(1.5, Math.abs(gInfo.in_drive), Math.abs(gInfo.rec_drive), Math.abs(gInfo.bias_drive), Math.abs(gInfo.z));
  const toW = (v) => Math.min(100, Math.round((Math.abs(v) / maxAbs) * 100));

  const body = document.getElementById("gate-calc-body");
  body.innerHTML = `
    <div class="drive-row">
      <div class="drive-lbl">
        <span style="color:#38BDF8;">1. From Today's Weather (W_ih · x_t):</span>
        <span style="color:#F8FAFC;font-weight:700;">${fmtSigned(gInfo.in_drive)}</span>
      </div>
      <div class="drive-bar-bg">
        <div class="drive-bar-fill" style="width:${toW(gInfo.in_drive)}%;background:${gInfo.in_drive >= 0 ? '#38BDF8' : '#F59E0B'};"></div>
      </div>
    </div>

    <div class="drive-row">
      <div class="drive-lbl">
        <span style="color:#C084FC;">2. From Yesterday's Hidden (W_hh · h_{t-1}):</span>
        <span style="color:#F8FAFC;font-weight:700;">${fmtSigned(gInfo.rec_drive)}</span>
      </div>
      <div class="drive-bar-bg">
        <div class="drive-bar-fill" style="width:${toW(gInfo.rec_drive)}%;background:${gInfo.rec_drive >= 0 ? '#C084FC' : '#F59E0B'};"></div>
      </div>
    </div>

    <div class="drive-row">
      <div class="drive-lbl">
        <span style="color:#94A3B8;">3. Plus Learned Gate Bias (b_ih + b_hh):</span>
        <span style="color:#F8FAFC;font-weight:700;">${fmtSigned(gInfo.bias_drive)}</span>
      </div>
      <div class="drive-bar-bg">
        <div class="drive-bar-fill" style="width:${toW(gInfo.bias_drive)}%;background:#64748B;"></div>
      </div>
    </div>

    <div class="formula-pill">
      <div>Pre-activation Sum: <code>z_t = ${fmtSigned(gInfo.in_drive, 3)} + (${fmtSigned(gInfo.rec_drive, 3)}) + (${fmtSigned(gInfo.bias_drive, 3)}) = <strong>${fmtSigned(gInfo.z, 4)}</strong></code></div>
      <div style="color:${m.col};margin-top:2px;">Gate Output: <code>${m.fn}(${fmtSigned(gInfo.z, 4)}) = <strong>${gInfo.val.toFixed(4)}</strong></code></div>
    </div>
  `;

  // Render Memory Carry-Forward & Survival Horizon Mini-Bars
  document.getElementById("mem-day-lbl").textContent = d.day;
  document.getElementById("hor-day-lbl").textContent = d.day;
  document.getElementById("hor-end-lbl").textContent = `Day ${d.day} (Today)`;
  document.getElementById("mem-c-lbl").textContent = `c_${d.day} = ${fmtSigned(d.c_t)}`;
  document.getElementById("mem-equation-line").innerHTML =
    `<span style="color:#C084FC;">Kept (${fmtSigned(d.retained)})</span> + <span style="color:#34D399;">Written Today (${fmtSigned(d.written)})</span> = <strong style="color:#FBBF24;">${fmtSigned(d.c_t)}</strong>`;

  const hContainer = document.getElementById("horizon-mini-bars");
  hContainer.innerHTML = "";
  const surv = d.survival;
  const maxS = Math.max(0.05, ...surv.map(v => Math.abs(v)));
  surv.forEach((sv, idx) => {
    const bar = document.createElement("div");
    bar.className = "h-bar";
    const pct = Math.max(6, Math.round((Math.abs(sv) / maxS) * 100));
    bar.style.height = `${pct}%`;
    bar.style.background = idx === surv.length - 1
      ? "#34D399"
      : (sv >= 0 ? "#38BDF8" : "#F59E0B");
    bar.title = `Written on Day ${idx + 1}: surviving contribution in Day ${d.day}'s cell state = ${fmtSigned(sv, 4)}`;
    hContainer.appendChild(bar);
  });
}

renderCircuit();
</script>
</body>
</html>
""".replace("__PAYLOAD_JSON__", payload_json)
    return html


def build_compact_memory_carryforward_chart(
    analysis: Dict[str, Any],
    neuron_number: int,
    selected_day: int,
) -> go.Figure:
    """
    Chart 1 (Compact 320px): Shows HOW the LSTM remembers moving forward from
    Day 1 to Day 30 by stacking:
      - Memory Kept from Yesterday: f_t * c_{t-1}
      - New Memory Written Today: i_t * g_t
      - Total Cell State: c_t
      - Exposed Hidden State: h_t
    """
    n = neuron_number - 1
    days = list(range(1, 31))
    dates = analysis["dates"]
    mem = analysis["memory_decomposition"]

    kept = mem["retained_memory"][:, n]
    written = mem["new_memory_written"][:, n]
    c_t = mem["c_current"][:, n]
    h_t = mem["h_current"][:, n]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=days,
            y=kept,
            name="Kept from Yesterday (f_t × c_{t-1})",
            marker_color="#818CF8",
            opacity=0.82,
            customdata=dates,
            hovertemplate="Day %{x} (%{customdata})<br>Kept Old Memory: %{y:+.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=days,
            y=written,
            name="Written Today (i_t × g_t)",
            marker_color="#34D399",
            opacity=0.88,
            customdata=dates,
            hovertemplate="Day %{x} (%{customdata})<br>New Memory Written: %{y:+.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=c_t,
            mode="lines+markers",
            name="Cell Memory (c_t = Kept + Written)",
            line=dict(color="#FBBF24", width=2.8),
            marker=dict(size=5),
            hovertemplate="Day %{x}<br>Total Cell State c_t: %{y:+.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=h_t,
            mode="lines",
            name="Hidden Output (h_t)",
            line=dict(color="#F472B6", width=2, dash="dot"),
            hovertemplate="Day %{x}<br>Hidden State h_t: %{y:+.4f}<extra></extra>",
        )
    )

    fig.add_vline(
        x=selected_day,
        line_width=1.6,
        line_dash="dash",
        line_color="#F59E0B",
    )

    fig.update_layout(
        barmode="relative",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=320,
        margin=dict(l=48, r=22, t=18, b=76),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(title="Day in 30-Day Window", gridcolor="#1E293B", dtick=2),
        yaxis=dict(title="Memory Value", gridcolor="#1E293B", zerolinecolor="#475569"),
    )
    return fig


def build_compact_weather_vs_gates_chart(
    analysis: Dict[str, Any],
    neuron_number: int,
    selected_day: int,
) -> go.Figure:
    """
    Chart 2 (Compact 320px): Dual-axis interactive chart correlating daily Max
    Temperature (°C) with the 4 LSTM gates (f_t, i_t, g_t, o_t) across Days 1..30.
    """
    n = neuron_number - 1
    days = list(range(1, 31))
    dates = analysis["dates"]
    temps = analysis["raw_features_30d"][:, 0]
    g = analysis["gate_data"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Shaded background temperature trace on secondary axis
    fig.add_trace(
        go.Scatter(
            x=days,
            y=temps,
            name="Max Temp (°C)",
            mode="lines",
            line=dict(color="rgba(251, 191, 36, 0.45)", width=2, dash="dot"),
            fill="tozeroy",
            fillcolor="rgba(251, 191, 36, 0.06)",
            customdata=dates,
            hovertemplate="Day %{x} (%{customdata})<br>Max Temp: %{y:.1f} °C<extra></extra>",
        ),
        secondary_y=True,
    )

    gate_Meta = [
        ("forget_gate", "Forget (f_t)", "#C084FC"),
        ("input_gate", "Input (i_t)", "#38BDF8"),
        ("candidate_gate", "Candidate (g_t)", "#34D399"),
        ("output_gate", "Output (o_t)", "#FB7185"),
    ]
    for g_key, g_lbl, col in gate_Meta:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=g[g_key][:, n],
                name=g_lbl,
                mode="lines+markers",
                line=dict(color=col, width=2.2),
                marker=dict(size=4),
                hovertemplate=f"{g_lbl}: %{{y:.4f}}<extra></extra>",
            ),
            secondary_y=False,
        )

    fig.add_vline(
        x=selected_day,
        line_width=1.6,
        line_dash="dash",
        line_color="#F59E0B",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=320,
        margin=dict(l=48, r=48, t=18, b=76),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.24,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(title="Day in 30-Day Window", gridcolor="#1E293B", dtick=2),
    )
    fig.update_yaxes(
        title_text="Gate Activation",
        gridcolor="#1E293B",
        range=[-1.08, 1.08],
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Temp (°C)",
        showgrid=False,
        secondary_y=True,
    )
    return fig


def build_compact_gate_drive_chart(
    story: Dict[str, Any],
) -> go.Figure:
    """
    Chart 3 (Compact 320px): Grouped bar chart showing HOW all 4 gates calculated
    their pre-activation z_t on the selected day from:
      - Today's Weather Input (W_ih * x_t)
      - Yesterday's Hidden State (W_hh * h_{t-1})
      - Gate Bias (b_ih + b_hh)
    """
    gb = story["gate_breakdown"]
    gate_keys = ["forget_gate", "input_gate", "candidate_gate", "output_gate"]
    labels = ["Forget (f_t)", "Input (i_t)", "Candidate (g_t)", "Output (o_t)"]

    in_vals = [gb[k]["input_drive"] for k in gate_keys]
    rec_vals = [gb[k]["recurrent_drive"] for k in gate_keys]
    bias_vals = [gb[k]["bias_drive"] for k in gate_keys]
    z_vals = [gb[k]["pre_activation_z"] for k in gate_keys]

    all_y = in_vals + rec_vals + bias_vals + z_vals + [0.0]
    y_min = min(all_y)
    y_max = max(all_y)
    y_span = max(1.0, y_max - y_min)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=in_vals,
            name="Today's Weather (W_ih · x_t)",
            marker_color="#38BDF8",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=rec_vals,
            name="Yesterday's Hidden (W_hh · h_{t-1})",
            marker_color="#C084FC",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=bias_vals,
            name="Learned Bias (b)",
            marker_color="#64748B",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=z_vals,
            mode="markers+text",
            name="Total Pre-Activation (z_t)",
            marker=dict(color="#FBBF24", size=10, symbol="diamond"),
            text=[f"z={v:+.2f}" for v in z_vals],
            textposition="top center",
            textfont=dict(size=10, color="#FDE68A"),
            cliponaxis=False,
        )
    )

    fig.update_layout(
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=320,
        margin=dict(l=48, r=22, t=20, b=76),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5, color="#E2E8F0"),
        ),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(
            title="Pre-Activation Drive",
            gridcolor="#1E293B",
            zerolinecolor="#475569",
            range=[y_min - 0.22 * y_span, y_max + 0.28 * y_span],
        ),
    )
    return fig


def build_compact_memory_horizon_or_heatmap_chart(
    analysis: Dict[str, Any],
    neuron_number: int,
    selected_day: int,
    mode_choice: str,
) -> go.Figure:
    """
    Chart 4 (Compact 300px): Either shows the exact Memory Survival Horizon for
    Day `selected_day` (how much of each earlier day k <= t is still stored inside
    c_t) OR a compact 64-neuron heatmap sorted by memory retention.
    """
    n = neuron_number - 1
    t = selected_day - 1

    if mode_choice.startswith("Memory Survival"):
        surv = analysis["memory_survival_horizon"][t, : t + 1, n]
        past_days = list(range(1, selected_day + 1))
        past_dates = analysis["dates"][:selected_day]
        colors = [
            "#34D399" if d_i == selected_day else ("#38BDF8" if v >= 0 else "#F59E0B")
            for d_i, v in zip(past_days, surv)
        ]

        fig = go.Figure(
            go.Bar(
                x=past_days,
                y=surv,
                marker_color=colors,
                customdata=past_dates,
                hovertemplate=(
                    "Origin: Day %{x} (%{customdata})<br>"
                    f"Surviving amount in Day {selected_day}'s c_t: %{{y:+.4f}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111827",
            plot_bgcolor="#0B0F19",
            height=295,
            margin=dict(l=48, r=22, t=18, b=48),
            xaxis=dict(title="Day When Info Was Originally Written", gridcolor="#1E293B", dtick=2),
            yaxis=dict(title=f"Surviving Share in c_{selected_day}", gridcolor="#1E293B", zerolinecolor="#475569"),
        )
        return fig

    # Otherwise render compact 64-neuron Forget Gate heatmap
    f_mat = analysis["gate_data"]["forget_gate"].T  # (64, 30)
    fig = go.Figure(
        go.Heatmap(
            z=f_mat,
            x=list(range(1, 31)),
            y=list(range(1, 65)),
            colorscale="Plasma",
            zmin=0,
            zmax=1,
            colorbar=dict(title="f_t", thickness=12),
            hovertemplate="Day %{x} · Neuron #%{y}<br>Forget Gate f_t: %{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        height=295,
        margin=dict(l=48, r=22, t=18, b=48),
        xaxis=dict(title="Day (1..30)", dtick=2),
        yaxis=dict(title="Neuron (1..64)", autorange="reversed"),
    )
    return fig


def render_gate_memory_explorer(artifacts: Dict[str, Any]) -> None:
    """
    Main Streamlit entry point for the Gate & Memory Explorer page.
    Designed so the user can effortlessly see data enter every gate, how each
    gate calculates its value, and how the LSTM remembers data while moving
    forward—with compact, side-by-side interactive charts that never overflow
    the full page.
    """
    _init_gate_explorer_state()

    st.subheader("🔬 Interactive LSTM Gate & Memory Explorer")
    st.caption(
        "Watch how daily weather data enters a single LSTM cell, how the **Forget ($f_t$)**, **Input ($i_t$)**, "
        "**Candidate ($g_t$)**, and **Output ($o_t$)** gates calculate their values, and how the **Cell State Memory Conveyor ($c_t$)** "
        "carries information forward from Day 1 to Day 30."
    )

    # ==========================================
    # 1. COMPACT TOP CONTROL BAR
    # ==========================================
    c1, c2, c3, c4, c5 = st.columns([1.05, 1.0, 1.35, 1.15, 0.95])

    model_options = ["baseline", "model_A", "model_B"]
    with c1:
        model_key = st.selectbox(
            "1. Trained Model",
            options=model_options,
            index=model_options.index(st.session_state["gm_model_key"]),
            format_func=lambda k: MODEL_DISPLAY_NAMES[k],
            key="gm_model_select",
        )
        st.session_state["gm_model_key"] = model_key

    cities_sorted = sorted(list(artifacts["city_to_id"].keys()))
    with c2:
        city = st.selectbox(
            "2. City",
            options=cities_sorted,
            index=(
                cities_sorted.index(st.session_state["gm_city"])
                if st.session_state["gm_city"] in cities_sorted
                else 0
            ),
            key="gm_city_select",
        )
        st.session_state["gm_city"] = city

    bounds = get_city_date_bounds(city)
    preset_labels = list(bounds["presets"].keys())
    with c3:
        preset = st.selectbox(
            "3. 30-Day Weather Window",
            options=preset_labels,
            index=(
                preset_labels.index(st.session_state["gm_window_preset"])
                if st.session_state["gm_window_preset"] in preset_labels
                else 0
            ),
            key="gm_preset_select",
        )
        st.session_state["gm_window_preset"] = preset
    active_end_date = bounds["presets"][preset]

    # Run gate & memory flow decomposition for the selected model, city, and window
    analysis = analyze_gate_and_memory_flow(
        model_key=model_key,
        city=city,
        artifacts=artifacts,
        end_date=active_end_date,
    )

    with c4:
        neuron_num = st.slider(
            "4. Select LSTM Neuron (1–64)",
            min_value=1,
            max_value=64,
            value=int(st.session_state["gm_neuron"]),
            step=1,
            key="gm_neuron_slider",
        )
        st.session_state["gm_neuron"] = neuron_num

    with c5:
        day_num = st.slider(
            "5. Focus Day (1–30)",
            min_value=1,
            max_value=30,
            value=int(st.session_state["gm_day"]),
            step=1,
            key="gm_day_slider",
        )
        st.session_state["gm_day"] = day_num

    # Quick Role-Based Neuron Picker Pills
    rankings = analysis["neuron_rankings"]
    top_keep = rankings["long_term_keepers"][:3]
    top_write = rankings["active_daily_writers"][:3]
    top_pred = rankings["top_forecast_drivers"][:3]

    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        st.caption(
            f"🐢 **Top Long-Term Memory Neurons** (High $f_t$): "
            + ", ".join([f"`#{u}`" for u in top_keep])
        )
    with q_col2:
        st.caption(
            f"⚡ **Top Daily Weather Writers** (High $|i_t \\cdot g_t|$): "
            + ", ".join([f"`#{u}`" for u in top_write])
        )
    with q_col3:
        st.caption(
            f"🎯 **Top Forecast Drivers** (High $|h_{{30}} \\cdot W_{{\\text{{fc}}}}|$): "
            + ", ".join([f"`#{u}`" for u in top_pred])
        )

    # ==========================================
    # 2. INTERACTIVE ANIMATED SINGLE-CELL CIRCUIT & SUB-STEP SIMULATOR
    # ==========================================
    circuit_html = build_gate_memory_circuit_html(
        analysis=analysis,
        initial_neuron=neuron_num,
        initial_day=day_num,
    )
    components.html(circuit_html, height=555, scrolling=False)

    # ==========================================
    # 3. NEAT, COMPACT 2×2 INTERACTIVE GRAPHS (NEVER FULL-PAGE)
    # ==========================================
    st.markdown(
        """
        <style>
        .gm-panel-hdr {
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
        .gm-panel-title {
            font-size: 13.5px;
            font-weight: 700;
            color: #F8FAFC;
            letter-spacing: 0.2px;
        }
        .gm-panel-sub {
            font-size: 11px;
            color: #94A3B8;
            font-family: Consolas, monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    story = extract_neuron_gate_story(
        analysis=analysis,
        neuron_number=neuron_num,
        day_number=day_num,
    )

    row1_left, row1_right = st.columns(2, gap="medium")
    with row1_left:
        st.markdown(
            f'<div class="gm-panel-hdr">'
            f'<span class="gm-panel-title">1. How Neuron #{neuron_num} Remembers Moving Forward</span>'
            f'<span class="gm-panel-sub">c_t = Kept + Written</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_memory_carryforward_chart(
                analysis=analysis,
                neuron_number=neuron_num,
                selected_day=day_num,
            ),
            use_container_width=True,
            theme=None,
        )
    with row1_right:
        st.markdown(
            f'<div class="gm-panel-hdr">'
            f'<span class="gm-panel-title">2. How Neuron #{neuron_num}\'s 4 Gates React to Daily Weather</span>'
            f'<span class="gm-panel-sub">Gates vs. Max Temp (°C)</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_weather_vs_gates_chart(
                analysis=analysis,
                neuron_number=neuron_num,
                selected_day=day_num,
            ),
            use_container_width=True,
            theme=None,
        )

    row2_left, row2_right = st.columns(2, gap="medium")
    with row2_left:
        st.markdown(
            f'<div class="gm-panel-hdr">'
            f'<span class="gm-panel-title">3. Inside Gate Math on Day {story["day_number"]} ({story["date"]})</span>'
            f'<span class="gm-panel-sub">Where Each Gate\'s Signal Comes From</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_gate_drive_chart(story=story),
            use_container_width=True,
            theme=None,
        )
    with row2_right:
        chart4_mode = st.radio(
            "Panel 4 View",
            options=[
                "Memory Survival Horizon (Past Days Still in c_t)",
                "All 64 Neurons × 30 Days Forget Gate Map",
            ],
            horizontal=True,
            label_visibility="collapsed",
            key="gm_chart4_radio",
        )
        panel4_title = (
            f"4. Memory Survival Horizon — Past Days Alive in Day {day_num}'s Cell State (c_{day_num})"
            if chart4_mode.startswith("Memory Survival")
            else "4. All 64 Neurons × 30 Days Forget Gate (f_t) Retention Map"
        )
        st.markdown(
            f'<div class="gm-panel-hdr">'
            f'<span class="gm-panel-title">{panel4_title}</span>'
            f'<span class="gm-panel-sub">Neuron #{neuron_num}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            build_compact_memory_horizon_or_heatmap_chart(
                analysis=analysis,
                neuron_number=neuron_num,
                selected_day=day_num,
                mode_choice=chart4_mode,
            ),
            use_container_width=True,
            theme=None,
        )
