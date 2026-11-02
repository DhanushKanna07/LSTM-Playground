#  LSTM Playground — Interactive Explainable AI for Weather Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](https://streamlit.io/)

**LSTM Playground** is an interactive educational web application that opens the "black box" of Long Short-Term Memory (**LSTM**) neural networks using **three real, trained PyTorch weather forecasting models** across **50 Indian cities** and **10 years of daily meteorological observations** (`182,620` daily records; `18,750` held-out test forecasts).

Unlike toy neural network demos that use synthetic data or random weights, every activation, gate value, weight matrix, cell-state update, and next-day maximum temperature prediction in this application is computed live from our **actual trained PyTorch checkpoints** (`.pth`) and **fitted scikit-learn scalers** (`.pkl`).

---

##  Table of Contents

1. [Project Highlights & Core Philosophy](#-project-highlights--core-philosophy)
2. [The Three Trained LSTM Architectures & Benchmark Results](#-the-three-trained-lstm-architectures--benchmark-results)
3. [Interactive Application Modules (How to Use the App)](#-interactive-application-modules-how-to-use-the-app)
   - [Module 1: Trained Architecture Explorer](#1-trained-architecture-explorer)
   - [Module 2: Interactive Gate & Memory Explorer](#2-interactive-gate--memory-explorer)
   - [Module 3: Model Comparison Explorer](#3-model-comparison-explorer)
4. [End-to-End ML Workflow (Step-by-Step Pipeline)](#-end-to-end-ml-workflow-step-by-step-pipeline)
5. [Repository Structure](#-repository-structure)
6. [How to Download, Install & Run](#-how-to-download-install--run)
7. [Running the Automated Verification Suites](#-running-the-automated-verification-suites)
8. [License (MIT)](#-license)

---

##  Project Highlights & Core Philosophy

* **100% Real Trained Weights & Data**: All visualizations are driven by three saved PyTorch checkpoints (`best_weather_lstm.pth`, `best_multivariate_model_A.pth`, `best_multivariate_model_B.pth`) and real 30-day historical weather windows across 50 Indian cities.
* **Exact Mathematical Transparency**: A custom PyTorch gate-extraction engine (`src/lstm_engine.py`) unrolls the 30-day sequence step by step ($t = 1 \dots 30$), decomposing the exact PyTorch weight slices (`[0:64]` Input, `[64:128]` Forget, `[128:192]` Candidate, `[192:256]` Output) and verifying numerical identity ($\le 10^{-6}$) against `torch.nn.LSTM`.
* **60fps Interactive What-If Perturbations & Neuron Ablation**: Users can drag live weather sliders (`Max Temp`, `Humidity`, `Rainfall`, `Wind Speed`), double-click any of the 64 LSTM neurons to mute/ablate it, or mute static geography features (`coastal_distance_km`) to see how the network's gates and Celsius forecast respond in real time.
* **Two-Tier Explanations**: Every module includes a toggle between **Beginner Mode** (intuitive analogies for students new to deep learning) and **Technical Mode** (exact LaTeX equations, tensor shapes, and matrix slice indices).

---

##  The Three Trained LSTM Architectures & Benchmark Results

All three models predict **next-day maximum temperature (`temperature_2m_max` in °C)** from a **30-day sliding window** of historical weather observations, evaluated on **18,750 held-out test samples** (375 test days per city across all 50 cities):

| Specification / Metric | Naive Persistence ($y_t = y_{t-1}$) | **Baseline V1** (`CityAwareLSTM`) | **Model A** (`MultivariateWeatherLSTM`) | **Model B** (`MultivariateWeatherLSTM`) |
| :--- | :---: | :---: | :---: | :---: |
| **Temporal Input Tensor** | Day 30 `temp_max` | `(B, 30, 1)` — Max Temp only | `(B, 30, 8)` — 8 Daily Features | `(B, 30, 8)` — 8 Daily Features |
| **LSTM Recurrent Core** | — | 64 Hidden Units (`16,896` params) | 64 Hidden Units (`18,944` params) | 64 Hidden Units (`18,944` params) |
| **Geography Bypass Track** | — | None (`0` dims) | `3` dims (`Lat`, `Lon`, `Elevation`) | `4` dims (`Lat`, `Lon`, `Elevation`, `Coastal Dist km`) |
| **City Embedding Lookup** | — | `nn.Embedding(50, 8)` (`400` params) | `nn.Embedding(50, 8)` (`400` params) | `nn.Embedding(50, 8)` (`400` params) |
| **Concatenated Vector → Dense Head** | — | `Linear(72 → 1)` `[64 + 8]` | `Linear(75 → 1)` `[64 + 3 + 8]` | `Linear(76 → 1)` `[64 + 4 + 8]` |
| **Total Trainable Parameters** | `0` | `17,369` | `19,420` | `19,421` (`+1` weight over Model A) |
| **Test MAE (°C) ↓** | `1.0502 °C` | `1.0373 °C` | `1.0057 °C` (`-3.05%` vs V1) | **`0.9827 °C`** (`-5.26%` vs V1) |
| **Test RMSE (°C) ↓** | `1.4905 °C` | `1.4527 °C` | `1.3972 °C` | **`1.3797 °C`** |
| **Test $R^2$ Score ↑** | `0.9474` | `0.9501` | `0.9538` | **`0.9550`** |
| **Cities Won (Lowest MAE / 50)** | — | `2 / 50` cities | `9 / 50` cities | **`39 / 50` cities** |

### Why Does Model B Outperform Model A with Only `+1` Extra Parameter?
In `MultivariateWeatherLSTM`, static geography features **bypass** the 30-day recurrent LSTM loop and concatenate directly with the Day-30 final hidden state $h_{30} \in \mathbb{R}^{64}$ and the 8-dimensional city embedding $e_{\text{city}} \in \mathbb{R}^8$ right before the final `nn.Linear` readout head. Adding explicit distance to the coastline (`coastal_distance_km`, computed from Natural Earth 10m coastline shapefiles) adds just **one scalar weight** to the final linear layer (`75 → 1` becomes `76 → 1`), allowing the model to distinguish maritime thermal buffering in coastal cities (`< 60 km` from the sea) from continental diurnal swings in inland plains.

---

##  Interactive Application Modules (How to Use the App)

### 1. Trained Architecture Explorer
Located in `src/architecture_explorer.py`, `src/playground_canvas.py`, `src/architecture_diagram.py`, and `src/architecture_inspector.py`:
* **Interactive Neural Network Playground Canvas**:
  * **30-Day Timeline Scrubber**: Click or drag any day from `Day 1` to `Day 30` to watch weather signals flow through the network.
  * **Live What-If Weather Sliders**: Perturb Day-$t$'s weather inputs (`temp_max`, `temp_min`, `temp_mean`, `humidity`, `precip_log`, `wind_max`, `day_of_year_sin/cos`) and watch the 4 LSTM gates, 64 hidden units, and final Celsius forecast update at 60fps.
  * **64-Neuron Interactive Lattice (`8×8` Grid)**: Click any neuron (`#1` to `#64`) to inspect its incoming weights and gate states, or **double-click** any neuron (or click `Mute Geo` / `Mute City`) to perform live **ablation experiments**.
* **4 Deep-Dive Inspection Tabs**:
  1. *Component & Tensor Flow Inspector*: Interactive SVG architecture diagram with clickable blocks and stage-by-stage tensor shapes for batch size $B \in \{1, 8, 16, 32, 64\}$.
  2. *Actual Trained Weights Inspector*: Heatmaps and distributions of the real learned PyTorch parameters (`W_ih`, `W_hh`, `b_ih`, `b_hh`, `city_embedding.weight`, `fc.weight`, `fc.bias`).
  3. *LSTM Neuron & Gate Inspector (64 Units)*: Ranks all 64 neurons by their actual linear readout contribution $w_{\text{fc}, u} \cdot h_{30, u}$ and plots their 30-day trajectories.
  4. *Prediction Walkthrough (10 Stages)*: Traces a real 30-day historical window through all 10 mathematical stages from raw CSV observations to the final inverse-scaled Celsius prediction.

### 2. Interactive Gate & Memory Explorer
Located in `src/gate_memory_explorer.py`, `src/gate_memory_inspector.py`, and `src/lstm_engine.py`:
* **Animated LSTM Cell Circuit Diagram**:
  * Click ** Play 30 Days** (or scrub the day slider / click any neuron `#1–#64`) to watch the 4 gates operate inside the LSTM cell:
    $$\begin{aligned}
    i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \quad &\text{(Input Gate — Write Filter)} \\
    f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \quad &\text{(Forget Gate — Keep Ratio)} \\
    g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \quad &\text{(Candidate — New Note)} \\
    o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \quad &\text{(Output Gate — Read Filter)} \\
    c_t &= \underbrace{f_t \odot c_{t-1}}_{\text{Kept from Yesterday}} + \underbrace{i_t \odot g_t}_{\text{Written Today}}, \qquad h_t = o_t \odot \tanh(c_t)
    \end{aligned}$$
  * Displays the exact live arithmetic inside each gate box and the **Memory Survival Horizon** bar showing how much of each earlier day's write ($k \le t$) still survives in $c_t$:
    $$c_t = \sum_{k=1}^{t} \left(\prod_{j=k+1}^{t} f_j\right) (i_k \odot g_k)$$
* **Compact 2×2 Interactive Plotly Grid**:
  1. *How Neuron `#u` Remembers Moving Forward*: Stacked bar decomposition of `Kept from Yesterday` ($f_t \cdot c_{t-1}$) + `New Written Today` ($i_t \cdot g_t$) = `Cell State` ($c_t$).
  2. *How Neuron `#u`'s 4 Gates React to Daily Weather*: 30-day trajectories of $f_t, i_t, g_t, o_t$ overlaid with daily `Max Temp (°C)`.
  3. *Inside Gate Math on Day $t$*: Pre-activation drive breakdown comparing `Today's Weather Drive` ($W_{ih} x_t$), `Yesterday's Memory Drive` ($W_{hh} h_{t-1}$), and `Learned Bias` ($b$).
  4. *Memory Survival Horizon / All 64 Neurons × 30 Days Forget Gate Heatmap*.

### 3. Model Comparison Explorer
Located in `src/model_comparison_explorer.py` and `src/model_comparison_inspector.py`:
* **4-Tier Benchmark Scoreboard**: Side-by-side summary of `Naive Persistence`, `Baseline V1`, `Model A`, and `Model B` across all 18,750 test samples.
* **Interactive 3-Model Head-to-Head Arena**:
  * Runs all three models simultaneously on the selected city and 30-day window.
  * Includes live **Day-30 Weather Sliders** (`Max Temp`, `Humidity`, `Rainfall`, `Wind Speed`) and **Geography Bypass Ablation** buttons (`Mute Coastal Dist (Model B)` and `Mute All Geo (A & B)`), demonstrating visually why `Baseline V1` is blind to humidity/rainfall/wind changes while `Model A` and `Model B` adapt immediately.
* **Compact 2×2 Comparison Charts & 50-City Leaderboard**:
  1. *45-Day Test Split Forecast Trajectory* (`Observed` vs `Baseline V1`, `Model A`, `Model B`).
  2. *Error Benchmark Comparison* (`All 50 Cities MAE` vs `Selected City MAE & RMSE`).
  3. *Geography & Coastal Impact across 50 Cities* (`Coastal < 60 km`, `Inland Plains`, `High Elevation ≥ 600 m`).
  4. *Top City Improvements / 18,750-Sample Test Residual Distribution*, plus expandable tables for the 13-row Architecture Comparison and 50-City Leaderboard.

---

##  End-to-End ML Workflow (Step-by-Step Pipeline)

If you want to understand or reproduce how the raw weather data was transformed into the trained models, explore the four Jupyter Notebooks in `notebooks/`:

1. **`notebooks/01_Dataset_Inspection.ipynb` — Exploratory Data Analysis**
   * Inspects `data/raw/weather_daily.csv` (`182,620` daily records across 50 Indian cities from `2016-01-01` to `2025-12-31`) and `data/raw/cities.csv`.
   * Checks for missing values, seasonal temperature regimes, precipitation skewness, and spatial coverage across India.
2. **`notebooks/02_Data_Preprocessing.ipynb` — Feature Engineering & Sequence Construction**
   * Merges daily weather records with city metadata into `data/processed/india_weather_merged.csv`.
   * Engineers cyclical calendar features (`day_of_year_sin`, `day_of_year_cos`) and log-transformed precipitation (`precipitation_log = log1p(precipitation_sum)`).
   * Computes exact distance from each city to the nearest coastline (`coastal_distance_km`) using Natural Earth 10m coastline shapefiles (`data/raw/coastline/ne_10m_coastline.shp`) via `geopandas` and `shapely`.
   * Builds 30-day sliding windows and chronological train/validation/test splits, saving fitted `MinMaxScaler` / `StandardScaler` objects in `data/processed/artifacts/`.
3. **`notebooks/03_LSTM_Model_Training.ipynb` — Univariate Baseline V1 (`CityAwareLSTM`)**
   * Trains the univariate `CityAwareLSTM` model (`models/best_weather_lstm.pth`) using 30 days of `temperature_2m_max` + an 8-dimensional learned `nn.Embedding(50, 8)` city vector.
   * Evaluates against the Naive Persistence baseline and exports `models/version1_results.json`.
4. **`notebooks/04_Multivariate_LSTM.ipynb` — Multivariate & Geography-Aware Models A & B (`MultivariateWeatherLSTM`)**
   * Trains **Model A** (8 daily weather features + 3 static geography features: `latitude`, `longitude`, `elevation_m`) and **Model B** (adding `coastal_distance_km` as the 4th static geography feature).
   * Saves checkpoints (`models/best_multivariate_model_A.pth`, `models/best_multivariate_model_B.pth`), held-out test tensors (`data/processed/artifacts/v2/test/`), and benchmark metrics (`models/multivariate_v2_results.json`).

---

##  Repository Structure

```text
LSTM_explainer/
├── app.py                                 # Main Streamlit entry point & navigation router
├── requirements.txt                       # Python dependencies required to run app & notebooks
├── LICENSE                                # MIT License
├── README.md                              # Project documentation & setup guide
├── .gitignore                             # Git ignore rules for venv, caches, and backups
│
├── src/                                   # Modular backend & frontend engines
│   ├── __init__.py
│   ├── artifacts.py                       # Unified loader for all .pth models, .pkl scalers & .json configs
│   ├── models.py                          # PyTorch model definitions (CityAwareLSTM, MultivariateWeatherLSTM)
│   ├── lstm_engine.py                     # Step-by-step mathematical LSTM gate & state extraction engine
│   ├── inference.py                       # Real-data 30-day preprocessing, scaling & next-day inference
│   ├── visualization.py                   # Core Plotly visualization helpers
│   ├── architecture_inspector.py          # Weight, tensor shape, 64-neuron & 10-stage walkthrough inspector
│   ├── architecture_diagram.py            # Interactive clickable SVG architecture diagram builder
│   ├── playground_canvas.py               # 60fps HTML/JS Neural Network Playground canvas & ablation engine
│   ├── architecture_explorer.py           # Streamlit UI for Module 1: Architecture Explorer
│   ├── gate_memory_inspector.py           # Cell decomposition (f_t*c_{t-1} + i_t*g_t) & survival horizon math
│   ├── gate_memory_explorer.py            # Streamlit UI for Module 2: Gate & Memory Explorer
│   ├── model_comparison_inspector.py      # 18,750-sample test evaluator & 3-model synchronized forward pass
│   └── model_comparison_explorer.py       # Streamlit UI for Module 3: Model Comparison Explorer
│
├── models/                                # Saved trained PyTorch checkpoints & benchmark JSONs
│   ├── best_weather_lstm.pth              # Trained Baseline V1 checkpoint (CityAwareLSTM)
│   ├── best_multivariate_model_A.pth      # Trained Model A checkpoint (8 weather + 3 geo)
│   ├── best_multivariate_model_B.pth      # Trained Model B checkpoint (8 weather + 4 geo incl. coast)
│   ├── version1_results.json              # Saved evaluation metrics for Baseline V1
│   └── multivariate_v2_results.json       # Saved evaluation metrics for Model A & Model B
│
├── data/                                  # Raw datasets, coastline shapefiles & fitted scalers
│   ├── raw/
│   │   ├── cities.csv                     # Coordinates & elevations for 50 Indian cities
│   │   ├── weather_daily.csv              # 10-year daily weather observations (182,620 rows)
│   │   └── coastline/                     # Natural Earth 10m coastline shapefiles
│   └── processed/
│       ├── india_weather_merged.csv       # Preprocessed dataset with engineered features
│       └── artifacts/                     # V1 & V2 scalers (.pkl), configs (.json), and test arrays (.npy)
│
├── notebooks/                             # End-to-end data science & model training notebooks
│   ├── 01_Dataset_Inspection.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_LSTM_Model_Training.ipynb
│   └── 04_Multivariate_LSTM.ipynb
│
└── test_*.py                              # Automated verification test suites
    ├── test_artifacts.py
    ├── test_lstm_engine.py
    ├── test_inference.py
    ├── test_visualization.py
    ├── test_architecture.py
    ├── test_gate_memory.py
    └── test_model_comparison.py
```

---

##  How to Download, Install & Run

Follow these step-by-step instructions to run **LSTM Playground** on your local machine (Windows, macOS, or Linux).

### Prerequisites
* **Python 3.10 – 3.12** installed ([Download Python](https://www.python.org/downloads/))
* **Git** installed ([Download Git](https://git-scm.com/downloads))
* *(Optional)* NVIDIA GPU with CUDA for faster batch evaluation (the app automatically falls back to CPU if CUDA is unavailable).

---

### Step 1: Download or Clone the Repository

**Option A — Clone via Git (Recommended):**
```bash
git clone https://github.com/<your-username>/LSTM_explainer.git
cd LSTM_explainer
```

**Option B — Download ZIP:**
1. Click the green **Code** button on GitHub and select **Download ZIP**.
2. Extract the ZIP archive to a folder on your computer.
3. Open a terminal (`PowerShell`, `Command Prompt`, or `bash`) inside the extracted `LSTM_explainer` directory.

---

### Step 2: Create and Activate a Virtual Environment

Creating an isolated virtual environment ensures clean package installation.

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Note: If PowerShell blocks script execution, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, or use Command Prompt: `.\venv\Scripts\activate.bat`)*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Required Dependencies

Upgrade `pip` and install all required packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional: If you have an NVIDIA GPU and want CUDA-accelerated PyTorch, install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/) before or after running `pip install -r requirements.txt`.)*

---

### Step 4: Launch the Interactive Streamlit Application

Start the Streamlit server from the project root directory:

```bash
streamlit run app.py
```

Your default web browser will automatically open at:
 **`http://localhost:8501`**

Use the **Navigation** sidebar on the left to switch between:
1. **Home** — Overview of the learning journey and loaded execution device (`cuda` or `cpu`).
2. **Architecture Explorer** — Interactive Neural Network Playground canvas, weight heatmaps, 64-neuron lattice, and 10-stage prediction walkthrough.
3. **Gate & Memory Explorer** — Animated 30-day LSTM Cell Circuit, gate math breakdown, and memory survival horizon.
4. **Model Comparison** — 18,750-sample test benchmark scoreboard, live 3-model Head-to-Head Arena, coastal impact analysis, and 50-city leaderboard.

---

##  Running the Automated Verification Suites

The repository includes comprehensive automated verification scripts that validate artifact loading, mathematical gate parity ($\le 10^{-6}$ error between custom gate unrolling and `torch.nn.LSTM`), real-data inference, and UI chart generation:

```bash
# 1. Verify artifact loader, model checkpoints, and scalers
python -m test_artifacts

# 2. Verify custom LSTM gate extraction engine against PyTorch nn.LSTM
python -m test_lstm_engine

# 3. Verify 30-day preprocessing and next-day Celsius inference
python -m test_inference

# 4. Verify Trained Architecture Explorer & Playground Canvas
python -m test_architecture

# 5. Verify Interactive Gate & Memory Explorer & Cell State Decomposition
python -m test_gate_memory

# 6. Verify Model Comparison Module & 18,750-sample Test Benchmark Parity
python -m test_model_comparison
```

All test suites run headlessly and exit with code `0` when all assertions pass.

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for full details.

```text
MIT License — Copyright (c) 2026 LSTM Playground Contributors
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

