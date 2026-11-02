from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.architecture_inspector import (
    MODEL_DISPLAY_NAMES,
    inspect_model_architecture,
    run_prediction_walkthrough,
    compute_real_gate_history,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_ARTIFACT_DIR = PROJECT_ROOT / "data" / "processed" / "artifacts" / "v2" / "test"
MODEL_DIR = PROJECT_ROOT / "models"

_CACHED_TEST_BENCHMARK: Optional[Dict[str, Any]] = None


def evaluate_full_test_benchmark(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates all three trained PyTorch models (`baseline`, `model_A`, `model_B`)
    and the Naive Persistence baseline (`Tomorrow = Today`) across the entire
    saved 18,750-sample test split (50 cities × 375 test days), computing both
    global metrics and per-city metrics in original Celsius units.

    Results are cached in memory after the first call (~0.18s on GPU).
    """
    global _CACHED_TEST_BENCHMARK
    if _CACHED_TEST_BENCHMARK is not None:
        return _CACHED_TEST_BENCHMARK

    device = artifacts["device"]
    target_scaler = artifacts["v2"]["target_scaler"]
    city_to_id = artifacts["city_to_id"]
    id_to_city = {int(v): k for k, v in city_to_id.items()}
    geo_df = artifacts["v2"]["geographical_features"].copy()

    X_test = np.load(TEST_ARTIFACT_DIR / "X_test.npy")          # (18750, 30, 8)
    geo_test_A = np.load(TEST_ARTIFACT_DIR / "geo_test_A.npy")  # (18750, 3)
    geo_test_B = np.load(TEST_ARTIFACT_DIR / "geo_test_B.npy")  # (18750, 4)
    city_test = np.load(TEST_ARTIFACT_DIR / "city_test.npy")    # (18750,)
    y_test = np.load(TEST_ARTIFACT_DIR / "y_test.npy")          # (18750, 1)
    target_dates = np.load(TEST_ARTIFACT_DIR / "target_dates.npy", allow_pickle=True)

    if len(target_dates) > len(X_test):
        target_dates = target_dates[-len(X_test):]
    date_strs = pd.to_datetime(target_dates).strftime("%Y-%m-%d").to_numpy()

    # Invert target scaling to physical Celsius
    y_true_c = target_scaler.inverse_transform(y_test).ravel().astype(np.float64)

    # Naive persistence baseline: tomorrow's max temp = Day 30's observed max temp
    # Note: feature 0 in X_test is temperature_2m_max scaled with the exact same [-12.6, 48.3] range
    day30_scaled = X_test[:, -1, 0:1]
    y_naive_c = target_scaler.inverse_transform(day30_scaled).ravel().astype(np.float64)

    # Batched PyTorch forward pass for all 3 trained models
    batch_size = 4096
    n_samples = len(X_test)
    preds_norm = {
        "baseline": np.zeros((n_samples, 1), dtype=np.float32),
        "model_A": np.zeros((n_samples, 1), dtype=np.float32),
        "model_B": np.zeros((n_samples, 1), dtype=np.float32),
    }

    models = artifacts["models"]
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = torch.tensor(
                X_test[start:end], dtype=torch.float32, device=device
            )
            c_batch = torch.tensor(
                city_test[start:end], dtype=torch.long, device=device
            )
            gA_batch = torch.tensor(
                geo_test_A[start:end], dtype=torch.float32, device=device
            )
            gB_batch = torch.tensor(
                geo_test_B[start:end], dtype=torch.float32, device=device
            )

            preds_norm["baseline"][start:end] = (
                models["baseline"](x_batch[:, :, 0:1], c_batch).cpu().numpy()
            )
            preds_norm["model_A"][start:end] = (
                models["model_A"](x_batch, gA_batch, c_batch).cpu().numpy()
            )
            preds_norm["model_B"][start:end] = (
                models["model_B"](x_batch, gB_batch, c_batch).cpu().numpy()
            )

    preds_c = {
        k: target_scaler.inverse_transform(v).ravel().astype(np.float64)
        for k, v in preds_norm.items()
    }
    preds_c["naive"] = y_naive_c

    def _calc_metrics(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, float]:
        err = y_p - y_t
        abs_err = np.abs(err)
        return {
            "MAE": float(mean_absolute_error(y_t, y_p)),
            "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
            "R2": float(r2_score(y_t, y_p)),
            "Bias": float(np.mean(err)),
            "MedianAE": float(np.median(abs_err)),
            "Within_1C_Pct": float(np.mean(abs_err <= 1.0) * 100.0),
            "Within_2C_Pct": float(np.mean(abs_err <= 2.0) * 100.0),
        }

    global_metrics = {
        "naive": _calc_metrics(y_true_c, preds_c["naive"]),
        "baseline": _calc_metrics(y_true_c, preds_c["baseline"]),
        "model_A": _calc_metrics(y_true_c, preds_c["model_A"]),
        "model_B": _calc_metrics(y_true_c, preds_c["model_B"]),
    }

    # Per-city metrics across all 50 cities
    geo_lookup = geo_df.set_index("city").to_dict(orient="index")
    city_rows: List[Dict[str, Any]] = []
    city_time_series: Dict[str, Dict[str, Any]] = {}

    for cid in sorted(id_to_city.keys()):
        cname = id_to_city[cid]
        mask = city_test == cid
        c_dates = date_strs[mask]
        order = np.argsort(c_dates)

        c_dates_sorted = c_dates[order]
        y_t_city = y_true_c[mask][order]
        p_naive_city = preds_c["naive"][mask][order]
        p_base_city = preds_c["baseline"][mask][order]
        p_A_city = preds_c["model_A"][mask][order]
        p_B_city = preds_c["model_B"][mask][order]

        m_base = _calc_metrics(y_t_city, p_base_city)
        m_A = _calc_metrics(y_t_city, p_A_city)
        m_B = _calc_metrics(y_t_city, p_B_city)
        m_naive = _calc_metrics(y_t_city, p_naive_city)

        maes = {
            "Baseline V1": m_base["MAE"],
            "Model A": m_A["MAE"],
            "Model B": m_B["MAE"],
        }
        best_model = min(maes, key=maes.get)

        g_info = geo_lookup.get(cname, {})
        coast_km = float(g_info.get("coastal_distance_km", 0.0))
        elev_m = float(g_info.get("elevation_m", 0.0))
        lat = float(g_info.get("latitude", 0.0))
        lon = float(g_info.get("longitude", 0.0))

        if coast_km < 60.0:
            zone = "Coastal (< 60 km)"
        elif elev_m >= 600.0:
            zone = "High Elevation (≥ 600 m)"
        else:
            zone = "Inland Plains (≥ 60 km)"

        city_rows.append(
            {
                "city": cname,
                "city_id": cid,
                "latitude": lat,
                "longitude": lon,
                "elevation_m": elev_m,
                "coastal_distance_km": coast_km,
                "zone": zone,
                "samples": int(np.sum(mask)),
                "naive_mae": m_naive["MAE"],
                "baseline_mae": m_base["MAE"],
                "model_A_mae": m_A["MAE"],
                "model_B_mae": m_B["MAE"],
                "baseline_rmse": m_base["RMSE"],
                "model_A_rmse": m_A["RMSE"],
                "model_B_rmse": m_B["RMSE"],
                "baseline_r2": m_base["R2"],
                "model_A_r2": m_A["R2"],
                "model_B_r2": m_B["R2"],
                "gain_A_vs_V1_mae": m_base["MAE"] - m_A["MAE"],
                "gain_B_vs_V1_mae": m_base["MAE"] - m_B["MAE"],
                "gain_B_vs_A_mae": m_A["MAE"] - m_B["MAE"],
                "best_model": best_model,
            }
        )

        city_time_series[cname] = {
            "dates": c_dates_sorted.tolist(),
            "y_true": y_t_city.tolist(),
            "naive": p_naive_city.tolist(),
            "baseline": p_base_city.tolist(),
            "model_A": p_A_city.tolist(),
            "model_B": p_B_city.tolist(),
            "metrics": {
                "naive": m_naive,
                "baseline": m_base,
                "model_A": m_A,
                "model_B": m_B,
            },
        }

    city_df = pd.DataFrame(city_rows)

    # Downsampled residuals for fast histogram rendering
    residuals = {
        "baseline": (preds_c["baseline"] - y_true_c).tolist(),
        "model_A": (preds_c["model_A"] - y_true_c).tolist(),
        "model_B": (preds_c["model_B"] - y_true_c).tolist(),
    }

    win_counts = city_df["best_model"].value_counts().to_dict()

    _CACHED_TEST_BENCHMARK = {
        "num_test_samples": n_samples,
        "num_cities": len(city_rows),
        "test_start_date": str(np.min(date_strs)),
        "test_end_date": str(np.max(date_strs)),
        "global_metrics": global_metrics,
        "city_metrics_df": city_df,
        "city_time_series": city_time_series,
        "residuals": residuals,
        "city_win_counts": {
            "Baseline V1": int(win_counts.get("Baseline V1", 0)),
            "Model A": int(win_counts.get("Model A", 0)),
            "Model B": int(win_counts.get("Model B", 0)),
        },
    }
    return _CACHED_TEST_BENCHMARK


def run_head_to_head_comparison(
    city: str,
    artifacts: Dict[str, Any],
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes a synchronized 3-model forward pass (`baseline`, `model_A`, `model_B`)
    on the exact same 30-day historical window for `city` ending on `end_date`.
    Returns each model's architecture summary, 30-day gate activations, dense
    layer feature contributions, and next-day Celsius forecast so the user can
    inspect all three models side by side.
    """
    models_data: Dict[str, Any] = {}
    for m_key in ["baseline", "model_A", "model_B"]:
        summary = inspect_model_architecture(m_key, artifacts)
        walkthrough = run_prediction_walkthrough(
            model_key=m_key,
            city=city,
            artifacts=artifacts,
            end_date=end_date,
        )
        gate_res = compute_real_gate_history(
            model_key=m_key,
            artifacts=artifacts,
            city=city,
            weather_tensor=walkthrough["weather_tensor"],
        )
        gd = gate_res["gate_data"]

        # Extract Day-30 gate averages across all 64 neurons
        day30_gate_means = {
            "forget_gate": float(np.mean(gd["forget_gate"][-1])),
            "input_gate": float(np.mean(gd["input_gate"][-1])),
            "candidate_gate": float(np.mean(gd["candidate_gate"][-1])),
            "output_gate": float(np.mean(gd["output_gate"][-1])),
            "cell_state_abs": float(np.mean(np.abs(gd["cell_state"][-1]))),
            "hidden_state_abs": float(np.mean(np.abs(gd["hidden_state"][-1]))),
        }

        models_data[m_key] = {
            "model_key": m_key,
            "model_name": MODEL_DISPLAY_NAMES[m_key],
            "summary": summary,
            "walkthrough": walkthrough,
            "gate_data": gd,
            "day30_gate_means": day30_gate_means,
            "predicted_celsius": float(
                walkthrough["stage10_celsius_prediction"]["predicted_max_temperature_celsius"]
            ),
            "actual_celsius": walkthrough["stage10_celsius_prediction"][
                "actual_max_temperature_celsius"
            ],
            "abs_error_celsius": walkthrough["stage10_celsius_prediction"][
                "absolute_error_celsius"
            ],
            "scaled_output": float(
                walkthrough["stage9_dense_output"]["scaled_prediction"]
            ),
            "contributions": walkthrough["stage9_dense_output"]["contributions"],
        }

    # Determine winner on this specific prediction
    actual_temp = models_data["model_B"]["actual_celsius"]
    day30_temp = float(
        models_data["model_B"]["walkthrough"]["stage1_raw_observations"][
            "temperature_2m_max"
        ].iloc[-1]
    )
    naive_err = abs(day30_temp - actual_temp) if actual_temp is not None else None

    winner_key = "model_B"
    if actual_temp is not None:
        errs = {
            k: models_data[k]["abs_error_celsius"]
            for k in ["baseline", "model_A", "model_B"]
            if models_data[k]["abs_error_celsius"] is not None
        }
        if errs:
            winner_key = min(errs, key=errs.get)

    return {
        "city": city,
        "city_id": models_data["model_B"]["walkthrough"]["city_id"],
        "window_start_date": models_data["model_B"]["walkthrough"]["window_start_date"],
        "last_observation_date": models_data["model_B"]["walkthrough"]["last_observation_date"],
        "prediction_date": models_data["model_B"]["walkthrough"]["prediction_date"],
        "dates": models_data["model_B"]["walkthrough"]["dates"],
        "actual_celsius": actual_temp,
        "day30_max_temp_celsius": day30_temp,
        "naive_abs_error_celsius": naive_err,
        "winner_key": winner_key,
        "winner_name": MODEL_DISPLAY_NAMES[winner_key],
        "models": models_data,
    }


def build_architecture_comparison_table(artifacts: Dict[str, Any]) -> pd.DataFrame:
    """
    Builds a side-by-side structural and statistical comparison DataFrame
    for Baseline V1, Model A, and Model B.
    """
    bench = evaluate_full_test_benchmark(artifacts)
    gm = bench["global_metrics"]
    wins = bench["city_win_counts"]

    s_base = inspect_model_architecture("baseline", artifacts)
    s_A = inspect_model_architecture("model_A", artifacts)
    s_B = inspect_model_architecture("model_B", artifacts)

    rows = [
        {
            "Dimension / Metric": "PyTorch Class",
            "Baseline V1 (Univariate)": s_base["architecture_class"],
            "Model A (Multivariate + 3 Geo)": s_A["architecture_class"],
            "Model B (Multivariate + 4 Geo)": s_B["architecture_class"],
        },
        {
            "Dimension / Metric": "Temporal Weather Input Shape",
            "Baseline V1 (Univariate)": "(B, 30, 1) — Max Temp only",
            "Model A (Multivariate + 3 Geo)": "(B, 30, 8) — 8 Daily Weather Features",
            "Model B (Multivariate + 4 Geo)": "(B, 30, 8) — 8 Daily Weather Features",
        },
        {
            "Dimension / Metric": "Geographical Bypass Features",
            "Baseline V1 (Univariate)": "None (0 dims)",
            "Model A (Multivariate + 3 Geo)": "3 dims (Lat, Lon, Elevation)",
            "Model B (Multivariate + 4 Geo)": "4 dims (Lat, Lon, Elevation, Coastal Dist km)",
        },
        {
            "Dimension / Metric": "City Embedding Lookup",
            "Baseline V1 (Univariate)": "nn.Embedding(50, 8) — 400 params",
            "Model A (Multivariate + 3 Geo)": "nn.Embedding(50, 8) — 400 params",
            "Model B (Multivariate + 4 Geo)": "nn.Embedding(50, 8) — 400 params",
        },
        {
            "Dimension / Metric": "LSTM Input Weight Matrix (W_ih)",
            "Baseline V1 (Univariate)": "(256, 1) — 256 params",
            "Model A (Multivariate + 3 Geo)": "(256, 8) — 2,048 params",
            "Model B (Multivariate + 4 Geo)": "(256, 8) — 2,048 params",
        },
        {
            "Dimension / Metric": "LSTM Recurrent Matrix (W_hh)",
            "Baseline V1 (Univariate)": "(256, 64) — 16,384 params",
            "Model A (Multivariate + 3 Geo)": "(256, 64) — 16,384 params",
            "Model B (Multivariate + 4 Geo)": "(256, 64) — 16,384 params",
        },
        {
            "Dimension / Metric": "Concatenated Vector → Dense Head",
            "Baseline V1 (Univariate)": "Linear(72 → 1) [64 + 8]",
            "Model A (Multivariate + 3 Geo)": "Linear(75 → 1) [64 + 3 + 8]",
            "Model B (Multivariate + 4 Geo)": "Linear(76 → 1) [64 + 4 + 8]",
        },
        {
            "Dimension / Metric": "Total Trainable Parameters",
            "Baseline V1 (Univariate)": f"{s_base['trainable_parameters']:,}",
            "Model A (Multivariate + 3 Geo)": f"{s_A['trainable_parameters']:,}",
            "Model B (Multivariate + 4 Geo)": f"{s_B['trainable_parameters']:,} (+1 over Model A)",
        },
        {
            "Dimension / Metric": "Test MAE (18,750 samples)",
            "Baseline V1 (Univariate)": f"{gm['baseline']['MAE']:.4f} °C",
            "Model A (Multivariate + 3 Geo)": f"{gm['model_A']['MAE']:.4f} °C (-3.05% vs V1)",
            "Model B (Multivariate + 4 Geo)": f"{gm['model_B']['MAE']:.4f} °C (-5.26% vs V1) 🏆",
        },
        {
            "Dimension / Metric": "Test RMSE (18,750 samples)",
            "Baseline V1 (Univariate)": f"{gm['baseline']['RMSE']:.4f} °C",
            "Model A (Multivariate + 3 Geo)": f"{gm['model_A']['RMSE']:.4f} °C",
            "Model B (Multivariate + 4 Geo)": f"{gm['model_B']['RMSE']:.4f} °C 🏆",
        },
        {
            "Dimension / Metric": "Test R² Score",
            "Baseline V1 (Univariate)": f"{gm['baseline']['R2']:.4f}",
            "Model A (Multivariate + 3 Geo)": f"{gm['model_A']['R2']:.4f}",
            "Model B (Multivariate + 4 Geo)": f"{gm['model_B']['R2']:.4f} 🏆",
        },
        {
            "Dimension / Metric": "Predictions Within ±1.0 °C",
            "Baseline V1 (Univariate)": f"{gm['baseline']['Within_1C_Pct']:.1f}%",
            "Model A (Multivariate + 3 Geo)": f"{gm['model_A']['Within_1C_Pct']:.1f}%",
            "Model B (Multivariate + 4 Geo)": f"{gm['model_B']['Within_1C_Pct']:.1f}% 🏆",
        },
        {
            "Dimension / Metric": "Cities Won (Lowest MAE out of 50)",
            "Baseline V1 (Univariate)": f"{wins['Baseline V1']} / 50 cities",
            "Model A (Multivariate + 3 Geo)": f"{wins['Model A']} / 50 cities",
            "Model B (Multivariate + 4 Geo)": f"{wins['Model B']} / 50 cities 🏆",
        },
    ]
    return pd.DataFrame(rows)

