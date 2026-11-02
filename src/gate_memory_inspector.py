import copy
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

from src.lstm_engine import LSTMGateEngine
from src.visualization import prepare_gate_data
from src.architecture_inspector import (
    inspect_model_architecture,
    run_prediction_walkthrough,
)


def analyze_gate_and_memory_flow(
    model_key: str,
    city: str,
    artifacts: Dict[str, Any],
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs a complete 30-day step-by-step decomposition of the trained LSTM cell
    for `model_key` and `city`, capturing:
      1. Exact pre-activation drives for all 4 gates (i_t, f_t, g_t, o_t):
         - Weather Input Drive: W_ih * x_t (plus per-feature breakdown)
         - Recurrent Hidden Drive: W_hh * h_{t-1}
         - Bias Drive: b_ih + b_hh
         - Total Pre-activation z_t = W_ih*x_t + W_hh*h_{t-1} + b_ih + b_hh
      2. Exact additive Cell State Memory Carry-Forward at each day t = 1..30:
         - c_{t-1} (incoming memory from yesterday)
         - retained_memory = f_t * c_{t-1} (kept from yesterday)
         - new_memory_written = i_t * g_t (written today)
         - c_t = retained_memory + new_memory_written
         - tanh_c_t = tanh(c_t)
         - h_t = o_t * tanh(c_t)
      3. Exact Memory Survival Horizon Matrix (30 x 30 x 64):
         How much of Day k's newly written memory (i_k * g_k) still survives inside
         Day t's cell state c_t after being multiplied by f_{k+1} * ... * f_t.
      4. Neuron Role Rankings across all 64 units (Long-Term Memory Keepers,
         Active Daily Writers, and Top Forecast Drivers).
    """
    summary = inspect_model_architecture(model_key, artifacts)
    walkthrough = run_prediction_walkthrough(
        model_key=model_key,
        city=city,
        artifacts=artifacts,
        end_date=end_date,
    )

    # Use a read-only CPU copy for exact float32 step-by-step parity with LSTMGateEngine
    cpu_model = copy.deepcopy(artifacts["models"][model_key]).cpu().eval()
    engine = LSTMGateEngine(cpu_model)
    engine.split_gate_weights(print_shapes=False)

    weather_tensor_3d = walkthrough["weather_tensor"].detach().cpu()  # (1, 30, F)
    sequence_2d = weather_tensor_3d.squeeze(0)  # (30, F)

    with torch.no_grad():
        history, final_h, final_c = engine.process_sequence(sequence_2d)
        pt_out, (pt_hn, pt_cn) = cpu_model.lstm(weather_tensor_3d)

    max_h_diff = float(torch.max(torch.abs(final_h - pt_hn[-1, 0])).item())
    max_c_diff = float(torch.max(torch.abs(final_c - pt_cn[-1, 0])).item())
    if max_h_diff > 1e-5 or max_c_diff > 1e-5:
        raise RuntimeError(
            f"Gate decomposition diverged from PyTorch nn.LSTM: "
            f"max_h_diff={max_h_diff}, max_c_diff={max_c_diff}"
        )

    gate_data = prepare_gate_data(history)  # 6 arrays of shape (30, 64)

    # Extract unpacked gate weights as float64 numpy arrays
    W_ih = engine.W_ih.detach().cpu().numpy().astype(np.float64)  # (256, F)
    W_hh = engine.W_hh.detach().cpu().numpy().astype(np.float64)  # (256, 64)
    b_ih = engine.b_ih.detach().cpu().numpy().astype(np.float64)  # (256,)
    b_hh = engine.b_hh.detach().cpu().numpy().astype(np.float64)  # (256,)
    b_total = b_ih + b_hh  # (256,)

    fc_weight = (
        cpu_model.fc.weight.detach().cpu().numpy()[0].astype(np.float64)
    )  # (fc_in,)

    scaled_x = sequence_2d.numpy().astype(np.float64)  # (30, F)
    T, F = scaled_x.shape
    H = summary["lstm_hidden_size"]  # 64

    # Gate row slices in PyTorch packed order: i (0:64), f (64:128), g (128:192), o (192:256)
    gate_slices = {
        "input_gate": (0, H),
        "forget_gate": (H, 2 * H),
        "candidate_gate": (2 * H, 3 * H),
        "output_gate": (3 * H, 4 * H),
    }

    # Pre-activation decomposition arrays: each (30, 64)
    pre_activations: Dict[str, Dict[str, np.ndarray]] = {}
    for g_name, (r0, r1) in gate_slices.items():
        w_ih_g = W_ih[r0:r1, :]  # (64, F)
        w_hh_g = W_hh[r0:r1, :]  # (64, 64)
        b_g = b_total[r0:r1]  # (64,)

        in_drive = np.zeros((T, H), dtype=np.float64)
        rec_drive = np.zeros((T, H), dtype=np.float64)
        bias_drive = np.tile(b_g, (T, 1))  # (30, 64)
        # Per-feature input contribution: (30, 64, F)
        feat_contribs = np.zeros((T, H, F), dtype=np.float64)

        for t in range(T):
            x_t = scaled_x[t]  # (F,)
            h_prev = (
                np.zeros(H, dtype=np.float64)
                if t == 0
                else gate_data["hidden_state"][t - 1].astype(np.float64)
            )
            in_drive[t] = w_ih_g @ x_t
            rec_drive[t] = w_hh_g @ h_prev
            # Individual feature contribution w_ih_g[n, f] * x_t[f]
            feat_contribs[t] = w_ih_g * x_t[np.newaxis, :]

        z_total = in_drive + rec_drive + bias_drive
        pre_activations[g_name] = {
            "input_drive": in_drive,
            "recurrent_drive": rec_drive,
            "bias_drive": bias_drive,
            "pre_activation": z_total,
            "feature_contributions": feat_contribs,
        }

    # Memory Carry-Forward Decomposition at each step t = 0..29: shape (30, 64)
    f_mat = gate_data["forget_gate"].astype(np.float64)
    i_mat = gate_data["input_gate"].astype(np.float64)
    g_mat = gate_data["candidate_gate"].astype(np.float64)
    o_mat = gate_data["output_gate"].astype(np.float64)
    c_mat = gate_data["cell_state"].astype(np.float64)
    h_mat = gate_data["hidden_state"].astype(np.float64)

    c_prev_mat = np.zeros((T, H), dtype=np.float64)
    c_prev_mat[1:, :] = c_mat[:-1, :]

    retained_memory_mat = f_mat * c_prev_mat  # f_t * c_{t-1}
    new_memory_written_mat = i_mat * g_mat  # i_t * g_t
    tanh_c_mat = np.tanh(c_mat)

    # Verify exact identity: c_t == retained_memory + new_memory_written
    recon_c_diff = float(
        np.max(np.abs(c_mat - (retained_memory_mat + new_memory_written_mat)))
    )
    if recon_c_diff > 1e-5:
        raise RuntimeError(
            f"Cell state decomposition mismatch: max diff = {recon_c_diff}"
        )

    # Memory Survival Horizon Matrix: survival[t, k, n] = how much of Day k's write
    # (i_k * g_k) still remains in c_t at Day t (for k <= t).
    # Note: sum_{k=0..t} survival[t, k, n] == c_mat[t, n] because c_0 = 0!
    memory_survival = np.zeros((T, T, H), dtype=np.float64)
    for t in range(T):
        for k in range(t + 1):
            write_k = new_memory_written_mat[k]  # (64,)
            if k == t:
                retention_factor = np.ones(H, dtype=np.float64)
            else:
                retention_factor = np.prod(f_mat[k + 1 : t + 1], axis=0)
            memory_survival[t, k, :] = write_k * retention_factor

    # Verify exact horizon sum identity: sum_{k=0..t} memory_survival[t, k, :] == c_mat[t, :]
    horizon_diff = float(
        np.max(np.abs(np.sum(memory_survival, axis=1) - c_mat))
    )
    if horizon_diff > 1e-5:
        raise RuntimeError(
            f"Memory survival horizon sum mismatch: max diff = {horizon_diff}"
        )

    # Neuron Role Rankings across the 64 units
    mean_forget = np.mean(f_mat, axis=0)  # (64,)
    mean_abs_write = np.mean(np.abs(new_memory_written_mat), axis=0)  # (64,)
    day30_dense_impact = np.abs(h_mat[-1, :] * fc_weight[:H])  # (64,)

    long_term_keepers = (np.argsort(-mean_forget) + 1).tolist()
    active_daily_writers = (np.argsort(-mean_abs_write) + 1).tolist()
    top_forecast_drivers = (np.argsort(-day30_dense_impact) + 1).tolist()

    neuron_profiles: List[Dict[str, Any]] = []
    for n_idx in range(H):
        avg_f = float(mean_forget[n_idx])
        avg_w = float(mean_abs_write[n_idx])
        if avg_f >= 0.75:
            role_badge = "Long-Term Memory Keeper"
        elif avg_f <= 0.35:
            role_badge = "Fast Daily Resetter"
        elif avg_w >= float(np.percentile(mean_abs_write, 75)):
            role_badge = "Active Weather Writer"
        else:
            role_badge = "Balanced Tracker"

        neuron_profiles.append(
            {
                "neuron_number": n_idx + 1,
                "role": role_badge,
                "mean_forget_gate": avg_f,
                "mean_input_gate": float(np.mean(i_mat[:, n_idx])),
                "mean_abs_write": avg_w,
                "final_c_30": float(c_mat[-1, n_idx]),
                "final_h_30": float(h_mat[-1, n_idx]),
                "fc_weight": float(fc_weight[n_idx]),
                "day30_Impact": float(h_mat[-1, n_idx] * fc_weight[n_idx]),
            }
        )

    raw_features_df = walkthrough["stage2_preprocessed_features"]

    return {
        "model_key": model_key,
        "model_name": summary["model_name"],
        "summary": summary,
        "walkthrough": walkthrough,
        "city": city,
        "dates": walkthrough["dates"],
        "prediction_date": walkthrough["prediction_date"],
        "temporal_features": summary["temporal_features"],
        "raw_features_30d": raw_features_df[summary["temporal_features"]].to_numpy(
            dtype=np.float64
        ),
        "scaled_features_30d": scaled_x,
        "gate_data": gate_data,
        "pre_activations": pre_activations,
        "memory_decomposition": {
            "c_prev": c_prev_mat,
            "retained_memory": retained_memory_mat,
            "new_memory_written": new_memory_written_mat,
            "c_current": c_mat,
            "tanh_c": tanh_c_mat,
            "h_current": h_mat,
        },
        "memory_survival_horizon": memory_survival,
        "neuron_rankings": {
            "long_term_keepers": long_term_keepers,
            "active_daily_writers": active_daily_writers,
            "top_forecast_drivers": top_forecast_drivers,
        },
        "neuron_profiles": neuron_profiles,
        "parity_checks": {
            "max_hidden_diff_vs_pytorch": max_h_diff,
            "max_cell_diff_vs_pytorch": max_c_diff,
            "cell_decomposition_diff": recon_c_diff,
            "memory_horizon_sum_diff": horizon_diff,
        },
    }


def extract_neuron_gate_story(
    analysis: Dict[str, Any],
    neuron_number: int = 25,
    day_number: int = 30,
) -> Dict[str, Any]:
    """
    Extracts a plain-English and exact numerical explanation for a single
    neuron (1..64) on a specific day (1..30), showing how data enters each
    gate, how each gate computes its value, and how memory moves forward.
    """
    if not (1 <= neuron_number <= 64):
        raise ValueError(f"neuron_number must be in 1..64, got {neuron_number}")
    if not (1 <= day_number <= 30):
        raise ValueError(f"day_number must be in 1..30, got {day_number}")

    n = neuron_number - 1
    t = day_number - 1

    dates = analysis["dates"]
    gate_data = analysis["gate_data"]
    mem = analysis["memory_decomposition"]
    pre = analysis["pre_activations"]
    feats = analysis["temporal_features"]

    f_t = float(gate_data["forget_gate"][t, n])
    i_t = float(gate_data["input_gate"][t, n])
    g_t = float(gate_data["candidate_gate"][t, n])
    o_t = float(gate_data["output_gate"][t, n])

    c_prev = float(mem["c_prev"][t, n])
    kept = float(mem["retained_memory"][t, n])
    written = float(mem["new_memory_written"][t, n])
    c_now = float(mem["c_current"][t, n])
    tanh_c = float(mem["tanh_c"][t, n])
    h_now = float(mem["h_current"][t, n])
    h_prev = 0.0 if t == 0 else float(gate_data["hidden_state"][t - 1, n])

    gate_breakdown = {}
    for g_key, g_label, act_fn in [
        ("forget_gate", "Forget Gate (f_t)", "sigmoid"),
        ("input_gate", "Input Gate (i_t)", "sigmoid"),
        ("candidate_gate", "Candidate Gate (g_t)", "tanh"),
        ("output_gate", "Output Gate (o_t)", "sigmoid"),
    ]:
        in_d = float(pre[g_key]["input_drive"][t, n])
        rec_d = float(pre[g_key]["recurrent_drive"][t, n])
        bias_d = float(pre[g_key]["bias_drive"][t, n])
        z_val = float(pre[g_key]["pre_activation"][t, n])
        f_contribs = {
            feats[f_idx]: float(pre[g_key]["feature_contributions"][t, n, f_idx])
            for f_idx in range(len(feats))
        }
        gate_breakdown[g_key] = {
            "label": g_label,
            "activation_fn": act_fn,
            "input_drive": in_d,
            "recurrent_drive": rec_d,
            "bias_drive": bias_d,
            "pre_activation_z": z_val,
            "activation_output": float(gate_data[g_key][t, n]),
            "feature_contributions": f_contribs,
        }

    # Memory survival slices up to day t: shape (t+1,)
    survival_up_to_t = analysis["memory_survival_horizon"][t, : t + 1, n].tolist()

    return {
        "neuron_number": neuron_number,
        "day_number": day_number,
        "date": dates[t],
        "profile": analysis["neuron_profiles"][n],
        "values": {
            "h_prev": h_prev,
            "c_prev": c_prev,
            "f_t": f_t,
            "i_t": i_t,
            "g_t": g_t,
            "o_t": o_t,
            "retained_memory": kept,
            "new_memory_written": written,
            "c_t": c_now,
            "tanh_c_t": tanh_c,
            "h_t": h_now,
        },
        "gate_breakdown": gate_breakdown,
        "survival_from_past_days": survival_up_to_t,
        "plain_english_steps": [
            (
                f"1. Inputs Arrive: Today's weather on Day {day_number} ({dates[t]}) "
                f"and yesterday's hidden summary (h_{day_number - 1} = {h_prev:+.4f}) enter Neuron #{neuron_number}."
            ),
            (
                f"2. Forget Gate Filters Yesterday's Memory: f_{day_number} opens to {f_t:.1%} ({f_t:.4f}), "
                f"multiplying yesterday's cell memory ({c_prev:+.4f}) to keep {kept:+.4f}."
            ),
            (
                f"3. Input & Candidate Gates Write Today's Info: Input Gate ({i_t:.1%}) × "
                f"Candidate Note ({g_t:+.4f}) creates a new memory write of {written:+.4f}."
            ),
            (
                f"4. Cell Conveyor Updates & Moves Forward: Kept memory ({kept:+.4f}) + "
                f"New write ({written:+.4f}) = Today's Cell State c_{day_number} ({c_now:+.4f})."
            ),
            (
                f"5. Output Gate Emits Working Summary: Output Gate ({o_t:.1%}) × "
                f"tanh(c_{day_number}) ({tanh_c:+.4f}) = Hidden State h_{day_number} ({h_now:+.4f}), "
                f"which passes forward to Day {day_number + 1}!"
            ),
        ],
    }

