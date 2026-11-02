from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.inference import WeatherInference
from src.lstm_engine import LSTMGateEngine
from src.visualization import prepare_gate_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEATHER_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "india_weather_merged.csv"

MODEL_DISPLAY_NAMES = {
    "baseline": "Baseline V1",
    "model_A": "Model A",
    "model_B": "Model B",
}

GATE_KEYS = [
    "input_gate",
    "forget_gate",
    "candidate_gate",
    "output_gate",
    "cell_state",
    "hidden_state",
]


class ArchitectureInspector:
    """
    Extracts and validates architecture dimensions, layer configurations,
    parameter counts, weight/bias tensor shapes, stage-by-stage tensor
    flow dimensions, layer explanations, and real trained weight statistics
    directly from a loaded PyTorch weather forecasting model.
    """

    def __init__(
        self,
        model: nn.Module,
        model_key: str,
        artifacts: Dict[str, Any],
    ):
        if model_key not in MODEL_DISPLAY_NAMES:
            raise ValueError(
                f"Unknown model_key '{model_key}'. "
                f"Expected one of {list(MODEL_DISPLAY_NAMES.keys())}."
            )

        self.model = model
        self.model_key = model_key
        self.artifacts = artifacts

        self.v1_config = artifacts["v1"]["config"]
        self.v2_config = artifacts["v2"]["config"]

    def extract_summary(self) -> Dict[str, Any]:
        """
        Extracts all architectural dimensions and parameter details from
        the actual PyTorch model instance and validates them against the
        saved configuration files.
        """
        # 1. Extract core PyTorch layer attributes directly from the loaded model
        lstm_layer: nn.LSTM = self.model.lstm
        embedding_layer: nn.Embedding = self.model.city_embedding
        fc_layer: nn.Linear = self.model.fc

        lstm_input_size = int(lstm_layer.input_size)
        lstm_hidden_size = int(lstm_layer.hidden_size)
        lstm_num_layers = int(lstm_layer.num_layers)
        bidirectional = bool(lstm_layer.bidirectional)
        batch_first = bool(lstm_layer.batch_first)

        city_vocab_size = int(embedding_layer.num_embeddings)
        city_embedding_dim = int(embedding_layer.embedding_dim)

        fc_input_size = int(fc_layer.in_features)
        fc_output_size = int(fc_layer.out_features)

        # 2. Derive geography dimension from actual dense layer input size
        num_directions = 2 if bidirectional else 1
        final_hidden_dim = lstm_hidden_size * num_directions
        geo_dim = fc_input_size - (final_hidden_dim + city_embedding_dim)

        if geo_dim < 0:
            raise ValueError(
                f"Invalid FC input dimension ({fc_input_size}) relative to "
                f"hidden size ({final_hidden_dim}) + embedding dim ({city_embedding_dim})."
            )

        # 3. Extract configuration metadata (sequence length & feature names)
        if self.model_key == "baseline":
            sequence_length = int(self.v1_config["sequence_length"])
            temporal_features = [self.v1_config["target"]]
            geo_features: List[str] = []
        else:
            sequence_length = int(self.v2_config["sequence_length"])
            temporal_features = list(self.v2_config["temporal_features"])
            geo_features = list(self.v2_config[self.model_key]["geo_features"])

        # 4. Extract parameter counts and tensor shapes via named_parameters()
        parameter_shapes: Dict[str, Tuple[int, ...]] = {}
        parameter_counts: Dict[str, int] = {}
        component_parameters: Dict[str, int] = {
            "city_embedding": 0,
            "lstm": 0,
            "fc": 0,
        }
        component_parameter_details: Dict[str, Dict[str, Dict[str, Any]]] = {
            "city_embedding": {},
            "lstm": {},
            "fc": {},
        }

        total_parameters = 0
        trainable_parameters = 0

        for name, param in self.model.named_parameters():
            shape = tuple(int(dim) for dim in param.shape)
            count = int(param.numel())

            parameter_shapes[name] = shape
            parameter_counts[name] = count
            total_parameters += count

            if param.requires_grad:
                trainable_parameters += count

            component_prefix = name.split(".")[0]
            short_param_name = ".".join(name.split(".")[1:])

            if component_prefix not in component_parameters:
                component_parameters[component_prefix] = 0
                component_parameter_details[component_prefix] = {}

            component_parameters[component_prefix] += count
            component_parameter_details[component_prefix][short_param_name] = {
                "full_name": name,
                "shape": shape,
                "numel": count,
                "requires_grad": bool(param.requires_grad),
            }

        device = str(next(self.model.parameters()).device)

        summary = {
            "model_key": self.model_key,
            "model_name": MODEL_DISPLAY_NAMES[self.model_key],
            "architecture_class": self.model.__class__.__name__,
            "device": device,
            "lstm_input_size": lstm_input_size,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers,
            "bidirectional": bidirectional,
            "batch_first": batch_first,
            "sequence_length": sequence_length,
            "temporal_features": temporal_features,
            "city_vocab_size": city_vocab_size,
            "city_embedding_dim": city_embedding_dim,
            "has_geography": geo_dim > 0,
            "geo_dim": geo_dim,
            "geo_features": geo_features,
            "fc_input_size": fc_input_size,
            "fc_output_size": fc_output_size,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "component_parameters": component_parameters,
            "component_parameter_details": component_parameter_details,
            "parameter_shapes": parameter_shapes,
            "parameter_counts": parameter_counts,
        }

        # 5. Validate that extracted model dimensions match configuration files
        validation = self.validate_against_config(summary)
        summary["config_validation"] = validation

        return summary

    def validate_against_config(self, summary: Dict[str, Any]) -> Dict[str, bool]:
        """
        Verifies that the dimensions extracted from the live PyTorch model
        agree with the saved V1/V2 configuration files.
        """
        checks: Dict[str, bool] = {}

        if self.model_key == "baseline":
            expected_input_size = int(self.v1_config["input_features"])
            expected_num_cities = int(self.v1_config["number_of_cities"])
            expected_seq_len = int(self.v1_config["sequence_length"])
            expected_hidden_size = int(self.v2_config["hidden_size"])
            expected_embedding_dim = int(self.v2_config["embedding_dim"])
            expected_geo_dim = 0
        else:
            expected_input_size = int(self.v2_config["input_size"])
            expected_num_cities = int(self.v2_config["num_cities"])
            expected_seq_len = int(self.v2_config["sequence_length"])
            expected_hidden_size = int(self.v2_config["hidden_size"])
            expected_embedding_dim = int(self.v2_config["embedding_dim"])
            expected_geo_dim = int(
                self.v2_config[self.model_key]["geo_dim"]
            )

        expected_fc_in = (
            expected_hidden_size + expected_embedding_dim + expected_geo_dim
        )

        checks["input_size_matches"] = (
            summary["lstm_input_size"] == expected_input_size
        )
        checks["hidden_size_matches"] = (
            summary["lstm_hidden_size"] == expected_hidden_size
        )
        checks["sequence_length_matches"] = (
            summary["sequence_length"] == expected_seq_len
        )
        checks["city_vocab_matches"] = (
            summary["city_vocab_size"] == expected_num_cities
            and summary["city_vocab_size"] == len(self.artifacts["city_to_id"])
        )
        checks["embedding_dim_matches"] = (
            summary["city_embedding_dim"] == expected_embedding_dim
        )
        checks["geo_dim_matches"] = (
            summary["geo_dim"] == expected_geo_dim
            and len(summary["geo_features"]) == expected_geo_dim
        )
        checks["fc_input_matches"] = (
            summary["fc_input_size"] == expected_fc_in
        )
        checks["fc_output_matches"] = summary["fc_output_size"] == 1
        checks["lstm_gate_packing_matches"] = (
            summary["parameter_shapes"]["lstm.weight_ih_l0"]
            == (4 * summary["lstm_hidden_size"], summary["lstm_input_size"])
            and summary["parameter_shapes"]["lstm.weight_hh_l0"]
            == (4 * summary["lstm_hidden_size"], summary["lstm_hidden_size"])
        )

        failed_checks = [
            name for name, passed in checks.items() if not passed
        ]
        if failed_checks:
            raise ValueError(
                f"Configuration validation failed for '{self.model_key}': "
                f"{failed_checks}"
            )

        return checks

    def get_tensor_flow(
        self,
        batch_size: int = 1,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates the stage-by-stage tensor dimensions and educational
        explanations for the given batch size.
        """
        if summary is None:
            summary = self.extract_summary()
        return calculate_tensor_flow(summary=summary, batch_size=batch_size)

    def get_component_details(
        self,
        component_id: str,
        batch_size: int = 1,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns complete Layer Inspector details for the selected component.
        """
        if summary is None:
            summary = self.extract_summary()
        return get_component_inspection_details(
            summary=summary,
            component_id=component_id,
            batch_size=batch_size,
        )

    def get_trained_weights_report(
        self,
        component_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts real trained weight statistics (shape, count, min, max, mean, std)
        from the loaded PyTorch model without modifying any parameters.
        """
        return extract_parameter_statistics(
            model=self.model,
            component_filter=component_filter,
        )


# ============================================================
# TENSOR DIMENSION ENGINE (CHECKPOINT 4)
# ============================================================

def calculate_tensor_flow(
    summary: Dict[str, Any],
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Calculates exact tensor dimensions at every stage of the forward pass
    using extracted model dimensions and a configurable batch size (default 1).

    Clearly distinguishes the full 30-day LSTM sequence output
    (batch, 30, 64) from the final Day-30 hidden state (batch, 64).
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(
            f"batch_size must be a positive integer, received {batch_size}."
        )

    B = int(batch_size)
    seq_len = int(summary["sequence_length"])
    in_features = int(summary["lstm_input_size"])
    hidden_size = int(summary["lstm_hidden_size"])
    has_geo = bool(summary["has_geography"])
    geo_dim = int(summary["geo_dim"])
    emb_dim = int(summary["city_embedding_dim"])
    fc_in = int(summary["fc_input_size"])
    fc_out = int(summary["fc_output_size"])

    weather_input_shape = (B, seq_len, in_features)
    lstm_sequence_output_shape = (B, seq_len, hidden_size)
    final_hidden_state_shape = (B, hidden_size)
    final_cell_state_shape = (B, hidden_size)
    geography_shape: Optional[Tuple[int, ...]] = (
        (B, geo_dim) if has_geo else None
    )
    city_embedding_shape = (B, emb_dim)
    concatenated_vector_shape = (B, fc_in)
    dense_output_shape = (B, fc_out)

    stages: List[Dict[str, Any]] = [
        {
            "key": "weather_input",
            "label": "Weather input",
            "shape": weather_input_shape,
            "axis_meaning": "(batch_size, sequence_length, temporal_features)",
            "description": (
                f"{B} sample(s) × {seq_len} consecutive days × "
                f"{in_features} scaled daily weather feature(s)."
            ),
            "beginner_explanation": (
                f"A {seq_len}-day calendar of scaled daily weather readings "
                f"({in_features} measurement(s) per day) fed into the LSTM one day at a time."
            ),
            "technical_explanation": (
                f"3D input tensor X of shape {weather_input_shape} (batch_first=True) "
                f"containing normalized temporal features: {summary['temporal_features']}."
            ),
        },
        {
            "key": "lstm_sequence_output",
            "label": "LSTM sequence output",
            "shape": lstm_sequence_output_shape,
            "axis_meaning": "(batch_size, sequence_length, hidden_size)",
            "description": (
                f"All {seq_len} daily hidden-state vectors produced step-by-step "
                f"across the {seq_len}-day window ({hidden_size} units per day)."
            ),
            "beginner_explanation": (
                f"As the LSTM reads each day from Day 1 to Day {seq_len}, it writes down "
                f"a {hidden_size}-number daily summary. Collecting all {seq_len} daily summaries "
                f"gives shape {lstm_sequence_output_shape}. Note: Only the final day's summary "
                f"is passed forward to the prediction head."
            ),
            "technical_explanation": (
                f"First return value `lstm_output` of `nn.LSTM`, shape {lstm_sequence_output_shape}, "
                f"containing h_t for every time step t = 1..{seq_len}. Distinct from `final_hidden_state`, "
                f"which slices only the last time step t = {seq_len}."
            ),
        },
        {
            "key": "final_hidden_state",
            "label": "Final hidden state",
            "shape": final_hidden_state_shape,
            "axis_meaning": "(batch_size, hidden_size)",
            "description": (
                f"The {hidden_size}-dimensional working summary after processing "
                f"Day {seq_len} (`hidden_state[-1]`)."
            ),
            "beginner_explanation": (
                f"The LSTM's final {hidden_size}-number summary at the end of Day {seq_len}, "
                f"distilling the entire {seq_len}-day weather trend into a single vector per sample."
            ),
            "technical_explanation": (
                f"Extracted via `hidden_state[-1]` from the final layer's hidden tensor "
                f"`h_n` of shape ({summary['lstm_num_layers']}, {B}, {hidden_size}), yielding "
                f"h_{seq_len} of shape {final_hidden_state_shape} (identical to `lstm_output[:, -1, :]`)."
            ),
        },
        {
            "key": "final_cell_state",
            "label": "Final cell state",
            "shape": final_cell_state_shape,
            "axis_meaning": "(batch_size, hidden_size)",
            "description": (
                f"The {hidden_size}-dimensional internal long-term memory conveyor "
                f"at Day {seq_len} (`cell_state[-1]`)."
            ),
            "beginner_explanation": (
                f"The internal memory notebook at Day {seq_len}. It carries long-term context "
                f"across the {seq_len} days and was used to compute the final hidden state."
            ),
            "technical_explanation": (
                f"Final cell state `c_n[-1]` of shape {final_cell_state_shape}. Updated recurrently "
                f"via c_t = f_t * c_{{t-1}} + i_t * g_t and used to compute h_t = o_t * tanh(c_t)."
            ),
        },
    ]

    if has_geo and geography_shape is not None:
        stages.append(
            {
                "key": "geography",
                "label": "Geography",
                "shape": geography_shape,
                "axis_meaning": "(batch_size, geo_dim)",
                "description": (
                    f"{geo_dim} static scaled geographical coordinates/physical attributes "
                    f"({', '.join(summary['geo_features'])}) that bypass the LSTM."
                ),
                "beginner_explanation": (
                    f"Static physical facts about the city ({', '.join(summary['geo_features'])}). "
                    f"Since a city's location and elevation do not change day-to-day, they skip "
                    f"the 30-day LSTM and join directly before the final prediction."
                ),
                "technical_explanation": (
                    f"2D auxiliary feature tensor of shape {geography_shape} standardized via "
                    f"`StandardScaler`. Bypasses recurrent gates and concatenates directly with `final_hidden`."
                ),
            }
        )

    stages.extend(
        [
            {
                "key": "city_embedding",
                "label": "City embedding",
                "shape": city_embedding_shape,
                "axis_meaning": "(batch_size, embedding_dim)",
                "description": (
                    f"Learned {emb_dim}-dimensional city identity vector looked up from "
                    f"the ({summary['city_vocab_size']}, {emb_dim}) embedding table."
                ),
                "beginner_explanation": (
                    f"An {emb_dim}-number 'personality profile' learned for each of the "
                    f"{summary['city_vocab_size']} Indian cities, capturing local micro-climate "
                    f"patterns. Like geography, it skips the LSTM and joins at concatenation."
                ),
                "technical_explanation": (
                    f"Output of `nn.Embedding({summary['city_vocab_size']}, {emb_dim})` indexed by "
                    f"`city_id` of shape ({B},), producing a dense vector of shape {city_embedding_shape}."
                ),
            },
            {
                "key": "concatenated_vector",
                "label": "Concatenated vector",
                "shape": concatenated_vector_shape,
                "axis_meaning": "(batch_size, combined_features)",
                "description": (
                    f"Combined feature vector of length {fc_in} formed by joining "
                    + (
                        f"final hidden state ({hidden_size}) + geography ({geo_dim}) + city embedding ({emb_dim})."
                        if has_geo
                        else f"final hidden state ({hidden_size}) + city embedding ({emb_dim})."
                    )
                ),
                "beginner_explanation": (
                    f"All clues glued side-by-side into one row of {fc_in} numbers: "
                    + (
                        f"{hidden_size} from the 30-day weather memory + {geo_dim} from geography + {emb_dim} from the city profile."
                        if has_geo
                        else f"{hidden_size} from the 30-day weather memory + {emb_dim} from the city profile."
                    )
                ),
                "technical_explanation": (
                    f"Result of `torch.cat(..., dim=1)` yielding shape {concatenated_vector_shape}: "
                    + (
                        f"`[final_hidden ({B}, {hidden_size}), geography ({B}, {geo_dim}), city_vector ({B}, {emb_dim})]`."
                        if has_geo
                        else f"`(final_hidden ({B}, {hidden_size}), city_vector ({B}, {emb_dim}))`."
                    )
                ),
            },
            {
                "key": "dense_output",
                "label": "Dense output",
                "shape": dense_output_shape,
                "axis_meaning": "(batch_size, output_dim)",
                "description": (
                    f"Single scaled next-day maximum temperature prediction per sample "
                    f"from `Linear({fc_in}, {fc_out})`."
                ),
                "beginner_explanation": (
                    f"The final weighted sum of all {fc_in} combined features plus a bias term, "
                    f"producing 1 scaled temperature prediction (which is then inverse-scaled to °C)."
                ),
                "technical_explanation": (
                    f"Output y_hat = combined @ W_fc^T + b_fc from `nn.Linear({fc_in}, {fc_out})`, "
                    f"shape {dense_output_shape} in normalized target space before inverse scaling."
                ),
            },
        ]
    )

    shapes_dict: Dict[str, Optional[Tuple[int, ...]]] = {
        "weather_input": weather_input_shape,
        "lstm_sequence_output": lstm_sequence_output_shape,
        "final_hidden_state": final_hidden_state_shape,
        "final_cell_state": final_cell_state_shape,
        "geography": geography_shape,
        "city_embedding": city_embedding_shape,
        "concatenated_vector": concatenated_vector_shape,
        "dense_output": dense_output_shape,
    }

    return {
        "model_key": summary["model_key"],
        "model_name": summary["model_name"],
        "batch_size": B,
        "shapes": shapes_dict,
        "stages": stages,
    }


def verify_runtime_tensor_shapes(
    model_key: str,
    artifacts: Dict[str, Any],
    batch_size: int = 1,
) -> Dict[str, Tuple[int, ...]]:
    """
    Executes the actual loaded PyTorch model submodules in `torch.no_grad()`
    mode and returns the real runtime tensor shapes to verify 100% agreement
    with `calculate_tensor_flow()`.
    """
    model = artifacts["models"][model_key]
    device = artifacts["device"]
    summary = inspect_model_architecture(model_key, artifacts)

    B = int(batch_size)
    seq_len = summary["sequence_length"]
    in_size = summary["lstm_input_size"]

    weather_input = torch.zeros(
        (B, seq_len, in_size),
        dtype=torch.float32,
        device=device,
    )
    city_ids = torch.zeros(
        (B,),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        lstm_output, (hidden_state, cell_state) = model.lstm(weather_input)
        final_hidden = hidden_state[-1]
        final_cell = cell_state[-1]
        city_vector = model.city_embedding(city_ids)

        if summary["has_geography"]:
            geography = torch.zeros(
                (B, summary["geo_dim"]),
                dtype=torch.float32,
                device=device,
            )
            combined = torch.cat(
                [final_hidden, geography, city_vector],
                dim=1,
            )
            geo_shape: Optional[Tuple[int, ...]] = tuple(geography.shape)
        else:
            combined = torch.cat(
                (final_hidden, city_vector),
                dim=1,
            )
            geo_shape = None

        dense_output = model.fc(combined)

    return {
        "weather_input": tuple(weather_input.shape),
        "lstm_sequence_output": tuple(lstm_output.shape),
        "final_hidden_state": tuple(final_hidden.shape),
        "final_cell_state": tuple(final_cell.shape),
        "geography": geo_shape,
        "city_embedding": tuple(city_vector.shape),
        "concatenated_vector": tuple(combined.shape),
        "dense_output": tuple(dense_output.shape),
    }


# ============================================================
# LAYER & COMPONENT INSPECTOR (CHECKPOINT 6)
# ============================================================

def get_component_inspection_details(
    summary: Dict[str, Any],
    component_id: str,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """
    Returns comprehensive Layer Inspector metadata for a selected component,
    including Beginner and Technical explanations, mathematical operations,
    actual parameter counts, and weight/bias shapes.
    """
    flow = calculate_tensor_flow(summary, batch_size=batch_size)
    shapes = flow["shapes"]

    seq_len = int(summary["sequence_length"])
    in_size = int(summary["lstm_input_size"])
    hidden_size = int(summary["lstm_hidden_size"])
    packed_rows = 4 * hidden_size
    has_geo = bool(summary["has_geography"])
    geo_dim = int(summary["geo_dim"])
    vocab_size = int(summary["city_vocab_size"])
    emb_dim = int(summary["city_embedding_dim"])
    fc_in = int(summary["fc_input_size"])
    fc_out = int(summary["fc_output_size"])
    param_shapes = summary["parameter_shapes"]
    comp_params = summary["component_parameters"]

    catalog: Dict[str, Dict[str, Any]] = {
        "weather_input": {
            "component_id": "weather_input",
            "component_name": "Weather Sequence Input",
            "category": "Temporal Input",
            "purpose": (
                f"Feeds {seq_len} consecutive days of scaled daily weather observations "
                f"({in_size} feature(s) per day) into the recurrent network."
            ),
            "input_dimensions": f"Raw DataFrame ({seq_len}, {in_size})",
            "output_dimensions": str(shapes["weather_input"]),
            "math_operation": (
                r"X_{\text{scaled}} = \text{MinMaxScaler}(X_{\text{raw}}) \in \mathbb{R}^{"
                + f"{batch_size} \\times {seq_len} \\times {in_size}"
                + r"}"
            ),
            "math_operation_text": (
                f"X_scaled = MinMaxScaler(X_raw) -> shape {shapes['weather_input']}"
            ),
            "parameter_count": 0,
            "weight_and_bias_shapes": {},
            "features_list": summary["temporal_features"],
            "beginner_explanation": (
                f"**Analogy:** Think of this as a {seq_len}-page weather diary leading up to tomorrow. "
                f"Each page represents one day and lists {in_size} measurement(s): "
                f"`{', '.join(summary['temporal_features'])}`.\n\n"
                f"**Plain-Language Dimensions `{shapes['weather_input']}`:**\n"
                f"- `{batch_size}`: Number of prediction examples processed at once.\n"
                f"- `{seq_len}`: 30 consecutive historical days (Day 1 to Day 30).\n"
                f"- `{in_size}`: Number of daily weather measurements recorded on each day.\n\n"
                f"**Why it exists:** Tomorrow's maximum temperature depends heavily on multi-day trends "
                f"(such as a 5-day warming spell or approaching monsoon rain) rather than just a single day."
            ),
            "technical_explanation": (
                f"**Forward-Pass Representation:** 3D tensor $X \\in \\mathbb{{R}}^{{{batch_size} \\times {seq_len} \\times {in_size}}}$ "
                f"arranged with `batch_first=True`.\n\n"
                f"- **Preprocessing:** Raw observations are transformed (including $\\log(1 + \\text{{precip}})$ and cyclical "
                f"$\\sin/\\cos$ day-of-year encoding in V2) and normalized via `MinMaxScaler` so all temporal inputs "
                f"lie in comparable numerical ranges before entering the LSTM's sigmoid/tanh gates.\n"
                f"- **Trainable Parameters:** `0` (input preprocessing uses fixed scaler statistics fit on the training split)."
            ),
        },
        "lstm": {
            "component_id": "lstm",
            "component_name": f"LSTM Recurrent Layer ({hidden_size} Hidden Units)",
            "category": "Recurrent Memory Core",
            "purpose": (
                f"Processes the {seq_len}-day weather sequence one day at a time, updating "
                f"a {hidden_size}-unit internal cell memory and emitting the Day-{seq_len} "
                f"Final Hidden State {shapes['final_hidden_state']}."
            ),
            "input_dimensions": str(shapes["weather_input"]),
            "output_dimensions": (
                f"Sequence: {shapes['lstm_sequence_output']} | "
                f"Final Hidden h_30: {shapes['final_hidden_state']} | "
                f"Final Cell c_30: {shapes['final_cell_state']}"
            ),
            "math_operation": (
                r"\begin{aligned}"
                r"i_t &= \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\"
                r"f_t &= \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\"
                r"g_t &= \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\"
                r"o_t &= \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\"
                r"c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \quad h_t = o_t \odot \tanh(c_t)"
                r"\end{aligned}"
            ),
            "math_operation_text": (
                "z_t = W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh -> split into (i_t, f_t, g_t, o_t)"
            ),
            "parameter_count": comp_params["lstm"],
            "weight_and_bias_shapes": {
                "W_ih_l0 (lstm.weight_ih_l0)": param_shapes["lstm.weight_ih_l0"],
                "W_hh_l0 (lstm.weight_hh_l0)": param_shapes["lstm.weight_hh_l0"],
                "b_ih_l0 (lstm.bias_ih_l0)": param_shapes["lstm.bias_ih_l0"],
                "b_hh_l0 (lstm.bias_hh_l0)": param_shapes["lstm.bias_hh_l0"],
            },
            "packed_matrix_info": {
                "W_ih_l0": param_shapes["lstm.weight_ih_l0"],
                "W_hh_l0": param_shapes["lstm.weight_hh_l0"],
                "b_ih_l0": param_shapes["lstm.bias_ih_l0"],
                "b_hh_l0": param_shapes["lstm.bias_hh_l0"],
                "packed_row_count": packed_rows,
                "single_gate_size": hidden_size,
                "packing_explanation": (
                    f"In PyTorch, `{packed_rows} = 4 × {hidden_size}` because the four gate transformations "
                    f"(Input `i`, Forget `f`, Candidate `g`, Output `o`, each with `{hidden_size}` units) "
                    f"are stacked vertically into a single matrix (`W_ih_l0` of shape `{param_shapes['lstm.weight_ih_l0']}` "
                    f"and `W_hh_l0` of shape `{param_shapes['lstm.weight_hh_l0']}`).\n"
                    f"- Rows `[0:64]` → Input Gate ($i_t$)\n"
                    f"- Rows `[64:128]` → Forget Gate ($f_t$)\n"
                    f"- Rows `[128:192]` → Candidate Gate ($g_t$)\n"
                    f"- Rows `[192:256]` → Output Gate ($o_t$)"
                ),
            },
            "gates_and_states": [
                {
                    "name": "Input Gate (i_t)",
                    "activation": "Sigmoid (0 to 1)",
                    "slice_rows": "[0:64]",
                    "w_ih_slice_shape": (hidden_size, in_size),
                    "w_hh_slice_shape": (hidden_size, hidden_size),
                    "beginner": (
                        "**The Write Filter:** Decides how strongly today's new weather reading should be "
                        "written into long-term memory (`0` = ignore today's update, `1` = fully record it)."
                    ),
                    "technical": (
                        f"$i_t = \\sigma(W_{{ii}} x_t + b_{{ii}} + W_{{hi}} h_{{t-1}} + b_{{hi}}) \\in (0, 1)^{{{hidden_size}}}$. "
                        f"Modulates the candidate activation $g_t$ via elementwise multiplication $i_t \\odot g_t$."
                    ),
                },
                {
                    "name": "Forget Gate (f_t)",
                    "activation": "Sigmoid (0 to 1)",
                    "slice_rows": "[64:128]",
                    "w_ih_slice_shape": (hidden_size, in_size),
                    "w_hh_slice_shape": (hidden_size, hidden_size),
                    "beginner": (
                        "**The Eraser / Retention Dial:** Decides how much of yesterday's long-term memory "
                        "still matters today (`1` = keep old trend, `0` = wipe outdated weather history)."
                    ),
                    "technical": (
                        f"$f_t = \\sigma(W_{{if}} x_t + b_{{if}} + W_{{hf}} h_{{t-1}} + b_{{hf}}) \\in (0, 1)^{{{hidden_size}}}$. "
                        f"Multiplies previous cell state $c_{{t-1}}$ directly, controlling gradient flow across time."
                    ),
                },
                {
                    "name": "Candidate Gate (g_t)",
                    "activation": "Tanh (-1 to +1)",
                    "slice_rows": "[128:192]",
                    "w_ih_slice_shape": (hidden_size, in_size),
                    "w_hh_slice_shape": (hidden_size, hidden_size),
                    "beginner": (
                        "**The New Note Draft:** Proposes new positive or negative adjustments (`-1` to `+1`) "
                        "based on today's weather before the Input Gate decides how much to keep."
                    ),
                    "technical": (
                        f"$g_t = \\tanh(W_{{ig}} x_t + b_{{ig}} + W_{{hg}} h_{{t-1}} + b_{{hg}}) \\in (-1, 1)^{{{hidden_size}}}$. "
                        f"Provides signed candidate content added to cell memory."
                    ),
                },
                {
                    "name": "Output Gate (o_t)",
                    "activation": "Sigmoid (0 to 1)",
                    "slice_rows": "[192:256]",
                    "w_ih_slice_shape": (hidden_size, in_size),
                    "w_hh_slice_shape": (hidden_size, hidden_size),
                    "beginner": (
                        "**The Read / Spokesperson Filter:** Chooses which parts of the internal memory notebook "
                        "are relevant to share right now as today's working summary (`h_t`)."
                    ),
                    "technical": (
                        f"$o_t = \\sigma(W_{{io}} x_t + b_{{io}} + W_{{ho}} h_{{t-1}} + b_{{ho}}) \\in (0, 1)^{{{hidden_size}}}$. "
                        f"Filters the squashed cell state $\\tanh(c_t)$ to form $h_t$."
                    ),
                },
                {
                    "name": "Cell State (c_t)",
                    "activation": "Linear Conveyor Belt",
                    "slice_rows": "(64,) state vector",
                    "w_ih_slice_shape": (hidden_size,),
                    "w_hh_slice_shape": (hidden_size,),
                    "beginner": (
                        "**Long-Term Memory Notebook:** Travels straight across all 30 days. Each day it keeps "
                        "a fraction of yesterday's notebook (`f_t * c_{t-1}`) and adds today's filtered note (`i_t * g_t`)."
                    ),
                    "technical": (
                        f"$c_t = f_t \\odot c_{{t-1}} + i_t \\odot g_t \\in \\mathbb{{R}}^{{{hidden_size}}}$. "
                        f"Additive update preserves long-range temporal gradients across the 30 steps."
                    ),
                },
                {
                    "name": "Hidden State (h_t)",
                    "activation": "o_t * tanh(c_t) (-1 to +1)",
                    "slice_rows": "(64,) state vector",
                    "w_ih_slice_shape": (hidden_size,),
                    "w_hh_slice_shape": (hidden_size,),
                    "beginner": (
                        "**Daily Working Summary:** The 64-number summary output at the end of each day. "
                        f"After Day {seq_len}, the final summary `h_30` of shape `{shapes['final_hidden_state']}` "
                        "is sent to the concatenation stage."
                    ),
                    "technical": (
                        f"$h_t = o_t \\odot \\tanh(c_t) \\in (-1, 1)^{{{hidden_size}}}$. "
                        f"Across 30 steps this forms `lstm_output` `{shapes['lstm_sequence_output']}`, while "
                        f"`hidden_state[-1]` extracts $h_{{30}} \\in \\mathbb{{R}}^{{{batch_size} \\times {hidden_size}}}$."
                    ),
                },
            ],
            "beginner_explanation": (
                f"**Analogy:** Imagine 64 weather analysts reading the 30-day diary page by page from Day 1 to Day 30. "
                f"Each analyst maintains a long-term notebook (**Cell State**) and a daily summary (**Hidden State**) "
                f"using four mental checkpoints (**Forget**, **Input**, **Candidate**, and **Output** gates).\n\n"
                f"**Sequence Output `{shapes['lstm_sequence_output']}` vs. Final Hidden State `{shapes['final_hidden_state']}`:**\n"
                f"- Across all 30 days, the analysts write 30 daily summaries → shape `{shapes['lstm_sequence_output']}`.\n"
                f"- Once Day 30 is finished, the model takes **only Day 30's final summary** (`{shapes['final_hidden_state']}`) "
                f"because it already distills the entire 30-day history."
            ),
            "technical_explanation": (
                f"**Parameter Count Derivation (`{comp_params['lstm']:,}` params):**\n"
                f"- Input-to-hidden weights `W_ih_l0`: `{param_shapes['lstm.weight_ih_l0']}` = `{packed_rows * in_size:,}`\n"
                f"- Hidden-to-hidden weights `W_hh_l0`: `{param_shapes['lstm.weight_hh_l0']}` = `{packed_rows * hidden_size:,}`\n"
                f"- Input bias `b_ih_l0`: `{param_shapes['lstm.bias_ih_l0']}` = `{packed_rows}`\n"
                f"- Hidden bias `b_hh_l0`: `{param_shapes['lstm.bias_hh_l0']}` = `{packed_rows}`\n"
                f"- Total: `{packed_rows * in_size} + {packed_rows * hidden_size} + {packed_rows} + {packed_rows} = {comp_params['lstm']:,}` trainable parameters."
            ),
        },
        "city_embedding": {
            "component_id": "city_embedding",
            "component_name": f"City Embedding Layer ({vocab_size} Cities × {emb_dim} Dims)",
            "category": "Learned Auxiliary Input (Bypass)",
            "purpose": (
                f"Maps a discrete integer city ID (0..{vocab_size - 1}) to a learned "
                f"{emb_dim}-dimensional continuous vector that bypasses the LSTM and joins at concatenation."
            ),
            "input_dimensions": f"({batch_size},) integer city_id",
            "output_dimensions": str(shapes["city_embedding"]),
            "math_operation": (
                r"e_{\text{city}} = E[\text{city\_id}] \in \mathbb{R}^{"
                + f"{batch_size} \\times {emb_dim}"
                + r"}, \quad E \in \mathbb{R}^{"
                + f"{vocab_size} \\times {emb_dim}"
                + r"}"
            ),
            "math_operation_text": (
                f"city_vector = E[city_id] -> row lookup in ({vocab_size}, {emb_dim}) matrix"
            ),
            "parameter_count": comp_params["city_embedding"],
            "weight_and_bias_shapes": {
                "city_embedding.weight": param_shapes["city_embedding.weight"],
            },
            "beginner_explanation": (
                f"**Analogy:** Every city has its own local personality (urban heat island, valley terrain, "
                f"monsoon timing). Instead of feeding a raw city ID number like `0` for Agartala or `9` for Chennai, "
                f"the model keeps a lookup table of **{vocab_size} rows and {emb_dim} columns** (`400` learned numbers).\n\n"
                f"**Why it bypasses the LSTM:** A city's identity does not change from Day 1 to Day 30! "
                f"So the model looks up the city's {emb_dim}-number profile once and merges it directly with the "
                f"LSTM's final summary."
            ),
            "technical_explanation": (
                f"**Lookup Operation:** `nn.Embedding(num_embeddings={vocab_size}, embedding_dim={emb_dim})` "
                f"stores weight matrix $E \\in \\mathbb{{R}}^{{{vocab_size} \\times {emb_dim}}}$ (`{comp_params['city_embedding']}` trainable parameters, no bias).\n\n"
                f"- Given integer index tensor `city_id` of shape `({batch_size},)`, forward execution performs "
                f"a gather operation returning $e_{{\\text{{city}}}} \\in \\mathbb{{R}}^{{{batch_size} \\times {emb_dim}}}$.\n"
                f"- **Architectural Routing:** Bypasses the recurrent layer completely and enters `torch.cat` along `dim=1`."
            ),
        },
        "concatenation": {
            "component_id": "concatenation",
            "component_name": "Feature Concatenation (torch.cat)",
            "category": "Feature Fusion",
            "purpose": (
                f"Fuses the temporal summary from the LSTM ({hidden_size} dims) with "
                + (
                    f"static geography ({geo_dim} dims) and city embedding ({emb_dim} dims) "
                    if has_geo
                    else f"the city embedding ({emb_dim} dims) "
                )
                + f"into a single {fc_in}-dimensional vector."
            ),
            "input_dimensions": (
                f"final_hidden {shapes['final_hidden_state']} + "
                f"geography {shapes['geography']} + "
                f"city_embedding {shapes['city_embedding']}"
                if has_geo
                else f"final_hidden {shapes['final_hidden_state']} + city_embedding {shapes['city_embedding']}"
            ),
            "output_dimensions": str(shapes["concatenated_vector"]),
            "math_operation": (
                r"z_{\text{combined}} = \text{concat}\big(h_{30},\; g,\; e_{\text{city}}\big) \in \mathbb{R}^{"
                + f"{batch_size} \\times {fc_in}"
                + r"}"
                if has_geo
                else (
                    r"z_{\text{combined}} = \text{concat}\big(h_{30},\; e_{\text{city}}\big) \in \mathbb{R}^{"
                    + f"{batch_size} \\times {fc_in}"
                    + r"}"
                )
            ),
            "math_operation_text": (
                f"torch.cat([final_hidden, geography, city_vector], dim=1) -> {shapes['concatenated_vector']}"
                if has_geo
                else f"torch.cat((final_hidden, city_vector), dim=1) -> {shapes['concatenated_vector']}"
            ),
            "parameter_count": 0,
            "weight_and_bias_shapes": {},
            "beginner_explanation": (
                f"**Analogy:** Before making the final forecast, the head meteorologist places all clues "
                f"side-by-side on one desk:\n"
                f"- **Indices `0..63` (`64` numbers):** What happened in the weather over the last 30 days (from the LSTM).\n"
                + (
                    f"- **Indices `64..{64 + geo_dim - 1}` (`{geo_dim}` numbers):** Where the city is physically located ({', '.join(summary['geo_features'])}).\n"
                    f"- **Indices `{64 + geo_dim}..{fc_in - 1}` (`{emb_dim}` numbers):** The city's learned 8-number climate profile.\n\n"
                    if has_geo
                    else f"- **Indices `64..71` (`{emb_dim}` numbers):** The city's learned 8-number climate profile.\n\n"
                )
                + f"Gluing them side-by-side creates a single row of **`{fc_in}` numbers** (`{shapes['concatenated_vector']}`)."
            ),
            "technical_explanation": (
                f"**Tensor Operation:** `torch.cat` along feature axis `dim=1` with `0` trainable parameters.\n\n"
                + (
                    f"- Exact concatenation order from `MultivariateWeatherLSTM.forward()`:\n"
                    f"  1. `final_hidden`: indices `[0:64]` shape `{shapes['final_hidden_state']}`\n"
                    f"  2. `geography`: indices `[64:{64 + geo_dim}]` shape `{shapes['geography']}`\n"
                    f"  3. `city_vector`: indices `[{64 + geo_dim}:{fc_in}]` shape `{shapes['city_embedding']}`\n"
                    if has_geo
                    else (
                        f"- Exact concatenation order from `WeatherLSTM.forward()`:\n"
                        f"  1. `final_hidden`: indices `[0:64]` shape `{shapes['final_hidden_state']}`\n"
                        f"  2. `city_vector`: indices `[64:72]` shape `{shapes['city_embedding']}`\n"
                    )
                )
                + f"- Resulting tensor shape: `{shapes['concatenated_vector']}`."
            ),
        },
        "fc": {
            "component_id": "fc",
            "component_name": f"Fully Connected Layer (Linear {fc_in} → {fc_out})",
            "category": "Dense Projection Layer",
            "purpose": (
                f"Computes a learned linear combination of all {fc_in} concatenated features "
                f"plus a scalar bias to output the scaled next-day maximum temperature prediction."
            ),
            "input_dimensions": str(shapes["concatenated_vector"]),
            "output_dimensions": str(shapes["dense_output"]),
            "math_operation": (
                r"\hat{y}_{\text{scaled}} = z_{\text{combined}} W_{\text{fc}}^{\top} + b_{\text{fc}} \in \mathbb{R}^{"
                + f"{batch_size} \\times {fc_out}"
                + r"}"
            ),
            "math_operation_text": (
                f"y_scaled = combined @ W_fc.T + b_fc  (W_fc: {param_shapes['fc.weight']}, b_fc: {param_shapes['fc.bias']})"
            ),
            "parameter_count": comp_params["fc"],
            "weight_and_bias_shapes": {
                "fc.weight": param_shapes["fc.weight"],
                "fc.bias": param_shapes["fc.bias"],
            },
            "feature_slice_map": (
                {
                    "final_hidden_weights": (0, 64),
                    "geography_weights": (64, 64 + geo_dim),
                    "city_embedding_weights": (64 + geo_dim, fc_in),
                }
                if has_geo
                else {
                    "final_hidden_weights": (0, 64),
                    "city_embedding_weights": (64, 72),
                }
            ),
            "beginner_explanation": (
                f"**Analogy:** This layer is the final scoring formula. It assigns an importance weight "
                f"to each of the `{fc_in}` numbers on the desk, multiplies each clue by its weight, adds them "
                f"all up, and adds 1 baseline offset (`bias`).\n\n"
                f"**Parameter Count (`{comp_params['fc']}` params):**\n"
                f"- `{fc_in}` weights in `fc.weight` `{param_shapes['fc.weight']}` (one per combined clue)\n"
                f"- `1` bias value in `fc.bias` `{param_shapes['fc.bias']}`\n"
                f"- Total = `{fc_in} + 1 = {comp_params['fc']}` trainable parameters."
            ),
            "technical_explanation": (
                f"**Affine Transformation:** `nn.Linear(in_features={fc_in}, out_features={fc_out})`.\n\n"
                f"- Weight matrix $W_{{\\text{{fc}}}} \\in \\mathbb{{R}}^{{1 \\times {fc_in}}}$ and bias $b_{{\\text{{fc}}}} \\in \\mathbb{{R}}^{{1}}$.\n"
                f"- Columns `[0:64]` weight the LSTM final hidden state $h_{{30}}$"
                + (
                    f", columns `[64:{64 + geo_dim}]` weight the geographical features `{summary['geo_features']}`, "
                    f"and columns `[{64 + geo_dim}:{fc_in}]` weight the city embedding vector."
                    if has_geo
                    else ", and columns `[64:72]` weight the city embedding vector."
                )
            ),
        },
        "output": {
            "component_id": "output",
            "component_name": "Predicted Next-Day Maximum Temperature",
            "category": "Prediction Output",
            "purpose": (
                "Outputs the model's scaled temperature prediction and converts it back to "
                "degrees Celsius (°C) via the saved target scaler."
            ),
            "input_dimensions": str(shapes["dense_output"]),
            "output_dimensions": f"{shapes['dense_output']} (scaled) → °C scalar",
            "math_operation": (
                r"\hat{T}_{t+1}\;(^\circ\text{C}) = \text{TargetScaler}^{-1}\big(\hat{y}_{\text{scaled}}\big)"
            ),
            "math_operation_text": (
                "T_celsius = target_scaler.inverse_transform(y_scaled)"
            ),
            "parameter_count": 0,
            "weight_and_bias_shapes": {},
            "beginner_explanation": (
                "**From Model Numbers to Real Celsius:** During training, temperatures were scaled down "
                "to a small `[0, 1]` range so the neural network could learn smoothly. The output node takes "
                "that scaled number and reverses the scaling formula to give you the actual forecast in **°C**."
            ),
            "technical_explanation": (
                f"**Inverse Scaling:** The dense layer emits $\\hat{{y}}_{{\\text{{scaled}}}} \\in \\mathbb{{R}}^{{{batch_size} \\times 1}}$. "
                f"`WeatherInference.inverse_transform_prediction()` applies the fitted `MinMaxScaler.inverse_transform` "
                f"(`v1['target_scaler']` for Baseline V1; `v2['target_scaler']` for Model A and Model B) to recover "
                f"the physical prediction in degrees Celsius."
            ),
        },
    }

    if has_geo:
        catalog["geography"] = {
            "component_id": "geography",
            "component_name": f"Geography Input ({geo_dim} Static Features)",
            "category": "Static Auxiliary Input (Bypass)",
            "purpose": (
                f"Supplies {geo_dim} standardized physical coordinates/attributes "
                f"({', '.join(summary['geo_features'])}) directly to the concatenation stage."
            ),
            "input_dimensions": f"City row ({geo_dim} features)",
            "output_dimensions": str(shapes["geography"]),
            "math_operation": (
                r"g = \text{StandardScaler}(g_{\text{raw}}) \in \mathbb{R}^{"
                + f"{batch_size} \\times {geo_dim}"
                + r"}"
            ),
            "math_operation_text": (
                f"g = geo_scaler.transform(g_raw) -> shape {shapes['geography']}"
            ),
            "parameter_count": 0,
            "weight_and_bias_shapes": {},
            "features_list": summary["geo_features"],
            "beginner_explanation": (
                f"**Analogy:** A city's physical geography—its **{', '.join(summary['geo_features'])}**—is "
                f"crucial for temperature (e.g., high-elevation Bengaluru is cooler; coastal Chennai is moderated "
                f"by the sea).\n\n"
                f"**Why it bypasses the LSTM:** A city's latitude, longitude, elevation, and coastal distance "
                f"do not change day by day! Feeding unchanging numbers into 30 daily LSTM steps would be redundant, "
                f"so they take a shortcut straight to the Concatenation block."
            ),
            "technical_explanation": (
                f"**Auxiliary Static Input:** $g \\in \\mathbb{{R}}^{{{batch_size} \\times {geo_dim}}}$ containing "
                f"`{summary['geo_features']}` scaled via `geo_scaler_A` (3 features) or `geo_scaler_B` (4 features).\n\n"
                f"- **Trainable Parameters:** `0` at the input stage; its linear projection weights live in columns "
                f"`[64:{64 + geo_dim}]` of `fc.weight`."
            ),
        }

    if component_id not in catalog:
        component_id = "lstm"

    return catalog[component_id]


# ============================================================
# ACTUAL TRAINED WEIGHTS INSPECTOR (CHECKPOINT 6)
# ============================================================

def extract_parameter_statistics(
    model: nn.Module,
    component_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Computes exact descriptive statistics (shape, num_values, min, max, mean, std)
    for every actual trained parameter tensor in the loaded PyTorch model.
    """
    stats_list: List[Dict[str, Any]] = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            component = name.split(".")[0]
            if component_filter is not None and component != component_filter:
                continue

            arr = param.detach().cpu().numpy().astype(np.float64)
            stats_list.append(
                {
                    "parameter_name": name,
                    "component": component,
                    "shape": tuple(int(d) for d in arr.shape),
                    "num_values": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
            )

    return stats_list


def get_lstm_gate_weight_statistics(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Splits the packed LSTM parameter tensors (`weight_ih_l0`, `weight_hh_l0`,
    `bias_ih_l0`, `bias_hh_l0`) into their 4 constituent gates
    (Input `i`, Forget `f`, Candidate `g`, Output `o`) and returns real
    statistics for each gate's sub-tensors.
    """
    gate_labels = [
        ("i", "Input Gate (i)", "Rows [0:64]"),
        ("f", "Forget Gate (f)", "Rows [64:128]"),
        ("g", "Candidate Gate (g)", "Rows [128:192]"),
        ("o", "Output Gate (o)", "Rows [192:256]"),
    ]

    with torch.no_grad():
        w_ih_chunks = model.lstm.weight_ih_l0.detach().cpu().chunk(4, dim=0)
        w_hh_chunks = model.lstm.weight_hh_l0.detach().cpu().chunk(4, dim=0)
        b_ih_chunks = model.lstm.bias_ih_l0.detach().cpu().chunk(4, dim=0)
        b_hh_chunks = model.lstm.bias_hh_l0.detach().cpu().chunk(4, dim=0)

    records: List[Dict[str, Any]] = []
    for idx, (code, label, row_slice) in enumerate(gate_labels):
        for tensor_label, chunk in [
            (f"W_ih [{code}]", w_ih_chunks[idx]),
            (f"W_hh [{code}]", w_hh_chunks[idx]),
            (f"b_ih [{code}]", b_ih_chunks[idx]),
            (f"b_hh [{code}]", b_hh_chunks[idx]),
        ]:
            arr = chunk.numpy().astype(np.float64)
            records.append(
                {
                    "gate_code": code,
                    "gate_name": label,
                    "packed_rows": row_slice,
                    "parameter": tensor_label,
                    "shape": tuple(int(d) for d in arr.shape),
                    "num_values": int(arr.size),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
            )
    return records


def slice_parameter_matrix(
    model: nn.Module,
    param_name: str,
    row_start: int = 0,
    row_end: Optional[int] = None,
    col_start: int = 0,
    col_end: Optional[int] = None,
    summary: Optional[Dict[str, Any]] = None,
    city_to_id: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Extracts a controlled 2D slice of a real trained parameter tensor along
    with human-readable row and column labels for safe tabular display and
    interactive heatmap visualization.
    """
    param_dict = dict(model.named_parameters())
    if param_name not in param_dict:
        raise ValueError(
            f"Parameter '{param_name}' not found. Available: {list(param_dict.keys())}"
        )

    with torch.no_grad():
        raw_arr = param_dict[param_name].detach().cpu().numpy()

    original_shape = tuple(int(d) for d in raw_arr.shape)

    # Convert 1D bias vectors (N,) into a 2D (1, N) row matrix for uniform slicing/heatmap
    if raw_arr.ndim == 1:
        matrix_2d = raw_arr.reshape(1, -1)
    else:
        matrix_2d = raw_arr

    total_rows, total_cols = matrix_2d.shape

    r0 = max(0, min(int(row_start), total_rows - 1))
    r1 = total_rows if row_end is None else max(r0 + 1, min(int(row_end), total_rows))
    c0 = max(0, min(int(col_start), total_cols - 1))
    c1 = total_cols if col_end is None else max(c0 + 1, min(int(col_end), total_cols))

    sub_matrix = matrix_2d[r0:r1, c0:c1]

    id_to_city = (
        {v: k for k, v in city_to_id.items()} if city_to_id is not None else {}
    )

    def _lstm_row_label(idx: int) -> str:
        gates = ["i (Input)", "f (Forget)", "g (Cand)", "o (Output)"]
        g_idx = min(idx // 64, 3)
        u_idx = (idx % 64) + 1
        return f"Row {idx} [{gates[g_idx]} U{u_idx}]"

    if param_name == "city_embedding.weight":
        row_labels = [
            f"{r}: {id_to_city.get(r, f'City {r}')}" for r in range(r0, r1)
        ]
        col_labels = [f"Emb Dim {c + 1}" for c in range(c0, c1)]
        xaxis_title = "Embedding Dimension"
        yaxis_title = "City (ID: Name)"
    elif param_name.startswith("lstm.weight_ih"):
        row_labels = [_lstm_row_label(r) for r in range(r0, r1)]
        if summary and len(summary.get("temporal_features", [])) == total_cols:
            feat_names = summary["temporal_features"]
            col_labels = [f"{c}: {feat_names[c]}" for c in range(c0, c1)]
        else:
            col_labels = [f"Feat {c + 1}" for c in range(c0, c1)]
        xaxis_title = "Input Temporal Feature"
        yaxis_title = "Packed Gate Row (256 = 4 × 64)"
    elif param_name.startswith("lstm.weight_hh"):
        row_labels = [_lstm_row_label(r) for r in range(r0, r1)]
        col_labels = [f"h_prev U{c + 1}" for c in range(c0, c1)]
        xaxis_title = "Previous Hidden Unit (h_{t-1})"
        yaxis_title = "Packed Gate Row (256 = 4 × 64)"
    elif param_name.startswith("lstm.bias_"):
        row_labels = [param_name]
        col_labels = [_lstm_row_label(c) for c in range(c0, c1)]
        xaxis_title = "Packed Gate Bias Index (256 = 4 × 64)"
        yaxis_title = "Bias Vector"
    elif param_name == "fc.weight":
        row_labels = ["Output Unit 1"]
        col_labels = []
        geo_feats = summary.get("geo_features", []) if summary else []
        geo_dim = len(geo_feats)
        for c in range(c0, c1):
            if c < 64:
                col_labels.append(f"{c}: LSTM h[{c + 1}]")
            elif c < 64 + geo_dim:
                col_labels.append(f"{c}: Geo ({geo_feats[c - 64]})")
            else:
                emb_idx = c - (64 + geo_dim) + 1
                col_labels.append(f"{c}: CityEmb[{emb_idx}]")
        xaxis_title = "Concatenated Input Feature Index"
        yaxis_title = "Dense Output"
    else:
        row_labels = [f"Row {r}" for r in range(r0, r1)]
        col_labels = [f"Col {c}" for c in range(c0, c1)]
        xaxis_title = "Column"
        yaxis_title = "Row"

    df_slice = pd.DataFrame(
        sub_matrix,
        index=row_labels,
        columns=col_labels,
    )

    return {
        "parameter_name": param_name,
        "original_shape": original_shape,
        "matrix_2d_shape": (total_rows, total_cols),
        "row_range": (r0, r1),
        "col_range": (c0, c1),
        "slice_shape": sub_matrix.shape,
        "matrix": sub_matrix,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "xaxis_title": xaxis_title,
        "yaxis_title": yaxis_title,
        "dataframe": df_slice,
    }


# ============================================================
# REAL HISTORICAL WINDOW & NEURON GATE INSPECTOR (CHECKPOINT 7)
# ============================================================

_CACHED_WEATHER_DF: Optional[pd.DataFrame] = None


def load_historical_weather_dataframe() -> pd.DataFrame:
    """
    Loads and caches the merged historical Indian city weather dataset
    (`data/processed/india_weather_merged.csv`) for real 30-day window selection.
    """
    global _CACHED_WEATHER_DF
    if _CACHED_WEATHER_DF is None:
        if not WEATHER_DATA_PATH.exists():
            raise FileNotFoundError(
                f"Historical weather file not found: {WEATHER_DATA_PATH}"
            )
        df = pd.read_csv(WEATHER_DATA_PATH)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["city", "date"]).reset_index(drop=True)
        _CACHED_WEATHER_DF = df
    return _CACHED_WEATHER_DF


def get_city_30day_window(
    city: str,
    end_date: Optional[str] = "2025-10-19",
    weather_df: Optional[pd.DataFrame] = None,
    sequence_length: int = 30,
) -> pd.DataFrame:
    """
    Selects a real 30-day historical weather window for a given city up to
    and including `end_date` (defaults to `2025-10-19`, predicting `2025-10-20`).
    """
    if weather_df is None:
        weather_df = load_historical_weather_dataframe()

    city_df = weather_df[weather_df["city"] == city].sort_values("date")
    if len(city_df) < sequence_length:
        raise ValueError(
            f"City '{city}' has fewer than {sequence_length} observations."
        )

    if end_date is not None:
        cutoff = pd.to_datetime(end_date)
        city_df = city_df[city_df["date"] <= cutoff]

    if len(city_df) < sequence_length:
        raise ValueError(
            f"Not enough historical rows for '{city}' on or before {end_date}."
        )

    return city_df.tail(sequence_length).copy().reset_index(drop=True)


def compute_real_gate_history(
    model_key: str,
    artifacts: Dict[str, Any],
    city: str = "Agartala",
    history_df: Optional[pd.DataFrame] = None,
    weather_tensor: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Reuses `LSTMGateEngine` (`src/lstm_engine.py`) and `prepare_gate_data`
    (`src/visualization.py`) to compute the 6 gate/state matrices of shape
    (30, 64) on a real 30-day scaled model input.

    Also verifies numerical agreement between `LSTMGateEngine` and PyTorch's
    native `model.lstm`.
    """
    model = artifacts["models"][model_key]
    inference = WeatherInference(artifacts)

    dates_list: List[str] = []

    if weather_tensor is None:
        if history_df is None:
            history_df = get_city_30day_window(city=city)
        prepared_df, date_col = inference.prepare_history(
            city=city,
            history_df=history_df,
            model_name=model_key,
        )
        dates_list = (
            prepared_df[date_col].dt.strftime("%Y-%m-%d").tolist()
        )
        if model_key == "baseline":
            weather_tensor = inference.prepare_baseline_input(prepared_df)
        else:
            weather_tensor = inference.prepare_temporal_input(
                prepared_df, date_col
            )
    else:
        if history_df is not None and "date" in history_df.columns:
            dates_list = (
                pd.to_datetime(history_df["date"])
                .dt.strftime("%Y-%m-%d")
                .tolist()
            )
        else:
            dates_list = [f"Day {d}" for d in range(1, 31)]

    # Run LSTMGateEngine on a read-only CPU copy (as verified in test_lstm_engine.py
    # to avoid cuDNN TF32 fused-kernel rounding differences on GPU while keeping
    # the original model in `artifacts["models"]` untouched on its target device).
    cpu_model = copy.deepcopy(model).cpu().eval()
    if weather_tensor.ndim == 3:
        sequence_2d = weather_tensor.detach().squeeze(0).cpu()
        batched_input = weather_tensor.detach().cpu()
    elif weather_tensor.ndim == 2:
        sequence_2d = weather_tensor.detach().cpu()
        batched_input = sequence_2d.unsqueeze(0)
    else:
        raise ValueError(
            f"Expected weather_tensor with 2 or 3 dims, got {weather_tensor.shape}"
        )

    # Reuse existing LSTMGateEngine and prepare_gate_data
    engine = LSTMGateEngine(cpu_model)
    engine.split_gate_weights(print_shapes=False)

    with torch.no_grad():
        history, manual_h, manual_c = engine.process_sequence(sequence_2d)
        pytorch_out, (pytorch_h, pytorch_c) = cpu_model.lstm(batched_input)

    # Verify agreement with native PyTorch LSTM
    max_h_diff = float(
        torch.max(torch.abs(manual_h - pytorch_h[-1, 0, :])).item()
    )
    max_c_diff = float(
        torch.max(torch.abs(manual_c - pytorch_c[-1, 0, :])).item()
    )

    gate_data = prepare_gate_data(history)

    return {
        "model_key": model_key,
        "model_name": MODEL_DISPLAY_NAMES[model_key],
        "city": city,
        "dates": dates_list,
        "weather_tensor": batched_input,
        "sequence_shape": tuple(sequence_2d.shape),
        "gate_data": gate_data,
        "final_hidden_state": manual_h.detach().cpu().numpy(),
        "final_cell_state": manual_c.detach().cpu().numpy(),
        "max_hidden_diff_vs_pytorch": max_h_diff,
        "max_cell_diff_vs_pytorch": max_c_diff,
    }


def extract_single_neuron_series(
    gate_data: Dict[str, np.ndarray],
    neuron_number: int = 1,
    dates: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extracts the 6 time-series arrays (length 30) and summary statistics for
    a single selected neuron (`neuron_number` in 1..64).
    """
    if not (1 <= int(neuron_number) <= 64):
        raise ValueError(
            f"neuron_number must be between 1 and 64, received {neuron_number}."
        )

    idx = int(neuron_number) - 1
    seq_len = gate_data["input_gate"].shape[0]
    if dates is None or len(dates) != seq_len:
        dates = [f"Day {d}" for d in range(1, seq_len + 1)]

    series: Dict[str, np.ndarray] = {}
    stats: List[Dict[str, Any]] = []

    display_labels = {
        "input_gate": "Input Gate (i_t)",
        "forget_gate": "Forget Gate (f_t)",
        "candidate_gate": "Candidate Gate (g_t)",
        "output_gate": "Output Gate (o_t)",
        "cell_state": "Cell State (c_t)",
        "hidden_state": "Hidden State (h_t)",
    }

    for key in GATE_KEYS:
        vec = gate_data[key][:, idx].astype(np.float64)
        series[key] = vec
        stats.append(
            {
                "key": key,
                "signal": display_labels[key],
                "day_1": float(vec[0]),
                "day_30": float(vec[-1]),
                "min": float(np.min(vec)),
                "max": float(np.max(vec)),
                "mean": float(np.mean(vec)),
                "std": float(np.std(vec)),
            }
        )

    df = pd.DataFrame(
        {
            "Day": list(range(1, seq_len + 1)),
            "Date": dates,
            "input_gate": series["input_gate"],
            "forget_gate": series["forget_gate"],
            "candidate_gate": series["candidate_gate"],
            "output_gate": series["output_gate"],
            "cell_state": series["cell_state"],
            "hidden_state": series["hidden_state"],
        }
    )

    return {
        "neuron_number": int(neuron_number),
        "neuron_index": idx,
        "dates": dates,
        "series": series,
        "statistics": stats,
        "dataframe": df,
    }


# ============================================================
# STEP-BY-STEP PREDICTION WALKTHROUGH (CHECKPOINT 8)
# ============================================================

def get_city_date_bounds(
    city: str,
    weather_df: Optional[pd.DataFrame] = None,
    sequence_length: int = 30,
) -> Dict[str, Any]:
    """
    Returns the earliest and latest valid 30-day window end-dates for a city,
    along with curated seasonal preset dates.
    """
    if weather_df is None:
        weather_df = load_historical_weather_dataframe()

    city_df = weather_df[weather_df["city"] == city].sort_values("date").reset_index(drop=True)
    if len(city_df) < sequence_length:
        raise ValueError(f"City '{city}' has fewer than {sequence_length} rows.")

    min_end_date = city_df["date"].iloc[sequence_length - 1]
    max_end_date = city_df["date"].iloc[-1]

    presets = {
        "Test Split Benchmark (Ends 2025-10-19 → Predicts 2025-10-20)": "2025-10-19",
        "Pre-Monsoon Heatwave (Ends 2025-05-15 → Predicts 2025-05-16)": "2025-05-15",
        "Active Monsoon Window (Ends 2025-07-20 → Predicts 2025-07-21)": "2025-07-20",
        "Winter Cool Period (Ends 2025-01-15 → Predicts 2025-01-16)": "2025-01-15",
    }

    return {
        "city": city,
        "min_end_date": min_end_date.strftime("%Y-%m-%d"),
        "max_end_date": max_end_date.strftime("%Y-%m-%d"),
        "total_days": len(city_df),
        "presets": presets,
    }


def run_prediction_walkthrough(
    model_key: str,
    city: str,
    artifacts: Dict[str, Any],
    end_date: Optional[str] = "2025-10-19",
    history_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Executes a complete 10-stage forward-pass prediction walkthrough using
    the actual trained PyTorch model and the existing `WeatherInference`
    preprocessing and inverse-scaling pipeline.

    Captures all 10 required stages:
      1. Original weather observations (30-day raw DataFrame)
      2. Preprocessed temporal features (30-day engineered feature DataFrame)
      3. Scaled input tensor (1, 30, 1) or (1, 30, 8)
      4. Geography tensor (None, (1, 3), or (1, 4)) + raw geography values
      5. City embedding (1, 8) + city_id
      6. LSTM sequence output (1, 30, 64)
      7. Final hidden state (1, 64) and final cell state (1, 64)
      8. Concatenated vector (1, 72), (1, 75), or (1, 76)
      9. Dense layer output (1, 1) in internal scaled space + contribution breakdown
     10. Inverse-scaled prediction in Celsius (°C) verified against `predict_next_day()`
    """
    inference = WeatherInference(artifacts)
    model = artifacts["models"][model_key]
    summary = inspect_model_architecture(model_key, artifacts)

    if history_df is None:
        history_df = get_city_30day_window(city=city, end_date=end_date)

    # 1. Stage 1: Original 30-day weather observations validated via WeatherInference
    raw_30_df, date_col = inference.prepare_history(
        city=city,
        history_df=history_df,
        model_name=model_key,
    )
    dates_list = raw_30_df[date_col].dt.strftime("%Y-%m-%d").tolist()
    last_obs_date = raw_30_df[date_col].iloc[-1]
    prediction_date_ts = last_obs_date + pd.Timedelta(days=1)
    prediction_date_str = prediction_date_ts.strftime("%Y-%m-%d")

    raw_display_cols = [
        c
        for c in [
            date_col,
            "city",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]
        if c in raw_30_df.columns
    ]
    stage1_raw_observations_df = raw_30_df[raw_display_cols].copy()

    # Check if actual next-day max temperature exists in the historical dataset
    actual_next_day_temp: Optional[float] = None
    try:
        full_weather_df = load_historical_weather_dataframe()
        match_row = full_weather_df[
            (full_weather_df["city"] == city)
            & (full_weather_df["date"] == prediction_date_ts)
        ]
        if len(match_row) == 1:
            actual_next_day_temp = float(
                match_row["temperature_2m_max"].iloc[0]
            )
    except Exception:
        actual_next_day_temp = None

    # 2. Stage 2 & Stage 3: Preprocessed temporal features & Scaled input tensor
    if model_key == "baseline":
        stage2_preprocessed_df = raw_30_df[["temperature_2m_max"]].astype(
            np.float32
        ).copy()
        stage2_preprocessed_df.insert(0, "date", dates_list)
        weather_tensor = inference.prepare_baseline_input(raw_30_df)
    else:
        temp_df = raw_30_df.copy()
        temp_df["precipitation_log"] = np.log1p(
            temp_df["precipitation_sum"].clip(lower=0)
        )
        day_of_year = temp_df[date_col].dt.dayofyear
        days_in_year = np.where(temp_df[date_col].dt.is_leap_year, 366, 365)
        angle = 2 * np.pi * (day_of_year - 1) / days_in_year
        temp_df["day_of_year_sin"] = np.sin(angle)
        temp_df["day_of_year_cos"] = np.cos(angle)

        stage2_preprocessed_df = (
            temp_df[inference.temporal_features].astype(np.float32).copy()
        )
        stage2_preprocessed_df.insert(0, "date", dates_list)
        weather_tensor = inference.prepare_temporal_input(raw_30_df, date_col)

    scaled_input_np = weather_tensor.detach().cpu().numpy()
    stage3_scaled_df = pd.DataFrame(
        scaled_input_np[0],
        columns=summary["temporal_features"],
    )
    stage3_scaled_df.insert(0, "date", dates_list)

    # 4. Stage 4: Geography tensor (if applicable)
    if summary["has_geography"]:
        geography_tensor = inference.prepare_geography_input(city, model_key)
        geography_scaled_np = geography_tensor.detach().cpu().numpy()
        city_geo_row = inference.geography_df[
            inference.geography_df["city"] == city
        ]
        geography_raw_df = city_geo_row[summary["geo_features"]].copy()
        stage4_geography = {
            "has_geography": True,
            "shape": tuple(geography_tensor.shape),
            "feature_names": summary["geo_features"],
            "raw_values": {
                col: float(geography_raw_df[col].iloc[0])
                for col in summary["geo_features"]
            },
            "scaled_values": {
                col: float(geography_scaled_np[0, i])
                for i, col in enumerate(summary["geo_features"])
            },
            "scaled_array": geography_scaled_np,
        }
    else:
        geography_tensor = None
        stage4_geography = {
            "has_geography": False,
            "shape": None,
            "feature_names": [],
            "raw_values": {},
            "scaled_values": {},
            "scaled_array": None,
        }

    # 5. Stage 5: City embedding
    city_id = int(inference.city_to_id[city])
    city_tensor = torch.tensor(
        [city_id],
        dtype=torch.long,
        device=inference.device,
    )

    # 6–9. Execute forward pass step-by-step on the actual loaded model
    with torch.no_grad():
        city_vector = model.city_embedding(city_tensor)
        lstm_output, (hidden_state, cell_state) = model.lstm(weather_tensor)
        final_hidden = hidden_state[-1]
        final_cell = cell_state[-1]

        if summary["has_geography"] and geography_tensor is not None:
            combined = torch.cat(
                [final_hidden, geography_tensor, city_vector],
                dim=1,
            )
        else:
            combined = torch.cat(
                (final_hidden, city_vector),
                dim=1,
            )

        dense_output = model.fc(combined)

        # Compute exact linear contribution breakdown of the dense layer
        fc_weight = model.fc.weight.detach().cpu().numpy()[0]  # (fc_in,)
        fc_bias = float(model.fc.bias.detach().cpu().numpy()[0])
        combined_np = combined.detach().cpu().numpy()[0]  # (fc_in,)
        elementwise_prod = combined_np * fc_weight

    lstm_contrib = float(np.sum(elementwise_prod[:64]))
    geo_dim = summary["geo_dim"]
    geo_contrib = (
        float(np.sum(elementwise_prod[64 : 64 + geo_dim]))
        if geo_dim > 0
        else 0.0
    )
    city_contrib = float(np.sum(elementwise_prod[64 + geo_dim :]))

    scaled_prediction = float(dense_output.item())

    # 10. Stage 10: Inverse-scaled prediction in Celsius & cross-check with predict_next_day
    celsius_prediction = inference.inverse_transform_prediction(
        scaled_prediction,
        model_key,
    )
    official_result = inference.predict_next_day(
        city=city,
        history_df=history_df,
        model_name=model_key,
    )

    if abs(celsius_prediction - official_result["predicted_max_temperature"]) > 1e-5:
        raise RuntimeError(
            "Walkthrough prediction diverged from WeatherInference.predict_next_day()."
        )

    lstm_seq_np = lstm_output.detach().cpu().numpy()  # (1, 30, 64)
    final_hidden_np = final_hidden.detach().cpu().numpy()  # (1, 64)
    final_cell_np = final_cell.detach().cpu().numpy()  # (1, 64)
    city_emb_np = city_vector.detach().cpu().numpy()  # (1, 8)
    concat_np = combined.detach().cpu().numpy()  # (1, fc_in)
    dense_np = dense_output.detach().cpu().numpy()  # (1, 1)

    return {
        "model_key": model_key,
        "model_name": summary["model_name"],
        "city": city,
        "city_id": city_id,
        "window_start_date": dates_list[0],
        "last_observation_date": dates_list[-1],
        "prediction_date": prediction_date_str,
        "dates": dates_list,
        "weather_tensor": weather_tensor,
        "stage1_raw_observations": stage1_raw_observations_df,
        "stage2_preprocessed_features": stage2_preprocessed_df,
        "stage3_scaled_input": {
            "shape": tuple(scaled_input_np.shape),
            "array": scaled_input_np,
            "dataframe": stage3_scaled_df,
        },
        "stage4_geography": stage4_geography,
        "stage5_city_embedding": {
            "city": city,
            "city_id": city_id,
            "shape": tuple(city_emb_np.shape),
            "vector": city_emb_np[0],
        },
        "stage6_lstm_output": {
            "shape": tuple(lstm_seq_np.shape),
            "array": lstm_seq_np,
            "day1_first8": lstm_seq_np[0, 0, :8],
            "day30_first8": lstm_seq_np[0, -1, :8],
        },
        "stage7_final_states": {
            "hidden_shape": tuple(final_hidden_np.shape),
            "cell_shape": tuple(final_cell_np.shape),
            "final_hidden": final_hidden_np[0],
            "final_cell": final_cell_np[0],
        },
        "stage8_concatenated_vector": {
            "shape": tuple(concat_np.shape),
            "vector": concat_np[0],
            "segments": {
                "final_hidden_slice": (0, 64),
                "geography_slice": (64, 64 + geo_dim) if geo_dim > 0 else None,
                "city_embedding_slice": (64 + geo_dim, summary["fc_input_size"]),
            },
        },
        "stage9_dense_output": {
            "shape": tuple(dense_np.shape),
            "scaled_prediction": scaled_prediction,
            "contributions": {
                "lstm_hidden_64_sum": lstm_contrib,
                "geography_sum": geo_contrib,
                "city_embedding_8_sum": city_contrib,
                "fc_bias": fc_bias,
                "total_scaled_sum": lstm_contrib + geo_contrib + city_contrib + fc_bias,
            },
        },
        "stage10_celsius_prediction": {
            "predicted_max_temperature_celsius": celsius_prediction,
            "actual_max_temperature_celsius": actual_next_day_temp,
            "absolute_error_celsius": (
                abs(celsius_prediction - actual_next_day_temp)
                if actual_next_day_temp is not None
                else None
            ),
            "unit": "Celsius",
            "official_inference_result": official_result,
        },
    }


def inspect_model_architecture(
    model_key: str,
    artifacts: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convenience function to inspect a single loaded model by key
    ('baseline', 'model_A', or 'model_B') using the shared artifacts dictionary.
    """
    model = artifacts["models"][model_key]
    inspector = ArchitectureInspector(
        model=model,
        model_key=model_key,
        artifacts=artifacts,
    )
    return inspector.extract_summary()


def inspect_all_models(
    artifacts: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Extracts and validates architecture summaries for all three trained models.
    """
    summaries = {}
    for model_key in ["baseline", "model_A", "model_B"]:
        summaries[model_key] = inspect_model_architecture(
            model_key=model_key,
            artifacts=artifacts,
        )
    return summaries


if __name__ == "__main__":
    from src.artifacts import load_all_artifacts

    loaded_artifacts = load_all_artifacts()
    gate_result = compute_real_gate_history(
        model_key="model_B",
        artifacts=loaded_artifacts,
        city="Agartala",
    )
    neuron_info = extract_single_neuron_series(
        gate_result["gate_data"],
        neuron_number=25,
        dates=gate_result["dates"],
    )
    print("\nNeuron 25 Gate & State Summary:")
    for item in neuron_info["statistics"]:
        print(
            f"  {item['signal']:22s} | Day1={item['day_1']:+.4f} | "
            f"Day30={item['day_30']:+.4f} | Min={item['min']:+.4f} | Max={item['max']:+.4f}"
        )
