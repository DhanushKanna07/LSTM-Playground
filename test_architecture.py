import numpy as np
from src.artifacts import load_all_artifacts
from src.architecture_inspector import (
    inspect_all_models,
    calculate_tensor_flow,
    verify_runtime_tensor_shapes,
    get_component_inspection_details,
    extract_parameter_statistics,
    get_lstm_gate_weight_statistics,
    slice_parameter_matrix,
    get_city_30day_window,
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
from src.playground_canvas import (
    build_playground_payload,
    build_interactive_playground_html,
)
from src.architecture_explorer import render_architecture_explorer


def run_architecture_extraction_tests(artifacts=None):
    """
    Verifies extracted architecture attributes, layer dimensions,
    parameter counts, and weight/bias matrix shapes across all three
    trained models (Baseline V1, Model A, and Model B).
    """
    if artifacts is None:
        artifacts = load_all_artifacts()

    summaries = inspect_all_models(artifacts)

    expected_specs = {
        "baseline": {
            "model_name": "Baseline V1",
            "architecture_class": "WeatherLSTM",
            "lstm_input_size": 1,
            "lstm_hidden_size": 64,
            "lstm_num_layers": 1,
            "bidirectional": False,
            "sequence_length": 30,
            "city_vocab_size": 50,
            "city_embedding_dim": 8,
            "has_geography": False,
            "geo_dim": 0,
            "geo_features": [],
            "fc_input_size": 72,
            "fc_output_size": 1,
            "total_parameters": 17625,
            "component_parameters": {
                "city_embedding": 400,
                "lstm": 17152,
                "fc": 73,
            },
            "parameter_shapes": {
                "city_embedding.weight": (50, 8),
                "lstm.weight_ih_l0": (256, 1),
                "lstm.weight_hh_l0": (256, 64),
                "lstm.bias_ih_l0": (256,),
                "lstm.bias_hh_l0": (256,),
                "fc.weight": (1, 72),
                "fc.bias": (1,),
            },
        },
        "model_A": {
            "model_name": "Model A",
            "architecture_class": "MultivariateWeatherLSTM",
            "lstm_input_size": 8,
            "lstm_hidden_size": 64,
            "lstm_num_layers": 1,
            "bidirectional": False,
            "sequence_length": 30,
            "city_vocab_size": 50,
            "city_embedding_dim": 8,
            "has_geography": True,
            "geo_dim": 3,
            "geo_features": ["latitude", "longitude", "elevation_m"],
            "fc_input_size": 75,
            "fc_output_size": 1,
            "total_parameters": 19420,
            "component_parameters": {
                "city_embedding": 400,
                "lstm": 18944,
                "fc": 76,
            },
            "parameter_shapes": {
                "city_embedding.weight": (50, 8),
                "lstm.weight_ih_l0": (256, 8),
                "lstm.weight_hh_l0": (256, 64),
                "lstm.bias_ih_l0": (256,),
                "lstm.bias_hh_l0": (256,),
                "fc.weight": (1, 75),
                "fc.bias": (1,),
            },
        },
        "model_B": {
            "model_name": "Model B",
            "architecture_class": "MultivariateWeatherLSTM",
            "lstm_input_size": 8,
            "lstm_hidden_size": 64,
            "lstm_num_layers": 1,
            "bidirectional": False,
            "sequence_length": 30,
            "city_vocab_size": 50,
            "city_embedding_dim": 8,
            "has_geography": True,
            "geo_dim": 4,
            "geo_features": [
                "latitude",
                "longitude",
                "elevation_m",
                "coastal_distance_km",
            ],
            "fc_input_size": 76,
            "fc_output_size": 1,
            "total_parameters": 19421,
            "component_parameters": {
                "city_embedding": 400,
                "lstm": 18944,
                "fc": 77,
            },
            "parameter_shapes": {
                "city_embedding.weight": (50, 8),
                "lstm.weight_ih_l0": (256, 8),
                "lstm.weight_hh_l0": (256, 64),
                "lstm.bias_ih_l0": (256,),
                "lstm.bias_hh_l0": (256,),
                "fc.weight": (1, 76),
                "fc.bias": (1,),
            },
        },
    }

    print("\n" + "=" * 56)
    print("CHECKPOINT 3: ARCHITECTURE EXTRACTION VERIFICATION")
    print("=" * 56)

    for model_key, expected in expected_specs.items():
        actual = summaries[model_key]
        print(f"\nVerifying {actual['model_name']} ({model_key})...")

        assert actual["model_name"] == expected["model_name"]
        assert actual["architecture_class"] == expected["architecture_class"]
        assert actual["lstm_input_size"] == expected["lstm_input_size"]
        assert actual["lstm_hidden_size"] == expected["lstm_hidden_size"]
        assert actual["lstm_num_layers"] == expected["lstm_num_layers"]
        assert actual["bidirectional"] == expected["bidirectional"]
        assert actual["sequence_length"] == expected["sequence_length"]
        assert actual["city_vocab_size"] == expected["city_vocab_size"]
        assert actual["city_embedding_dim"] == expected["city_embedding_dim"]
        assert actual["has_geography"] == expected["has_geography"]
        assert actual["geo_dim"] == expected["geo_dim"]
        assert actual["geo_features"] == expected["geo_features"]
        assert actual["fc_input_size"] == expected["fc_input_size"]
        assert actual["fc_output_size"] == expected["fc_output_size"]
        assert actual["total_parameters"] == expected["total_parameters"]
        assert actual["trainable_parameters"] == expected["total_parameters"]
        assert actual["component_parameters"] == expected["component_parameters"]
        assert actual["parameter_shapes"] == expected["parameter_shapes"]
        assert all(actual["config_validation"].values())

        print(f"  [PASS] Input size          : {actual['lstm_input_size']}")
        print(f"  [PASS] Hidden size         : {actual['lstm_hidden_size']}")
        print(f"  [PASS] Sequence length     : {actual['sequence_length']}")
        print(
            f"  [PASS] City embedding      : "
            f"({actual['city_vocab_size']}, {actual['city_embedding_dim']})"
        )
        print(
            f"  [PASS] Geography dim       : {actual['geo_dim']} "
            f"{actual['geo_features']}"
        )
        print(
            f"  [PASS] Dense layer         : "
            f"{actual['fc_input_size']} -> {actual['fc_output_size']}"
        )
        print(
            f"  [PASS] Trainable params    : {actual['trainable_parameters']:,} "
            f"{actual['component_parameters']}"
        )
        print("  [PASS] Weight/bias shapes  : All 7 parameter tensors verified")

    return summaries


def run_tensor_dimension_tests(artifacts=None, summaries=None):
    """
    Verifies the Tensor Dimension Engine for Baseline V1, Model A, and Model B
    against expected specifications (for default batch_size=1 and custom
    batch_size=8) and against live PyTorch runtime execution.
    """
    if artifacts is None:
        artifacts = load_all_artifacts()
    if summaries is None:
        summaries = inspect_all_models(artifacts)

    expected_b1_shapes = {
        "baseline": {
            "weather_input": (1, 30, 1),
            "lstm_sequence_output": (1, 30, 64),
            "final_hidden_state": (1, 64),
            "final_cell_state": (1, 64),
            "geography": None,
            "city_embedding": (1, 8),
            "concatenated_vector": (1, 72),
            "dense_output": (1, 1),
        },
        "model_A": {
            "weather_input": (1, 30, 8),
            "lstm_sequence_output": (1, 30, 64),
            "final_hidden_state": (1, 64),
            "final_cell_state": (1, 64),
            "geography": (1, 3),
            "city_embedding": (1, 8),
            "concatenated_vector": (1, 75),
            "dense_output": (1, 1),
        },
        "model_B": {
            "weather_input": (1, 30, 8),
            "lstm_sequence_output": (1, 30, 64),
            "final_hidden_state": (1, 64),
            "final_cell_state": (1, 64),
            "geography": (1, 4),
            "city_embedding": (1, 8),
            "concatenated_vector": (1, 76),
            "dense_output": (1, 1),
        },
    }

    print("\n" + "=" * 56)
    print("CHECKPOINT 4: TENSOR DIMENSION ENGINE VERIFICATION")
    print("=" * 56)

    for model_key, expected_shapes in expected_b1_shapes.items():
        summary = summaries[model_key]
        flow_b1 = calculate_tensor_flow(summary, batch_size=1)
        runtime_b1 = verify_runtime_tensor_shapes(
            model_key, artifacts, batch_size=1
        )

        assert flow_b1["shapes"] == expected_shapes
        assert flow_b1["shapes"] == runtime_b1
        assert (
            flow_b1["shapes"]["lstm_sequence_output"]
            != flow_b1["shapes"]["final_hidden_state"]
        )

        print(f"\n--- {summary['model_name']} (batch_size = 1) ---")
        for stage in flow_b1["stages"]:
            print(
                f"  [PASS] {stage['label'] + ':':22s} {str(stage['shape']):12s} | "
                f"{stage['description']}"
            )

        flow_b8 = calculate_tensor_flow(summary, batch_size=8)
        runtime_b8 = verify_runtime_tensor_shapes(
            model_key, artifacts, batch_size=8
        )
        assert flow_b8["shapes"] == runtime_b8
        for stage_key, shape_b1 in expected_shapes.items():
            if shape_b1 is None:
                assert flow_b8["shapes"][stage_key] is None
            else:
                expected_b8 = (8,) + shape_b1[1:]
                assert flow_b8["shapes"][stage_key] == expected_b8

        print(
            f"  [PASS] Dynamic batch scaling verified for batch_size = 8 "
            f"(Concatenated: {flow_b8['shapes']['concatenated_vector']} -> "
            f"Output: {flow_b8['shapes']['dense_output']})"
        )


def run_architecture_diagram_tests(summaries):
    """
    Verifies the Interactive Architecture Diagram structure and Plotly
    figure generation for Baseline V1, Model A, and Model B.
    """
    print("\n" + "=" * 56)
    print("CHECKPOINT 5: ARCHITECTURE DIAGRAM VERIFICATION")
    print("=" * 56)

    for model_key, summary in summaries.items():
        data = get_architecture_diagram_data(
            summary=summary,
            batch_size=1,
            mode="Technical",
            selected_component="lstm",
        )
        fig = build_architecture_diagram(
            summary=summary,
            batch_size=1,
            mode="Technical",
            selected_component="lstm",
        )

        node_ids = [n["id"] for n in data["nodes"]]
        edge_pairs = [(e["source"], e["target"]) for e in data["edges"]]
        bypass_edges = [
            (e["source"], e["target"]) for e in data["edges"] if e["is_bypass"]
        ]

        if model_key == "baseline":
            assert node_ids == [
                "weather_input",
                "lstm",
                "city_embedding",
                "concatenation",
                "fc",
                "output",
            ]
            assert bypass_edges == [("city_embedding", "concatenation")]
        else:
            assert node_ids == [
                "weather_input",
                "lstm",
                "geography",
                "city_embedding",
                "concatenation",
                "fc",
                "output",
            ]
            assert bypass_edges == [
                ("geography", "concatenation"),
                ("city_embedding", "concatenation"),
            ]

        for src, dst in edge_pairs:
            if src in ("geography", "city_embedding"):
                assert dst == "concatenation"

        assert len(fig.data) == len(edge_pairs) + 1

        print(f"\n--- {summary['model_name']} ({model_key}) ---")
        print(f"  [PASS] Nodes ({len(node_ids)})      : {node_ids}")
        print(f"  [PASS] Bypass edges   : {bypass_edges}")
        print(
            f"  [PASS] Plotly figure  : {len(fig.data)} traces, "
            f"{len(fig.layout.shapes)} shapes, {len(fig.layout.annotations)} annotations"
        )


def run_layer_and_weight_inspector_tests(artifacts, summaries):
    """
    Verifies Checkpoint 6: Layer Inspector details (including packed 256-row
    LSTM matrices and all 6 gates/states) and Actual Trained Weights Inspector
    statistics, controlled matrix slicing, and heatmap generation.
    """
    print("\n" + "=" * 56)
    print("CHECKPOINT 6: LAYER & WEIGHT INSPECTOR VERIFICATION")
    print("=" * 56)

    city_to_id = artifacts["city_to_id"]

    for model_key, summary in summaries.items():
        model = artifacts["models"][model_key]
        in_size = summary["lstm_input_size"]
        fc_in = summary["fc_input_size"]

        lstm_details = get_component_inspection_details(summary, "lstm", batch_size=1)
        assert lstm_details["parameter_count"] == summary["component_parameters"]["lstm"]
        assert lstm_details["packed_matrix_info"]["W_ih_l0"] == (256, in_size)
        assert lstm_details["packed_matrix_info"]["W_hh_l0"] == (256, 64)
        assert lstm_details["packed_matrix_info"]["b_ih_l0"] == (256,)
        assert lstm_details["packed_matrix_info"]["b_hh_l0"] == (256,)
        assert len(lstm_details["gates_and_states"]) == 6

        emb_details = get_component_inspection_details(
            summary, "city_embedding", batch_size=1
        )
        assert emb_details["parameter_count"] == 400
        assert emb_details["weight_and_bias_shapes"]["city_embedding.weight"] == (50, 8)

        fc_details = get_component_inspection_details(summary, "fc", batch_size=1)
        assert fc_details["parameter_count"] == fc_in + 1
        assert fc_details["weight_and_bias_shapes"]["fc.weight"] == (1, fc_in)
        assert fc_details["weight_and_bias_shapes"]["fc.bias"] == (1,)

        param_stats = extract_parameter_statistics(model)
        assert len(param_stats) == 7
        total_vals = sum(item["num_values"] for item in param_stats)
        assert total_vals == summary["total_parameters"]

        for item in param_stats:
            assert np.isfinite(item["min"])
            assert np.isfinite(item["max"])
            assert np.isfinite(item["mean"])
            assert np.isfinite(item["std"])
            assert item["min"] <= item["mean"] <= item["max"]

        gate_stats = get_lstm_gate_weight_statistics(model)
        assert len(gate_stats) == 16

        slice_w_hh = slice_parameter_matrix(
            model=model,
            param_name="lstm.weight_hh_l0",
            row_start=0,
            row_end=16,
            col_start=0,
            col_end=16,
            summary=summary,
            city_to_id=city_to_id,
        )
        assert slice_w_hh["slice_shape"] == (16, 16)
        heatmap_fig = build_weight_heatmap(slice_w_hh, summary["model_name"])
        assert len(heatmap_fig.data) == 1

        print(f"\n--- {summary['model_name']} ({model_key}) ---")
        print(
            f"  [PASS] LSTM packed matrices : W_ih={lstm_details['packed_matrix_info']['W_ih_l0']}, "
            f"W_hh={lstm_details['packed_matrix_info']['W_hh_l0']} (256 = 4 × 64)"
        )
        print(
            f"  [PASS] Embedding & Dense    : Emb=(50, 8), FC=({fc_in}, 1) -> shapes & explanations verified"
        )
        print(
            f"  [PASS] Controlled slicing   : lstm.weight_hh_l0 sliced to {slice_w_hh['slice_shape']} & heatmap built"
        )


def run_neuron_inspector_tests(artifacts, summaries):
    """
    Verifies Checkpoint 7: LSTM Neuron Inspector integration using
    `LSTMGateEngine` (`src/lstm_engine.py`) and `prepare_gate_data`
    (`src/visualization.py`) on a real 30-day historical weather sequence.
    """
    print("\n" + "=" * 56)
    print("CHECKPOINT 7: LSTM NEURON INSPECTOR VERIFICATION")
    print("=" * 56)

    history_df = get_city_30day_window(city="Agartala", end_date="2025-10-19")
    assert len(history_df) == 30

    expected_gates = [
        "input_gate",
        "forget_gate",
        "candidate_gate",
        "output_gate",
        "cell_state",
        "hidden_state",
    ]

    for model_key, summary in summaries.items():
        gate_result = compute_real_gate_history(
            model_key=model_key,
            artifacts=artifacts,
            city="Agartala",
            history_df=history_df,
        )
        gate_data = gate_result["gate_data"]

        assert gate_result["sequence_shape"] == (30, summary["lstm_input_size"])
        assert gate_result["max_hidden_diff_vs_pytorch"] < 1e-5
        assert gate_result["max_cell_diff_vs_pytorch"] < 1e-5

        for gate_name in expected_gates:
            assert gate_name in gate_data
            assert gate_data[gate_name].shape == (30, 64)
            assert np.isfinite(gate_data[gate_name]).all()

        neuron_25 = extract_single_neuron_series(
            gate_data=gate_data,
            neuron_number=25,
            dates=gate_result["dates"],
        )
        assert neuron_25["neuron_number"] == 25
        assert neuron_25["neuron_index"] == 24
        assert len(neuron_25["statistics"]) == 6
        assert len(neuron_25["dataframe"]) == 30

        grid_fig = build_neuron_six_timeseries_grid(
            gate_data=gate_data,
            neuron_number=25,
            dates=gate_result["dates"],
            model_name=summary["model_name"],
            city="Agartala",
        )
        combined_fig = build_neuron_combined_gates_chart(
            gate_data=gate_data,
            neuron_number=25,
            dates=gate_result["dates"],
            model_name=summary["model_name"],
        )
        heatmap_fig = build_gate_activation_heatmap(
            gate_data=gate_data,
            gate_key="input_gate",
            dates=gate_result["dates"],
            model_name=summary["model_name"],
        )

        assert len(grid_fig.data) == 6
        assert len(combined_fig.data) == 4
        assert len(heatmap_fig.data) == 1

        print(f"\n--- {summary['model_name']} ({model_key}) ---")
        print(
            f"  [PASS] Real 30-day input    : Agartala ({gate_result['dates'][0]} to {gate_result['dates'][-1]}), "
            f"shape {gate_result['sequence_shape']}"
        )
        print(
            f"  [PASS] Gate matrices (6)    : All 6 arrays verified at shape (30, 64)"
        )
        print(
            f"  [PASS] PyTorch parity       : max |h_diff| = {gate_result['max_hidden_diff_vs_pytorch']:.2e}, "
            f"max |c_diff| = {gate_result['max_cell_diff_vs_pytorch']:.2e}"
        )
        print(
            f"  [PASS] Neuron 25 plots      : 6-subplot grid ({len(grid_fig.data)} traces), "
            f"combined gates ({len(combined_fig.data)} traces), 64-unit heatmap built"
        )


def run_prediction_walkthrough_tests(artifacts, summaries):
    """
    Verifies Checkpoint 8: 10-stage Prediction Walkthrough integration with
    `WeatherInference` across Baseline V1, Model A, and Model B.
    """
    print("\n" + "=" * 56)
    print("CHECKPOINT 8: PREDICTION WALKTHROUGH VERIFICATION")
    print("=" * 56)

    for model_key, summary in summaries.items():
        wt = run_prediction_walkthrough(
            model_key=model_key,
            city="Agartala",
            artifacts=artifacts,
            end_date="2025-10-19",
        )

        in_size = summary["lstm_input_size"]
        geo_dim = summary["geo_dim"]
        fc_in = summary["fc_input_size"]

        assert len(wt["stage1_raw_observations"]) == 30
        assert len(wt["stage2_preprocessed_features"]) == 30
        assert wt["stage3_scaled_input"]["shape"] == (1, 30, in_size)

        if summary["has_geography"]:
            assert wt["stage4_geography"]["shape"] == (1, geo_dim)
        else:
            assert wt["stage4_geography"]["shape"] is None

        assert wt["stage5_city_embedding"]["shape"] == (1, 8)
        assert wt["stage6_lstm_output"]["shape"] == (1, 30, 64)
        assert wt["stage7_final_states"]["hidden_shape"] == (1, 64)
        assert wt["stage7_final_states"]["cell_shape"] == (1, 64)
        assert wt["stage8_concatenated_vector"]["shape"] == (1, fc_in)
        assert wt["stage9_dense_output"]["shape"] == (1, 1)

        scaled_pred = wt["stage9_dense_output"]["scaled_prediction"]
        contrib_sum = wt["stage9_dense_output"]["contributions"]["total_scaled_sum"]
        assert abs(scaled_pred - contrib_sum) < 1e-5

        celsius_pred = wt["stage10_celsius_prediction"][
            "predicted_max_temperature_celsius"
        ]
        official_pred = wt["stage10_celsius_prediction"][
            "official_inference_result"
        ]["predicted_max_temperature"]
        assert abs(celsius_pred - official_pred) < 1e-5
        assert abs(celsius_pred - scaled_pred) > 1.0

        temp_fig = build_walkthrough_temperature_chart(wt)
        concat_fig = build_concatenated_vector_chart(wt)
        assert len(temp_fig.data) >= 3
        assert len(concat_fig.data) == (3 if summary["has_geography"] else 2)

        print(f"\n--- {summary['model_name']} ({model_key}) ---")
        print(
            f"  [PASS] Stages 1–3 (Inputs)  : 30 raw days -> preprocessed -> scaled tensor {wt['stage3_scaled_input']['shape']}"
        )
        print(
            f"  [PASS] Stages 4–5 (Bypass)  : Geography={wt['stage4_geography']['shape']}, "
            f"CityEmb={wt['stage5_city_embedding']['shape']} (Agartala ID={wt['city_id']})"
        )
        print(
            f"  [PASS] Stages 6–8 (States)  : LSTM Seq={wt['stage6_lstm_output']['shape']} -> "
            f"Final h_30={wt['stage7_final_states']['hidden_shape']} -> Concat={wt['stage8_concatenated_vector']['shape']}"
        )
        print(
            f"  [PASS] Stage 9 (Scaled Out) : {scaled_pred:.6f} "
            f"(LSTM={wt['stage9_dense_output']['contributions']['lstm_hidden_64_sum']:+.4f}, "
            f"Geo={wt['stage9_dense_output']['contributions']['geography_sum']:+.4f}, "
            f"City={wt['stage9_dense_output']['contributions']['city_embedding_8_sum']:+.4f}, "
            f"Bias={wt['stage9_dense_output']['contributions']['fc_bias']:+.4f})"
        )
        print(
            f"  [PASS] Stage 10 (Celsius)   : Predicted = {celsius_pred:.2f} °C "
            f"(Observed on {wt['prediction_date']} = {wt['stage10_celsius_prediction']['actual_max_temperature_celsius']:.2f} °C)"
        )


def run_interactive_playground_and_explorer_tests(artifacts, summaries):
    """
    Verifies Checkpoint 9 & 10: Interactive Neural Network Playground payload
    and HTML generation (`src/playground_canvas.py`) and Streamlit page
    entry point (`src/architecture_explorer.py`).
    """
    print("\n" + "=" * 56)
    print("CHECKPOINT 9 & 10: INTERACTIVE PLAYGROUND & EXPLORER VERIFICATION")
    print("=" * 56)

    assert callable(render_architecture_explorer)

    for model_key, summary in summaries.items():
        wt = run_prediction_walkthrough(
            model_key=model_key,
            city="Coimbatore",
            artifacts=artifacts,
            end_date="2025-10-19",
        )
        gate_res = compute_real_gate_history(
            model_key=model_key,
            artifacts=artifacts,
            city="Coimbatore",
            weather_tensor=wt["weather_tensor"],
        )
        payload = build_playground_payload(
            model_key=model_key,
            summary=summary,
            artifacts=artifacts,
            walkthrough=wt,
            gate_result=gate_res,
            batch_size=1,
            mode="Technical",
            selected_neuron=25,
        )
        html_str = build_interactive_playground_html(payload)

        assert len(payload["weights"]["W_ih"]) == 256
        assert len(payload["weights"]["W_ih"][0]) == summary["lstm_input_size"]
        assert len(payload["weights"]["W_hh"]) == 256
        assert len(payload["weights"]["W_hh"][0]) == 64
        assert len(payload["weights"]["fc_weight"]) == summary["fc_input_size"]
        assert len(payload["raw_features_30d"]) == 30
        assert len(payload["scaled_features_30d"]) == 30
        assert "runLiveForwardPass" in html_str
        assert "neuron-grid-64" in html_str
        assert "synapse-svg" in html_str

        print(
            f"  [PASS] {summary['model_name']:12s} ({model_key:8s}) : "
            f"Interactive Playground payload & HTML verified ({len(html_str):,} bytes)"
        )

    print("\n" + "=" * 56)
    print("ALL CHECKPOINT 3, 4, 5, 6, 7, 8, 9 & 10 TESTS PASSED SUCCESSFULLY!")
    print("=" * 56)


if __name__ == "__main__":
    loaded_artifacts = load_all_artifacts()
    extracted_summaries = run_architecture_extraction_tests(loaded_artifacts)
    run_tensor_dimension_tests(loaded_artifacts, extracted_summaries)
    run_architecture_diagram_tests(extracted_summaries)
    run_layer_and_weight_inspector_tests(loaded_artifacts, extracted_summaries)
    run_neuron_inspector_tests(loaded_artifacts, extracted_summaries)
    run_prediction_walkthrough_tests(loaded_artifacts, extracted_summaries)
    run_interactive_playground_and_explorer_tests(loaded_artifacts, extracted_summaries)
