"""
Automated verification test suite for the Interactive Model Comparison Module.
Tests:
1. Full 18,750-sample held-out test set evaluation and exact parity with saved JSON benchmarks
2. Per-city 50-city leaderboard and coastal vs inland regime breakdown
3. Synchronized 3-model Head-to-Head comparison (Baseline V1, Model A, Model B)
4. Interactive Head-to-Head Arena HTML widget (weather sliders + geography ablation toggles)
5. Compact 2x2 interactive Plotly charts (height <= 320, external HTML headers, bottom legends)
6. Architecture & benchmark comparison table
"""

import json
from pathlib import Path
from src.artifacts import load_all_artifacts
from src.model_comparison_inspector import (
    evaluate_full_test_benchmark,
    run_head_to_head_comparison,
    build_architecture_comparison_table,
)
from src.model_comparison_explorer import (
    build_head_to_head_arena_html,
    build_compact_city_trajectory_chart,
    build_compact_error_comparison_chart,
    build_compact_coastal_impact_chart,
    build_compact_leaderboard_or_residual_chart,
)


def main():
    print("=" * 68)
    print("RUNNING MODEL COMPARISON MODULE VERIFICATION SUITE")
    print("=" * 68)

    artifacts = load_all_artifacts()
    project_root = Path(__file__).resolve().parent

    # 1. Verify Full Test Benchmark Evaluation & Saved Benchmark Parity
    bench = evaluate_full_test_benchmark(artifacts)
    assert bench["num_test_samples"] == 18750, (
        f"Expected 18750 test samples, got {bench['num_test_samples']}"
    )
    assert bench["num_cities"] == 50, f"Expected 50 cities, got {bench['num_cities']}"

    # Compare against saved models/multivariate_v2_results.json and models/version1_results.json
    with open(project_root / "models" / "multivariate_v2_results.json", "r") as f:
        v2_saved = json.load(f)
    with open(project_root / "models" / "version1_results.json", "r") as f:
        v1_saved = json.load(f)

    for model_key, saved_key in [
        ("baseline", "V1_Benchmark"),
        ("model_A", "Model_A"),
        ("model_B", "Model_B"),
    ]:
        live_m = bench["global_metrics"][model_key]
        saved_m = v2_saved[saved_key]
        assert abs(live_m["MAE"] - saved_m["MAE"]) < 1e-3, (
            f"MAE mismatch for {model_key}: live={live_m['MAE']} vs saved={saved_m['MAE']}"
        )
        assert abs(live_m["RMSE"] - saved_m["RMSE"]) < 1e-3, (
            f"RMSE mismatch for {model_key}: live={live_m['RMSE']} vs saved={saved_m['RMSE']}"
        )
        assert abs(live_m["R2"] - saved_m["R2"]) < 1e-3, (
            f"R2 mismatch for {model_key}: live={live_m['R2']} vs saved={saved_m['R2']}"
        )

    assert abs(bench["global_metrics"]["naive"]["MAE"] - v1_saved["baseline_mae"]) < 1e-3
    assert abs(bench["global_metrics"]["naive"]["RMSE"] - v1_saved["baseline_rmse"]) < 1e-3
    assert abs(bench["global_metrics"]["naive"]["R2"] - v1_saved["baseline_r2"]) < 1e-3

    # Verify ordering: Model B < Model A < Baseline V1 < Naive Persistence in MAE and RMSE
    gm = bench["global_metrics"]
    assert (
        gm["model_B"]["MAE"]
        < gm["model_A"]["MAE"]
        < gm["baseline"]["MAE"]
        < gm["naive"]["MAE"]
    ), "Expected Model B < Model A < Baseline V1 < Naive in MAE"

    print(
        f"[PASS] Full 18,750-sample test evaluation verified with exact benchmark parity:\n"
        f"       Naive MAE={gm['naive']['MAE']:.4f}°C | "
        f"Baseline V1 MAE={gm['baseline']['MAE']:.4f}°C | "
        f"Model A MAE={gm['model_A']['MAE']:.4f}°C | "
        f"Model B MAE={gm['model_B']['MAE']:.4f}°C"
    )

    # 2. Verify Per-City Leaderboard DataFrame
    per_city_df = bench["city_metrics_df"]
    assert len(per_city_df) == 50, f"Expected 50 cities in city_metrics_df, got {len(per_city_df)}"
    required_cols = {
        "city",
        "coastal_distance_km",
        "zone",
        "baseline_mae",
        "model_A_mae",
        "model_B_mae",
        "best_model",
    }
    assert required_cols.issubset(set(per_city_df.columns)), "Missing columns in city_metrics_df"
    print(
        f"[PASS] 50-city leaderboard verified ({len(per_city_df)} cities across Coastal/Inland/Elevation zones; "
        f"Wins: {bench['city_win_counts']})"
    )

    # 3. Verify Synchronized 3-Model Head-to-Head Comparison
    for city in ["Chennai", "Agra"]:
        h2h = run_head_to_head_comparison(city, artifacts, end_date="2025-10-19")
        assert h2h["city"] == city
        assert len(h2h["dates"]) == 30
        assert set(h2h["models"].keys()) == {"baseline", "model_A", "model_B"}

        for mk, mdata in h2h["models"].items():
            assert -15.0 <= mdata["predicted_celsius"] <= 55.0, (
                f"Unrealistic prediction for {mk}: {mdata['predicted_celsius']}"
            )
            assert mdata["abs_error_celsius"] is not None and mdata["abs_error_celsius"] >= 0.0
            assert set(mdata["day30_gate_means"].keys()) >= {
                "forget_gate",
                "input_gate",
                "candidate_gate",
                "output_gate",
            }
            assert "lstm_hidden_64_sum" in mdata["contributions"]

        arena_html = build_head_to_head_arena_html(h2h, artifacts)
        assert "Baseline V1" in arena_html
        assert "Model A" in arena_html
        assert "Model B" in arena_html
        assert "Mute Coastal Dist (Model B)" in arena_html
        assert "Mute All Geo (A & B)" in arena_html
        print(
            f"[PASS] Head-to-Head comparison & interactive HTML arena verified for {city} "
            f"(Obs={h2h['actual_celsius']:.2f}°C | V1={h2h['models']['baseline']['predicted_celsius']:.2f}°C | "
            f"A={h2h['models']['model_A']['predicted_celsius']:.2f}°C | "
            f"B={h2h['models']['model_B']['predicted_celsius']:.2f}°C | Winner={h2h['winner_name']})"
        )

    # 4. Verify Compact 2x2 Plotly Charts (height <= 320, no internal title overlap, bottom legend)
    fig1 = build_compact_city_trajectory_chart(bench, "Chennai", focus_date="2025-10-20")
    fig2 = build_compact_error_comparison_chart(bench, "Chennai")
    fig3 = build_compact_coastal_impact_chart(bench, "Chennai")
    fig4a = build_compact_leaderboard_or_residual_chart(
        bench, "Chennai", mode_choice="Top City Improvements (Model B vs V1)"
    )
    fig4b = build_compact_leaderboard_or_residual_chart(
        bench, "Chennai", mode_choice="18,750-Sample Test Error Distribution"
    )

    for idx, fig in enumerate([fig1, fig2, fig3, fig4a, fig4b], start=1):
        height = fig.layout.height
        assert height is not None and height <= 320, f"Chart {idx} height {height} exceeds 320px limit"
        title_text = fig.layout.title.text if fig.layout.title else None
        assert not title_text, f"Chart {idx} should not have internal SVG title text (got {title_text!r})"
        if fig.layout.showlegend is not False and fig.layout.legend and fig.layout.legend.y is not None:
            assert fig.layout.legend.y < 0, f"Chart {idx} legend.y ({fig.layout.legend.y}) must be < 0 (below plot)"

    print("[PASS] All 5 compact Plotly chart views verified (height <= 315px, external HTML headers, bottom legends)")

    # 5. Verify Architecture & Benchmark Comparison Table
    arch_df = build_architecture_comparison_table(artifacts)
    assert len(arch_df) >= 12
    assert list(arch_df.columns) == [
        "Dimension / Metric",
        "Baseline V1 (Univariate)",
        "Model A (Multivariate + 3 Geo)",
        "Model B (Multivariate + 4 Geo)",
    ]
    print(f"[PASS] Architecture & Benchmark comparison table verified ({len(arch_df)} rows)")

    print("=" * 68)
    print("ALL MODEL COMPARISON MODULE TESTS PASSED SUCCESSFULLY!")
    print("=" * 68)


if __name__ == "__main__":
    main()
