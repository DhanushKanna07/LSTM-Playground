import numpy as np
from src.artifacts import load_all_artifacts
from src.gate_memory_inspector import (
    analyze_gate_and_memory_flow,
    extract_neuron_gate_story,
)
from src.gate_memory_explorer import (
    build_gate_memory_circuit_html,
    build_compact_memory_carryforward_chart,
    build_compact_weather_vs_gates_chart,
    build_compact_gate_drive_chart,
    build_compact_memory_horizon_or_heatmap_chart,
    render_gate_memory_explorer,
)


def run_gate_memory_tests():
    artifacts = load_all_artifacts()
    assert callable(render_gate_memory_explorer)

    print("\n" + "=" * 56)
    print("GATE & MEMORY EXPLORER VERIFICATION")
    print("=" * 56)

    for model_key in ["baseline", "model_A", "model_B"]:
        analysis = analyze_gate_and_memory_flow(
            model_key=model_key,
            city="Coimbatore",
            artifacts=artifacts,
            end_date="2025-10-19",
        )

        # Verify parity checks
        p = analysis["parity_checks"]
        assert p["max_hidden_diff_vs_pytorch"] < 1e-5
        assert p["max_cell_diff_vs_pytorch"] < 1e-5
        assert p["cell_decomposition_diff"] < 1e-5
        assert p["memory_horizon_sum_diff"] < 1e-5

        # Verify story extraction on Neuron 25, Day 30
        story = extract_neuron_gate_story(
            analysis=analysis,
            neuron_number=25,
            day_number=30,
        )
        assert len(story["plain_english_steps"]) == 5
        assert len(story["survival_from_past_days"]) == 30
        assert abs(sum(story["survival_from_past_days"]) - story["values"]["c_t"]) < 1e-5

        # Verify interactive cell circuit HTML & all 4 compact charts
        html_str = build_gate_memory_circuit_html(analysis, initial_neuron=25, initial_day=30)
        assert "cell-svg" in html_str
        assert "setSubStep" in html_str

        fig1 = build_compact_memory_carryforward_chart(analysis, 25, 30)
        fig2 = build_compact_weather_vs_gates_chart(analysis, 25, 30)
        fig3 = build_compact_gate_drive_chart(story)
        fig4 = build_compact_memory_horizon_or_heatmap_chart(
            analysis, 25, 30, "Memory Survival Horizon (Past Days Still in c_t)"
        )

        assert fig1.layout.height <= 350
        assert fig2.layout.height <= 350
        assert fig3.layout.height <= 350
        assert fig4.layout.height <= 350
        assert fig1.layout.legend.y < 0
        assert fig2.layout.legend.y < 0
        assert fig3.layout.legend.y < 0

        print(
            f"  [PASS] {analysis['model_name']:12s} ({model_key:8s}) : "
            f"Cell decomposition diff={p['cell_decomposition_diff']:.2e}, "
            f"Horizon sum diff={p['memory_horizon_sum_diff']:.2e}, "
            f"Circuit HTML ({len(html_str):,} bytes) & 4 compact charts verified"
        )

    print("\n" + "=" * 56)
    print("ALL GATE & MEMORY EXPLORER TESTS PASSED SUCCESSFULLY!")
    print("=" * 56)


if __name__ == "__main__":
    run_gate_memory_tests()

