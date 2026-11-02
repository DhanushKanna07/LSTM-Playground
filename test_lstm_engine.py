
import numpy as np
import torch

from src.artifacts import load_all_artifacts
from src.lstm_engine import LSTMGateEngine
from src.visualization import prepare_gate_data


# 1. Load trained artifacts
artifacts = load_all_artifacts()

# Use Model B on CPU for strict numerical verification
model = artifacts["models"]["model_B"].cpu()

print("Model B loaded on:", next(model.parameters()).device)


# 2. Initialize the gate engine
engine = LSTMGateEngine(model)
engine.split_gate_weights(print_shapes=False)


# 3. Load an actual test sequence
X_test = np.load(
    "data/processed/artifacts/v2/test/X_test.npy"
)

sequence = torch.tensor(
    X_test[0],
    dtype=torch.float32
)

print("\nInput sequence shape:", sequence.shape)


# 4. Run our manual LSTM implementation
history, manual_h, manual_c = engine.process_sequence(sequence)


# 5. Run PyTorch's original LSTM
with torch.no_grad():
    pytorch_output, (pytorch_h, pytorch_c) = model.lstm(
        sequence.unsqueeze(0)
    )


# 6. Compare all 30 hidden states
manual_outputs = torch.stack(
    [step["hidden_state"] for step in history]
)

pytorch_outputs = pytorch_output.squeeze(0)

torch.testing.assert_close(
    manual_outputs,
    pytorch_outputs,
    rtol=1e-5,
    atol=1e-6
)

# 7. Compare final cell and hidden states
torch.testing.assert_close(
    manual_h,
    pytorch_h[-1, 0, :],
    rtol=1e-5,
    atol=1e-6
)

torch.testing.assert_close(
    manual_c,
    pytorch_c[-1, 0, :],
    rtol=1e-5,
    atol=1e-6
)


# 8. Validate the recorded gate history
expected_gates = [
    "input_gate",
    "forget_gate",
    "candidate_gate",
    "output_gate",
    "cell_state",
    "hidden_state"
]

assert len(history) == 30

for day in history:
    for gate in expected_gates:
        assert day[gate].shape == (64,)
        assert torch.isfinite(day[gate]).all()


print("\n" + "=" * 40)
print("LSTM GATE ENGINE VERIFICATION PASSED")
print("=" * 40)

print("Sequence length:", len(history))
print("Hidden units:", engine.hidden_size)
print("Gate history:", len(expected_gates), "values per day")
print("All hidden states match PyTorch.")
print("Final cell state matches PyTorch.")
print("All gate values are finite.")

print("\nGATE HISTORY INSPECTION")
print("-----------------------")

print("Number of days:", len(history))

for key, value in history[0].items():
    print(key, ":", value.shape)

print("\nFirst day, first 5 input gate values:")
print(history[0]["input_gate"][:5])

print("\nLast day, first 5 cell state values:")
print(history[-1]["cell_state"][:5])

gate_data = prepare_gate_data(history)

print("\nVISUALIZATION DATA VERIFICATION")
print("--------------------------------")

for name, matrix in gate_data.items():
    print(name, ":", matrix.shape)

    assert matrix.shape == (30, 64)
    assert np.isfinite(matrix).all()

print("\nVisualization data prepared successfully!")