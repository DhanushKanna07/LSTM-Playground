
import numpy as np


def prepare_gate_data(history):
    """
    Convert LSTM gate history into NumPy matrices.

    Expected input:
        history: List of 30 dictionaries.
                 Each dictionary contains six tensors of shape (64,).

    Returns:
        Dictionary containing six arrays of shape (30, 64).
    """

    gate_names = [
        "input_gate",
        "forget_gate",
        "candidate_gate",
        "output_gate",
        "cell_state",
        "hidden_state"
    ]

    gate_data = {}

    for gate in gate_names:

        values = [
            step[gate].detach().cpu().numpy()
            for step in history
        ]

        gate_data[gate] = np.stack(values, axis=0)

    return gate_data
