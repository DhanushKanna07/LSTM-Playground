
import torch
import torch.nn as nn


class LSTMGateEngine:

    def __init__(self, model):

        self.model = model

        # Access the trained PyTorch LSTM layer.
        self.lstm = model.lstm

        # Extract architecture information.
        self.input_size = self.lstm.input_size
        self.hidden_size = self.lstm.hidden_size
        self.num_layers = self.lstm.num_layers

        # Check supported architecture.
        if self.num_layers != 1:
            raise ValueError(
                "Only single-layer LSTMs are supported."
            )

        if self.lstm.bidirectional:
            raise ValueError(
                "Bidirectional LSTMs are not supported."
            )

        # Extract actual trained parameters.
        self.W_ih = self.lstm.weight_ih_l0.detach()
        self.W_hh = self.lstm.weight_hh_l0.detach()

        self.b_ih = self.lstm.bias_ih_l0.detach()
        self.b_hh = self.lstm.bias_hh_l0.detach()

        print("LSTM Gate Engine Initialized")
        print("----------------------------")
        print("Input size:", self.input_size)
        print("Hidden size:", self.hidden_size)
        print("Number of layers:", self.num_layers)
        print("W_ih shape:", self.W_ih.shape)
        print("W_hh shape:", self.W_hh.shape)
        print("b_ih shape:", self.b_ih.shape)
        print("b_hh shape:", self.b_hh.shape)

    # -------------------------------------
    # Split packed weights into four gates
    # -------------------------------------

    def split_gate_weights(self, print_shapes=True):

        gate_names = ["i", "f", "g", "o"]

        W_ih_parts = self.W_ih.chunk(4, dim=0)
        W_hh_parts = self.W_hh.chunk(4, dim=0)

        b_ih_parts = self.b_ih.chunk(4, dim=0)
        b_hh_parts = self.b_hh.chunk(4, dim=0)

        self.gates = {}

        for index, name in enumerate(gate_names):

            self.gates[name] = {
                "W_ih": W_ih_parts[index],
                "W_hh": W_hh_parts[index],
                "b_ih": b_ih_parts[index],
                "b_hh": b_hh_parts[index]
            }

        if print_shapes:

            print("\nGATE WEIGHT EXTRACTION")
            print("----------------------")

            for name, params in self.gates.items():

                print(f"\nGate: {name}")

                for parameter_name, tensor in params.items():
                    print(parameter_name, "shape:", tensor.shape)

        return self.gates

    def calculate_gates(self, x_t, h_prev):

        if not hasattr(self, "gates"):
            raise RuntimeError(
                "Call split_gate_weights() first."
            )

        # Calculate the input-to-hidden contribution.
        input_contribution = torch.matmul(
            self.W_ih, x_t
        )

        # Calculate the hidden-to-hidden contribution.
        hidden_contribution = torch.matmul(
            self.W_hh, h_prev
        )

        # Add both biases.
        z = (
            input_contribution
            + hidden_contribution
            + self.b_ih
            + self.b_hh
        )

        # Split into four gate pre-activations.
        z_i, z_f, z_g, z_o = z.chunk(4, dim=0)

        # Apply activation functions.
        input_gate = torch.sigmoid(z_i)
        forget_gate = torch.sigmoid(z_f)
        candidate_gate = torch.tanh(z_g)
        output_gate = torch.sigmoid(z_o)

        return {
            "input_gate": input_gate,
            "forget_gate": forget_gate,
            "candidate_gate": candidate_gate,
            "output_gate": output_gate
        }

    def update_memory(self, gate_values, c_prev):

        # Retrieve gate activations.
        input_gate = gate_values["input_gate"]
        forget_gate = gate_values["forget_gate"]
        candidate_gate = gate_values["candidate_gate"]
        output_gate = gate_values["output_gate"]

        # Update cell state.
        c_current = (
            forget_gate * c_prev
            + input_gate * candidate_gate
        )

        # Update hidden state.
        h_current = (
            output_gate * torch.tanh(c_current)
        )

        return {
            "cell_state": c_current,
            "hidden_state": h_current
        }
    def process_sequence(self, sequence):
        h_prev = torch.zeros(
            self.hidden_size,
            device=sequence.device
        )

        c_prev = torch.zeros(
            self.hidden_size,
            device=sequence.device
        )

        history = []

        for t in range(sequence.shape[0]):

            x_t = sequence[t]

            gate_values = self.calculate_gates(x_t, h_prev)

            memory = self.update_memory(gate_values, c_prev)

            c_prev = memory["cell_state"]
            h_prev = memory["hidden_state"]

            history.append({
                **gate_values,
                "cell_state": c_prev,
                "hidden_state": h_prev
            })

        return history, h_prev, c_prev