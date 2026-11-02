
import torch
import torch.nn as nn


# ============================================================
# MODEL 1: BASELINE WEATHER LSTM (V1)
# ============================================================

class WeatherLSTM(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_cities=50,
        embedding_dim=8
    ):
        super(WeatherLSTM, self).__init__()

        # City embedding
        self.city_embedding = nn.Embedding(
            num_cities,
            embedding_dim
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Fully connected output layer
        self.fc = nn.Linear(
            hidden_size + embedding_dim,
            1
        )

    def forward(self, x, city_id):

        lstm_output, (hidden_state, cell_state) = self.lstm(x)

        final_hidden = hidden_state[-1]

        city_vector = self.city_embedding(city_id)

        combined = torch.cat(
            (final_hidden, city_vector),
            dim=1
        )

        prediction = self.fc(combined)

        return prediction


# ============================================================
# MODEL 2 & 3: MULTIVARIATE WEATHER LSTM (V2)
# ============================================================

class MultivariateWeatherLSTM(nn.Module):

    def __init__(
        self,
        input_size=8,
        hidden_size=64,
        num_cities=50,
        embedding_dim=8,
        geo_dim=3
    ):
        super(MultivariateWeatherLSTM, self).__init__()

        # City embedding
        self.city_embedding = nn.Embedding(
            num_cities,
            embedding_dim
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Fully connected output layer
        combined_size = (
            hidden_size +
            embedding_dim +
            geo_dim
        )

        self.fc = nn.Linear(
            combined_size,
            1
        )

    def forward(self, weather, geography, city_id):

        lstm_output, (hidden_state, cell_state) = self.lstm(weather)

        final_hidden = hidden_state[-1]

        city_vector = self.city_embedding(city_id)

        combined = torch.cat(
            [final_hidden, geography, city_vector],
            dim=1
        )

        prediction = self.fc(combined)

        return prediction


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(model_name, num_cities=50):

    if model_name == "baseline":

        return WeatherLSTM(
            input_size=1,
            hidden_size=64,
            num_cities=num_cities,
            embedding_dim=8
        )

    elif model_name == "model_A":

        return MultivariateWeatherLSTM(
            input_size=8,
            hidden_size=64,
            num_cities=num_cities,
            embedding_dim=8,
            geo_dim=3
        )

    elif model_name == "model_B":

        return MultivariateWeatherLSTM(
            input_size=8,
            hidden_size=64,
            num_cities=num_cities,
            embedding_dim=8,
            geo_dim=4
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )
