
import numpy as np
import pandas as pd
import torch


class WeatherInference:
    """
    Runs next-day maximum temperature predictions using
    the trained baseline, Model A, or Model B.

    Input:
        Raw weather observations for a city.

    Output:
        Predicted next-day maximum temperature in Celsius.
    """

    def __init__(self, artifacts):
        self.artifacts = artifacts

        self.device = artifacts["device"]
        self.models = artifacts["models"]

        self.v1 = artifacts["v1"]
        self.v2 = artifacts["v2"]

        self.city_to_id = artifacts["city_to_id"]

        self.sequence_length = self.v2["config"]["sequence_length"]

        self.temporal_features = self.v2["config"]["temporal_features"]

        self.geo_features_A = self.v2["config"]["model_A"]["geo_features"]
        self.geo_features_B = self.v2["config"]["model_B"]["geo_features"]

        self.geography_df = self.v2["geographical_features"]

    # --------------------------------------------------
    # 1. Validate and prepare the weather history
    # --------------------------------------------------

    def prepare_history(self, city, history_df, model_name):
        """
        Validates the input and selects the latest 30 observations.
        """

        if model_name not in self.models:
            raise ValueError(
                "Invalid model. Choose baseline, model_A, or model_B."
            )

        if city not in self.city_to_id:
            raise ValueError(f"City '{city}' is not available.")

        if not isinstance(history_df, pd.DataFrame):
            raise TypeError("history_df must be a pandas DataFrame.")

        df = history_df.copy()

        # If the dataframe contains multiple cities,
        # retain only the requested city.
        if "city" in df.columns:
            df = df[df["city"] == city].copy()

        # Identify the date column.
        date_column = None

        for column in ["date", "time", "datetime"]:
            if column in df.columns:
                date_column = column
                break

        if date_column is None:
            raise ValueError(
                "The input dataframe must contain a date, time, or datetime column."
            )

        df[date_column] = pd.to_datetime(
            df[date_column],
            errors="coerce"
        )

        if df[date_column].isna().any():
            raise ValueError("Some dates could not be converted.")

        df = df.sort_values(date_column)

        if len(df) < self.sequence_length:
            raise ValueError(
                f"At least {self.sequence_length} observations are required. "
                f"Received {len(df)}."
            )

        # Keep only the latest 30 observations.
        df = df.tail(self.sequence_length).copy()

        if df[date_column].duplicated().any():
            raise ValueError("Duplicate dates found in the input history.")

        return df, date_column

    # --------------------------------------------------
    # 2. Prepare baseline input
    # --------------------------------------------------

    def prepare_baseline_input(self, history_df):
        """
        Baseline uses only maximum temperature.

        Shape before LSTM:
            (1, 30, 1)
        """

        column = "temperature_2m_max"

        if column not in history_df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

        values = history_df[[column]].astype(np.float32)

        scaler = self.v1["target_scaler"]

        scaled_values = scaler.transform(values)

        tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        return tensor

    # --------------------------------------------------
    # 3. Prepare V2 temporal input
    # --------------------------------------------------

    def prepare_temporal_input(self, history_df, date_column):
        """
        Builds the eight temporal features in the exact
        order used during V2 training.
        """

        df = history_df.copy()

        required_columns = [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max"
        ]

        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required weather columns: {missing}"
            )

        # Rainfall transformation used during training.
        df["precipitation_log"] = np.log1p(
            df["precipitation_sum"].clip(lower=0)
        )

        # Seasonal features.
        day_of_year = df[date_column].dt.dayofyear

        days_in_year = np.where(
            df[date_column].dt.is_leap_year,
            366,
            365
        )

        angle = 2 * np.pi * (day_of_year - 1) / days_in_year

        df["day_of_year_sin"] = np.sin(angle)
        df["day_of_year_cos"] = np.cos(angle)

        # Preserve the exact feature order from config.json.
        values = df[self.temporal_features].astype(np.float32)
        

        if not np.isfinite(values.to_numpy(dtype=np.float32)).all():
            raise ValueError(
                "The temporal input contains missing or infinite values."
            )

        scaler = self.v2["temporal_scaler"]

        scaled_values = scaler.transform(values)

        tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        return tensor

    # --------------------------------------------------
    # 4. Prepare geographical input
    # --------------------------------------------------


    def prepare_geography_input(self, city, model_name):

        if model_name == "model_A":
            geo_features = self.geo_features_A
            scaler = self.v2["geo_scaler_A"]

        elif model_name == "model_B":
            geo_features = self.geo_features_B
            scaler = self.v2["geo_scaler_B"]

        else:
            raise ValueError(
                "Geographical input is only used by Model A and Model B."
            )

        city_row = self.geography_df[
            self.geography_df["city"] == city
        ]

        if len(city_row) != 1:
            raise ValueError(
                f"Could not find exactly one geographical record for {city}."
            )

        # Preserve feature names expected by the scaler.
        values = city_row[geo_features].astype(np.float32)

        # Scale while values is still a DataFrame.
        scaled_values = scaler.transform(values)

        tensor = torch.tensor(
            scaled_values,
            dtype=torch.float32,
            device=self.device  
        )

        return tensor

    # --------------------------------------------------
    # 5. Convert scaled prediction back to Celsius
    # --------------------------------------------------

    def inverse_transform_prediction(self, prediction, model_name):

        if model_name == "baseline":
            scaler = self.v1["target_scaler"]
        else:
            scaler = self.v2["target_scaler"]

        prediction_array = np.array(
            [[prediction]],
            dtype=np.float32
        )

        feature_names = getattr(scaler, "feature_names_in_", None)

        if feature_names is not None:
            prediction_input = pd.DataFrame(
                prediction_array,
                columns=feature_names
            )
        else:
            prediction_input = prediction_array

        actual_temperature = scaler.inverse_transform(
            prediction_input
        )

        return float(actual_temperature[0, 0])

    # --------------------------------------------------
    # 6. Predict next-day maximum temperature
    # --------------------------------------------------

    def predict_next_day(self, city, history_df, model_name="model_B"):
        """
        Predicts the maximum temperature for the day
        immediately following the final observation.

        history_df must contain raw weather observations.
        """

        df, date_column = self.prepare_history(
            city,
            history_df,
            model_name
        )

        city_id = self.city_to_id[city]

        city_tensor = torch.tensor(
            [city_id],
            dtype=torch.long,
            device=self.device
        )

        # Prepare inputs according to the selected model.
        if model_name == "baseline":

            weather_tensor = self.prepare_baseline_input(df)

            with torch.no_grad():
                prediction = self.models["baseline"](
                    weather_tensor,
                    city_tensor
                )

        else:

            weather_tensor = self.prepare_temporal_input(
                df,
                date_column
            )

            geography_tensor = self.prepare_geography_input(
                city,
                model_name
            )

            with torch.no_grad():
                prediction = self.models[model_name](
                    weather_tensor,
                    geography_tensor,
                    city_tensor
                )

        # Convert the model's scaled output to Celsius.
        scaled_prediction = prediction.item()

        temperature_celsius = self.inverse_transform_prediction(
            scaled_prediction,
            model_name
        )

        last_date = df[date_column].iloc[-1]
        prediction_date = last_date + pd.Timedelta(days=1)

        return {
            "city": city,
            "model": model_name,
            "last_observation_date": last_date.strftime("%Y-%m-%d"),
            "prediction_date": prediction_date.strftime("%Y-%m-%d"),
            "predicted_max_temperature": temperature_celsius,
            "unit": "Celsius"
        }
