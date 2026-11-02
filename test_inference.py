
import numpy as np
import pandas as pd
import torch

from src.artifacts import load_all_artifacts
from src.inference import WeatherInference


# -----------------------------------------
# 1. Load models and inference engine
# -----------------------------------------

artifacts = load_all_artifacts()
inference = WeatherInference(artifacts)

device = artifacts["device"]


# -----------------------------------------
# 2. Load saved test arrays
# -----------------------------------------

TEST_DIR = "data/processed/artifacts/v2/test"

X_test = np.load(f"{TEST_DIR}/X_test.npy")
geo_test_A = np.load(f"{TEST_DIR}/geo_test_A.npy")
geo_test_B = np.load(f"{TEST_DIR}/geo_test_B.npy")
city_test = np.load(f"{TEST_DIR}/city_test.npy")
y_test = np.load(f"{TEST_DIR}/y_test.npy")
target_dates = np.load(
    f"{TEST_DIR}/target_dates.npy",
    allow_pickle=True
)

print("\nTEST ARRAY SHAPES")
print("-----------------")
print("X_test:", X_test.shape)
print("geo_test_A:", geo_test_A.shape)
print("geo_test_B:", geo_test_B.shape)
print("city_test:", city_test.shape)
print("y_test:", y_test.shape)
print("target_dates:", target_dates.shape)


# -----------------------------------------
# DIAGNOSE TEST ARRAY ALIGNMENT
# -----------------------------------------

print("\nDATE AND CITY DIAGNOSTICS")
print("-------------------------")

print("Agartala city ID:", artifacts["city_to_id"]["Agartala"])

print("Unique city IDs in test:", np.unique(city_test))

print("First 10 test dates:")
print(target_dates[:10])

print("\nLast 10 test dates:")
print(target_dates[-10:])

print("\nDate range:")
print("Start:", pd.to_datetime(target_dates).min())
print("End:", pd.to_datetime(target_dates).max())

print("\nUnique date count:", len(np.unique(target_dates)))

print("\nSamples belonging to Agartala:")

agartala_indices = np.where(
    city_test == artifacts["city_to_id"]["Agartala"]
)[0]

print("Count:", len(agartala_indices))

if len(agartala_indices) > 0:
    print("First 10 Agartala dates:")
    print(target_dates[agartala_indices[:10]])

    print("Last 10 Agartala dates:")
    print(target_dates[agartala_indices[-10:]])


# -----------------------------------------
# 3. Find Agartala sample on 20 Oct 2025
# -----------------------------------------

weather_df = pd.read_csv(
    "data/processed/india_weather_merged.csv"
)

weather_df["date"] = pd.to_datetime(weather_df["date"])

city = "Agartala"
target_date = pd.Timestamp("2025-10-20")

city_id = artifacts["city_to_id"][city]

# Convert dates into a comparable format.
dates = pd.to_datetime(target_dates)

# Find matching test sample.
indices = np.where(
    (city_test == city_id) &
    (dates == target_date)
)[0]

if len(indices) != 1:
    raise ValueError(
        f"Expected one matching test sample, found {len(indices)}."
    )

idx = int(indices[0])

print("\nMATCHED TEST SAMPLE")
print("-------------------")
print("Index:", idx)
print("City:", city)
print("Target date:", dates[idx])


# -----------------------------------------
# 4. Predict directly from saved tensors
# -----------------------------------------

city_tensor = torch.tensor(
    [city_id],
    dtype=torch.long,
    device=device
)

weather_tensor = torch.tensor(
    X_test[idx:idx+1],
    dtype=torch.float32,
    device=device
)

geo_A_tensor = torch.tensor(
    geo_test_A[idx:idx+1],
    dtype=torch.float32,
    device=device
)

geo_B_tensor = torch.tensor(
    geo_test_B[idx:idx+1],
    dtype=torch.float32,
    device=device
)

with torch.no_grad():

    pred_A_saved = artifacts["models"]["model_A"](
        weather_tensor,
        geo_A_tensor,
        city_tensor
    ).item()

    pred_B_saved = artifacts["models"]["model_B"](
        weather_tensor,
        geo_B_tensor,
        city_tensor
    ).item()


# Convert saved-tensor predictions to Celsius.
saved_A_celsius = inference.inverse_transform_prediction(
    pred_A_saved, "model_A"
)

saved_B_celsius = inference.inverse_transform_prediction(
    pred_B_saved, "model_B"
)


# -----------------------------------------
# 5. Run raw-data inference
# -----------------------------------------

city_df = weather_df[
    weather_df["city"] == city
].copy()

city_df = city_df.sort_values("date")

history = city_df[
    city_df["date"] < target_date
].tail(30)

if len(history) != 30:
    raise ValueError("Expected 30 historical observations.")

raw_A = inference.predict_next_day(
    city, history, "model_A"
)

raw_B = inference.predict_next_day(
    city, history, "model_B"
)


# -----------------------------------------
# 6. Compare predictions
# -----------------------------------------

comparison = pd.DataFrame([
    {
        "Model": "Model A",
        "Saved Tensor Prediction": saved_A_celsius,
        "Raw Data Prediction": raw_A["predicted_max_temperature"],
        "Difference": abs(
            saved_A_celsius -
            raw_A["predicted_max_temperature"]
        )
    },
    {
        "Model": "Model B",
        "Saved Tensor Prediction": saved_B_celsius,
        "Raw Data Prediction": raw_B["predicted_max_temperature"],
        "Difference": abs(
            saved_B_celsius -
            raw_B["predicted_max_temperature"]
        )
    }
])

print("\nINFERENCE VERIFICATION")
print("----------------------")
print(comparison.round(6).to_string(index=False))


# -----------------------------------------
# 7. Check numerical agreement
# -----------------------------------------

for _, row in comparison.iterrows():

    if row["Difference"] < 1e-4:
        print(f'{row["Model"]}: PASSED')
    else:
        print(f'{row["Model"]}: MISMATCH')
