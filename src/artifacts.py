
import json
import joblib
import torch

from pathlib import Path

from src.models import create_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

ARTIFACT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "artifacts"
)

V2_ARTIFACT_DIR = ARTIFACT_DIR / "v2"


# ============================================================
# DEVICE CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FILE VALIDATION
# ============================================================

def check_file(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


# ============================================================
# LOAD JSON FILE
# ============================================================

def load_json(path):

    check_file(path)

    with open(path, "r") as file:
        return json.load(file)


# ============================================================
# LOAD V1 ARTIFACTS
# ============================================================

def load_v1_artifacts():

    print("Loading V1 artifacts...")

    config = load_json(
        ARTIFACT_DIR / "config.json"
    )

    scaler = joblib.load(
        ARTIFACT_DIR / "temperature_scaler.pkl"
    )

    city_to_id = load_json(
        ARTIFACT_DIR / "city_to_id.json"
    )

    return {
        "config": config,
        "target_scaler": scaler,
        "city_to_id": city_to_id
    }


# ============================================================
# LOAD V2 ARTIFACTS
# ============================================================

def load_v2_artifacts():

    print("Loading V2 artifacts...")

    config = load_json(
        V2_ARTIFACT_DIR / "config.json"
    )

    temporal_scaler = joblib.load(
        V2_ARTIFACT_DIR / "temporal_scaler.pkl"
    )

    target_scaler = joblib.load(
        V2_ARTIFACT_DIR / "target_scaler.pkl"
    )

    geo_scaler_A = joblib.load(
        V2_ARTIFACT_DIR / "geo_scaler_A.pkl"
    )

    geo_scaler_B = joblib.load(
        V2_ARTIFACT_DIR / "geo_scaler_B.pkl"
    )

    city_to_id = load_json(
        V2_ARTIFACT_DIR / "city_to_id.json"
    )

    geographical_features = (
        V2_ARTIFACT_DIR / "geographical_features.csv"
    )

    check_file(geographical_features)

    import pandas as pd

    geographical_features = pd.read_csv(
        geographical_features
    )

    return {
        "config": config,
        "temporal_scaler": temporal_scaler,
        "target_scaler": target_scaler,
        "geo_scaler_A": geo_scaler_A,
        "geo_scaler_B": geo_scaler_B,
        "city_to_id": city_to_id,
        "geographical_features": geographical_features
    }


# ============================================================
# LOAD TRAINED MODEL CHECKPOINT
# ============================================================

def load_trained_model(model_name, city_to_id):

    print(f"Loading {model_name}...")

    model_paths = {
        "baseline": MODEL_DIR / "best_weather_lstm.pth",
        "model_A": MODEL_DIR / "best_multivariate_model_A.pth",
        "model_B": MODEL_DIR / "best_multivariate_model_B.pth"
    }

    if model_name not in model_paths:
        raise ValueError(
            f"Unknown model name: {model_name}"
        )

    model_path = model_paths[model_name]

    check_file(model_path)

    # Construct the exact architecture
    model = create_model(
        model_name,
        num_cities=len(city_to_id)
    ).to(DEVICE)

    # Load trained weights
    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# LOAD ALL THREE TRAINED MODELS AND ARTIFACTS
# ============================================================

def load_all_artifacts():

    print("\n================================")
    print("LSTM PLAYGROUND - ARTIFACT LOADER")
    print("================================")

    print("Device:", DEVICE)

    v1 = load_v1_artifacts()
    v2 = load_v2_artifacts()

    # Check city mappings are consistent
    if v1["city_to_id"] != v2["city_to_id"]:
        raise ValueError(
            "V1 and V2 city mappings do not match!"
        )

    city_to_id = v2["city_to_id"]

    # Load trained models
    baseline = load_trained_model(
        "baseline",
        city_to_id
    )

    model_A = load_trained_model(
        "model_A",
        city_to_id
    )

    model_B = load_trained_model(
        "model_B",
        city_to_id
    )

    print("\nAll artifacts loaded successfully!")

    return {
        "device": DEVICE,

        "models": {
            "baseline": baseline,
            "model_A": model_A,
            "model_B": model_B
        },

        "v1": v1,
        "v2": v2,

        "city_to_id": city_to_id
    }
