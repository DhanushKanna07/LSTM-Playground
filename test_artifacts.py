from src.artifacts import load_all_artifacts

# Load everything
artifacts = load_all_artifacts()

print("\nMODEL VERIFICATION")

for name, model in artifacts["models"].items():

    print(f"\n{name.upper()}")

    print("Architecture:", model.__class__.__name__)
    print("Device:", next(model.parameters()).device)
    print("Training mode:", model.training)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", parameters)

print("\nAvailable cities:", len(artifacts["city_to_id"]))
print("\nArtifact loading test completed!")