import json

def save_params(params, model_path, model_name):
    # Try to load existing parameters from the file
    try:
        with open(model_path, 'r') as f:
            existing_params = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, initialize with an empty array
        existing_params = {}

    # Append the new parameters to the existing array
    existing_params[model_name] = params

    # Save the updated array to the file
    with open(model_path, 'w') as f:
        json.dump(existing_params, f)

def load_params(model_path):
    with open(model_path, 'r') as f:
        return json.load(f)