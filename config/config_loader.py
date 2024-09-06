import yaml

def load_config(config_path):
    """Loads configuration from the provided YAML file."""
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f"Error loading configuration: {exc}")
            return None
