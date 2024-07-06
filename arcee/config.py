import json
import os
from pathlib import Path

from typer import get_app_dir


def get_configuration_path() -> Path:
    """Gets the configuration file path.
    Uses typer/click's get_app_dir to get the configuration file path
    and allows an ARCEE_CONFIG_LOCATION override.
    """
    env_location = os.getenv("ARCEE_CONFIG_LOCATION")
    if env_location:
        env_path: Path = Path(env_location)
        if not env_path.is_file():
            raise FileNotFoundError(f"Configuration file not found at {env_path}")
        return env_path

    app_dir = get_app_dir("arcee")
    conf_path: Path = Path(app_dir) / "config.json"

    return conf_path


def write_configuration_value(key: str, value: str) -> None:
    """Writes a configuration value to the configuration file.
    Args:
        key (string): The name of the configuration variable.
        value (string): The value of the configuration variable.
    """
    conf_path = get_configuration_path()
    config = {}

    if conf_path.is_file():
        with open(conf_path, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                pass
    else:
        conf_path.touch()

    config[key] = value

    with open(conf_path, "w") as f:
        json.dump(config, f)


def get_conditional_configuration_variable(key: str, default: str) -> str:
    """Retrieves the configuration variable conditionally.
        ##1. check if variable is in environment
        ##2. check if variable is in config file
        ##3. return default value
    Args:
        key (string): The name of the configuration variable.
        default (string): The default value of the configuration variable.
    Returns:
        string: The value of the conditional configuration variable.
    """
    conf_location = get_configuration_path()

    if os.path.exists(conf_location):
        with open(conf_location) as f:
            config = json.load(f)
    else:
        config = {}

    return (os.getenv(key) or config.get(key)) or default


ARCEE_API_URL = get_conditional_configuration_variable("ARCEE_API_URL", "https://app.arcee.ai/api")
ARCEE_APP_URL = get_conditional_configuration_variable("ARCEE_APP_URL", "https://app.arcee.ai")
ARCEE_API_KEY = get_conditional_configuration_variable("ARCEE_API_KEY", "")
ARCEE_API_VERSION = get_conditional_configuration_variable("ARCEE_API_VERSION", "v2")
ARCEE_ORG = get_conditional_configuration_variable("ARCEE_ORG", "")
