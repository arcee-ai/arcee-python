import json
import os


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

    os_name = os.name

    if os_name == "nt":
        default_path = os.getenv("USERPROFILE", "") + "\\arcee\\config.json"
    else:
        default_path = os.getenv("HOME", "") + "/.config/arcee/config.json"

    # default configuration location
    conf_location = os.getenv(
        "ARCEE_CONFIG_LOCATION",
        default=default_path,
    )

    if os.path.exists(conf_location):
        with open(conf_location) as f:
            config = json.load(f)
    else:
        config = {}

    return (os.getenv(key) or config.get(key)) or default


ARCEE_API_URL = get_conditional_configuration_variable("ARCEE_API_URL", "https://api.arcee.ai")
ARCEE_APP_URL = get_conditional_configuration_variable("ARCEE_APP_URL", "https://app.arcee.ai")
ARCEE_API_KEY = get_conditional_configuration_variable("ARCEE_API_KEY", "")
ARCEE_API_VERSION = get_conditional_configuration_variable("ARCEE_API_VERSION", "v2")
