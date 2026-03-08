import json
import os

# Get the directory where the current script is located (package directory)
package_directory = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(package_directory, 'config.json')
DEFAULT_CONFIG = {
    'max_tokens': 2000,
    'language': 'Japanese',
    'model': 'gpt-4.1-mini',
}


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return dict(DEFAULT_CONFIG)

    with open(CONFIG_FILE, 'r', encoding='utf-8') as file:
        try:
            user_config = json.load(file)
        except json.JSONDecodeError:
            user_config = {}

    config = dict(DEFAULT_CONFIG)
    if isinstance(user_config, dict):
        config.update(user_config)
    return config

def add_key(api_key):
    config = load_config()
    config['openai_api_key'] = api_key
    with open(CONFIG_FILE, 'w', encoding='utf-8') as file:
        json.dump(config, file, indent=4)

def show_key():
    config = load_config()
    return config.get('openai_api_key', 'No key found')


def get_setting(key, default=None):
    config = load_config()
    if key in config:
        return config[key]
    return default

def reset_config():
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
