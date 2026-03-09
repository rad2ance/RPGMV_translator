import json
import os

# Get the directory where the current script is located (package directory)
package_directory = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(package_directory, 'config.json')
DEFAULT_CONFIG = {
    'max_tokens': 2000,
    'language': 'Japanese',
    'model': 'gpt-4.1-mini',
    'target_language': 'Chinese',
    'quality': {
        'key_coverage_rate_threshold': 0.9,
        'missing_key_abs_threshold': 2,
        'max_on_the_fly_retries': 2,
        'max_consecutive_bad_calls': 5,
        'missing_marker_prefix': '__MISSING_TRANSLATION__',
        'cost_confirmation_usd': 10.0,
    },
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
    config['quality'] = dict(DEFAULT_CONFIG['quality'])
    if isinstance(user_config, dict):
        config.update(user_config)
        if isinstance(user_config.get('quality'), dict):
            config['quality'].update(user_config['quality'])

        # Backward compatibility for older flat quality keys.
        legacy_keys = [
            'key_coverage_rate_threshold',
            'missing_key_abs_threshold',
            'max_on_the_fly_retries',
            'max_consecutive_bad_calls',
            'missing_marker_prefix',
            'cost_confirmation_usd',
        ]
        for legacy_key in legacy_keys:
            if legacy_key in user_config:
                config['quality'][legacy_key] = user_config[legacy_key]
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
    # Recreate config with defaults instead of removing the file.
    with open(CONFIG_FILE, 'w', encoding='utf-8') as file:
        json.dump(DEFAULT_CONFIG, file, indent=4)
