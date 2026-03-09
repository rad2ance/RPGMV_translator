import argparse
import os
import unittest
from rpgmv_translator.translate import RPGMVTranslator  # Import the RPGMVTranslator class
from rpgmv_translator.request_controller import GPTRequestController
import rpgmv_translator.config_manager as config_manager
import rpgmv_translator.utils as utils


def _run_integration_tests():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    integration_dir = os.path.join(repo_root, 'tests', 'integration')
    suite = unittest.defaultTestLoader.discover(start_dir=integration_dir, pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main():
    config = config_manager.load_config()
    default_language = config.get('language', 'Japanese')
    default_max_tokens = int(config.get('max_tokens', 2000))
    default_model = config.get('model', 'gpt-4.1-mini')
    default_target_language = config.get('target_language', 'Chinese')
    quality = config.get('quality', {})
    quality_config = {
        'key_coverage_rate_threshold': float(quality.get('key_coverage_rate_threshold', 0.9)),
        'missing_key_abs_threshold': int(quality.get('missing_key_abs_threshold', 2)),
        'max_on_the_fly_retries': int(quality.get('max_on_the_fly_retries', 2)),
        'max_consecutive_bad_calls': int(quality.get('max_consecutive_bad_calls', 5)),
        'missing_marker_prefix': quality.get('missing_marker_prefix', '__MISSING_TRANSLATION__'),
        'cost_confirmation_usd': float(quality.get('cost_confirmation_usd', 10.0)),
    }

    parser = argparse.ArgumentParser(description="RPGMV Translator Command Line Tool")
    parser.add_argument('-addkey', '--addkey', type=str, help='Add API key to config')
    parser.add_argument('-showkey', '--showkey', action='store_true', help='Show API key')
    parser.add_argument('-reset', '--reset', action='store_true', help='Reset config')
    parser.add_argument('-test', '--test', action='store_true', help='Run integration tests')
    parser.add_argument('-translate', '--translate', type=str, nargs='?', const=os.getcwd(), default=None, help='Start translating. Specify the directory path (optional).')
    parser.add_argument('-restore', '--restore', type=str, nargs='?', const=os.getcwd(), default=None, help='Restore data from .old backups. Specify the directory path (optional).')
    parser.add_argument('--translate-csv', type=str, default=None, help='Translate a specific CSV file directly.')
    parser.add_argument('--column', type=str, default=None, help='CSV source column to translate (used with --translate-csv).')
    parser.add_argument('--output-csv', type=str, default=None, help='Output CSV path for --translate-csv (defaults to input path).')
    parser.add_argument('--target-column', type=str, default='translated_text', help='Output translated column name for --translate-csv.')
    parser.add_argument('--id-column', type=str, default=None, help='Optional stable row id column for resume support in --translate-csv mode.')
    parser.add_argument('--language', type=str, default=default_language, help='Tokenizer language. Uses config default if set.')
    parser.add_argument('--max-tokens', type=int, default=default_max_tokens, help='Max token batch size per request. Uses config default if set.')
    parser.add_argument('--model', type=str, default=default_model, help='OpenAI model to use for translation. Uses config default if set.')
    parser.add_argument('--target-language', type=str, default=default_target_language, help='Target translation language. Uses config default if set.')

    args = parser.parse_args()

    if args.addkey:
        config_manager.add_key(args.addkey)
        print("API key added to config.")
    elif args.test:
        raise SystemExit(_run_integration_tests())
    elif args.showkey:
        key = config_manager.show_key()
        print(f"API Key: {key}")
    elif args.reset:
        confirm = input("Reset config to defaults and remove stored API key? Type 'yes' to continue: ").strip().lower()
        if confirm in {'y', 'yes'}:
            config_manager.reset_config()
            print("Config reset to defaults.")
        else:
            print("Config reset cancelled.")
    elif args.restore is not None:
        try:
            utils.restore_from_backup(args.restore)
            print(f"Data successfully restored from backups in {args.restore}")
        except Exception as e:
            print(f"Error: {e}")
    elif args.translate is not None:
        directory = args.translate
        print("Starting translation...")
        translator = RPGMVTranslator(directory)
        translator.translate()
        print("Translation completed.")
    elif args.translate_csv is not None:
        if not args.column:
            parser.error("--column is required when using --translate-csv")
        input_csv = args.translate_csv
        output_csv = args.output_csv or input_csv

        controller = GPTRequestController(
            max_tokens=args.max_tokens,
            language=args.language,
            model=args.model,
            target_language=args.target_language,
            quality_config=quality_config,
        )
        controller.process_arbitrary_csv(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            source_column=args.column,
            target_column=args.target_column,
            id_column=args.id_column,
        )
        print(f"CSV translation completed: {output_csv}")

if __name__ == '__main__':
    main()
