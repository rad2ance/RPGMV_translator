import argparse
import os
from rpgmv_translator.translate import RPGMVTranslator  # Import the RPGMVTranslator class
from rpgmv_translator.request_controller import GPTRequestController
import rpgmv_translator.config_manager as config_manager
import rpgmv_translator.utils as utils

def main():
    parser = argparse.ArgumentParser(description="RPGMV Translator Command Line Tool")
    parser.add_argument('-addkey', '--addkey', type=str, help='Add API key to config')
    parser.add_argument('-showkey', '--showkey', action='store_true', help='Show API key')
    parser.add_argument('-reset', '--reset', action='store_true', help='Reset config')
    parser.add_argument('-translate', '--translate', type=str, nargs='?', const=os.getcwd(), default=None, help='Start translating. Specify the directory path (optional).')
    parser.add_argument('-restore', '--restore', type=str, nargs='?', const=os.getcwd(), default=None, help='Restore data from .old backups. Specify the directory path (optional).')
    parser.add_argument('--translate-csv', type=str, default=None, help='Translate a specific CSV file directly.')
    parser.add_argument('--column', type=str, default=None, help='CSV source column to translate (used with --translate-csv).')
    parser.add_argument('--output-csv', type=str, default=None, help='Output CSV path for --translate-csv (defaults to input path).')
    parser.add_argument('--target-column', type=str, default='translated_text', help='Output translated column name for --translate-csv.')
    parser.add_argument('--id-column', type=str, default=None, help='Optional stable row id column for resume support in --translate-csv mode.')
    parser.add_argument('--language', type=str, default='Japanese', help='Tokenizer language (default: Japanese).')
    parser.add_argument('--max-tokens', type=int, default=300, help='Max token batch size per request.')

    args = parser.parse_args()

    if args.addkey:
        config_manager.add_key(args.addkey)
        print("API key added to config.")
    elif args.showkey:
        key = config_manager.show_key()
        print(f"API Key: {key}")
    elif args.reset:
        config_manager.reset_config()
        print("Config reset.")
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

        controller = GPTRequestController(max_tokens=args.max_tokens, language=args.language)
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
