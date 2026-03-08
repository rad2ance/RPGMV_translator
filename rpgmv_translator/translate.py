import os
import sys
from rpgmv_translator.utils import is_rpgmv_folder, duplicate_json_files
from rpgmv_translator.json_handler import JSONHandler
from rpgmv_translator.request_controller import GPTRequestController
from rpgmv_translator.utils import read_progress_log, update_progress_log
import rpgmv_translator.config_manager as config_manager

class RPGMVTranslator:
    def __init__(self, path):
        # Check if the path is an existing file
        if os.path.isfile(path) and path.endswith('.json'):
            self.directory = os.path.dirname(path)
            self.specific_file = os.path.basename(path)
        elif path.endswith('.json'):
            # Attempt to concatenate with the absolute working directory
            abs_working_dir = os.path.abspath(os.getcwd())
            full_path = os.path.join(abs_working_dir, path)
            if os.path.isfile(full_path):
                self.directory = os.path.dirname(full_path)
                self.specific_file = os.path.basename(full_path)
            else:
                raise ValueError(f"The file path {path} does not exist.")
        else:
            # If a directory is given, use it directly
            self.directory = path
            self.specific_file = None


        self.json_handler = JSONHandler(self.directory, self.specific_file)

        print(f"Operating on directory: {self.directory}")
        if self.specific_file:
            print(f"Specific file to process: {self.specific_file}")
        else:
            print("Processing all JSON files in the directory.")


    def translate(self):
        if not is_rpgmv_folder(self.directory):
            raise ValueError("The path is not a valid RPGMV folder.")

        progress = read_progress_log(self.directory)

        if not progress.get('duplicate_json_files'):
            duplicate_json_files(self.directory, self.specific_file)
            update_progress_log(self.directory, 'duplicate_json_files')

        if not progress.get('read_and_process_jsons'):
            self.json_handler.read_and_process_jsons()
            update_progress_log(self.directory, 'read_and_process_jsons')

        if not progress.get('process_csv'):
            # Assuming GPTRequestController has been properly implemented
            controller = GPTRequestController(
                max_tokens=int(config_manager.get_setting('max_tokens', 2000)),
                language=config_manager.get_setting('language', 'Japanese'),
                model=config_manager.get_setting('model', 'gpt-4.1-mini'),
            )
            controller.process_csv(self.json_handler.original_csv, self.json_handler.translated_csv)
            update_progress_log(self.directory, 'process_csv')

        if not progress.get('update_jsons_with_translations'):
            self.json_handler.update_jsons_with_translations()
            update_progress_log(self.directory, 'update_jsons_with_translations')

def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    translator = RPGMVTranslator(directory)
    translator.translate()

if __name__ == "__main__":
    main()
