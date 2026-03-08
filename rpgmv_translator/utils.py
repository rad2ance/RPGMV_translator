import os
import shutil
import re
# import nltk
# from nltk.tokenize import word_tokenize


def read_progress_log(directory):
    progress_log_path = os.path.join(directory, 'progress.log')
    progress = {}
    if os.path.exists(progress_log_path):
        with open(progress_log_path, 'r') as file:
            for line in file:
                key, value = line.strip().split(':')
                progress[key] = value == 'completed'
    return progress

def update_progress_log(directory, step):
    progress_log_path = os.path.join(directory, 'progress.log')
    with open(progress_log_path, 'a') as file:
        file.write(f"{step}:completed\n")

def is_rpgmv_folder(directory):
    required_folders = ['audio', 'data', 'img']
    www_path = os.path.join(directory, 'www')

    if os.path.exists(www_path) and all(os.path.exists(os.path.join(www_path, folder)) for folder in required_folders):
        return True
    elif all(os.path.exists(os.path.join(directory, folder)) for folder in required_folders):
        return True
    return False


def duplicate_json_files(directory, specific_file=None):
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.json') and (specific_file is None or filename == specific_file):
                original_path = os.path.join(dirpath, filename)
                backup_path = os.path.join(dirpath, f"{filename}.old")
                if not os.path.exists(backup_path):
                    shutil.copy2(original_path, backup_path)


def restore_from_backup(directory):
    if not is_rpgmv_folder(directory):
        raise ValueError("The specified directory is not a valid RPGMV folder.")

    # List of generated files that should be removed if they exist
    new_files = [
        'original.csv',
        'translated.csv',
        'translated.csv.part',
        'progress.log',
    ]

    # Remove new files if they exist
    for new_file in new_files:
        new_file_path = os.path.join(directory, new_file)
        if os.path.exists(new_file_path):
            os.remove(new_file_path)

    restored_any = False
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.json.old'):
                backup_path = os.path.join(dirpath, filename)
                original_path = os.path.join(dirpath, filename.replace('.old', ''))
                shutil.copy2(backup_path, original_path)
                os.remove(backup_path)
                restored_any = True

    if not restored_any:
        raise FileNotFoundError("No .old backup files found to restore.")


def contains_japanese(text):
    # Regular expression that matches Japanese Hiragana and Katakana
    # Hiragana: U+3040 to U+309F, Katakana: U+30A0 to U+30FF

    japanese_regex = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'

    # Check for the presence of Hiragana, Katakana, or Kanji characters
    if re.search(japanese_regex, text):
        return True

    return False

def contains_japanese_strict(text):
    # Regular expression that matches Japanese Hiragana and Katakana characters
    # Hiragana: U+3040 to U+309F, Katakana: U+30A0 to U+30FF
    japanese_strict_regex = r'[\u3040-\u309F\u30A0-\u30FF]'

    # Check for the presence of Hiragana or Katakana characters strictly
    if re.search(japanese_strict_regex, text):
        return True

    return False
def is_valid_field_to_translate(text):
    # Detect Japanese using contains_japanese function
    if contains_japanese(text):
        # Check for the presence of equal sign, slash, or other non-Japanese characters
        if re.search(r'[$=/<>{}_]', text):
            return False
        # Check for backslashes that are not followed by 'n'
        if re.search(r'\\(?!n)', text):
            return False
        else:
            return True
    else:
        return False
