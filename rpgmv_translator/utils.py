import os
import shutil
import re
# import nltk
# from nltk.tokenize import word_tokenize


def is_rpgmv_folder(directory):
    required_folders = ['audio', 'data', 'img']
    www_path = os.path.join(directory, 'www')

    if os.path.exists(www_path) and all(os.path.exists(os.path.join(www_path, folder)) for folder in required_folders):
        return True
    elif all(os.path.exists(os.path.join(directory, folder)) for folder in required_folders):
        return True
    return False


def duplicate_json_files(directory):
    if not is_rpgmv_folder(directory):
        raise ValueError("The specified directory is not a valid RPGMV folder.")

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.json'):
                original_path = os.path.join(dirpath, filename)
                backup_path = os.path.join(dirpath, f"{filename}.old")
                shutil.copy2(original_path, backup_path)

def restore_from_backup(directory):
    if not is_rpgmv_folder(directory):
        raise ValueError("The specified directory is not a valid RPGMV folder.")

    restored_any = False
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.json.old'):
                backup_path = os.path.join(dirpath, filename)
                original_path = os.path.join(dirpath, filename.replace('.old', ''))
                shutil.copy2(backup_path, original_path)
                restored_any = True

    if not restored_any:
        raise FileNotFoundError("No .old backup files found to restore.")


def contains_japanese(text):
    # Regular expression that matches Japanese Hiragana and Katakana
    # Hiragana: U+3040 to U+309F, Katakana: U+30A0 to U+30FF
    japanese_regex = r'[\u3040-\u309F\u30A0-\u30FF]'

    # Check for the presence of Hiragana or Katakana
    if re.search(japanese_regex, text):
        return True

    # If needed, add additional logic here to handle Kanji more selectively
    # ...

    return False
