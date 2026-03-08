import os
import json
import csv
import uuid
import re
from rpgmv_translator.utils import is_valid_field_to_translate

class JSONHandler:
    def __init__(self, directory, specific_file=None):
        self.directory = directory
        self.specific_file = specific_file
        self.original_csv = os.path.join(directory, 'original.csv')
        self.translated_csv = os.path.join(directory, 'translated.csv')

    def read_and_process_jsons(self):
        existing_entries = self._load_existing_entries(self.original_csv)
        new_entries = {}

        for dirpath, dirnames, filenames in os.walk(self.directory):
            for file_name in filenames:
                if file_name.endswith('.json') and (self.specific_file is None or file_name == self.specific_file):
                    file_path = os.path.join(dirpath, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        self._process_json(data, existing_entries, new_entries)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)

        self._write_new_entries_to_csv(new_entries, self.original_csv)

    def _process_json(self, data, existing_entries, new_entries):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self._process_json(value, existing_entries, new_entries)
                elif isinstance(value, str):
                    data[key] = self._process_string(value, existing_entries, new_entries)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    self._process_json(item, existing_entries, new_entries)
                elif isinstance(item, str):
                    data[i] = self._process_string(item, existing_entries, new_entries)

    def _process_string(self, value, existing_entries, new_entries):
        if self._is_xml_like(value):
            return self._process_xml_like_field(value, existing_entries, new_entries)
        elif is_valid_field_to_translate(value):
            entry_uuid = existing_entries.get(value, str(uuid.uuid4()))
            existing_entries[value] = entry_uuid
            new_entries[value] = entry_uuid
            return entry_uuid
        else:
            return value

    def _load_existing_entries(self, csv_path):
        entries = {}
        try:
            with open(csv_path, mode='r', encoding='utf-8', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    row_uuid = row.get('uuid')
                    text = row.get('text')
                    if row_uuid and text:
                        entries[text] = row_uuid
        except FileNotFoundError:
            pass
        return entries

    def _is_xml_like(self, value):
        # Determine if the value is XML-like
        return '<' in value and '>' in value

    def _process_xml_like_field(self, value, existing_entries, new_entries):
        # Process only specific parts of the XML-like field
        def replace_func(match):
            original_text = match.group(1)
            entry_uuid = existing_entries.get(original_text, str(uuid.uuid4()))
            existing_entries[original_text] = entry_uuid
            new_entries[original_text] = entry_uuid
            return f"<hintId:{entry_uuid}>"

        return re.sub(r'<hint:(.*?)>', replace_func, value)

    def _write_new_entries_to_csv(self, new_entries, csv_path):
        # Check if the file is new or empty (i.e., needs a header)
        file_is_new = not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0

        with open(csv_path, mode='a', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # If the file is new, write the header first
            if file_is_new:
                writer.writerow(['uuid', 'text'])

            # Write the new entries
            for text, entry_uuid in new_entries.items():
                writer.writerow([entry_uuid, text])

    def update_jsons_with_translations(self):
        translated_entries = self._load_translated_entries(self.translated_csv, self.original_csv)

        for dirpath, dirnames, filenames in os.walk(self.directory):
            for file_name in filenames:
                if file_name.endswith('.json') and (self.specific_file is None or file_name == self.specific_file):
                    file_path = os.path.join(dirpath, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        self._update_json(data, translated_entries)
                    with open(file_path, 'w', encoding='utf-8') as file:
                        json.dump(data, file, ensure_ascii=False, indent=4)

    def _update_json(self, data, translated_entries):
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    self._update_json(value, translated_entries)
                elif isinstance(value, str):
                    if self._is_xml_like(value):
                        updated_value = self._update_xml_like_field(value, translated_entries)
                        data[key] = updated_value
                    elif value in translated_entries:
                        data[key] = translated_entries[value]
        elif isinstance(data, list):
            for item in data:
                self._update_json(item, translated_entries)

    def _update_xml_like_field(self, value, translated_entries):
        # Update only specific parts of the XML-like field
        def replace_func(match):
            uuid_text = match.group(1)
            translated_text = translated_entries.get(uuid_text, uuid_text)  # Fallback to original if not translated
            return f"<hint:{translated_text}>"

        return re.sub(r'<hintId:(.*?)>', replace_func, value)

    def _load_translated_entries(self, translated_csv, original_csv):
        entries = {}
        try:
            with open(translated_csv, mode='r', encoding='utf-8', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    row_uuid = row.get('uuid')
                    translated_text = row.get('translated_text')
                    if row_uuid and translated_text is not None:
                        entries[row_uuid] = translated_text
        except FileNotFoundError:
            with open(original_csv, mode='r', encoding='utf-8', newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    row_uuid = row.get('uuid')
                    text = row.get('text')
                    if row_uuid and text is not None:
                        entries[row_uuid] = text
        return entries
