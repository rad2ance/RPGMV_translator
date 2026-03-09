import csv
import os

from rpgmv_translator.entity.const import DEFAULT_SOURCE_ENCODING


def read_csv_rows(csv_path, required_columns=None):
    with open(csv_path, "r", encoding=DEFAULT_SOURCE_ENCODING, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError("Input CSV has no header row.")

        if required_columns:
            missing = [col for col in required_columns if col not in fieldnames]
            if missing:
                raise ValueError(
                    f"Input CSV is missing required columns {missing}. Found: {fieldnames}"
                )

        rows = list(reader)
    return rows, fieldnames


def row_identifier(row, id_column, index):
    if id_column:
        return row.get(id_column)
    return str(index)


def build_pending_items(rows, source_column, id_column, existing_translations, overwrite_existing):
    pending_items = []
    for index, row in enumerate(rows):
        row_id = row_identifier(row, id_column, index)
        if row_id is None or row_id == "":
            continue

        source_text = (row.get(source_column) or "").strip()
        if not source_text:
            continue

        if not overwrite_existing and row_id in existing_translations and existing_translations[row_id]:
            continue

        pending_items.append((row_id, source_text))
    return pending_items


def build_row_lookup(rows, source_column, id_column):
    row_lookup = {}
    for index, row in enumerate(rows):
        row_id = row_identifier(row, id_column, index)
        if row_id is None or row_id == "":
            continue
        row_lookup[row_id] = row.get(source_column) or ""
    return row_lookup


def get_temp_path(file_path, suffix=".part"):
    return f"{file_path}{suffix}"


def load_translation_map(file_path, id_column, translation_column):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, "r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames or id_column not in reader.fieldnames or translation_column not in reader.fieldnames:
            return {}

        output_map = {}
        for row in reader:
            row_id = row.get(id_column)
            if row_id is None:
                continue
            output_map[row_id] = row.get(translation_column, "")
        return output_map


def load_existing_output_map(output_csv_path, target_column, id_column):
    if not os.path.exists(output_csv_path):
        return {}

    with open(output_csv_path, "r", encoding="utf-8", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        if not reader.fieldnames or target_column not in reader.fieldnames:
            return {}

        output_map = {}
        for index, row in enumerate(reader):
            row_id = row_identifier(row, id_column, index)
            if row_id is None:
                continue
            output_map[row_id] = row.get(target_column, "")
        return output_map


def write_translation_map(file_path, id_column, translation_column, items):
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [id_column, translation_column]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row_id, translated_text in items.items():
            writer.writerow({id_column: row_id, translation_column: translated_text})


def append_translation_map(file_path, id_column, translation_column, items):
    current = load_translation_map(file_path, id_column, translation_column)
    current.update(items)
    write_translation_map(file_path, id_column, translation_column, current)


def collect_marker_items_from_map(translated_map, row_lookup, is_missing_marker_fn):
    items = []
    for row_id, translated in translated_map.items():
        if is_missing_marker_fn(translated):
            original_text = row_lookup.get(row_id, "")
            if original_text:
                items.append((row_id, original_text))
    return items
