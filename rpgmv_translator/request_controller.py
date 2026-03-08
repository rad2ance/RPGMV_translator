import csv
import json
import os

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from rpgmv_translator.translator.gpt_translator import GPTTranslator


class GPTRequestController:
    def __init__(self, max_tokens, language, model="gpt-4.1-mini"):
        self.max_tokens = max_tokens
        self.language = language
        self.model = model
        self.api_key = self._load_api_key_from_config("config.json")
        self.translator = GPTTranslator(self.api_key)
        self.tokenizer = self._select_tokenizer(language)

    def _load_api_key_from_config(self, file_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, file_name)

        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
                api_key = config.get("openai_api_key")
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("API key not found in config file or OPENAI_API_KEY.")
                return api_key
        except FileNotFoundError as exc:
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                return env_key
            raise FileNotFoundError(f"Config file not found at path: {config_path}") from exc

    def _select_tokenizer(self, language):
        if language == "English":
            from rpgmv_translator.tokenizer.english_tokenizer import EnglishTokenizer
            return EnglishTokenizer()
        if language == "Japanese":
            from rpgmv_translator.tokenizer.japanese_tokenizer import JapaneseTokenizer
            return JapaneseTokenizer()
        raise ValueError(f"Unsupported language: {language}")

    def _estimate_tokens_and_price(self, texts):
        return self.translator.estimate_tokens_and_price(texts, self.tokenizer, self.model)

    def process_csv(self, original_csv_path, translated_csv_path):
        with open(original_csv_path, "r", encoding="utf-8-sig", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            required_columns = {"uuid", "text"}
            if not reader.fieldnames or not required_columns.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"Input CSV must contain columns {sorted(required_columns)}. "
                    f"Found: {reader.fieldnames}"
                )
            rows = list(reader)

        temp_translated_csv_path = self._get_temp_path(translated_csv_path)
        source_progress_path = temp_translated_csv_path if os.path.exists(temp_translated_csv_path) else translated_csv_path
        existing_translations = self._load_translation_map(source_progress_path, "uuid", "translated_text")
        self._write_translation_map(temp_translated_csv_path, "uuid", "translated_text", existing_translations)

        processed_uuids = set(existing_translations.keys())
        pending_items = []
        for row in rows:
            row_uuid = row.get("uuid", "")
            text = row.get("text", "")
            if row_uuid and row_uuid not in processed_uuids and text:
                pending_items.append((row_uuid, text))

        token_count, estimated_price = self._estimate_tokens_and_price([text for _, text in pending_items])
        print(f"Total tokens to translate: {token_count}")
        print(f"Estimated price for translation: ${estimated_price:.2f}")

        self._translate_items_in_batches(
            pending_items,
            on_batch_translated=lambda batch: self._append_translation_map(
                temp_translated_csv_path, "uuid", "translated_text", batch
            ),
        )
        os.replace(temp_translated_csv_path, translated_csv_path)

    def process_arbitrary_csv(self, input_csv_path, output_csv_path, source_column, target_column="translated_text", id_column=None):
        with open(input_csv_path, "r", encoding="utf-8-sig", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                raise ValueError("Input CSV has no header row.")
            if source_column not in reader.fieldnames:
                raise ValueError(f"Source column '{source_column}' was not found in CSV header: {reader.fieldnames}")
            if id_column and id_column not in reader.fieldnames:
                raise ValueError(f"ID column '{id_column}' was not found in CSV header: {reader.fieldnames}")
            rows = list(reader)
            input_fieldnames = list(reader.fieldnames)

        progress_path = self._get_temp_path(output_csv_path)
        existing_translations = self._load_existing_output_map(progress_path, target_column, id_column)
        if not existing_translations:
            existing_translations = self._load_existing_output_map(output_csv_path, target_column, id_column)

        temp_id_column = id_column or "__row_index__"
        self._write_translation_map(progress_path, temp_id_column, target_column, existing_translations)
        pending_items = []

        for index, row in enumerate(rows):
            row_id = row.get(id_column) if id_column else str(index)
            source_text = (row.get(source_column) or "").strip()
            if not source_text:
                continue
            if row_id in existing_translations and existing_translations[row_id]:
                continue
            pending_items.append((row_id, source_text))

        token_count, estimated_price = self._estimate_tokens_and_price([text for _, text in pending_items])
        print(f"Total tokens to translate: {token_count}")
        print(f"Estimated price for translation: ${estimated_price:.2f}")

        new_translations = self._translate_items_in_batches(
            pending_items,
            on_batch_translated=lambda batch: self._append_translation_map(
                progress_path, temp_id_column, target_column, batch
            ),
        )

        output_fieldnames = list(input_fieldnames)
        if target_column not in output_fieldnames:
            output_fieldnames.append(target_column)

        temp_output_path = self._get_temp_path(output_csv_path, suffix=".output.tmp")
        with open(temp_output_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                row_id = row.get(id_column) if id_column else str(index)
                source_text = row.get(source_column) or ""

                translated = None
                if row_id in new_translations:
                    translated = new_translations[row_id]
                elif row_id in existing_translations:
                    translated = existing_translations[row_id]
                elif source_text.strip():
                    translated = source_text

                row[target_column] = translated if translated is not None else row.get(target_column, "")
                writer.writerow(row)

        os.replace(temp_output_path, output_csv_path)
        if os.path.exists(progress_path):
            os.remove(progress_path)

    def _translate_items_in_batches(self, items, on_batch_translated=None):
        translations = {}
        batch_ids = []
        batch_texts = []
        current_token_count = 0

        for item_id, text in tqdm(items, total=len(items), desc="Translating..."):
            token_count = self.tokenizer.get_token_count(text)

            if token_count > self.max_tokens:
                if batch_texts:
                    batch_translations = self.translator.translate(batch_texts, model=self.model)
                    batch_map = {}
                    for translated_id, translated_text in zip(batch_ids, batch_translations):
                        batch_map[translated_id] = translated_text
                        translations[translated_id] = translated_text
                    if on_batch_translated and batch_map:
                        on_batch_translated(batch_map)
                    batch_ids = []
                    batch_texts = []
                    current_token_count = 0

                split_texts = self._split_text(text, self.max_tokens)
                translated_segments = self.translator.translate(split_texts, model=self.model)
                translations[item_id] = "".join(translated_segments)
                if on_batch_translated:
                    on_batch_translated({item_id: translations[item_id]})
                continue

            if current_token_count + token_count > self.max_tokens and batch_texts:
                batch_translations = self.translator.translate(batch_texts, model=self.model)
                batch_map = {}
                for translated_id, translated_text in zip(batch_ids, batch_translations):
                    batch_map[translated_id] = translated_text
                    translations[translated_id] = translated_text
                if on_batch_translated and batch_map:
                    on_batch_translated(batch_map)
                batch_ids = []
                batch_texts = []
                current_token_count = 0

            batch_ids.append(item_id)
            batch_texts.append(text)
            current_token_count += token_count

        if batch_texts:
            batch_translations = self.translator.translate(batch_texts, model=self.model)
            batch_map = {}
            for translated_id, translated_text in zip(batch_ids, batch_translations):
                batch_map[translated_id] = translated_text
                translations[translated_id] = translated_text
            if on_batch_translated and batch_map:
                on_batch_translated(batch_map)

        return translations

    def _split_text(self, text, max_tokens):
        if hasattr(self.tokenizer, "split_text_by_max_tokens"):
            return self.tokenizer.split_text_by_max_tokens(text, max_tokens)

        # Fallback for non-token-id tokenizers.
        tokens = self.tokenizer.tokenize(text)
        segments = []
        current_segment = []
        current_token_count = 0

        for token in tokens:
            token_text = str(token)
            token_count = self.tokenizer.get_token_count(token_text)
            if current_token_count + token_count <= max_tokens:
                current_segment.append(token_text)
                current_token_count += token_count
            else:
                segments.append("".join(current_segment))
                current_segment = [token_text]
                current_token_count = token_count

        if current_segment:
            segments.append("".join(current_segment))

        return segments

    def _get_processed_uuids(self, translated_csv_path):
        return set(self._load_translation_map(translated_csv_path, "uuid", "translated_text").keys())

    def _load_existing_output_map(self, output_csv_path, target_column, id_column):
        if not os.path.exists(output_csv_path):
            return {}

        with open(output_csv_path, "r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames or target_column not in reader.fieldnames:
                return {}

            output_map = {}
            for index, row in enumerate(reader):
                row_id = row.get(id_column) if id_column else str(index)
                if row_id is None:
                    continue
                output_map[row_id] = row.get(target_column, "")
            return output_map

    def _get_temp_path(self, file_path, suffix=".part"):
        return f"{file_path}{suffix}"

    def _load_translation_map(self, file_path, id_column, translation_column):
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

    def _write_translation_map(self, file_path, id_column, translation_column, items):
        with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [id_column, translation_column]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_id, translated_text in items.items():
                writer.writerow({id_column: row_id, translation_column: translated_text})

    def _append_translation_map(self, file_path, id_column, translation_column, items):
        current = self._load_translation_map(file_path, id_column, translation_column)
        current.update(items)
        self._write_translation_map(file_path, id_column, translation_column, current)

    def _write_to_csv(self, file_path, row_uuid, translated_text):
        file_has_header = os.path.isfile(file_path) and os.path.getsize(file_path) > 0
        with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["uuid", "translated_text"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if not file_has_header:
                writer.writeheader()

            writer.writerow({"uuid": row_uuid, "translated_text": translated_text})
