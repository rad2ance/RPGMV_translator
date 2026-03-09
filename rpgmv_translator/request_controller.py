import csv
import json
import os
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from rpgmv_translator.entity.const import (
    DEFAULT_MISSING_MARKER_PREFIX,
    YES_VALUES,
)
from rpgmv_translator.csv_translation_helpers import (
    append_translation_map,
    build_pending_items,
    build_row_lookup,
    collect_marker_items_from_map,
    get_temp_path,
    load_existing_output_map,
    load_translation_map,
    read_csv_rows,
    row_identifier,
    write_translation_map,
)
from rpgmv_translator.translator.gpt_translator import GPTTranslator


class GPTRequestController:
    def __init__(
        self,
        max_tokens,
        language,
        model="gpt-4.1-mini",
        target_language="Chinese",
        custom_prompt="",
        quality_config=None,
    ):
        self.max_tokens = max_tokens
        self.language = language
        self.model = model
        self.target_language = target_language
        self.custom_prompt = custom_prompt or ""
        self.api_key = self._load_api_key_from_config("config.json")
        self.translator = GPTTranslator(self.api_key)
        self.tokenizer = self._select_tokenizer(language)

        quality_config = quality_config or {}
        self.key_coverage_rate_threshold = float(quality_config.get("key_coverage_rate_threshold", 0.9))
        self.missing_key_abs_threshold = int(quality_config.get("missing_key_abs_threshold", 2))
        self.max_on_the_fly_retries = int(quality_config.get("max_on_the_fly_retries", 2))
        self.max_consecutive_bad_calls = int(quality_config.get("max_consecutive_bad_calls", 5))
        self.missing_marker_prefix = quality_config.get("missing_marker_prefix", DEFAULT_MISSING_MARKER_PREFIX)
        self.cost_confirmation_usd = float(quality_config.get("cost_confirmation_usd", 10.0))

        self._reset_call_stats()

    def _reset_call_stats(self):
        self.total_calls = 0
        self.complete_calls = 0
        self.above_threshold_calls = 0
        self.consecutive_bad_calls = 0

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
            return EnglishTokenizer(model_name=self.model)
        if language == "Japanese":
            from rpgmv_translator.tokenizer.japanese_tokenizer import JapaneseTokenizer
            return JapaneseTokenizer(model_name=self.model)
        raise ValueError(f"Unsupported language: {language}")

    def _estimate_tokens_and_price(self, texts):
        return self.translator.estimate_tokens_and_price(texts, self.tokenizer, self.model)

    def process_csv(self, original_csv_path, translated_csv_path):
        start_time = time.time()
        self._reset_call_stats()

        rows, _ = read_csv_rows(original_csv_path, required_columns=["uuid", "text"])

        temp_translated_csv_path = get_temp_path(translated_csv_path)
        source_progress_path = temp_translated_csv_path if os.path.exists(temp_translated_csv_path) else translated_csv_path
        existing_translations = load_translation_map(source_progress_path, "uuid", "translated_text")
        source_has_any = any((row.get("text") or "").strip() for row in rows)
        overwrite_existing = self._confirm_precheck(
            target_has_values=bool(existing_translations),
            source_has_values=source_has_any,
            target_column="translated_text",
        )
        if overwrite_existing:
            existing_translations = {}

        write_translation_map(temp_translated_csv_path, "uuid", "translated_text", existing_translations)
        pending_items = build_pending_items(
            rows=rows,
            source_column="text",
            id_column="uuid",
            existing_translations=existing_translations,
            overwrite_existing=overwrite_existing,
        )

        token_count, estimated_price = self._estimate_tokens_and_price([text for _, text in pending_items])
        print(f"Total tokens to translate: {token_count}")
        print(f"Estimated price for translation: ${estimated_price:.2f}")
        self._confirm_cost_if_needed(estimated_price)

        self._translate_items_in_batches(
            pending_items,
            on_batch_translated=lambda batch: append_translation_map(
                temp_translated_csv_path, "uuid", "translated_text", batch
            ),
        )

        retry_items = collect_marker_items_from_map(
            load_translation_map(temp_translated_csv_path, "uuid", "translated_text"),
            row_lookup=build_row_lookup(rows, "text", "uuid"),
            is_missing_marker_fn=self._is_missing_marker,
        )
        if retry_items:
            self._translate_items_in_batches(
                retry_items,
                on_batch_translated=lambda batch: append_translation_map(
                    temp_translated_csv_path, "uuid", "translated_text", batch
                ),
            )

        os.replace(temp_translated_csv_path, translated_csv_path)
        self._print_run_summary(start_time)

    def process_arbitrary_csv(self, input_csv_path, output_csv_path, source_column, target_column="translated_text", id_column=None):
        start_time = time.time()
        self._reset_call_stats()

        rows, input_fieldnames = read_csv_rows(input_csv_path)
        if source_column not in input_fieldnames:
            raise ValueError(f"Source column '{source_column}' was not found in CSV header: {input_fieldnames}")
        if id_column and id_column not in input_fieldnames:
            raise ValueError(f"ID column '{id_column}' was not found in CSV header: {input_fieldnames}")

        progress_path = get_temp_path(output_csv_path)
        existing_translations = load_existing_output_map(progress_path, target_column, id_column)
        if not existing_translations:
            existing_translations = load_existing_output_map(output_csv_path, target_column, id_column)

        any_target_populated = any((row.get(target_column) or "").strip() for row in rows)
        source_has_any = any((row.get(source_column) or "").strip() for row in rows)
        overwrite_existing = self._confirm_precheck(
            target_has_values=any_target_populated,
            source_has_values=source_has_any,
            target_column=target_column,
        )

        if overwrite_existing:
            existing_translations = {}

        temp_id_column = id_column or "__row_index__"
        write_translation_map(progress_path, temp_id_column, target_column, existing_translations)
        pending_items = build_pending_items(
            rows=rows,
            source_column=source_column,
            id_column=id_column,
            existing_translations=existing_translations,
            overwrite_existing=overwrite_existing,
        )

        token_count, estimated_price = self._estimate_tokens_and_price([text for _, text in pending_items])
        print(f"Total tokens to translate: {token_count}")
        print(f"Estimated price for translation: ${estimated_price:.2f}")
        self._confirm_cost_if_needed(estimated_price)

        new_translations = self._translate_items_in_batches(
            pending_items,
            on_batch_translated=lambda batch: append_translation_map(
                progress_path, temp_id_column, target_column, batch
            ),
        )

        marker_retry_items = collect_marker_items_from_map(
            new_translations,
            row_lookup=build_row_lookup(rows, source_column, id_column),
            is_missing_marker_fn=self._is_missing_marker,
        )
        if marker_retry_items:
            marker_retry_translations = self._translate_items_in_batches(
                marker_retry_items,
                on_batch_translated=lambda batch: append_translation_map(
                    progress_path, temp_id_column, target_column, batch
                ),
            )
            new_translations.update(marker_retry_translations)

        output_fieldnames = list(input_fieldnames)
        if target_column not in output_fieldnames:
            output_fieldnames.append(target_column)

        temp_output_path = get_temp_path(output_csv_path, suffix=".output.tmp")
        with open(temp_output_path, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=output_fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows):
                row_id = row_identifier(row, id_column, index)
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
        self._print_run_summary(start_time)

    def _translate_items_in_batches(self, items, on_batch_translated=None):
        translations = {}
        batch_ids = []
        batch_texts = []
        current_token_count = 0

        for item_id, text in tqdm(items, total=len(items), desc="Translating..."):
            token_count = self.tokenizer.get_token_count(text)

            if token_count > self.max_tokens:
                if batch_texts:
                    batch_map = self._translate_batch_with_quality(batch_ids, batch_texts)
                    translations.update(batch_map)
                    if on_batch_translated and batch_map:
                        on_batch_translated(batch_map)
                    batch_ids = []
                    batch_texts = []
                    current_token_count = 0

                split_texts = self._split_text(text, self.max_tokens)
                split_ids = [f"{item_id}__chunk_{idx}" for idx, _ in enumerate(split_texts)]
                split_map = self._translate_batch_with_quality(split_ids, split_texts)
                ordered_segments = [split_map.get(split_id, self._build_missing_marker(split_id)) for split_id in split_ids]
                joined = "".join(segment for segment in ordered_segments if not self._is_missing_marker(segment))
                translations[item_id] = joined if joined else self._build_missing_marker(item_id)
                if on_batch_translated:
                    on_batch_translated({item_id: translations[item_id]})
                continue

            if current_token_count + token_count > self.max_tokens and batch_texts:
                batch_map = self._translate_batch_with_quality(batch_ids, batch_texts)
                translations.update(batch_map)
                if on_batch_translated and batch_map:
                    on_batch_translated(batch_map)
                batch_ids = []
                batch_texts = []
                current_token_count = 0

            batch_ids.append(item_id)
            batch_texts.append(text)
            current_token_count += token_count

        if batch_texts:
            batch_map = self._translate_batch_with_quality(batch_ids, batch_texts)
            translations.update(batch_map)
            if on_batch_translated and batch_map:
                on_batch_translated(batch_map)

        return translations

    def _translate_batch_with_quality(self, batch_ids, batch_texts):
        unique_texts = set(batch_texts)
        attempts = 0
        final_mapping = {}

        while attempts <= self.max_on_the_fly_retries:
            raw_mapping = self.translator.translate_to_mapping(
                batch_texts,
                model=self.model,
                target_language=self.target_language,
                custom_prompt=self.custom_prompt,
            )
            valid_mapping = {k: v for k, v in raw_mapping.items() if k in unique_texts and isinstance(v, str)}

            coverage, missing_count, should_retry = self._evaluate_mapping_quality(valid_mapping, unique_texts, attempts)

            final_mapping = valid_mapping

            if self.consecutive_bad_calls >= self.max_consecutive_bad_calls:
                raise RuntimeError(
                    f"Early stop: {self.consecutive_bad_calls} consecutive low-quality calls "
                    f"(coverage={coverage:.2%}, missing={missing_count})."
                )

            if not should_retry:
                break

            attempts += 1

        result = {}
        for row_id, original_text in zip(batch_ids, batch_texts):
            translated = final_mapping.get(original_text)
            if translated is None:
                translated = self._build_missing_marker(row_id)
            result[row_id] = translated

        return result

    def _evaluate_mapping_quality(self, valid_mapping, unique_texts, attempts):
        coverage = (len(valid_mapping) / len(unique_texts)) if unique_texts else 1.0
        missing_count = len(unique_texts) - len(valid_mapping)

        self.total_calls += 1
        if missing_count == 0:
            self.complete_calls += 1

        meets_threshold = (
            coverage >= self.key_coverage_rate_threshold
            or missing_count < self.missing_key_abs_threshold
        )
        if meets_threshold:
            self.above_threshold_calls += 1
            self.consecutive_bad_calls = 0
        else:
            self.consecutive_bad_calls += 1

        should_retry = (
            coverage < self.key_coverage_rate_threshold
            and missing_count >= self.missing_key_abs_threshold
            and attempts < self.max_on_the_fly_retries
        )
        return coverage, missing_count, should_retry

    def _split_text(self, text, max_tokens):
        if hasattr(self.tokenizer, "split_text_by_max_tokens"):
            return self.tokenizer.split_text_by_max_tokens(text, max_tokens)

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

    def _confirm_precheck(self, target_has_values, source_has_values, target_column):
        if not source_has_values:
            answer = input("Proceed? source column has no translatable values. Type 'yes' to continue: ").strip().lower()
            if answer not in YES_VALUES:
                raise RuntimeError("Translation cancelled by user.")

        if not target_has_values:
            return False

        answer = input(
            f"Proceed? target column '{target_column}' already has values and will be overwritten. "
            "Type 'yes' to continue: "
        ).strip().lower()
        if answer not in YES_VALUES:
            raise RuntimeError("Translation cancelled by user.")
        return True

    def _confirm_cost_if_needed(self, estimated_price):
        if estimated_price <= self.cost_confirmation_usd:
            return
        answer = input(
            f"Proceed? estimated cost ${estimated_price:.2f} is above ${self.cost_confirmation_usd:.2f}. "
            "Type 'yes' to continue: "
        ).strip().lower()
        if answer not in YES_VALUES:
            raise RuntimeError("Translation cancelled by user.")

    def _build_missing_marker(self, row_id):
        return f"{self.missing_marker_prefix}:{row_id}"

    def _is_missing_marker(self, value):
        return isinstance(value, str) and value.startswith(f"{self.missing_marker_prefix}:")

    def _print_run_summary(self, start_time):
        elapsed = time.time() - start_time
        if self.total_calls == 0:
            print("Call quality summary: no API calls were required.")
            print(f"Time taken: {elapsed:.2f}s")
            return

        complete_rate = self.complete_calls / self.total_calls
        above_threshold_rate = self.above_threshold_calls / self.total_calls
        print(f"Complete-success call rate: {complete_rate:.2%}")
        print(f"Above-threshold call rate: {above_threshold_rate:.2%}")
        print(f"Time taken: {elapsed:.2f}s")
