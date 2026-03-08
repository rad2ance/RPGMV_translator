import json
import os
import time

import openai

from rpgmv_translator.translator.translator_base import AbstractTranslator
from rpgmv_translator.utils import contains_japanese_strict


class GPTTranslator(AbstractTranslator):
    # USD pricing per 1K tokens (input/output) used for estimation only.
    MODEL_PRICING_PER_1K = {
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "gpt-4.1": {"input": 0.0020, "output": 0.0080},
    }

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.prompt = None
        self.enhanced_prompt = False

    def translate(self, texts, model="gpt-4.1-mini", split_attempt=False):
        if not texts:
            return []

        max_retries = 6 if not split_attempt else 2
        attempts = 0
        retry_delay = 1
        self.enhanced_prompt = False
        self.prompt = self._build_prompt(texts)

        while attempts < max_retries:
            try:
                request_kwargs = {
                    "messages": [
                        {"role": "system", "content": self.prompt},
                        {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
                    ],
                    "model": model,
                    "response_format": {"type": "json_object"},
                }
                try:
                    response = self.client.chat.completions.create(**request_kwargs)
                except TypeError:
                    # Backward compatibility for older SDK signatures that do not accept response_format.
                    request_kwargs.pop("response_format", None)
                    response = self.client.chat.completions.create(**request_kwargs)

                response_text = response.choices[0].message.content
                translated_dict = self._extract_dict_from_response(response_text)

                if self._is_valid_response(texts, translated_dict):
                    return [translated_dict.get(original_text, original_text) for original_text in texts]

                print(f"Invalid response or format: {response_text}.")
                attempts += 1
                if attempts > 2 and model != "gpt-4.1":
                    model = "gpt-4.1"
                    print("Switching to gpt-4.1 model.")
                elif attempts > 4 and not split_attempt:
                    midpoint = len(texts) // 2
                    first_half = self.translate(texts[:midpoint], model, split_attempt=True)
                    second_half = self.translate(texts[midpoint:], model, split_attempt=True)
                    return first_half + second_half
                time.sleep(retry_delay)
            except openai.RateLimitError as e:
                print(f"Rate limit exceeded, retrying in {retry_delay} seconds: {e}")
                time.sleep(retry_delay)
                retry_delay *= 2
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.AuthenticationError,
                openai.APIError,
                openai.BadRequestError,
            ) as e:
                print(f"API error: {e}")
                break

        raise Exception("Failed to get valid translation after retries.")

    def estimate_tokens_and_price(self, texts, tokenizer, model):
        input_tokens = sum(tokenizer.get_token_count(text) for text in texts if text)

        pricing = self.MODEL_PRICING_PER_1K.get(model)
        if pricing is None:
            # Fallback to mini pricing for unknown models.
            pricing = self.MODEL_PRICING_PER_1K["gpt-4.1-mini"]

        # Simple heuristic: output tokens are usually close to input tokens for JP->ZH translation.
        estimated_output_tokens = input_tokens
        estimated_price = (
            (input_tokens / 1000.0) * pricing["input"]
            + (estimated_output_tokens / 1000.0) * pricing["output"]
        )
        return input_tokens, estimated_price

    def _extract_dict_from_response(self, full_response):
        if not full_response:
            print("Failed to parse response: empty content.")
            return None

        full_response = full_response.strip()
        try:
            parsed = json.loads(full_response)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start_index = full_response.find("{")
        end_index = full_response.rfind("}")

        if start_index != -1 and end_index != -1:
            extracted_content = full_response[start_index : end_index + 1].strip()
            try:
                return json.loads(extracted_content)
            except json.JSONDecodeError:
                print("Failed to parse the extracted content as JSON.")
                return None

        print("Failed to find dictionary-like content in the response.")
        return None

    def _build_prompt(self, texts):
        json_list = json.dumps(texts, ensure_ascii=False)
        return (
            "Translate Japanese strings to Chinese. "
            "Return exactly one JSON object where each key is the exact original source string and each value is the translated Chinese string. "
            "Do not translate English-only strings. Do not return any text outside the JSON object. "
            f"Input array: {json_list}"
        )

    def _build_enhanced_prompt(self, texts):
        json_list = json.dumps(texts, ensure_ascii=False)
        return (
            "Translate Japanese strings to Chinese. "
            "Return exactly one JSON object where each key is the exact original source string and each value is the translated Chinese string. "
            "Do not translate English-only strings. Ensure Chinese output values do not contain Japanese kana when translation is expected. "
            "Do not return any text outside the JSON object. "
            f"Input array: {json_list}"
        )

    def _is_valid_response(self, original_texts, translated_texts):
        if not isinstance(translated_texts, dict):
            print("Invalid response: The response is not a dictionary.")
            return False

        unique_original_texts = set(original_texts)
        if len(translated_texts) < len(unique_original_texts) * 0.9:
            print(
                "Invalid response: Insufficient length. "
                f"Expected at least {len(unique_original_texts) * 0.9}, got {len(translated_texts)}."
            )
            return False

        if not all(key in unique_original_texts for key in translated_texts.keys()):
            print("Invalid response: Some keys in the translated text are not found in the original text.")
            return False

        japanese_count = sum(contains_japanese_strict(text) for text in translated_texts.values())
        if japanese_count > len(translated_texts) * 0.2:
            print("Invalid response: More than 20% of translated texts contain Japanese.")
            if not self.enhanced_prompt:
                self.prompt = self._build_enhanced_prompt(original_texts)
                self.enhanced_prompt = True
            return False

        return True
