import json
import os
import time

import openai

from rpgmv_translator.translator.translator_base import AbstractTranslator


class GPTTranslator(AbstractTranslator):
    # USD pricing per 1K tokens (input/output) used for estimation only.
    MODEL_PRICING_PER_1K = {
        "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
        "gpt-4.1": {"input": 0.0020, "output": 0.0080},
    }

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key)

    def translate(self, texts, model="gpt-4.1-mini", split_attempt=False, target_language="Chinese"):
        del split_attempt  # Compatibility with prior interface.
        if not texts:
            return []

        translated_map = self.translate_to_mapping(texts, model=model, target_language=target_language)
        return [translated_map.get(original_text, original_text) for original_text in texts]

    def translate_to_mapping(self, texts, model="gpt-4.1-mini", target_language="Chinese"):
        if not texts:
            return {}

        max_retries = 3
        attempts = 0
        retry_delay = 1
        prompt = self._build_prompt(texts, target_language)

        while attempts < max_retries:
            try:
                request_kwargs = {
                    "messages": [
                        {"role": "system", "content": prompt},
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
                if isinstance(translated_dict, dict):
                    return translated_dict

                print(f"Invalid response format: {response_text}")
                attempts += 1
                time.sleep(retry_delay)
            except openai.RateLimitError as e:
                print(f"Rate limit exceeded, retrying in {retry_delay} seconds: {e}")
                attempts += 1
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
                attempts += 1
                time.sleep(retry_delay)

        return {}

    def estimate_tokens_and_price(self, texts, tokenizer, model):
        input_tokens = sum(tokenizer.get_token_count(text) for text in texts if text)

        pricing = self.MODEL_PRICING_PER_1K.get(model)
        if pricing is None:
            # Fallback to mini pricing for unknown models.
            pricing = self.MODEL_PRICING_PER_1K["gpt-4.1-mini"]

        # Simple heuristic: output tokens are usually close to input tokens for JP->target translation.
        estimated_output_tokens = input_tokens
        estimated_price = (
            (input_tokens / 1000.0) * pricing["input"]
            + (estimated_output_tokens / 1000.0) * pricing["output"]
        )
        return input_tokens, estimated_price

    def _extract_dict_from_response(self, full_response):
        if not full_response:
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
                return None

        return None

    def _build_prompt(self, texts, target_language):
        json_list = json.dumps(texts, ensure_ascii=False)
        return (
            f"Translate Japanese strings to {target_language}. "
            "Return exactly one JSON object where each key is the exact original source string and each value is the translated target-language string. "
            "Do not translate English-only strings. "
            "Do not return any text outside the JSON object. "
            f"Input array: {json_list}"
        )
