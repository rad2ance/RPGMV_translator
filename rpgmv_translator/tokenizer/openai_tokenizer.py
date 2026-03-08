from rpgmv_translator.tokenizer.tokenizer_base import AbstractTokenizer

import tiktoken


class OpenAITokenizer(AbstractTokenizer):
    def __init__(self, model_name="gpt-4.1-mini"):
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")

    def tokenize(self, text):
        return self.encoding.encode(text or "")

    def get_token_count(self, text):
        return len(self.tokenize(text))

    def split_text_by_max_tokens(self, text, max_tokens):
        token_ids = self.tokenize(text)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if not token_ids:
            return []

        segments = []
        for i in range(0, len(token_ids), max_tokens):
            chunk = token_ids[i : i + max_tokens]
            segments.append(self.encoding.decode(chunk))
        return segments
