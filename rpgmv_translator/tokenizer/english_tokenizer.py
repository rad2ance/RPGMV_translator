from rpgmv_translator.tokenizer.openai_tokenizer import OpenAITokenizer


class EnglishTokenizer(OpenAITokenizer):
    def __init__(self, model_name="gpt-4.1-mini"):
        super().__init__(model_name=model_name)
