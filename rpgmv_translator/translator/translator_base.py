# abstract_translator.py

from abc import ABC, abstractmethod

class AbstractTranslator(ABC):

    @abstractmethod
    def translate(self, texts):
        """
        Translate a list of texts from the source language to the target language.
        """
        pass

    @abstractmethod
    def estimate_tokens_and_price(self, texts, tokenizer, model):
        """
        Estimate input tokens and translation price for this translator/model.
        """
        pass
