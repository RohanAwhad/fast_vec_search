from .base import TextPreprocessor

class NomicEmbedPreprocessor(TextPreprocessor):
  def preprocess_queries(self, queries: list[str], **kwargs) -> list[str]:
    return [f'search_query: {text}' for text in queries]

  def preprocess_corpus(self, corpus: list[str], **kwargs) -> list[str]:
    return [f'search_document: {text}' for text in corpus]
