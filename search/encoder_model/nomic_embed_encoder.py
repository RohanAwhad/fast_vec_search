import numpy as np
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from . import utils
from .base import EncoderModel
from text_preprocessor.base import TextPreprocessor


class NomicEmbedEncoder(EncoderModel):
  def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", matryoshka_dim=None, text_preprocessor: TextPreprocessor | None=None, **kwargs):
    self.model = SentenceTransformer(model_name, **kwargs)
    self.matryoshka_dim = matryoshka_dim  # used for matryoshka embeddings
    self.text_preprocessor = text_preprocessor

  def encode_queries(self, queries: list[str], batch_size: int, **kwargs) -> np.ndarray:
    """Normalize embeddings for cosine similarity = dot product"""
    if self.text_preprocessor: queries = self.text_preprocessor.preprocess_queries(queries)
    embeddings = self.model.encode(
        queries,
        batch_size=batch_size,
        **kwargs
    )
    embeddings = self._apply_matryoshka_transformation(embeddings)
    return embeddings

  def encode_corpus(self, corpus: list[str], batch_size: int, **kwargs) -> np.ndarray:
    if self.text_preprocessor: corpus = self.text_preprocessor.preprocess_corpus(corpus)
    embeddings = self.model.encode(
        corpus,
        batch_size=batch_size,
        **kwargs
    )
    if self.matryoshka_dim is not None: embeddings = utils.apply_matryoshka_transformation(embeddings, self.matryoshka_dim)
    return embeddings
