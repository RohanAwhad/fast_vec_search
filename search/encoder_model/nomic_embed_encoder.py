import numpy as np
import torch
import torch.nn.functional as F

from sentence_transformers import SentenceTransformer

from .base import EncoderModel
from text_preprocessor.base import TextPreprocessor


class NomicEmbedEncoder(EncoderModel):
  def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", embed_dim=None, text_preprocessor: TextPreprocessor | None=None, **kwargs):
    self.model = SentenceTransformer(model_name, **kwargs)
    self.embed_dim = embed_dim  # used for matryoshka embeddings
    self.text_preprocessor = text_preprocessor

  def encode_queries(self, queries: list[str], batch_size: int, **kwargs) -> np.ndarray:
    """Normalize embeddings for cosine similarity = dot product"""
    if self.text_preprocessor: queries = self.text_preprocessor.preprocess_queries(queries)
    embeddings = self.model.encode(
        queries,
        batch_size=batch_size,
        convert_to_tensor=True,
        **kwargs
    )
    embeddings = self._apply_matryoshka_transformation(embeddings)
    return embeddings

  def encode_corpus(self, corpus: list[str], batch_size: int, **kwargs) -> np.ndarray:
    if self.text_preprocessor: corpus = self.text_preprocessor.preprocess_corpus(corpus)
    embeddings = self.model.encode(
        corpus,
        batch_size=batch_size,
        convert_to_tensor=True,
        **kwargs
    )
    embeddings = self._apply_matryoshka_transformation(embeddings)
    return embeddings

  def _apply_matryoshka_transformation(self, embeddings: torch.Tensor) -> np.ndarray:
    if self.embed_dim is not None:
      embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
      embeddings = embeddings[:, :self.embed_dim]
      embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()
