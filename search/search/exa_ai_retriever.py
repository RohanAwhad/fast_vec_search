import ctypes
import functools
import numpy as np
import os

BYTE_SIZE = 8
SUBVECTOR_SIZE = 8

@functools.lru_cache()
def _get_lib():

  lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libexa_search.so'))
  # Define argument types
  lib.get_binary_matrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
  ]

  lib.quantize.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int
  ]

  lib.instantiate_lookup_table.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
  ]

  lib.compile_scores.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
    ctypes.c_int,
  ]

  return lib

from .base import BaseSearch
from encoder_model.base import EncoderModel

BYTE_SIZE = 8
SUBVECTOR_SIZE = 8

class ExaAISearch(BaseSearch):
  def __init__(self, model: EncoderModel, batch_size: int = 128, corpus_chunk_size: int = 50000, matryoshka_dim: int = 256):
    self.model = model
    self.batch_size = batch_size
    self.corpus_chunk_size = corpus_chunk_size
    self.matryoshka_dim = matryoshka_dim

  def search(self, corpus, corpus_ids, corpus_embeddings, query_ids, query_embeddings, top_k, **kwargs) -> dict[str, dict[str, float]]:
    # setup vec search
    lib = _get_lib()
    matrix_B = np.random.randn(256, BYTE_SIZE).astype(np.float32)
    lib.get_binary_matrix(matrix_B.ravel())

    # Convert corpus to ordered list of texts
    corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
    db = corpus_embeddings

    # index corpus
    DB_SIZE = db.shape[0]
    assert db.shape[1] == self.matryoshka_dim, f'need embedding dimension size == {self.matryoshka_dim}, but got "{db.shape[1]}"'
    compressed = np.zeros((DB_SIZE, self.matryoshka_dim//8), dtype=np.uint8)
    lib.quantize(compressed.ravel(), db[:, :self.matryoshka_dim].ravel(), DB_SIZE)

    # Encode queries and corpus in batches
    query_emb = query_embeddings

    # Compute scores
    assert query_emb.shape[1] == self.matryoshka_dim, f'need embedding dimension size == {self.matryoshka_dim}, but got "{query_emb.shape[1]}"'
    lookup_table = np.zeros((len(query_ids) * self.matryoshka_dim//SUBVECTOR_SIZE, 256), dtype=np.float32)
    lib.instantiate_lookup_table(lookup_table.ravel(), query_emb.astype(np.float32).ravel(), matrix_B.T.ravel(), len(query_ids))

    scores = np.zeros((len(query_emb), DB_SIZE), dtype=np.float32)
    lib.compile_scores(compressed.ravel(), lookup_table, scores.ravel(), len(query_ids), len(corpus_ids))

    # Convert to BEIR format results
    results = {}
    for q_idx, qid in enumerate(query_ids):
        doc_scores = scores[q_idx]
        top_indices = np.argpartition(doc_scores, -top_k)[-top_k:]
        results[qid] = {
            corpus_ids[d_idx]: float(doc_scores[d_idx])
            for d_idx in top_indices
        }
    return results
