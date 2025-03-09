import faiss
import numpy as np
from typing import Any, Type

from .base import BaseIndex, BaseDB
from .models import Document, Result, RankedResults

# ===
# C Lib
# ===
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

# ===
# Vec DB
# ===

class ExaAIIndex(BaseIndex):
  def __init__(self, ids, documents, binary_db, binary_centroids, cluster_members):
    self.ids = ids
    self.documents = documents
    self.binary_db = binary_db
    self.binary_centroids = binary_centroids
    self.cluster_members = cluster_members
    self.subvector_size = SUBVECTOR_SIZE
    self.matrix_B = np.empty((256, BYTE_SIZE), dtype=np.float32)

    self.lib = _get_lib()
    self.lib.get_binary_matrix(self.matrix_B.ravel())

  def query(self, query_embeddings: list[list[float]], n_results: int, filters = None) -> list[RankedResults]:

    ret = []
    BATCH_SIZE = 128
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    embd_dim = query_embeddings.shape[1]

    for i in range(0, len(query_embeddings), BATCH_SIZE):
      # Encode queries and corpus in batches
      query_emb = query_embeddings[i:i+BATCH_SIZE]

      # Compute scores
      subvector_scores = np.zeros((len(query_emb), embd_dim//self.subvector_size, embd_dim), dtype=np.float32)
      self.lib.instantiate_lookup_table(subvector_scores.ravel(), query_emb.astype(np.float32).ravel(), self.matrix_B.T.ravel(), len(query_emb))

      centroid_scores = np.zeros((len(query_emb), len(self.binary_centroids)), dtype=np.float32)
      self.lib.compile_scores(self.binary_centroids.ravel(), subvector_scores, centroid_scores.ravel(), len(query_emb), len(self.binary_centroids))


      for i, cs in enumerate(centroid_scores):
        top_cluster_idx = cs.argmax()
        candidate_indices = np.array(self.cluster_members[top_cluster_idx])
        candidate_binary_db = np.array([self.binary_db[i] for i in candidate_indices], dtype=np.uint8)

        candidate_scores = np.zeros((1, len(candidate_binary_db)), dtype=np.float32)
        self.lib.compile_scores(candidate_binary_db.ravel(), subvector_scores[i].ravel(), candidate_scores.ravel(), 1, len(candidate_binary_db))
        local_top_k = candidate_scores[0].argsort()[::-1][:n_results]
        global_top_k = candidate_indices[local_top_k]

        results = RankedResults(results=[
          Result(document=Document(doc_id=self.ids[idx], text=self.documents[idx]), score=candidate_scores[0][score_idx])
          for idx, score_idx in zip(global_top_k, local_top_k)
        ])
        ret.append(results)


    return ret


class ExaAIDB(BaseDB):
  def __init__(
    self,
    embedding_size: int,
    n_centroids: int,
    min_cluster_size: int,
  ):
    self.embedding_size = embedding_size
    self.n_centroids = n_centroids
    self.min_cluster_size = min_cluster_size

  def create_collection(self, name: str, metadata: dict[str, Type] | None = None, **kwargs) -> BaseDB:
    if metadata is not None: raise NotImplementedError('metadata is not yet supported')
    self.collection_name = name
    self.corpus, self.corpus_ids, self.corpus_embeddings, self.compressed_embeddings = [], [], [], None
    self._corpus_ids_set = set()
    return self

  def add(self, ids: list[str | int], documents: list[str], embeddings: list[list[float]], metadata: list[dict[str, Any]] | None = None, **kwargs) -> None:
    if metadata is not None: raise NotImplementedError('metadata is not yet supported')
    assert len(ids) == len(documents) == len(embeddings), \
      f'need len of ids, docs and embeddings to be equal, but got {len(doc_ids)}, {len(documents)}, and {len(embeddings)}, respectively'

    corpus = {}
    for i, d, e in zip(ids, documents, embeddings): corpus[i] = {'document': d, 'embedding': e}

    unseen_doc_ids = list(set(ids).difference(self._corpus_ids_set))
    documents = [corpus[idx]['document'] for idx in unseen_doc_ids]
    embeddings = [corpus[idx]['embedding'] for idx in unseen_doc_ids]
    compressed_embeddings = self._compress(embeddings)

    self.corpus_ids.extend(unseen_doc_ids)
    self.corpus.extend(documents)
    self.corpus_embeddings.extend(embeddings)
    if self.compressed_embeddings is None:
      self.compressed_embeddings = compressed_embeddings
    else:
      self.compressed_embeddings = np.concat([self.compressed_embeddings, compressed_embeddings], dtype=np.uint8)

  def _compress(self, db):
    lib = _get_lib()

    # compress
    db = np.array(db, dtype=np.float32)
    DB_SIZE = db.shape[0]
    assert db.shape[1] == self.embedding_size, f'need embedding dimension size == {self.embedding_size}, but got "{db.shape[1]}"'
    compressed = np.zeros((DB_SIZE, self.embedding_size//8), dtype=np.uint8)
    lib.quantize(compressed.ravel(), db[:, :self.embedding_size].ravel(), DB_SIZE)
    return compressed

  def create_index(self):

    # Create clusters
    n_centroids = max(1, min(self.n_centroids, len(self.corpus_ids) // self.min_cluster_size))
    binary_centroids, cluster_members = self._create_clusters(self.corpus_embeddings, n_centroids)
    return ExaAIIndex(
      ids=self.corpus_ids,
      documents=self.corpus,
      binary_db=self.compressed_embeddings,
      binary_centroids=binary_centroids,
      cluster_members=cluster_members,
    )

  def _create_clusters(self, db, n_centroids, niter=100):
    db = np.array(db)
    kmeans = faiss.Kmeans(d=self.embedding_size, k=n_centroids, niter=niter, verbose=True)
    print('training...')
    kmeans.train(db.astype(np.float32))
    print('trained')
    binary_centroids = self._compress(kmeans.centroids)

    _, cluster_labels = kmeans.index.search(db.astype(np.float32), 1)
    cluster_labels = cluster_labels.reshape(-1)
    cluster_members = [[] for _ in range(n_centroids)]
    for i, label in enumerate(cluster_labels):
      cluster_members[label].append(i)

    return binary_centroids, cluster_members
