import faiss
import numpy as np
from typing import Dict

from .base import BaseSearch
from encoder_model.base import EncoderModel

class PyExaAISearch(BaseSearch):
  def __init__(self,
               model: EncoderModel,
               batch_size: int = 128,
               matryoshka_dim: int = 256,
               embed_size: int = 256,
               subvector_size: int = 8,
               n_centroids: int = 10):
    self.model = model
    self.batch_size = batch_size
    self.matryoshka_dim = matryoshka_dim
    self.embed_size = embed_size
    self.subvector_size = subvector_size
    self.n_centroids = n_centroids
    self.binary_db = None
    self.centroids = None
    self.binary_centroids = None
    self.cluster_members = None
    self.cluster_labels = None
    self._min_cluster_size = 100_000

  def compress(self, db):
    _tmp = list(map(lambda x: '0' if x < 0 else '1', db.reshape(-1)))
    binary_db = []
    for i in range(0, len(_tmp), self.embed_size):
      _ = _tmp[i:i+self.embed_size]
      binary_db.append([])
      for j in range(0, self.embed_size, self.subvector_size):
        binary_db[-1].append(''.join(_[j:j+self.subvector_size]))
    return binary_db

  def instantiate_lookup_table(self, query):
    keys = [f"{i:08b}" for i in range(2**self.subvector_size)]
    b = np.array([list(map(lambda x: 1 if x=='1' else -1, x)) for x in keys])
    values = query.reshape(-1, self.subvector_size) @ b.T
    return values

  def compile_scores(self, subvector_scores, binary_db):
    scores = []
    for y in binary_db:
      score = sum([subvector_scores[i][int(x, 2)] for i, x in enumerate(y)])
      scores.append(score)
    return np.array(scores)

  def create_clusters(self, db):
    kmeans = faiss.Kmeans(d=self.matryoshka_dim, k=self.n_centroids, niter=100, verbose=True)
    print('training...')
    kmeans.train(db.astype(np.float32))
    print('trained')
    binary_centroids = self.compress(kmeans.centroids)

    _, cluster_labels = kmeans.index.search(db.astype(np.float32), 1)
    cluster_labels = cluster_labels.reshape(-1)
    cluster_members = [[] for _ in range(self.n_centroids)]
    for i, label in enumerate(cluster_labels):
      cluster_members[label].append(i)

    return binary_centroids, cluster_members

  def search(self,
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str],
            top_k: int,
            score_function: str,
            **kwargs) -> Dict[str, Dict[str, float]]:

    # Process and encode corpus
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
    corpus_embeddings = self.model.encode_corpus(corpus_texts, self.batch_size, convert_to_tensor=True)
    self.binary_db = self.compress(corpus_embeddings)

    # Create clusters
    self.n_centroids = min(self.n_centroids, len(corpus_ids) // self._min_cluster_size)
    self.binary_centroids, self.cluster_members = self.create_clusters(corpus_embeddings)

    # Process queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = self.model.encode_queries(query_texts, self.batch_size, convert_to_tensor=True)

    results = {}
    for qid, query_embedding in zip(query_ids, query_embeddings):
      subvector_scores = self.instantiate_lookup_table(query_embedding)
      centroid_scores = self.compile_scores(subvector_scores, self.binary_centroids)
      top_cluster_idx = centroid_scores.argmax()
      candidate_indices = self.cluster_members[top_cluster_idx]
      candidate_binary_db = [self.binary_db[i] for i in candidate_indices]
      candidate_scores = self.compile_scores(subvector_scores, candidate_binary_db)

      local_top_k = candidate_scores.argsort()[::-1][:top_k]
      global_top_k = candidate_indices[local_top_k]

      results[qid] = {
        corpus_ids[idx]: float(candidate_scores[local_idx])
        for local_idx, idx in zip(local_top_k, global_top_k)
      }

    return results
