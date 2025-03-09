import faiss
faiss.omp_set_num_threads(1)
import numpy as np
from typing import Any, Dict
from rerankers import Reranker

from .base import BaseSearch
from encoder_model.base import EncoderModel
from retriever.exaai import ExaAIIndex, ExaAIDB


# # ===
# # Vector DB
# # ===
# from abc import ABC, abstractmethod
# from pydantic import BaseModel
# from typing import Any, Type
#
# class Document(BaseModel):
#   doc_id: str|int
#   text: str
#   metadata: dict[str, Any] | None = None
#
#
# class Result(BaseModel):
#   document: Document
#   score: float
#
# class RankedResults(BaseModel):
#   results: list[Result]
#   def __getitem__(self, idx): return self.results[idx]
#
#
# class BaseIndex(ABC):
#   @abstractmethod
#   def query(self, query_embeddings: list[list[float]], n_results: int, filters=None) -> list[RankedResults]: pass
#
# class BaseDB(ABC):
#   @abstractmethod
#   def create_collection(self, name: str, metadata: dict[str, Type] | None = None, **kwargs) -> "BaseDB": pass
#
#   @abstractmethod
#   def add(self, ids: list[str|int], documents: list[str], embeddings: list[list[float]], metadata: list[dict[str, Any]] | None = None, **kwargs) -> None: pass
#
#   @abstractmethod
#   def create_index(self) -> BaseIndex: pass
#
#
# # concrete implementation
# from tqdm import tqdm
#
# class ExaAIIndex(BaseIndex):
#   def __init__(self, ids, documents, binary_db, binary_centroids, cluster_members, subvector_size):
#     self.ids = ids
#     self.documents = documents
#     self.binary_db = binary_db
#     self.binary_centroids =binary_centroids
#     self.cluster_members = cluster_members
#     self.subvector_size = subvector_size
#
#   def query(self, query_embeddings: list[list[float]], n_results: int, filters = None) -> list[RankedResults]:
#     ret = []
#     for query_embedding in tqdm(query_embeddings, total=len(query_embeddings)):
#       subvector_scores = self._instantiate_lookup_table(query_embedding)
#       centroid_scores = self._compile_scores(subvector_scores, self.binary_centroids)
#       top_cluster_idx = centroid_scores.argmax()
#       candidate_indices = np.array(self.cluster_members[top_cluster_idx])
#       candidate_binary_db = [self.binary_db[i] for i in candidate_indices]
#       candidate_scores = self._compile_scores(subvector_scores, candidate_binary_db)
#
#       local_top_k = candidate_scores.argsort()[::-1][:n_results]
#       global_top_k = candidate_indices[local_top_k]
#
#       results = RankedResults(results=[
#         Result(document=Document(doc_id=self.ids[idx], text=self.documents[idx]), score=candidate_scores[score_idx])
#         for idx, score_idx in zip(global_top_k, local_top_k)
#       ])
#       ret.append(results)
#     return ret
#
#   def _instantiate_lookup_table(self, query):
#     keys = [f"{i:08b}" for i in range(2**self.subvector_size)]
#     b = np.array([list(map(lambda x: 1 if x=='1' else -1, x)) for x in keys])
#     values = query.reshape(-1, self.subvector_size) @ b.T
#     return values
#
#   def _compile_scores(self, subvector_scores, binary_db):
#     scores = []
#     for y in binary_db:
#       score = sum([subvector_scores[i][int(x, 2)] for i, x in enumerate(y)])
#       scores.append(score)
#     return np.array(scores)
#
# class ExaAIDB(BaseDB):
#   def __init__(
#     self,
#     embedding_size: int,
#     subvector_size: int,
#     n_centroids: int,
#     min_cluster_size: int,
#   ):
#     self.embedding_size = embedding_size
#     self.subvector_size = subvector_size
#     self.n_centroids = n_centroids
#     self.min_cluster_size = min_cluster_size
#
#   def create_collection(self, name: str, metadata: dict[str, Type] | None = None, **kwargs) -> BaseDB:
#     if metadata is not None: raise NotImplementedError('metadata is not yet supported')
#     self.collection_name = name
#     self.corpus, self.corpus_ids, self.corpus_embeddings, self.compressed_embeddings = [], [], [], []
#     self._corpus_ids_set = set()
#     return self
#
#   def add(self, ids: list[str | int], documents: list[str], embeddings: list[list[float]], metadata: list[dict[str, Any]] | None = None, **kwargs) -> None:
#     if metadata is not None: raise NotImplementedError('metadata is not yet supported')
#     assert len(ids) == len(documents) == len(embeddings), \
#       f'need len of ids, docs and embeddings to be equal, but got {len(doc_ids)}, {len(documents)}, and {len(embeddings)}, respectively'
#
#     corpus = {}
#     for i, d, e in zip(ids, documents, embeddings): corpus[i] = {'document': d, 'embedding': e}
#
#     unseen_doc_ids = list(set(ids).difference(self._corpus_ids_set))
#     documents = [corpus[idx]['document'] for idx in unseen_doc_ids]
#     embeddings = [corpus[idx]['embedding'] for idx in unseen_doc_ids]
#     compressed_embeddings = self._compress(embeddings)
#
#     self.corpus_ids.extend(unseen_doc_ids)
#     self.corpus.extend(documents)
#     self.corpus_embeddings.extend(embeddings)
#     self.compressed_embeddings.extend(compressed_embeddings)
#
#   def _compress(self, db):
#     db = np.array(db)
#     _tmp = list(map(lambda x: '0' if x < 0 else '1', db.reshape(-1)))
#     binary_db = []
#     for i in range(0, len(_tmp), self.embedding_size):
#       _ = _tmp[i:i+self.embedding_size]
#       binary_db.append([])
#       for j in range(0, self.embedding_size, self.subvector_size):
#         binary_db[-1].append(''.join(_[j:j+self.subvector_size]))
#     return binary_db
#
#
#   def create_index(self):
#     # Create clusters
#     n_centroids = max(1, min(self.n_centroids, len(self.corpus_ids) // self.min_cluster_size))
#     binary_centroids, cluster_members = self._create_clusters(self.corpus_embeddings, n_centroids)
#     return ExaAIIndex(
#       ids=self.corpus_ids,
#       documents=self.corpus,
#       binary_db=self.compressed_embeddings,
#       binary_centroids=binary_centroids,
#       cluster_members=cluster_members,
#       subvector_size=self.subvector_size,
#     )
#
#   def _create_clusters(self, db, n_centroids, niter=100):
#     db = np.array(db)
#     kmeans = faiss.Kmeans(d=self.embedding_size, k=n_centroids, niter=niter, verbose=True)
#     print('training...')
#     kmeans.train(db.astype(np.float32))
#     print('trained')
#     binary_centroids = self._compress(kmeans.centroids)
#
#     _, cluster_labels = kmeans.index.search(db.astype(np.float32), 1)
#     cluster_labels = cluster_labels.reshape(-1)
#     cluster_members = [[] for _ in range(n_centroids)]
#     for i, label in enumerate(cluster_labels):
#       cluster_members[label].append(i)
#
#     return binary_centroids, cluster_members


# usage:
# exaai_collection = ExaAIDB(embedding_size=256, subvector_size=8, n_centroids=10, min_cluster_size=100_000).create_collection('test')
# exaai_collection.add(ids=[1,2], documents=['hello', 'world'], embeddings = np.random.rand(2, 256).tolist())
# exaai_collection.create_index()
# results = exaai_collection.query(query_embeddings=np.random.rand(1, 256).tolist(), n_results=10)

# ===
# Search
# ===
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


  def search(self,
             corpus,
             corpus_ids,
             corpus_embeddings,
             queries,
             query_ids,
             query_embeddings,
             top_k,
             **kwargs) -> Dict[str, Dict[str, float]]:

    import time
    documents = [corpus[idx]['text'] for idx in corpus_ids]
    exaai_collection = ExaAIDB(embedding_size=256, n_centroids=10, min_cluster_size=100_000).create_collection('test')
    start = time.monotonic()
    exaai_collection.add(ids=corpus_ids, documents=documents, embeddings=corpus_embeddings)
    exaai_index: ExaAIIndex = exaai_collection.create_index()
    end1 = time.monotonic()
    results = exaai_index.query(query_embeddings=query_embeddings, n_results=top_k)
    end2 = time.monotonic()
    print(f"Creating Index took: {end1-start:0.3f} secs")
    print(f"Querying took      : {end2-end1:0.3f} secs")

    ret = {}
    for qid, res in zip(query_ids, results):
      ret[qid] = {x.document.doc_id: x.score for x in res.results}
    return ret
