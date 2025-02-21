from qdrant_client import QdrantClient, models
from typing import Dict
import uuid
from encoder_model.base import EncoderModel
from .base import BaseSearch

class QdrantSearch(BaseSearch):
  def __init__(self,
               model: EncoderModel,
               batch_size: int = 128,
               matryoshka_dim: int = 256):
    self.model = model
    self.batch_size = batch_size
    self.matryoshka_dim = matryoshka_dim
    self._client = QdrantClient(host="localhost", port=6333)

  def search(self,
            corpus,
            corpus_ids,
            corpus_embeddings,
            query_ids,
            query_embeddings,
            top_k,
            **kwargs) -> Dict[str, Dict[str, float]]:

    # Create unique collection name for this session
    collection_name = f"session_{uuid.uuid4().hex}"

    # Create collection with matryoshka dimension
    self._client.recreate_collection(
      collection_name=collection_name,
      vectors_config=models.VectorParams(
        size=self.matryoshka_dim,
        distance=models.Distance.COSINE
      )
    )

    # Upsert documents
    cid2did = {x: i for i, x in enumerate(corpus_ids)}
    for i in range(0, len(corpus_ids), self.batch_size):
      self._client.upsert(
        collection_name=collection_name,
        points=[
          models.PointStruct(
            id=cid2did[doc_id],
            vector=embedding.tolist(),
            payload={"text": corpus[doc_id]['text']}
          )
          for doc_id, embedding in zip(corpus_ids[i:i+self.batch_size], corpus_embeddings[i:i+self.batch_size])
        ]
      )

    # Process queries
    results = {}
    for qid, query_emb in zip(query_ids, query_embeddings):
      # Search Qdrant
      hits = self._client.search(
        collection_name=collection_name,
        query_vector=query_emb,
        limit=top_k
      )

      # Convert to BEIR format with negative distances
      results[qid] = {
        corpus_ids[int(hit.id)]: hit.score
        for hit in hits
      }

    return results

