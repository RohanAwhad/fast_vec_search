import chromadb
from typing import Dict
from .base import BaseSearch
from encoder_model.base import EncoderModel

CHROMADB_BATCH_SIZE = 4096
class ChromaDBSearch(BaseSearch):
    def __init__(self,
                 model: EncoderModel,
                 batch_size: int = 128,
                 matryoshka_dim: int = 256):
        self.model = model
        self.batch_size = batch_size
        self.matryoshka_dim = matryoshka_dim
        self._client = chromadb.Client()  # In-memory client

    def search(self,
            corpus,
            corpus_ids,
            corpus_embeddings,
            query_ids,
            query_embeddings,
            top_k,
            **kwargs) -> Dict[str, Dict[str, float]]:

        # Create fresh collection for this search session
        collection = self._client.create_collection(
            name="search_session",
            metadata={"hnsw:space": "ip"}  # Inner product similarity
        )

        # Add to Chroma
        corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
        for i in range(0, len(corpus_ids), CHROMADB_BATCH_SIZE):
            collection.add(
                ids=corpus_ids[i: i+CHROMADB_BATCH_SIZE],
                documents=corpus_texts[i: i+CHROMADB_BATCH_SIZE],
                embeddings=corpus_embeddings[i:i+CHROMADB_BATCH_SIZE].tolist()
            )

        # Perform search
        results = collection.query(
            query_embeddings=query_embeddings.tolist(),
            n_results=top_k,
            include=["distances"]
        )

        # Format results to BEIR spec
        beir_results = {}
        for q_idx, qid in enumerate(query_ids):
            scores = results["distances"][q_idx]
            doc_ids = results["ids"][q_idx]
            beir_results[qid] = {
                doc_id: -1.0 * float(score)  # beir uses similarity => inv. dist
                for doc_id, score in zip(doc_ids, scores)
            }

        return beir_results

