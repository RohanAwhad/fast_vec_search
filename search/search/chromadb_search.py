import chromadb
from typing import Dict
from .base import BaseSearch
from encoder_model.base import EncoderModel

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
              corpus: Dict[str, Dict[str, str]],
              queries: Dict[str, str],
              top_k: int,
              score_function: str,
              **kwargs) -> Dict[str, Dict[str, float]]:

        # Create fresh collection for this search session
        collection = self._client.create_collection(
            name="search_session",
            metadata={"hnsw:space": "ip"}  # Inner product similarity
        )

        # Process corpus
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]

        # Encode and truncate corpus embeddings
        corpus_embeddings = self.model.encode_corpus(
            corpus_texts,
            self.batch_size,
            convert_to_tensor=True
        )

        # Add to Chroma
        collection.add(
            ids=corpus_ids,
            documents=corpus_texts,
            embeddings=corpus_embeddings.tolist()
        )

        # Process queries
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        # Encode and truncate query embeddings
        query_embeddings = self.model.encode_queries(
            query_texts,
            self.batch_size,
            convert_to_tensor=True
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

