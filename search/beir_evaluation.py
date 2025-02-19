# /// script
# dependencies = [
#   "sentence-transformers",
#   "numpy",
#   "datasets",
#   "tqdm",
#   "einops",
#   "beir",
# ]
# ///

import json
import logging
import os

from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search import BaseSearch

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join("./datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")



class EncoderModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs):
        self.model = SentenceTransformer(model_name, **kwargs)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        """Normalize embeddings for cosine similarity = dot product"""
        return self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,  # Critical for dot product = cosine
            convert_to_numpy=True,
            **kwargs
        )

    def encode_corpus(self, corpus: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return self.model.encode(
            corpus,
            batch_size=batch_size,
            normalize_embeddings=True,  # Must match query normalization
            convert_to_numpy=True,
            **kwargs
        )



class RetrievalSystem(BaseSearch):
    def __init__(self, model: EncoderModel, batch_size: int = 128,
                 corpus_chunk_size: int = 50000, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size

    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str],
              top_k: int, score_function: str, **kwargs) -> Dict[str, Dict[str, float]]:
        # Convert corpus to ordered list of texts
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]

        # Encode queries and corpus in batches
        query_ids = list(queries.keys())
        query_emb = self.model.encode_queries([queries[qid] for qid in query_ids], self.batch_size)
        corpus_emb = self.model.encode_corpus(corpus_texts, self.batch_size)

        # Compute scores using dot product (cosine similarity with normalized embeddings)
        scores = query_emb @ corpus_emb.T  # Matrix multiplication for batch dot products

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


encoder = EncoderModel("sentence-transformers/all-MiniLM-L6-v2")
retriever = RetrievalSystem(encoder)
evaluator = EvaluateRetrieval(retriever)

# On BEIR dataset:
results = evaluator.retrieve(corpus, queries)

k_values = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)

results_dir = os.path.join("./results")
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, f"{dataset}.json"), 'w') as f:
  json.dump(dict(ndcg=ndcg, recall=recall, precision=precision), f)

# needs beir from main branch, but there is a bug in the code. "faiss is undefined"
# util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
# util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
