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

# ===
# Load up C lib
# ===
import ctypes
import functools
import numpy as np

EMBED_SIZE = 256
BYTE_SIZE = 8
SUBVECTOR_SIZE = 8

@functools.lru_cache()
def get_lib():
  lib = ctypes.CDLL('./libexa_search.so')

  # Define argument types
  lib.get_binary_matrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
  ]

  lib.quantize.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int
  ]

  lib.get_scores.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.uint8),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
    ctypes.c_int
  ]
  return lib


# ===
# Evaluation on BEIR
# ===
import json
import logging
import os

from typing import Dict

import numpy as np

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search import BaseSearch

from encoder_model.base import EncoderModel

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
# TODO: understand what is a logging handler?


#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join("./datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


class RetrievalSystem(BaseSearch):
  def __init__(self, model: EncoderModel, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
    self.model = model
    self.batch_size = batch_size
    self.corpus_chunk_size = corpus_chunk_size

  def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, score_function: str, **kwargs) -> Dict[str, Dict[str, float]]:
    # setup vec search
    lib = get_lib()
    matrix_B = np.random.randn(256, BYTE_SIZE).astype(np.float32)
    lib.get_binary_matrix(matrix_B.ravel())

    # Convert corpus to ordered list of texts
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
    db = self.model.encode_corpus(corpus_texts, self.batch_size)

    # index corpus
    DB_SIZE = db.shape[0]
    assert db.shape[1] >= EMBED_SIZE, f'need embedding dimension size >= {EMBED_SIZE}, but got "{db.shape[1]}"'
    compressed = np.zeros((DB_SIZE, EMBED_SIZE//8), dtype=np.uint8)
    lib.quantize(compressed.ravel(), db[:, :EMBED_SIZE].ravel(), DB_SIZE)

    # Encode queries and corpus in batches
    query_ids = list(queries.keys())
    query_emb = self.model.encode_queries([queries[qid] for qid in query_ids], self.batch_size)

    # Compute scores
    assert query_emb.shape[1] == EMBED_SIZE, f'need embedding dimension size == {EMBED_SIZE}, but got "{query_emb.shape[1]}"'
    scores = np.zeros((len(query_emb), DB_SIZE), dtype=np.float32)
    lib.get_scores(
      query_emb.astype(np.float32).ravel(),
      compressed.ravel(),
      matrix_B.T.ravel(),
      scores.ravel(),
      len(query_emb),
      DB_SIZE,
    )

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


from text_preprocessor.nomic_embed_preprocessor import NomicEmbedPreprocessor
from encoder_model.nomic_embed_encoder import NomicEmbedEncoder

model_name = 'nomic-ai/nomic-embed-text-v1.5'
encoder = NomicEmbedEncoder(model_name=model_name, embed_dim=EMBED_SIZE, text_preprocessor=NomicEmbedPreprocessor(), trust_remote_code=True)
retriever = RetrievalSystem(encoder, batch_size=8)
evaluator = EvaluateRetrieval(retriever)

# On BEIR dataset:
results = evaluator.retrieve(corpus, queries)

k_values = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)

results_dir = os.path.join("./results", model_name.replace('/', '_'))
os.makedirs(results_dir, exist_ok=True)

with open(os.path.join(results_dir, f"{dataset}.json"), 'w') as f:
  json.dump(dict(ndcg=ndcg, recall=recall, precision=precision), f)

# needs beir from main branch, but there is a bug in the code. "faiss is undefined"
# util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
# util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
