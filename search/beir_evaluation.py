# /// script
# dependencies = [
#   "sentence-transformers",
#   "numpy",
#   "datasets",
#   "tqdm",
#   "einops",
#   "beir",
#   "chromadb",
# ]
# ///

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--retrieval_type', type=str, choices=['exa_ai', 'dense', 'chromadb'], default='dense', help='Type of retrieval to use')
args = argparser.parse_args()

# ===
# Evaluation on BEIR
# ===
import json
import logging
import os

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from encoder_model.nomic_embed_encoder import NomicEmbedEncoder
from text_preprocessor.nomic_embed_preprocessor import NomicEmbedPreprocessor

from search.chromadb_search import ChromaDBSearch
from search.exa_ai_retriever import ExaAISearch

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


EMBED_DIM = 256
model_name = 'tomaarsen/mpnet-base-nli-matryoshka'
encoder = NomicEmbedEncoder(model_name=model_name, matryoshka_dim=EMBED_DIM, text_preprocessor=NomicEmbedPreprocessor(), trust_remote_code=True)

if args.retrieval_type == 'dense':
  model = DRES(encoder, batch_size=8)
  evaluator = EvaluateRetrieval(model, score_function="dot") # or "dot" for dot product "cos_sim" for cosine similarity
  results = evaluator.retrieve(corpus, queries)

elif args.retrieval_type == 'exa_ai':
  retriever = ExaAISearch(encoder, batch_size=8, matryoshka_dim=EMBED_DIM)
  evaluator = EvaluateRetrieval(retriever)
  results = evaluator.retrieve(corpus, queries)
elif args.retrieval_type == 'chromadb':
  retriever = ChromaDBSearch(encoder, batch_size=8, matryoshka_dim=EMBED_DIM)
  evaluator = EvaluateRetrieval(retriever)
  results = evaluator.retrieve(corpus, queries)
else:
  raise ValueError('Invalid retrieval type mentioned')

k_values = [1,3,5,10,50,100,1000]
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)

# save
results_dir = os.path.join("./results", model_name.replace('/', '_'), args.retrieval_type)
os.makedirs(results_dir, exist_ok=True)
fn = os.path.join(results_dir, f"{dataset}.json")

with open(fn, 'w') as f:
  json.dump(dict(ndcg=ndcg, recall=recall, precision=precision), f)

# needs beir from main branch, but there is a bug in the code. "faiss is undefined"
# util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
# util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
