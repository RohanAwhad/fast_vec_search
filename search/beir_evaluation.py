import argparse

from torch import narrow_copy

argparser = argparse.ArgumentParser()
argparser.add_argument('--retrieval_type', type=str, choices=['exa_ai', 'py_exa_ai', 'dense', 'chromadb', 'qdrant'], default='py_exa_ai', help='Type of retrieval to use')
argparser.add_argument(
  '--model_name',
  type=str,
  choices=[
    'tomaarsen/mpnet-base-nli-matryoshka',
    'nomic-ai/nomic-embed-text-v1.5',
  ],
  default='tomaarsen/mpnet-base-nli-matryoshka',
  help='model to use')
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
from search.qdrant_search import QdrantSearch
from search.py_exaai_search import PyExaAISearch

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
encoder = NomicEmbedEncoder(model_name=args.model_name, matryoshka_dim=EMBED_DIM, text_preprocessor=NomicEmbedPreprocessor(), trust_remote_code=True)

if args.retrieval_type == 'dense': retriever = DRES(encoder, batch_size=8)
elif args.retrieval_type == 'exa_ai': retriever = ExaAISearch(encoder, batch_size=8, matryoshka_dim=EMBED_DIM)
elif args.retrieval_type == 'py_exa_ai': retriever = PyExaAISearch(encoder, batch_size=8, matryoshka_dim=EMBED_DIM)
elif args.retrieval_type == 'chromadb': retriever = ChromaDBSearch(encoder, batch_size=8, matryoshka_dim=EMBED_DIM)
elif args.retrieval_type == 'qdrant': retriever = QdrantSearch(encoder, batch_size=8, matryoshka_dim=EMBED_DIM)
else: raise ValueError('Invalid retrieval type mentioned')


evaluator = EvaluateRetrieval(retriever, score_function="dot") # or "dot" for dot product "cos_sim" for cosine similarity
results = evaluator.retrieve(corpus, queries)

k_values = [1,3,5,10,50,100,1000]
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)

# save
results_dir = os.path.join("./results", args.model_name.replace('/', '_'), args.retrieval_type)
os.makedirs(results_dir, exist_ok=True)
fn = os.path.join(results_dir, f"{dataset}.json")

with open(fn, 'w') as f:
  json.dump(dict(ndcg=ndcg, recall=recall, precision=precision), f)

# needs beir from main branch, but there is a bug in the code. "faiss is undefined"
# util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
# util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)
