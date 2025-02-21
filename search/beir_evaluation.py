import argparse

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
argparser.add_argument(
  '--dataset',
  type=str,
  choices=[
    'scifact',
    'msmarco',
    'trec-covid',
    'nfcorpus',
    'nq',
    'hotpotqa',
    'fiqa',
    'arguana',
    'webis-touche2020',
    'cqadupstack',
    'quora',
    'dbpedia-entity',
    'scidocs',
    'fever',
    'climate-fever',
  ],
  default='scifact',
  help='dataset to use')
argparser.add_argument('--device', type=str, choices=['mps', 'cpu', 'cuda'], default=None, help='batch size for encoding')
argparser.add_argument('--batch_size', type=int, default=8, help='batch size for encoding')
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
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{args.dataset}.zip"
out_dir = os.path.join("./datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


EMBED_DIM = 256
encoder = NomicEmbedEncoder(model_name=args.model_name, matryoshka_dim=EMBED_DIM, text_preprocessor=NomicEmbedPreprocessor(), trust_remote_code=True, device=args.device)
def get_retriever(retrieval_type, encoder, batch_size):
  retriever_factory = {
    'dense': lambda encoder, batch_size: DRES(encoder, batch_size=batch_size),
    'exa_ai': lambda encoder, batch_size: ExaAISearch(encoder, batch_size=batch_size, matryoshka_dim=EMBED_DIM),
    'py_exa_ai': lambda encoder, batch_size: PyExaAISearch(encoder, batch_size=batch_size, matryoshka_dim=EMBED_DIM),
    'chromadb': lambda encoder, batch_size: ChromaDBSearch(encoder, batch_size=batch_size, matryoshka_dim=EMBED_DIM),
    'qdrant': lambda encoder, batch_size: QdrantSearch(encoder, batch_size=batch_size, matryoshka_dim=EMBED_DIM)
  }

  if retrieval_type not in retriever_factory:
    raise ValueError(f'Invalid retrieval type: {retrieval_type}')
  return retriever_factory[retrieval_type](encoder, batch_size)

retriever = get_retriever(args.retrieval_type, encoder, args.batch_size)

evaluator = EvaluateRetrieval(retriever, score_function="dot") # or "dot" for dot product "cos_sim" for cosine similarity
results = evaluator.retrieve(corpus, queries)

k_values = [1,3,5,10,50,100,1000]
ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)

# save
results_dir = os.path.join("./results", args.model_name.replace('/', '_'), args.retrieval_type)
os.makedirs(results_dir, exist_ok=True)
fn = os.path.join(results_dir, f"{args.dataset}.json")

with open(fn, 'w') as f:
  json.dump(dict(ndcg=ndcg, recall=recall, precision=precision), f, indent=2)

# needs beir from main branch, but there is a bug in the code. "faiss is undefined"
# util.save_runfile(os.path.join(results_dir, f"{args.dataset}.run.trec"), results)
# util.save_results(os.path.join(results_dir, f"{args.dataset}.json"), ndcg, _map, recall, precision, mrr)
