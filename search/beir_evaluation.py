import argparse
import pickle

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

from search.base import BaseSearch
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


# ===
# Encode
# ===
EMBED_DIM = 256
encoder = NomicEmbedEncoder(model_name=args.model_name, matryoshka_dim=EMBED_DIM, text_preprocessor=NomicEmbedPreprocessor(), trust_remote_code=True, device=args.device)

embed_dir = os.path.join('./embeddings', args.dataset)
os.makedirs(embed_dir, exist_ok=True)
embeddings_fn = os.path.join(embed_dir, f"{args.model_name.replace('/', '_')}.pkl")
if not os.path.exists(embeddings_fn):
  corpus_ids = list(corpus.keys())
  corpus_ids.sort(key=lambda x: len(corpus[x]['text']), reverse=True)  # sorts based on len for batch packing
  corpus_texts = [corpus[cid]['text'] for cid in corpus_ids]
  corpus_embeddings = encoder.encode_corpus(
    corpus_texts,
    args.batch_size,
    convert_to_tensor=True
  )
  # encode queries
  query_ids = list(queries.keys())
  query_texts = [queries[qid] for qid in query_ids]
  query_embeddings = encoder.encode_queries(
    query_texts,
    args.batch_size,
    convert_to_tensor=True
  )
  with open(embeddings_fn, 'wb') as f:
    pickle.dump(dict(
      corpus_ids=corpus_ids,
      corpus_embeddings=corpus_embeddings,
      query_ids=query_ids,
      query_embeddings=query_embeddings,
    ), f)

with open(embeddings_fn, 'rb') as f:
  _ = pickle.load(f)
  corpus_ids = _['corpus_ids']
  corpus_embeddings = _['corpus_embeddings']
  query_ids = _['query_ids']
  query_embeddings = _['query_embeddings']

def get_retriever(retrieval_type, encoder, batch_size) -> BaseSearch:
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

k_values = [1,3,5,10,50,100,1000]
retriever = get_retriever(args.retrieval_type, encoder, args.batch_size)
results = retriever.search(corpus, corpus_ids, corpus_embeddings, query_ids, query_embeddings, top_k=max(k_values))
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, k_values)

# save
results_dir = os.path.join("./results", args.model_name.replace('/', '_'), args.retrieval_type)
os.makedirs(results_dir, exist_ok=True)
fn = os.path.join(results_dir, f"{args.dataset}.json")

with open(fn, 'w') as f:
  json.dump(dict(ndcg=ndcg, recall=recall, precision=precision), f, indent=2)

# needs beir from main branch, but there is a bug in the code. "faiss is undefined"
# util.save_runfile(os.path.join(results_dir, f"{args.dataset}.run.trec"), results)
# util.save_results(os.path.join(results_dir, f"{args.dataset}.json"), ndcg, _map, recall, precision, mrr)
