# /// script
# dependencies = [
#   "sentence-transformers",
#   "datasets",
#   "tqdm",
#   "einops",
# ]
# ///

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


class VectorSearch:
  def __init__(self, db, ids):
    assert len(ids) == len(db), 'expected len of ids and db to be equal'
    self.DB_SIZE = db.shape[0]
    assert db.shape[1] >= EMBED_SIZE, f'need embedding dimension size >= {EMBED_SIZE}, but got "{db.shape[1]}"'

    self.lib = get_lib()
    self.matrix_B = np.random.randn(256, BYTE_SIZE).astype(np.float32)
    self.lib.get_binary_matrix(self.matrix_B.ravel())

    self.compressed = np.zeros((self.DB_SIZE, EMBED_SIZE//8), dtype=np.uint8)
    self.lib.quantize(self.compressed.ravel(), db[:, :EMBED_SIZE].ravel(), self.DB_SIZE)
    self.ids = ids

  def query(self, x, k=10):
    "query based on given query embeddings"
    assert x.shape[1] >= EMBED_SIZE, f'need embedding dimension size >= {EMBED_SIZE}, but got "{x.shape[1]}"'
    scores = np.zeros(self.DB_SIZE, dtype=np.float32)
    self.lib.get_scores(
      x[:, :EMBED_SIZE].astype(np.float32).ravel(),
      self.compressed.ravel(),
      self.matrix_B.T.ravel(),
      scores.ravel(),
      len(x),
      self.DB_SIZE,
    )

    top_k = list(scores.argsort()[::-1][:k])
    ret = []
    for i, x in enumerate(top_k):
      ret.append({
        'id': self.ids[x],
        'rank': i,
        'score': scores[x]
      })
    return ret

  def append(self, x):
    "add new vectors for retrieval"
    raise NotImplementedError



if __name__ == '__main__':
  import hashlib
  import os
  import pickle
  import random
  from tqdm import tqdm

  import torch
  import torch.nn.functional as F
  from sentence_transformers import SentenceTransformer, CrossEncoder
  from datasets import load_dataset

  # Load the model without truncation (full 768 dimensions)
  model = SentenceTransformer("tomaarsen/mpnet-base-nli-matryoshka", trust_remote_code=True)
  cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
  dataset = load_dataset('squad', split='validation')
  hashed_context = [hashlib.md5(x['context'].encode('utf-8')).hexdigest() for x in dataset]
  hash2idx = {x: i for i, x in enumerate(hashed_context)}

  @torch.no_grad()
  def embed(text, is_query=False):
    text = f'search_query: {text}' if is_query else f'search_document: {text}'
    embeddings = model.encode([text], convert_to_tensor=True)
    embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
    embeddings = embeddings[:, :EMBED_SIZE]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

  if not os.path.exists('squad_embeddings.pkl'):
    hash2embeddings = {}
    for i, x in tqdm(enumerate(hashed_context), total=len(hashed_context)):
      if x not in hash2embeddings:
        hash2embeddings[x] = embed(dataset[i]['context']).reshape(1, -1)

    ids, embeddings = [], []
    for h, e in  hash2embeddings.items():
      ids.append(h)
      embeddings.append(e)
    db = np.vstack(embeddings).astype(np.float32)

    # Save the database and queries
    with open('squad_embeddings.pkl', 'wb') as f:
      pickle.dump({
        'ids': ids,
        'db': db,
      }, f)

  # Load saved embeddings
  with open('squad_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    ids = data['ids']
    db = data['db'].astype(np.float32)


  vs = VectorSearch(db, ids)
  print(f'Number of vectors:', vs.DB_SIZE)

  def rerank(question, docs, k):
    candidates = [(question, d) for d in docs]
    rerank_scores = cross_encoder.predict(candidates)
    top_indices = np.argpartition(np.array(rerank_scores), -k)[-k:]
    top_indices = top_indices[np.argsort(np.array(rerank_scores)[top_indices])[::-1]]
    return [docs[idx] for idx in top_indices]


  # Test multiple random queries
  for _ in range(5):
    idx = random.randint(0, db.shape[0])
    query = embed(dataset[idx]['question'], is_query=True).reshape(1, -1).astype(np.float32)
    context_hash = hashlib.md5(dataset[idx]['context'].encode('utf-8')).hexdigest()
    
    print("\nQuestion:", dataset[idx]['question'])
    print("Expected context:", dataset[idx]['context'][:100], "...")
    
    ans = vs.query(query, k=50)
    docs = [dataset[hash2idx[x['id']]]['context'] for x in ans]
    reranked_docs = rerank(dataset[idx]['question'], docs, k=10)

    print("\nTop 10 retrieved contexts:")
    for i, x in enumerate(reranked_docs):
      print(f"\n - {i+1}.", x[:100], "...")

    idx_s = (query @ db.T).reshape(-1).argsort()[::-1][:3]
    print('\nTop 3 Exact Search retrieved contexts:')
    for i, x in enumerate(idx_s):
      print(f"\n{i+1}.", dataset[hash2idx[ids[x]]]['context'][:100], "...")

  # Evaluate retrieval accuracy
  total = 100  # Test on subset
  K = [1, 3, 5, 10, 50, 100]
  def evaluate_accuracy(vs, dataset, hash2idx):
    hits = {k: 0 for k in K}
    test_indices = random.sample(range(len(dataset)), total)

    for idx in tqdm(test_indices):
      query = embed(dataset[idx]['question'], is_query=True).reshape(1, -1).astype(np.float32)
      context_hash = hashlib.md5(dataset[idx]['context'].encode('utf-8')).hexdigest()
      
      ans = vs.query(query, k=max(K))
      docs = [dataset[hash2idx[x['id']]]['context'] for x in ans]
      reranked_docs = rerank(dataset[idx]['question'], docs, k=max(K))
      reranked_hashes = [hashlib.md5(x.encode('utf-8')).hexdigest() for x in reranked_docs]
      
      for k in K:
        if context_hash in reranked_hashes[:k]:
          hits[k] += 1

    print("\nRetrieval Accuracy:")
    for k in K:
      print(f"Top-{k}: {hits[k]/total*100:.2f}%")

  evaluate_accuracy(vs, dataset, hash2idx)
