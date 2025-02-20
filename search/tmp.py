# efficient matmul between float and binary arry
import numpy as np

ORIGINAL_EMBED_SIZE = 4096
EMBED_SIZE = 256
SUBVECTOR_SIZE = 8
DB_SIZE = 10_000

def compress(db):
  _tmp = list(map(lambda x: '0' if x < 0 else '1', db.reshape(-1)))
  binary_db = []
  for i in range(0, len(_tmp), EMBED_SIZE):
    _ = _tmp[i:i+EMBED_SIZE]
    binary_db.append([])
    for j in range(0, EMBED_SIZE, SUBVECTOR_SIZE):
      binary_db[-1].append(''.join(_[j:j+SUBVECTOR_SIZE]))
  return binary_db

def instantiate_lookup_table(query):
  keys = [f"{i:08b}" for i in range(2**SUBVECTOR_SIZE)]
  b = np.array([list(map(lambda x: 1 if x=='1' else -1, x)) for x in keys])
  values = query.reshape(-1, SUBVECTOR_SIZE) @ b.T
  return values

def compile_scores(subvector_scores, binary_db):
  scores = []
  for y in binary_db:
    score = sum([subvector_scores[i][int(x, 2)] for i, x in enumerate(y)])
    scores.append(score)
  return np.array(scores)

def preprocess(db):
  return compress(db[:, :EMBED_SIZE])

def main(query):
  subvector_scores = instantiate_lookup_table(query[:EMBED_SIZE])
  scores = compile_scores(subvector_scores, binary_db)
  exact_scores = (query.reshape(1, ORIGINAL_EMBED_SIZE) @ db.T).reshape(-1)

  binary_top_10 = scores.argsort()[::-1][:10]
  exact_top_10 = exact_scores.argsort()[::-1][:10]

  binary_top_100 = scores.argsort()[::-1][:100]
  exact_top_100 = exact_scores.argsort()[::-1][:100]

  print('\n\nBinary dot product ranks:', binary_top_10)
  print(' Exact dot product  ranks:', exact_top_10)
  print('Common in top 100: ', len(set(binary_top_100).intersection(exact_top_100)))



import faiss

def create_clusters():
  ncentroids = 100
  kmeans = faiss.Kmeans(d=EMBED_SIZE, k=ncentroids, niter=100, verbose=False)
  kmeans.train(db[:, :EMBED_SIZE].astype(np.float32))
  centroids = kmeans.centroids
  _, cluster_labels = kmeans.index.search(db[:, :EMBED_SIZE].astype(np.float32), 1)
  cluster_labels = cluster_labels.reshape(-1)
  
# create lists of indices for each cluster
  cluster_members = [[] for _ in range(ncentroids)]
  for i, label in enumerate(cluster_labels):
    cluster_members[label].append(i)
    
  binary_centroids = preprocess(centroids)
  return centroids, binary_centroids, cluster_members, cluster_labels

def search_with_clusters(query, binary_centroids, cluster_members, top_k=10):
  subvector_scores = instantiate_lookup_table(query[:EMBED_SIZE])
  centroid_scores = compile_scores(subvector_scores, binary_centroids)
  
  top_2_clusters = centroid_scores.argsort()[::-1][:2]
  candidate_indices = np.concatenate([cluster_members[i] for i in top_2_clusters])
  candidate_binary_db = [binary_db[i] for i in candidate_indices]
  candidate_scores = compile_scores(subvector_scores, candidate_binary_db)
  
  local_top_k = candidate_scores.argsort()[::-1][:top_k]
  global_top_k = candidate_indices[local_top_k]
  
  exact_scores = (query.reshape(1, ORIGINAL_EMBED_SIZE) @ db.T).reshape(-1)
  exact_top_k = exact_scores.argsort()[::-1][:top_k]
  
  print(f'\nClustered Binary top {top_k}:', global_top_k)
  print(f'            Exact top {top_k}:', exact_top_k)


db = np.random.rand(DB_SIZE, ORIGINAL_EMBED_SIZE)*2-1
binary_db = preprocess(db)
centroids, binary_centroids, cluster_members, cluster_labels = create_clusters()

if __name__ == '__main__':
  try:
    while True:
      query = np.random.rand(ORIGINAL_EMBED_SIZE)*2-1
      search_with_clusters(query, binary_centroids, cluster_members, top_k=10)
      input('\npress enter to check for next answer:')
  except KeyboardInterrupt:
    print('byee')
