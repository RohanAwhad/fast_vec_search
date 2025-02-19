# python implementation that works
import numpy as np

DB_SIZE = 10
EMBED_SIZE = 256
BYTE_SIZE = 8
SUBVECTOR_SIZE = 8

def compress(db):
  _tmp = list(map(lambda x: '0' if x < 0 else '1', db.reshape(-1)))
  binary_db = []
  for i in range(0, len(_tmp), EMBED_SIZE):
    _ = _tmp[i:i+EMBED_SIZE]
    binary_db.append([])
    for j in range(0, EMBED_SIZE, SUBVECTOR_SIZE):
      binary_db[-1].append(''.join(_[j:j+SUBVECTOR_SIZE]))
  return binary_db

def generate_subvector_scores(query):
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
  subvector_scores = generate_subvector_scores(query[:EMBED_SIZE])
  scores = compile_scores(subvector_scores, binary_db)
  exact_scores = (query.reshape(1, EMBED_SIZE) @ db.T).reshape(-1)

  binary_top_10 = scores.argsort()[::-1][:10]
  exact_top_10 = exact_scores.argsort()[::-1][:10]

  binary_top_100 = scores.argsort()[::-1][:100]
  exact_top_100 = exact_scores.argsort()[::-1][:100]

  print('\n\nBinary dot product ranks:', binary_top_10)
  print(' Exact dot product  ranks:', exact_top_10)
  print('Common in top 100: ', len(set(binary_top_100).intersection(exact_top_100)))

db = np.random.rand(DB_SIZE, EMBED_SIZE)*2-1
binary_db = preprocess(db)


# ===
# C Implementation to check
# ===

import ctypes
import functools
import numpy as np


@functools.lru_cache()
def get_lib():
  lib = ctypes.CDLL('./libexa_search.so')

  # Define argument types
  lib.get_binary_matrix.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
  ]

  lib.instantiate_lookup_table.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
  ]

  lib.quantize.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
  ]

  lib.get_scores.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.uint8),
    np.ctypeslib.ndpointer(dtype=np.float32),
    np.ctypeslib.ndpointer(dtype=np.float32),
    ctypes.c_int,
    ctypes.c_int,
  ]
  return lib

lib = get_lib()
compressed = np.zeros((DB_SIZE, EMBED_SIZE//8), dtype=np.uint8)
lib.quantize(compressed, db[:, :EMBED_SIZE].ravel().astype(np.float32), DB_SIZE)


# use numpy to see if compressed has all values equal to binary_db
# Convert binary_db to numpy array for comparison
binary_np = np.zeros((DB_SIZE, EMBED_SIZE//8), dtype=np.uint8)
for i, row in enumerate(binary_db):
  for j, byte_str in enumerate(row):
    binary_np[i,j] = int(byte_str, 2)

# Compare arrays
are_equal = np.array_equal(compressed, binary_np)
print("Arrays are equal:", are_equal)

if not are_equal:
  print("\nDifferences:")
  for i in range(DB_SIZE):
    if not np.array_equal(compressed[i], binary_np[i]):
      print(f"Row {i}:")
      print("compressed:", compressed[i])
      print("binary_db:", binary_np[i])


# test matrix_b
keys = [f"{i:08b}" for i in range(2**SUBVECTOR_SIZE)]
b = np.array([list(map(lambda x: 1 if x=='1' else -1, x)) for x in keys]).astype(np.float32)

matrix_B = np.random.randn(256, BYTE_SIZE).astype(np.float32)
lib.get_binary_matrix(matrix_B)

print("Matrix B valid?", np.array_equal(matrix_B, b))

# pretty print matrix b
print("\nMatrix B:")
for i in range(min(10, len(matrix_B))):
  print(f"Row {i}: {matrix_B[i]}")
print("...")

# test lookup table
lookup = np.zeros((EMBED_SIZE//SUBVECTOR_SIZE, 256), dtype=np.float32)
query = np.random.rand(EMBED_SIZE)*2-1
query_np = query.astype(np.float32)
subvector_scores = generate_subvector_scores(query_np)
subvector_scores = query_np.reshape(EMBED_SIZE//SUBVECTOR_SIZE, SUBVECTOR_SIZE) @ matrix_B.T

print("A shape:", query_np.reshape(EMBED_SIZE//SUBVECTOR_SIZE, SUBVECTOR_SIZE).shape)
print("B shape:", matrix_B.T.shape)

# Test C implementation lookup table
lib.instantiate_lookup_table(
  lookup.ravel(),
  query_np.ravel(),
  matrix_B.T.ravel(),
  1
)

print("\nLookup table valid?", np.allclose(lookup, subvector_scores, atol=1e-5))


if not np.allclose(lookup, subvector_scores, atol=1e-5):
  # Print first few elements of each
  print("\nSubvector scores first 5 elements:")
  for i in range(min(5, EMBED_SIZE//SUBVECTOR_SIZE)):
    print(f"Row {i}: {subvector_scores[i][:5]}")

  print("\nLookup table first 5 elements:")
  for i in range(min(5, EMBED_SIZE//SUBVECTOR_SIZE)):
    print(f"Row {i}: {lookup[i][:5]}")

  # Print absolute differences
  print("\nAbsolute differences first 5 elements:")
  diffs = np.abs(lookup - subvector_scores)
  for i in range(min(5, EMBED_SIZE//SUBVECTOR_SIZE)):
    print(f"Row {i}: {diffs[i][:5]}")

  # Print statistics
  print("\nStatistics:")
  print(f"Max difference: {np.max(diffs)}")
  print(f"Mean difference: {np.mean(diffs)}")
  print(f"Std difference: {np.std(diffs)}")

# test final scores
query = np.random.rand(1, EMBED_SIZE).astype(np.float32)
subvector_scores = generate_subvector_scores(query[:, :EMBED_SIZE])
py_scores = compile_scores(subvector_scores, binary_db)

# numpy all close
query = query.astype(np.float32).ravel()
scores_c = np.zeros(DB_SIZE, dtype=np.float32)
lib.get_scores(query.ravel(), compressed.ravel(), matrix_B.T.ravel(), scores_c.ravel(), 1, DB_SIZE)

print("Scores match?", np.allclose(scores_c, py_scores))

if not np.allclose(scores_c, py_scores):
  print("\nScore differences:")
  print("Python scores:", py_scores)
  print("C scores:", scores_c)
  print("Max difference:", np.max(np.abs(scores_c - py_scores)))
  print("Mean difference:", np.mean(np.abs(scores_c - py_scores)))

  binary_top_10 = py_scores.argsort()[::-1][:10]
  binary_top_100 = py_scores.argsort()[::-1][:100]
  binary_top_10_c = scores_c.argsort()[::-1][:10]
  binary_top_100_c = scores_c.argsort()[::-1][:100]
  print("\nTop 10 indices match?", np.array_equal(binary_top_10_c, binary_top_10))
  print("Common in top 100:", len(set(binary_top_100_c).intersection(set(binary_top_100))))

def benchmark():
  import time
  num_runs = 1000
  
  start = time.time()
  for _ in range(num_runs):
    subvector_scores = generate_subvector_scores(query[:EMBED_SIZE])
    py_scores = compile_scores(subvector_scores, binary_db)
  py_time = (time.time() - start) / num_runs
  
  start = time.time()
  for _ in range(num_runs):
    lib.get_scores(query, compressed, matrix_B, scores_c, EMBED_SIZE, DB_SIZE)
  c_time = (time.time() - start) / num_runs
  
  print(f"\nBenchmark results (average over {num_runs} runs):")
  print(f"Python implementation: {py_time*1000:.3f} ms")
  print(f"C implementation: {c_time*1000:.3f} ms")
  print(f"Speedup: {py_time/c_time:.2f}x")

benchmark()


# these were all single query, now I want to test the final scores for a batch of 2 queries at a time
# Test batch processing
batch_size = 2
queries = np.random.rand(batch_size, EMBED_SIZE).astype(np.float32) * 2 - 1

# Python implementation
py_batch_scores = []
for q in queries:
  subvector_scores = generate_subvector_scores(q)
  scores = compile_scores(subvector_scores, binary_db)
  py_batch_scores.append(scores)
py_batch_scores = np.array(py_batch_scores)

# C implementation
c_batch_scores = np.zeros((batch_size, DB_SIZE), dtype=np.float32)
lib.get_scores(
  queries.ravel(),
  compressed.ravel(),
  matrix_B.T.ravel(),
  c_batch_scores.ravel(),
  batch_size,
  DB_SIZE
)

print("Batch scores match?", np.allclose(c_batch_scores, py_batch_scores))

if not np.allclose(c_batch_scores, py_batch_scores):
  print("\nBatch score differences:")
  print("Max difference:", np.max(np.abs(c_batch_scores - py_batch_scores)))
  print("Mean difference:", np.mean(np.abs(c_batch_scores - py_batch_scores)))
