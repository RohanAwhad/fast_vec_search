# this is a decently fast, correct kmeans implementation
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
  def __init__(self, n_clusters=8, max_iters=300, tol=1e-4):
    self.n_clusters = n_clusters
    self.max_iters = max_iters
    self.tol = tol
    self.centroids = None
    self.labels_ = None
    self.inertia_ = None

  def kmeans_plus_plus(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
    n_samples = X.shape[0]

    # First centroid: random point
    centroids = [X[np.random.randint(n_samples)]]

    # Pick remaining centroids
    for _ in range(1, n_clusters):
      # Compute distances to closest centroid
      dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centroids], axis=0)

      # Choose next centroid with probability proportional to dist^2
      probs = dists / dists.sum()
      new_centroid_idx = np.random.choice(n_samples, p=probs)
      centroids.append(X[new_centroid_idx])

    return np.array(centroids)

  def fit(self, X):
    n_samples, n_features = X.shape

    # Initialize centroids randomly
    self.centroids = self.kmeans_plus_plus(X, self.n_clusters)

    # Sample points for centroid update
    sample_size = min(256 * self.n_clusters, n_samples)
    sample_idx = np.random.choice(n_samples, sample_size, replace=False)
    X_sample = X[sample_idx]

    for _ in range(self.max_iters):
      old_centroids = self.centroids.copy()

      # Assign samples to closest centroids
      distances = self.compute_l2_distances(X_sample, self.centroids)
      # distances = np.sqrt(((X_sample[:, np.newaxis] - self.centroids) ** 2).sum(axis=2))
      sample_labels = distances.argmin(axis=1)

      # Update centroids using sampled points
      for k in range(self.n_clusters):
        mask = sample_labels == k
        if np.any(mask):
          self.centroids[k] = X_sample[mask].mean(axis=0)

      # Check convergence
      if np.all(np.abs(old_centroids - self.centroids) < self.tol):
        break

    # Final assignment for all points
    distances = self.compute_l2_distances(X, self.centroids)
    self.labels_ = distances.argmin(axis=1)
    self.inertia_ = ((X - self.centroids[self.labels_]) ** 2).sum()

    return self

  def compute_l2_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    # this is god awful slow and memory heavy
    # dists = np.sqrt(np.sum((X1.reshape(m, 1, D) - X2.reshape(1, n, D)) ** 2, axis=-1))

    X1_squared = np.sum(X1**2, axis=1, keepdims=True)  # m*D FLOPs
    X2_squared = np.sum(X2**2, axis=1).reshape(1, -1)  # n*D FLOPs
    X1X2 = np.dot(X1, X2.T)  # m*n*D FLOPs

    return X1_squared + X2_squared - 2*X1X2
    # dists = np.sqrt(X1_squared + X2_squared - 2*X1X2)  # 3*m*n FLOPs  # can skip sqrt
    # return dists

  def predict(self, X):
    distances = self.compute_l2_distances(X, self.centroids)
    return distances.argmin(axis=1)

  def fit_predict(self, X):
    return self.fit(X).predict(X)


# write a test to compare sklearn and this implementation
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
import time

# Generate synthetic data
n_samples = 10000
n_features = 2
n_clusters = 5
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Test custom implementation
custom_kmeans = KMeans(n_clusters=n_clusters, max_iters=300)
start_time = time.time()
custom_labels = custom_kmeans.fit_predict(X)
custom_time = time.time() - start_time
custom_inertia = custom_kmeans.inertia_

# Test sklearn implementation
sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, max_iter=300, n_init=1, random_state=42)
start_time = time.time()
sklearn_labels = sklearn_kmeans.fit_predict(X)
sklearn_time = time.time() - start_time
sklearn_inertia = sklearn_kmeans.inertia_

print(f"Custom KMeans:")
print(f"Time: {custom_time:.4f} seconds")
print(f"Inertia: {custom_inertia:.4f}")
print("\nSklearn KMeans:")
print(f"Time: {sklearn_time:.4f} seconds")
print(f"Inertia: {sklearn_inertia:.4f}")

# now test output and speed with faiss
import faiss
import numpy as np
import time

# Test with FAISS
def faiss_kmeans(X, n_clusters, n_iter=300):
  X = X.astype(np.float32)
  d = X.shape[1]
  kmeans = faiss.Kmeans(d, n_clusters, niter=n_iter, verbose=False)
  start_time = time.time()
  kmeans.train(X)
  faiss_time = time.time() - start_time

  centroids = kmeans.centroids
  _, labels = kmeans.index.search(X, 1)
  labels = labels.reshape(-1)

  return labels, centroids, faiss_time, kmeans.obj[-1]

# Generate larger dataset for comparison
n_samples = 100000
n_features = 2
n_clusters = 5
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Run all implementations
custom_kmeans = KMeans(n_clusters=n_clusters)
sklearn_kmeans = SklearnKMeans(n_clusters=n_clusters, n_init=1, random_state=42)

start = time.time()
custom_labels = custom_kmeans.fit_predict(X)
custom_time = time.time() - start

start = time.time()
sklearn_labels = sklearn_kmeans.fit_predict(X)
sklearn_time = time.time() - start

faiss_labels, faiss_centroids, faiss_time, faiss_inertia = faiss_kmeans(X, n_clusters)

# Print results
print("Timing Comparison:")
print(f"Custom:  {custom_time:.4f}s")
print(f"Sklearn: {sklearn_time:.4f}s")
print(f"FAISS:   {faiss_time:.4f}s")

print("\nInertia Comparison:")
print(f"Custom:  {custom_kmeans.inertia_:.4f}")
print(f"Sklearn: {sklearn_kmeans.inertia_:.4f}")
print(f"FAISS:   {faiss_inertia:.4f}")


# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=custom_labels, cmap='viridis')
plt.scatter(custom_kmeans.centroids[:, 0], custom_kmeans.centroids[:, 1],
           c='red', marker='x', s=200, linewidths=3)
plt.title('Custom KMeans')

plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis')
plt.scatter(sklearn_kmeans.cluster_centers_[:, 0], sklearn_kmeans.cluster_centers_[:, 1],
           c='red', marker='x', s=200, linewidths=3)
plt.title('Sklearn KMeans')

plt.subplot(133)
plt.scatter(X[:, 0], X[:, 1], c=faiss_labels, cmap='viridis')
plt.scatter(faiss_centroids[:, 0], faiss_centroids[:, 1],
           c='red', marker='x', s=200, linewidths=3)
plt.title('FAISS KMeans')

plt.tight_layout()
plt.show()

# Scaling comparison
sizes = [1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]
custom_times = []
sklearn_times = []
faiss_times = []

for size in sizes:
  X, _ = make_blobs(n_samples=size, n_features=n_features, centers=n_clusters, random_state=42)

  start = time.time()
  custom_kmeans.fit(X)
  custom_times.append(time.time() - start)

  start = time.time()
  sklearn_kmeans.fit(X)
  sklearn_times.append(time.time() - start)

  _, _, faiss_time, _ = faiss_kmeans(X, n_clusters)
  faiss_times.append(faiss_time)

plt.figure(figsize=(10, 6))
plt.plot(sizes, custom_times, 'o-', label='Custom')
plt.plot(sizes, sklearn_times, 'o-', label='Sklearn')
plt.plot(sizes, faiss_times, 'o-', label='FAISS')
plt.xlabel('Dataset Size')
plt.ylabel('Time (seconds)')
plt.title('Scaling Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.xscale('log')
plt.show()


