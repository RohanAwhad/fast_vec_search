import numpy as np
import torch
import torch.nn.functional as F

def apply_matryoshka_transformation(embeddings: torch.Tensor, matryoshka_dim) -> np.ndarray:
  embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
  embeddings = embeddings[:, :matryoshka_dim]
  embeddings = F.normalize(embeddings, p=2, dim=1)
  return embeddings.cpu().numpy()
