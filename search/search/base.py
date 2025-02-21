# copied from https://github.com/beir-cellar/beir/blob/v2.0.0/beir/retrieval/search/base.py
# didn't wanna make you install beir for this

from abc import ABC, abstractmethod
from typing import Dict

class BaseSearch(ABC):

  @abstractmethod
  def search(self, 
             corpus: dict[str, dict[str, str]],
             corpus_ids: list[str],
             corpus_embeddings: "np.ndarray",
             query_ids: list[str],
             query_embeddings: "np.ndarray",
             top_k: int, 
             **kwargs) -> Dict[str, Dict[str, float]]:
    """Performs semantic search on corpus using queries.

    Args:
      corpus: Dict mapping doc IDs to dict containing 'text' field
      queries: Dict mapping query IDs to query strings
      top_k: Number of top results to return per query
      **kwargs: Additional arguments (unused)

    Returns:
      Dict mapping query IDs to dict of {doc_id: score} for top k matches
    """
    pass
