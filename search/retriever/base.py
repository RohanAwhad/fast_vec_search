from abc import ABC, abstractmethod
from typing import Any, Type

from .models import RankedResults


class BaseIndex(ABC):
  @abstractmethod
  def query(self, query_embeddings: list[list[float]], n_results: int, filters=None) -> list[RankedResults]: pass

class BaseDB(ABC):
  @abstractmethod
  def create_collection(self, name: str, metadata: dict[str, Type] | None = None, **kwargs) -> "BaseDB": pass

  @abstractmethod
  def add(self, ids: list[str|int], documents: list[str], embeddings: list[list[float]], metadata: list[dict[str, Any]] | None = None, **kwargs) -> None: pass

  @abstractmethod
  def create_index(self) -> BaseIndex: pass
