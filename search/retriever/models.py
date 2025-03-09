from pydantic import BaseModel
from typing import Any

class Document(BaseModel):
  doc_id: str|int
  text: str
  metadata: dict[str, Any] | None = None


class Result(BaseModel):
  document: Document
  score: float

class RankedResults(BaseModel):
  results: list[Result]
  def __getitem__(self, idx): return self.results[idx]

