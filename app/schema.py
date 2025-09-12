import uuid
from typing import List, Optional
from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    embeddings: List[float]
    dimension: int


class ModelResponse(BaseModel):
    model_id: uuid.UUID
    model_name: str
    dimension: int
