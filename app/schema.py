import uuid
from typing import List, Optional
from pydantic import BaseModel


class NormalizeRequest(BaseModel):
    text: str


class NormalizeResponse(BaseModel):
    normalized_text: str


class EmbeddingRequest(BaseModel):
    text1: str
    text2: str


class EmbeddingResponse(BaseModel):
    embedding1: List[float]
    embedding2: List[float]
    cosine_similarity: float
    angular_similarity_radians: float
    developer_value: Optional[str] = None


class ModelResponse(BaseModel):
    model_id: uuid.UUID
    model_name: str
