from typing import List

from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    LOG_DIR: str = "log"

    EMBEDDING_MODEL_NAMES: List[str] = [
        "intfloat/multilingual-e5-base",
        "ai-forever/sbert_large_mt_nlu_ru",
        "Qwen/Qwen3-Embedding-4B"
    ]

    LLM_MODEL_NAME: str = "Qwen/Qwen3-8B"

    VLLM_URL: str = "http://localhost:8080"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

os.makedirs(settings.LOG_DIR, exist_ok=True)
